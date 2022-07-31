import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import tensorrt as trt

import torch
from torchvision import transforms, datasets

from core.raft import RAFT
import ctypes

def parse_args():
  parser = argparse.ArgumentParser(
    description='Simple testing funtion for Monodepthv2 models.')

  parser.add_argument('--name', default='raft', help="name your experiment")
  parser.add_argument('--stage', help="determines which dataset to use for training")
  parser.add_argument('--restore_ckpt', help="restore checkpoint")
  parser.add_argument('--small', action='store_true', help='use small model')
  parser.add_argument('--validation', type=str, nargs='+')

  parser.add_argument('--lr', type=float, default=0.00002)
  parser.add_argument('--num_steps', type=int, default=100000)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
  # parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
  parser.add_argument('--gpus', type=int, nargs='+', default=[0])
  parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

  parser.add_argument('--iters', type=int, default=12)
  parser.add_argument('--wdecay', type=float, default=.00005)
  parser.add_argument('--epsilon', type=float, default=1e-8)
  parser.add_argument('--clip', type=float, default=1.0)
  parser.add_argument('--dropout', type=float, default=0.0)
  parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
  parser.add_argument('--add_noise', action='store_true')

  return parser.parse_args()


def export_to_onnx(args):

  pth_file_path = '../checkpoints/'
  full_pth_path = os.path.join(pth_file_path, "raft.pth")
  print(f"[trace] Loading model from {full_pth_path}")

  if not os.path.exists(full_pth_path):
    print(f'[trace] weight file {full_pth_path} not exist, exit...')
    exit(-1)

  output_model_name = 'raft.onnx'
  if os.path.exists(output_model_name):
    print(f'[trace] target file {full_pth_path} exists, go to next routine..')
    return

  device = torch.device("cuda")
  loaded_dict_enc = torch.load(full_pth_path, map_location=device)
  new_dict = {}
  for key in loaded_dict_enc:
    value = loaded_dict_enc[key]
    new_key = key.replace("module.", "")
    new_dict[new_key] = value

  model = RAFT(args)
  model.load_state_dict(new_dict)
  model.eval()
  model = model.to(device)
  print(f'[trace] model weight file loaded')

  input0 = torch.FloatTensor(1, 3, 440, 1024).to(device)
  input1 = torch.FloatTensor(1, 3, 440, 1024).to(device)
  input_tuple = (input0, input1)

  print(f'[trace] start the test run')
  example_output = model(input0, input1, iters = 1, test_mode = True)

  print('[trace] start to export the onnx file')
  print(torch.__version__)
  torch.onnx.export(model,  # model being run
                    args=input_tuple,  # model input (or a tuple for multiple inputs)
                    f=output_model_name,
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=['input0', 'input1',],  # the model's input names
                    output_names=['output',],
                    dynamic_axes = {'input0': {0: 'batch_size'},  # variable length axes
                                    'input1': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}}
  )
  pass

def GiB(val):
  return val * 1 << 30

def _get_calibrator(args):
  #Calibrator(train_loader, cache_file=calibration_cache)
  import sys
  sys.path.append('../core')
  import datasets
  train_loader = datasets.fetch_dataloader(args)
  import utility
  calibration_cache = "calibration.cache"
  calib = utility.Calibrator(train_loader, cache_file=calibration_cache)
  return calib
  pass

def build_engine(args, mode):

  onnx_file_path = 'raft-simp.onnx'
  if not os.path.exists(onnx_file_path):
    print("[trace] ONNX file {} not found, exit -1".format(onnx_file_path))
    exit(-1)

  TRT_LOGGER = trt.Logger()
  EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

  """Takes an ONNX file and creates a TensorRT engine to run inference with"""
  with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
      builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(
    TRT_LOGGER) as runtime:

    print("[trace] loading ONNX file from path {}...".format(onnx_file_path))
    with open(onnx_file_path, "rb") as model:
      print("Beginning ONNX file parsing")
      if not parser.parse(model.read()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
          print(parser.get_error(error))
        return None
      print("Completed parsing of ONNX file")

    # Parse model file
    import utility
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(16))
    calib = _get_calibrator(args)

    if mode == utility.CalibratorMode.INT8:
      config.set_flag(trt.BuilderFlag.INT8)
    elif mode == utility.CalibratorMode.FP16:
      config.set_flag(trt.BuilderFlag.FP16)
    elif mode == utility.CalibratorMode.TF32:
      config.set_flag(trt.BuilderFlag.TF32)
    elif mode == utility.CalibratorMode.FP32:
      # do nothing since this is the default branch
      # config.set_flag(trt.BuilderFlag.FP32)
      pass
    else:
      print(f'[trace] unknown calibrator mode: {mode.name}, exit')
      exit(-1)

    # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
    print("[trace] Building an engine from file {}; this may take a while...".format(onnx_file_path))
    pic_dim_col = 440
    pic_dim_width = 1024
    config.int8_calibrator = calib
    engine_file_path = f'raft.{mode.name}.engine'
    profile = builder.create_optimization_profile()
    min_shape = (1, 3, pic_dim_col, pic_dim_width)
    opt_shape = (8, 3, pic_dim_col, pic_dim_width)
    max_shape = (16, 3, pic_dim_col, pic_dim_width)
    profile.set_shape(network.get_input(0).name, min_shape, opt_shape, max_shape)
    profile.set_shape(network.get_input(1).name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, config)
    if plan == None:
      print("[trace] builder.build_serialized_network failed, exit -1")
      exit(-1)
    engine = runtime.deserialize_cuda_engine(plan)
    print("[trace] Completed creating Engine")
    with open(engine_file_path, "wb") as f:
      f.write(plan)
    return engine


def onnx_to_trt(args):

  TRT_LOGGER = trt.Logger()
  trt.init_libnvinfer_plugins(TRT_LOGGER, '')
  so_file = "/home/noname/projects/deeplearning/tensorrt/trt-samples-for-hackathon-cn/old/GridSamplerPlugin.so"
  #so_file = "/home/noname/projects/deeplearning/tensorrt/gridSamplePlugin/build/libgridSamplePlugin.so"
  if not os.path.exists(so_file):
    print(f'[trace] target so file {so_file} not exist, exit')
    exit(-1)

  ctypes.CDLL(so_file)
  # node type: GridSample
  registry = trt.get_plugin_registry()
  for c in registry.plugin_creator_list:
    print("plugin name:", c.name, "plugin namespace:", c.plugin_namespace)

  EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
  import utility
  mode: utility.CalibratorMode = utility.CalibratorMode.INT8
  engine_file_path = f'raft-simp.{mode.name}.engine'
  if os.path.exists(engine_file_path):
    # If a serialized engine exists, use it instead of building an engine.
    '''
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
      return runtime.deserialize_cuda_engine(f.read())
    '''
    print(f'[trace] target engine file {engine_file_path} already exists, do nothing')
    exit(0)
  else:
    build_engine(args, mode)

  pass

def main(args):
  print(f'[trace] working in the main function')
  export_to_onnx(args)
  onnx_to_trt(args)
  pass


if __name__ == '__main__':
  args = parse_args()
  main(args)
