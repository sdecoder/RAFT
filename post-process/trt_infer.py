import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from core.utils import flow_viz

import torch
import torchvision.transforms
from torchvision import transforms, datasets

import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import cv2
import torch
import argparse
import ctypes

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

parser = argparse.ArgumentParser(
  description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--models', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Which dataset to use')


def load_engine(trt_runtime, engine_path):

  trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")
  print(f'[trace] loading TensorRT engine from file {engine_path}')
  if not os.path.exists(engine_path):
    print(f'[trace] TensorRT engine file {engine_path} does not exist, exit')
    exit(-1)

  with open(engine_path, 'rb') as f:
    engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    print(f'[trace] engine has been loaded.')
    return engine

  return None

class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()

def load_plugin():

  print(f'[trace] start to load plugin')
  so_file = '/home/noname/projects/deeplearning/tensorrt/plugins/gridSamplePlugin/plugin/GridSamplePlugin.so'
  if not os.path.exists(so_file):
    print(f'[trace] TensorRT plugin file {so_file} does not exist, exit')
    exit(-1)

  ctypes.CDLL(so_file)
  # node type: GridSample
  registry = trt.get_plugin_registry()
  for c in registry.plugin_creator_list:
    print("plugin name:", c.name, "plugin namespace:", c.plugin_namespace)

  print(f'[trace] plugin loaded')
  pass

def allocate_buffers_for_engine(engine):
  """Allocates host and device buffer for TRT engine inference.
  This function is similair to the one in common.py, but
  converts network outputs (which are np.float32) appropriately
  before writing them to Python buffer. This is needed, since
  TensorRT plugins doesn't support output type description, and
  in our particular case, we use NMS plugin as network output.
  Args:
      engine (trt.ICudaEngine): TensorRT engine
  Returns:
      inputs [HostDeviceMem]: engine input memory
      outputs [HostDeviceMem]: engine output memory
      bindings [int]: buffer to device bindings
      stream (cuda.Stream): cuda stream for engine inference synchronization
  """
  print('[trace] reach func@allocate_buffers_for_engine')
  inputs = []
  outputs = []
  bindings = []
  stream = cuda.Stream()

  binding_to_type = {}
  binding_to_type['input0'] = np.float32
  binding_to_type['input1'] = np.float32

  binding_to_type['output'] = np.float32
  binding_to_type['1262'] = np.float32
  binding_to_type['1637'] = np.float32
  binding_to_type['2012'] = np.float32
  binding_to_type['2387'] = np.float32
  binding_to_type['2762'] = np.float32
  binding_to_type['3137'] = np.float32
  binding_to_type['3512'] = np.float32
  binding_to_type['3887'] = np.float32
  binding_to_type['4262'] = np.float32
  binding_to_type['4637'] = np.float32
  binding_to_type['5012'] = np.float32

  # Current NMS implementation in TRT only supports DataType.FLOAT but
  # it may change in the future, which could brake this sample here
  # when using lower precision [e.g. NMS output would not be np.float32
  # anymore, even though this is assumed in binding_to_type]


  for binding in engine:
    print(f'[trace] current binding: {str(binding)}')
    _binding_shape = engine.get_binding_shape(binding)
    _volume = trt.volume(_binding_shape)
    size = _volume * engine.max_batch_size
    print(f'[trace] current binding size: {size}')
    dtype = binding_to_type[str(binding)]
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
      inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
      outputs.append(HostDeviceMem(host_mem, device_mem))

  print(f'[trace] done with memory bindings')
  return inputs, outputs, bindings, stream


def infer(engine):

  print(f'[start] to infer')
  inputs, outputs, bindings, stream = allocate_buffers_for_engine(engine)

  def _load_img(image_path):
    im = Image.open(image_path)
    im = im.resize((1024, 440))
    #im.show()
    trans_to_tensor = torchvision.transforms.ToTensor()
    _tensor = trans_to_tensor(im)
    return _tensor

  demo_frame0 = '../demo-frames/frame_0016.png'
  demo_frame1 = '../demo-frames/frame_0017.png'
  img0 = _load_img(demo_frame0)
  img1 = _load_img(demo_frame1)

  print(f'[trace] source images loaded')
  np.copyto(inputs[0].host, img0.ravel())
  np.copyto(inputs[1].host, img1.ravel())
  [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
  # Run inference.

  batch_size = 1
  context = engine.create_execution_context()
  context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
  [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
  stream.synchronize()
  return outputs

def render(outputs):
  print(f'[trace] start to render output data')
  ndarray_data = outputs[11].host
  new_shape = (2, 440, 1024)
  ndarray_data.resize(new_shape, refcheck=True)
  tensor = torch.from_numpy(ndarray_data)
  flo = tensor.permute(1, 2, 0).cpu().numpy()
  flo = flow_viz.flow_to_image(flo)

  img = Image.fromarray(flo, 'RGB')

  img.show()
  # map flow to rgb image

  # import matplotlib.pyplot as plt
  # plt.imshow(img_flo / 255.0)
  # plt.show()

  #cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
  #cv2.waitKey()

  pass

def convert_onnx_to_engine():

  EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
  with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
      builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(
    TRT_LOGGER) as runtime:
    config.max_workspace_size = 1 << 28  # 256MiB
    builder.max_batch_size = 1

  onnx_file_path = 'raft-simp.onnx'
  if not os.path.exists(onnx_file_path):
    print("[trace] ONNX file {} not found, exit -1".format(onnx_file_path))
    exit(-1)

  print("[trace] Loading ONNX file from path {}...".format(onnx_file_path))
  with open(onnx_file_path, "rb") as model:
    print("Beginning ONNX file parsing")
    if not parser.parse(model.read()):
      print("ERROR: Failed to parse the ONNX file.")
      for error in range(parser.num_errors):
        print(parser.get_error(error))
      return None
    print("Completed parsing of ONNX file")

  print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
  plan = builder.build_serialized_network(network, config)
  if plan == None:
    print("[trace] builder.build_serialized_network failed, exit -1")
    exit(-1)
  else:
    engine_file_path = 'raft-simp.engine'
    with open(engine_file_path, "wb") as f:
      f.write(plan)
  engine = runtime.deserialize_cuda_engine(plan)
  print("Completed creating Engine")
  return engine

def main():

  print("[trace] reach the main entry")

  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if args.cuda else "cpu")
  load_plugin()
  engine_file_path = 'raft-simp.engine'
  #engine_file_path = '/home/noname/projects/deeplearning/optical-flow/RAFT/post-process/polygraphy_capability_dumps/test.engine'
  #engine = convert_onnx_to_engine()
  engine = load_engine(trt_runtime, engine_file_path)
  if not engine:
    print(f'[trace] failed to load the engine file, exit')
    exit(-1)

  outputs = infer(engine)
  render(outputs)
  print(f'[trace] end of the main point')
  pass

if __name__ == "__main__":
  main()
  pass
