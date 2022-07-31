# RAFT
This repository contains the source code for our paper:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

<img src="RAFT.png">

## Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```Shell
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

## Demos
Pretrained models can be downloaded by running
```Shell
./download_models.sh
```
or downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

You can demo a trained model on a sequence of frames
```Shell
python demo.py --model=models/raft-things.pth --path=demo-frames
```

## Required Data
To evaluate/train RAFT, you will need to download the required datasets.
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)


By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```

## Training
We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
./train_standard.sh
```

If you have a RTX GPU, training can be accelerated using mixed precision. You can expect similiar results in this setting (1 GPU)
```Shell
./train_mixed.sh
```

## (Optional) Efficent Implementation
You can optionally use our alternate (efficent) implementation by compiling the provided cuda extension
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.

## TensorRT inference

* Once the training is done, the model can be exported to the ONNX file first.
* The ONNX file can NOT be converted to TensorRT engine file directly due to lacking of support for GridSampler operation.

### Solution:
1. Go to: https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/old/plugins
2. Build the GridSamplerPlugin.cpp as Linux .so file:
```shell
nvcc -g -Xcompiler -fPIC -shared -o GridSamplePlugin.so gridSamplerPlugin.cpp -lnvinfer -lcudnn
```
3. Load the so file during the engine build:
```python
import ctypes
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
ctypes.CDLL(so_file)
```
4. Follow the generic TensorRT conversion for engine generation.
5. The command line way would be:
```shell
trtexec --buildOnly --verbose --onnx=raft-simp.onnx --saveEngine=raft-simp.engine --plugins=GridSamplePlugin.so --workspace=4096
```

Failed when trying to convert to FP16 engine. Errors copied as following:
```log
trtexec --buildOnly --verbose --onnx=raft-simp.onnx --saveEngine=raft.FP16.engine --plugins=GridSamplePlugin.so --workspace=4096 --fp16 
[07/31/2022-10:41:46] [V] [TRT] Fastest Tactic: 0x000000000000001c Time: 0.173225
[07/31/2022-10:41:46] [V] [TRT] --------------- Timing Runner: PWN(PWN(onnx::Mul_176_clone_2 + (Unnamed Layer* 295) [Shuffle] + Mul_714, Div_718), PWN(onnx::Sub_178_clone_3 + (Unnamed Layer* 301) [Shuffle], Sub_720)) (PointWise)
[07/31/2022-10:41:46] [V] [TRT] PointWise has no valid tactics for this config, skipping
[07/31/2022-10:41:46] [V] [TRT] >>>>>>>>>>>>>>> Chose Runner Type: PointWiseV2 Tactic: 0x000000000000001c
[07/31/2022-10:41:46] [V] [TRT] =============== Computing costs for 
[07/31/2022-10:41:46] [V] [TRT] *************** Autotuning format combination: Float(7040,7040,128,1), Float(162,18,2,1) -> Float(81,81,9,1) ***************
[07/31/2022-10:41:46] [V] [TRT] --------------- Timing Runner: GridSample_730 (PluginV2)
[07/31/2022-10:41:46] [V] [TRT] Tactic: 0x0000000000000000 Time: 0.352631
[07/31/2022-10:41:46] [V] [TRT] Fastest Tactic: 0x0000000000000000 Time: 0.352631
[07/31/2022-10:41:46] [V] [TRT] >>>>>>>>>>>>>>> Chose Runner Type: PluginV2 Tactic: 0x0000000000000000
[07/31/2022-10:41:46] [V] [TRT] *************** Autotuning format combination: Half(7040,7040,128,1), Half(162,18,2,1) -> Half(81,81,9,1) ***************
[07/31/2022-10:41:46] [V] [TRT] --------------- Timing Runner: GridSample_730 (PluginV2)
[07/31/2022-10:41:58] [W] [TRT] GPU error during getBestTactic: GridSample_730 : the launch timed out and was terminated
[07/31/2022-10:41:58] [V] [TRT] =============== Computing costs for 
[07/31/2022-10:41:58] [V] [TRT] *************** Autotuning format combination: Float(7040,7040,128,1) -> Float(1728,1728,64,1) ***************
[07/31/2022-10:41:58] [V] [TRT] --------------- Timing Runner: AveragePool_526 (TiledPooling)
[07/31/2022-10:41:58] [W] [TRT] GPU error during getBestTactic: AveragePool_526 : the launch timed out and was terminated
[07/31/2022-10:41:58] [V] [TRT] *************** Autotuning format combination: Half(7040,7040,128,1) -> Half(1728,1728,64,1) ***************
[07/31/2022-10:41:58] [V] [TRT] --------------- Timing Runner: AveragePool_526 (TiledPooling)
[07/31/2022-10:41:58] [V] [TRT] TiledPooling has no valid tactics for this config, skipping
[07/31/2022-10:41:58] [V] [TRT] --------------- Timing Runner: AveragePool_526 (CudnnPooling)
[07/31/2022-10:41:58] [W] [TRT] GPU error during getBestTactic: AveragePool_526 : the launch timed out and was terminated
[07/31/2022-10:41:58] [E] [TRT] plugin/instanceNormalizationPlugin/instanceNormalizationPlugin.cu (177) - Cudnn Error in terminate: 4 (CUDNN_STATUS_INTERNAL_ERROR)
terminate called after throwing an instance of 'nvinfer1::plugin::CudnnError'
  what():  std::exception
fish: Job 1, 'trtexec --buildOnly --verbose -…' terminated by signal SIGABRT (Abort)
```

### Evaluation

```shell
trtexec --loadEngine=raft.FP32.engine --plugins=GridSamplerPlugin.so  --batch=1024 --streams=1 --verbose --avgRuns=16
[07/31/2022-10:50:17] [I] === Performance summary ===
[07/31/2022-10:50:17] [I] Throughput: 5560.21 qps
[07/31/2022-10:50:17] [I] Latency: min = 196.026 ms, max = 198.59 ms, mean = 197.778 ms, median = 198.062 ms, percentile(99%) = 198.59 ms
[07/31/2022-10:50:17] [I] Enqueue Time: min = 181.541 ms, max = 184.13 ms, mean = 183.331 ms, median = 183.623 ms, percentile(99%) = 184.13 ms
[07/31/2022-10:50:17] [I] H2D Latency: min = 3.85986 ms, max = 3.92706 ms, mean = 3.88376 ms, median = 3.87628 ms, percentile(99%) = 3.92706 ms
[07/31/2022-10:50:17] [I] GPU Compute Time: min = 178.025 ms, max = 180.634 ms, mean = 179.814 ms, median = 180.129 ms, percentile(99%) = 180.634 ms
[07/31/2022-10:50:17] [I] D2H Latency: min = 13.2493 ms, max = 14.2151 ms, mean = 14.0807 ms, median = 14.1177 ms, percentile(99%) = 14.2151 ms
[07/31/2022-10:50:17] [I] Total Host Walltime: 3.13082 s
[07/31/2022-10:50:17] [I] Total GPU Compute Time: 3.05683 s
[07/31/2022-10:50:17] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[07/31/2022-10:50:17] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[07/31/2022-10:50:17] [I] Explanations of the performance metrics are printed in the verbose logs.
 
```

```shell
trtexec --loadEngine=raft.FP32.engine --plugins=GridSamplerPlugin.so --batch=4096 --streams=1 --verbose --avgRuns=16
[07/31/2022-10:51:48] [I] === Performance summary ===
[07/31/2022-10:51:48] [I] Throughput: 22153.6 qps
[07/31/2022-10:51:48] [I] Latency: min = 196.777 ms, max = 202.45 ms, mean = 198.494 ms, median = 198.579 ms, percentile(99%) = 202.45 ms
[07/31/2022-10:51:48] [I] Enqueue Time: min = 182.278 ms, max = 187.959 ms, mean = 184.055 ms, median = 184.087 ms, percentile(99%) = 187.959 ms
[07/31/2022-10:51:48] [I] H2D Latency: min = 3.86035 ms, max = 3.95959 ms, mean = 3.8778 ms, median = 3.87085 ms, percentile(99%) = 3.95959 ms
[07/31/2022-10:51:48] [I] GPU Compute Time: min = 178.785 ms, max = 184.374 ms, mean = 180.54 ms, median = 180.573 ms, percentile(99%) = 184.374 ms
[07/31/2022-10:51:48] [I] D2H Latency: min = 13.262 ms, max = 14.198 ms, mean = 14.0754 ms, median = 14.1191 ms, percentile(99%) = 14.198 ms
[07/31/2022-10:51:48] [I] Total Host Walltime: 3.14315 s
[07/31/2022-10:51:48] [I] Total GPU Compute Time: 3.06919 s
[07/31/2022-10:51:48] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[07/31/2022-10:51:48] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[07/31/2022-10:51:48] [I] Explanations of the performance metrics are printed in the verbose logs.
[07/31/2022-10:51:48] [V] 
```

```shell
trtexec --loadEngine=raft.FP32.engine --plugins=GridSamplerPlugin.so --batch=8192 --streams=1 --verbose --avgRuns=16
[07/31/2022-10:54:54] [I] === Performance summary ===
[07/31/2022-10:54:54] [I] Throughput: 44282.4 qps
[07/31/2022-10:54:54] [I] Latency: min = 197.01 ms, max = 200.25 ms, mean = 198.634 ms, median = 198.731 ms, percentile(99%) = 200.25 ms
[07/31/2022-10:54:54] [I] Enqueue Time: min = 182.512 ms, max = 185.705 ms, mean = 184.162 ms, median = 184.448 ms, percentile(99%) = 185.705 ms
[07/31/2022-10:54:54] [I] H2D Latency: min = 3.86938 ms, max = 3.97961 ms, mean = 3.90051 ms, median = 3.89319 ms, percentile(99%) = 3.97961 ms
[07/31/2022-10:54:54] [I] GPU Compute Time: min = 178.994 ms, max = 182.148 ms, mean = 180.63 ms, median = 180.917 ms, percentile(99%) = 182.148 ms
[07/31/2022-10:54:54] [I] D2H Latency: min = 13.2424 ms, max = 14.2108 ms, mean = 14.1036 ms, median = 14.1479 ms, percentile(99%) = 14.2108 ms
[07/31/2022-10:54:54] [I] Total Host Walltime: 3.14491 s
[07/31/2022-10:54:54] [I] Total GPU Compute Time: 3.0707 s
[07/31/2022-10:54:54] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[07/31/2022-10:54:54] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[07/31/2022-10:54:54] [I] Explanations of the performance metrics are printed in the verbose logs.
```

```shell
trtexec --loadEngine=raft.FP32.engine --plugins=GridSamplerPlugin.so --batch=32768 --streams=1 --verbose --avgRuns=16
[07/31/2022-10:56:48] [I] === Performance summary ===
[07/31/2022-10:56:48] [I] Throughput: 178359 qps
[07/31/2022-10:56:48] [I] Latency: min = 195.566 ms, max = 199.309 ms, mean = 197.33 ms, median = 197.453 ms, percentile(99%) = 199.309 ms
[07/31/2022-10:56:48] [I] Enqueue Time: min = 181.084 ms, max = 184.749 ms, mean = 182.888 ms, median = 183.122 ms, percentile(99%) = 184.749 ms
[07/31/2022-10:56:48] [I] H2D Latency: min = 3.86218 ms, max = 3.91199 ms, mean = 3.87888 ms, median = 3.875 ms, percentile(99%) = 3.91199 ms
[07/31/2022-10:56:48] [I] GPU Compute Time: min = 177.583 ms, max = 181.196 ms, mean = 179.374 ms, median = 179.621 ms, percentile(99%) = 181.196 ms
[07/31/2022-10:56:48] [I] D2H Latency: min = 13.2102 ms, max = 14.21 ms, mean = 14.0768 ms, median = 14.1218 ms, percentile(99%) = 14.21 ms
[07/31/2022-10:56:48] [I] Total Host Walltime: 3.12322 s
[07/31/2022-10:56:48] [I] Total GPU Compute Time: 3.04936 s
[07/31/2022-10:56:48] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[07/31/2022-10:56:48] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[07/31/2022-10:56:48] [I] Explanations of the performance metrics are printed in the verbose logs.
```
