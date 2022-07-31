from enum import Enum
from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

import timm

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import cv2
import pydicom
# from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom

import sys
import pycuda.driver as cuda
import pycuda.autoinit

class CalibratorMode(Enum):
  INT8 = 0
  FP16 = 1
  TF32 = 2
  FP32 = 3

import tensorrt as trt
class Calibrator(trt.IInt8EntropyCalibrator2):

  def __init__(self, dataloader, cache_file):

    print(f'[trace] Calibrator.init: checking dataloder type')
    assert isinstance(dataloader, DataLoader)
    # Whenever you specify a custom constructor for a TensorRT class,
    # you MUST call the constructor of the parent explicitly.
    trt.IInt8EntropyCalibrator2.__init__(self)
    self.cache_file = cache_file
    # Every time get_batch is caled, the next batch of size batch_size will be copied to the device and returned.

    self.dataloader = dataloader
    self.batch_size = dataloader.batch_size
    print(f'[trace] batch_size for the DataLoader is {self.batch_size}')
    print(f'[trace] batchlen for the internal DataLoader is {len(self.dataloader)}')
    print(f'[trace] datalen for the internal DataSet is {len(self.dataloader.dataset)}')
    self.current_index = 0

    # Allocate enough memory for a whole batch.
    channel = 3
    pic_dim_col = 440
    pic_dim_width = 1024
    single_data_bytes = channel * pic_dim_col * pic_dim_width
    self.device_input0 = cuda.mem_alloc(single_data_bytes * self.batch_size * 4)
    self.device_input1 = cuda.mem_alloc(single_data_bytes * self.batch_size * 4)

  def get_batch_size(self):
    return self.batch_size

  # TensorRT passes along the names of the engine bindings to the get_batch function.
  # You don't necessarily have to use them, but they can be useful to understand the order of
  # the inputs. The bindings list is expected to have the same ordering as 'names'.
  def get_batch(self, names):

    print(f'[trace] the names.type? {type(names)}, value: {names}')
    print(f'[trace] current index: {self.current_index}, data loader length: {len(self.dataloader)}')
    if self.current_index >= len(self.dataloader):
      print(f'[trace] self.current_index > len(self.dataloader): boundary encountered, return None;')
      return None

    objects = next(iter(self.dataloader))
    input0, input1, flow, valid = objects
    print(f"[trace] calibrating batch {self.current_index} containing {len(input0)} images")
    #batch = self.data[self.current_index: self.current_index + self.batch_size].ravel()
    _elements0 = input0.ravel().numpy()
    _elements1 = input1.ravel().numpy()
    cuda.memcpy_htod(self.device_input0, _elements0)
    cuda.memcpy_htod(self.device_input1, _elements1)

    self.current_index += 1
    return [self.device_input0, self.device_input1]

  def read_calibration_cache(self):
    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
    print(f'[trace] MNISTEntropyCalibrator: read_calibration_cache: {self.cache_file}')
    if os.path.exists(self.cache_file):
      with open(self.cache_file, "rb") as f:
        return f.read()

  def write_calibration_cache(self, cache):
    print(f'[trace] MNISTEntropyCalibrator: write_calibration_cache: {cache}')
    with open(self.cache_file, "wb") as f:
      f.write(cache)


def rand_bbox(size, lam):
  W = size[0]
  H = size[1]
  cut_rat = np.sqrt(1. - lam)
  cut_w = np.int(W * cut_rat)
  cut_h = np.int(H * cut_rat)

  # uniform
  cx = np.random.randint(W)
  cy = np.random.randint(H)

  bbx1 = np.clip(cx - cut_w // 2, 0, W)
  bby1 = np.clip(cy - cut_h // 2, 0, H)
  bbx2 = np.clip(cx + cut_w // 2, 0, W)
  bby2 = np.clip(cy + cut_h // 2, 0, H)
  return bbx1, bby1, bbx2, bby2


def get_img(path):
  im_bgr = cv2.imread(path)
  im_rgb = im_bgr[:, :, ::-1]
  # print(im_rgb)
  return im_rgb


CFG = {
  'fold_num': 5,
  'seed': 719,
  'model_arch': 'tf_efficientnet_b4_ns',
  'img_size': 512,
  'epochs': 20,
  'train_bs': 128,
  'valid_bs': 32,
  'T_0': 10,
  'lr': 1e-4,
  'min_lr': 1e-6,
  'weight_decay': 1e-6,
  'num_workers': 4,
  'accum_iter': 2,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
  'verbose_step': 1,
  'device': 'cuda:0'
}


def build_engine_common_routine(network, builder, config, runtime, engine_file_path):
  input_batch_size = 1
  input_channel = 1
  input_image_width = 28
  input_image_height = 28
  network.get_input(0).shape = [input_batch_size, input_channel, input_image_width, input_image_height]
  plan = builder.build_serialized_network(network, config)
  if plan == None:
    print("[trace] builder.build_serialized_network failed, exit -1")
    exit(-1)
  engine = runtime.deserialize_cuda_engine(plan)
  print("[trace] Completed creating Engine")
  with open(engine_file_path, "wb") as f:
    f.write(plan)
  return engine

  pass


from albumentations import (
  HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
  Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
  IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
  IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
  ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2


def get_train_transforms():
  return Compose([
    RandomResizedCrop(CFG['img_size'], CFG['img_size']),
    Transpose(p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ShiftScaleRotate(p=0.5),
    HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    CoarseDropout(p=0.5),
    Cutout(p=0.5),
    ToTensorV2(p=1.0),
  ], p=1.)


def get_valid_transforms():
  return Compose([
    CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
    Resize(CFG['img_size'], CFG['img_size']),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0),
  ], p=1.)


class CassvaImgClassifier(nn.Module):
  def __init__(self, model_arch, n_class, pretrained=False):
    super().__init__()
    self.model = timm.create_model(model_arch, pretrained=pretrained)
    n_features = self.model.classifier.in_features
    self.model.classifier = nn.Linear(n_features, n_class)
    '''
    self.model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
        nn.Linear(n_features, n_class, bias=True)
    )
    '''

  def forward(self, x):
    x = self.model(x)
    return x


def prepare_dataloader(df, trn_idx, val_idx, data_root='../input/cassava-leaf-disease-classification/train_images/'):
  print(f'[trace] exec@prepare_dataloader')
  from catalyst.data.sampler import BalanceClassSampler
  train_ = df.loc[trn_idx, :].reset_index(drop=True)
  valid_ = df.loc[val_idx, :].reset_index(drop=True)

  train_ds = CassavaDataset(train_, data_root, transforms=get_train_transforms(), output_label=True,
                            one_hot_label=False, do_fmix=False, do_cutmix=False)
  valid_ds = CassavaDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True)

  train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=CFG['train_bs'],
    pin_memory=False,
    drop_last=False,
    shuffle=True,
    num_workers=CFG['num_workers'],
    # sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
  )
  val_loader = torch.utils.data.DataLoader(
    valid_ds,
    batch_size=CFG['valid_bs'],
    num_workers=CFG['num_workers'],
    shuffle=False,
    pin_memory=False,
  )
  return train_loader, val_loader


from apex import amp
def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):

  print(f'[trace] exec@train_one_epoch')
  model.train()
  t = time.time()
  running_loss = None
  scaler = GradScaler()
  pbar = tqdm(enumerate(train_loader), total=len(train_loader))
  for step, (imgs, image_labels) in pbar:

    imgs = imgs.to(device).float()
    image_labels = image_labels.to(device).long()

    # print(image_labels.shape, exam_label.shape)
    with autocast():

      from torchsummary import summary
      image_preds = model(imgs)  # output = model(input)
      # print(image_preds.shape, exam_pred.shape)

      loss = loss_fn(image_preds, image_labels)

      with amp.scale_loss(loss, optimizer) as scaled_loss:
        #scaled_loss.backward()
        scaler.scale(scaled_loss).backward()

      if running_loss is None:
        running_loss = loss.item()
      else:
        running_loss = running_loss * .99 + loss.item() * .01

      if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
        # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None and schd_batch_update:
          scheduler.step()

      if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
        description = f'epoch {epoch} loss: {running_loss:.4f}'

        pbar.set_description(description)

  delta = time.time() - t
  print(f'[trace] epoch {epoch} takes {delta} seconds')
  if scheduler is not None and not schd_batch_update:
    scheduler.step()


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):

  print(f'[trace] exec@valid_one_epoch')
  model.eval()
  t = time.time()
  loss_sum = 0
  sample_num = 0
  image_preds_all = []
  image_targets_all = []

  pbar = tqdm(enumerate(val_loader), total=len(val_loader))
  for step, (imgs, image_labels) in pbar:
    imgs = imgs.to(device).float()
    image_labels = image_labels.to(device).long()

    image_preds = model(imgs)  # output = model(input)
    # print(image_preds.shape, exam_pred.shape)
    image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
    image_targets_all += [image_labels.detach().cpu().numpy()]
    loss = loss_fn(image_preds, image_labels)
    loss_sum += loss.item() * image_labels.shape[0]
    sample_num += image_labels.shape[0]

    if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
      description = f'epoch {epoch} loss: {loss_sum / sample_num:.4f}'
      pbar.set_description(description)

  image_preds_all = np.concatenate(image_preds_all)
  image_targets_all = np.concatenate(image_targets_all)
  print('validation multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))

  if scheduler is not None:
    if schd_loss_update:
      scheduler.step(loss_sum / sample_num)
    else:
      scheduler.step()


class MyCrossEntropyLoss(_WeightedLoss):
  def __init__(self, weight=None, reduction='mean'):
    super().__init__(weight=weight, reduction=reduction)
    self.weight = weight
    self.reduction = reduction

  def forward(self, inputs, targets):
    lsm = F.log_softmax(inputs, -1)

    if self.weight is not None:
      lsm = lsm * self.weight.unsqueeze(0)

    loss = -(targets * lsm).sum(-1)

    if self.reduction == 'sum':
      loss = loss.sum()
    elif self.reduction == 'mean':
      loss = loss.mean()

    return loss


class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()


def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
