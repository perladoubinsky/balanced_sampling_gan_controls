# python 3.7
"""Predicts the celebA attributes."""

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models

from .base_predictor import BasePredictor
from .model_celeba import ResNet50

__all__ = ['CelebAPredictor']

FEATURE_DIM = 512

class CelebAPredictor(BasePredictor):
    
  def __init__(self):
    super().__init__('celeba')

  def build(self):
    self.net = ResNet50(pretrained=True)

  def load(self):
    # Transform for input images.
    self.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    # Load pre-trained weights.
    checkpoint = torch.load(self.weight_path)
    state_dict = {k.replace('module.', ''): v
                  for k, v in checkpoint['state_dict'].items()}
    self.net.load_state_dict(state_dict)    

  def _predict(self, images):
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')
    if images.dtype != np.uint8:
      raise ValueError(f'Images should be with dtype `numpy.uint8`!')
    if not (len(images.shape) == 4 and
            0 < images.shape[0] <= self.batch_size and
            images.shape[3] == self.image_channels):
      raise ValueError(f'Images should be with shape [batch_size, height '
                       f'width, channel], where `batch_size` no larger than '
                       f'{self.batch_size}, and `channel` equals to '
                       f'{self.image_channels}!\n'
                       f'But {images.shape} received!')

    xs = [self.transform(Image.fromarray(img)).unsqueeze(0) for img in images]
    xs = torch.cat(xs, dim=0).to(self.run_device)

    outputs = self.net(xs)
    (out_smile, out_age, out_gender, out_glasses) = outputs
    softmax = nn.Softmax(dim=1)
    preds_smile = self.get_value(softmax(out_smile)[:,1])
    preds_age = self.get_value(softmax(out_age)[:,1])
    preds_gender = self.get_value(softmax(out_gender)[:,1])
    preds_glasses = self.get_value(softmax(out_glasses)[:,1])

    if self.use_cuda:
      torch.cuda.empty_cache()

    return {'Smile': preds_smile,
            'Age': preds_age,
            'Gender': preds_gender,
            'Glasses': preds_glasses
           }

  def predict(self, images, **kwargs):
    return self.batch_run(images, self._predict)
