import math
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

def pixel_shuffle(input, upscale_factor, depth_first=False):
    batch_size, channels, in_height, in_width = input.size()
    channels //= upscale_factor ** 2

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    if not depth_first:
        input_view = input.contiguous().view(
            batch_size, channels, upscale_factor, upscale_factor,
            in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
        return shuffle_out.view(batch_size, channels, out_height, out_width)
    else:
        input_view = input.contiguous().view(batch_size, upscale_factor, upscale_factor, channels, in_height, in_width)
        shuffle_out = input_view.permute(0, 4, 1, 5, 2, 3).contiguous().view(batch_size, out_height, out_width,
                                                                             channels)
        return shuffle_out.permute(0, 3, 1, 2)

class DarkDataset(Dataset):
    def __init__(self, txt_path, ps = 512):
        
        self.ps = ps
        lines = open(txt_path).read().split('\n')
        self.data_info = np.array([line.split()[0:2] for line in lines])
        self.image_arr = self.data_info[:, 0]
        self.output_arr = self.data_info[:, 1]
        self.data_len = len(self.data_info)

    def __getitem__(self, index):
      single_image_name = self.image_arr[index]
      single_output_name = self.output_arr[index]
      img_exp = float(os.path.basename(single_image_name)[9:-5])
      out_exp = float(os.path.basename(single_output_name)[9:-5])
      ampl_ratio = np.minimum(out_exp / img_exp, 300.0)
      
      img_as_img = np.transpose(np.load(single_image_name) * ampl_ratio, (2,0,1)) # Normalised input image of dim 4 * H * W
      output_as_img = np.transpose(np.load(single_output_name)/255, (2,0,1)) # Normalised ouput image of dim 3 * H * W
      
      if self.ps:
        H, W = img_as_img.shape[1:3]
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        img_patch = img_as_img[:, yy:yy + self.ps, xx:xx + self.ps]
        output_patch = output_as_img[:, yy*2:yy*2 + self.ps*2 , xx*2:xx*2 + self.ps*2]

        if np.random.randint(3, size = 1)[0] == 1: # Vertical flip
          #print('V')
          img_patch = np.flip(img_patch, axis = 1)
          output_patch = np.flip(output_patch, axis = 1)

        if np.random.randint(3, size = 1)[0] == 1: # Horizontal flip
          #print('H')
          img_patch = np.flip(img_patch, axis = 2)
          output_patch = np.flip(output_patch, axis = 2)
        
        if np.random.randint(3, size = 1)[0] == 1: # Random transpose
          #print('T')
          img_patch = np.transpose(img_patch, (0, 2, 1))
          output_patch = np.transpose(output_patch, (0, 2, 1))
        
        return torch.from_numpy(img_patch.copy()).float(), torch.from_numpy(output_patch.copy()).float()
      else:
        return torch.from_numpy(img_as_img.copy()).float(), torch.from_numpy(output_as_img.copy()).float()
              
    def __len__(self):
        return self.data_len

class UNet(nn.Module):

  def contraction_block(self, in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(ConvLayer(in_channels, out_channels, kernel_size, 1),
                          nn.ReLU(),
                          nn.InstanceNorm2d(out_channels),
                          ConvLayer(out_channels, out_channels, kernel_size, 1),
                          nn.ReLU(),
                          nn.InstanceNorm2d(out_channels)
              )
    return block
  
  def expansion_block(self, in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(ConvLayer(in_channels, out_channels, kernel_size, 1),
                          nn.ReLU(),
                          nn.InstanceNorm2d(out_channels),
                          ConvLayer(out_channels, out_channels, kernel_size, 1),
                          nn.ReLU(),
                          nn.InstanceNorm2d(out_channels)
              )
    return block
  
  def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            required_H = upsampled.shape[2]
            H = bypass.shape[2]
            required_W = upsampled.shape[3]
            W = bypass.shape[3]
            if H % 2 == 0:
              bypass = bypass[:, :, (H//2) - (required_H//2) : (H//2) + (required_H//2), :]
            else:
              bypass = bypass[:, :, (H//2) - (required_H//2) + 1 : (H//2) + (required_H//2) + 1, :]
            
            if W % 2 == 0:
              bypass = bypass[:, :, :, (W//2) - (required_W//2) : (W//2) + (required_W//2)]
            else:
              bypass = bypass[:, :, :, (W//2) - (required_W//2) + 1 : (W//2) + (required_W//2) + 1]
            return torch.cat((upsampled, bypass), 1)
      
  def __init__(self):
    super(UNet, self).__init__()

    self.maxpool = nn.MaxPool2d(2, stride = 2)
    self.conv1 = self.contraction_block(4, 32, 3)
    self.conv2 = self.contraction_block(32, 64, 3)
    self.conv3 = self.contraction_block(64, 128, 3)
    self.conv4 = self.contraction_block(128, 256, 3)
    self.conv5 = self.contraction_block(256, 512, 3)
    
    self.upconv1 = UpsampleConvLayer(512, 256, 3, 1, upsample = 2)
    self.conv6 =  self.expansion_block(512, 256, 3)

    self.upconv2 = UpsampleConvLayer(256, 128, 3, 1, upsample = 2)
    self.conv7 =  self.expansion_block(256, 128, 3)

    self.upconv3 = UpsampleConvLayer(128, 64, 3, 1, upsample = 2)
    self.conv8 =  self.expansion_block(128, 64, 3)

    self.upconv4 = UpsampleConvLayer(64, 32, 3, 1, upsample = 2)
    self.conv9 =  self.expansion_block(64, 32, 3)

    self.conv10  = ConvLayer(32, 12, 1, 1)

  def forward(self, x):
    conv1o = self.conv1(x)
    conv2o= self.conv2(self.maxpool(conv1o))
    conv3o= self.conv3(self.maxpool(conv2o))
    conv4o= self.conv4(self.maxpool(conv3o))
    conv5o= self.conv5(self.maxpool(conv4o))

    expan1 = self.crop_and_concat(self.upconv1(conv5o), conv4o, crop=True)
    conv6o = self.conv6(expan1)
    expan2 = self.crop_and_concat(self.upconv2(conv6o), conv3o, crop = True)
    conv7o = self.conv7(expan2)
    expan3 = self.crop_and_concat(self.upconv3(conv7o), conv2o, crop = True)
    conv8o = self.conv8(expan3)
    expan4 = self.crop_and_concat(self.upconv4(conv8o), conv1o, crop = True)
    conv9o = self.conv9(expan4)
    conv10o = self.conv10(conv9o)
    depth_to_space_conv = pixel_shuffle(conv10o, upscale_factor=2, depth_first=True)
    return depth_to_space_conv

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size//2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode = 'nearest', scale_factor = self.upsample)
        
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size//2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out