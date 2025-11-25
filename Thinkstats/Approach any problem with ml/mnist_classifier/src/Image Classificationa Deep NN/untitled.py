#simple_unet.py

import torch
import torch.nn as nn
from torch.nn import functional as F

def double_conv(in_channels,out_channels):
    conv=nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3)
        nn.ReLU(inplace=True)
        nn.Conv2d(out_channels,out_channels,kernel_size=3)
        nn.ReLU(inplace=True)
    )
    return conv



def crop_tensor(tensor,target_tensor):
    target_size=target_tensor.size()[2]
    tensor_size=tensor.size()[2]
    delta=tensor_size-target_size
    delta=delta//2
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]


class UNet(