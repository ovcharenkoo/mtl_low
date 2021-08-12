import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import utils.backbone as backbone


""" ========================================================
Experiment 1. UNet-L

Parts of the U-Net model are from here
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #             self.up = F.interpolate()
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet_ext(nn.Module):
    def __init__(self, n_channels, n_classes, kernel_size=3, bilinear=True, s=2):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, s * 16)
        self.down1 = Down(s * 16, s * 32)
        self.down2 = Down(s * 32, s * 64)
        self.down3 = Down(s * 64, s * 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(s * 128, s * 256 // factor)
        self.up1 = Up(s * 256, s * 128 // factor, bilinear)
        self.up2 = Up(s * 128, s * 64 // factor, bilinear)
        self.up3 = Up(s * 64, s * 32 // factor, bilinear)
        self.up4 = Up(s * 32, s * 16, bilinear)
        self.outc = OutConv(s * 16, n_classes)
        
    @autocast()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

""" ========================================================
Experiment 2. Multi-L
Based on https://github.com/shepnerd/inpainting_gmcnn
"""
class PureUpsampling(nn.Module):
    def __init__(self, scale=2, mode='bilinear'):
        super().__init__()
        assert isinstance(scale, int)
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        h, w = x.size(2) * self.scale, x.size(3) * self.scale
        if self.mode == 'nearest':
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode)
        else:
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode, align_corners=True)
        return xout

    
class Mixer(nn.Module):
    def __init__(self, n_channels, n_classes, kernel_size=3, bilinear=True, s=2):
        super().__init__()
        ch = 8
        self.using_norm = True
        self.norm = F.instance_norm
        self.act = F.elu
        
        # network structure
        self.EB1 = []
        self.EB1_pad_rec = [3, 3, 3, 3, 3, 3, 6, 12, 24, 48, 3, 3, 0]
        self.EB1.append(nn.Conv2d(n_channels, ch, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch, ch * 2, kernel_size=7, stride=2))
        self.EB1.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=7, stride=2))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=2))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=4))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=8))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=16))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(PureUpsampling(scale=4))
        self.EB1 = nn.ModuleList(self.EB1)
        
        self.EB2 = []
        self.EB2_pad_rec = [2, 2, 2, 2, 2, 2, 4, 8, 16, 32, 2, 2, 0, 2, 2, 0]
        self.EB2.append(nn.Conv2d(n_channels, ch, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch, ch * 2, kernel_size=5, stride=2))
        self.EB2.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=5, stride=2))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=2))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=4))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=8))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=16))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(PureUpsampling(scale=2, mode='nearest'))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 2, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=5, stride=1))
        self.EB2.append(PureUpsampling(scale=2))
        self.EB2 = nn.ModuleList(self.EB2)
        
        self.EB3 = []
        self.EB3_pad_rec = [1, 1, 1, 1, 1, 1, 2, 4, 8, 16, 1, 1, 0, 1, 1, 0, 1, 1]
        self.EB3.append(nn.Conv2d(n_channels, ch, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2))
        self.EB3.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=3, stride=2))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=2))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=4))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=8))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=16))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(PureUpsampling(scale=2, mode='nearest'))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 2, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=3, stride=1))
        self.EB3.append(PureUpsampling(scale=2, mode='nearest'))
        self.EB3.append(nn.Conv2d(ch * 2, ch, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch, ch, kernel_size=3, stride=1))
        self.EB3 = nn.ModuleList(self.EB3)
        
        self.decoding_layers = []
        self.decoding_layers.append(nn.Conv2d(56, ch // 2, kernel_size=3, stride=1))
        self.decoding_layers.append(nn.Conv2d(ch // 2, n_classes, kernel_size=3, stride=1))
 
        self.decoding_pad_rec = [1, 1]
        self.decoding_layers = nn.ModuleList(self.decoding_layers)

        # padding operations
        padlen = 49
        self.pads = [0] * padlen
        for i in range(padlen):
            self.pads[i] = nn.ReflectionPad2d(i)
        self.pads = nn.ModuleList(self.pads)
    
    @autocast()
    def forward(self, x):
        x1, x2, x3 = x, x, x
        
        for i, layer in enumerate(self.EB1):
            pad_idx = self.EB1_pad_rec[i]
            x1 = layer(self.pads[pad_idx](x1))
            if self.using_norm:
                x1 = self.norm(x1)
            if pad_idx != 0:
                x1 = self.act(x1)
                
        for i, layer in enumerate(self.EB2):
            pad_idx = self.EB2_pad_rec[i]
            x2 = layer(self.pads[pad_idx](x2))
            if self.using_norm:
                x2 = self.norm(x2)
            if pad_idx != 0:
                x2 = self.act(x2)
        
        for i, layer in enumerate(self.EB3):
            pad_idx = self.EB3_pad_rec[i]
            x3 = layer(self.pads[pad_idx](x3))
            if self.using_norm:
                x3 = self.norm(x3)
            if pad_idx != 0:
                x3 = self.act(x3)
        
        x_d = torch.cat((x3.clone(), x2.clone(), x1.clone()), 1)
        x_d = self.act(self.decoding_layers[0](self.pads[self.decoding_pad_rec[0]](x_d)))
        x_d = self.decoding_layers[1](self.pads[self.decoding_pad_rec[1]](x_d))
        return x_d, x_d
    
    
""" ========================================================
Experiments 3-5. Multi-{LM, LC, LCM}
"""
# Model head (M)
class HeadOld(nn.Module):
    def __init__(self, layers=[(56, 16), (16, 1)], kernel_sizes=[3, 3], strides=[1, 1]):
        super().__init__()
        self.act = F.leaky_relu
        self.decoding_layers = []
        for ilayer, (ch_inp, ch_out) in enumerate(layers):
            self.decoding_layers.append(nn.Conv2d(ch_inp, ch_out, 
                                                  kernel_size=kernel_sizes[ilayer], stride=strides[ilayer]))
        self.decoding_pad_rec = [1, 1]
        self.decoding_layers = nn.ModuleList(self.decoding_layers)
        
        # padding operations
        padlen = 49
        self.pads = [0] * padlen
        for i in range(padlen):
            self.pads[i] = nn.ReflectionPad2d(i)
        self.pads = nn.ModuleList(self.pads)
    
    @autocast()
    def forward(self, x):
        x = self.act(self.decoding_layers[0](self.pads[self.decoding_pad_rec[0]](x)))
        x = self.decoding_layers[1](self.pads[self.decoding_pad_rec[1]](x))
        return torch.clip(x, -1, 1)
    
    
# Low-frequency data head (L)
class Head(nn.Module):
    def __init__(self, layers, layers_out, kernel_sizes, strides, pads):
        super().__init__()
        self.act = F.leaky_relu
        self.decoding_layers = []
        
        # Upscale to the target size
        for ilayer, (ch_inp, ch_out) in enumerate(layers):
            self.decoding_layers.append(nn.ConvTranspose2d(ch_inp, ch_out, 
                                                            kernel_size=kernel_sizes[ilayer], 
                                                            stride=strides[ilayer],
                                                            padding=pads[ilayer]))
            self.decoding_layers.append(nn.LeakyReLU())
        
        # After reaching the target size, do this
        self.out_layers = []
        self.out_layers.append(nn.Conv2d(layers_out[0], layers_out[0], kernel_size=3, padding=1))
        self.out_layers.append(nn.Conv2d(layers_out[0], layers_out[1],kernel_size=1))
        
        # Convert to Pytorch modules
        self.decoding_layers = nn.ModuleList(self.decoding_layers)
        self.out_layers = nn.ModuleList(self.out_layers)
        
    @autocast()
    def forward(self, x):
        for layer in self.decoding_layers:
            x = layer(x)
        for layer in self.out_layers:
            x = layer(x)
        return torch.clip(x, -1, 1)

# Encoder
# Based on https://github.com/shepnerd/inpainting_gmcnn
class Encoder(nn.Module):
    def __init__(self, n_channels, kernel_size=3, bilinear=True, s=2, ch=8):
        super().__init__()
        self.using_norm = True
        self.norm = F.instance_norm
        self.act = F.leaky_relu
        
        # network structure
        self.EB1 = []
        self.EB1_pad_rec = [3, 3, 3, 3, 3, 3, 6, 12, 24, 48, 3, 3, 0]
        self.EB1.append(nn.Conv2d(n_channels, ch, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch, ch * 2, kernel_size=7, stride=2))
        self.EB1.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=7, stride=2))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=2))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=4))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=8))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=16))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1 = nn.ModuleList(self.EB1)
        
        self.EB2 = []
        self.EB2_pad_rec = [2, 2, 2, 2, 2, 2, 4, 8, 16, 32, 2, 2, 0, 2, 2, 0]
        self.EB2.append(nn.Conv2d(n_channels, ch, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch, ch * 2, kernel_size=5, stride=2))
        self.EB2.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=5, stride=2))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=2))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=4))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=8))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=16))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2 = nn.ModuleList(self.EB2)
        
        self.EB3 = []
        self.EB3_pad_rec = [1, 1, 1, 1, 1, 1, 2, 4, 8, 16, 1, 1, 0, 1, 1, 0, 1, 1]
        self.EB3.append(nn.Conv2d(n_channels, ch, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2))
        self.EB3.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=3, stride=2))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=2))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=4))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=8))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=16))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3 = nn.ModuleList(self.EB3)
        
        self.decoding_pad_rec = [1, 1]
        self.decoding_layers = []
        
        self.decoding_layers.append(nn.Conv2d(96, 96, kernel_size=3, stride=1))
        self.decoding_layers.append(nn.Conv2d(96, 96, kernel_size=3, stride=1))
        self.decoding_layers = nn.ModuleList(self.decoding_layers)
        
        # padding operations
        padlen = 49
        self.pads = [0] * padlen
        for i in range(padlen):
            self.pads[i] = nn.ReflectionPad2d(i)
        self.pads = nn.ModuleList(self.pads)
    
    @autocast()
    def forward(self, x):
        x1, x2, x3 = x, x, x
        
        for i, layer in enumerate(self.EB1):
            pad_idx = self.EB1_pad_rec[i]
            x1 = layer(self.pads[pad_idx](x1))
            if self.using_norm:
                x1 = self.norm(x1)
            if pad_idx != 0:
                x1 = self.act(x1)
                
        for i, layer in enumerate(self.EB2):
            pad_idx = self.EB2_pad_rec[i]
            x2 = layer(self.pads[pad_idx](x2))
            if self.using_norm:
                x2 = self.norm(x2)
            if pad_idx != 0:
                x2 = self.act(x2)
        
        for i, layer in enumerate(self.EB3):
            pad_idx = self.EB3_pad_rec[i]
            x3 = layer(self.pads[pad_idx](x3))
            if self.using_norm:
                x3 = self.norm(x3)
            if pad_idx != 0:
                x3 = self.act(x3)
        
        x = torch.cat((x3.clone(), x2.clone(), x1.clone()), 1)
        x = self.act(self.decoding_layers[0](self.pads[self.decoding_pad_rec[0]](x)))
        return x
    

    
""" ========================================================
Classes for inference and manipulation of multiple models.
"""

class Wrapper(backbone.BaseModel):
    def __init__(self, encoder, head1, head2, load_dir, gan=False):
        super().__init__()
        self.device = 0
        self.load_dir = load_dir
        
        # Init architectures
        self.net_encoder = encoder
        self.net_data = head1
        self.net_model = head2
        
        self.loss_weights = [torch.ones((1,), requires_grad=False) for _ in range(3)]
            
        self.phases = ['train', 'val']
        self.running_metrics_encoder = {}
        self.running_metrics_lr = {'train': {'lr_data': [], 'lr_model': []}}
        self.running_metrics_data = {'train': {'w1': [], 'w2': [], 'w3': [], 'w_s1s2s3': []}}
        for p in self.phases:
            self.running_metrics_encoder[p] = {'data': [], 'model': []}
            
        self.model_names = ['_encoder', '_data', '_model']

        self.load_networks(load_dir, 0)
        self.load_history(load_dir)
        self.load_lr_history(load_dir)
        self.load_sigmas(load_dir)
        
        self.net_encoder = self.net_encoder.to(self.device)
        self.net_data = self.net_data.to(self.device)
        self.net_model = self.net_model.to(self.device)
        
        self.net_encoder.eval()
        self.net_data.eval()
        self.net_model.eval()
    
    def load_sigmas(self, load_dir):
        print(f'Load sigmas from {load_dir}')
        try:
            available_weights = self.loss_weights
            self.loss_weights = [available_weights[iw].clone() * np.load(os.path.join(load_dir, f's{iw}.npy')).astype(np.float32) 
                                 for iw in range(len(self.loss_weights))]
        except Exception as e:
            print(f'Failed to load sigmas from {load_dir}! {e}')
        
    def from_numpy(self, x):
        if len(x.shape) < 4:
            ndim_old = len(x.shape)
            ndiff = 4 - ndim_old
            for _ in range(ndiff):
                x = np.expand_dims(x, 0)
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

    
class OldWrapper(backbone.BaseModel):
    def __init__(self, ar, load_dir):
        super().__init__()
        self.device = 0
        
        # Init architectures
        self.net_g = ar
        self.model_names = ['_g']
        self.load_networks(load_dir, 0)
        
        self.net_g = self.net_g.to(self.device)
        self.net_g.eval()
        
        
class Blend:
    def __init__(self, ens):
        self.ens = ens
    
    def predict(self, h, pred_idx=0, pred_chan=0, return_type='mean'):
        preds = [net.predict(h, pred_idx, pred_chan, return_type) for net in self.ens]
        if return_type == 'list':
            preds = [p for sublist in preds for p in sublist]
            
        if return_type == 'list':
            return preds
        else:
            return sum(preds) / len(preds)
        
        
class Ensemble(Wrapper):
    def __init__(self, encoder, head1, head2, load_dir, num_ens=1,
                 old=False, single_out=False, gan=False):
        self.nets = []
        self.old=old
        self.single_out = single_out
        if isinstance(num_ens, int):
            num_ens = [i for i in range(num_ens)]
            
        for i in num_ens:
            this_load_dir = load_dir[:-1]+f'_{i}' if len(num_ens) > 1 else load_dir
            
            if old:
                self.nets.append(OldWrapper(copy.deepcopy(encoder), this_load_dir))
            else:
                self.nets.append(Wrapper(copy.deepcopy(encoder), 
                                         copy.deepcopy(head1), 
                                         copy.deepcopy(head2), this_load_dir, gan))
        self.device = 0
    
    def encode(self, h):
        if isinstance(h, np.ndarray):
            h = self.from_numpy(h)
 
        preds = []
        for net in self.nets:
            x = net.net_encoder(h)
            preds.append(x.detach().cpu().numpy())
        return sum(preds) / len(self.nets)
    
    def predict(self, h, pred_idx=0, pred_chan=0, return_type='mean'):
        if isinstance(h, np.ndarray):
            h = self.from_numpy(h)
 
        preds = []
        for net in self.nets:
            if self.single_out:
                lup = net.net_g(h)
                mp = torch.zeros_like(h)
            else:
                if self.old:
                    lup, mp = net.net_g(h)
                else:
                    x = net.net_encoder(h)
                    lup = net.net_data(x)
                    mp = net.net_model(x)

            pred = (lup, mp)
            pl = pred[pred_idx]
            
            if pl.shape[1] > 1:
                outs = pl.detach().cpu().numpy()[0, 1,...], pl.detach().cpu().numpy()[0, 0,...]
            else:
                outs = pl.detach().cpu().numpy()[0, 0,...], pl.detach().cpu().numpy()[0, 0,...]
            preds.append(outs)
            
        preds = [p[pred_chan].astype(np.float32) for p in preds]
        if return_type == 'list':
            return preds
        else:
            return sum(preds) / len(preds)