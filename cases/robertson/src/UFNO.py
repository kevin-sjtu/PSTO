


import torch
from torch import nn
import torch.nn.functional as F
from FNO import FNOBlock, SpectralConv2d, xavier_conv_
from Unet import UNetCore


class UFourierBlock(nn.Module):
    def __init__(self, channels, modes, unet_depth=2):
        super().__init__()
        self.spectral=SpectralConv2d(channels, channels, modes, modes)
        self.local=nn.Conv2d(channels, channels, 1)
        self.unet=UNetCore(channels, channels, channels, unet_depth)
        xavier_conv_(self.local)

    def forward(self,x):
        return F.gelu(self.spectral(x)+self.local(x)+self.unet(x))


class UFNO(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=23,
                 modes=16, n_layers=4, dt=1e-6):
        super().__init__()
        if n_layers < 1: raise ValueError('U-FNO requires at least one layer')
        self.hidden_channels,self.n_layers,self.modes=hidden_channels,n_layers,modes
        self.output_channels=output_channels
        self.register_buffer('time_step',torch.tensor(float(dt)))
        self.lift=nn.Conv2d(input_channels,hidden_channels,1)
        n_fourier=n_layers//2
        self.blocks=nn.ModuleList([FNOBlock(hidden_channels,modes) if i<n_fourier
                                  else UFourierBlock(hidden_channels,modes)
                                  for i in range(n_layers)])
        self.projection=nn.Conv2d(hidden_channels,output_channels,1)
        xavier_conv_(self.lift);xavier_conv_(self.projection)

    def forward(self,x):
        z=self.lift(x)
        for block in self.blocks:z=block(z)
        prediction = x[:,:self.output_channels]+self.time_step.to(x.dtype)*self.projection(z)
        return prediction.clamp(-1.0, 1.0)
