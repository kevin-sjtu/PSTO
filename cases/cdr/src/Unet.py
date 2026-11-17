
import math
import torch
from torch import nn
import torch.nn.functional as F
from FNO import xavier_conv_


def periodic_resize(x, size):
    height, width = x.shape[-2:]
    gy = (torch.arange(size[0], device=x.device, dtype=x.dtype) + .5) * height / size[0] + 1.0
    gx = (torch.arange(size[1], device=x.device, dtype=x.dtype) + .5) * width / size[1] + 1.0
    gy, gx = torch.meshgrid(2 * gy / (height + 2) - 1, 2 * gx / (width + 2) - 1, indexing='ij')
    grid = torch.stack((gx, gy), -1)[None].expand(x.shape[0], -1, -1, -1)
    return F.grid_sample(F.pad(x, (1, 1, 1, 1), mode='circular'), grid, align_corners=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = math.gcd(out_channels, 8)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="circular", bias=False),
            nn.GroupNorm(groups, out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="circular", bias=False),
            nn.GroupNorm(groups, out_channels), nn.GELU())
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d): xavier_conv_(layer)

    def forward(self, x):
        return self.net(x)


class UNetCore(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, depth):
        super().__init__()
        if depth < 1: raise ValueError('U-Net depth must be positive')
        self.encoders = nn.ModuleList()
        widths = [hidden_channels * 2**i for i in range(depth)]
        current = in_channels
        for width in widths:
            self.encoders.append(DoubleConv(current, width)); current = width
        self.bottleneck = DoubleConv(current, current * 2)
        current *= 2
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths):
            self.upconvs.append(nn.ConvTranspose2d(current, width, 2, stride=2))
            self.decoders.append(DoubleConv(width * 2, width)); current = width
        self.output = nn.Conv2d(current, out_channels, 1)
        xavier_conv_(self.output)
        for layer in self.upconvs: xavier_conv_(layer)

    def forward(self, x):
        skips=[]
        for encoder in self.encoders:
            x=encoder(x); skips.append(x)
            x=F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="circular")
            x=F.avg_pool2d(x, 2)
        x=self.bottleneck(x)
        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x=up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x=periodic_resize(x, skip.shape[-2:])
            x=decoder(torch.cat((skip,x),dim=1))
        return self.output(x)


class Unet(nn.Module):
    def __init__(self, num_phys_vars, hidden_channels=23, n_layers=4, dt=1e-6):
        super().__init__()
        self.hidden_channels, self.n_layers = hidden_channels, n_layers
        self.register_buffer('time_step', torch.tensor(float(dt)))
        self.lift = nn.Conv2d(num_phys_vars, hidden_channels, 1)
        self.core = UNetCore(hidden_channels, hidden_channels, hidden_channels, n_layers)
        self.projection = nn.Conv2d(hidden_channels, num_phys_vars, 1)
        xavier_conv_(self.lift); xavier_conv_(self.projection)

    def forward(self, x):
        prediction = x+self.time_step.to(x.dtype)*self.projection(self.core(self.lift(x)))
        return prediction.clamp(-1.0, 1.0)
