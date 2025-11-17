import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FFTCircularLowPass(nn.Module):
    def __init__(self, Ns: int, img_size: tuple[int, int] = (512, 512)):
        super().__init__()
        H, W = img_size
        R = H / Ns     
        r = R / 2.0    
        yy, xx = torch.meshgrid(
            torch.arange(H) - H // 2,
            torch.arange(W) - W // 2,
            indexing="ij"
        )
        dist = torch.sqrt(xx**2 + yy**2)
        mask = (dist <= r).float()            
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Xf = torch.fft.fftn(x, dim=(-2, -1))
        Xf = torch.fft.fftshift(Xf, dim=(-2, -1))
        Xf = Xf * self.mask
        Xf = torch.fft.ifftshift(Xf, dim=(-2, -1))
        xr = torch.fft.ifftn(Xf, dim=(-2, -1)).real
        return xr
    
class WCVD(nn.Module):
    def __init__(
        self,
        ns_list              = (3, 5, 7), 
        in_channels          = 1,
        mid_channels         = 4,
        out_channels      = 1,
        Nt                   = 3,         
        num_slayers          = 3
    ):
        super().__init__()

        self.subnets = nn.ModuleList([
            BlindCNN(
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                Nt=Nt,
                Ns=ks,
                num_slayers=num_slayers
            )
            for ks in ns_list
        ])

        init_w = torch.ones(len(ns_list), dtype=torch.float32) / len(ns_list)
        self.mix_weights = nn.Parameter(init_w)
        img_size=(1024,1024)
        self.lowpass = FFTCircularLowPass(Ns=min(ns_list), img_size=img_size)

    def forward(self, x):
        preds = [net(x) for net in self.subnets]        # list of (B,C,T',X,Y)
        preds = torch.stack(preds, dim=0)                  # (K,B,C,T',X,Y)

        weights = F.softmax(self.mix_weights, dim=0)       # (K,)
        weights = weights.view(-1, 1, 1, 1, 1, 1)          # broadcast to preds

        combined = (weights * preds).sum(dim=0)            # (B,C,T',X,Y)
        return combined


def special_padding_3d(input_tensor, pad):
    d_pad, h_pad, w_pad = pad
    if h_pad > 0:
        input_tensor = torch.cat([input_tensor[:,:, :, -h_pad:, :], input_tensor, input_tensor[:,:, :, :h_pad, :]], dim=3)
    if w_pad > 0:
        input_tensor = torch.cat([input_tensor[:,:, :, :, -w_pad:], input_tensor, input_tensor[:,:, :, :, :w_pad]], dim=4)
    return input_tensor

class BlindConv3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, Nt=3,Ns=5):
        super(BlindConv3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(Nt,Ns,Ns),
            padding=0,
            bias=False
        )
        self.Nt=Nt
        self.Ns=Ns
        nn.init.kaiming_normal_(self.conv.weight)
        Tcenter = Nt//2
        Scenter=Ns//2
        with torch.no_grad():
            self.conv.weight[..., Tcenter, Scenter, Scenter] = 0.0
        self.conv.weight.register_hook(self._zero_central_grad)

    def _zero_central_grad(self, grad):
        Tcenter = self.Nt // 2
        Scenter=self.Ns//2
        grad[..., Tcenter, Scenter, Scenter] = 0.0
        return grad

    def forward(self, x):
        padding = (self.Nt//2, self.Ns//2, self.Ns//2)
        x = special_padding_3d(x, padding)
        return self.conv(x)

class BlindCNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        mid_channels=4,
        out_channels=1,
        Nt=3, Ns=5,
        num_slayers=3,
    ):
        super(BlindCNN, self).__init__()       
        layers = []
        conv_in_channels = in_channels
        conv_out_channels = mid_channels
        layers.append(BlindConv3D(
            in_channels=conv_in_channels,
            out_channels=conv_out_channels,
            Nt=Nt,Ns=Ns
        ))
          
        for j in range(num_slayers):
            layers.append(nn.Conv3d(
            mid_channels,
            mid_channels,
            kernel_size=1,
            padding=0,
            bias=False))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        layers.append(nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False))
        layers.append(nn.ReLU())            
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
