import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class FFTCircularLowPass(nn.Module):
    """
    Low-pass filter: FFT → centred circular mask → IFFT.
    The mask diameter is 512 / Ns (≙ radius = 256 / Ns).
    """
    def __init__(self, Ns: int, img_size: tuple[int, int] = (512, 512)):
        super().__init__()
        H, W = img_size
        # ---- build a (H, W) binary mask ----
        R = H / Ns       # diameter in frequency pixels
        r = R / 2.0          # radius
        yy, xx = torch.meshgrid(
            torch.arange(H) - H // 2,
            torch.arange(W) - W // 2,
            indexing="ij"
        )
        dist = torch.sqrt(xx**2 + yy**2)
        mask = (dist <= r).float()              # 1 inside circle, 0 outside
        # shape (1,1,1,H,W) to broadcast over (B,C,T,H,W)
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        Xf = torch.fft.fftn(x, dim=(-2, -1))
        Xf = torch.fft.fftshift(Xf, dim=(-2, -1))
        Xf = Xf * self.mask
        Xf = torch.fft.ifftshift(Xf, dim=(-2, -1))
        xr = torch.fft.ifftn(Xf, dim=(-2, -1)).real          # drop imag part
        return xr
class MultiScaleBlindDenoiser(nn.Module):
    """
    Take three self-blind CNNs with different kernel sizes, combine
    their outputs with a learnable convex combination, and compare
    that mixture to the noisy input through MSE.
    """
    def __init__(
        self,
        ns_list              = (3, 5, 7),   # spatial kernel sizes
        in_channels          = 1,
        mid_channels         = 8,
        out_channels      = 1,
        Nt                   = 3,           # temporal kernel size
        num_slayers          = 1
    ):
        super().__init__()

        # Three independent back-bones (weights *not* shared)
        self.subnets = nn.ModuleList([
            BlindUNettest(
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                Nt=Nt,
                Ns=ks,
                num_slayers=num_slayers
            )
            for ks in ns_list
        ])

        # Learnable mixing weights (initialised to 1/len(ns_list))
        init_w = torch.ones(len(ns_list), dtype=torch.float32) / len(ns_list)
        self.mix_weights = nn.Parameter(init_w)
        #img_size=(512,512)
        img_size=(1024,1024)
        self.lowpass = FFTCircularLowPass(Ns=min(ns_list), img_size=img_size)

    def forward(self, x):
        """
        x: (B, C, T, X, Y)

        Returns
        -------
        combined : (B, C, T_out, X, Y)
            Weighted sum of the three subnet outputs where
                T_out = T - (Nt - 1)
        """

        # Individual predictions – we only use the main 3-D output
        preds = [net(x) for net in self.subnets]        # list of (B,C,T',X,Y)
        preds = torch.stack(preds, dim=0)                  # (K,B,C,T',X,Y)

        # Positive weights that sum to 1
        weights = F.softmax(self.mix_weights, dim=0)       # (K,)
        weights = weights.view(-1, 1, 1, 1, 1, 1)          # broadcast to preds

        combined = (weights * preds).sum(dim=0)            # (B,C,T',X,Y)
        return combined
    
def normalizer(data):
    maxer=np.max(data)
    miner=np.min(data)
    diff=(maxer-miner)
    data=(data-miner)/diff
    return data,diff,miner
def calc_mse(data1,data2):
    mse = np.mean((data1 - data2) ** 2)
    return mse
def special_padding_3d(input_tensor, pad):
    # input_tensor shape: (batch, channels, depth, height, width)
    # Pads periodic in spatial dimension, reflection in time dimension
    d_pad, h_pad, w_pad = pad

    # Pad along depth (front and back)
    #print("input_tensor",input_tensor.size(),pad)
 
    
    # Pad along height (top and bottom)
    if h_pad > 0:
        input_tensor = torch.cat([input_tensor[:,:, :, -h_pad:, :], input_tensor, input_tensor[:,:, :, :h_pad, :]], dim=3)
    
    # Pad along width (left and right)
    if w_pad > 0:
        input_tensor = torch.cat([input_tensor[:,:, :, :, -w_pad:], input_tensor, input_tensor[:,:, :, :, :w_pad]], dim=4)

    #if d_pad > 0:
    #    input_tensor = torch.cat([input_tensor[:,:, -d_pad:, :, :], input_tensor, input_tensor[:,:, :d_pad, :, :]], dim=2)
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
        #print("xshape",x.size)
        x = special_padding_3d(x, padding)
        return self.conv(x)

class BlindUNettest(nn.Module):
    def __init__(
        self,
        in_channels=1,
        mid_channels=1,
        out_channels=1,
        Nt=3, Ns=5,
        num_slayers=1,
    ):
        super(BlindUNettest, self).__init__()       
        layers = []
        # Define the input and output channels for each layer
        conv_in_channels = in_channels
        conv_out_channels = mid_channels
        # Add the CustomConv3D layer
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
        # Use nn.Sequential to stack the layers
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)