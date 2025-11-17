import os, sys
import torch
import numpy as np
from dudvd_ensemble import *

datname=sys.argv[1]
outpath=f'../output'
os.makedirs(outpath,exist_ok=True)
dtype = torch.float32
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

datread=np.load(f'../data/{datname}.npy')
if len(datread.shape)==3:
    data=torch.from_numpy(datread).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
if len(data.shape)!=5:
    print("Input data shape needs to be (Nt,Nx,Ny)")
    quit()

denoise_net = MultiScaleBlindDenoiser(num_slayers=3,mid_channels=4).to(device)
denoise_net=torch.load(f"../weights/denoising_model.pt",weights_only=False).to(device)
img_size=(data.shape[3],data.shape[4])
lowpass = FFTCircularLowPass(Ns=3, img_size=img_size).to(device)
denoised_data=lowpass(denoise_net(data))
np.save(f'{outpath}/{datname}_denoised.npy',denoised_data.detach().cpu().numpy()[0,0,:,:,:])
