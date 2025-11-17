import os, sys
import torch
import numpy as np
from wcvd import *

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

denoise_net = WCVD(num_slayers=4,mid_channels=3).to(device)
state_dict=torch.load(f"../weights/denoising_model_weights.pt",map_location=device,weights_only=True)
denoise_net.load_state_dict(state_dict)
denoise_net.eval()
denoise_net.to(device)
img_size=(data.shape[3],data.shape[4])
lowpass = FFTCircularLowPass(Ns=3, img_size=img_size).to(device)
denoised_data=lowpass(denoise_net(data))
np.save(f'{outpath}/{datname}_denoised.npy',denoised_data.detach().cpu().numpy()[0,0,:,:,:])
