import sys
from pathlib import Path

import numpy as np
import torch

from wcvd import FFTCircularLowPass, WCVD


if len(sys.argv) not in (2, 3):
    raise SystemExit(
        "Usage: python inference.py <data name or .npy path> [weight name]"
    )

repo_path = Path(__file__).resolve().parents[1]
data_argument = Path(sys.argv[1])
if data_argument.suffix == ".npy":
    input_path = data_argument
else:
    input_path = repo_path / "data" / f"{data_argument}.npy"

output_path = repo_path / "output"
output_path.mkdir(parents=True, exist_ok=True)
if len(sys.argv) == 3:
    weight_name = Path(sys.argv[2]).name
    if not weight_name.endswith(".pt"):
        weight_name = f"{weight_name}.pt"
else:
    weight_name = "denoising_model_weights.pt"
weights_path = repo_path / "weights" / weight_name

# Increase or decrease this value depending on the available GPU memory.
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_read = np.load(input_path)
if data_read.ndim != 3:
    raise ValueError(
        f"Input data must have shape (Nt, Nx, Ny), but received {data_read.shape}."
    )
if data_read.shape[0] < 3:
    raise ValueError("Input data must contain at least three frames.")

# Input intensities should use scaling similar to the model's training data.
video = torch.from_numpy(np.asarray(data_read, dtype=np.float32))

denoise_net = WCVD(num_slayers=4, mid_channels=3).to(device)
state_dict = torch.load(weights_path, map_location=device, weights_only=True)
denoise_net.load_state_dict(state_dict)
denoise_net.eval()

image_size = (video.shape[1], video.shape[2])
lowpass = FFTCircularLowPass(Ns=3, img_size=image_size).to(device)
number_of_windows = video.shape[0] - 2
denoised_batches = []

# Three input frames produce one denoised center frame. The output therefore
# contains Nt - 2 frames and excludes the first and last input frames.
with torch.inference_mode():
    for start in range(0, number_of_windows, batch_size):
        stop = min(start + batch_size, number_of_windows)
        windows = torch.stack(
            [video[index:index + 3] for index in range(start, stop)]
        )
        windows = windows.unsqueeze(1).to(device)
        predictions = lowpass(denoise_net(windows))
        denoised_batches.append(predictions[:, 0, 0].cpu())

denoised_data = torch.cat(denoised_batches, dim=0).numpy()
save_path = output_path / f"{input_path.stem}_denoised.npy"
np.save(save_path, denoised_data)

print(f"Device: {device}")
print(f"Input shape: {data_read.shape}")
print(f"Output shape: {denoised_data.shape}")
print(f"Saved denoised data to: {save_path}")
