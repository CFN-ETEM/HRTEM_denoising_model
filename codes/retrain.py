import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from wcvd import FFTCircularLowPass, WCVD


if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python retrain.py <data name or .npy path> <new weight name>"
    )

repo_path = Path(__file__).resolve().parents[1]
data_argument = Path(sys.argv[1])
if data_argument.suffix == ".npy":
    input_path = data_argument
else:
    input_path = repo_path / "data" / f"{data_argument}.npy"

weight_name = Path(sys.argv[2]).name
if not weight_name.endswith(".pt"):
    weight_name = f"{weight_name}.pt"

output_path = repo_path / "output"
output_path.mkdir(parents=True, exist_ok=True)
weights_path = repo_path / "weights"
weights_path.mkdir(parents=True, exist_ok=True)
starting_weights_path = weights_path / "denoising_model_weights.pt"
new_weights_path = weights_path / weight_name

# These training values can be changed for a specific dataset.
number_of_epochs = 300
batch_size = 4
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_read = np.load(input_path)
if data_read.ndim != 3:
    raise ValueError(
        f"Input data must have shape (Nt, Nx, Ny), but received {data_read.shape}."
    )
if data_read.shape[0] < 3:
    raise ValueError("Input data must contain at least three frames.")

# Keep the original video unchanged for the final denoising step.
# Input intensities should use scaling similar to the original training data.
video = torch.from_numpy(np.asarray(data_read, dtype=np.float32))
number_of_windows = video.shape[0] - 2

denoise_net = WCVD(num_slayers=4, mid_channels=3).to(device)
state_dict = torch.load(starting_weights_path, map_location=device, weights_only=True)
denoise_net.load_state_dict(state_dict)
denoise_net.train()

image_size = (video.shape[1], video.shape[2])
lowpass = FFTCircularLowPass(Ns=3, img_size=image_size).to(device)

# Adam is a typical optimizer. This line can be changed to use another one.
optimizer = Adam(denoise_net.parameters(), lr=learning_rate)
lr_scheduler = ReduceLROnPlateau(optimizer, patience=10)
loss_function = nn.MSELoss()

print(f"Device: {device}")
print(f"Training data shape: {data_read.shape}")

for epoch in range(1, number_of_epochs + 1):
    augmented_video = video

    # Transposition is used only for square images so the image size is unchanged.
    if video.shape[1] == video.shape[2] and np.random.random() > 0.5:
        augmented_video = torch.transpose(augmented_video, -2, -1)
    if np.random.random() > 0.5:
        augmented_video = torch.flip(augmented_video, (-1,))
    if np.random.random() > 0.5:
        augmented_video = torch.flip(augmented_video, (-2,))

    shuffled_indices = torch.randperm(number_of_windows)
    total_loss = 0.0

    for start in range(0, number_of_windows, batch_size):
        batch_indices = shuffled_indices[start:start + batch_size]
        index_list = batch_indices.tolist()
        windows = torch.stack(
            [augmented_video[index:index + 3] for index in index_list]
        )
        targets = augmented_video[batch_indices + 1]
        windows = windows.unsqueeze(1).to(device)
        targets = targets.unsqueeze(1).unsqueeze(2).to(device)

        optimizer.zero_grad()
        predictions = lowpass(denoise_net(windows))
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(index_list)

    mean_loss = total_loss / number_of_windows
    lr_scheduler.step(mean_loss)

    if epoch == 1 or epoch % 10 == 0 or epoch == number_of_epochs:
        current_learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{number_of_epochs}, "
            f"loss: {mean_loss:.6g}, learning rate: {current_learning_rate:.3g}"
        )

# Save a state dictionary that can be loaded directly by inference.py.
torch.save(denoise_net.state_dict(), new_weights_path)

denoise_net.eval()
denoised_batches = []

# Denoise the unchanged video so the saved result has its original orientation.
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

print(f"Output shape: {denoised_data.shape} (first and last frames are excluded)")
print(f"Saved retrained weights to: {new_weights_path}")
print(f"Saved denoised data to: {save_path}")
