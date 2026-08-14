# HRTEM Denoising Model

This repository contains the Weight Centric Video Denoising model from following paper:

Lee, B., Li, M., Yang, J.C. et al. Machine learning pipeline for denoising low signal-to-noise ratio and out-of-distribution transmission electron microscopy datasets. npj Comput Mater (2026). https://doi.org/10.1038/s41524-026-02193-9

Questions about this work can be directed to: blee2@bnl.gov

It supports:

- Inference using the supplied pretrained model weights.
- Self-supervised retraining on a user's noisy HRTEM video.
- Automatic use of an NVIDIA CUDA GPU when one is available, with CPU fallback.

## Repository Structure

```text
codes/
  inference.py       Run inference with pretrained or retrained weights
  retrain.py         Retrain the model on a noisy video and denoise it
  wcvd.py            Model and low-pass filter definitions
data/
  au.npy              Example HRTEM video
  ...
weights/
  denoising_model_weights.pt
output/               Denoised videos are saved here
requirements.txt      Python dependencies
```

## Installation

Clone the repository and enter its root directory:

```bash
git clone https://github.com/CFN-ETEM/HRTEM_denoising_model.git
cd HRTEM_denoising_model
```

Creating a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The code was tested with Python 3.12.2, NumPy 1.26.4, and PyTorch 2.5.1. If a particular CUDA build of PyTorch is required, install the appropriate PyTorch package for the system before installing the requirements.

## Input Data

Input videos must be NumPy `.npy` arrays with shape:

```text
(Nt, Nx, Ny)
```

Here, `Nt` is the number of video frames and `Nx` and `Ny` are the spatial dimensions. At least three frames are required. Data are converted to 32-bit floating point before they are passed to the model.

For best results, use the same preprocessing and similar intensity scaling as the data used to train the supplied model weights.

The model uses three consecutive frames to predict their center frame. Therefore, an input with shape `(Nt, Nx, Ny)` produces an output with shape `(Nt - 2, Nx, Ny)`. The first and last input frames are not included in the output.

## Inference with Pretrained Weights

From the repository root, denoise the included example with:

```bash
python codes/inference.py au
```

The name `au` is resolved as `data/au.npy`. A direct path to another `.npy` file can also be used:

```bash
python codes/inference.py /path/to/video.npy
```

By default, inference uses:

```text
weights/denoising_model_weights.pt
```

The denoised video is saved as:

```text
output/<input_name>_denoised.npy
```

For example, `data/au.npy` produces `output/au_denoised.npy`.

### Inference with Retrained Weights

To use another weight file from the `weights` directory, provide its name as the second argument:

```bash
python codes/inference.py au my_retrained_weights
```

This command loads `weights/my_retrained_weights.pt`. Including the `.pt` extension is optional.

The inference batch size is set by `batch_size` near the beginning of `codes/inference.py`. Reduce it if GPU memory is limited, or increase it if more GPU memory is available.

## Retraining and Denoising

Retraining fine-tunes the supplied pretrained weights on a specified noisy video. Clean target data are not required. The blind-spot model is trained to predict each center frame from neighboring spatial and temporal information.

Run retraining with:

```bash
python codes/retrain.py au my_retrained_weights
```

An explicit input path can also be used:

```bash
python codes/retrain.py /path/to/video.npy my_retrained_weights
```

The default training settings are:

- 300 epochs
- Adam optimizer
- Learning rate of `0.001`
- Batch size of `4`
- `ReduceLROnPlateau` learning-rate scheduler

These values are defined near the beginning of `codes/retrain.py` as `number_of_epochs`, `learning_rate`, and `batch_size`. The optimizer can be changed where `optimizer` is defined.

After retraining, the script saves:

```text
weights/my_retrained_weights.pt
output/<input_name>_denoised.npy
```

The saved weight file can later be supplied to `codes/inference.py`. Running inference or retraining again on the same input name overwrites the corresponding denoised output file.

## GPU and Memory Use

The scripts automatically select CUDA when `torch.cuda.is_available()` is true. Otherwise, they run on the CPU.

Both scripts process the video as batches of overlapping three-frame windows. If a memory error occurs, lower the `batch_size` value in the corresponding script.
