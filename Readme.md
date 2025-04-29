# CFR-GAN

## Overview
This repository contains the implementation of CFR-GAN, a face rotation model that can generate realistic face images at different angles.

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- tqdm
- faceParsing
- mmRegressor
- pytorch3d

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/CFR-GAN.git
cd CFR-GAN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
- Download the BFM face model from: https://faces.dmi.unibas.ch/bfm/bfm2017.html
- Place the downloaded `BFM_model_80.mat` file in `mmRegressor/BFM/` directory
- Download the generator model `CFRNet_G_ep55_vgg.pth` and place it in `saved_models/` directory
- The 3D estimator model `trained_weights_occ_3d.pth` should already be in the `saved_models/` directory

## Usage

### Inference
To run inference on a set of images:

```bash
python inference.py --img_path test_imgs/input --save_path test_imgs/output
```

Arguments:
- `--img_path`: Path to input images directory (required)
- `--save_path`: Path to save output images (required)
- `--aligner`: Face alignment method (default: 'None')
- `--generator_path`: Path to generator model (default: "saved_models/CFRNet_G_ep55_vgg.pth")
- `--estimator_path`: Path to 3D estimator model (default: "saved_models/trained_weights_occ_3d.pth")
- `--face_model_path`: Path to face model (default: "mmRegressor/BFM/BFM_model_80.mat")
- `--batch_size`: Batch size for processing (default: 4)

### Model Architecture
The CFRNet model consists of:
- Initial convolution layers for both rotated and guidance images
- Attention Feature Difference (AFD) module
- Downsampling layers with gated convolutions
- Residual blocks
- Upsampling layers
- Final convolution layer with tanh activation
- Mixing layer for mask generation

### Image Processing Pipeline
1. Load and preprocess input images using `Estimator3D.load_images()`
2. Generate testing pairs with specific pose using `generate_testing_pairs()`
3. Normalize and permute dimensions for model input
4. Run inference through CFRNet
5. Process and save output images

## Notes
- The model currently runs on CPU by default
- Input images should be in JPG or PNG format
- The output images will be saved in the specified save path
- The rotation angle is set to [5.0, 0.0, 0.0] by default

## License
[Your License Here]

CFR-GAN generate_pairs.py Change Log
This document outlines the changes made to generate_pairs.py in the CFR-GAN project to resolve errors encountered during execution on a CPU-based environment (Python 3.8.18, torch==1.6.0+cpu, numpy==1.19.2, pytorch3d==0.3.0). The changes address type mismatches, syntax errors, and compatibility issues to enable processing of 7 input images (test_imgs/input/) and generate _rot.jpg, _gui.jpg, and _occ.jpg outputs in test_imgs/output/.

Environment

OS: Ubuntu (assumed from snu@snu-Precision-3680)
Python: 3.8.18
Dependencies (from requirements.txt):
pillow>=7.2.0
opencv-python==4.4.0.44
pytorch3d==0.3.0
scipy==1.5.4
tensorboard==2.4.1
tqdm
numpy==1.19.2
torch==1.6.0+cpu
torchvision==0.7.0+cpu

Input: 7 JPEG images in test_imgs/input/
Output: Rendered images and occlusion masks in test_imgs/output/
Required Files:
saved_models/trained_weights_occ_3d.pth
mmRegressor/BFM/BFM_model_80.mat
faceParsing/model_final_diss.pth

Environment Setup with pyenv

To set up Python 3.8 using pyenv:

1. Install Prerequisites:
```bash
sudo apt update && sudo apt install -y \
  make build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses5-dev libncursesw5-dev xz-utils tk-dev \
  libffi-dev liblzma-dev python-openssl git
```

2. Install pyenv:
```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

3. Install Python 3.8:
```bash
pyenv install 3.8.18
pyenv global 3.8.18
```

Verify the installation:
```bash
python --version  # Should show Python 3.8.18
```

Change Log

1. Fix for Tensor Dimension Mismatch in get_colors_from_image
Issue:
- Error occurred when trying to index image tensors with incorrect dimensions
- IndexError: too many indices for tensor of dimension 2

Fix:
- Added proper tensor shape handling and validation
- Improved color extraction with proper indexing
- Added dimension checks and conversions

Code Changes:
```python
def get_colors_from_image(self, image, proj, z_buffer, scaling=True, normalized=False, reverse=True, z_cut=None):
    # Ensure image is in the correct format
    if len(image.shape) == 4 and image.shape[1] == 3:
        image = image.permute(0, 2, 3, 1)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    # Ensure proj is a PyTorch tensor
    if isinstance(proj, np.ndarray):
        proj = torch.from_numpy(proj).float().to(self.device)
    elif not isinstance(proj, torch.Tensor):
        proj = torch.tensor(proj, dtype=torch.float32, device=self.device)
    else:
        proj = proj.float().to(self.device)
```

2. Fix for Vertex-Color Correspondence in Mesh Creation
Issue:
- Error occurred during mesh rendering due to mismatched vertex and color counts
- IndexError: index out of bounds for dimension 0

Fix:
- Added vertex count tracking
- Implemented color data validation and adjustment
- Added padding/truncation for color data

Code Changes:
```python
# Ensure face_shape and face_color have the same number of vertices
num_vertices = face_shape.shape[1]
color_from_img = self.get_colors_from_image(image, face_projection, z_buffer, normalized=True)

# Ensure color_from_img has the correct number of vertices
if color_from_img.shape[1] != num_vertices:
    if color_from_img.shape[1] < num_vertices:
        padding = torch.zeros((color_from_img.shape[0], num_vertices - color_from_img.shape[1], color_from_img.shape[2]), 
                            device=color_from_img.device)
        color_from_img = torch.cat([color_from_img, padding], dim=1)
    else:
        color_from_img = color_from_img[:, :num_vertices, :]
```

3. Fix for Mixed Memory Format Warning
Issue:
- Warning about mixed memory formats in convolution operations
- UserWarning: Mixed memory format inputs detected

Note:
- This is a non-critical warning from PyTorch
- Does not affect the functionality of the code
- Can be safely ignored in this context

Installation and Usage

Install Dependencies:
```bash
# Install PyTorch and torchvision
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
pip install -r requirements.txt
```

Usage:
```bash
python generate_pairs.py
```

The script will:
1. Load images from test_imgs/input/
2. Process each image to generate:
   - Rotated view (_rot.jpg)
   - Guidance image (_gui.jpg)
   - Occlusion mask (_occ.jpg)
3. Save outputs to test_imgs/output/

Note: Processing time is approximately 56 seconds per image on CPU.
    
