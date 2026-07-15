# single-frame-fluorescence-restoration
This repository contains the code for physics-informed end-to-end learning for single-frame fluorescence image denoising and out-of-focus background removal.

## Environment requirements
The code was developed and tested with Python 3.9.13. All required Python packages and their versions are listed in requirements.txt.

To install the dependencies, run: pip install -r requirements.txt

## Project structure

```text
single-frame-fluorescencerestoration/
├── dataset/                         # Dataset loading and preprocessing scripts
├── demo_data/                       # Example fluorescence images for demonstration
├── path_models/                     # Pretrained model checkpoints
├── Utils/                           # Utility functions and network-related modules
│
├── generate_randimage.py            # Generate simulated training images
├── noise_parameter_estimation.py    # Estimate noise parameters from noisy images
│
├── train_denoise_wide_field.py      # Train the wide-field fluorescence denoising model
├── train_denoise_only_noisy_img.py  # Train the model using only noisy experimental images
├── train_remove_Bg_and_denoise.py   # Train the joint background-removal and denoising model
│
├── Infer_denoise_wide_field.py      # Perform inference for wide-field fluorescence denoising
├── Infer_denoise_only_noisy_img.py  # Perform inference for models trained on noisy images
├── Infer_remove_Bg_and_denoise.py   # Perform joint background removal and denoising
│
├── requirements.txt                 # Python package dependencies
├── README.md                        # Project documentation
├── LICENSE                          # Software license
└── .gitignore                       # Files and directories excluded from Git



```

## Train
To train the models from scratch, follow these steps:

- Run `generate_randimage.py` to generate 10,000 random images in each subfolder of `dataset/`. The validation set contains one tenth as many images as the training set.
- Change the relative dataset and model-saving paths in the corresponding training scripts.
- Run the appropriate script:
  - `train_denoise_wide_field.py` for wide-field fluorescence denoising.
  - `train_denoise_only_noisy_img.py` for training with only noisy experimental images.
  - `train_remove_Bg_and_denoise.py` for joint background removal and denoising.

## Infer

在demo 文件夹里给了一些实力图像，在path_models 提供了一些预先训练的模型，可分别运行Infer_denoise_wide_field.py 查看运行结果。





If you use this code, please cite our paper:
[paper information]