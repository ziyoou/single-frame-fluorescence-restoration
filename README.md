# single-frame-fluorescence-restoration
This repository contains the code for physics-informed end-to-end learning for single-frame fluorescence image denoising and out-of-focus background removal.

## Environment requirements
The code was developed and tested with Python 3.9.13. All required Python packages and their versions are listed in requirements.txt.

To install the dependencies, run: pip install -r requirements.txt

## Project structure

```text
single-frame-fluorescencerestoration/
├── dataset/                         # Simulated datasets
├── demo_data/                       # Example fluorescence images for demonstration
├── path_models/                     # Directory for model checkpoints, including pretrained model weights provided by the authors
├── Utils/                           # Utility functions and network-related modules
│
├── generate_randimage.py            # Generate artificial random images
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

The provided training and inference scripts use the parameter configurations adopted in this study. The default settings are provided only as examples and may not be directly applicable to arbitrary fluorescence datasets. Several pretrained model weights and representative inference results are also included for demonstration and evaluation.

## Train

Before initiating model training, specify the key imaging and noise parameters, including the numerical aperture (NA), emission wavelength, pixel size, noise parameters,  peak photon count and other task-specific parameters. Once these parameters have been determined, follow the steps below to train the models from scratch:

- Run `generate_randimage.py` to generate 10,000 random images (512 × 512 pixels) in each subfolder of `dataset/`. The validation set contains one tenth as many images as the training set.
- Change the relative dataset and model-saving paths in the corresponding training scripts.
- Run the appropriate script:
  - `train_denoise_wide_field.py` for wide-field fluorescence denoising.
  - Before running `train_denoise_only_noisy_img.py`, run `noise_parameter_estimation.py` to estimate the noise parameters from the noisy images. In addition, estimate the point-spread function (PSF); here, the PSF is approximated by a Gaussian function, whose width must be determined. Then use the estimated parameters to train the model with `train_denoise_only_noisy_img.py`.
  - `train_remove_Bg_and_denoise.py` for joint background removal and denoising.


## Inference

To evaluate the models, use the example images in `demo_data/` together with the model checkpoints generated during training and saved in `path_models/`. After specifying the paths to the input images, model checkpoint, and output directory, run one of the following scripts:

- `Infer_denoise_wide_field.py`: wide-field fluorescence denoising. The inference results are saved in `demo_data\denoise\experimental_wide_field_img`. Representative restored images are shown below:

<img src=".\demo_data\denoise\experimental_wide_field_img\Result_microtubule.png" alt="Result_corn_stem" style="zoom:80%;" /> 

<img src=".\demo_data\denoise\experimental_wide_field_img\Result_oleander.png" alt="Result_corn_stem" style="zoom:80%;" />

- `Infer_denoise_only_noisy_img.py`: denoising using a model trained exclusively on noisy images. The inference result is saved in `demo_data\denoise\only_noisy_img_Myosin_IIA`. A representative restored image is shown below:

<img src=".\demo_data\denoise\only_noisy_img_Myosin_IIA\Result_Myosin_IIA.png" alt="Result_corn_stem" style="zoom:80%;" />
- `Infer_remove_Bg_and_denoise.py`: simultaneous out-of-focus background removal and denoising. The inference results are saved in `demo_data\remove_background`. A representative comparison is shown below:

<img src=".\demo_data\remove_background\Result_color\Compare_z0000.png" alt="Result_corn_stem" style="zoom:30%;" />

If you use this code, please cite our paper:
[paper information]


