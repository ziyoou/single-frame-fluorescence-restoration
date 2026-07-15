# single-frame-fluorescence-restoration
This repository contains the code for physics-informed end-to-end learning for single-frame fluorescence image denoising and out-of-focus background removal.

## Requirements
- Python 3.9.13
- PyTorch 1.9.0
- NumPy
- SciPy
- matplotlib


请按一下步骤
1. 调用generate_randimage.py，生成用于训练去噪、去背景的随机图像，保存到 dataset 文件下。


pip install -r requirements.txt
python train.py
python train.py
python infer.py

Training data are generated using a physics-based forward model.
Example input data are provided in `data_example/`.

If you use this code, please cite our paper:
[paper information]
