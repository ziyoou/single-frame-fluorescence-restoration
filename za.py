

import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from Utils.SFHformer import SFHformer_m
import Utils.SFHformer as SFHformer
import tifffile


def load_tiff_stack(path):
    """
    读取 TIFF stack。
    输出 shape 统一为 (Z, Y, X)。
    """
    stack = tifffile.imread(path)

    # 单张二维图也允许输入，自动视为 Z=1
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]

    if stack.ndim != 3:
        raise ValueError(
            f"输入 TIFF 应为二维图像或三维 stack，"
            f"当前 shape 为 {stack.shape}。"
        )

    return stack


input_stack_path = r".\demo_data\remove_background\raw_noise_stack.tif"
stack = load_tiff_stack(input_stack_path)
stack = stack[:10]
# 保存为 npy
output_npy_path = r".\demo_data\remove_background\raw_noise_stack.npy"
np.save(output_npy_path, stack)
print(f"npy 文件已保存至: {output_npy_path}")
print(f"数组形状: {stack.shape}")



# model_path = (r"path_models\remove_background\ep_135.pth")

# try:
#     model = torch.load(
#         model_path,
#         map_location="cpu",
#         weights_only=False
#     )
# except TypeError:
#     # 兼容不支持 weights_only 参数的旧版 PyTorch
#     model = torch.load(
#         model_path,
#         map_location="cpu"
#     )

# # 修复旧版 PyTorch 保存的 GELU 对象
# for module in model.modules():
#     if isinstance(module, torch.nn.GELU):
#         if not hasattr(module, "approximate"):
#             module.approximate = "none"

# model = model.to("cuda")
# model.eval()

# torch.save(
#     model.state_dict(),
#     r"path_models\remove_background\ep_135_Only_noisy_img_state_dict.pth"
# )
















# model_path = (r"C:\202605_YXX_code\20260701-修改文章-回意见\code_0701\7_only_noisy_image_denoise_BioSR\Models_pretrain\2026-06-19-21-42_sfh_m_RAND_IMG/best_valid_model.pth")

# try:
#     model = torch.load(
#         model_path,
#         map_location="cpu",
#         weights_only=False
#     )
# except TypeError:
#     # 兼容不支持 weights_only 参数的旧版 PyTorch
#     model = torch.load(
#         model_path,
#         map_location="cpu"
#     )

# # 修复旧版 PyTorch 保存的 GELU 对象
# for module in model.modules():
#     if isinstance(module, torch.nn.GELU):
#         if not hasattr(module, "approximate"):
#             module.approximate = "none"

# model = model.to("cuda")
# model.eval()

# torch.save(
#     model.state_dict(),
#     "path_models\denoise\only_noisy_img_Myosin_IIA\Only_noisy_img_state_dict.pth"
# )





# output_folder = r"demo_data\denoise\only_noisy_img_Myosin_IIA"
# name_tag = "Myosin_IIA"   #"corn_stem"      #"pituitary"# "oleander"

# raw_datafile = r"C:\202605_YXX_code\20260701-修改文章-回意见\code_0701\7_only_noisy_image_denoise_BioSR\Datasets_raw\Myosin-IIA\Cell_032.tif"
# Img = np.array(Image.open(raw_datafile)).astype('float32')

# GT_datafile = r"C:\202605_YXX_code\20260701-修改文章-回意见\code_0701\7_only_noisy_image_denoise_BioSR\Datasets_GT\Myosin-IIA\Cell_032.tif"

# MEAN_Img = np.array(Image.open(GT_datafile)).astype('float32')

# # MEAN_Img 本身已经是 NumPy 数组
# np.save(
#     os.path.join(output_folder, f"MEAN_Img_{name_tag}.npy"),
#     MEAN_Img.astype(np.float32)
# )
# # lr_image 是 Torch Tensor，需要先转换成 NumPy 数组

# np.save(
#     os.path.join(output_folder, f"Single_Noise_Img_{name_tag}.npy"),
#     Img
# )
# print(MEAN_Img.shape)
# print(Img.shape)

















#################   下面四个是  实验 宽场  细胞样品的去噪   和  训练好的模型


############ 模型 旧版本 不兼容

# model_path = (
#     r"E:\2025-lab\20251110_第二个工作_神经网络去噪"
#     r"\mode_save_SFHformer\1125\ep_385.pth"
# )

# try:
#     model = torch.load(
#         model_path,
#         map_location="cpu",
#         weights_only=False
#     )
# except TypeError:
#     # 兼容不支持 weights_only 参数的旧版 PyTorch
#     model = torch.load(
#         model_path,
#         map_location="cpu"
#     )

# # 修复旧版 PyTorch 保存的 GELU 对象
# for module in model.modules():
#     if isinstance(module, torch.nn.GELU):
#         if not hasattr(module, "approximate"):
#             module.approximate = "none"

# model = model.to("cuda")
# model.eval()

# torch.save(
#     model.state_dict(),
#     "ep_385_state_dict.pth"
# )









# dataFile = r"E:\2025-lab\229电脑\二楼共聚焦电脑\2025\Fig20251126-1-上午-中区细胞骨架-532-550滤波-bsi-11bit-3sensitivity\下午\060na40x60ms-a\Default/"
# background = r"E:\2025-lab\229电脑\二楼共聚焦电脑\2025\Fig20251126-1-上午-中区细胞骨架-532-550滤波-bsi-11bit-3sensitivity\下午\无光背景-40ms\AVG_Default.tif"
# output_folder = r"demo_data\denoise"
# name_tag = "microtubule"   #"corn_stem"      #"pituitary"# "oleander"


# Img_bg = np.array(Image.open(background)).astype('float32')
# lr_images = []

# M = 400
# for i in range(M):
#     if i > 99:
#         datafile = os.path.join(dataFile + "img_channel000_position000_time000000" + f"{i:02d}" + "_z000.tif")
#     else:
#         datafile = os.path.join(dataFile +"img_channel000_position000_time0000000"+ f"{i:02d}"+"_z000.tif")
#     Img = np.array(Image.open(datafile)).astype('float32')
#     Img = Img - Img_bg
#     Img = (Img - np.min(Img)) / (np.max(Img) - np.min(Img)+ 1e-12)
#     lr_images.append(Img)
# lr_images = np.stack(lr_images, axis=0)
# lr_images = np.expand_dims(lr_images, axis=0)
# MEAN_Img=np.mean(lr_images,axis=1,keepdims=True).squeeze()
# MEAN_Img = (MEAN_Img-np.min(MEAN_Img))/(np.max(MEAN_Img)-np.min(MEAN_Img)+ 1e-12)
# lr_image = lr_images[:,0:1,:,:]


# # MEAN_Img 本身已经是 NumPy 数组
# np.save(
#     os.path.join(output_folder, f"MEAN_Img_{name_tag}.npy"),
#     MEAN_Img.astype(np.float32)
# )
# # lr_image 是 Torch Tensor，需要先转换成 NumPy 数组
# lr_image_np = lr_image.squeeze().astype(np.float32)
# np.save(
#     os.path.join(output_folder, f"Single_Noise_Img_{name_tag}.npy"),
#     lr_image_np
# )
# print(MEAN_Img.shape)
# print(lr_image_np.shape)










# dataFile = r"E:\2025-lab\229电脑\二楼共聚焦电脑\2025\\FigS20251126-2-下午-060na40x-淘宝-大科教学-532-550滤波-bsi-11bit-3sensitivity\玉米经横切-荧光红染色-2ms-b\Default/"
# #output_folder = r"E:\2025-lab\229电脑\二楼共聚焦电脑\2025\Fig20251126-1-上午-中区细胞骨架-532-550滤波-bsi-11bit-3sensitivity\下午\060na40x40ms-b/"

# background = r"E:\2025-lab\229电脑\二楼共聚焦电脑\2025\FigS20251126-2-下午-060na40x-淘宝-大科教学-532-550滤波-bsi-11bit-3sensitivity\无光背景-2ms\AVG_Default.tif"
# output_folder = r"demo_data\denoise"

# Img_bg = np.array(Image.open(background)).astype('float32')
# lr_images = []

# M = 100
# for i in range(M):
#     if i > 99:
#         datafile = os.path.join(dataFile + "img_channel000_position000_time000000" + f"{i:02d}" + "_z000.tif")
#     else:
#         datafile = os.path.join(dataFile +"img_channel000_position000_time0000000"+ f"{i:02d}"+"_z000.tif")
#     Img = np.array(Image.open(datafile)).astype('float32')
#     Img = Img - Img_bg
#     Img = (Img - np.min(Img)) / (np.max(Img) - np.min(Img)+ 1e-12)
#     lr_images.append(Img)
# lr_images = np.stack(lr_images, axis=0)
# lr_images = np.expand_dims(lr_images, axis=0)
# MEAN_Img=np.mean(lr_images,axis=1,keepdims=True).squeeze()
# MEAN_Img = (MEAN_Img-np.min(MEAN_Img))/(np.max(MEAN_Img)-np.min(MEAN_Img)+ 1e-12)
# lr_image = lr_images[:,0:1,:,:]

# name_tag = "corn_stem"      #"pituitary"# "oleander"
# # MEAN_Img 本身已经是 NumPy 数组
# np.save(
#     os.path.join(output_folder, f"MEAN_Img_{name_tag}.npy"),
#     MEAN_Img.astype(np.float32)
# )
# # lr_image 是 Torch Tensor，需要先转换成 NumPy 数组
# lr_image_np = lr_image.squeeze().astype(np.float32)
# np.save(
#     os.path.join(output_folder, f"Single_Noise_Img_{name_tag}.npy"),
#     lr_image_np
# )
# print(MEAN_Img.shape)
# print(lr_image_np.shape)










# dataFile = r"E:\2025-lab\229电脑\二楼共聚焦电脑\2025\FigS20251126-2-下午-060na40x-淘宝-大科教学-532-550滤波-bsi-11bit-3sensitivity\夹竹桃叶横切荧光黄染色-4ms-a\Default/"

# output_folder = r"demo_data\denoise"
# background = r"E:\2025-lab\229电脑\二楼共聚焦电脑\2025\FigS20251126-2-下午-060na40x-淘宝-大科教学-532-550滤波-bsi-11bit-3sensitivity\无光背景-4ms\AVG_Default.tif"

# Img_bg = np.array(Image.open(background)).astype('float32')
# lr_images = []

# M = 100
# for i in range(M):
#     if i > 99:
#         datafile = os.path.join(dataFile + "img_channel000_position000_time000000" + f"{i:02d}" + "_z000.tif")
#     else:
#         datafile = os.path.join(dataFile +"img_channel000_position000_time0000000"+ f"{i:02d}"+"_z000.tif")
#     Img = np.array(Image.open(datafile)).astype('float32')
#     Img = Img - Img_bg
#     Img = (Img - np.min(Img)) / (np.max(Img) - np.min(Img)+ 1e-12)
#     lr_images.append(Img)
# lr_images = np.stack(lr_images, axis=0)
# lr_images = np.expand_dims(lr_images, axis=0)
# MEAN_Img=np.mean(lr_images,axis=1,keepdims=True).squeeze()
# MEAN_Img = (MEAN_Img-np.min(MEAN_Img))/(np.max(MEAN_Img)-np.min(MEAN_Img)+ 1e-12)
# lr_image = lr_images[:,0:1,:,:]

# name_tag = "oleander"      #"pituitary"# "oleander"
# # MEAN_Img 本身已经是 NumPy 数组
# np.save(
#     os.path.join(output_folder, f"MEAN_Img_{name_tag}.npy"),
#     MEAN_Img.astype(np.float32)
# )
# # lr_image 是 Torch Tensor，需要先转换成 NumPy 数组
# lr_image_np = lr_image.squeeze().astype(np.float32)
# np.save(
#     os.path.join(output_folder, f"Single_Noise_Img_{name_tag}.npy"),
#     lr_image_np
# )
# print(MEAN_Img.shape)
# print(lr_image_np.shape)









# dataFile = r"E:\2025-lab\229电脑\二楼共聚焦电脑\2025\FigS20251126-2-下午-060na40x-淘宝-大科教学-532-550滤波-bsi-11bit-3sensitivity\脑垂体切片-荧光黄染色-4ms-a\Default/"
# output_folder = r"demo_data\denoise"
# background = r"E:\2025-lab\229电脑\二楼共聚焦电脑\2025\FigS20251126-2-下午-060na40x-淘宝-大科教学-532-550滤波-bsi-11bit-3sensitivity\无光背景-4ms\AVG_Default.tif"

# Img_bg = np.array(Image.open(background)).astype('float32')
# lr_images = []

# M = 100
# for i in range(M):
#     if i > 99:
#         datafile = os.path.join(dataFile + "img_channel000_position000_time000000" + f"{i:02d}" + "_z000.tif")
#     else:
#         datafile = os.path.join(dataFile +"img_channel000_position000_time0000000"+ f"{i:02d}"+"_z000.tif")
#     Img = np.array(Image.open(datafile)).astype('float32')
#     Img = Img - Img_bg
#     Img = (Img - np.min(Img)) / (np.max(Img) - np.min(Img)+ 1e-12)
#     lr_images.append(Img)
# lr_images = np.stack(lr_images, axis=0)
# lr_images = np.expand_dims(lr_images, axis=0)
# MEAN_Img=np.mean(lr_images,axis=1,keepdims=True).squeeze()
# MEAN_Img = (MEAN_Img-np.min(MEAN_Img))/(np.max(MEAN_Img)-np.min(MEAN_Img)+ 1e-12)
# lr_image = lr_images[:,4:5,:,:]

# name_tag = "pituitary"# "oleander"
# # MEAN_Img 本身已经是 NumPy 数组
# np.save(
#     os.path.join(output_folder, f"MEAN_Img_{name_tag}.npy"),
#     MEAN_Img.astype(np.float32)
# )
# # lr_image 是 Torch Tensor，需要先转换成 NumPy 数组
# lr_image_np = lr_image.squeeze().astype(np.float32)
# np.save(
#     os.path.join(output_folder, f"Single_Noise_Img_{name_tag}.npy"),
#     lr_image_np
# )
# print(MEAN_Img.shape)
# print(lr_image_np.shape)