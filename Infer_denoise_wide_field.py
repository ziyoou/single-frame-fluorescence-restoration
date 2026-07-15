
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import os
import numpy as np
import torch
from Utils.SFHformer import SFHformer_m

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from matplotlib.colors import LinearSegmentedColormap

output_folder = r"./demo_data\denoise\wide_field/"
name_tag = "microtubule"   #"corn_stem"  #"pituitary"# "oleander"
# 读取 NumPy 数据
MEAN_Img = np.load(
    os.path.join(output_folder, f"MEAN_Img_{name_tag}.npy")
).astype(np.float32)

lr_image_np = np.load(
    os.path.join(output_folder, f"Single_Noise_Img_{name_tag}.npy")
).astype(np.float32)
lr_image = torch.from_numpy(lr_image_np).unsqueeze(0).unsqueeze(0)

model = SFHformer_m()  # 参数必须与训练时一致

state_dict = torch.load(
    "path_models\denoise\wide_field\ep_385_state_dict.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model = model.cuda()
model.eval()

with torch.no_grad():
    sr_tensor = model(lr_image.to("cuda"))  # 超分辨率图像

sr_tensor = sr_tensor[:, 0, :, :].squeeze()
sr_image = (sr_tensor.cpu())
sr_image=sr_image.numpy()
sr_image = (sr_image-np.min(sr_image))/(np.max(sr_image)-np.min(sr_image)+ 1e-12)   #  归一化

lr_image = lr_image.squeeze()
lr_image = lr_image.numpy()
lr_image = (lr_image-np.min(lr_image))/(np.max(lr_image)-np.min(lr_image)+ 1e-12)   #  归一化

MEAN_Img = (MEAN_Img-np.min(MEAN_Img))/(np.max(MEAN_Img)-np.min(MEAN_Img)+ 1e-12)   #  归一化
data_range = 1.0
raw_psnr_value = compare_psnr(MEAN_Img, lr_image,data_range=data_range)
raw_ssim_value, _ = compare_ssim(MEAN_Img, lr_image, data_range=data_range,full=True)
# raw_pearsonr_value, _ = pearsonr(MEAN_Img, lr_image)
raw_rmse_value = np.sqrt(np.mean((MEAN_Img - lr_image) ** 2))

sn2n_psnr_value = compare_psnr(MEAN_Img, sr_image,data_range=data_range)
sn2n_ssim_value, _ = compare_ssim(MEAN_Img, sr_image, data_range=data_range,full=True)
# sn2n_pearsonr_value, _ = pearsonr(MEAN_Img, sr_image)
sn2n_rmse_value = np.sqrt(np.mean((MEAN_Img - sr_image) ** 2))

def add_psnr_ssim_labels(ax, psnr, ssim,mse):
    text = f"PSNR: {psnr:.2f}\nSSIM: {ssim:.4f}\nRMSE: {mse:.4f}"
    ax.text(0.95, 0.05, text, color='white', transform=ax.transAxes,
            fontsize=10, ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0.8))

c580 = (1.0, 0.9, 0.0)
cmap_580 = LinearSegmentedColormap.from_list("black_to_580nm", [(0, 0, 0), c580], N=256)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(MEAN_Img, cmap=cmap_580)
axs[0].set_title('GT')
axs[0].axis('off')
axs[1].imshow(lr_image, cmap=cmap_580)
axs[1].set_title('Raw')
axs[1].axis('off')
add_psnr_ssim_labels(axs[1], raw_psnr_value, raw_ssim_value,raw_rmse_value)
axs[2].imshow(sr_image, cmap=cmap_580)
axs[2].set_title('inference')
axs[2].axis('off')
add_psnr_ssim_labels(axs[2], sn2n_psnr_value, sn2n_ssim_value,sn2n_rmse_value)
fig.savefig(os.path.join(output_folder, f"Result_{name_tag}.tiff"), dpi=300, bbox_inches='tight')
#plt.show()


