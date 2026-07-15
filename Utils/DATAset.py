
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.signal import convolve2d
import glob
import random

def normalize_minmax_numpy(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + eps)

class CustomDataset_denoise(Dataset):   #  [N C  H   W]   C  the channels
    def __init__(self, hr_dir, a,b1,b2,photon1,photon2,S,edge,PSF_ALL,trans=None):
        self.a = a
        self.b1 = b1
        self.b2 = b2
        self.photon1 = photon1
        self.photon2 = photon2
        self.S = S
        self.edge = edge
        self.trans = trans
        self.hr_file_paths = hr_dir
        self.PSF_ALL = PSF_ALL
    def __len__(self):
        return len(glob.glob(os.path.join(self.hr_file_paths, '*.png')))
    def __getitem__(self, index):
        data_path1 = os.path.join(self.hr_file_paths,f"{index:04d}.png")
        hr_img = np.array(Image.open(data_path1)).astype('float32') / 255
        hr_img = self.trans(hr_img[:, :, 0]).numpy().squeeze()
        hr_img = normalize_minmax_numpy(hr_img)
        Noise_lr_Img = []
        Denoise_lr_Img = []
        RAND = random.randint(self.photon1, self.photon2)
        z_psf = random.randint(80, 120)
        psf = self.PSF_ALL[z_psf]
        
        lr_img = convolve2d(hr_img, psf, mode='same')
        lr_img = lr_img[self.edge:self.S + self.edge, self.edge:self.S + self.edge]
        lr_img = normalize_minmax_numpy(lr_img)
        Denoise_lr_Img.append(lr_img)
        noise_lr_img = self.a*lr_img*RAND
        b = random.uniform(self.b1,self.b2)
        std_dev = np.sqrt(self.a* noise_lr_img + b)
        noise_lr_img = noise_lr_img + np.random.normal(0.0, std_dev)
        noise_lr_img = normalize_minmax_numpy(noise_lr_img)
        noise_lr_img = noise_lr_img.astype('float32')
        Noise_lr_Img.append(noise_lr_img)

        Noise_lr_Img = np.stack(Noise_lr_Img, axis=0)
        Denoise_lr_Img = np.stack(Denoise_lr_Img, axis=0)
        return torch.tensor(Noise_lr_Img), torch.tensor(Denoise_lr_Img)

class CustomDataset_denoise_new(Dataset):   #  [N C  H   W]   C  the channels
    def __init__(self, hr_dir, a, b, photon1, photon2, S, edge, PSF, trans=None):
        self.a = a
        self.b = b
        self.photon1 = photon1
        self.photon2 = photon2
        self.S = S
        self.edge = edge
        self.trans = trans
        self.hr_file_paths = hr_dir
        self.PSF = [np.asarray(psf, dtype=np.float32) for psf in PSF]

    def __len__(self):
        return len(glob.glob(os.path.join(self.hr_file_paths, '*.png')))

    def __getitem__(self, index):
        data_path1 = os.path.join(self.hr_file_paths, f"{index:04d}.png")
        hr_img = np.array(Image.open(data_path1)).astype('float32') / 255
        hr_img = self.trans(hr_img[:, :, 0]).numpy().squeeze()
        hr_img = normalize_minmax_numpy(hr_img)

        psf = random.choice(self.PSF)
        lr_img = convolve2d(hr_img, psf, mode='same')
        lr_img = lr_img[self.edge:self.S + self.edge, self.edge:self.S + self.edge]
        lr_img = normalize_minmax_numpy(lr_img)

        # RAND 表示最大期望光子数
        PHOTON = lr_img * float(random.randint(int(self.photon1), int(self.photon2)))
        clean_img = self.a * PHOTON
        noise_var = self.a * clean_img + self.b
        gaussian_noise = np.random.normal(
            loc=0.0,
            scale=np.sqrt(np.maximum(noise_var, 0.0)),
        ).astype('float32')

        # Poisson + Gaussian 噪声合成
        noise_lr_img = clean_img + gaussian_noise
        scale = float(np.max(noise_lr_img))

        clean_img = clean_img / scale
        noise_lr_img = noise_lr_img / scale

        # [1, H, W] 单通道输出
        Noise_lr_Img = noise_lr_img[np.newaxis, ...].astype(np.float32)
        Denoise_lr_Img = clean_img[np.newaxis, ...].astype(np.float32)
        return torch.from_numpy(Noise_lr_Img), torch.from_numpy(Denoise_lr_Img)




class CustomDataset_remove_Bg(Dataset):
    def __init__(
        self,
        hr_dir,
        RANG,
        S,
        edge,
        PSF_FFT_conven,
        PSF_confocal_FFT_conven,
        a,
        b,
        photon1,
        photon2,
        trans=None,
    ):
        self.hr_dir = hr_dir
        self.RANG = int(RANG)
        self.S = int(S)
        self.edge = int(edge)

        self.a = a
        self.b = b
        self.photon1 = photon1
        self.photon2 = photon2
        self.trans = trans

        # 预先读取每个文件夹中的 png 路径
        self.files = []
        for d in self.hr_dir:
            fs = sorted(glob.glob(os.path.join(d, "*.png")))

            if len(fs) == 0:
                raise RuntimeError(f"Empty folder: {d}")

            self.files.append(fs)

        # 防止不同文件夹图片数不一致
        self.length = min(len(fs) for fs in self.files)

        # PSF FFT shape: [H, W//2+1, Z]
        H, W2, Z = PSF_FFT_conven.shape
        W = 2 * (W2 - 1)

        self.psf_conv_r = np.ascontiguousarray(
            PSF_FFT_conven[:, :W2, :]
        )

        self.psf_design_r = np.ascontiguousarray(
            PSF_confocal_FFT_conven[:, :W2, :]
        )

        self.H = H
        self.W = W
        self.Z = Z
        self.W2 = W2

    def __len__(self):
        return self.length

    def _read_one(self, path):
        # 读取单通道灰度图
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # 保持你原来的 transform 调用方式
        if self.trans is not None:
            out = self.trans(arr)

            if isinstance(out, torch.Tensor):
                arr = out.numpy().squeeze().astype(np.float32)
            else:
                arr = np.asarray(out, dtype=np.float32)

        return normalize_minmax_numpy(arr)

    def __getitem__(self, index):
        N = len(self.hr_dir)

        # -------------------------------------------------
        # 1. 读取 N 张图像，shape: [N, H, W]
        # -------------------------------------------------
        imgs = [
            self._read_one(self.files[i][index])
            for i in range(N)
        ]

        imgs = np.stack(imgs, axis=0)

        # 某些 np_transforms 会输出 [H, W, 1]
        # 堆叠后变成 [N, H, W, 1]，这里恢复为 [N, H, W]
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs[..., 0]

        if imgs.ndim != 3:
            raise ValueError(
                f"imgs 应为 [N, H, W]，但当前 shape 为 {imgs.shape}"
            )

        F_imgs = np.fft.rfft2(
            imgs,
            axes=(-2, -1),
        ).astype(np.complex64, copy=False)

        # -------------------------------------------------
        # 3. 每张图选择一个对应 z 的 PSF
        # -------------------------------------------------
        lows = (
            np.arange(N) * (2 * self.RANG) / N
        ).astype(int)

        highs = (
            (np.arange(N) + 1) * (2 * self.RANG) / N
        ).astype(int)

        highs = np.maximum(highs, lows + 1)

        z_idx = np.random.randint(lows, highs)

        # PSF: [H, W2, N] -> [N, H, W2]
        psf_conv = np.take(
            self.psf_conv_r,
            z_idx,
            axis=2,
        )
        psf_conv = np.moveaxis(psf_conv, 2, 0)

        psf_design = np.take(
            self.psf_design_r,
            z_idx,
            axis=2,
        )
        psf_design = np.moveaxis(psf_design, 2, 0)

        # -------------------------------------------------
        # 4. 核心加速：
        # 原来是 N 次 conventional irfft2 + N 次 confocal irfft2。
        # 现在先在频域求和，各只做 1 次 irfft2。
        # -------------------------------------------------
        sum_fft_conven = np.sum(
            F_imgs * psf_conv,
            axis=0,
        )

        sum_fft_design = np.sum(
            F_imgs * psf_design,
            axis=0,
        )

        sum_img_conven_full = np.fft.irfft2(
            sum_fft_conven,
            s=(self.H, self.W),
        ).astype(np.float32)

        sum_img_design_full = np.fft.irfft2(
            sum_fft_design,
            s=(self.H, self.W),
        ).astype(np.float32)

        # -------------------------------------------------
        # 5. 裁剪边缘
        # -------------------------------------------------
        e = self.edge
        S = self.S

        sum_img_conven = sum_img_conven_full[
            e:e + S,
            e:e + S,
        ]

        sum_img_design = sum_img_design_full[
            e:e + S,
            e:e + S,
        ]

        # 防止少量 FFT 数值误差产生负数，导致 Poisson 报错
        sum_img_conven = np.maximum(sum_img_conven, 0.0)
        sum_img_design = np.maximum(sum_img_design, 0.0)

        # -------------------------------------------------
        # 6. 原有归一化与噪声模型，保持不变
        # -------------------------------------------------
        sum_img_design = sum_img_design / (
            sum_img_design.max() + 1e-8
        )

        sum_img_conven = sum_img_conven / (
            sum_img_conven.max() + 1e-8
        )

        RAND_photon = random.randint(
            self.photon1,
            self.photon2,
        )

        sum_img_conven = sum_img_conven * RAND_photon

        sum_img_conven = np.random.poisson(
            sum_img_conven
        ).astype(np.float32)

        read_noise_var = random.uniform(
            0.8 * self.b,
            1.2 * self.b,
        )

        gaussian_noise = np.random.normal(
            loc=0.0,
            scale=np.sqrt(read_noise_var),
            size=sum_img_conven.shape,
        ).astype(np.float32)

        sum_img_conven = (
            self.a * sum_img_conven
            + gaussian_noise
        )

        sum_img_conven = sum_img_conven / (
            sum_img_conven.max() + 1e-8
        )

        # -------------------------------------------------
        # 7. 输出 Tensor，shape: [1, S, S]
        # -------------------------------------------------
        Conven_Img = np.ascontiguousarray(
            sum_img_conven[None, ...],
            dtype=np.float32,
        )

        Design_Img = np.ascontiguousarray(
            sum_img_design[None, ...],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(Conven_Img),
            torch.from_numpy(Design_Img),
        )