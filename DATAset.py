
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
    def __init__(self, hr_dir, a,b1,b2,photon1,photon2,Scale,M,S,edge,PSF_ALL,trans=None):
        self.a = a
        self.b1 = b1
        self.b2 = b2
        self.photon1 = photon1
        self.photon2 = photon2
        self.Scale = Scale
        self.M = M
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
        for i in range(self.M):
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

class CustomDataset_remove_Bg(Dataset):
    def __init__(self, hr_dir, RANG, S, edge,
                 PSF_FFT_conven, PSF_confocal_FFT_conven,
                 trans=None):
        self.hr_dir = hr_dir
        self.RANG = int(RANG)
        self.S = int(S)
        self.edge = int(edge)
        self.trans = trans

        # 1) 预先把每个文件夹的 png 列表缓存好（避免 __len__ 和 __getitem__ 里反复 glob）
        self.files = []
        for d in self.hr_dir:
            fs = sorted(glob.glob(os.path.join(d, "*.png")))
            if len(fs) == 0:
                raise RuntimeError(f"Empty folder: {d}")
            self.files.append(fs)

        self.length = min(len(fs) for fs in self.files)  # 防止某个文件夹张数更少

        # 2) 用 rfft2：只保留 fft2 的前半谱（沿最后一维）
        #    PSF_FFT_* 形状应为 (H, W, Z)，我们取 (H, W//2+1, Z)
        H, W2, Z = PSF_FFT_conven.shape
        W = 2* (W2 -1)
        self.psf_conv_r = np.ascontiguousarray(PSF_FFT_conven[:, :W2, :].astype(np.complex64))
        self.psf_design_r = np.ascontiguousarray(PSF_confocal_FFT_conven[:, :W2, :].astype(np.complex64))

        self.H = H
        self.W = W
        self.Z = Z
        self.W2 = W2

    def __len__(self):
        return self.length

    def _read_one(self, path):
        # 读灰度（避免 hr_img[:,:,0]）
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W)

        # 如果 trans 是 torch 版 transforms，会返回 tensor，这里兼容一下
        if self.trans is not None:
            out = self.trans(arr)  # 你原来是 self.trans(hr_img[:,:,0])
            if isinstance(out, torch.Tensor):
                arr = out.numpy().squeeze().astype(np.float32)
            else:
                arr = np.asarray(out, dtype=np.float32)

        return normalize_minmax_numpy(arr)

    def __getitem__(self, index):
        N = len(self.hr_dir)
        # 1) 读 N 张图并堆叠成 (N,H,W)
        imgs = [self._read_one(self.files[i][index]) for i in range(N)]
        imgs = np.stack(imgs, axis=0)  # (N,H,W)
        # 2) 批量 rfft2（一次算完 N 张）
        F_imgs = np.fft.rfft2(imgs, axes=(-2, -1))  # (N,H,W2)
        # 3) 为每张图选一个 Z_psf（等价你原来的 random.randint 分段）
        #    你原式：randint(i*2*RANG/N, (i+1)*2*RANG/N - 1)（两端包含）
        #    用 numpy：randint(low, high_exclusive)
        lows = (np.arange(N) * (2 * self.RANG) / N).astype(int)
        highs = ((np.arange(N) + 1) * (2 * self.RANG) / N).astype(int)
        highs = np.maximum(highs, lows + 1)  # 防止空区间
        z_idx = np.random.randint(lows, highs)  # (N,)

        # 4) 取对应 psf（得到 (N,H,W2)）
        psf_conv = np.take(self.psf_conv_r, z_idx, axis=2)      # (H,W2,N)
        psf_conv = np.moveaxis(psf_conv, 2, 0)                  # (N,H,W2)
        psf_design = np.take(self.psf_design_r, z_idx, axis=2)  # (H,W2,N)
        psf_design = np.moveaxis(psf_design, 2, 0)              # (N,H,W2)

        # 5) 批量 irfft2 得到卷积结果 (N,H,W)
        conv_imgs = np.fft.irfft2(F_imgs * psf_conv, s=(self.H, self.W), axes=(-2, -1)).astype(np.float32)
        design_imgs = np.fft.irfft2(F_imgs * psf_design, s=(self.H, self.W), axes=(-2, -1)).astype(np.float32)

        # 6) 裁剪 + 求和
        e = self.edge
        S = self.S
        conv_imgs = conv_imgs[:, e:e+S, e:e+S]
        design_imgs = design_imgs[:, e:e+S, e:e+S]

        sum_img_conven = conv_imgs.sum(axis=0)
        sum_img_design = design_imgs.sum(axis=0)

        # 7) 噪声 + 归一化
        sum_img_conven = normalize_minmax_numpy(sum_img_conven)

        RAND = random.randint(200, 300)  #100  到两百个  光子数
        sum_img_conven = 0.93 * sum_img_conven * RAND
        std_dev = np.sqrt(0.93 * sum_img_conven + 5)
        sum_img_conven += np.random.normal(loc=0.0, scale=std_dev)
        sum_img_conven += np.random.normal(0.0, 0.001, size=sum_img_conven.shape).astype(np.float32)
        sum_img_conven = normalize_minmax_numpy(sum_img_conven)

        sum_img_design = normalize_minmax_numpy(sum_img_design)

        # 8) [C,H,W] 并用 from_numpy（比 torch.tensor 快）
        Conven_Img = sum_img_conven[None, ...].astype(np.float32)
        Design_Img = sum_img_design[None, ...].astype(np.float32)

        return torch.from_numpy(Conven_Img), torch.from_numpy(Design_Img)


