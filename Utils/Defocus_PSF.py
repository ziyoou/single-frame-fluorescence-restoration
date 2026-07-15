import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # 第一类一阶 Bessel 函数
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'


#  离焦PSF 的函数
def defocused_psf(NA, wavelength0, n, pixel_size, z):
    """
    用角谱法计算介质折射率 n 下, 离焦距离 z (µm) 处的 PSF

    参数
    ----
    NA          : 数值孔径 (已包含 n)
    wavelength0 : 真空波长 (µm)
    n           : 环境折射率
    size        : 计算网格大小 (像素数, size x size)
    pixel_size  : 像素大小 (µm)
    z           : 离焦距离 (µm)

    返回
    ----
    psf         : 归一化 PSF (二维 numpy 数组)
    """

    # 有效波长 & 波矢
    wavelength_eff = wavelength0 / n
    k = 2 * np.pi / wavelength_eff  # [rad/µm]

    size = 256
    # 空间频率坐标（相机平面）
    fx = np.fft.fftfreq(size, d=pixel_size)  # [µm^-1]
    fy = np.fft.fftfreq(size, d=pixel_size)
    # print(fx[:10])
    # print("************************")

    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))
    # print(FX[2,0]-FX[2,1])
    # print("************************")

    # pupil 半径 (cutoff freq)
    f_cutoff = NA / wavelength0   # [1/µm]
    pupil = (FX**2 + FY**2 <= f_cutoff**2).astype(float)

    # 传播算子 H(u,v;z)
    # 注意空间频率要转换为传播方向分量
    fx_phys = FX * 2 * np.pi
    fy_phys = FY * 2 * np.pi
    inside = k**2 - (fx_phys**2 + fy_phys**2)
    inside[inside <= 0] = 0  # 过滤倏逝波
    W = np.sqrt(inside)
    H = np.exp(1j * W * z)

    # 离焦 pupil → 逆 FFT → 场分布
    pupil_prop = pupil * H
    field = np.fft.ifft2(np.fft.ifftshift(pupil_prop))
    psf = np.abs(field)**2
    #psf /= psf.max()
    psf_defocus = np.fft.fftshift(psf)


    ## 裁剪PSF
    center = np.array(psf_defocus.shape) // 2
    # 提取中心行的强度值
    center_row = psf_defocus[center[0], :]

    # 提取中心行的右半部分强度值
    right_half_row = center_row[center[1]:]

    max_value = np.max(right_half_row)
    max_index = np.argmax(right_half_row)
    # print(max_index)
    # print("****************")
    threshold = 0.1 * max_value

    for i in range(max_index, len(right_half_row)):
        if right_half_row[i] < threshold:
            threshold_index = i
            break
    else:
        threshold_index = len(right_half_row) - 1  # 如果没有找到，设置为最后一个索引

    cropped_psf = psf_defocus[center[0] - threshold_index + 1:center[0] + threshold_index,
                  center[1] - threshold_index + 1:center[1] + threshold_index]

    return cropped_psf.astype(np.float32)



"""
NA = 0.6
n = 1   # 物镜的环境折射率
Scale = 1
wavelength = 580e-9
pixelsize = 6.5/40*1e-6
pixelsize = pixelsize/Scale


z = 0.8 *wavelength/(NA*NA)
#z = 0
#示例：水中的 PSF (n=1.33)
cropped_psf = defocused_psf(NA, wavelength, n, pixelsize, z)

print(cropped_psf.shape)
plt.figure(figsize=(6, 6))
plt.imshow(cropped_psf, cmap="inferno")
plt.title(f"Cropped PSF (NA={NA}, λ0={wavelength*1e6:.1f} µm, n={n}, z={z*1e6:.1f} µm)")
plt.xlabel("x [µm]")
plt.ylabel("y [µm]")
plt.colorbar(label="Normalized Intensity")
plt.show()
"""
#
# Z0 = 2*wavelength/(NA*NA)
# PSF_ALL = []
# for i in range(20):
#     z = Z0*i/20
#     psf_z = defocused_psf(NA, wavelength, n, pixelsize, z)
#     PSF_ALL.append(psf_z)
#
# a=PSF_ALL[0]
# b=PSF_ALL[18]
# print(a.shape)
# print(b.shape)
# print(type(a))
# NA = 0.6
# wavelength0 = 590e-9
# n = 1
# pixel_size = 6.5/40*1e-6









