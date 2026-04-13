

import numpy as np
from math import pi
from scipy.special import j0, j1  # 0,1 阶 Bessel 函数
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
from scipy.signal import convolve2d
def microsc_psf(params):
    """
    Compute 3D PSF for fluorescence microscopy.

    This implementation is adapted with reference to:
    J. Li, F. Xue, T. Blu,
    Fast and accurate three-dimensional point spread function computation
    for fluorescence microscopy,
    J. Opt. Soc. Am. A 34, 1029-1034 (2017).
    """

    # ---- 复制参数并填默认值 ----
    p = dict(params)
    if 'size' not in p:
        raise ValueError("Please set params['size'] = (nx, ny, nz)")
    nx, ny, nz = p['size']

    p.setdefault('numBasis', 200)       # Bessel 基函数个数
    p.setdefault('numSamp', 2000)       # pupil 半径方向采样点数
    p.setdefault('overSampling', 2)     # 径向过采样倍率

    p.setdefault('NA', 1.4)
    p.setdefault('lambda', 610e-9)
    p.setdefault('M', 100)

    p.setdefault('ns', 1.33)            # specimen RI
    p.setdefault('ng0', 1.5)            # coverslip RI (design)
    p.setdefault('ng', 1.5)             # coverslip RI (exp)
    p.setdefault('ni0', 1.5)            # immersion RI (design)
    p.setdefault('ni', 1.5)             # immersion RI (exp)

    p.setdefault('ti0', 150e-6)         # working distance (design)
    p.setdefault('tg0', 170e-6)         # coverslip thickness (design)
    p.setdefault('tg', 170e-6)          # coverslip thickness (exp)

    p.setdefault('resLateral', 100e-9)  # lateral pixel size
    p.setdefault('resAxial', 250e-9)    # axial sampling
    p.setdefault('pZ', 2000e-9)         # emitter depth in specimen

    # ---- 像平面 & 轴向坐标 ----
    x0 = (nx - 1) / 2.0
    y0 = (ny - 1) / 2.0
    xp, yp = x0, y0

    maxRadius = int(round(((nx - x0) ** 2 + (ny - y0) ** 2) ** 0.5)) + 1
    R = np.arange(0, p['overSampling'] * maxRadius) / p['overSampling']  # 像素半径
    r = R * p['resLateral']  # 物理半径 (m)

    Ti = p['ti0'] + p['resAxial'] * (np.arange(nz) - (nz - 1.0) / 2.0)

    # ---- pupil 半径 ρ 采样 ----
    a = 0.0
    b = min(
        1.0,
        p['ns'] / p['NA'],
        p['ni'] / p['NA'],
        p['ni0'] / p['NA'],
        p['ng0'] / p['NA'],
        p['ng'] / p['NA'],
    )
    L = p['numSamp']
    Rho = np.linspace(a, b, L)[:, None]  # (L,1)

    # ---- 1. 用 Bessel 系列逼近 exp(iW) ----
    NN = p['numBasis']
    k0 = 2 * pi / p['lambda']

    A = k0 * p['NA'] * r          # (Nr,)
    A2 = A ** 2
    Ab = A * b                    # A*b

    # 缩放基频 an（Li & Blu 的经验做法）
    k00 = 2 * pi / (545e-9)       # min wavelength
    factor1 = k0 / k00
    NA0 = 1.4                     # max NA
    factor2 = p['NA'] / NA0

    an = (3 * np.arange(1, NN + 1) - 2).astype(float)  # (NN,)
    an = an * factor1 * factor2

    # 构造拟合矩阵 J (L×NN)
    anRho = Rho @ an[None, :]     # (L,NN)
    J = j0(anRho)

    # 解析核 Ele(r, an)
    J0A = j0(Ab)                  # (Nr,)
    J1A = A * j1(Ab)              # (Nr,)

    anJ0A = J0A[:, None] * an[None, :]  # (Nr,NN)
    anb = an * b
    an2 = an ** 2
    B1 = j1(anb)                  # (NN,)
    B0 = j0(anb)                  # (NN,)

    Ele = anJ0A * B1[None, :] - J1A[:, None] * B0[None, :]  # (Nr,NN)
    domin = an2[None, :] - A2[:, None]                      # (Nr,NN)
    Ele = Ele * b / domin                                   # (Nr,NN)

    # ---- Gibson–Lanni OPD: specimen + immersion + coverslip ----
    C1 = p['ns'] * p['pZ']
    C2 = p['ni'] * (Ti - p['ti0'])          # (nz,)
    C3 = p['ng'] * (p['tg'] - p['tg0'])

    OPDs = C1 * np.sqrt(1 - (p['NA'] * Rho / p['ns']) ** 2)             # (L,1)
    OPDi = np.sqrt(1 - (p['NA'] * Rho / p['ni']) ** 2) * C2[None, :]   # (L,nz)
    OPDg = C3 * np.sqrt(1 - (p['NA'] * Rho / p['ng']) ** 2)             # (L,1)

    OPD = OPDi + (OPDs + OPDg)             # (L,nz)
    W = k0 * OPD
    Ffun = np.cos(W) + 1j * np.sin(W)      # = exp(iW)

    # 求 Bessel 展开系数 Ci： J (L×NN) * Ci (NN×nz) ≈ Ffun (L×nz)
    Ci, *_ = np.linalg.lstsq(J, Ffun, rcond=None)  # (NN,nz)

    # ---- 2. 得到每个 z 的径向强度 PSF0(r,z) ----
    ciEle = Ele @ Ci               # (Nr,nz)
    PSF0 = np.abs(ciEle) ** 2      # (Nr,nz)

    # ---- 3. 径向 → 笛卡尔网格插值 ----
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    rPixel = np.sqrt((X - xp) ** 2 + (Y - yp) ** 2)

    over = p['overSampling']
    index = np.floor(rPixel * over).astype(int)
    max_index = len(R) - 2
    index = np.clip(index, 0, max_index)
    index1 = index
    index2 = index + 1

    R_index = R[index1]
    disR = (rPixel - R_index) * over
    disR1 = 1.0 - disR

    PSF = np.zeros((ny, nx, nz), dtype=float)
    for zi in range(nz):
        h = PSF0[:, zi]     # (Nr,)
        slice_ = h[index2] * disR + h[index1] * disR1
        PSF[:, :, zi] = slice_

    # 归一化
    PSF /= PSF.max()
    return PSF

def Cutting_PSF(psf_defocus):
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


def PSF_confocal(params_PSF_exc,params_PSF_em,pinhole_AU):
    PSF_exc = microsc_psf(params_PSF_exc)
    airy_diameter = 1.22 * params_PSF_em['lambda'] / params_PSF_em['NA']    # [m] 探测面上的1 AU直径
    pinhole_radius = (pinhole_AU * airy_diameter)/2
    radius_samples = max(1, int(np.ceil(pinhole_radius / params_PSF_em['resLateral'])))

    R = radius_samples
    size = 2 * R + 1
    y, x = np.ogrid[-R:R + 1, -R:R + 1]
    mask = x ** 2 + y ** 2 <= R ** 2  # 圆形区域
    kernel = np.zeros((size, size), dtype=float)
    kernel[mask] = 1.0


    PSF_em = microsc_psf(params_PSF_em)
    PSF_em_eff = np.empty_like(PSF_em)
    for zi in range(PSF_em.shape[2]):  # 记得修改  对应的数据集
        #PSF_em_eff[:, :, zi] = np.convolve(PSF_em[:, :, zi], kernel, mode='same')
        PSF_em_eff[:, :, zi] = convolve2d(PSF_em[:, :, zi],kernel,mode='same',boundary='fill')
        #xy_slice = Cutting_PSF(xy_slice)
        #PSF_ALL.append(xy_slice)
    PSF_conf = PSF_exc * PSF_em_eff
    PSF_conf /= PSF_conf.max()
    return PSF_conf
