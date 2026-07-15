# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from Utils.sfhformer_haze import sfhformer_haze
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
# 输入：平场校正后的三维 TIFF stack
input_stack_path = r".\demo_data\remove_background\raw_noise_stack.tif"

# 输出：网络推理后的三维 TIFF stack
out_stack_path = r".\demo_data\remove_background\infer_denoise_stack.tif"
compare_folder = r".\demo_data\remove_background\Result_color"
os.makedirs(compare_folder, exist_ok=True)
# ===================== Parameters =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = sfhformer_haze()  # 参数必须与训练时一致

state_dict = torch.load(
    r"path_models\remove_background\ep_135_Only_noisy_img_state_dict.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model = model.cuda()
model.eval()

# 保持你原来的裁剪逻辑。
# 若不希望裁剪，改为 None。

# "per_slice"：与原先逐张 .npy 推理时的归一化方式一致
# "global"：所有 Z 层使用同一个最大值，更严格保留层间相对亮度
normalization_mode = "per_slice"

def pad_to_multiple(x, multiple=8):
    """将 H、W 补齐到 multiple 的整数倍，避免网络尺寸不匹配。"""
    _, _, h, w = x.shape

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, pad=(0, pad_w, 0, pad_h), mode="reflect")

    return x, h, w


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




def main():
    print(f"Using device: {device}")

    if not os.path.isfile(input_stack_path):
        raise FileNotFoundError(f"找不到输入 stack：\n{input_stack_path}")

    os.makedirs(os.path.dirname(os.path.abspath(out_stack_path)), exist_ok=True)

    # ---------- 读取三维输入 ----------
    stack = load_tiff_stack(input_stack_path)
    print(f"Input stack shape (Z, Y, X): {stack.shape}")
    print(f"Input dtype: {stack.dtype}")

    nz, height, width = stack.shape

    # 仅在 global 模式下计算一次全 stack 最大值
    if normalization_mode == "global":
        global_max_gray = max(float(np.max(stack)), 1e-6)
    elif normalization_mode != "per_slice":
        raise ValueError(
            "normalization_mode 只能是 'per_slice' 或 'global'。"
        )

    # 推理输出为 uint16，因此估计是否需要 BigTIFF
    estimated_output_bytes = nz * height * width * np.dtype(np.uint16).itemsize
    use_bigtiff = estimated_output_bytes > 3.8 * 1024**3

    print(f"Writing output stack: {out_stack_path}")
    print(f"BigTIFF: {use_bigtiff}")

    start_time = time.perf_counter()

    # 逐层写出，避免把整个预测结果全部堆在内存中
    with tifffile.TiffWriter(out_stack_path, bigtiff=use_bigtiff) as tif:
        with torch.inference_mode():
            for z in range(nz):
                # 当前 Z 层，转 float32
                arr = stack[z].astype(np.float32)

                # 归一化方式
                if normalization_mode == "per_slice":
                    max_gray = max(float(arr.max()), 1e-6)
                else:
                    max_gray = global_max_gray

                x_np = np.ascontiguousarray(arr / max_gray, dtype=np.float32)
                x = torch.from_numpy(x_np)[None, None, :, :].to(device)

                # 补齐到网络要求的尺寸
                x_pad, original_h, original_w = pad_to_multiple(x, multiple=8)

                # 网络推理
                pred = model(x_pad)

                # 某些模型可能返回 tuple/list
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]

                # 裁掉 padding，恢复到原尺寸
                pred = pred[:, :, :original_h, :original_w]

                # 恢复到原始强度尺度
                pred_raw = pred[0, 0].float().cpu().numpy() * max_gray

                # # 防止 NaN/Inf，并写为 16-bit
                pred_raw = np.nan_to_num(
                    pred_raw,
                    nan=0.0,
                    posinf=65535.0,
                    neginf=0.0
                )

                pred_u16 = np.clip(
                    np.rint(pred_raw),
                    0,
                    65535
                ).astype(np.uint16)
                # 每一页对应一个 Z 层
                tif.write(
                    pred_u16,
                    photometric="minisblack",
                    compression=None
                )
                c580 = (1.0, 0.9, 0.0)
                cmap_580 = LinearSegmentedColormap.from_list("black_to_580nm", [(0, 0, 0), c580], N=256)

                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(arr, cmap=cmap_580, vmin=arr.min(), vmax=arr.max())
                axs[0].set_title("Raw")
                axs[0].axis("off")

                axs[1].imshow(pred_u16, cmap=cmap_580, vmin=pred_u16.min(), vmax=pred_u16.max())
                axs[1].set_title("Denoised")
                axs[1].axis("off")

                fig.savefig(
                    os.path.join(compare_folder, f"Compare_z{z:04d}.tiff"),
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close(fig)

                print(f"[{z + 1:>4d}/{nz}] Processed Z = {z}")

    elapsed = time.perf_counter() - start_time

    print("\nDone.")
    print(f"Output stack saved to:\n{out_stack_path}")
    print(f"Output shape (Z, Y, X): ({nz}, {height}, {width})")
    print(f"Total inference + writing time: {elapsed:.2f} s")
    print(f"Average time per frame: {elapsed / nz:.4f} s/frame")


if __name__ == "__main__":
    main()