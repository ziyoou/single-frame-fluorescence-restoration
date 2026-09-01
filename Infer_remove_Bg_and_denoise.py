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
# Input: flat-field-corrected 3D TIFF stack
input_stack_path = r".\demo_data\remove_background\raw_noise_stack.npy"

# Output: network-inferred 3D TIFF stack
out_stack_path = r".\demo_data\remove_background\infer_denoise_stack.tif"
compare_folder = r".\demo_data\remove_background\Result_color"
os.makedirs(compare_folder, exist_ok=True)
# ===================== Parameters =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = sfhformer_haze()  # Parameters must match those used during training

state_dict = torch.load(
    r"path_models\remove_background\ep_135_Only_noisy_img_state_dict.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model = model.cuda()
model.eval()

# Preserve the original cropping logic.
# Set to None to disable cropping.

# "per_slice": use the same normalization as the original per-image .npy inference
# "global": use one maximum for all Z slices to better preserve relative intensity across slices
normalization_mode = "per_slice"

def pad_to_multiple(x, multiple=8):
    """Pad H and W to multiples of ``multiple`` to satisfy the network size requirement."""
    _, _, h, w = x.shape

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, pad=(0, pad_w, 0, pad_h), mode="reflect")

    return x, h, w


def load_tiff_stack(path):
    """
    Load a TIFF stack and return it with shape (Z, Y, X).
    """
    stack = tifffile.imread(path)

    # Accept a single 2D image by treating it as Z=1
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

    

    os.makedirs(os.path.dirname(os.path.abspath(out_stack_path)), exist_ok=True)

    # ---------- Load the 3D input ----------
    stack = np.load(input_stack_path)
    print(f"Input stack shape (Z, Y, X): {stack.shape}")
    print(f"Input dtype: {stack.dtype}")

    nz, height, width = stack.shape

    # Compute the maximum over the full stack once in global mode
    if normalization_mode == "global":
        global_max_gray = max(float(np.max(stack)), 1e-6)
    elif normalization_mode != "per_slice":
        raise ValueError(
            "normalization_mode 只能是 'per_slice' 或 'global'。"
        )

    # Estimate whether BigTIFF is required for the uint16 inference output
    estimated_output_bytes = nz * height * width * np.dtype(np.uint16).itemsize
    use_bigtiff = estimated_output_bytes > 3.8 * 1024**3

    print(f"Writing output stack: {out_stack_path}")
    print(f"BigTIFF: {use_bigtiff}")

    start_time = time.perf_counter()

    # Write one slice at a time to avoid holding the full prediction in memory
    with tifffile.TiffWriter(out_stack_path, bigtiff=use_bigtiff) as tif:
        with torch.inference_mode():
            for z in range(nz):
                # Convert the current Z slice to float32
                arr = stack[z].astype(np.float32)

                # Select the normalization method
                if normalization_mode == "per_slice":
                    max_gray = max(float(arr.max()), 1e-6)
                else:
                    max_gray = global_max_gray

                x_np = np.ascontiguousarray(arr / max_gray, dtype=np.float32)
                x = torch.from_numpy(x_np)[None, None, :, :].to(device)

                # Pad to the size required by the network
                x_pad, original_h, original_w = pad_to_multiple(x, multiple=8)

                # Run network inference
                pred = model(x_pad)

                # Some models may return a tuple or list
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]

                # Remove padding and restore the original size
                pred = pred[:, :, :original_h, :original_w]

                # Restore the original intensity scale
                pred_raw = pred[0, 0].float().cpu().numpy() * max_gray

                # Guard against NaN/Inf and write as 16-bit
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
                # Each page corresponds to one Z slice
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
                axs[1].set_title("Denoised and remove Bg")
                axs[1].axis("off")

                fig.savefig(
                    os.path.join(compare_folder, f"Compare_z{z:04d}.png"),
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
