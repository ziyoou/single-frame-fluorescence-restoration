# -*- coding: utf-8 -*-
import datetime
import json
import os
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import Utils.np_transforms as np_transforms
from Utils.DATAset import *
from Utils.np_transforms import *
from Utils.SFHformer import SFHformer_m

# ===================== 全局配置 =====================
# 固定随机种子
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

# 路径配置
CURRENT_DATE = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
IMAGE_DIR_TRAIN = r"./dataset/denoise/rand_image_512/"
IMAGE_DIR_EVAL = r"./dataset/denoise/rand_image_512_eval/"

PATH_MODEL = rf"./path_models\denoise\only_noisy_img_Myosin_IIA"

PSF_FWHM_MIN = 2
PSF_FWHM_MAX = 5
PSF_NUM = 200
PSF_CROP_REL_THRESHOLD = 0.05

BATCH_SIZE = 2
BATCH_PER_EP = 600
LEARNING_RATE = 2e-4
EPOCHS = 200
S = 256

# 可视化参数
VIS_EVERY = 10
VIS_NUM = 30

# 噪声参数
#BIAS = 102.0
#MAX_GRAY = 400.0
A = 2.241111
B = -5.234278   #  b的平方
PHOTON1 = 20
PHOTON2 = 80


def norm01_tensor(x, eps=1e-9):
    """
    x: [N, C, H, W]
    对每张图单独归一化到 0-1，用于 TensorBoard 可视化。
    """
    x = x[:, 0:1, ...]
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return torch.clamp((x - x_min) / (x_max - x_min + eps), 0, 1)


def make_gaussian_psf_from_fwhm(fwhm, rel_threshold=0.05):
    sigma = float(fwhm) / 2.354820045
    radius = max(3, int(np.ceil(5.0 * sigma)))

    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))

    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
    threshold = psf[cy, cx] * float(rel_threshold)
    yy_idx, xx_idx = np.where(psf >= threshold)
    crop_radius = int(max(
        np.max(np.abs(yy_idx - cy)),
        np.max(np.abs(xx_idx - cx)),
    ))

    psf = psf[
        cy - crop_radius:cy + crop_radius + 1,
        cx - crop_radius:cx + crop_radius + 1,
    ]
    psf = psf / (np.sum(psf) + 1e-12)
    return psf.astype(np.float32)


def build_gaussian_psf_list():
    fwhm_grid = np.linspace(PSF_FWHM_MIN, PSF_FWHM_MAX, PSF_NUM)
    psf_list = [
        make_gaussian_psf_from_fwhm(fwhm, PSF_CROP_REL_THRESHOLD)
        for fwhm in fwhm_grid
    ]
    return psf_list, fwhm_grid


def main():
    # ===================== 日志与设备 =====================
    path = "Photo1_%sP2_%s_a=%s_b=%s" % (
        str(PHOTON1),
        str(PHOTON2),
        str(A),
        str(B),
    )
    log_dir = os.path.join(PATH_MODEL, path)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # ===================== PSF list =====================
    psf_list, fwhm_grid = build_gaussian_psf_list()
    EDGE = max(psf.shape[0] // 2 for psf in psf_list)

    config_path = os.path.join(PATH_MODEL, "config.json")
    config = {
        "CURRENT_DATE": CURRENT_DATE,
        "IMAGE_DIR_TRAIN": IMAGE_DIR_TRAIN,
        "IMAGE_DIR_EVAL": IMAGE_DIR_EVAL,
        "PATH_MODEL": PATH_MODEL,
        "BATCH_SIZE": BATCH_SIZE,
        "BATCH_PER_EP": BATCH_PER_EP,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": EPOCHS,
        "S": S,
        "VIS_EVERY": VIS_EVERY,
        "VIS_NUM": VIS_NUM,
        "EDGE": EDGE,
        "A": A,
        "B": B,
        "PHOTON1": PHOTON1,
        "PHOTON2": PHOTON2,
        "PSF_FWHM_MIN": PSF_FWHM_MIN,
        "PSF_FWHM_MAX": PSF_FWHM_MAX,
        "PSF_NUM": PSF_NUM,
        "PSF_CROP_REL_THRESHOLD": PSF_CROP_REL_THRESHOLD,
        "PSF_FWHM_LIST": fwhm_grid.tolist(),
        "PSF_SHAPES": [list(psf.shape) for psf in psf_list],
        "PSF_SUMS": [float(psf.sum()) for psf in psf_list],
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===================== 数据集与 DataLoader =====================
    train_transforms = np_transforms.Compose([
        np_transforms.RandomCrop(S + 2 * EDGE),
        np_transforms.RandomHorizontalFlip(),
        np_transforms.RandomVerticalFlip(),
        np_transforms.ToTensor(),
    ])

    eval_transforms = np_transforms.Compose([
        np_transforms.RandomCrop(S + 2 * EDGE),
        np_transforms.RandomHorizontalFlip(),
        np_transforms.RandomVerticalFlip(),
        np_transforms.ToTensor(),
    ])

    data_train = CustomDataset_denoise_new(  
        IMAGE_DIR_TRAIN,
        A,
        B,
        PHOTON1,
        PHOTON2,
        S,
        EDGE,
        psf_list,
        train_transforms,
    )
    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    data_eval = CustomDataset_denoise_new(
        IMAGE_DIR_EVAL,
        A,
        B,
        PHOTON1,
        PHOTON2,
        S,
        EDGE,
        psf_list,
        eval_transforms,
    )
    eval_loader = torch.utils.data.DataLoader(
        data_eval,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # ===================== 模型与优化器 =====================
    model = SFHformer_m().cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        EPOCHS,
        4e-8,
    )
    criterion = nn.L1Loss()

    loss_history = []
    min_valid_mse = float("inf")
    early_stop_counter = 0
    start_ep = 0

    # ===================== 断点恢复 =====================
    checkpoint_path = os.path.join(PATH_MODEL, "ep_60checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start_ep = checkpoint["epoch"]
        min_valid_mse = checkpoint.get("min_valid_mse", float("inf"))
        early_stop_counter = checkpoint.get("early_stop_counter", 0)

        print(
            "Resuming from the checkpoint: ep and min_valid_mse",
            start_ep,
            min_valid_mse,
        )

        np.random.set_state(checkpoint["np_rand_state"])
        torch.set_rng_state(checkpoint["torch_rand_state"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.load_state_dict(checkpoint["model"])

    for ep in range(start_ep, EPOCHS):
        print(f"Epoch {ep}/{EPOCHS}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        # ===================== 训练 =====================
        model.train()
        train_l2_step = 0
        loss_all = 0
        t1 = default_timer()

        for i, (noise_img, true_img) in enumerate(train_loader):
            train_loss = 0
            if i >= BATCH_PER_EP:
                break

            noise_img = noise_img.to(device)
            true_img = true_img.to(device)

            optimizer.zero_grad(set_to_none=True)

            output = model(noise_img)
            train_loss = criterion(output, true_img)
            train_loss += 0.1 * fft_l1_loss(output, true_img, criterion)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            train_l2_step += 1
            loss_all += train_loss.item()

            if (i + 1) % 100 == 0:
                print(
                    f"Train_Batch {i + 1}/{BATCH_PER_EP}, "
                    f"Loss_all:{train_loss.item():.6f}"
                )

        avg_loss = loss_all / train_l2_step
        loss_history.append(avg_loss)

        # ===================== 验证 =====================
        model.eval()
        valid_mse = 0

        vis_low_list = []
        vis_high_list = []
        vis_output_list = []
        vis_count = 0

        with torch.no_grad():
            for i, (lowimg, highimg) in enumerate(eval_loader):
                lowimg = lowimg.to(device)
                highimg = highimg.to(device)
                output = model(lowimg)

                valid_mse += criterion(output, highimg).item()
                valid_mse += 0.1 * fft_l1_loss(output, highimg, criterion).item()

                if ep % VIS_EVERY == 0 and vis_count < VIS_NUM:
                    remain_num = VIS_NUM - vis_count
                    take_num = min(remain_num, lowimg.shape[0])

                    vis_low_list.append(lowimg[:take_num].detach().cpu())
                    vis_high_list.append(highimg[:take_num].detach().cpu())
                    vis_output_list.append(output[:take_num].detach().cpu())
                    vis_count += take_num

        avg_valid_mse = valid_mse / len(eval_loader)

        # ===================== TensorBoard 记录 =====================
        if ep % VIS_EVERY == 0 and vis_count > 0:
            xx = torch.cat(vis_low_list, dim=0)
            yy = torch.cat(vis_high_list, dim=0)
            im = torch.cat(vis_output_list, dim=0)

            writer.add_images(
                "Noise_image",
                norm01_tensor(xx),
                ep,
                dataformats="NCHW",
            )
            writer.add_images(
                "Real_image",
                norm01_tensor(yy),
                ep,
                dataformats="NCHW",
            )
            writer.add_images(
                "Guss_output_Denoise_imag",
                norm01_tensor(im),
                ep,
                dataformats="NCHW",
            )

        writer.add_scalar("valid_MSEloss", avg_valid_mse, ep)
        writer.add_scalar("train_loss", avg_loss, ep)

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning Rate", current_lr, ep)

        t2 = default_timer()
        print(f"Epoch {ep} completed in {t2 - t1:.2f} seconds.")
        print(f"Validation MSE Loss: {avg_valid_mse:.6f}")
        print(f"Average  Loss: {avg_loss:.6f}")

        # ===================== 保存与早停 =====================
        if min_valid_mse == float("inf"):
            improve_ratio = float("inf")
        else:
            improve_ratio = (min_valid_mse - avg_valid_mse) / (min_valid_mse + 1e-12)

        is_best = avg_valid_mse < min_valid_mse
        # is_significant_improved = is_best and improve_ratio > EARLY_STOP_MIN_RATIO

        if is_best:
            min_valid_mse = avg_valid_mse

            torch.save(model, os.path.join(PATH_MODEL, "best_valid_model.pth"))
            torch.save(
                {
                    "epoch": ep,
                    "min_valid_mse": min_valid_mse,
                    "early_stop_counter": early_stop_counter,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "np_rand_state": np.random.get_state(),
                    "scheduler": scheduler.state_dict(),
                    "torch_rand_state": torch.get_rng_state(),
                },
                os.path.join(PATH_MODEL, "best_valid_checkpoint.pth"),
            )

            print(
                f"Best model saved at epoch {ep}. "
                f"Best validation loss: {min_valid_mse:.6f}, "
                f"Improve ratio: {improve_ratio * 100:.4f}%"
            )

        # if is_significant_improved:
        #     early_stop_counter = 0
        #     print("Validation loss significantly improved. Early stop counter reset to 0.")
        # else:
        #     if ep > EARLY_STOP_START_EPOCH:
        #         early_stop_counter += 1
        #
        #     print(
        #         f"No significant improvement. "
        #         f"Improve ratio: {improve_ratio * 100:.4f}%, "
        #         f"Early stop counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}"
        #     )

        if ep % 20 == 0:
            torch.save(
                {
                    "epoch": ep,
                    "min_valid_mse": min_valid_mse,
                    "early_stop_counter": early_stop_counter,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "np_rand_state": np.random.get_state(),
                    "scheduler": scheduler.state_dict(),
                    "torch_rand_state": torch.get_rng_state(),
                },
                os.path.join(PATH_MODEL, "ep_" + str(ep) + "checkpoint.pth"),
            )

        scheduler.step()

        # if early_stop_counter >= EARLY_STOP_PATIENCE:
        #     print(f"Early stopping triggered at epoch {ep}.")
        #     print(f"Best validation loss: {min_valid_mse:.6f}")
        #
        #     torch.save(
        #         {
        #             "epoch": ep,
        #             "min_valid_mse": min_valid_mse,
        #             "early_stop_counter": early_stop_counter,
        #             "model": model.state_dict(),
        #             "optimizer": optimizer.state_dict(),
        #             "np_rand_state": np.random.get_state(),
        #             "scheduler": scheduler.state_dict(),
        #             "torch_rand_state": torch.get_rng_state(),
        #         },
        #         os.path.join(PATH_MODEL, "early_stop_checkpoint.pth"),
        #     )
        #     break


if __name__ == "__main__":
    main()
