
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg'
import torch
import numpy as np
import os
from Utils.DATAset import CustomDataset_remove_Bg
import torch.nn as nn
from timeit import default_timer
import Utils.np_transforms as np_transforms
from torch.utils.tensorboard import SummaryWriter
import datetime
from Utils.Defocus_PSF import *
from Utils.sfhformer_haze import sfhformer_haze
from Utils.microscPSF import *
import json
# Get the current date
current_date = datetime.datetime.now().strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD'

torch.manual_seed(0)   # Seed the random number generator so that neural network training
np.random.seed(0)      # behaves identically across runs and remains reproducible

image_dir_train = [
    r"dataset\\remove_background\rand_image_512\1/",
    r"dataset\\remove_background\rand_image_512\2/",
    r"dataset\\remove_background\rand_image_512\3/",
    r"dataset\\remove_background\rand_image_512\4/",
    r"dataset\\remove_background\rand_image_512\5/",
    r"dataset\\remove_background\rand_image_512\6/",
    r"dataset\\remove_background\rand_image_512\7/",
    r"dataset\\remove_background\rand_image_512\8/",
    r"dataset\\remove_background\rand_image_512\9/",
    r"dataset\\remove_background\rand_image_512\10/"
]
image_dir_eval = [
    r"dataset\\remove_background\rand_image_512_eval\1/",
    r"dataset\\remove_background\rand_image_512_eval\2/",
    r"dataset\\remove_background\rand_image_512_eval\3/",
    r"dataset\\remove_background\rand_image_512_eval\4/",
    r"dataset\\remove_background\rand_image_512_eval\5/",
    r"dataset\\remove_background\rand_image_512_eval\6/",
    r"dataset\\remove_background\rand_image_512_eval\7/",
    r"dataset\\remove_background\rand_image_512_eval\8/",
    r"dataset\\remove_background\rand_image_512_eval\9/",
    r"dataset\\remove_background\rand_image_512_eval\10/"
]

batch_size = 2
batch_per_ep = 500  # Number of training batches per epoch
learning_rate=2e-4
epochs = 200
S = 256
Scale = 1
NA = 0.7
n = 1  # Refractive index of the objective environment
ns = 1  # Refractive index of the sample medium
wavelength_em = 580e-9
#pixelsize = 6.5/40*1e-6
pixelsize = 6.5/60*1e-6
pixelsize = pixelsize/Scale

alpha = 1.3
beta = 0.7
# Noise parameters
a = 0.93
b = 12
photon1 = 60
photon2 = 150
RANG = 400   # Half of the sampled z range: RANG/100 * wavelength_em / (NA * NA)
edge=12*np.ceil((0.61*wavelength_em/NA)/pixelsize)
edge = int(edge)

params_PSF_em = {
        'size': (255, 255, 1200),  # nx, ny, nz; odd values keep the cropped PSF symmetric
        'NA': NA,
        'lambda': wavelength_em,  # Emission wavelength
        'ns': ns,  # Refractive index of the sample medium
        'ni0': n, 'ni': n,  # 1 for an air objective and 1.5 for an oil objective
        'ng0': 1, 'ng': 1,  # Design and actual refractive indices of the coverslip
        'tg0': 0e-6, 'tg': 0e-6,  # Design and actual coverslip thicknesses
        'ti0': 0e-3,  # Working distance; the specified range for a 40x/0.6 NA objective is 3.6--2.8 mm
        'resLateral': pixelsize,  # Lateral sampling: 100 nm
        'resAxial': wavelength_em / (NA * NA) / 100,  # Axial sampling: 250 nm
        'pZ': 0e-9,  # Axial emitter distance from the coverslip surface; designed for 0--200 nm
    }


CURRENT_DATE = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

path_model = rf"./path_models\remove_background"
def normalize_minmax_torch(x, eps=1e-12):
    # x: torch.Tensor
    return (x - x.amin()) / (x.amax() - x.amin() + eps)

def main():

    os.makedirs(path_model, exist_ok=True)
    path = 'alpha=%s_beta=%s_NA=%s_ns=%s_n=%s_Date=%s' % (str(alpha), str(beta), str(NA), str(ns), str(n), current_date)
    writer = SummaryWriter(os.path.join(path_model, path))

    config = {
        "current_date": current_date,
        "CURRENT_DATE": CURRENT_DATE,
        "seed": 0,
        "batch_size": batch_size,
        "batch_per_ep": batch_per_ep,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "S": S,
        "Scale": Scale,
        "NA": NA,
        "n": n,
        "ns": ns,
        "wavelength_em": wavelength_em,
        "pixelsize": pixelsize,
        "alpha": alpha,
        "beta": beta,
        "noise_a":a,
        "noise_b":b,
        "photon1": photon1,
        "photon2": photon2,
        "RANG": RANG,
        "edge": edge,
        "params_PSF_em": {k: (str(v) if isinstance(v, tuple) else v) for k, v in params_PSF_em.items()},
        "image_dir_train": image_dir_train,
        "image_dir_eval": image_dir_eval,
        "weight_decay": 1e-5,
        "scheduler_eta_min": 4e-8,
        "fft_loss_weight": 0.1,
        "num_workers": 1,
        "criterion": "L1Loss",
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "model": "sfhformer_haze",
        "path_model": path_model,
    }
    config_path = os.path.join(path_model, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Config saved to {config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(edge)
    # 2025-11-24: use the corrected PSF model
    psf_conven = microsc_psf(params_PSF_em)
    mid_z = psf_conven.shape[2] // 2

    H = S + 2 * edge
    W = S + 2 * edge

    PSF_FFT_conven1 = []
    PSF_confocal_FFT_conven1 = []
    for zi in range(mid_z - RANG, mid_z + RANG, 1):  # Update this for the corresponding dataset
        psf = Cutting_PSF(psf_conven[:, :, zi]).astype(np.float32, copy=False)
        # psf = psf_conven[:, :, zi].astype(np.float32, copy=False)
        #psf = torch.from_numpy(PSF_conven).float()
        if abs(zi-mid_z) > int(beta*100):
            scale = np.exp(-alpha * ((abs(zi - mid_z) / 100.0 - beta) ** 2) / (beta ** 2)).astype(psf.dtype)
        else:
            scale = 1
        psf_confocal = psf * scale
        # --- Key change: use rfft2 to generate a half-spectrum (last dimension becomes W//2+1) ---
        F_psf_bg1 = np.fft.rfft2(psf, s=(H, W))  # shape: (H, W//2+1); retain only the nonredundant half-spectrum for real-valued input
        F_psf_confocal_bg1 = np.fft.rfft2(psf_confocal, s=(H, W))  # shape: (H, W//2+1)

        PSF_FFT_conven1.append(F_psf_bg1.astype(np.complex64, copy=False))
        PSF_confocal_FFT_conven1.append(F_psf_confocal_bg1.astype(np.complex64, copy=False))

    PSF_FFT_conven = np.stack(PSF_FFT_conven1, axis=2)
    PSF_confocal_FFT_conven = np.stack(PSF_confocal_FFT_conven1,axis=2)

    Data_train=CustomDataset_remove_Bg(image_dir_train,RANG,S,edge,PSF_FFT_conven,PSF_confocal_FFT_conven,a,b,photon1,photon2,np_transforms.Compose([np_transforms.RandomCrop(S+2*edge),np_transforms.RandomHorizontalFlip(),np_transforms.RandomVerticalFlip()]))
    train_loader = torch.utils.data.DataLoader(Data_train, batch_size=batch_size, shuffle=True, num_workers=0)

    Data_eval = CustomDataset_remove_Bg(image_dir_eval, RANG,S,edge,PSF_FFT_conven,PSF_confocal_FFT_conven,a,b,photon1,photon2,np_transforms.Compose([np_transforms.RandomCrop(S+2*edge),
                                                         np_transforms.RandomHorizontalFlip(),
                                                         np_transforms.RandomVerticalFlip()]))
    eval_loader = torch.utils.data.DataLoader(Data_eval, batch_size = batch_size, shuffle=True, num_workers=0)

    model = sfhformer_haze().cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,weight_decay=1e-5)  # 2024 Optica: remove the extra loss term equal to weight_decay times the sum of squared weights

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 4e-8)  # 2024
    criterion = nn.L1Loss()
    loss_history = []

    min_valid_mse = 1    # Set the minimum validation loss
    start_ep = 0
    # Resume from a checkpoint
    if os.path.isfile(os.path.join(path_model,"ep_120checkpoint.pth")):
        checkpoint = torch.load(os.path.join(path_model,"ep_120checkpoint.pth"), map_location='cpu')
        start_ep = checkpoint['epoch']
        #min_valid_mse = checkpoint['min_valid_mse']
        print("Resuming from the checkpoint: ep and min_valid_mse", start_ep,min_valid_mse)
        np.random.set_state(checkpoint['np_rand_state'])

        torch.set_rng_state(checkpoint['torch_rand_state'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])

    for ep in range(start_ep, epochs):
        print(f"Epoch {ep}/{epochs}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        ################### Train ###################
        model.train()
        train_l2_step = 0
        LOSS_ALL=0   # Accumulated loss
        t1 = default_timer()
        for i, (noise_img, Ture_img) in enumerate(train_loader):
            # Break when reaching the batch_per_ep limit
            train_loss=0
            if i >= batch_per_ep:
                break
            # Load data onto the GPU (or CPU)
            noise_img = noise_img.to(device)
            Ture_img = Ture_img.to(device)
            #print(f"Batch {i}: lowimg dtype={lowimg.dtype}, highimg dtype={highimg.dtype}")
            # Forward pass
            output = model(noise_img)
            #print(output.dtype)
            train_loss += criterion(output,Ture_img)

            fft_diff = torch.fft.fft2(
                output - Ture_img,
                dim=(-2, -1),
            )
            # Equivalent to stacking the real and imaginary parts before applying L1Loss
            fft_loss = 0.5 * (
                fft_diff.real.abs().mean()
                + fft_diff.imag.abs().mean()
            )
            #print(pred_fft3.dtype)
            train_loss += 0.1 * fft_loss
            #print(train_loss.dtype)
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # Clip gradients to prevent exploding gradients
            optimizer.step()
            train_l2_step += 1
            LOSS_ALL +=train_loss.item()
            # Print loss information for the current batch
            if (i + 1) % 100 == 0:  # Print every 10 batches
                print(f"Train_Batch {i + 1}/{batch_per_ep}, Loss_all:{train_loss.item():.6f}")

        avg_loss = LOSS_ALL / train_l2_step
        loss_history.append(avg_loss)  # Record the average loss for each epoch

        ################### valid ###################
        # Record the training time for each epoch
        model.eval()  # Set the model to evaluation mode
        valid_mse = 0
        valid_define_loss = 0
        xx_list = []
        yy_list = []
        im_list = []
        with torch.no_grad():
            for i, (lowimg, highimg) in enumerate(eval_loader):
                if i >= 10:
                   break
                #print(f"Batch {i}: lowimg dtype={lowimg.dtype}, highimg dtype={highimg.dtype}")
                lowimg = lowimg.to(device)
                highimg = highimg.to(device)
                output = model(lowimg)
                valid_mse += criterion(output, highimg).item()     # Validation loss without the Fourier-transform term
                xx_list.append(lowimg.cpu().numpy())
                yy_list.append(highimg.cpu().numpy())
                im_list.append(output.cpu().numpy())
        avg_valid_mse = valid_mse / len(eval_loader)

        xx = np.vstack(xx_list).reshape((-1,) + lowimg.shape[1:])
        yy = np.vstack(yy_list).reshape((-1,) + highimg.shape[1:])
        im = np.vstack(im_list).reshape((-1,) + output.shape[1:])

        writer.add_images('Noise_image', np.clip(
            (xx[:, 0:1, ...] - xx[:, 0:1, ...].min()) / (xx[:, 0:1, ...].max() - xx[:, 0:1, ...].min()), 0, 1), ep,
                          dataformats='NCHW')
        writer.add_images('Real_image', np.clip(
            (yy[:, 0:1, ...] - yy[:, 0:1, ...].min()) / (yy[:, 0:1, ...].max() - yy[:, 0:1, ...].min()), 0, 1), ep,
                          dataformats='NCHW')
        writer.add_images('Guss_output_Denoise_imag', np.clip(
            (im[:, 0:1, ...] - im[:, 0:1, ...].min()) / (im[:, 0:1, ...].max() - im[:, 0:1, ...].min()+1e-9), 0, 1), ep,
                          dataformats='NCHW')
        writer.add_scalar('valid_MSEloss', avg_valid_mse, ep)
        writer.add_scalar('train_loss', avg_loss, ep)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, ep)
        t2 = default_timer()
        print(f"Epoch {ep} completed in {t2 - t1:.2f} seconds.")
        print(f"Validation MSE Loss: {avg_valid_mse:.6f}")
        print(f"Average  Loss: {avg_loss:.6f}")
        ################### Save Model ###################
        if avg_valid_mse < min_valid_mse and ep > 10:
            torch.save(model, os.path.join(path_model, "ep_" + str(ep) + ".pth"))
            min_valid_mse = avg_valid_mse
        if (ep) % 30 == 0:
            torch.save({'epoch': ep,
                        'min_valid_mse': min_valid_mse,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'scheduler': scheduler.state_dict(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"ep_" + str(ep)+"checkpoint.pth"))
        #scheduler.step(avg_valid_mse)
        scheduler.step()
    torch.save({'epoch': ep,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'np_rand_state': np.random.get_state(),
                'scheduler': scheduler.state_dict(),
                'torch_rand_state': torch.get_rng_state(),
                }, os.path.join(path_model,  "ep_" + str(epochs) + "checkpoint.pth"))
    print("Training and Valid completed!")
    writer.close()  # Close the writer after flushing the event log

if __name__ == '__main__':
    main()







