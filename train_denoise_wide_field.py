
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import torch
import numpy as np
from Utils.DATAset import *
from timeit import default_timer
import Utils.np_transforms as np_transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import datetime
from Utils.SFHformer import SFHformer_m
from Utils.microscPSF import *
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
torch.manual_seed(0)
np.random.seed(0)
image_dir_train = r"./dataset/denoise/rand_image_512/"
image_dir_eval  = r"./dataset/denoise/rand_image_512_eval/"

path_model = r"./path_models/denoise/wide_field"

batch_size = 2
batch_per_ep = 600
learning_rate=2e-4
epochs = 400
S = 256
NA = 0.6
wavelength = 580e-9
pixelsize = 6.5/40*1e-6
edge=np.ceil((0.61*wavelength/NA)/pixelsize)
edge = int(edge)
#####  wide-field imaging;      noise paramates
##### Prime BSI Express sCMOS camera (Teledyne Photometrics) operated in 11-bit sensitivity mode
a = 0.93
b1 = 4
b2 = 8

######  The maximum number of photons S
photon1 = 20
photon2 = 100

n = 1   # Environmental refractive index of the objective lens
ns = 1  # Refractive index of the medium in which the sample is located

params_PSF_low = {
        # 'size': (255, 255, 64),
        'NA': NA,
        'lambda': wavelength,
        'ns': ns,
        'ni0': n, 'ni': n,
        'ng0': 1.5, 'ng': 1.5,
        'tg0': 170e-6, 'tg': 170e-6,
        'ti0': 2.8e-3,
        'resLateral': pixelsize,
        # 'resAxial': 20e-9,
        'pZ': 100e-9,
    }

def main():
    path = 'Epochs=%d_a=%s_b1=%s_b2=%s_Date=%s' % (epochs,str(a),str(b1),str(b2),current_date)
    log_dir = os.path.join(path_model, path)
    os.makedirs(log_dir, exist_ok=True)  
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    params_PSF_low['resAxial'] = wavelength / (NA * NA) / 100
    params_PSF_low['size'] = (255, 255, 500)
    psf_low = microsc_psf(params_PSF_low)
    mid_z = psf_low.shape[2] // 2
    PSF_ALL = []
    for zi in range(mid_z - 100, mid_z + 100, 1):  
        xy_slice = psf_low[:, :, zi]
        xy_slice = Cutting_PSF(xy_slice)
        PSF_ALL.append(xy_slice)
    Data_train=CustomDataset_denoise(image_dir_train,a,b1,b2,photon1,photon2,S,edge,PSF_ALL,
                                np_transforms.Compose([np_transforms.RandomCrop(S+2*edge),
                                                                                np_transforms.RandomHorizontalFlip(),
                                                                                np_transforms.RandomVerticalFlip(),
                                                                                np_transforms.ToTensor()
                                                                        ]))
    train_loader = torch.utils.data.DataLoader(Data_train, batch_size=batch_size, shuffle=True, num_workers=1)
    Data_eval = CustomDataset_denoise( image_dir_eval,a,b1,b2,photon1,photon2,S,edge,PSF_ALL,np_transforms.Compose([np_transforms.RandomCrop(S+2*edge),np_transforms.RandomHorizontalFlip(),
                                                         np_transforms.RandomVerticalFlip(),
                                                         np_transforms.ToTensor()
                                                         ]))
    eval_loader = torch.utils.data.DataLoader(Data_eval, batch_size = batch_size, shuffle=True, num_workers=1)
    model = SFHformer_m().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 4e-8)
    criterion = nn.L1Loss()
    loss_history = []
    min_valid_mse = 1  
    start_ep = 0
    if os.path.isfile(os.path.join(path_model,"ep_160checkpoint.pth")):
        checkpoint = torch.load(os.path.join(path_model,"ep_160checkpoint.pth"), map_location='cpu')
        start_ep = checkpoint['epoch']
        #min_valid_mse = checkpoint['min_valid_mse']
        print("Resuming from the checkpoint: ep and min_valid_mse", start_ep,min_valid_mse)
        np.random.set_state(checkpoint['np_rand_state'])

        torch.set_rng_state(checkpoint['torch_rand_state'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!warning!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("!!!!!!!!!!!!!!!!Temporarily change step size!!!!!!!!!!!!!!!!!")
        # scheduler.step_size = 50
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])

    for ep in range(start_ep, epochs):
        print(f"Epoch {ep}/{epochs}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        ################### Train ###################
        model.train()
        train_l2_step = 0
        LOSS_ALL=0   
        t1 = default_timer()
        for i, (noise_img, Ture_img) in enumerate(train_loader):
            # Break when reaching the batch_per_ep limit
            train_loss=0
            if i >= batch_per_ep:
                break
            noise_img = noise_img.to(device)
            Ture_img = Ture_img.to(device)
            output = model(noise_img)
            #print(output.dtype)
            train_loss += criterion(output,Ture_img)

            label_fft3 = torch.fft.fft2(output, dim=(-2, -1))
            label_fft = torch.stack((label_fft3.real, label_fft3.imag), -1)
            pred_fft3 = torch.fft.fft2(Ture_img, dim=(-2, -1))
            pred_fft = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            #print(pred_fft3.dtype)
            train_loss += 0.1*criterion(pred_fft,label_fft)
            #print(train_loss.dtype)
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
            optimizer.step()
            train_l2_step += 1
            LOSS_ALL +=train_loss.item()
            if (i + 1) % 100 == 0:  
                print(f"Train_Batch {i + 1}/{batch_per_ep}, Loss_all:{train_loss.item():.6f}")

        avg_loss = LOSS_ALL / train_l2_step
        loss_history.append(avg_loss)  

        ################### valid ###################
        model.eval()  
        valid_mse = 0
        xx_list = []
        yy_list = []
        im_list = []
        with torch.no_grad():
            for i, (lowimg, highimg) in enumerate(eval_loader):
                lowimg = lowimg.to(device)
                highimg = highimg.to(device)
                output = model(lowimg)
                label_fft3 = torch.fft.fft2(output, dim=(-2, -1))
                label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)
                pred_fft3 = torch.fft.fft2(highimg, dim=(-2, -1))
                pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)
                valid_mse +=criterion(output, highimg).item()+0.1 * criterion(pred_fft3, label_fft3).item()
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
        if (ep) % 50 == 0:
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
    writer.close() 

if __name__ == '__main__':
    main()


