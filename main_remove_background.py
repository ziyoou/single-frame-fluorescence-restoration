
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
from DATAset import *
import torch.nn as nn
from timeit import default_timer
import np_transforms
from torch.utils.tensorboard import SummaryWriter
import datetime
from sfhformer_haze import sfhformer_haze
from microscPSF import *

current_date = datetime.datetime.now().strftime('%Y-%m-%d')
torch.manual_seed(0)
np.random.seed(0)

image_dir_train = [
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2\1/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2\2/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2\3/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2\4/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2\5/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2\6/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2\7/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2\8/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2\9/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2\10/"
]
image_dir_eval = [
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2_eval\1/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2_eval\2/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2_eval\3/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2_eval\4/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2_eval\5/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2_eval\6/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2_eval\7/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2_eval\8/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2_eval\9/",
    r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_dataset_20260118_gamma\gamma_2_eval\10/"
]

M = 1
batch_size = 2
batch_per_ep = 70

learning_rate=2e-4
epochs = 20
S = 256
Scale = 1
NA = 0.6
n = 1
ns = 1
wavelength_em = 580e-9
pixelsize = 6.5/40*1e-6
pixelsize = pixelsize/Scale

alpha = 1.5
beta = 0.7
RANG = 400   #  z方向取得  范围的一半   RANG/100 *  wavelength_em / (NA * NA)
edge=12*np.ceil((0.61*wavelength_em/NA)/pixelsize)
edge = int(edge)

params_PSF_em = {
        'size': (255, 255, 1200),  # nx, ny, nz   奇数 最后裁剪出来的psf  对称
        'NA': NA,
        'lambda': wavelength_em,  # 发射波长
        'ns': ns,  # 样品所处介质的折射率
        'ni0': n, 'ni': n,  # 空气镜为1 油镜为1.5
        'ng0': 1, 'ng': 1,  # 设计载玻片的折射率  实际载玻片的折射率
        'tg0': 0e-6, 'tg': 0e-6,  # 设计载玻片的厚度  实际载玻片的厚度
        'ti0': 0e-3,  # 工作距离   对于0.6na  40x 的物镜官网给的是 3.6--2.8mm
        'resLateral': pixelsize,  # 横向采样 100 nm
        'resAxial': wavelength_em / (NA * NA) / 100,  # 轴向采样 250 nm
        'pZ': 0e-9,  # 轴向发射点距离盖玻片表面的距离  紧贴表面  设计  0-200nm
    }

path_model = r"./path_models/remove_background"
def normalize_minmax_torch(x, eps=1e-12):
    return (x - x.amin()) / (x.amax() - x.amin() + eps)

def main():
    os.makedirs(path_model, exist_ok=True)
    path = 'Epochs=%d_alpha=%s_beta=%s_NA=%s__Date=%s' % (epochs, str(alpha), str(beta), str(NA), current_date)
    writer = SummaryWriter(os.path.join(path_model, path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    psf_conven = microsc_psf(params_PSF_em)
    mid_z = psf_conven.shape[2] // 2
    H = S + 2 * edge
    W = S + 2 * edge
    PSF_FFT_conven1 = []
    PSF_confocal_FFT_conven1 = []
    for zi in range(mid_z - RANG, mid_z + RANG, 1):  # 记得修改  对应的数据集
        psf = Cutting_PSF(psf_conven[:, :, zi]).astype(np.float32, copy=False)
        if abs(zi-mid_z) > int(beta*100):
            scale = np.exp(-alpha * ((abs(zi - mid_z) / 100.0 - beta) ** 2) / (beta ** 2)).astype(psf.dtype)
        else:
            scale = 1
        psf_confocal = psf * scale
        F_psf_bg1 = np.fft.rfft2(psf, s=(H, W))
        F_psf_confocal_bg1 = np.fft.rfft2(psf_confocal, s=(H, W))
        PSF_FFT_conven1.append(F_psf_bg1.astype(np.complex64, copy=False))
        PSF_confocal_FFT_conven1.append(F_psf_confocal_bg1.astype(np.complex64, copy=False))
    PSF_FFT_conven = np.stack(PSF_FFT_conven1, axis=2)
    PSF_confocal_FFT_conven = np.stack(PSF_confocal_FFT_conven1,axis=2)
    Data_train= CustomDataset_remove_Bg(image_dir_train,RANG,S,edge,PSF_FFT_conven,PSF_confocal_FFT_conven,
                                np_transforms.Compose([np_transforms.RandomCrop(S+2*edge),
                                                                                np_transforms.RandomHorizontalFlip(),
                                                                                np_transforms.RandomVerticalFlip(),
                                                                                np_transforms.ToTensor()
                                                                        ]))
    train_loader = torch.utils.data.DataLoader(Data_train, batch_size=batch_size, shuffle=True, num_workers=1)
    Data_eval = CustomDataset_remove_Bg(image_dir_eval, RANG,S,edge,PSF_FFT_conven,PSF_confocal_FFT_conven,
                                  np_transforms.Compose([np_transforms.RandomCrop(S+2*edge),
                                                         np_transforms.RandomHorizontalFlip(),
                                                         np_transforms.RandomVerticalFlip(),
                                                         np_transforms.ToTensor()
                                                         ]))
    eval_loader = torch.utils.data.DataLoader(Data_eval, batch_size = batch_size, shuffle=True, num_workers=1)
    model = sfhformer_haze().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 4e-8)
    criterion = nn.L1Loss()
    loss_history = []

    min_valid_mse = 1    #  设置验证集的最小损失  值
    start_ep = 0
    # 从检查点  开始运行
    if os.path.isfile(os.path.join(path_model,"ep_120checkpoint.pth")):
        checkpoint = torch.load(os.path.join(path_model,"ep_120checkpoint.pth"), map_location='cpu')
        start_ep = checkpoint['epoch']
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
        LOSS_ALL=0   #  累加损失
        t1 = default_timer()

        for i, (noise_img, Ture_img) in enumerate(train_loader):
            # Break when reaching the batch_per_ep limit
            train_loss=0
            if i >= batch_per_ep:
                break
            # 将数据加载到 GPU（或 CPU）
            noise_img = noise_img.to(device)
            Ture_img = Ture_img.to(device)
            # 前向传播
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  #  梯度裁剪  防止梯度爆炸
            optimizer.step()
            train_l2_step += 1
            LOSS_ALL +=train_loss.item()
            # 打印当前 batch 的损失信息
            if (i + 1) % 100 == 0:  # 每 10 个 batch 打印一次
                print(f"Train_Batch {i + 1}/{batch_per_ep}, Loss_all:{train_loss.item():.6f}")

        avg_loss = LOSS_ALL / train_l2_step
        loss_history.append(avg_loss)  # 记录每个 epoch 的平均损失

        ################### valid ###################
        model.eval()  # 将模型设置为评估模式
        valid_mse = 0
        valid_define_loss = 0
        xx_list = []
        yy_list = []
        im_list = []
        with torch.no_grad():
            for i, (lowimg, highimg) in enumerate(eval_loader):
                #if i >= 10:
                #    break
                #print(f"Batch {i}: lowimg dtype={lowimg.dtype}, highimg dtype={highimg.dtype}")
                lowimg = lowimg.to(device)
                highimg = highimg.to(device)
                output = model(lowimg)
                valid_mse += criterion(output, highimg).item()     #  验证的   损失函数  去掉  傅里叶变换部分

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







