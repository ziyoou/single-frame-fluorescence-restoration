import multiprocessing
import randimage
import matplotlib.pyplot as plt
import numpy as np
import os


wt_dir = r"./dataset/denoise/rand_image_512/"  # SPECIFY THE WRITING DIRECTORY HERE FOR THE 100K IMAGES

L = 512  # Image size (LxL pixels)  方便裁剪  将尺寸扩大两倍
M = 10000  # 高分辨图  数量
def gen_random_image(i):
    while True:
        # 生成图像并二值化
        tmp = randimage.get_random_image((L, L))
        tmp = np.matmul(tmp, [0.2989, 0.5870, 0.1140])
        # 保存有效图像
        plt.imsave(os.path.join(wt_dir, f"{i:04d}.png"), tmp, cmap='gray')
        return i

def main():
    pool = multiprocessing.Pool(8)  # NO. OF POOLS NEED TO BE ADJUSTED BASED ON YOUR HARDWARE
    ii = pool.map(gen_random_image, range(M))

if __name__ == '__main__':
    main()


