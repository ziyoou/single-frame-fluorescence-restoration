import multiprocessing
import randimage
import matplotlib.pyplot as plt
import numpy as np
import os
import random

wt_dir = r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_512_0827_eval/"  # SPECIFY THE WRITING DIRECTORY HERE FOR THE 100K IMAGES

#wt_dir =r"E:\2025-lab\第二个工作-深度学习-blind-sim\06_27_speckle_resolution_youzi\rand_image_eval/"

L = 512  # Image size (LxL pixels)  方便裁剪  将尺寸扩大两倍
M = 500  # 高分辨图  数量
def gen_random_image(i):
    while True:
        # 生成图像并二值化
        tmp = randimage.get_random_image((L, L))
        tmp = np.matmul(tmp, [0.2989, 0.5870, 0.1140])
        #threshold = np.median(tmp) + random.uniform(-0.1, 0.1)
        # threshold = 0.5 + random.uniform(-0.2, 0.2)
        # tmp = (tmp > threshold).astype(np.uint8)

        # 判断是否为全 0 或全 1，若无效则继续生成
        if np.all(tmp == 0) or np.all(tmp == 1):
            continue

        # 保存有效图像
        plt.imsave(os.path.join(wt_dir, f"{i:04d}.png"), tmp, cmap='gray')
        return i

def main():
    pool = multiprocessing.Pool(8)  # NO. OF POOLS NEED TO BE ADJUSTED BASED ON YOUR HARDWARE
    ii = pool.map(gen_random_image, range(M))

if __name__ == '__main__':
    main()