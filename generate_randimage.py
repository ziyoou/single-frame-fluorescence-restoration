import multiprocessing
import randimage
import matplotlib.pyplot as plt
import numpy as np
import os


wt_dir = r"./dataset/denoise/rand_image_512/"  # SPECIFY THE WRITING DIRECTORY HERE FOR THE 100K IMAGES

L = 512  # Image size (LxL pixels); doubled for convenient cropping
M = 10000  # Number of high-resolution images
def gen_random_image(i):
    while True:
        # Generate and binarize the image
        tmp = randimage.get_random_image((L, L))
        tmp = np.matmul(tmp, [0.2989, 0.5870, 0.1140])
        # Save valid images
        plt.imsave(os.path.join(wt_dir, f"{i:04d}.png"), tmp, cmap='gray')
        return i

def main():
    pool = multiprocessing.Pool(8)  # NO. OF POOLS NEED TO BE ADJUSTED BASED ON YOUR HARDWARE
    ii = pool.map(gen_random_image, range(M))

if __name__ == '__main__':
    main()


