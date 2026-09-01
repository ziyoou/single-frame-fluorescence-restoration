import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # First-order Bessel function of the first kind
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg'


# Defocused PSF function
def defocused_psf(NA, wavelength0, n, pixel_size, z):
    """
    Calculate the PSF at defocus distance z (µm) in a medium with refractive
    index n using the angular spectrum method.

    Parameters
    ----
    NA          : Numerical aperture (including n)
    wavelength0 : Vacuum wavelength (µm)
    n           : Refractive index of the medium
    size        : Computational grid size in pixels (size x size)
    pixel_size  : Pixel size (µm)
    z           : Defocus distance (µm)

    Returns
    ----
    psf         : Normalized PSF (2D NumPy array)
    """

    # Effective wavelength and wave vector
    wavelength_eff = wavelength0 / n
    k = 2 * np.pi / wavelength_eff  # [rad/µm]

    size = 256
    # Spatial-frequency coordinates in the camera plane
    fx = np.fft.fftfreq(size, d=pixel_size)  # [µm^-1]
    fy = np.fft.fftfreq(size, d=pixel_size)
    # print(fx[:10])
    # print("************************")

    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))
    # print(FX[2,0]-FX[2,1])
    # print("************************")

    # Pupil radius (cutoff frequency)
    f_cutoff = NA / wavelength0   # [1/µm]
    pupil = (FX**2 + FY**2 <= f_cutoff**2).astype(float)

    # Propagation operator H(u,v;z)
    # Convert spatial frequencies to propagation-direction components
    fx_phys = FX * 2 * np.pi
    fy_phys = FY * 2 * np.pi
    inside = k**2 - (fx_phys**2 + fy_phys**2)
    inside[inside <= 0] = 0  # Filter evanescent waves
    W = np.sqrt(inside)
    H = np.exp(1j * W * z)

    # Defocused pupil → inverse FFT → field distribution
    pupil_prop = pupil * H
    field = np.fft.ifft2(np.fft.ifftshift(pupil_prop))
    psf = np.abs(field)**2
    #psf /= psf.max()
    psf_defocus = np.fft.fftshift(psf)


    ## Crop the PSF
    center = np.array(psf_defocus.shape) // 2
    # Extract intensity values from the center row
    center_row = psf_defocus[center[0], :]

    # Extract intensity values from the right half of the center row
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
        threshold_index = len(right_half_row) - 1  # Use the last index if no threshold crossing is found

    cropped_psf = psf_defocus[center[0] - threshold_index + 1:center[0] + threshold_index,
                  center[1] - threshold_index + 1:center[1] + threshold_index]

    return cropped_psf.astype(np.float32)



"""
NA = 0.6
n = 1   # Refractive index of the objective environment
Scale = 1
wavelength = 580e-9
pixelsize = 6.5/40*1e-6
pixelsize = pixelsize/Scale


z = 0.8 *wavelength/(NA*NA)
#z = 0
# Example: PSF in water (n=1.33)
cropped_psf = defocused_psf(NA, wavelength, n, pixelsize, z)

print(cropped_psf.shape)
plt.figure(figsize=(6, 6))
plt.imshow(cropped_psf, cmap="inferno")
plt.title(f"Cropped PSF (NA={NA}, λ0={wavelength*1e6:.1f} µm, n={n}, z={z*1e6:.1f} µm)")
plt.xlabel("x [µm]")
plt.ylabel("y [µm]")
plt.colorbar(label="Normalized Intensity")
plt.show()
"""
#
# Z0 = 2*wavelength/(NA*NA)
# PSF_ALL = []
# for i in range(20):
#     z = Z0*i/20
#     psf_z = defocused_psf(NA, wavelength, n, pixelsize, z)
#     PSF_ALL.append(psf_z)
#
# a=PSF_ALL[0]
# b=PSF_ALL[18]
# print(a.shape)
# print(b.shape)
# print(type(a))
# NA = 0.6
# wavelength0 = 590e-9
# n = 1
# pixel_size = 6.5/40*1e-6









