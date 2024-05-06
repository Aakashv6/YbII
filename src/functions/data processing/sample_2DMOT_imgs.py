import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Fitting Module
from lmfit.models import gaussian2d

H = 2748
W = 3840

def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y):
    """Calculate the 2D Gaussian value for coordinates x, y with specified mean and standard deviation."""
    prefactor = 1 / (2 * np.pi * sigma_x * sigma_y)
    exponent = -((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2))
    return prefactor * np.exp(exponent)

# Background noise has -
# 1. Gaussian noise
# 2. Salt & Pepper noise
def noise_img(gn=0, spn=0):

    GN = 255 * np.random.normal(0, 1, (H, W))
    SPN = 50 * np.random.randint(2, size=(H, W))

    return gn * GN + spn * SPN

# 2DMOT background has -
# 1. Yb fluorescence => elliptical 2D gaussians
# 2. 2DMOT => Circular 2D gaussian
def bg_img(mu=np.array([[0, 0], [0, 0]]), sigma=np.array([[1, 1], [1, 1]])):

    dark_bg = np.zeros((H, W), dtype=float)

    # Parameters
    mu_x1, mu_y1 = mu[0,0], mu[0,1]
    mu_x2, mu_y2 = mu[1,0], mu[1,1]

    sigma_x1, sigma_y1 = sigma[0,0], sigma[0,1]
    sigma_x2, sigma_y2 = sigma[1,0], sigma[1,1]

    # Create a meshgrid for the x and y values
    x1 = np.linspace(-2, 2, 3840)
    y1 = np.linspace(-10, 10, 2748)
    X1, Y1 = np.meshgrid(x1, y1)

    x2 = np.linspace(-10, 10, 3840)
    y2 = np.linspace(-2, 2, 2748)
    X2, Y2 = np.meshgrid(x2, y2)

    # Compute the 2D Gaussian matrix
    Zx = 255 * (1/10) * gaussian_2d(X1, Y1, mu_x1, mu_y1, sigma_x1, sigma_y1)
    Zy = 255 * (1/10) * gaussian_2d(X2, Y2, mu_x2, mu_y2, sigma_x2, sigma_y2)

    img_bg = dark_bg + Zx + Zy

    return img_bg

def mot2d_img(mu=[0, 0], sigma=[1, 1]):

    # Create a meshgrid for the x and y values
    # Parameters
    mu_x, mu_y = mu[0], mu[1]
    sigma_x, sigma_y = sigma[0], sigma[1]

    cc = np.random.randint(low = -5, high = +6, size=(2, 2))
    x = np.linspace(-50 + cc[0, 0], 50 + cc[0, 1], 3840)
    y = np.linspace(-50 + cc[1, 0], 50 + cc[1, 1], 2748)
    X, Y = np.meshgrid(x, y)

    mot2d = 255 * gaussian_2d(X, Y, mu_x, mu_y, sigma_x, sigma_y)

    return mot2d

def mot2d_img_gen(dir="C:/Users/aak6a/YbII/data/generated_2DMOT_images/", num_imgs=10, plot_sample=False):

    for i in range(num_imgs):
        img_sig = noise_img(0.01, 0.01) + bg_img() + mot2d_img()
        img_bg = noise_img(0.01, 0.01) + bg_img()

        im_sign = Image.fromarray(img_sig).convert("L")
        im_bg = Image.fromarray(img_bg).convert("L")

        im_sig_name = dir + "2DMOT_exp_" + str(i).zfill(2) + "_0_ms_power_" + str(2*i).zfill(3) + "_9_mW_fL_" + str(2*i).zfill(3) + "_" + str(5*i).zfill(3) + "_THz.bmp"
        im_bg_name = dir + "2DMOT_exp_" + str(i).zfill(2) + "_0_ms_power_" + str(2*i).zfill(3) + "_6_mW_fL_" + str(2*i).zfill(3) + "_" + str(5*i).zfill(3) + "_THz_bg.bmp"

        im_sign.save(im_sig_name)
        im_bg.save(im_bg_name)
    
    if plot_sample:
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(noise_img(0.01, 0.01) + bg_img() + mot2d_img(), cmap='grey', vmin=0, vmax=255)
        ax[0].set_title('2DMOT')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')

        ax[1].imshow(noise_img(0.01, 0.01) + bg_img(), cmap='grey', vmin=0, vmax=255)
        ax[1].set_title('Background')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')

        fig.suptitle('Generated 2DMOT Image sample')

    plt.show()

mot2d_img_gen(num_imgs=10, plot_sample=True)