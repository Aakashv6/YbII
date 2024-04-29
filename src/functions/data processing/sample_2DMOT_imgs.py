import numpy as np
import matplotlib.pyplot as plt

# Fitting Module
from lmfit.models import gaussian2d

H = 2748
W = 3840

dark_bg = np.zeros((H, W), dtype=float)

def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y):
    """Calculate the 2D Gaussian value for coordinates x, y with specified mean and standard deviation."""
    prefactor = 1 / (2 * np.pi * sigma_x * sigma_y)
    exponent = -((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2))
    return prefactor * np.exp(exponent)

# Parameters
mu_x1, mu_y1 = 0, 0
mu_x2, mu_y2 = 0, 0 

sigma_x1, sigma_y1 = 1, 1
sigma_x2, sigma_y2 = 1, 1

# Create a meshgrid for the x and y values
x1 = np.linspace(-2, 2, 3840)
y1 = np.linspace(-10, 10, 2748)
X1, Y1 = np.meshgrid(x1, y1)

x2 = np.linspace(-10, 10, 3840)
y2 = np.linspace(-2, 2, 2748)
X2, Y2 = np.meshgrid(x2, y2)

# Compute the 2D Gaussian matrix
Zx = 255 * (1/2) * gaussian_2d(X1, Y1, mu_x1, mu_y1, sigma_x1, sigma_y1)
Zy = 255 * (1/2) * gaussian_2d(X2, Y2, mu_x2, mu_y2, sigma_x2, sigma_y2)
GN = 255 * (1/10) * np.random.normal(0, 1, (H, W))
SPN = 100 * np.random.randint(2, size=(H, W))

# Create a meshgrid for the x and y values
# Parameters
mu_x, mu_y = 0, 0
sigma_x, sigma_y = 1, 1

x = np.linspace(-50, 50, 3840)
y = np.linspace(-50, 50, 2748)
X, Y = np.meshgrid(x, y)

mot2d = 255 * gaussian_2d(X, Y, mu_x, mu_y, sigma_x, sigma_y)

# Plotting
fig, ax = plt.subplots(1, 1)

ax.imshow(Zx + Zy + GN + SPN + mot2d, cmap='grey', vmin=0, vmax=255)
ax.set_title('2D Gaussian distribution')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()