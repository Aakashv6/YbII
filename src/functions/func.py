import numpy as np

def gaussian_dist(x, amp, x0, sigma, c=0):
    return amp * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-1 * ((x - x0)**2 / (2 * sigma**2))) + c

def lorentzian_dist(x, amp, x0, gamma, c=0):
    return amp * (1 / np.pi) * (gamma / (gamma**2 + (x - x0)**2)) + c