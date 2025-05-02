import numpy as np

def gaussian_dist(x, amp, x0, sigma, c=0):
    return amp * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-1 * ((x - x0)**2 / (2 * sigma**2))) + c

def lorentzian_dist(x, amp, x0, gamma, c=0):
    return amp * (1 / np.pi) * (gamma / (gamma**2 + (x - x0)**2)) + c

# 07/15 2 step piecewise function for fitting the transition from the CS MOT to the G MOT
def two_piecewise_func(t, a, b, tau, t0):
    return np.where(t < t0, a+b, a * np.exp(- (t - t0) / tau) + b)

# 07/15 3 steps piecewise function for fitting the transition from the CS MOT to the G MOT and the decay of the G MOT
def three_piecewise_func(x, x0, x1, h, m, tau):
    condlist = [x < x0, (x >= x0) & (x <= x1), x > x1]
    funclist = [
        lambda x: h,
        lambda x: m * (x-x0) + h,
        lambda x: (m * (x1-x0) + h) * np.exp(- (x - x1) / tau) + 0
    ]
    return np.piecewise(x, condlist, funclist)