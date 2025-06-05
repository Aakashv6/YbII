import math
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def splitDecimal(num: float) -> tuple:
    df_round = np.round(num, 3)
    res = np.array([math.modf(float(df_round))])
    if num < 0:
        return (-int(np.round(np.abs(res[0][1]), 0)), int(np.round(np.abs(res[0][0]) * 10, 3)))
    else:
        return (int(np.round(res[0][1], 0)), int(np.round(res[0][0], 3)))
# FYI, w = sqrt(2) * sigma

FM_polynomial = np.array([6.19975510e-07,  5.50488951e-05, -8.78214143e-05, -1.09927015e-02,
  1.96484719e-03,  8.57576696e-01, -1.81343982e-01]) # Channel 4 AOM FM curve centered around 96.9375 MHz, deviation = 6.65 MHz, see 05.14.25 Notion page

def fibout_dia(wl, fl, mfd):
    return (4 * wl * fl) / (mfd * np.pi)

def I_sat(p, w0, Is):
    return p / (np.pi * w0**2 * Is)

def getFMVVec(vec_det):
    """
    vec_det: vector of desired detunings in MHz, e.g., np.arange(-2.5, -1.4, 0.1)
    return: a list of tuples with (detuning, FM voltage) pairs
    """    # Sample x and y
    x_sample = np.linspace(-10, 10, 1000)
    y_sample = np.polyval(FM_polynomial, x_sample)
    # Invert numerically
    f_inv = interp1d(y_sample, x_sample, bounds_error=False, fill_value="extrapolate")
    x_val = f_inv(vec_det)
    res = [(round(e[0], 2), e[1]) for e in list(zip(vec_det, x_val))]
    return res

def getBgsub(img: plt.figure, bg: plt.figure, img_type: type=np.uint8):
    """
    Subtracts the background from the image.
    
    Args:
        img (np.array): The image to be background subtracted.
        bg (np.array): The background image to be subtracted.
    
    Returns:
        np.array: The background-subtracted image.
    """
    res = np.array(img, dtype=float) - np.array(bg, dtype=float)
    res[res < 0] = 0
    res = res.astype(img_type)
    return res

def exp_decay(x: np.array, a: float, b: float, c: float) -> np.array:
    """
    Exponential decay function with a constant offset.

    Args:
        x (np.array): independent variable, typically a time array.
        a (float): amplitude of the exponential decay.
        b (float): time constant, f=a/e if x=b.
        c (float): constant offset.

    Returns:
        np.array: evaluated exponential decay function at each point in x.
    """
    return a * np.exp(-x / b) + c

def gaussian(x, amp, x0, sigma, c):
    return amp * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-1 * ((x - x0)**2 / (2 * sigma**2))) + c

def gaussian2d(x, y, th, A, B, x0, y0, sx, sy):
    Xrel = x - x0
    Yrel = y - y0
    Xrot = np.cos(th) * Xrel + np.sin(th) * Yrel
    Yrot = -np.sin(th) * Xrel + np.cos(th) * Yrel
    return A * np.exp(- Xrot ** 2 / (2 * sx**2) - Yrot ** 2 / (2 * sy ** 2)) + B

def lorentzian(x, amp, x0, gamma, c=0):
    return amp * (1 / np.pi) * (gamma / (gamma**2 + (x - x0)**2)) + c

# 07/15 2 step piecewise function for fitting the transition from the CS MOT to the G MOT
def two_piecewise(t, a, b, tau, t0):
    return np.where(t < t0, a+b, a * np.exp(- (t - t0) / tau) + b)

# 07/15 3 steps piecewise function for fitting the transition from the CS MOT to the G MOT and the decay of the G MOT
def three_piecewise(x, x0, x1, h, m, tau):
    condlist = [x < x0, (x >= x0) & (x <= x1), x > x1]
    funclist = [
        lambda x: h,
        lambda x: m * (x-x0) + h,
        lambda x: (m * (x1-x0) + h) * np.exp(- (x - x1) / tau) + 0
    ]
    return np.piecewise(x, condlist, funclist)