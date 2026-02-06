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
        return (int(np.round(res[0][1], 0)), int(np.round(res[0][0] * 10, 3)))
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

def getBgsub(
    img: plt.figure,
    bg: plt.figure,
    img_type: type | np.dtype | None = None,
):
    """
    Subtracts the background from the image.

    Args:
        img (np.array): The image to be background subtracted.
        bg (np.array): The background image to be subtracted.
        img_type (np.dtype, optional): Desired dtype for the result. Defaults to the
            dtype of ``img`` and converts Pillow's int32 representation of 16-bit PNGs
            back to ``uint16`` to avoid losing dynamic range.

    Returns:
        np.array: The background-subtracted image.
    """
    img_arr = np.asarray(img)
    bg_arr = np.asarray(bg)

    if img_arr.shape != bg_arr.shape:
        raise ValueError("img and bg must have the same shape.")

    if img_type is None:
        target_dtype = img_arr.dtype
        if (
            np.issubdtype(target_dtype, np.signedinteger)
            and target_dtype.itemsize > 2
        ):
            # Pillow loads 16-bit PNGs as int32; check if the data actually fits
            # inside uint16 so we can write out a genuine 16-bit image.
            img_min = img_arr.min()
            bg_min = bg_arr.min()
            if img_min >= 0 and bg_min >= 0:
                combined_max = max(img_arr.max(), bg_arr.max())
                if combined_max <= np.iinfo(np.uint16).max:
                    target_dtype = np.uint16
    else:
        target_dtype = np.dtype(img_type)

    res = img_arr.astype(np.float64, copy=False) - bg_arr.astype(np.float64, copy=False)
    np.maximum(res, 0, out=res)
    return res.astype(target_dtype, copy=False)

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
