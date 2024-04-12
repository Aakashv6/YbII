import numpy as np

def fibout_dia(wl, fl, mfd):
    return (4 * wl * fl) / (mfd * np.pi)

def I_sat(p, w0, Is):
    return p / (np.pi * w0**2 * Is)