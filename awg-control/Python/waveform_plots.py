import scipy.signal as scisig
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# file = np.load("data/waveform_data.npz", allow_pickle=True)
# signal = file['signal']
# wfm = file['wfm'].item()
# sr = wfm.sample_rate

sr = 614.4e6

signal = np.load("data/test.npy")
f, t, Sxx = scisig.stft(signal, fs=sr, nperseg=256*100)
f /= 1e6
t *= 1e3
Sxx[abs(Sxx) < 0.01] = 0
f_plot = np.logical_and(f > 60, f < 100)
fig, ax = plt.subplots()
im = ax.pcolormesh(t, f[f_plot], np.abs(Sxx[f_plot, :]), shading='gouraud')
plt.title("Signal Spectrogram Frequency")
plt.ylabel('Frequency [MHz]')
plt.xlabel('Time [ms]')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
if True:
    plt.savefig("figs/Spectrogram-frequency.png", dpi=1200)

fig, ax = plt.subplots()
im = ax.pcolormesh(t, f[f_plot], np.angle(Sxx[f_plot, :]), shading='gouraud')
plt.title("Signal Spectrogram Phase")
plt.ylabel('Frequency [MHz]')
plt.xlabel('Time [ms]')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
if True:
    plt.savefig("figs/Spectrogram-phase.png", dpi=1200)
