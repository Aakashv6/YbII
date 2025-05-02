from lib.waveform import *
import scipy.signal as scisig
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
import itertools
from pathlib import Path
# import cupy as cp

def plot(signal):
    sr = SAMPLING_RATE
    f, t, Sxx = scisig.stft(signal, fs=sr, nperseg=256*100)
    f /= 1e6
    t *= 1e3
    # Sxx[abs(Sxx) < 0.01] = 0
    f_plot = np.logical_and(f > 90, f < 120)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(t, f[f_plot], np.abs(Sxx[f_plot, :]), shading='gouraud')
    plt.title("Signal Spectrogram Frequency")
    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Time [ms]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    plt.savefig("figs/Spectrogram-frequency.png", dpi=1200)
    

def obj(phi: np.ndarray):
    phi_pairs = phi[
        np.stack(np.triu_indices(len(phi), k=1), axis=-1)
    ]
    E_diff = np.sum([
        np.exp(1j * (phi_p[0] - phi_p[1])) + np.exp(1j * (phi_p[0] + phi_p[1]))
        for phi_p in phi_pairs
    ])
    return np.sqrt(E_diff.imag**2 + E_diff.real**2)

def obj_2(
    x: float,
    wfm: Waveform,
    idx: int,
):
    nt = len(wfm.omega)
    pairs = [(idx, p) for p in range(nt)]
    pairs.pop(idx)
    phases = wfm.phi.copy()
    phases[idx] = x
    phi_pairs = phases[pairs]
    E_minus = np.sum([
        np.exp(1j * (phi_p[0] - phi_p[1] ))
        for phi_p in phi_pairs
    ])
    return np.sqrt(E_minus.imag**2 + E_minus.real**2)

wfm_path = Path("data/array20.npz") # Path to the waveform file, modify this!!!!

wfm = Waveform()
wfm.from_file(wfm_path)
lower = [0]
upper = [np.pi * 2]
bound_range = Bounds(lower, upper)

init_err = obj(wfm.phi)

x0 = wfm.phi
results = minimize(
    obj, x0,
    method="trust-constr", bounds = bound_range,
    options={"disp": True},
    args=()
)
if results.success:
    wfm.phi = results.x
    
# for i in range(1, len(wfm.omega)-1):
#     x0 = wfm.phi[i]
#     result = minimize(
#         obj_2, x0,
#         method="trust-constr", bounds = bound_range,
#         options={"disp": True},
#         args=(wfm, i)
#     )
#     wfm.phi[i] = result.x if result.success else wfm.phi[i]
    
final_err = obj(wfm.phi)
print(f"initial amplitude: {init_err}, final amplitude: {final_err}")
# save_wfm(wfm, wfm_path.parent.joinpath(wfm_path.stem + "_optm.npz"))
save_wfm(wfm, wfm_path)




