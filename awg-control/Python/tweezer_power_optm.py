from PIL import Image
import lmfit
import numpy as np
import numpy.linalg as la
from typing import List, Tuple
import lib.plotting.pyplotdefs as pd
import matplotlib.pyplot as plt
from lib.AWG import *
from lib.waveform import save_wfm
import time
import scipy.ndimage as ndimage
from lib import basler
from pypylon import pylon
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from pathlib import Path
import os
from matplotlib.patches import Rectangle
from datetime import datetime
import cv2

def find_peaks(
    img: np.ndarray,
    plot_peaks=False,
) -> np.ndarray:
    peakfilter_size = (20, 20)
    # print(img.max())
    threshold = img.max() / 3.5
    img_max = ndimage.maximum_filter(img, size=peakfilter_size)
    img_min = ndimage.minimum_filter(img, size=peakfilter_size)
    diff = (img_max - img_min) > threshold
    img_max[diff == 0] = 0
    labeled, num_objs = ndimage.label(img_max)
    img_slices = ndimage.find_objects(labeled)
    xc = []
    yc = []
    # pd.Plotter().imshow(img).colorbar().savefig(f"figs/img.png")
    print(f"{num_objs} peaks found")
    for k, slice in enumerate(img_slices):
        xc.append(int(slice[0].start + (slice[0].stop - slice[0].start) / 2))
        yc.append(int(slice[1].start + (slice[1].stop - slice[1].start) / 2))

    peak_locs = np.sort(np.array([xc,yc]).T, axis=0)
    # peak_locs = process_image(img)
    # sys.exit(0)
    if plot_peaks:
        P = pd.Plotter().imshow(img).colorbar()
        ax = P.ax
        w = CROP_W
        h = CROP_H
        # w = peakfilter_size[0]
        # h = peakfilter_size[1]
        for k, (i0, j0) in enumerate(peak_locs):
            ax.add_patch(
                Rectangle(
                    (j0 - h // 2, i0 - w // 2),
                    # (j0, i0),
                    w, h,
                    edgecolor='red',
                    facecolor='none',
                    lw=0.05
                )
            )
            ax.annotate(
                k,
                (j0,i0),
                fontsize=1,
                color='r',
                alpha=1,
                antialiased=True
            )
        P.grid(False)
        P.savefig(PROBE_DET_FILE.parent.joinpath("peaks.png"))
        P.close()
            

    return peak_locs

def get_powers(
    img: np.ndarray,
    peak_locs,
    plot_crop=False,
) -> np.ndarray:
    w = CROP_W
    h = CROP_H
    # background = img[:1000, :1000].mean()
    subimgs = [
        img[i0 - h // 2 : i0 + h // 2, j0 - w // 2 : j0 + w // 2]
        for i0, j0 in peak_locs
    ]
    power = np.array([subimg.sum() for subimg in subimgs])
    if plot_crop:
        try:
            os.mkdir(PROBE_DET_FILE.parent.joinpath("peaks"))
        except FileExistsError:
            pass
        for k, subimg in enumerate(subimgs):
            pd.Plotter().imshow(subimg).colorbar().savefig(
                PROBE_DET_FILE.parent.joinpath(f"peaks/peak_{k}.png")
            ).close()
    return power

def plot(basler_cam: basler, init_power: np.ndarray, peak_locs: np.ndarray):
    img = grab_multi(basler_cam, 50)
    corrected_power = get_powers(img, peak_locs)
    (pd.Plotter()
        .plot(corrected_power[::-1], marker='.', linestyle="-", color="C0", label="corrected")
        .plot(init_power[::-1], marker='.', linestyle="-", color="C1", label="initial")
        .set_xlabel("tweezer index")
        .set_ylabel("power (arb unit)")
        .legend()
        .set_title("")
        # .savefig("figs/corrected_power.png")
        .savefig(PROBE_DET_FILE.parent.joinpath("corrected_power.png"))
        .close()
    )
    (pd.Plotter()
        .plot(corrected_power[::-1] / corrected_power.max(), marker='.', linestyle="-", color="C0", label="corrected")
        .plot(init_power[::-1] / init_power.max(), marker='.', linestyle="-", color="C1", label="initial")
        .set_xlabel("tweezer index")
        .set_ylabel("power (normalized)")
        .legend()
        .set_title("")
        .savefig(PROBE_DET_FILE.parent.joinpath("corrected_power_norm.png"))
        .close()
    )

def init_basler():
    basler_cam = basler.connect_camera()
    basler_cam.PixelFormat.SetValue("Mono12")
    basler_cam.ExposureTime.SetValue(50.0) #1200
    basler_cam.Gain.SetValue(0.0)
    basler_cam.AcquisitionMode.SetValue("Continuous")
    basler_cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return basler_cam

def initilze_awg():
    array_wfm = Waveform()
    array_wfm.from_file(TWZR_WFM_FILE)
    save_wfm(array_wfm, BACKUP_FILE)
    sig = create_static_array(array_wfm)
    condition = [SPCSEQ_ENDLOOPONTRIG, SPCSEQ_ENDLOOPALWAYS]
    awg = AWG()
    awg.open(id=0)
    awg.set_sampling_rate(int(614.4e6))
    awg.toggle_channel(0, amplitude=2500)
    awg.set_trigger(EXT0=SPC_TM_POS)
    awg.set_sequence_mode(2)  # partition AWG memory into 2 segments
    awg.write_segment(sig, segment=0)
    awg.write_segment(sig, segment=1)
    awg.configure_step(step=0, segment=0, nextstep=1, loop=1, condition=condition[0])
    awg.configure_step(step=1, segment=1, nextstep=0, loop=1, condition=condition[0])
    awg.run()
    awg.force_trigger()
    return awg, array_wfm  

def grab(camera):
    res = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    img = res.Array
    res.Release()
    return np.array(img)

def grab_multi(camera: basler, n: int):
    # img_arr = np.zeros((n, 2748, 3840))
    # for i in range(n):
    #     img_arr[i] = grab(camera)
    # return img_arr.mean(axis=0)
    img_arr = []
    for i in range(n):
        res = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        # res.Release()
        img_arr.append(np.array(res.Array))
    return np.mean(img_arr, axis=0)
    

def reload_awg(awg: AWG, wfm: Waveform):
    sig = create_static_array(wfm)
    condition = [SPCSEQ_ENDLOOPONTRIG, SPCSEQ_ENDLOOPALWAYS]
    awg.stop()
    # awg.reset()
    # awg.set_sampling_rate(int(614.4e6))
    # awg.toggle_channel(0, amplitude=2500)
    # awg.set_trigger(EXT0=SPC_TM_POS)
    # awg.set_sequence_mode(2)  # partition AWG memory into 2 segments
    awg.write_segment(sig, segment=0)
    awg.write_segment(sig, segment=1)
    # awg.configure_step(step=0, segment=0, nextstep=1, loop=1, condition=condition[0])
    # awg.configure_step(step=1, segment=1, nextstep=0, loop=1, condition=condition[0])
    awg.run()
    awg.force_trigger()

def save_weights(save_file: str):
    basler_cam = init_basler()
    img = grab_multi(basler_cam, 100)
    powers = get_powers(img, find_peaks(img))
    weights = np.mean(powers) / powers
    np.save(save_file, weights)

def get_weighted_mask(type, init_power=None):
    # probe detuning based weights
    if type == 'probe':
        # polarz = -1.24e-3 # kHz/uk horizontally polarized tweezer 7.27.23
        polarz = -5.6e-3 # MHz/uk vertically polarized tweezer 8.8.23
        probe_det = np.load(PROBE_DET_FILE)[::-1]
        depth_diff = (probe_det - np.mean(probe_det)) / polarz
        approx_depth = MEAN_DEPTH + depth_diff  # average depth of 575uk is free param
        weighted_mask = (approx_depth / np.mean(approx_depth))
        weighted_mask *= init_power.mean() / init_power  # maintain current power distribution
    # fixed targer power distribution
    if type == 'target':
        weighted_mask = np.load(TARGET_WEIGHT_FILE)

    return weighted_mask

def doit_fixed_step(max_loop: int, step_size: float, err_tol = 0.1):
    
    
    
    basler_cam = init_basler()
    awg, twzr_wfm = initilze_awg()
    time.sleep(5.0)
    init_img = grab_multi(basler_cam, 10)
    peak_locs = find_peaks(init_img, plot_peaks=True)

    if len(peak_locs) != NT:
        basler.disconnect_camera(basler_cam)
        awg.stop()
        awg.close()
        print(f"found {len(peak_locs)} peaks, {NT} peaks specified, exiting")
        sys.exit(0)

    init_power = get_powers(init_img, peak_locs, plot_crop=False)
    weighted_mask = get_weighted_mask(type=OPTM_TYPE, init_power=init_power)

    # do iteration
    power_errs = np.zeros((max_loop, len(peak_locs)))
    for i in range(max_loop):
        img = grab_multi(basler_cam, 50)
        powers = get_powers(img, peak_locs)
        power_errs[i] = (weighted_mask*powers - np.median(weighted_mask*powers)) / np.median(weighted_mask*powers)
        # power_errs[i] = get_power_diff(img, weighted_mask, peak_locs)
        print(f"\r on loop {i:2}/{max_loop-1}, avg error: {np.mean(abs(power_errs[i])):.4f}", end="")
        if np.mean(abs(power_errs[i])) <= err_tol:
            power_errs[i:] = power_errs[i]
            break
        twzr_wfm.amplitude = twzr_wfm.amplitude - step_size * power_errs[i]
        reload_awg(awg, twzr_wfm)
        time.sleep(0.5)
    print()

    if SAVE_FILE is not None:
        save_wfm(twzr_wfm, SAVE_FILE)
    
    (pd.Plotter()
        .plot(np.mean(abs(power_errs), axis=1), marker='.', linestyle="-", color="C0", label="average abs error")
        .set_xlabel("iteration index")
        .set_ylabel("error (normlized)")
        .legend()
        .savefig("figs/error_trace.png")
    )

    # img = grab_multi(basler_cam, 50)
    corrected_power = powers
    (pd.Plotter()
        .plot(corrected_power[::-1], marker='.', linestyle="-", color="C0", label="corrected")
        .plot(init_power[::-1], marker='.', linestyle="-", color="C1", label="initial")
        .set_xlabel("tweezer index")
        .set_ylabel("power (arb unit)")
        .legend()
        .set_title("")
        # .savefig("figs/corrected_power.png")
        .savefig(PROBE_DET_FILE.parent.joinpath("corrected_power.png"))
        .close()
    )

    img = grab_multi(basler_cam, 50)
    corrected_power = get_powers(img, peak_locs)
    (pd.Plotter()
        .plot(corrected_power[::-1], marker='.', linestyle="-", color="C0", label="corrected")
        .plot(init_power[::-1], marker='.', linestyle="-", color="C1", label="initial")
        .set_xlabel("tweezer index")
        .set_ylabel("power (arb unit)")
        .legend()
        .set_title("")
        # .savefig("figs/corrected_power.png")
        .savefig(PROBE_DET_FILE.parent.joinpath("corrected_power_2.png"))
        .close()
    )

    (pd.Plotter()
        .plot(corrected_power[::-1] / corrected_power.max(), marker='.', linestyle="-", color="C0", label="corrected")
        .plot(init_power[::-1] / init_power.max(), marker='.', linestyle="-", color="C1", label="initial")
        .set_xlabel("tweezer index")
        .set_ylabel("power (normalized)")
        .legend()
        .set_title("")
        .savefig(PROBE_DET_FILE.parent.joinpath("corrected_power_norm.png"))
        .close()
    )
      
    awg.stop()
    awg.close()
    basler.disconnect_camera(basler_cam)


# def doit_nonlinear_optm(max_variation=2):

#     def minimize_obj_func(amp):
#         # global twzr_wfm, awg, weighted_mask, basler_cam, peak_locs
#         twzr_wfm.amplitude = amp
#         reload_awg(awg, twzr_wfm)
#         time.sleep(0.1)
#         img = grab_multi(basler_cam, 1)
#         current_power = get_powers(img, peak_locs)
#         # current_power_norm = (current_power - current_power.mean()) / current_power.std()
#         # target_power_norm = (target_power - target_power.mean()) / target_power.std()
#         # diff = np.mean(abs(target_power_norm - current_power_norm))
#         weighted_power = current_power * weighted_mask
#         error = (weighted_power - np.mean(weighted_power)) / np.mean(weighted_power)
#         return abs(error).mean()
    
#     basler_cam = init_basler()
#     awg, twzr_wfm = initilze_awg()
#     time.sleep(5)
    
#     init_img = grab_multi(basler_cam, 10)
#     peak_locs = find_peaks(init_img, plot_peaks=True)
#     init_power = get_powers(init_img, peak_locs, plot_crop=False)

#     # fixed targer power distribution
#     # target_img = np.load(TARGET_IMG_FILE)
#     # target_power = get_powers(target_img, find_peaks(target_img), plot_indiv=False)
#     # weighted_mask = target_power.mean() / target_power

#     # probe detuning calculated target power distribution
#     probe_det = np.load(PROBE_DET_FILE)[::-1]
#     depth_diff = (probe_det - probe_det.mean()) / -1.24e-3 # -1.24 kHz/uk, measured on 7.27.23
#     approx_depth = MEAN_DEPTH + depth_diff  # average depth
#     weighted_mask = (approx_depth / np.median(approx_depth)) ** 2
#     weighted_mask *= init_power.mean() / init_power

#     x0 = twzr_wfm.amplitude
#     max_variation = 2.0
#     bound_range = Bounds([x/max_variation for x in x0], [x*max_variation for x in x0])
#     # sum_constraint = LinearConstraint(np.ones(len(x0)), 0, 2**15-1)
#     result = minimize(
#         minimize_obj_func, x0,
#         method='trust-constr', bounds=bound_range,# constraints=sum_constraint,
#         options={"disp": True, "maxiter": 30},
#         # args=(target_ratios, twzr_array, basler_cam, awg),
#     )
#     save_wfm(twzr_wfm, SAVE_FILE)
    
#     plot(basler_cam, init_power, peak_locs)
#     awg.stop()
#     awg.close()
#     basler.disconnect_camera(basler_cam)

if __name__ == "__main__":
    TODAY_PATH = Path("data").joinpath(datetime.now().strftime("%Y%m%d"))

    TWZR_WFM_FILE = TODAY_PATH.joinpath("array20_df=0.9MHz.npz")
    
    PROBE_DET_FILE = (
        # Path("C:/Users/Covey Lab/Documents/Andor Solis/atomic_data")
        Path("/home/coveylab/Documents/Data/atomic_data")
        .joinpath("20240422")
        .joinpath("probe-scan_004")
        .joinpath("probe-det.npy")
    )
    
    SAVE_FILE = TODAY_PATH.joinpath("array20_df=0.9MHz_init.npz")
    BACKUP_FILE = TODAY_PATH.joinpath("backup.npz")

    NT = 20 # number of peaks to find
    CROP_W = 45 # crop width
    CROP_H = 40 # crop height

    MAX_LOOP = 100
    STEP_SIZE = 35 # tune this for convergence
    MEAN_DEPTH = 1000 # estimated average depth, tune this if under/over correcting
    ERROR_TOLERANCE = 0.0025 # error threshold
    OPTM_TYPE = "probe" # use 'probe' or 'target' to calculate target power distribution
    doit_fixed_step(max_loop=MAX_LOOP, step_size=STEP_SIZE, err_tol=ERROR_TOLERANCE)

    # MAX_VARIATION = 1.1
    # doit_nonlinear_optm(max_variation=MAX_VARIATION)

    # save_weights("data/20230808_array20_optm-weights.npy")

    # test_grab()
    