import numpy as np
from typing import List, Tuple
import lib.plotting.pyplotdefs as pd
import matplotlib.pyplot as plt
from lib.AWG import *
from lib.waveform import save_wfm
import time
import scipy.ndimage as ndimage
from lib import basler
from pypylon import pylon
from pathlib import Path
import os
from matplotlib.patches import Rectangle
from datetime import datetime

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

def grab_multi(
    camera: basler,
    peak_locs: np.ndarray,
    n: int,
):
    w = CROP_W
    h = CROP_H
    power_arr = []
    for i in range(n):
        res = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        # res.Release()
        img = np.array(res.Array)
        powers = [
            img[i0 - h // 2 : i0 + h // 2, j0 - w // 2 : j0 + w // 2].sum()
            for i0, j0 in peak_locs
        ]
        power_arr.append(powers)
    return [np.mean(power_arr, axis=0), np.std(power_arr, axis=0)]
    

def reload_awg(awg: AWG, wfm: Waveform):
    sig = create_static_array(wfm)
    awg.stop()
    awg.write_segment(sig, segment=0)
    awg.write_segment(sig, segment=1)
    awg.run()
    awg.force_trigger()

def get_weighted_mask(init_power):
    # polarz = -1.24e-3 # kHz/uk horizontally polarized tweezer 7.27.23
    polarz = -5.6e-3 # MHz/uk vertically polarized tweezer 8.8.23
    probe_det = np.load(PROBE_DET_FILE)[::-1]
    depth_diff = (probe_det - np.mean(probe_det)) / polarz
    approx_depth = MEAN_DEPTH + depth_diff  # average depth of 575uk is free param
    weighted_mask = (approx_depth / np.mean(approx_depth))
    weighted_mask *= init_power.mean() / init_power  # maintain current power distribution

    return weighted_mask

def doit_fixed_step(max_loop: int, step_size: float, err_tol = 0.1):
    basler_cam = init_basler()
    awg, twzr_wfm = initilze_awg()
    time.sleep(0.1)
    init_img = grab(basler_cam)
    peak_locs = find_peaks(init_img, plot_peaks=True)

    if len(peak_locs) != NT:
        basler.disconnect_camera(basler_cam)
        awg.stop()
        awg.close()
        print(f"found {len(peak_locs)} peaks, {NT} peaks specified, exiting")
        sys.exit(0)

    init_power = grab_multi(basler_cam, peak_locs, SHOT_PER_LOOP)
    weighted_mask = get_weighted_mask(init_power[0])

    # do iteration
    power_errs = []
    for i in range(max_loop):
        powers = grab_multi(basler_cam, peak_locs, SHOT_PER_LOOP)
        p_err = (weighted_mask*powers[0] - np.median(weighted_mask*powers[0])) \
            / np.median(weighted_mask*powers[0])
        power_errs.append(np.mean(abs(p_err)))
        print(f"\r on loop {i:2}/{max_loop-1}, avg error: {power_errs[i]:.4f}", end="")
        if power_errs[i] <= err_tol:
            break
        twzr_wfm.amplitude = twzr_wfm.amplitude - step_size * p_err
        reload_awg(awg, twzr_wfm)
        time.sleep(0.1)
    print()
    # print(len(power_errs), len(power_errs[0]))
    if SAVE_FILE is not None:
        save_wfm(twzr_wfm, SAVE_FILE)
    
    (pd.Plotter()
        .plot(power_errs, marker='.', linestyle="-", color="C0", label="average abs error")
        .set_xlabel("iteration index")
        .set_ylabel("error (normlized)")
        .legend()
        .savefig(PROBE_DET_FILE.parent.joinpath("error_trace.png"))
        .close()
    )

    corrected_power = powers
    x = np.arange(0, NT, 1)
    (pd.Plotter()
        .errorbar(x, corrected_power[0][::-1], corrected_power[1][::-1], marker='.', linestyle="-", color="C0", label="corrected")
        .errorbar(x, init_power[0][::-1], init_power[1][::-1], marker='.', linestyle="-", color="C1", label="initial")
        .set_xlabel("tweezer index")
        .set_ylabel("power (arb unit)")
        .legend()
        .set_title("")
        .savefig(PROBE_DET_FILE.parent.joinpath("corrected_power.png"))
        .close()
    )

    corrected_power = grab_multi(basler_cam, peak_locs, SHOT_PER_LOOP)
    (pd.Plotter()
        .errorbar(x, corrected_power[0][::-1], corrected_power[1][::-1], marker='.', linestyle="-", color="C0", label="corrected")
        .errorbar(x, init_power[0][::-1], init_power[1][::-1], marker='.', linestyle="-", color="C1", label="initial")
        .set_xlabel("tweezer index")
        .set_ylabel("power (arb unit)")
        .legend()
        .set_title("")
        .savefig(PROBE_DET_FILE.parent.joinpath("corrected_power_2.png"))
        .close()
    )
      
    awg.stop()
    awg.close()
    basler.disconnect_camera(basler_cam)


if __name__ == "__main__":
    TODAY_PATH = Path("data").joinpath(datetime.now().strftime("%Y%m%d"))

    TWZR_WFM_FILE = TODAY_PATH.joinpath("array20_df=0.9MHz.npz")
    
    PROBE_DET_FILE = (
        # Path("C:/Users/Covey Lab/Documents/Andor Solis/atomic_data")
        Path("/home/coveylab/Documents/Data/atomic_data")
        .joinpath("20240506")
        .joinpath("probe-scan_007")
        .joinpath("probe-det.npy")
    )
    
    SAVE_FILE = TODAY_PATH.joinpath("array20_df=0.9MHz_init.npz")
    BACKUP_FILE = TODAY_PATH.joinpath("backup.npz")

    NT = 20 # number of peaks to find
    CROP_W = 45 # crop width
    CROP_H = 40 # crop height

    SHOT_PER_LOOP = 20 # number of shots to average over for each loop
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
    