from PIL import Image
import numpy as np
import lib.peaks as peaks
import cv2
import lib.plotting.pyplotdefs as pd
from matplotlib.pyplot import Rectangle, Circle
import lib.basler as basler
import lib.AWG as AWG
import lib.waveform as waveform
from pypylon import pylon
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sys import exit

def init_basler(exposure: int, gain: int) -> pylon.InstantCamera:
    camera = basler.connect_camera()
    # camera.PixelFormat.SetValue("Mono12")
    camera.ExposureTime.SetValue(exposure)
    camera.Gain.SetValue(gain)
    camera.AcquisitionMode.SetValue("Continuous")
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return camera

def init_awg(
    wfm_file0: str,
    wfm_file1: str,
)-> tuple[AWG.AWG, tuple[waveform.Waveform, waveform.Waveform]]:
    
    array_H = waveform.Waveform()
    array_V = waveform.Waveform()
    array_H.from_file(wfm_file0)
    array_V.from_file(wfm_file1)
    sig_H = waveform.create_static_array(array_H)
    sig_V = waveform.create_static_array(array_V)
    sig = np.empty((sig_H.shape[0] + sig_V.shape[0]), dtype=sig_H.dtype)
    sig[0::2] = sig_H
    sig[1::2] = sig_V
    
    condition = [AWG.SPCSEQ_ENDLOOPONTRIG, AWG.SPCSEQ_ENDLOOPALWAYS]
    awg = AWG.AWG()
    awg.open(id=0)
    awg.set_sampling_rate(int(614.4e6))
    awg.set_trigger(EXT0=AWG.SPC_TM_POS)
    awg.set_sequence_mode(2)
    awg.write_segment(sig, segment=0)
    # awg.write_segment(sig, segment=1)
    awg.configure_step(step=0, segment=0, nextstep=0, loop=1, condition=condition[0])
    # awg.configure_step(step=1, segment=1, nextstep=0, loop=1, condition=condition[0])
    awg.toggle_channel([0, 3], amplitude=1000)
    awg.run()
    awg.force_trigger()
    return awg, (array_H, array_V)

def reload_awg(
    awg: AWG.AWG,
    wfms: tuple[waveform.Waveform, waveform.Waveform]
):
    sig_H = waveform.create_static_array(wfms[0])
    sig_V = waveform.create_static_array(wfms[1])
    sig = np.empty((sig_H.shape[0] + sig_V.shape[0]), dtype=sig_H.dtype)
    sig[0::2] = sig_H
    sig[1::2] = sig_V
    awg.stop()
    awg.reset()
    awg.set_sampling_rate(int(614.4e6))
    awg.set_trigger(EXT0=AWG.SPC_TM_POS)
    awg.set_sequence_mode(2)
    awg.write_segment(sig, segment=0)
    # awg.write_segment(sig, segment=1)
    awg.configure_step(step=0, segment=0, nextstep=0, loop=1, condition=AWG.SPCSEQ_ENDLOOPONTRIG)
    # awg.configure_step(step=1, segment=1, nextstep=0, loop=1, condition=condition[0])
    awg.toggle_channel([0, 3], amplitude=1000)
    awg.run()
    awg.force_trigger()


def find_peaks(
    img: np.ndarray,
    array_dim: tuple[int, int],
) -> list:
    peak_finder = peaks.Cv2HoughCircles(
        gaussian_blur_params={"ksize": (3,3),"sigmaX": 0.0,"sigmaY": 0.0},
        dp=1.5,
        method=cv2.HOUGH_GRADIENT,
        minDist=200,
        #maxDist=200,
        param1=10,
        param2=8,
        minRadius=15,
        maxRadius=30,
    )
    rot = IMAGE_ROTATION_ANGLE * np.pi / 180
    peak_locs, proc = peak_finder.find_peaks(img)
    peak_locs = [
        (y*np.sin(rot) + x*np.cos(rot), y*np.cos(rot) - x*np.sin(rot))
        for (x,y) in peak_locs
    ]
    peak_locs.sort(key=lambda xy: -xy[1])
    for i in range(array_dim[0]):
        peak_locs[i*array_dim[1]:(i+1)*array_dim[1]] = \
            sorted(peak_locs[i*array_dim[1]:(i+1)*array_dim[1]], key=lambda xy: -xy[0])
    peak_locs = [
        (int(y*np.sin(-rot) + x*np.cos(-rot)), int(y*np.cos(-rot) - x*np.sin(-rot)))
        for (x,y) in peak_locs
    ]
    return peak_locs

def label_peaks(
    img: np.ndarray,
    peak_locs: list,
    crop_dim: tuple[int, int],
    save_path: str,
) -> None:
    P = pd.Plotter().imshow(img).colorbar().grid(False)
    ax = P.ax
    for i, (x,y) in enumerate(peak_locs):
        ax.add_patch(
            # Circle(
            #     (x,y), 25,
            #     edgecolor='red',
            #     facecolor='none',
            #     lw=0.05
            # )
            Rectangle(
                (x-crop_dim[0]//2, y-crop_dim[1]//2),
                crop_dim[0], crop_dim[1],
                edgecolor='red',
                facecolor='none',
                lw=0.1
            )
        )
        ax.annotate(
            i,
            (x-30,y-30),
            fontsize=2,
            color='g',
            antialiased=True,
        )
    P.savefig(save_path, dpi=1200)
    P.close()
    
def grab_image(
    camera: pylon.InstantCamera,
    repeat: int,
) -> np.ndarray:
    imgs = []
    for i in range(repeat):
        res = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        imgs.append(res.Array)
    return np.array(imgs)

def grab_power(
    camera: pylon.InstantCamera,
    peak_locs: np.ndarray,
    repeat: int,
    crop_dim: tuple[int, int],
    array_dim: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    w = crop_dim[0]
    h = crop_dim[1]
    power_arr = []
    for i in range(repeat):
        res = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        img = np.array(res.Array)
        # powers = [
        #     img[j0 - h // 2 : j0 + h // 2, i0 - w // 2 : i0 + w // 2].sum()
        #     for i0, j0 in peak_locs
        # ]
        powers = [
            img[j0 - h // 2 : j0 + h // 2, i0 - w // 2 : i0 + w // 2].max()
            for i0, j0 in peak_locs
        ]
        power_arr.append(powers)
    return [
        np.mean(power_arr, axis=0).reshape(array_dim),
        np.std(power_arr, axis=0).reshape(array_dim)
    ]

def get_weighted_mask(
    target_powers: np.ndarray,
    init_powers: np.ndarray=None,
) -> np.ndarray:
    mask = target_powers / np.mean(target_powers)
    if init_powers is not None:
        mask *= init_powers.mean() / init_powers
    return mask

def main():
    basler_cam = init_basler(exposure=BASLER_EXPOSURE, gain=0.0)
    awg, (array_H, array_V) = init_awg(WFM_FILE0, WFM_FILE1)
    time.sleep(1.0)
    
    img = grab_image(basler_cam, 1)[0]
    if AUTO_FIND_PEAKS:
        peak_locs = find_peaks(img, TARGET_POWERS.shape)
        np.save(Path(DATA_SAVE_DIR).joinpath("peak_locs.npy"), peak_locs)
    else:
        peak_locs = np.load(MANUAL_PEAK_FILE)
        
    label_peaks(img, peak_locs, CROP_DIM, Path(FIG_SAVE_DIR).joinpath("initial_crop.png"))
    if len(peak_locs) != TARGET_POWERS.flatten().shape[0]:
        print(f"found {np.array(peak_locs).shape} peaks, want {TARGET_POWERS.shape}")
        awg.stop()
        awg.close()
        basler.disconnect_camera(basler_cam)
        exit(0)

    awg.stop()
    time.sleep(0.2)
    bnd_powers = grab_power(
        basler_cam,
        peak_locs,
        repeat=SHOT_PER_LOOP,
        crop_dim=CROP_DIM,
        array_dim=TARGET_POWERS.shape
    )[0]
    
    awg.run()
    awg.force_trigger()
    time.sleep(1.0)

    init_powers = grab_power(
        basler_cam,
        peak_locs,
        repeat=SHOT_PER_LOOP,
        crop_dim=CROP_DIM,
        array_dim=TARGET_POWERS.shape,
    )
    init_powers[0] -= bnd_powers
    
    (pd.Plotter()
        .imshow(bnd_powers[:, ::-1], cmap='gray')
        .set_xticklabels([])
        .set_yticklabels([])
        .colorbar()
        .grid(False)
        .savefig(Path(FIG_SAVE_DIR).joinpath("bnd_power.png"))
        .close()
    )
    (pd.Plotter()
        .imshow(
            init_powers[0][:, ::-1],
            cmap="gray",
        )
        .set_xticklabels([])
        .set_yticklabels([])
        .colorbar()
        .grid(False)
        .savefig(Path(FIG_SAVE_DIR).joinpath("init_power.png"))
        .close()
    )
    
    weighted_mask = get_weighted_mask(TARGET_POWERS)
    
    column_errs = []
    row_errs = []
    total_errs = []
    for i in range(MAX_LOOP):
        powers = grab_power(
            basler_cam,
            peak_locs,
            repeat=SHOT_PER_LOOP,
            crop_dim=CROP_DIM,
            array_dim=TARGET_POWERS.shape,
        )
        powers[0] -= bnd_powers
        p_err = (weighted_mask*powers[0] - np.median(weighted_mask*powers[0])) \
            / np.median(weighted_mask*powers[0])
        col_err = p_err.mean(axis=0)
        row_err = p_err.mean(axis=1)
        column_errs.append(abs(col_err).mean())
        row_errs.append(abs(row_err).mean())
        total_errs.append(np.mean(abs(p_err)))
        print(
            f"\r loop {i:2}/{MAX_LOOP-1}, "
            f"col err: {column_errs[-1]:.4f}, "
            f"row err: {row_errs[-1]:.4f}, "
            f"tot err: {total_errs[-1]:.4f}",
            end=""
        )
        # if column_errs[-1] <= ERROR_TOLERANCE:
        #     break
        if np.mean(abs(p_err)) <= ERROR_TOLERANCE or i == MAX_LOOP - 1:
            np.save("data/test_amp0.npy", array_H.amplitude)
            break
        array_H.amplitude = array_H.amplitude - STEP_SIZE * col_err
        array_H.amplitude[array_H.amplitude < 100] = 100
        array_V.amplitude = array_V.amplitude - STEP_SIZE * row_err
        array_V.amplitude[array_V.amplitude < 100] = 100
        reload_awg(awg, (array_H, array_V))
        time.sleep(0.15)
    print()
        
    final_powers = grab_power(
        basler_cam,
        peak_locs,
        repeat=SHOT_PER_LOOP,
        crop_dim=CROP_DIM,
        array_dim=TARGET_POWERS.shape,
    )
    final_img = grab_image(
        basler_cam,
        10
    )
    final_img = np.mean(final_img, axis=0)
    np.save("data/pictures/final20x20.npy", final_img)
    array_H.save_wfm(WFM_FILE0_SAVE)
    array_V.save_wfm(WFM_FILE1_SAVE)
    np.save("data/test_amp1.npy", array_H.amplitude)
    awg.stop()
    awg.close()
    basler.disconnect_camera(basler_cam)
    
    (pd.Plotter()
        .plot(column_errs, marker='.', linestyle="-", color="C0", label="col")
        .plot(row_errs, marker='.', linestyle="-", color="C1", label="row")
        .plot(total_errs, marker=".", linestyle="-", color="C2", label="tot")
        .set_xlabel("iterations")
        .set_ylabel("avg error")
        .legend()
        .savefig(Path(FIG_SAVE_DIR).joinpath("error_trace.png"))
        .close()
    )
    (pd.Plotter()
        .imshow(
            final_img,
            cmap="jet",
        )
        .set_xticklabels([])
        .set_yticklabels([])
        .colorbar()
        .grid(False)
        .savefig(Path(FIG_SAVE_DIR).joinpath("final_img.png"))
        .close()
    )
    (pd.Plotter()
        .imshow(
            final_powers[0][:, ::-1],
            cmap="gray",
        )
        .set_xticklabels([])
        .set_yticklabels([])
        .colorbar()
        .grid(False)
        .savefig(Path(FIG_SAVE_DIR).joinpath("final_power.png"))
        .close()
    )
    # (pd.Plotter()
    #     .imshow(
    #         (final_powers[0] - init_powers[0])[:, ::-1],
    #         cmap="gray",
    #         # origin="lower",
    #         # extent=[final_powers.shape[1], 0, final_powers.shape[0], 0]
    #     )
    #     .set_xticklabels([])
    #     .set_yticklabels([])
    #     .colorbar()
    #     .grid(False)
    #     .savefig(Path(FIG_SAVE_DIR).joinpath("power_diff.png"))
    #     .close()
    # )
    
def debug_peak_finder():
    # img = np.array(plt.imread("data/pictures/maxish_20x20.tiff"))
    img = np.array(plt.imread("figs/initial_20241023.tiff"))
    sorted_peaks = find_peaks(img, ARRAY_DIM) # circle detection and sorting
    # manual adjustment
    sorted_peaks = np.array(sorted_peaks, dtype=int)
    print(sorted_peaks.shape)
    sorted_peaks[0, 0] += 20
    sorted_peaks[0, 1] += 3
    sorted_peaks[1, 1] -= 18
    sorted_peaks[2, 0] -= 10
    sorted_peaks[3, 0] += 2
    sorted_peaks[3, 1] += 10
    sorted_peaks[4, 0] -= 4
    sorted_peaks[4, 1] += 6
    sorted_peaks[5, 0] += 4
    sorted_peaks[5, 1] -= 10
    sorted_peaks[6, 0] -= 13
    sorted_peaks[6, 1] -= 8
    # sorted_peaks[7, 0] += 5
    sorted_peaks[7, 1] += 10
    sorted_peaks[8, 0] -= 2
    sorted_peaks[8, 1] += 5
    sorted_peaks[9, 1] -= 10
    sorted_peaks[10, 0] += 2
    sorted_peaks[11, 0] -= 4
    sorted_peaks[11, 1] -= 10
    sorted_peaks[12, 0] += 6
    sorted_peaks[12, 1] -= 8
    sorted_peaks[13, 0] -= 15
    sorted_peaks[14, 0] -= 27
    sorted_peaks[14, 1] -= 35
    sorted_peaks[15, 0] -= 2
    sorted_peaks[15, 1] -= 11
    sorted_peaks[16, 0] += 7
    sorted_peaks[16, 1] -= 17
    sorted_peaks[17, 0] -= 2
    sorted_peaks[18, 0] -= 10
    sorted_peaks[18, 1] += 5
    sorted_peaks[19,0] -= 21
    sorted_peaks[19,1] -= 30
    sorted_peaks[20,0] += 14
    sorted_peaks[20,1] -= 50
    sorted_peaks[21,0] -= 5
    sorted_peaks[21,1] -= 20
    sorted_peaks[22,0] -= 10
    sorted_peaks[22,1] -= 20
    sorted_peaks[23,0] -= 10
    sorted_peaks[23,1] -= 10
    sorted_peaks[24,0] -= 40
    sorted_peaks[24,1] -= 40

    # sorted_peaks[23,0] += 10
    # sorted_peaks[23,1] += 10
    
    # sorted_peaks[26,0] -= 5
    # sorted_peaks[29,0] += 30
    # sorted_peaks[29,1] -= 10
    # sorted_peaks[33,0] += 25
    # sorted_peaks[33,1] -= 50
    # sorted_peaks[34,0] += 30
    # sorted_peaks[34,1] += 30
    # sorted_peaks[37,0] -= 10
    # sorted_peaks[37,1] += 30
    # sorted_peaks[38,1] -= 10
    
    # sorted_peaks[39,0] += 10
    # sorted_peaks[40,0] += 5
    # sorted_peaks[40,1] -= 20
    # sorted_peaks[41,0] -= 10
    # sorted_peaks[41,1] += 40
    # sorted_peaks[42,1] += 30
    # sorted_peaks[45,0] -= 10
    # sorted_peaks[45,1] += 50
    # sorted_peaks[46,0] += 40
    # sorted_peaks[46,1] += 20
    
    # sorted_peaks[52,0] += 10
    # sorted_peaks[53,0] += 10
    # sorted_peaks[54,0] -= 10
    # sorted_peaks[57,1] += 10
    # sorted_peaks[59,0] -= 20
    # sorted_peaks[59,1] += 75
    # sorted_peaks[60,0] -= 10
    # sorted_peaks[60,1] -= 20
    # sorted_peaks[63,0] += 15
    # sorted_peaks[64,0] += 5
    # sorted_peaks[64,1] += 10
    
    # sorted_peaks[66,1] += 25
    # sorted_peaks[70,0] += 10
    # sorted_peaks[70,1] += 20
    # sorted_peaks[72,0] -= 20
    # sorted_peaks[72,1] += 60
    # sorted_peaks[76,1] += 5
    # sorted_peaks[77,0] += 3
    
    # sorted_peaks[78,0] += 5
    # sorted_peaks[78,1] += 5
    # sorted_peaks[79,1] -= 5
    # sorted_peaks[85,1] += 5
    
    # sorted_peaks[94,1] += 10
    # sorted_peaks[97,1] += 10
    # sorted_peaks[101,1] += 10
    
    # sorted_peaks[104,0] += 5
    # sorted_peaks[113,1] -= 5
    
    # sorted_peaks[117,1] += 10
    # sorted_peaks[121,1] += 15
    # sorted_peaks[122,0] += 5
    # sorted_peaks[122,1] += 30
    # sorted_peaks[123,0] += 90
    # sorted_peaks[123,1] -= 40
    # sorted_peaks[124,0] += 60
    # sorted_peaks[124,1] -= 30
    # sorted_peaks[125,0] += 15
    # sorted_peaks[125,1] += 25
    # sorted_peaks[126,1] += 25
    # sorted_peaks[127,1] += 30
    
    # sorted_peaks[130,0] += 5
    # sorted_peaks[130,1] += 5
    # sorted_peaks[136,0] -= 20
    # sorted_peaks[136,1] -= 130
    # sorted_peaks[138,0] -= 5
    # sorted_peaks[138,1] += 10
    
    # sorted_peaks[147,1] += 10
    # sorted_peaks[150,1] += 50
    
    # sorted_peaks[157,0] += 10
    # sorted_peaks[159,1] += 5
    # sorted_peaks[160,0] -= 5
    # sorted_peaks[164,0] -= 15
    # sorted_peaks[165,0] += 15
    # sorted_peaks[166,0] -= 10
    # sorted_peaks[166,1] += 3
    # sorted_peaks[167,1] += 3
    # --------------------- 
    # label_peaks(img, sorted_peaks, (5, 5), "figs/labeled_peaks_5x5_new.png")
    label_peaks(img, sorted_peaks, (20,20), "figs/debug.png")
    np.save(MANUAL_PEAK_FILE, sorted_peaks)


if __name__ == "__main__":
    # WFM_FILE0 = "data/array5_H_11x11friend_optm.npz"
    # WFM_FILE1 = "data/array5_V_11x11friend_optm.npz"
    WFM_FILE0 = "data/array5_H_NOT5x5guy.npz"
    WFM_FILE1 = "data/array5_V_NOT5x5guy.npz"
    # WFM_FILE0 = "data/array5_H_5x5friend_optm6.npz"
    # WFM_FILE1 = "data/array5_V_5x5friend_optm6.npz"
    FIG_SAVE_DIR = "figs"
    DATA_SAVE_DIR = "data"
    # WFM_FILE0_SAVE = DATA_SAVE_DIR + Path(WFM_FILE0).stem + "_1" + Path(WFM_FILE0).suffix
    # WFM_FILE1_SAVE = DATA_SAVE_DIR + Path(WFM_FILE1).stem + "_1" + Path(WFM_FILE1).suffix
    # WFM_FILE0_SAVE = "data/array5_H_11x11friend_optm2.npz"
    # WFM_FILE1_SAVE = "data/array5_V_11x11friend_optm2.npz"
    WFM_FILE0_SAVE = "data/array5_H_5x5friend_optm6.npz"
    WFM_FILE1_SAVE = "data/array5_V_5x5friend_optm6.npz"

    AUTO_FIND_PEAKS = True
    IMAGE_ROTATION_ANGLE = 0 # rotates image counter-clockwise
    MANUAL_PEAK_FILE = "data/peaks_friend.npy"
    ARRAY_DIM = (5,5)
    TARGET_POWERS = np.ones(ARRAY_DIM)
    
    # BASLER_EXPOSURE = 10990.0
    BASLER_EXPOSURE = 9030.0
    # CROP_DIM = (60, 60)
    CROP_DIM = (50, 50)
    SHOT_PER_LOOP = 30 # number of shots to average over for each loop
    MAX_LOOP = 3
    STEP_SIZE = 10 # tune this for convergence
    ERROR_TOLERANCE = 0.03 # error threshold
    
    main()
    # debug_peak_finder()
    # img = np.array(plt.imread("data/pictures/main_optm.bmp"))
    # peak_locs = np.load("data/peaks_full.npy")
    # debug(
    #     img,
    #     peak_locs,
    #     array_dim=(10,10)
    # )