# region imports
import datetime, imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go
import pandas as pd
from YbII.constants import const
from YbII.functions import func
from YbII.functions import gaussian2d
# from YbII.functions import pyplotdefs as ppd # DONT USE YET
# endregion

# region functions
def getMOTS0(p, mot_type):
    """
    s0 = 2 * (I_0 * T + I_0 * T ** 3) / I_sat
    Args:
        p (float): measured total power before all VPs
        mot_type (string): 'BLUE2D', 'BLUE3D', or 'GREEN3D' for the mot_type of MOT
    Returns:
        float: saturation parameter s0 for the MOT
    """
    if mot_type == 'BLUE2D':
        return p / (np.pi * const.w1_mot2d * const.w2_mot2d / 2) * (const.T_VP_mot2d + const.T_VP_mot2d ** 3) / const.Isat_1s0_1p1
    elif mot_type == 'BLUE3D':
        return p / (np.pi * const.w_mot3d_blue ** 2 / 2) * (const.T_VP_mot3d_blue + const.T_VP_mot3d_blue ** 3) / const.Isat_1s0_1p1
    elif mot_type == 'GREEN3D':
        return p / (np.pi * const.w_mot3d_green ** 2 / 2) * (const.T_VP_mot3d_green + const.T_VP_mot3d_green ** 3) / const.Isat_1s0_3p1
    else:
        raise ValueError("Invalid mot_type. Use 'BLUE2D', 'BLUE3D', or 'GREEN3D'.")

def rhoMOT(f, p, mot_type):
    """
    rho = s0 / 2 / (1 + s0 + (2 * (omega_l - omega_0) / gamma) ** 2)
    Args:
        f (float): frequency of the blue cooling beam. For the green beam, it is currently the FM detuning on the green AOM, centered around 96.9375 MHz
        p (float): measured total power before all entry VPs
    Returns: 
        float: probability of the atom being in the excited state (from Eva Casotti's thesis)
    """
    if mot_type == 'BLUE2D':
        s0 = getMOTS0(p, 'BLUE2D')
        omega_l = 2 * np.pi * f
        delta = omega_l - const.w0_1s0_1p1_32_06042025
        gamma = const.gamma_1s0_1p1
    elif mot_type == 'BLUE3D':
        s0 = getMOTS0(p, 'BLUE3D')
        omega_l = 2 * np.pi * (f + 10e6)
        delta = omega_l - const.w0_1s0_1p1_32_06042025
        gamma = const.gamma_1s0_1p1
    elif mot_type == 'GREEN3D':
        s0 = getMOTS0(p, 'GREEN3D')
        omega_l = (const.f_556_l + const.f_556_AOM_CMOT * 2 + f * 2) * 2 * np.pi
        delta = omega_l - const.w0_1s0_3p1_32
        gamma = const.gamma_1s0_3p1
    else:
        raise ValueError("Invalid mot_type. Use 'BLUE2D', 'BLUE3D', or 'GREEN3D'.")
    return s0 / 2 / (1 + s0 + (2 * delta / gamma) ** 2)

def getAtomNumber(img: np.ndarray, t_exp: float, f: float, p: float, wavelength: str, camera: str, mot_type: str) -> float:
    """
    Estimate the number of atoms in the imaged MOT based on the image data and camera parameters.
    n = gamma_tot / gamma_atom = I_sum * saturation_capacity / quantum_efficiency / bit_depth / collection_efficiency / t_exp / gamma / rho_ee

    Args:
        img (np.ndarray): background-subtracted ROI data (currently needs to be a 2D array; if we need to analyze colored images, we need to modify this script to sum the RGB channels)
        t_exp (float): exposure time of the imaging camera in seconds
        f (float): for the bMOT, it is wavemeter frequency reading of the 399; for the gMOT, it is the FM detuning on the green AOM, centered around 96.9375 MHz, all in Hz
        p (float): measured total power before all entry VPs, in W
        wavelength (str): 'BLUE' or 'GREEN', for the blue or green MOT, respectively
        camera (str): camera model, currently supports 'acA3800 14um', 'acA3800 14uc', and 'flir'
        mot_type (str): type of the MOT, currently supports 'BLUE2D', 'BLUE3D', and 'GREEN3D'

    Raises:
        ValueError: Invalid camera. Use 'acA3800 14um', 'acA3800 14uc', or 'flir'.
        ValueError: Invalid wavelength. Use 'BLUE' or 'GREEN'.
        ValueError: Invalid mot_type. Use 'BLUE2D', 'BLUE3D', or 'GREEN3D'.

    Returns:
        float: the estimated number of atoms in the imaged MOT
    """
    
    I_sum = np.sum(img)
    QE = 0
    sat_cap = 0
    bit_depth = 0
    d0 = const.d0_mot2d_end if mot_type == 'BLUE2D' else const.d0_mot3d # distance from the MOT to the first lens, in meters
    ################################################
    eff = (0.0254/2) ** 2 / (4 * d0 ** 2) # collection efficiency of the imaging setup, assuming using 1 inch lens
    if camera == 'acA3800 14um':
        if wavelength == 'BLUE':
            QE = const.acA3800_14um['BLUE']['QE'] # quantum efficiency of acA3800 for 399 nm light  
        elif wavelength == 'GREEN':
            QE = const.acA3800_14um['GREEN']['QE'] # quantum efficiency of acA3800 for 556 nm light
        sat_cap = const.acA3800_14um['sat_cap'] # saturation capacity of the camera
        bit_depth = 255 # bit depth of the camera
    elif camera == 'acA3800 14uc':
        raise ValueError('acA3800 14uc parameters not fully implemented')
    elif camera == 'flir':
        if wavelength == 'BLUE':
            QE = const.flir['BLUE']['QE'] # quantum efficiency of the flir camera for 399 nm light 
        elif wavelength == 'GREEN':
            QE = const.flir['GREEN']['QE'] # quantum efficiency of the flir camera for 556 nm light
        sat_cap = const.flir['sat_cap'] # saturation capacity of the camera
        bit_depth = 2 ** 12 - 1
    else:
        raise ValueError("Invalid camera. Use 'acA3800 14um', 'acA3800 14uc', or 'flir'.")
    ppi = sat_cap / QE / bit_depth # photon per pixel per intensity
    gamma_tot = I_sum * ppi / eff / t_exp # total photon emission rate
    if wavelength == 'BLUE':
        gamma_atom = const.gamma_1s0_1p1 * rhoMOT(f, p, mot_type) # photon emission rate of a single atom
    elif wavelength == 'GREEN': 
        gamma_atom = const.gamma_1s0_3p1 * rhoMOT(f, p, mot_type)
    else:
        raise ValueError("Invalid wavelength. Use 'BLUE' or 'GREEN'.")
    print('rhoMOT:', rhoMOT(f, p, mot_type))
    gamma_tot = I_sum * ppi / eff / t_exp 
    return gamma_tot / gamma_atom

def getImagedAtomNumber(img: np.ndarray, img_bg: np.ndarray, t_exp: float, freq: float, p: float, wavelength: str, fit_param: dict[str, float | None], camera: str, mot_type: str, background: bool) -> tuple:
    """
    get the number of imaged MOT and fit the MOT size from the image data. Note that this code needs to be modified for colored images. 
    Args:
        img (np.ndarray): MOT image data, usually loaded via imageio.imread()
        img_bg (np.ndarray): MOT background data
        t_exp (float): exposure time of the imaging camera in seconds
        freq (float): for the bMOT, it is wavemeter frequency reading of the 399; for the gMOT, it is the FM detuning on the green AOM, centered around 96.9375 MHz, all in Hz
        p (float): measured total power before all entry VPs, in W
        d0 (float): distance from the MOT to the first lens, in meters. As of 06/02/25, it is 0.395 m for the 2DMOT & 0.2 m for both 3DMOTs.
        background (bool): choose whether to subtract the background image from the MOT image data or not
        wavelength (str): 'BLUE' or 'GREEN', for the blue or green MOT, respectively
        fit_param (dict[str, float | None]): a dictionary with fit parameter names as its keys and initial values as its values. Can enter all or just a few parameters. The parameters are {'x0', 'y0', 'box_x', 'box_y'}.
        camera (str): .  ''.
        mot_type (str): .  ''.

    Returns:
        _type_: _description_
    """
    if background:
        img_res = np.abs(np.array(img, dtype=float) - np.array(img_bg, dtype=float))
        img_res[img_res < 0] = 0
    else:
        img_res = img
    img_res_og = img_res.copy() # keep the original image for plotting

    bin_size = 10 # binning 10 pixels to 1 for faster fitting
    if mot_type == 'BLUE2D':
        mag = const.f0_mot2d / const.f1_mot2d # magnification of the 2D MOT imaging setup
        box_x = 75 if fit_param == None or 'box_x' not in fit_param.keys() or fit_param['box_x'] is None else fit_param['box_x']
        box_y = 75 if fit_param == None or 'box_y' not in fit_param.keys() or fit_param['box_y'] is None else fit_param['box_y']
    else: # 'BLUE3D' 'GREEN3D'
        mag = const.f0_mot3d / const.f1_mot3d
        box_x = 100 if fit_param == None or 'box_x' not in fit_param.keys() or fit_param['box_x'] is None else fit_param['box_x']
        box_y = 100 if fit_param == None or 'box_y' not in fit_param.keys() or fit_param['box_y'] is None else fit_param['box_y']
    if camera == 'acA3800 14um':
        pixel_size = (const.acA3800_14um['pixel_size'] / mag) ** 2
    elif camera == 'flir':
        pixel_size = (const.flir['pixel_size'] / mag) ** 2
    x0 = None
    y0 = None
    # if fit_param is not None, try to extract x0 and y0 from it
    if fit_param != None:
        x0 = fit_param['x0'] if fit_param['x0'] is not None else None
        y0 = fit_param['y0'] if fit_param['y0'] is not None else None
    # if x0 and y0 are provided, find the MOT ROI
    if x0 is not None and y0 is not None:
        img_res = img_res[y0-box_y:y0+box_y, x0-box_x:x0+box_x]
    # otherwise, find the center of the MOT from the fit
    else:
        fit_params, x_bin, y_bin, d_bin = gaussian2d.lmfit_gaussian(img_res, pixel_size, bin_size, {'x0': None, 'y0': None})
        x0 = int(fit_params['x0'].value / np.sqrt(pixel_size))
        y0 = int(fit_params['y0'].value / np.sqrt(pixel_size))
        img_res = img_res[y0-box_y:y0+box_y, x0-box_x:x0+box_x]
        x0 = box_x
        y0 = box_y

    fit_params, x_bin, y_bin, d_bin = gaussian2d.lmfit_gaussian(img_res, pixel_size, bin_size, {'x0': x0, 'y0': y0})

    atom_num = getAtomNumber(img_res, t_exp, freq, p, wavelength, camera, mot_type)

    print('Atom number:', atom_num)

    return img_res_og, img_res, fit_params, bin_size, pixel_size, box_x, box_y, x_bin, y_bin, d_bin, atom_num

def plotMOTNumber(img, img_bg, t_exp, freq, p, wavelength, fit_param, camera, mot_type, filename, background=True, fit_interact=False, show_fit=True):
    """
    _summary_

    Args:
        img (_type_): _description_
        img_bg (_type_): _description_
        t_exp (_type_): _description_
        freq (_type_): _description_
        p (_type_): _description_
        d0 (_type_): _description_
        background (_type_): _description_
        wavelength (_type_): _description_
        fit_param (_type_): _description_
        camera (str): _description_.
        mot_type (str): _description_.
        filename (str): _description_.

    Returns:
        _type_: _description_
    """
    img_res_og, img_res, fit_params, bin_size, pixel_size, box_x, box_y, x_bin, y_bin, d_bin, atom_num = getImagedAtomNumber(img, img_bg, t_exp, freq, p, wavelength, fit_param, camera=camera, mot_type=mot_type, background=background)

    # rotate the og by 180 degrees
    # img_res_og = np.rot90(img_res_og, 2)

    A = fit_params['A'].value
    B = fit_params['B'].value
    sx = fit_params['sx'].value
    sy = fit_params['sy'].value
    x0 = fit_params['x0'].value
    y0 = fit_params['y0'].value
    th = fit_params['theta'].value
    wx = sx * np.sqrt(2)
    wy = sy * np.sqrt(2)
    u = np.linspace(0, img_res.shape[1], img_res.shape[1] // bin_size) * np.sqrt(pixel_size)
    v = np.linspace(0, img_res.shape[0], img_res.shape[0] // bin_size) * np.sqrt(pixel_size)
    x = np.outer(u, np.ones(np.size(v)))
    y = np.outer(np.ones(np.size(u)), v)
    z = func.gaussian2d(x, y, th, A, B, x0, y0, sx, sy)

    fig, axs = plt.subplots(1, 2, figsize=(5, 5), dpi=300)
    axs[0].imshow(img_res_og, vmin=0, vmax=100)
    axs[0].set_xlim([0, img_res_og.shape[1]])
    axs[0].set_ylim([0, img_res_og.shape[0]])
    axs[0].set_xlabel('Pixel')
    axs[0].set_ylabel('Pixel')

    x1, x2, y1, y2 = x0-box_x, x0+box_x, y0-box_y, y0+box_y

    axins = axs[0].inset_axes([0.6, 0.6, 0.3, 0.3], xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.imshow(img_res, vmin=0, vmax=100, origin='lower', extent=(x1, x2, y1, y2))
    axs[0].indicate_inset_zoom(axins, edgecolor="black")
    # turn off the grid for inset
    axins.grid(False)
    axins.set_xticks([])
    axins.set_yticks([])

    scatter = go.Scatter3d(x=x_bin, y=y_bin, z=d_bin, mode='markers', marker=dict(size=2))
    if show_fit:
        opacity = 0.8
    else:
        opacity = 0.0
    surface = go.Surface(x=x, y=y, z=z, colorscale='Viridis', opacity=opacity, showscale=False)
    fit_fig = go.Figure(data=[scatter, surface], layout=go.Layout(scene=dict(aspectmode='cube')))
    fit_fig.update_layout(
        width=900, height=900,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            xaxis=dict(range=[0, img_res.shape[1] * np.sqrt(pixel_size)] ),
            yaxis=dict(range=[0, img_res.shape[0] * np.sqrt(pixel_size)] ),
            camera=dict(
                eye=dict(x=1, y=1, z=1.75)  # Adjust the camera position for better view
            )
        )
    )
    fit_params.pretty_print()
    if fit_interact:
        fit_fig.show()
    buf = BytesIO()
    fit_fig.write_image(buf, format="png")
    buf.seek(0)
    fit_fig = np.array(Image.open(buf))
    axs[1].axis('off')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].imshow(fit_fig, aspect='equal', extent=[0, 1, 0, 1], origin='upper')
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)

    fit_str = f" (x0={x0:.0f} px, y0={y0:.0f} px, wx={wx:.1f} mm, wy={wy:.1f} mm, A={A:.1f}, B={B:.1f}, theta={th:.2f} rad)\n" if show_fit else ""
    fig.suptitle(mot_type + f'#Atom ~ {atom_num:.1e}\n' + fit_str + f'ROI size: ({box_x * 2:.0f} px, {box_y * 2:.0f} px)=({box_x * 2 * np.sqrt(pixel_size):.1f}mm, {box_y * 2 * np.sqrt(pixel_size):.2f}mm)\n' + 'File name: ' + filename, fontsize=5)
    fig.tight_layout()
    plt.close()
    return img_res, atom_num, fig


# dir: str, directory of the images
# ext: str, file extension of the images
def var_extract(dir, ext, keywords=[["exp", 1e-3]], save_params=True):
    # Get a list of file names in the directory
    file_names = os.listdir(dir)
    # Filter to include only files, not directories
    # file_names = [file for file in file_names if os.path.isfile(os.path.join(dir, file))]
    # Further filter to find files ending with '_bg'
    # bg_files = [file for file in file_names if file.endswith("_bg.bmp")]
    # filter to include only files ending with '.bmp'
    sig_ext = ext
    bg_ext = "_bg" + ext
    file_names = [file for file in file_names if file.endswith(sig_ext) and not file.endswith(bg_ext)]
    exp_params = []
    for f in file_names:
        # f_data = f[:-7] + ".bmp"
        f_data = f
        vars_file = []
        split_vars = f_data.split("_")
        vars_file.append(f_data)
        for k in keywords:
            try:
                s = split_vars.index(k[0]) + 1
                e = s + 2
                val = float(".".join(split_vars[s : e])) * k[1]
                vars_file.append(val)
            except:
                print("Parameter not in the file name")
                continue
        ctime = os.path.getctime(dir + f)
        # Convert to a datetime object
        ctime_datetime = datetime.datetime.fromtimestamp(ctime)
        # Convert to a string in the desired format
        ctime_str = ctime_datetime.strftime('%Y-%m-%d %H:%M:%S')
        # Append to vars_file
        vars_file.append(ctime_str)
        exp_params.append(vars_file)
    exp_params = np.array(exp_params)
    if save_params:
        # insert a "filename" column at the beginning of keywords
        keywords.insert(0, ["filename", 1])
        np.savetxt(dir + "data.csv", exp_params, delimiter=",", fmt='%s', header=",".join([k[0] for k in keywords] + ["time"]))
# var_extract("C:/Users/YbII Cobra/Desktop/3D MOT/0510/VertArm QWPsweep/", keywords=[["exp", 1e-3], ["vcoil", 1], ["pavg", 1e-3], ["d", 1e-3], ["verta", 1]])

def fit_data(dir, save_dir, t_exp, freq, df, p, d0, box_x, box_y, background, wavelength, camera, ext, global_background=None, fit_override=None, param_init=None, constraints=None, plot_save=True, save_data=False):
    exp_params = np.loadtxt(dir + "data.csv", dtype=str, delimiter=",")
    headers = pd.read_csv(dir + "data.csv").columns.tolist()
    # # Get a list of file names in the directory
    # file_names = os.listdir(dir)
    # # Filter to include only files, not directories
    # file_names = [file for file in file_names if os.path.isfile(os.path.join(dir, file))]
    # # Further filter to find files ending with '_bg'
    # bg_files = [file for file in file_names if file.endswith("_bg.bmp")]
    atom_nums = []
    atom_nums_bgs = []
    data_list = []

    for f in range(exp_params.shape[0]):
        t_exp_hlp = t_exp
        f_hlp = freq
        p_hlp = p
        df_hlp = df
        if 't' in headers:
            t_exp_hlp = float(exp_params[f, headers.index('t')])
        elif t_exp == -1:
            raise ValueError('t_exp must be provided in the data file or as an argument')
        if 'f' in headers:
            f_hlp = float(exp_params[f, headers.index('f')])
        elif f == -1:
            raise ValueError('f must be provided in the data file or as an argument')
        if 'df' in headers:
            df_hlp += float(exp_params[f, headers.index('df')])
        elif df == -1:
            raise ValueError('df must be provided in the data file or as an argument')
        if 'p' in headers:
            p_hlp = float(exp_params[f, headers.index('p')])
        elif 'bPtot' in headers:
            p_hlp = float(exp_params[f, headers.index('bPtot')])
        f_data = exp_params[f, 0]
        f_bg = f_data[:-4] + '_bg' + ext

        img_data = imageio.imread(dir + f_data)

        if background: 
            if global_background == None:
                img_bg = imageio.imread(dir + f_bg)
            else:
                img_bg = imageio.imread(dir + global_background)
        else:
            img_bg = None

        img_res, x0, y0, wx, wy, bgx, bgy, x_data, x_fit, y_data, y_fit, atom_num, atom_num_bg = plotMOTNumber(img_data, img_bg, f_data, t_exp_hlp, f_hlp, p_hlp, d0, box_x, box_y, background, wavelength, fit_override, param_init, constraints, plot_save, save_dir=save_dir, camera=camera)

        atom_nums.append(atom_num)
        atom_nums_bgs.append(atom_num_bg)
        data_list.append(f_data)

    atom_nums = np.array([atom_nums])
    atom_nums_bgs = np.array([atom_nums_bgs])
    data_list = np.array([data_list])

    exp_params = np.concatenate((exp_params, atom_nums.T), axis=1)
    exp_params = np.concatenate((exp_params, atom_nums_bgs.T), axis=1)
    exp_params = np.concatenate((exp_params, data_list.T), axis=1)
    exp_params = np.savetxt(dir + "exp_params.csv", exp_params, delimiter=",", fmt='%s')
# endregion