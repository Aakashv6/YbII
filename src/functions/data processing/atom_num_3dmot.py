# region imports

import sys, os
sys.path.append('../YBII')

import numpy as np
import matplotlib.pyplot as plt

# Fitting Module
import lmfit
from lmfit.models import gaussian2d

# Image Processing Module
import imageio
import matplotlib.pyplot as plt
import pandas as pd

from src.functions.math import func as mathf
from src.functions.optics import func as optf
from src.constants import const

import datetime

# endregion

# region functions
# p is the measured power per beam before any VP, w1 and w2 are the beam waist radii
# s0 = 2 * (I_0 * T + I_0 * T ** 3) / I_sat
def get3DMOTS0(p, w=3e-3):
    return p / (np.pi * w ** 2 / 2) * (0.955 + 0.955 ** 3) / const.Isat_1s0_1p1

# probability of the atom being in the excited state (from Eva Casotti's thesis)
# f is frequency of the cooling beam
# p is the measured power per beam before any VP
def rho3DMOT(f, p):
    # w_atom = const.w0_1s0_1p1_32
    # w_l = f + 
    return get3DMOTS0(p) / 2 / (1 + get3DMOTS0(p) + (2 * (2 * np.pi * f + 2 * np.pi * 10e6 - const.w0_1s0_1p1_32) / const.gamma_1s0_1p1) ** 2)

# sum up the pixel values in the region of interest
# x0, y0, wx, wy are the parameters of the Gaussian fit
# img_res is the background-subtracted image data
# f is frequency of the cooling beam
# p is the measured power per beam before any VP
# d0 is the image distance from the MOT to the first lens
def getAtomNumber(x0, y0, box_x, box_y, bgx, bgy, img, t_exp, f, p, d0, wavelength, camera='acA3800 14um'):
    
    x0 = int(x0)
    y0 = int(y0)
    wx = int(box_x)
    wy = int(box_y)
    I_sum = np.sum(img[y0-box_y:y0+box_y, x0-box_x:x0+box_x])

    area = (2 * box_x) * (2 * box_y)
    noise_bg = ((bgx + bgy)/2) * area

    I_sum_bg = I_sum - noise_bg
    QE = 0
    cap_sat = 0

    QE399_14um = 0.42 # quantum efficiency of acA3800 for 399 nm light
    QE556_14um = 0.37 # quantum efficiency of acA3800 for 556 nm light
    
    QE399_14um = 0.37 # quantum efficiency of the 14uc camera mono mode at 399 nm
    QE399_14uc = 0.17 # quantum efficiency of the 14uc camera RGB mode at 399 nm
    cap_sat_acA3800 = 2800 # saturation capacity of the camera

    #### Flir grasshopper 3 camera
    QE399_flir = 0.45 
    QE556_flir = 0.7 # quantum efficiency of the 14uc camera mono mode at 556 nm
    cap_sat_flir = 10482 # saturation capacity of the camera
    ################################################
    eff = (0.0254/2) ** 2 / (4 * d0 ** 2) # collection efficiency of the imaging setup, assuming using 1 inch lens
    if camera == 'acA3800 14um':
        if wavelength == 'blue':
            QE = QE399_14um  
        elif wavelength == 'green':
            QE = QE556_14um
        cap_sat = cap_sat_acA3800
    elif camera == 'acA3800 14uc':
        raise ValueError('acA3800 14uc parameters not fully implemented')
        # if wavelength == 'blue':
        #     QE = QE399_14uc 
        # elif wavelength == 'green':
            # QE = QE556_14uc
        # cap_sat = cap_sat_acA3800
    elif camera == 'flir':
        if wavelength == 'blue':
            QE = QE399_flir 
        elif wavelength == 'green':
            QE = QE556_flir
        cap_sat = cap_sat_flir
    ppi = cap_sat / QE / 255 # photon per pixel per intensity
    gamma_tot = I_sum * ppi / eff / t_exp # total photon emission rate
    gamma_atom = const.gamma_1s0_1p1 * rho3DMOT(f, p) # photon emission rate of a single atom

    if wavelength == 'green': gamma_atom = 1

    gamma_tot_bg = I_sum_bg * ppi / eff / t_exp 
    return gamma_tot / gamma_atom, gamma_tot_bg / gamma_atom


# get the number of imaged atoms
# img is the MOT image data, img_bg is the background image data
# t_exp is the exposure time of the camera
# f is frequency of the cooling beam
# p is the measured power per beam before any VP
# param_init is an 1D array of the form [Xa, X\sigma, X0, Xc_c, Ya, Y\sigma, Y0, Yc_c]
# constraints is a 7x2 2D array of the form [Xa[min, max], X\sigma[min, max], X0[min, max], ...]
# fit_override is an 1D array for overriding the results of the fitted parameters
def getImagedAtomNumber(img, img_bg, t_exp, f, p, d0, box_x, box_y, background, wavelength, fit_override=None, param_init=None, constraints=None, camera='acA3800 14um'):
    x_rg = np.arange(0, img.shape[1])
    y_rg = np.arange(0, img.shape[0])

    if background:
        img_res = np.abs(np.array(img, dtype=float) - np.array(img_bg, dtype=float))
        img_res[img_res < 0] = 0
    else:
        img_res = img

    x_data = np.max(img_res, axis=0)
    y_data = np.max(img_res, axis=1)

    # also in /src/functions/math/func.py
    def GaussianWBaseline(x, center, sigma, amplitude, c_c):
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + c_c

    if wavelength == 'blue':
        model = lmfit.Model(GaussianWBaseline)

        # adjust the initial parameters if needed
        if param_init!=None:
            print(
                'hi'
            )
            params_x = model.make_params(c_c=param_init[3], center=param_init[2], sigma=param_init[1], amplitude=param_init[0])
            params_y = model.make_params(c_c=param_init[7], center=param_init[6], sigma=param_init[5], amplitude=param_init[4])
        else: 
            params_x = model.make_params(c_c=0,
            center=np.argmax(x_data), sigma=50, amplitude=np.max(x_data))
            params_y = model.make_params(c_c=0, center=np.argmax(y_data), sigma=50, amplitude=np.max(y_data))

        # Set constraints if needed
        if constraints!=None:
            params_x['amplitude'].set(min=constraints[0, 0])
            params_x['amplitude'].set(max=constraints[0, 1])

            params_x['sigma'].set(min=constraints[1, 0])
            params_x['sigma'].set(max=constraints[1, 0])

            params_x['center'].set(min=constraints[2, 0])
            params_x['center'].set(max=constraints[2, 1])

            params_x['c_c'].set(min=constraints[3, 0])
            params_x['c_c'].set(max=constraints[3, 1])

            params_y['amplitude'].set(min=constraints[4, 0])
            params_y['amplitude'].set(max=constraints[4, 1])

            params_y['sigma'].set(min=constraints[5, 0])
            params_y['sigma'].set(max=constraints[5, 1])

            params_y['center'].set(min=constraints[6, 0])
            params_y['center'].set(max=constraints[6, 1])

            params_y['c_c'].set(min=constraints[7, 0])
            params_y['c_c'].set(max=constraints[7, 1])

        result_x = model.fit(x_data, params_x, x=x_rg)
        result_y = model.fit(y_data, params_y, x=y_rg)

        x_fit = result_x.best_fit
        y_fit = result_y.best_fit

        x0 = result_x.best_values['center']
        y0 = result_y.best_values['center']

        wx = 2 * result_x.best_values['sigma']
        wy = 2 * result_y.best_values['sigma']

        bgx = result_x.best_values['c_c']
        bgy = result_y.best_values['c_c']

    elif wavelength == 'green':
        bgx, bgy, x_fit, y_fit, x0, y0, wx, wy = 0, 0, 0, 0, 0, 0, 0, 0

    ##### TODO change fit_override to a dict
    if fit_override!=None:
        x0 = fit_override[0] if fit_override[0]!=None else x0
        y0 = fit_override[1] if fit_override[1]!=None else y0
        # wx = fit_override[2] if fit_override[2]!=None else wx 
        # wy = fit_override[3] if fit_override[3]!=None else wy

    atom_num, atom_num_bg = getAtomNumber(x0, y0, box_x, box_y, bgx, bgy, img_res, t_exp, f, p, d0, wavelength, camera)

    return img_res, x0, y0, wx, wy, bgx, bgy, x_data, x_fit, y_data, y_fit, atom_num, atom_num_bg

# plot the image data along with the Gaussian fit and the atom number
# img is the MOT image data, img_bg is the background image data
# t_exp is the exposure time of the camera
# df is the detuning from the Yb171 1S0 -> 1P1 transition in angular frequency units
# p is the measured power per beam before any VP
def plotMOTNumber(img, img_bg, f_data, t_exp, df, p, d0, box_x, box_y, background, wavelength, fit_override=None, param_init=None, constraints=None, plot_save=True, save_dir="./results/3DMOT_results/0709/test/", camera='acA3800 14um'):
    
    # print(f'df = {df}')

    img_res, x0, y0, wx, wy, bgx, bgy, x_data, x_fit, y_data, y_fit, atom_num, atom_num_bg = getImagedAtomNumber(img, img_bg, t_exp, df, p, d0, box_x, box_y, background, wavelength, fit_override, param_init, constraints, camera=camera)

    if True:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        x_rg = np.arange(0, img.shape[1])
        y_rg = np.arange(0, img.shape[0])

        axs[0].imshow(img_res, vmin=0, vmax=100)

        axs[0].set_xlim([0, img.shape[1]])
        axs[0].set_ylim([0, img.shape[0]])
        axs[0].set_xlabel('Pixel')
        axs[0].set_ylabel('Pixel')
        # axs[0].set_title(f_data)

        axs[0].hlines(y0-box_y, x0-box_x, x0+box_x, color='b', linestyles='dashed')
        axs[0].hlines(y0+box_y, x0-box_x, x0+box_x, color='b', linestyles='dashed')
        axs[0].vlines(x0-box_x, y0-box_y, y0+box_y, color='r', linestyles='dashed')
        axs[0].vlines(x0+box_x, y0-box_y, y0+box_y, color='r', linestyles='dashed')

        # print(len(x_rg), len(x_data))
        axs[1].scatter(x_rg, x_data)
        axs[1].scatter(y_rg, y_data)

        if wavelength == 'blue':
            axs[1].plot(x_rg, x_fit, 'r', label='x Fit')
            axs[1].plot(y_rg, y_fit, 'b', label='y Fit')

        axs[1].vlines(x0-box_x, 0, 100, color='r', linestyles='dashed', label='x bounds')
        axs[1].vlines(x0+box_x, 0, 100, color='r', linestyles='dashed')
        axs[1].vlines(y0-box_y, 0, 100, color='b', linestyles='dashed', label='y bounds')
        axs[1].vlines(y0+box_y, 0, 100, color='b', linestyles='dashed')
        
        axs[1].set_xlabel('Pixel')
        axs[1].set_ylabel('Intensity')
        axs[1].legend()
        if wavelength == 'blue':
            fig.suptitle('3D MOT #Atom ~ {:.1e}; x0:'.format(round(atom_num, -2)) + str(round(x0, 2)) + ' wx:' + str(round(wx, 2)) + ' y0:' + str(round(y0, 2)) + ' wy:' + str(round(wy, 2)) + ' waist_y:' + str(round(4 * wy * 1.67 * 10**(-3), 2)) + 'mm')
        elif wavelength == 'green':
            fig.suptitle(r'(Green CPS) = {:.1e}'.format(atom_num) +  '\n t_exp='+str(t_exp) + ' s')
    if plot_save == True:
        plt.savefig(save_dir + f_data[:-4] + '.png')
    plt.close()
    return img_res, x0, y0, wx, wy, bgx, bgy, x_data, x_fit, y_data, y_fit, atom_num, atom_num_bg

def var_extract(dir, keywords=[["exp", 1e-3]], save_params=True):

    # Get a list of file names in the directory
    file_names = os.listdir(dir)

    # Filter to include only files, not directories
    # file_names = [file for file in file_names if os.path.isfile(os.path.join(dir, file))]
    # Further filter to find files ending with '_bg'
    # bg_files = [file for file in file_names if file.endswith("_bg.bmp")]
    # filter to include only files ending with '.bmp'
    file_names = [file for file in file_names if file.endswith(".bmp") and not file.endswith("_bg.bmp")]

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

        
        # print(vars_file)
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
        np.savetxt(data_folder + "data.csv", exp_params, delimiter=",", fmt='%s', header=",".join([k[0] for k in keywords] + ["time"]))

# var_extract("C:/Users/YbII Cobra/Desktop/3D MOT/0510/VertArm QWPsweep/", keywords=[["exp", 1e-3], ["vcoil", 1], ["pavg", 1e-3], ["d", 1e-3], ["verta", 1]])

def fit_data(dir, save_dir, t_exp, df, p, d0, box_x, box_y, background, wavelength, camera, global_background=None, fit_override=None, param_init=None, constraints=None, plot_save=True, save_data=False):
    exp_params = np.loadtxt(data_folder + "data.csv", dtype=str, delimiter=",")
    headers = pd.read_csv(data_folder + "data.csv").columns.tolist()
    # # Get a list of file names in the directory
    # file_names = os.listdir(dir)

    # # Filter to include only files, not directories
    # file_names = [file for file in file_names if os.path.isfile(os.path.join(dir, file))]
    # # Further filter to find files ending with '_bg'
    # bg_files = [file for file in file_names if file.endswith("_bg.bmp")]

    atom_nums = []
    atom_nums_bgs = []
    f_data_list = []


    for f in range(exp_params.shape[0]):
        t_exp_hlp = t_exp
        df_hlp = df
        p_hlp = p
        if 't' in headers:
            t_exp_hlp = float(exp_params[f, headers.index('t')])
        if 'f' in headers:
            df_hlp = float(exp_params[f, headers.index('f')])
            if 'df' in headers:
                df_hlp += float(exp_params[f, headers.index('df')])
        if 'p' in headers:
            p_hlp = float(exp_params[f, headers.index('p')])
        
        f_data = exp_params[f, 0]
        f_bg = f_data[:-4] + "_bg.bmp"

        img_data = imageio.imread(dir + f_data)

        if background: 
            if global_background == None:
                img_bg = imageio.imread(dir + f_bg)
            else:
                img_bg = imageio.imread(dir + global_background)
        else:
            img_bg = None

        img_res, x0, y0, wx, wy, bgx, bgy, x_data, x_fit, y_data, y_fit, atom_num, atom_num_bg = plotMOTNumber(img_data, img_bg, f_data, t_exp_hlp, df_hlp, p_hlp, d0, box_x, box_y, background, wavelength, fit_override, param_init, constraints, plot_save, save_dir=save_dir, camera=camera)

        atom_nums.append(atom_num)
        atom_nums_bgs.append(atom_num_bg)
        f_data_list.append(f_data)

    atom_nums = np.array([atom_nums])
    atom_nums_bgs = np.array([atom_nums_bgs])
    f_data_list = np.array([f_data_list])

    exp_params = np.concatenate((exp_params, atom_nums.T), axis=1)
    exp_params = np.concatenate((exp_params, atom_nums_bgs.T), axis=1)
    exp_params = np.concatenate((exp_params, f_data_list.T), axis=1)
    exp_params = np.savetxt(data_folder + "exp_params.csv", exp_params, delimiter=",", fmt='%s')
# endregion

#####################CHANGE THE BELOW EACH TIME YOU RUN###########

data_folder = r"C:\Users\YbII Cobra\Desktop\3D MOT\0731\green\flir\Vay/"
# save_dir = "./results/3DMOT_results/0718/T vs P/"
save_dir = "./results/3DMOT_results/0731/green_VRR/Vay/"

 
var_extract(data_folder, keywords=[["t", 1e-6], ['theta', 1]])

t_exp = 500015e-6  #this does not matter
freqL = 751.5270397e12
d0 = 195e-3

p = 59.3e-3

# region image processing parameters
box_xy_blue_flir = [242,242]
box_xy_blue_basler = [250,250]
# box_xy_green_flir = [121,121]
box_xy_green_flir = [150,150]

box_xy_green_basler = [250,250]
# override [x0, y0]

center_blue_basler=[1650, 1750]
center_green_basler=[1650, 1750]
center_blue_flir=[925, 1075]
center_green_flir = [1050,1050]
# endregion

# global_background = r'tau_1_0_ms_t-exp_79995_0_us.bmp'
# global_background = r't_50015_0_us_bg.bmp'
global_background = None

bool_bg = 1

camera_options = ['acA3800 14um', 'acA3800 14uc', 'flir']
wavelengths = ['blue', 'green']

'''
CHOOSE CAMERA AND WAVELENGTH HERE
'''
camera = camera_options[2]
wavelength = wavelengths[1]
''''''
# fit_override = [1070,1075]
if camera == 'acA3800 14um' or camera == 'acA3800 14uc': 
    if wavelength == 'blue':
        fit_override = center_blue_basler
        box_xy = box_xy_blue_basler
    elif wavelength == 'green':
        fit_override = center_green_basler
        box_xy = box_xy_green_basler
elif camera == 'flir': 
    if wavelength == 'blue':
        fit_override = center_blue_flir
        box_xy = box_xy_blue_flir
    elif wavelength == 'green':
        fit_override = center_green_flir
        box_xy = box_xy_green_flir

plot_save = True

fit_data(
    dir = data_folder, save_dir = save_dir, 
    t_exp = t_exp, df = freqL, p = p, d0 = d0, 
    box_x = box_xy[0], box_y = box_xy[1], 
    fit_override = fit_override,
    background = bool_bg, 
    wavelength = wavelength, 
    camera = camera, 
    global_background = global_background, plot_save= plot_save
    )

# def fit_data(dir, save_dir, t_exp, df, p, d0, box_x, box_y, background, wavelength, camera, global_background=None, fit_override=None, param_init=None, constraints=None, plot_save=True, save_data=False):