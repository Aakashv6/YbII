import numpy as np

# Physical constants in SI units
c = 299792458
h = 6.62607015 * 10**(-34)
h_bar = h / (2 * np.pi)
kB = 1.380649 * 10**(-23)

# Material constants
rho_cu = 1.7 * 10**(-8)

# 171Yb constants
gamma_1s0_1p1 = 2 * np.pi * 29.1 * 10**6
gamma_1s0_3p1 = 2 * np.pi * 0.183 * 10**6
Isat_1s0_1p1 = 59.97 * 10**(-3) * 10**(4)
Isat_1s0_3p1 = 0.139 * 10**(-3) * 10**(4)

f_399_l = 751.526978e12 # 399 nm laser frequency
f_556_l = 539.390067e12 # 556 nm laser frequency
f_556_AOM_CMOT = 96.9375e6 # 556 nm AOM frequency for CMOT

w0_1s0_1p1_32 = 2 * np.pi * 751.527103 * 10**12 # wavemeter reading
w0_1s0_1p1_32_2025 = 2 * np.pi * (751.527103 * 10**12 - 100e6) # wavemeter reading
w0_1s0_1p1_32_june2025 = 2 * np.pi * (751.527103 * 10**12 - 100e6 + 50e6)# wavemeter reading
w0_1s0_1p1_32_06042025 = w0_1s0_1p1_32_2025 + 2 * np.pi * 25e6 # 399 resonance estimate after the June HVAC failure
w0_1s0_1p1_32_06052025 = w0_1s0_1p1_32_06042025 - 2 * np.pi * 6e6 # 399 resonance estimate after the June HVAC failure
w0_1s0_1p1_32_06122025 = w0_1s0_1p1_32_06052025 - 2 * np.pi * 15e6 # 399 resonance estimate after the June HVAC failure



# 05/28/25 Estimated from YbI CMOT AOM frequency, assuming gamma/2 red detuning
w0_1s0_3p1_32 = 2 * np.pi * (f_556_l + f_556_AOM_CMOT * 2) + gamma_1s0_3p1 / 2

omega_aom_399_2DMOT = 2 * np.pi * 150 * 10**6
omega_aom_399_3DMOT = 2 * np.pi * 160 * 10**6

# Optical parameters
w1_mot2d = 3.29e-3 # minor waist of the 2D MOT beam
w2_mot2d = 8.66e-3 # major waist of the 2D MOT beam
T_VP_mot2d = 0.915 # power transmission of the 2D MOT viewports
d0_mot2d_end = 395e-3 # distance from the 2D MOT to the 1st lens 
f0_mot2d = 400e-3 # focal length of the 1st lens in the 2D MOT setup
f1_mot2d = 100e-3 # focal length of the 2nd lens in the 2D MOT setup

w_mot3d_blue = 3e-3 # waist of the blue 3D MOT beam
w_mot3d_green = 3e-3 # waist of the red 3D MOT beam
T_VP_mot3d_blue = 0.915 # blue power transmission of the 3D MOT viewports
T_VP_mot3d_green = 0.915 # blue power transmission of the 3D MOT viewports
d0_mot3d = 200e-3
f0_mot3d = 200e-3 # focal length of the 1st lens in the 3D MOT setup
f1_mot3d = 50e-3 # focal length of the 2nd lens in the 3D MOT setup

acA3800_14um = {
    'pixel_size': 1.67e-6, # pixel size in meters
    'sat_cap': 2800,
    'BLUE': {
        'QE': 0.42, 
    }, 
    'GREEN': {
        'QE': 0.37, 
    }
}

acA3800_14uc = {
    'pixel_size': 1.67e-6, # pixel size in meters
    'sat_cap': 2800,
    'BLUE': {
        'QE': 0.025, # Not verified yet
    }, 
    'GREEN': {
        'QE': 0.37, # Not verified yet
    }
}

acA1440_220um = {
    'pixel_size': 3.45e-6, # pixel size in meters
    'sat_cap': 10531,
    'BLUE': {
        'QE': 0.4, 
    }, 
    'GREEN': {
        'QE': 0.6, 
    }
}

flir = {
    'pixel_size': np.nan, 
    'sat_cap': 10482,
    'BLUE': {
        'QE': 0.45, 
    }, 
    'GREEN': {
        'QE': 0.7, 
    }
}