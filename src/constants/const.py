import numpy as np

# Physical constants in SI units
c = 299792458
h = 6.62607015 * 10**(-34)
h_bar = h / (2 * np.pi)
kB = 1.380649 * 10**(-23)

# Material constants
rho_cu = 1.7 * 10**(-8)

# 171Yb constants
w0_1s0_1p1_32 = 2 * np.pi * 751.527103 * 10**12 # wavemeter reading
gamma_1s0_1p1 = 2 * np.pi * 29.1 * 10**6
gamma_1S0_3P1 = 2 * np.pi * 0.183 * 10**6
Isat_1s0_1p1 = 59.97 * 10**(-3) * 10**(4)
Isat_1S0_3P1 = 0.139 * 10**(-3) * 10**(4)
omega_aom_399_2DMOT = 2 * np.pi * 150 * 10**6
omega_aom_399_3DMOT = 2 * np.pi * 160 * 10**6