from typing import Dict, Tuple, Any, List
import numpy as np
from scipy.interpolate import interp1d
# import cupy as cp
import scipy.optimize as spopt
import sys

class Waveform:
    def __init__(
        self, 
        f_start: int=0,
        df: int=0,
        nt: int=0,
        sample_rate: int=614.4e6,
        freq_res: int=1e3
    ):
        """
        helper class to store basic waveform information.
        :param f_start: starting frequency tone of the array.
        :param df: differential frequency between neighboring tweezers.
        :param nt: total number of tweezers to create from the starting frequency.
        :param sample_rate: sampling rate of the AWG to generate correct number of samples.
        :param freq_res: frequency resolution involved with this waveform
        """

        self.amplitude = 2 ** 10 * np.ones(nt)
        self.omega = 2 * np.pi * np.linspace(f_start, f_start + df * (nt-1), nt, endpoint=True)
        self.phi = 2 * np.pi * np.random.rand(nt)  # random initial phases from 0-2pi
        self.sample_rate: int = sample_rate
        self.freq_res = freq_res
        # formula for minimum sample length
        # sample_len_min = 512 * m
        # sample_len_min * freq_resolution / sampling_rate = k, k % 2 == 0
        self.sample_len_min = 2 * sample_rate / np.gcd(int(sample_rate), int(freq_res))
        # self.sample_len_min = 4 * sample_rate / np.gcd(int(sample_rate), int(df))
        # assert (self.sample_len_min * freq_res / self.sample_rate) % 2 == 0, "frequency resolution requirement not met"
        # assert self.sample_len_min % 512 == 0, "sample length not integer multiple of 512"

    def copy(self, other):
        self.omega = other.omega
        self.amplitude = other.amplitude
        self.phi = other.phi
        self.sample_rate = other.sample_rate
        self.freq_res = other.freq_res
        self.sample_len_min = other.sample_len_min
    
    def set_amplitudes(self, amps: np.ndarray) -> bool:
        if np.sum(amps) >= 2**15 - 1:
            print("warning: amplitudes too high")
            return False
        self.amplitude = amps[:len(self.amplitude)]
        return True


    def set_phase(self, phase: np.ndarray) -> bool:
        self.phi = phase[:len(self.phi)]
        return True

    def from_file(self, filename: str):
        data = np.load(filename)
        self.omega = data['omega']
        self.amplitude = data['amplitude']
        self.phi = data['phi']
        self.sample_rate = data['sample_rate']
        self.freq = data['freq_res']
        
    def save_wfm(self, filename:str):
        np.savez(
            filename,
            omega=self.omega,
            amplitude=self.amplitude,
            phi=self.phi,
            sample_rate=self.sample_rate,
            freq_res=self.freq_res,
        )
        
    def save_csv(self, filename: str):
        with open(filename, "w") as file:
            file.write(f"{self.sample_rate}\n")
            file.write(f"{int(self.freq_res)}\n")
        with open(filename, "a") as file:
            np.savetxt(file, np.round(self.omega / 2 / np.pi), fmt="%d", delimiter='', newline="")
            file.write("\n")
            np.savetxt(file, self.phi, fmt="%f", delimiter='', newline="")
            file.write("\n")
            np.savetxt(file, self.amplitude, fmt="%f", delimiter='', newline="")
            file.write("\n")


def create_static_array(
    wfm: Waveform,
    full: bool=False,
) -> np.ndarray:
    """
    create a static array signal from an initialized wfm object
    :param wfm: initialized Waveform object
    :param full: set True to return a full signal matrix, defaults to False
    :return: either a 1D or 2D np array
    """
    freq_res = wfm.freq_res
    if (wfm.sample_len_min * freq_res / wfm.sample_rate) % 2 != 0:
        wfm.sample_len_min = 4 * wfm.sample_rate / np.gcd(int(wfm.sample_rate), int(freq_res))
    assert (wfm.sample_len_min * freq_res / wfm.sample_rate) % 2 == 0, "frequency resolution requirement not met"
    assert wfm.sample_len_min % 512 == 0, "sample length not integer multiple of 512"
    
    # construct time axis, t_total(s) = sample_len / sample_rate, dt = t_total / sample_len
    t = np.arange(wfm.sample_len_min) / wfm.sample_rate

    # calculate individual sin waves, sig_mat[i] corresponds to data for ith tweezer
    # sin_mat = wfm.amplitude * np.sin(np.outer(wfm.omega,t) + np.expand_dims(wfm.phi, axis=1))  # shape=(number of tweezers x sample_len)
    sin_mat = np.sin(
        np.outer(wfm.omega, t) + np.expand_dims(wfm.phi, axis=1)
        # shape=(number of tweezers x sample_len)
    )
    sin_mat = (wfm.amplitude * sin_mat.T).T  # this works, trust me
    # sum up all rows to get final signal
    sig = np.sum(sin_mat, axis=0)
    if np.max(sig) >= 2 ** 15 - 1:
        print("Signal amp exceeds 2^15-1")
    if full:
        return sin_mat.astype(np.int16)
    return sig.astype(np.int16)

def stack_left(i_start, i_end, offset, stack_size=0):
    # calculate first index where the reduced path algorithm is applied
    #     threshold = 0.01
    #     cutoff = np.ceil(np.log(threshold) / np.log(1-load_p))
    #     cutoff = int(cutoff)
    #     print(cutoff)
    if stack_size == 0:
        stack_size = np.floor((i_end - i_start) / 2)
    stack_last = int(stack_size + i_start) - 1
    dist_mod = (i_end - i_start - stack_size) / (i_end - i_start)  # max_distance ratio
    dist_add = offset

    # get a list of moves to pre-generate
    moves = []
    max_dist = 0
    for i in range(i_start, i_end):
        moves.append([])
        j_max = i if i < stack_last else stack_last
        dist = np.ceil((i - i_start) * dist_mod + dist_add)
        j_min = int(i - dist) if i - dist >= i_start else i_start
        for j in range(j_min, j_max + 1):
            moves[i - i_start].append(j)  # add all paths between j_min and j_max
            if max_dist < abs(j-i):
                max_dist = abs(j-i)
    return moves, max_dist


def stack_right(i_start, i_end, offset, stack_size=0):
    moves, max_dist = stack_left(i_start, i_end, offset=offset, stack_size=stack_size)
    moves.reverse()
    for i in range(len(moves)):
        moves[i].reverse()
        for j in range(len(moves[i])):
            moves[i][j] = i_end - 1 - moves[i][j] + i_start
    return moves, max_dist


def create_path_table_gpu(
        wfm: Waveform, t_idx, pre_paths=None, save_path=None,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], np.ndarray]:
    """
    create a dim-3 look up table where the table[i,j] contains a sine wave to move tweezer i to tweezer j
    :param pre_paths: list of pre-defined paths to generate signals for
    :param save_path: file saving path
    :param t_idx: indices of tweezer positions in target pattern
    :param wfm: waveform object already initialized with basic parameters.
    :return: dictionary containing rearrange paths
    """
    import cupy as cp

    if pre_paths is None:
        pre_paths = []
        for i in range(len(wfm.omega)):
            for j in range(len(wfm.omega)):
                pre_paths.append((i,j))
        # pre_paths = [(i,j) for i in range(len(wfm.omega) for j in range(len(wfm.omega)))]
    
    # interpolate optimal amplitudes
    # data = np.load("data/optimal_amps.npz")
    w = wfm.omega
    a = wfm.amplitude
    omega_interp = interp1d(w, a, kind='cubic')

    dw_max = 0
    for i,j in pre_paths:
        dw = abs(wfm.omega[j] - wfm.omega[i])
        if dw_max < dw: dw_max = dw

    # setup basic variables
    twopi = 2 * np.pi
    vmax = twopi * 40e3 * 1e6  # convert units, 20 kHz/us -> 20e3 * 1e6 Hz/s
    t_max = 2 * dw_max / vmax  # Longest move sets the maximum moving time
    a_max = vmax * 2 / t_max  # maximum acceleration, negative sign because of magic
    # get number of samples required for longest move,this sets the size of lookup table
    sample_len = int(np.ceil(t_max * wfm.sample_rate))
    sample_len += wfm.sample_len_min - sample_len % wfm.sample_len_min
    sample_len = int(sample_len)
    a_max = vmax * 2 / (sample_len / wfm.sample_rate)
    # now we calculate all possible trajectories, go to Group Notes/Projects/Rearrangement for detail
    path_table = {}  # lookup table to store all moves
    static_sig = cp.zeros(sample_len)  # for fast real-time waveform generation purposes
    t = cp.arange(sample_len) / int(wfm.sample_rate)  # time series
    dt = t[1]
    t += dt
    nt = len(wfm.omega)

    diagonal_mat = cp.sin(
        cp.outer(cp.array(wfm.omega), t) + cp.expand_dims(cp.array(wfm.phi), axis=1)
        # shape=(number of tweezers x sample_len)
    )
    diagonal_mat = (cp.array(wfm.amplitude) * diagonal_mat.T).T  # this works, trust me
    # diagonal_mat = cp.array(diagonal_mat)
    static_sig = cp.sum(diagonal_mat[t_idx], axis=0)

    # iterate!
    time_counter = 0
    for i, j in pre_paths:
        time_counter += 1
        if time_counter % 100 == 0:
            print(time_counter)
        if i == j:
            continue

        omega_i = wfm.omega[i]
        omega_j = wfm.omega[j]
        path = cp.zeros(sample_len)

        # I advise reading through the notes page first before going further
        dw = omega_j - omega_i  # delta omega in the equation
        adw = abs(dw)
        t_tot = np.sqrt(abs(4 * dw / a_max))  # calculate minimum time to complete move
        end = int(np.round(t_tot * wfm.sample_rate))  # convert to an index in samples
        t_tot = end / wfm.sample_rate + t[1]

        '''
        phi_j = wfm.phi[j] % twopi  # wrap around two pi
        phi_i = wfm.phi[i] % twopi
        dphi = (phi_j - phi_i) % twopi  # delta phi in the equation
        if dphi < 0: dphi = abs(dphi) + twopi - phi_i  # warp around for negative phase shift
        t_tot += 12 * np.pi / adw - (
                (t_tot - 6 * dphi / adw) %
                (12 * np.pi / adw))  # extend move time to arrive at the correct phase
        '''

        a = 4 * (omega_i - omega_j) / (t_tot ** 2)  # adjust acceleration accordingly to ensure we still get to omega_j
        half = int(end / 2) + 1  # index of sample half-way through the move where equation changes
        t1 = t[:half]  # first half of the move, slicing to make life easier
        t2 = t[half:end] - t_tot / 2  # time series for second half of the move

        # interpolate amplitudes during the move
        amps = cp.zeros(sample_len)
        inst_w = cp.zeros(end)
        inst_w[0] = omega_i
        inst_w[-1] = omega_j
        inst_w[1:half] = omega_i - 0.5 * a * t1[1:] ** 2
        inst_w[half:end - 1] = omega_i - \
                               a / 2 * (t_tot / 2) ** 2 - \
                               a * t_tot / 2 * t2[:-1] + \
                               a / 2 * t2[:-1] ** 2
        sw = omega_i
        bw = omega_j
        if omega_i > omega_j:
            sw = omega_j
            bw = omega_i
        inst_w[inst_w < sw] = sw
        inst_w[inst_w > bw] = bw
        amps[:end] = cp.array(omega_interp(inst_w.get()))
        amps[end:] = wfm.amplitude[j]

        # calculate sine wave
        path[:half] = wfm.phi[i] + omega_i * t1 - a / 6 * t1 ** 3  # t<=T/2
        path[half:end] = path[half-1] + \
                         (omega_i - a / 2 * (t_tot / 2) ** 2) * t2 - \
                         a / 2 * t_tot / 2 * t2 ** 2 + \
                         a / 6 * t2 ** 3  # t>=T/2
        path[end:] = path[end-1] + omega_j * (t[end:] - t[end-1])
        path = amps * cp.sin(path)
        if i != j:
            path -= diagonal_mat[j]
        path = cp.asnumpy(path).astype(np.int16)
        path_table[(i, j)] = path

    static_sig = cp.asnumpy(static_sig).astype(np.int16)

    # save stuff if prompted
    if save_path is not None:
        np.savez(save_path, table=path_table, static_sig=static_sig, wfm=wfm, t_idx=t_idx)

    return path_table, static_sig


def get_rearrange_paths(
        f_idx: np.ndarray,
        t_idx: np.ndarray,
) -> np.ndarray:
    """
    Finds the minimum weight perfect matching between f_idx and t_idx
    :param f_idx: indices of tweezer positions filled with atoms.
    :param t_idx: indices of tweezer positions in target pattern.
    :returns: 2d numpy array containing moving path trajectories
    """
    if len(f_idx) < len(t_idx):
        return np.array([])
    cm = abs(np.subtract.outer(f_idx, t_idx))
    row, col = spopt.linear_sum_assignment(cm)
    return np.stack([f_idx[row], t_idx]).T

def create_moving_array(
        path_table: Dict,
        sig: np.ndarray,
        filled_idx: np.ndarray,
        target_idx: np.ndarray,
):
    """
    create a rearranging signal that moves tweezers as specified by paths.
    :param sig: initially a static-array-generating waveform, this function
    modifies sig directly
    :param path_table: lookup table returned from create_path_table_reduced().
    :param filled_idx: indices of tweezer positions filled with atoms.
    :param target_idx: indices of tweezer positions in target pattern.

    """
    paths = get_rearrange_paths(filled_idx, target_idx)
    # print(paths)
    if len(paths) == 0:
        return
    for i, j in paths:
        if i == j:
            # print("skip")
            continue  # skip stationary paths
        if (i, j) in path_table:
            # print("hey")
            sig += path_table[(i, j)]
    return


def create_moving_array_GPUOTF(
        path_table: Dict,
        sig: np.ndarray,
        filled_idx: np.ndarray,
        target_idx: np.ndarray,
):
    """
    same function as above, with running gpu arrays on the fly
    """
    paths = get_rearrange_paths(filled_idx, target_idx)
    if len(paths) == 0:
        return
    n_moves = len(paths)
    for k in range(n_moves):
        (i,j) = paths[k]
        if i == j:
            continue
        if (i, j) in path_table:
            sig += cp.array(path_table[(i, j)], dtype=cp.int16)
    return


def create_moving_signal_single(
        omega_i, omega_f, sample_rate, signal_time, amp=2**12, phi=0
):
    min_len = 2 * sample_rate / (1e3)
    sample_len = sample_rate * signal_time
    sample_len += min_len - sample_len % min_len
    sample_len = int(sample_len)

    t = np.arange(sample_len) / sample_rate
    t += t[1]
    t_tot = sample_len / sample_rate + t[1]
    a = 4 * (omega_i - omega_f) / (t_tot ** 2)
    end = sample_len
    half = int(end / 2) + 1
    t1 = t[:half]
    t2 = t[half:end] - t_tot / 2

    signal = np.zeros(sample_len)

    signal[:half] = phi + omega_i * t1 - a / 6 * t1 ** 3  # t<=T/2
    # ph = wfm.phi[i] + omega_i * t_tot / 2 + a / 6 * (t_tot / 2) ** 3
    signal[half:end] = signal[half - 1] + \
                     (omega_i - a / 2 * (t_tot / 2) ** 2) * t2 - \
                     a / 2 * t_tot / 2 * t2 ** 2 + \
                     a / 6 * t2 ** 3  # t>=T/2
    signal[end:] = signal[end - 1] + omega_f * (t[end:] - t[end - 1])
    phi_end = signal[-1]
    signal = amp * np.sin(signal)
    return signal.astype(np.int16), phi_end


def create_static_signal_single(
        omega, sample_rate, sample_len, amp=2 ** 12, phi=0
):
    # min_len = 2 * sample_rate / (1e3)
    # sample_len = sample_rate * signal_time
    # sample_len += min_len - sample_len % min_len
    # sample_len = int(sample_len)

    t = np.arange(sample_len) / sample_rate
    t += t[1]
    signal = phi + omega * t
    phi_end = signal[-1]
    signal = amp * np.sin(signal)
    return signal.astype(np.int16)


def tricky_trick(
    wfm: Waveform,
    site_index: List[int],
    df: float, # (Hz)
    tau_move: float=None, # (s)
    tau_stay: float=None, # (s)
    n: int=1,
) -> np.ndarray:
    if tau_move is None:
        tau_move = n / df - tau_stay
        while tau_move < 0:
            n += 1
            tau_move = n / df - tau_stay
    elif tau_stay is None:
        tau_stay = n / df - tau_move
        while tau_stay < 0:
            n += 1
            tau_stay = n / df - tau_move
    else:
        n = np.ceil((tau_move + tau_stay) * df)
        tau_stay = n / df - tau_move
    print(f"stay = {tau_stay*1e6:.1f} us")
    print(f"move = {tau_move*1e6:.1f} us")
    
    df *= np.pi * 2
    tau_total = 2 * tau_move + tau_stay
    if tau_total == 0:
        return create_static_array(wfm)
    sample_len = tau_total * wfm.sample_rate
    sample_len += wfm.sample_len_min - sample_len % wfm.sample_len_min
    sample_len = int(sample_len)
    t = np.arange(sample_len) / wfm.sample_rate
    accel = -4 * df / tau_move**2
    print(
          f"accleration: {accel / 2 / np.pi * 289e-9 / 65e3:.2f} m/s^2"
          f", {accel / 2 / np.pi * 289e-9 / 65e3 / 9.8067:.2f} g"
    )
    move_len = int(tau_move * wfm.sample_rate)
    move_len_half = int(move_len / 2) + 1
    idx_move0_half = move_len_half
    idx_stay_start = move_len
    idx_stay_end = move_len + int(tau_stay * wfm.sample_rate)
    idx_move1_half = idx_stay_end + move_len_half
    idx_move1_end = idx_stay_end + move_len
    t0 = t[:idx_move0_half]
    t1 = t[idx_move0_half:idx_stay_start] - t[idx_move0_half - 1]
    t2 = t[idx_stay_start:idx_stay_end] - t[idx_stay_start - 1]
    t3 = t[idx_stay_end:idx_move1_half] - t[idx_stay_end - 1]
    t4 = t[idx_move1_half:idx_move1_end] - t[idx_move1_half - 1]
    t5 = t[idx_move1_end:] - t[idx_move1_end - 1]
    # print(t0.shape)
    signal = np.zeros(sample_len)
    for k, omega_i in enumerate(wfm.omega):
        if k not in site_index:
            signal += wfm.amplitude[k] * np.sin(omega_i * t + wfm.phi[k])
            continue
        omega_f = omega_i + df
        sig = np.zeros(sample_len)
        
        sig[:idx_move0_half] = wfm.phi[k] \
            + omega_i * t0 \
            - accel / 6 * t0**3
            
        sig[idx_move0_half:idx_stay_start] = \
            sig[idx_move0_half - 1] \
            + (omega_i - accel / 2 * (tau_move / 2)**2) * t1 \
            - accel / 2 * tau_move / 2 * t1**2 \
            + accel / 6 * t1**3
        
        sig[idx_stay_start:idx_stay_end] = sig[idx_stay_start - 1] \
            + omega_f * t2

        sig[idx_stay_end:idx_move1_half] = sig[idx_stay_end - 1] \
            + omega_f * t3 + accel / 6 * t3**3

        sig[idx_move1_half:idx_move1_end] = sig[idx_move1_half - 1] \
            + (omega_f + accel / 2 * (tau_move / 2)**2) * t4 \
            + accel / 2 * tau_move / 2 * t4**2 \
            - accel / 6 * t4**3

        sig[idx_move1_end:] = sig[idx_move1_end-1] \
            + omega_i * t5

        signal += wfm.amplitude[k] * np.sin(sig)

    return signal.astype(np.int16)


def create_move_then_back(omega_i, omega_f, sample_rate, move_time, stay_time):
    move0, phi0 = create_moving_signal_single(
        omega_i, omega_f, sample_rate, move_time
    )
    stay, phi1 = create_static_signal_single(
        omega_f, sample_rate, stay_time, phi=phi0
    )
    move1, phi2 = create_moving_signal_single(
        omega_f, omega_i, sample_rate, move_time, phi=phi1
    )
    signal = np.concatenate((move0, stay, move1))
    return signal

def save_wfm(wfm:Waveform, filename:str, signal=None):
    np.savez(
        filename,
        omega=wfm.omega,
        amplitude=wfm.amplitude,
        phi=wfm.phi,
        sample_rate=wfm.sample_rate,
        freq_res=wfm.freq_res,
        signal=signal,
    )


