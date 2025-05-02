import numpy as np

from lib.waveform import *
import os

array = Waveform(
    55e6,
    25e6,
    3,
    freq_res=10e3
)
# save_wfm(array, "data/array_70-3-20.npz")
save_wfm(array, "data/array21.npz")

# twzr.phi = phase
# sig = create_static_array(twzr, False)
# print(sig.shape)
# save_wfm(twzr, sig, 'data/array-5.npz')
# empty = np.zeros(sample_len)

# table generations
# tgt = np.zeros(nt)
# tgt[int(nt/4)+1:int(3*nt/4)] = 1
# t_idx = np.nonzero(tgt)[0]
# print(tgt)
# # savepath = 'table/table-half_31_120722'
# create_path_table_reduced_gpu(twzr, t_idx, dist_offset=np.inf, partition=True)

# moving waveform testings
# data = np.load("table/table-half_31_120722.npz", allow_pickle=True)
# table = data['table'].item()
# twzr = data['wfm'].item()
# static_sig = data['static_sig']
# target = data['target']
# t_idx = np.nonzero(target)[0]

# f_idx = np.array([5,8,20])
# create_moving_array_reduced(table, static_sig, f_idx, t_idx)
# np.savez('data/test.npz', signal=static_sig, wfm=twzr)

# data = np.load('data/table_5.npz', allow_pickle=True)
# table = data['table']
# sig = data['static_sig']
# paths, off = get_rearrange_paths(f_idx, t_idx)
# create_moving_array_reduced(sig, table, paths, off)

# path = 'data/move_trial'
# np.savez(path, signal=sig, wfm=twzr)

