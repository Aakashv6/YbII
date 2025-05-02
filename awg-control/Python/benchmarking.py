import numpy as np

from lib.waveform import *
from cupyx.profiler import benchmark
import cupy as cp


def test(f_idx, t_idx):
    create_moving_array_reduced(_table_cp, _sig, f_idx, t_idx)
    return cp.asnumpy(_sig)


def count(paths, path_table, off):
    counter = 0
    for i, j in paths:
        if i == j:
            continue
        if (i, j) in path_table:
            counter += 1
        # else:
        #     print((i, j), "not in path table")
    for i in off:
        counter += 1
    return counter


data = np.load("table/table-half_31_120722.npz", allow_pickle=True)
_table = data['table'].item()
twzr = data['wfm'].item()
static_sig = data['static_sig']
target = data['target']
t_idx = np.nonzero(target)[0]
nt = twzr.omega.size

_table_cp = {}
# _sig = np.copy(static_sig)
for key in _table:
    _table_cp[key] = cp.array(_table[key])

n_repeat = 500
times = {
    'total_t': np.zeros((nt + 1, 2)),
    'gpu_t': np.zeros((nt + 1, 2)),
    'cpu_t': np.zeros((nt + 1, 2))
}
filling_ratio = np.zeros((nt+1, 2))
n_move = np.zeros((nt+1, 2))

for i in range(nt+1):
    # _sig = cp.array(static_sig)
    _sig = np.copy(static_sig)
    f_prob = i/nt
    t_t = np.zeros(n_repeat)
    g_t = np.zeros(n_repeat)
    c_t = np.zeros(n_repeat)
    ratio = np.zeros(n_repeat)
    nm = np.zeros(n_repeat)
    print(i, f_prob)
    for j in range(n_repeat):
        # filled = np.random.rand(nt)
        # tr = filled < f_prob
        # fa = filled >= f_prob
        # filled[tr] = 1
        # filled[fa] = 0
        filled = np.random.binomial(1, f_prob, nt)
        f_idx = np.nonzero(filled)[0]
        b = benchmark(
            test,
            (f_idx, t_idx),
            n_repeat=1
        )
        # stuff to save
        ratio[j] = f_idx.size / nt
        g_t[j] = b.gpu_times
        c_t[j] = b.cpu_times
        t_t[j] = b.gpu_times + b.cpu_times
        paths, off = get_rearrange_paths(f_idx, t_idx)
        nm[j] = count(paths, _table_cp, off)

    n_move[i,0] = np.mean(nm)
    n_move[i,1] = np.var(nm)
    times['gpu_t'][i,0] = np.mean(g_t)
    times['gpu_t'][i,1] = np.var(g_t)
    times['cpu_t'][i,0] = np.mean(c_t)
    times['cpu_t'][i,1] = np.var(c_t)
    times['total_t'][i,0] = np.mean(t_t)
    times['total_t'][i,1] = np.var(t_t)
    filling_ratio[i,0] = np.mean(ratio)
    filling_ratio[i,1] = np.var(ratio)


np.savez(
    f"data/120822_{nt}-half_wfm-and-transfer_CPU.npz",
    wfm=twzr,
    target=target,
    filling_ratio=filling_ratio,
    times=times,
    n_move=n_move
)

