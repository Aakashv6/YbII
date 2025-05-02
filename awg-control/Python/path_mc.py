import numpy as np

from lib.waveform import *
import matplotlib.pyplot as plt

mc_file_path = 'tables/16_MC-path.npz'

nt = 16
target = np.zeros(nt)
target[int(nt / 4):int(3 * nt / 4)] = 1
t_idx = np.nonzero(target)[0]
trial_repeat = 1000000
path_counter = {}
load_prob = [0.6, 0.9]
n_trials = trial_repeat * len(load_prob)
if mc_file_path is None:
    for i in load_prob:
        for j in range(trial_repeat):
            filled = np.random.binomial(1, i, nt)
            f_idx = np.nonzero(filled)[0]

            paths = get_rearrange_paths(f_idx, t_idx)
            for x,y in paths:
                if x == y: continue
                if (x,y) in path_counter:
                    path_counter[(x,y)] += 1
                else:
                    path_counter[(x,y)] = 0
    if True:
        np.savez(
            f"tables/{nt}_MC-path.npz",
            path_counter=path_counter,
            t_idx=t_idx,
            n_trials=trial_repeat*len(load_prob)
        )
else:
    mc_file = np.load(mc_file_path, allow_pickle=True)
    path_counter = mc_file['path_counter'].item()
    n_trials = mc_file['n_trials']

#-------------------------------------------------------------------------------------------

cutoffs = [0, 0.001, 0.005, 0.01, 0.05, 0.1]
success_rates = {}
n_samples = 20
loading_prob = np.linspace(0,1,n_samples)
for coff in cutoffs:
    path_table = {}
    useful_keys = [i for i,j in path_counter.items() if j >= coff*n_trials]
    for key in useful_keys:
        path_table[key] = 0

    n_repeat = 500
    n_move = np.zeros((n_samples, 2))
    missing = np.zeros((n_samples, 2))
    success = np.zeros((n_samples, 2))
    for i in range(n_samples):
        f_prob = loading_prob[i]
        nm = np.zeros(n_repeat)
        suc = np.zeros(n_repeat)
        for j in range(n_repeat):
            filled = np.random.binomial(1, f_prob, nt)
            f_idx = np.nonzero(filled)[0]

            paths = get_rearrange_paths(f_idx, t_idx)
            #         nm[j] = len(paths)

            for x, y in paths:
                if x != y and (x, y) in path_table:
                    filled[y] = 1
                    nm[j] += 1
            if np.array_equal(filled[t_idx], target[t_idx]):
                suc[j] = 1
            else:
                nm[j] = 0

        n_move[i, 0] = np.mean(nm)
        success[i, 0] = np.mean(suc)
        n_move[i, 1] = np.std(nm)
        success[i, 1] = np.std(suc)
    success_rates[(coff, len(path_table.keys()))] = success

x = loading_prob * 100
fig, ax1 = plt.subplots()

for p,d in success_rates:
    sucs = success_rates[(p,d)]
    # ax1.errorbar(x=x, y=sucs[:,0], yerr=sucs[:,1], fmt='.', label=f"{p*100}%, {d}")
    ax1.errorbar(x=x, y=sucs[:,0], fmt='.', label=f"{p*100}%, {d}")
ax1.set_ylabel("success probability")
ax1.set_ylim(0, 1.1)
ax1.set_xticks(np.arange(0,110,10))
ax1.set_xlim(0,100)
ax1.set_yticks(np.arange(0,1.2,0.1))
ax1.set_xlabel("loading probability (%)")
ax1.set_title(f"{nt} sites success rate")
ax1.grid()
ax1.legend()
# ax2.grid()
plt.savefig(f"fig/MC-success-rates_{nt}.jpeg", dpi=1200)
