from lib.AWG import *
import code
# import readline
# import rlcompleter
import os
# import argparse
from lib.waveform import *
import sys
from datetime import datetime, timedelta
from pathlib import Path
#from memory_profiler import profile
import time

try:
    profile
except NameError:
    no_profile = lambda x: x
    profile = no_profile

def get_step():
    step = int32(0)
    spcm_dwGetParam_i32(awg.card, SPC_SEQMODE_STATUS, byref(step))
    return step.value

def trigger():
    global awg
    # step = get_step()
    awg.force_trigger()
    print("triggering...")
    # print("current step:", step)

def stop():
    global awg
    awg.stop()
    print("stopping awg output...")

def start():
    global awg
    if not awg.is_connected():
        sys.exit(0)
    awg.run()
    awg.force_trigger()
    print("starting awg output...")

def close():
    global awg
    awg.stop()
    awg.reset()
    awg.close()
    print("closing awg and quitting...")
    quit()

np.random.seed(0)
sampling_rate = int(614.4e6)

# array_H = Waveform(
#     60e6,
#     2e6,
#     40,
#     freq_res=10e3,
# )
# print(array_H.sample_len_min)
# array_V = Waveform()
# array_V.copy(array_H)

# array_H = Waveform(
#     60e6,
#     10e6,
#     5,
#     freq_res=10e3)
# array_V = Waveform(
#     60e6,
#     10e6,
#     5,
#     freq_res=10e3)
# array_H.save_wfm("data/array5_H_NOT5x5guy.npz")
# array_V.save_wfm("data/array5_V_NOT5x5guy.npz")

start_freq = 60e6 #unit in Hz
step_size = 10e6 #unit in Hz

array_H = Waveform(
    40e6,
    10e6,
    10,
    freq_res=10e3)
array_V = Waveform(
    40e6,
    10e6,
    10,
    freq_res=10e3)

print(array_V.phi)
array_H.save_wfm("data/array5_H_NOT5x5guy.npz")
array_V.save_wfm("data/array5_V_NOT5x5guy.npz")

# array_H = Waveform()
# array_V = Waveform()
# # # array_H.from_file("data/array5_H_smolboi_optm6.npz")
# # # array_V.from_file("data/array5_V_smolboi_optm6.npz")
# array_H.from_file("data/array5_H_5x5friend_optm6.npz")
# array_V.from_file("data/array5_V_5x5friend_optm6.npz")


# array_H.from_file("data/main.npz")
# array_V.from_file("data/main_half.npz")
#array_H.from_file("data/array20/array20_H_mod1.npz")
#array_V.from_file("data/array20/array20_H_mod1.npz")
sig_H = create_static_array(array_H, full=False)
sig_V = create_static_array(array_V, full=False)

print(array_H.omega / 2 / np.pi / 1e6)
'''
these two signals are written onto segment 0 and 1 of the AWG memory
make sure you interleave your data correctly
'''
sig_segment0 = np.empty(sig_H.shape[0] + sig_V.shape[0], dtype=np.int16)
sig_segment0[0::2] = sig_H
sig_segment0[1::2] = sig_V

sig_segment1 = np.empty(sig_H.shape[0] + sig_V.shape[0], dtype=np.int16)
sig_segment1[0::2] = sig_H
sig_segment1[1::2] = sig_V

# sig_segment0 = sig_H
# sig_segment1 = sig_segment0


'''
Setting AWG data replay settings
SPCSEQ_ENDLOOPONTRIG for infinite loop until trigger
SPCSEQ_ENDLOOPALWAYS for finite loop, then goto nextstep at trigger.
'''
condition = [SPCSEQ_ENDLOOPONTRIG, SPCSEQ_ENDLOOPALWAYS]
awg = AWG()
awg.open(id=0)  # switch between AWG cards if there are multiple
awg.set_sampling_rate(sampling_rate)
awg.set_trigger(EXT0=SPC_TM_POS) # set trigger on positive edge of EXT0
awg.set_sequence_mode(2)  # partition AWG memory into 2 segments
awg.write_segment(sig_segment1, segment=0) # write the static signal to segment 0
# awg.write_segment(sig_segment1, segment=1) # write the static signal to segment 1

# sequence control, read the doc for more details
awg.configure_step(step=0, segment=0, nextstep=0, loop=1, condition=condition[0])
# awg.configure_step(step=1, segment=1, nextstep=0, loop=1, condition=condition[0])

'''
channel output settings
channel 0 and channel 3 are connected to H/V AODs
allowed combinations are [0,1], [0,2], [0,3], [1,2], [1,3], [2,3], [0,1,2,3]
'''
awg.toggle_channel([0,1], amplitude=1000)
# awg.toggle_channel([0], amplitude=2500)

start()
# console stuff
vars = globals()
vars.update(locals())
code.InteractiveConsole(vars).interact("")
