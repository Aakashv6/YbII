from lib.AWG import *
import code
# import readline
import rlcompleter
import os
import argparse
from lib.waveform import *
import sys
from datetime import datetime, timedelta
from pathlib import Path

# _step = int32(0)

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
    awg.close()
    print("closing awg and quitting...")
    exit(0)

# argument parsing, more functionalities to be added later...
parser = argparse.ArgumentParser(description='AWG control')
parser.add_argument(
    '-id',
    type=int,
    default=0,
    help='0 for top AWG, 1 for bot AWG, default to 0'
)
args = parser.parse_args()
awg_id = ''
if args.id == 0:
    awg_id = b'/dev/spcm0'
elif args.id == 1:
    awg_id = b'/dev/spcm1'
else:
    print("invalid id")
    exit(0)

sampling_rate = int(614.4e6)

today_path = Path("data").joinpath(datetime.now().strftime("%Y%m%d"))
today_path.mkdir(parents=True, exist_ok=True)

previous_counter = 1
previous_path = Path("data").joinpath(
    (datetime.now() - timedelta(previous_counter)).strftime("%Y%m%d")
)
while not previous_path.exists():
    previous_counter += 1
    previous_path = Path("data").joinpath(
        (datetime.now() - timedelta(previous_counter)).strftime("%Y%m%d")
    )

init_file = "array20_df=0.9MHz_init.npz"
save_file = 'array20_df=0.9MHz.npz'

array20_90 = Waveform()

if today_path.joinpath(init_file).is_file():
    array20_90.from_file(today_path.joinpath(init_file))
else:
    array20_90.from_file(previous_path.joinpath(init_file))
    save_wfm(array20_90, today_path.joinpath(init_file))
print(array20_90.amplitude)
print()

sig0 = create_static_array(array20_90, full=False)
sig1 = sig0

save_wfm(array20_90, today_path.joinpath(save_file))

# array20_90.save_csv('data/array20_df=0.9MHz.csv')

# AWG stuff

# condition for awg step sequence. 
# 0 for infinite loop until trigger
# 1 for finite loop, then goto nextstep afterwards.
condition = [SPCSEQ_ENDLOOPONTRIG, SPCSEQ_ENDLOOPALWAYS]
awg = AWG()
awg.open(id=0)  # change this to 0 for top AWG, 1 for bot AWG
awg.set_sampling_rate(sampling_rate)
awg.toggle_channel(0, amplitude=2500)
awg.set_trigger(EXT0=SPC_TM_POS)
awg.set_sequence_mode(2)  # partition AWG memory into 2 segments
awg.write_segment(sig0, segment=0)
awg.write_segment(sig1, segment=1)
# awg.write_segment(empty, segment=1)
awg.configure_step(step=0, segment=0, nextstep=1, loop=1, condition=condition[0])
awg.configure_step(step=1, segment=1, nextstep=0, loop=1, condition=condition[0])
# awg.configure_step(step=1, segment=1, nextstep=0, loop=1, condition=SPCSEQ_ENDLOOPONTRIG)
start()

# console
vars = globals()
vars.update(locals())
readline.set_completer(rlcompleter.Completer(vars).complete)
readline.parse_and_bind("tab: complete")
code.InteractiveConsole(vars).interact("")
