from .pyspcm import *
from .spcm_tools import *
import sys
import numpy as np
# from .config import Config
from .waveform import Waveform, create_static_array

#
# **************************************************************************
# Class for controlling the AWG
#
# User must install the AWG driver for Windows 8/9/10/11, found in
# https://spectrum-instrumentation.com/support/downloads.php
# Look under model M4i 6622-x8
#
# The python module provided in pyspcm is in essence a c/c++ wrapper,
# hence the use of ctype variables in this code.
# **************************************************************************
#


class AWG:

    def __init__(self):
        """
        constructor for the AWG class
        """
        self.card = None  # holds the spcm card instance, this is used by all internal calls
        self.card_idx = 0  # index of connected awg
        self.card_type = int32(0)  # type of card in a 32-bit mask
        self.serial_number = int32(0)  # serial number
        self.sample_rate = int64(0)  # sampling rate of the AWG, this sets the "speed" of the AWG
        self.mem_size = int64(0)  # maximum memory size of the AWG
        self.full_scale = int32(0)  # full scale scaling of output voltage, used for data calculation
        self.channel = [0,0,0,0]  # activated channels
        self.ch_amp = [0,0,0,0]  # channel output amplitude
        self.mode = ""  # current mode AWG is running on

    def is_connected(self):
        return self.card is not None and self.card

    def open(self, id=0, remote=False) -> bool:
        """
        opens and initializes instance variables
        :param remote: flag to determine remote connection, default is False
        """
        if remote:
            self.card = spcm_hOpen(create_string_buffer(b'TCPIP::192.168.1.10::inst0::INSTR'))
        else:
            if id == 0:
                self.card = spcm_hOpen(create_string_buffer(b'/dev/spcm0'))
                self.card_idx = 0
            else:
                self.card = spcm_hOpen(create_string_buffer(b'/dev/spcm1'))
                self.card_idx = 1
        if not self.is_connected():
            print("no card found...\n")
            return False
        else:
            spcm_dwGetParam_i32(self.card, SPC_PCITYP, byref(self.card_type))
            spcm_dwGetParam_i32(self.card, SPC_PCISERIALNO, byref(self.serial_number))
            spcm_dwGetParam_i64(self.card, SPC_SAMPLERATE, byref(self.sample_rate))
            spcm_dwGetParam_i64(self.card, SPC_PCIMEMSIZE, byref(self.mem_size))
            spcm_dwGetParam_i32(self.card, SPC_MIINST_MAXADCVALUE,
                                byref(self.full_scale))  # full scale value for data generation purpose
            name = szTypeToName(self.card_type.value)
            sys.stdout.write("Card: {0} sn {1:05d}\n".format(name, self.serial_number.value))
            sys.stdout.write("Max sample Rate: {:.1f} MHz\n".format(self.sample_rate.value / 1000000))
            sys.stdout.write("Memory size: {:.0f} MBytes\n\n".format(self.mem_size.value / 1024 / 1024))
        return True
        # self.check_error()

    def close(self):
        spcm_vClose(self.card)
        self.card = None

    def check_error(self, message="") -> bool:
        """
        Note: this function currently causes seg fault,
        likely due to incorrect string buffer type.
        error checking helper.
        :param message: caller defined string for debugging purposes
        :return: 1 if error is found, 0 otherwise
        """
        err_reg = uint32(0)
        err_val = int32(0)
        err_text = create_string_buffer(256)
        err_code = spcm_dwGetErrorInfo_i32(self.card, byref(err_reg), byref(err_val), byref(err_text))
        if err_code:
            print(
                f"{message}\n"
                f"error code (see spcerr.py): {hex(err_code)}\n"
                f"error text: {err_text.value}\n"
                f"error register: {err_reg.value}\n"
                f"error val: {err_val.value}\n"
            )
            self.close()
            return True
        return False
    
    def run(self):
        """
        start the card, enable trigger and wait for trigger to start
        """
        if not self.is_connected():
            return
        spcm_dwSetParam_i32(self.card, SPC_M2CMD,
                            M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
        self.check_error("Checking error at run")

    def stop(self):
        """
        stop the card, this is different from closing the card
        """
        if not self.is_connected():
            return
        spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_STOP)
        self.check_error("Checking error at stop")

    def reset(self):
        """
        resets the board, this clears all onboard memory and settings
        """
        if not self.is_connected():
            return
        spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_RESET)
        self.sample_rate = int64(0)  # sampling rate of the AWG, this sets the "speed" of the AWG
        self.channel = [0,0,0,0]  # activated channels
        self.ch_amp = [0,0,0,0]  # channel output amplitude
        self.mode = ""  # current mode AWG is running on

        # self.check_error("Checking error at reset")

    def force_trigger(self):
        """
        force a trigger event, this completely mimics an actual trigger event
        """
        if not self.is_connected():
            return
        spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_FORCETRIGGER)
        # print("forcing a software trigger (like real)")
        # self.check_error("Checking error at force_trigger")

    def set_sampling_rate(self, sr: int):
        """
        set sampling rate
        :param sr: 64bit integer between 50MHz and 625MHz
        """
        if not self.is_connected():
            return
        self.sample_rate = int64(sr)
        spcm_dwSetParam_i64(self.card, SPC_SAMPLERATE, sr)
        # self.check_error("Checking error at set_sampling_rate")
        # print(f"Setting sampling rate to {self.sample_rate.value / 1e6} MHz")

    def toggle_channel(self, ch: list[int], amplitude: int=1000, stoplvl: str="ZERO"):
        """
        enable/disable individual channel and set its parameters
        :param ch: enables channel 0-3.
        :param amplitude: sets output amplitude of each channel between 80-2500mV, default level is 1000mV.
        :param stoplvl: sets channels pause behavior, "ZERO", "LOW", "HIGH", "HOLDLAST
        """
        if not self.is_connected():
            return
        stopmask = {"ZERO":     SPCM_STOPLVL_ZERO,
                    "LOW":      SPCM_STOPLVL_LOW,
                    "HIGH":     SPCM_STOPLVL_HIGH,
                    "HOLDLAST": SPCM_STOPLVL_HOLDLAST}
        if not self.is_connected():
            return
        enable_mask = 0

        for c in ch:
            self.channel[c] = 1
            enable_mask = enable_mask | int(2**c)
        spcm_dwSetParam_i64(self.card, SPC_CHENABLE, enable_mask)

        for c in ch:
            # c = uint32(c)
            spcm_dwSetParam_i32(self.card, SPC_ENABLEOUT0 + c * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0), 1)
            spcm_dwSetParam_i32(self.card, SPC_AMP0 + c * (SPC_AMP1 - SPC_AMP0), amplitude)
            spcm_dwSetParam_i32(self.card, SPC_CH0_STOPLEVEL + c * (SPC_CH1_STOPLEVEL - SPC_CH0_STOPLEVEL), stopmask[stoplvl])
    
    def get_channel_count(self) -> int:
        if not self.is_connected():
            return
        chcount = int32(0)
        spcm_dwGetParam_i32(self.card, SPC_CHCOUNT, byref(chcount))
        return chcount.value

    def set_trigger(self, **kwargs):
        """
        set trigger behavior
        @TODO: add more functionality
        :param args:
        :param kwargs:
        """
        if not self.is_connected():
            return
        for key,value in kwargs.items():
            if key == "EXT0":
                spcm_dwSetParam_i32(self.card, SPC_TRIG_ORMASK, SPC_TMASK_EXT0)  # using external channel 0 as trigger
                spcm_dwSetParam_i32(self.card, SPC_TRIG_TERM, 1) # 0: 50 Ohm, 1: high impedance
                spcm_dwSetParam_i32(self.card, SPC_TRIG_EXT0_ACDC, COUPLING_DC)
                spcm_dwSetParam_i32(self.card, SPC_TRIG_EXT0_LEVEL0, 1000) # -10000 to 10000 mV
                spcm_dwSetParam_i32(self.card, SPC_TRIG_EXT0_MODE, value)
            if key == "EXT1":
                spcm_dwSetParam_i32(self.card, SPC_TRIG_ORMASK, SPC_TMASK_EXT1)  # using external channel 1 as trigger
                spcm_dwSetParam_i32(self.card, SPC_TRIG_EXT1_MODE, value)
        # self.check_error("Checking error at set_trigger")

    def get_aligned_buf(self, size):
        """
        returns a numpy array at a page-aligned memory location
        :param size: number of samples used for data calculation
        :return: 
        """
        data_length_bytes = int(size * 2 * np.sum(self.channel))
        buffer = pvAllocMemPageAligned(data_length_bytes)  # buffer now holds a page-aligned location
        buffer_data = cast(addressof(buffer), ptr16)  # cast it to int16 array
        # array = np.frombuffer(buffer_data, dtype=int16)
        return buffer, buffer_data

    def set_sequence_mode(self, nseg: int):
        """
        set the AWG mode to sequence replay, and divide memory into segments.
        :param nseg: number of segments the memory is divided into, must be powers of 2.
        """
        if not self.is_connected():
            return
        self.mode = "Sequence Replay"
        spcm_dwSetParam_i32(self.card, SPC_CARDMODE,            SPC_REP_STD_SEQUENCE)  # Sequence replay mode
        spcm_dwSetParam_i32(self.card, SPC_SEQMODE_MAXSEGMENTS, nseg)  # set number of sequences the memory is divided into
        spcm_dwSetParam_i32(self.card, SPC_SEQMODE_STARTSTEP,   0)  # set starting step to be 0
        # self.check_error("Checking error at set_sequence_mode")

    # @profile
    def write_segment(self, data: np.ndarray, segment: int):
        """
        write data onto a specified segment in sequence replay mode
        :param data: numpy array containing waveform data
        :param segment: the segment to write on
        :return:
        """
        if not self.is_connected():
            return

        if self.mode != "Sequence Replay":
            print("Wrong method, current mode is: " + self.mode)
            return
        chcount = int32(0)
        spcm_dwGetParam_i32(self.card, SPC_CHCOUNT, byref(chcount))
        # if data.dtype != int:
        #     sys.stdout.write("data must be in int type\n")
        #     return
        if data.size > self.mem_size.value / chcount.value / 2:
            sys.stdout.write("data is too big")
            return
        spcm_dwSetParam_i32(self.card, SPC_SEQMODE_WRITESEGMENT, segment)  # set current segment to write on
        spcm_dwSetParam_i32(self.card, SPC_SEQMODE_SEGMENTSIZE,  data.size)  # set size of segment in unit of samples
        
        # data transfer
        sample_len = data.size
        buflength = uint32(sample_len * 2)  # samples * (2 bytes/sample)
        data_ptr = data.ctypes.data_as(ptr16)  # cast data array into a c-like array
        buffer = pvAllocMemPageAligned(sample_len * 2)  # buffer now holds a page-aligned location
        buffer_data = cast(addressof(buffer), ptr16)  # cast it to int16 array
        memmove(buffer_data, data_ptr, sample_len * 2)  # moving data into the page-aligned block
        spcm_dwDefTransfer_i64(self.card, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, 0, byref(buffer), 0, buflength)
        # self.check_error("Checking error at write_segment")
        spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
        spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_DATA_STOPDMA)
        # self.check_error("Checking error at write_segment")
        buffer = None

    def configure_step(self, step: int, segment: int, nextstep: int, loop: int, condition: int):
        """
        configure a step in the sequence replay mode.
        :param step: the step to be configured, sequence starts at step 0 on default, max step is 32.
        :param segment: memory segment to be associated with the configured step.
        :param nextstep: index of next step in the sequence.
        :param loop: number of times the current step is repeated before checking for next step condition.
        :param condition: behavior after current step is repeated "loop" amount of times.
        list of conditions: SPCSEQ_ENDLOOPALWAYS, SPCSEQ_ENDLOOPONTRIG, SPCSEQ_END, see manual for detail.
        """
        if not self.is_connected():
            return
        if self.mode != "Sequence Replay":
            print("Wrong method, current mode is: " + self.mode)
            return
        mask = (condition << 32) | (loop << 32) | (nextstep << 16) | segment  # 64-bit mask
        spcm_dwSetParam_i64(self.card, SPC_SEQMODE_STEPMEM0 + step, int64(mask))
        # self.check_error("Checking error at configure_step")

