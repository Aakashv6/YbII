from .AWG import AWG
from .config import Config

def connect(awg: AWG, id=0) -> bool:
    awg.open(id)

def load_config(awg: AWG, config: Config):
    awg.set_config(config)

def start(awg: AWG):
    print("starting awg output")
    awg.run()
    awg.force_trigger

def trigger(awg: AWG):
    print("triggering...")
    awg.force_trigger()

def stop(awg: AWG):
    print("stopping awg output")
    awg.stop()

def disconnect(awg: AWG):
    print("disconnecting awg")
    awg.stop()
    awg.close()    
    