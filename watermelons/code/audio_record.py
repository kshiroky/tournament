import os
import sounddevice as sd
from scipy.io.wavfile import write

def recorder(path_to_dir, label):
    """This function records and saves the sound"""
    # sampling_frequency
    freq = 44100
    # recording duration
    duration = 5
    recording = sd.rec(int(duration * freq),
            samplerate=freq, channels=2)
    sd.wait()
    num_of_image = len(os.listdir(path_to_dir))
    write(os.path.join(path_to_dir, f"recording{num_of_image}.wav)"), freq, recording)
    return None