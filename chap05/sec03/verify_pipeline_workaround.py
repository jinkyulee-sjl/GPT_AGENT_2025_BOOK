import os
import torch
import soundfile as sf
import numpy as np
import wave
import struct
import math
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# 1. Create dummy audio
def create_sine_wave_wav(filename, duration=1, sample_rate=16000, frequency=440):
    n_samples = int(sample_rate * duration)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)

# 2. Define workaround function
def load_audio(file_path):
    data, sr = sf.read(file_path)
    waveform = torch.from_numpy(data).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()
    return {"waveform": waveform, "sample_rate": sr}

def verify():
    filename = "verify_audio.wav"
    create_sine_wave_wav(filename)
    print(f"Created {filename}")

    try:
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        print("Loading pipeline...")
        if token:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        else:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        
        print("Pipeline loaded. Moving to GPU if available...")
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda:0"))
            print("Moved to GPU.")
        
        print("Loading audio with workaround...")
        io = load_audio(filename)
        print(f"Audio loaded: {io['waveform'].shape}, SR: {io['sample_rate']}")
        
        print("Running diarization...")
        diarization = pipeline(io)
        print("Diarization successful!")
        print(diarization)
        
    except Exception as e:
        print("Verification FAILED.")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Removed {filename}")

if __name__ == "__main__":
    verify()
