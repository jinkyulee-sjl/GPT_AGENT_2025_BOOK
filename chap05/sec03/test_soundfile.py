import wave
import struct
import math
import os
import torch
import numpy as np

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

def test_soundfile():
    filename = "test_audio_sf.wav"
    create_sine_wave_wav(filename)
    print(f"Created {filename}")
    
    try:
        import soundfile as sf
        print("soundfile imported successfully.")
        
        data, samplerate = sf.read(filename)
        print(f"Loaded with soundfile. Shape: {data.shape}, SR: {samplerate}")
        
        # Convert to torch tensor (channels, time)
        # soundfile returns (time, channels) or just (time,) for mono
        waveform = torch.from_numpy(data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
            
        print(f"Converted to tensor: {waveform.shape}")
        return True
        
    except ImportError:
        print("soundfile package is NOT installed.")
        return False
    except Exception as e:
        print(f"Failed to load with soundfile: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Removed {filename}")

if __name__ == "__main__":
    test_soundfile()
