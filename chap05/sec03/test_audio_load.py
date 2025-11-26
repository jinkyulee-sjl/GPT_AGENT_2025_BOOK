import wave
import struct
import math
import os
import torch
import torchaudio

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

def test_load():
    filename = "test_audio.wav"
    create_sine_wave_wav(filename)
    print(f"Created {filename}")
    
    try:
        print("Available backends:", torchaudio.list_audio_backends())
    except:
        print("Could not list backends.")

    try:
        # Try to force soundfile backend if available
        if 'soundfile' in torchaudio.list_audio_backends():
            torchaudio.set_audio_backend("soundfile")
            print("Set backend to soundfile.")
        
        print("Attempting to load with torchaudio...")
        waveform, sample_rate = torchaudio.load(filename, backend="soundfile") # Explicitly pass backend if supported in load
        print(f"Success! Loaded waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
        return True
    except Exception as e:
        print(f"Failed to load audio: {e}")
        # import traceback
        # traceback.print_exc()
        
        # Try without backend arg if the above failed (maybe backend arg is not supported in this version)
        try:
             print("Retrying without explicit backend arg...")
             waveform, sample_rate = torchaudio.load(filename)
             print(f"Success! Loaded waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
             return True
        except Exception as e2:
             print(f"Retry failed: {e2}")

        return False
    finally:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Removed {filename}")

if __name__ == "__main__":
    test_load()
