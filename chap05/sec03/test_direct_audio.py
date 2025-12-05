import os, traceback, sys
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Setup FFmpeg path
ffmpeg_path = r"D:\Study\GPT_AGENT_2025_BOOK\ffmpeg-7.1.1-full_build-shared\bin"
path_list = os.environ.get('PATH', '').split(os.pathsep)
if ffmpeg_path not in path_list:
    os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(ffmpeg_path)

def test_direct_audio():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    audio_file_path = r"D:/Study/GPT_AGENT_2025_BOOK/chap05/audio/싼기타_비싼기타.mp3"
    
    print("Loading pipeline...")
    pipeline_obj = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    
    if torch.cuda.is_available():
        pipeline_obj.to(torch.device("cuda:0"))
        print("Pipeline on GPU")
    else:
        print("Pipeline on CPU")

    print("\n=== TEST: Direct audio file path ===")
    try:
        result = pipeline_obj(audio_file_path)
        print("SUCCESS: Direct path works!")
        print(f"Result type: {type(result)}")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {str(e)[:200]}")
        return False

if __name__ == "__main__":
    success = test_direct_audio()
    sys.exit(0 if success else 1)
