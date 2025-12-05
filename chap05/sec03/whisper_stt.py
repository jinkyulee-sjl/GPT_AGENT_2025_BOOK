import os, traceback

# Fix for PyTorch 2.6+ weights_only=True default causing WeightsUnpickler errors
# Must be set BEFORE importing torch
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

import torch
import pandas as pd
# import soundfile as sf # 버전 충돌문제 해결했으므로 이제 필요 없음
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# 1. 추가하려는 경로를 정의합니다.
# Windows 경로를 다룰 때는 raw string(r"...")을 사용하는 것이 좋습니다.
ffmpeg_path = r"D:\Study\GPT_AGENT_2025_BOOK\ffmpeg-7.1.1-full_build-shared\bin"

# 2. 현재 PATH 환경 변수를 가져옵니다.
# os.pathsep은 운영체제에 맞는 경로 구분자(';' 또는 ':')를 자동으로 사용해줍니다.
path_list = os.environ.get('PATH', '').split(os.pathsep)

# 3. 추가하려는 경로가 이미 PATH에 있는지 확인합니다.
if ffmpeg_path not in path_list:
    # 4. PATH에 없는 경우에만 맨 앞에 추가합니다.
    os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
    print(f"'{ffmpeg_path}' 경로를 PATH에 추가했습니다.")
else:
    print(f"'{ffmpeg_path}' 경로는 이미 PATH에 존재합니다.")

# (선택 사항) 변경된 PATH 확인
print("\n--- 현재 PATH ---")
print(os.environ['PATH'])

# python 3.8 이후 버전에서는 os.add_dll_directory()를 사용하여 DLL 경로를 추가해 주지 않으면
# DLL 파일을 자동으로 불러오지 않는다.
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(ffmpeg_path)

import shutil
print("shutil.which('ffmpeg') ->", shutil.which("ffmpeg"))

def whisper_stt(
    audio_file_path: str,
    output_file_path: str = "./output.csv"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure device_idx is defined for pipeline: 0 for first GPU, -1 for CPU
    device_idx = 0 if torch.cuda.is_available() else -1

    # openai/whisper-large-v3-turbo 모델도 정상 동작 함. 속도를 위해서는 small 모델을 사용.
    model_id = "openai/whisper-small"

    # gtx-1660ti 사용시 float16 으로 설정하면 NaN 발생. 불가피하게 float32로 설정
    dtype = torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        processor=processor,
        dtype=dtype,
        device=device_idx,
        return_timestamps=True,
        generate_kwargs={"language": "ko", "task": "transcribe"},
    )

    result = pipe(audio_file_path)
    print("DEBUG: result keys:", result.keys())
    print("DEBUG: result text:", result.get("text", "")[:100])
    print("DEBUG: chunks count:", len(result.get("chunks", [])))
    df = whisper_to_dataframe(result, output_file_path)

    return result, df


def whisper_to_dataframe(result, output_file_path):
    start_end_text = []

    if "chunks" in result:
        for chunk in result["chunks"]:
            start = chunk["timestamp"][0]
            end = chunk["timestamp"][1]
            text = chunk["text"].strip()
            start_end_text.append([start, end, text])
            
    df = pd.DataFrame(start_end_text, columns=["start", "end", "text"])
    df.to_csv(output_file_path, index=False, sep="|")

    return df

# ==============================================================================
# [Workaround] torchcodec/torchaudio compatibility fix
# ==============================================================================
# Since torchcodec is incompatible with the current PyTorch version, we use
# soundfile to load audio and pass the waveform directly to the pipeline.

# Pytorch 와 torchcodec/torchaudio가 버전 충돌하는 문제를 해결했으므로
# 아래의 load_audio() 함수는 필요 없다.
# Pytoch Version 2.3.1 - cu121 로 다운그레이드 했음
# Python Version 3.12.6 로 다운그레이드 했음

def load_audio(file_path):
    """
    Load audio file using soundfile and convert to PyTorch tensor.
    Returns a dictionary suitable for pyannote.audio pipeline.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    # Load audio using soundfile (returns numpy array)
    # data shape: (time, channels) or (time,)
    data, sr = sf.read(file_path)
    
    # Convert to PyTorch tensor
    waveform = torch.from_numpy(data).float()
    
    # Ensure shape is (channels, time)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()
        
    return {"waveform": waveform, "sample_rate": sr}

# Example Usage:
# audio_file = "path/to/your/audio.wav"
# io = load_audio(audio_file)
# diarization = pipeline(io)


def speaker_diarization(audio_file_path: str, output_rttm_file_path: str,output_csv_file_path: str):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    # Try to instantiate the pyannote pipeline. If it fails, capture and print guidance.
    try:
        if token:
            print("HF token found in environment (value not shown).")
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        else:
            print("No HF_TOKEN in environment. If the model is gated, you must set HF_TOKEN with a token that has Read permission.")
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        print("Pipeline loaded successfully.")
    except Exception as e:
        print("Failed to load Pipeline.from_pretrained()")
        traceback.print_exc()
        print("If the model is gated: create an HF token with Read permission and either set it as HF_TOKEN in your environment or pass it with token=token.")
        print("You can also accept the model terms on the Hugging Face model page if required.")
        raise

    # GPU: try to move pipeline to GPU but handle failures gracefully
    try:
        if torch.cuda.is_available():
            try:
                diarization_pipeline.to(torch.device("cuda:0"))
                print("CUDA available — diarization pipeline moved to GPU.")
            except Exception as e:
                print("Could not move diarization pipeline to GPU (possibly OOM). Continuing on CPU.")
                traceback.print_exc()
        else:
            print("CUDA is not available — using CPU.")
    except NameError:
        # diarization_pipeline variable might not exist if instantiation failed
        print("Diarization pipeline not available to move to GPU. Skipping GPU move.")


    # Pytorch 와 torchcodec/torchaudio가 버전 충돌하는 문제를 해결했으므로
    # 아래의 코드는 필요 없다.
    # Pytoch Version 2.3.1 - cu121 로 다운그레이드 했음
    # Python Version 3.12.6 로 다운그레이드 했음

    # io = load_audio(audio_file_path) # 이제 필요 없음

    # Run the pipeline
    try:
        diarization_result = diarization_pipeline(audio_file_path)
        # diarization_result = diarization_pipeline(io) # 이제 필요 없음
        print("Diarization pipeline executed successfully.")
    except Exception as e:
        print("Failed to execute pipeline()")
        traceback.print_exc()
        raise

    # dump the diarization output to disk using  RTTM format
    with open(output_rttm_file_path, "w", encoding="utf-8") as rttm:
        # diarization_result.speaker_diarization.write_rttm(rttm) # 이제 필요 없음   
        diarization_result.write_rttm(rttm)

    # 판다스 데이터프레임으로 변환
    df_rttm = pd.read_csv(output_rttm_file_path, # rttm 파일 경로
        sep=" ", # 구분자
        header=None, # 헤더 없음
        names=['type', 'file', 'chnl', 'start', 'duration', 'C1', 'C2', 'speaker_id', 'C3', 'C4']
        )
    
    df_rttm["end"] = df_rttm["start"] + df_rttm["duration"]

    # speaker_id를 기반으로 화자별로 구간 나누기
    df_rttm["number"] = None
    df_rttm.at[0, "number"] = 0

    for i in range(1, len(df_rttm)):
        if df_rttm.at[i, "speaker_id"] == df_rttm.at[i-1, "speaker_id"]:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"]
        else:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"] + 1

    df_rttm_grouped = df_rttm.groupby("number").agg(
        start=pd.NamedAgg(column="start", aggfunc="min"),
        end=pd.NamedAgg(column="end", aggfunc="max"),
        speaker_id=pd.NamedAgg(column="speaker_id", aggfunc="first")
    )

    df_rttm_grouped["duration"] = df_rttm_grouped["end"] - df_rttm_grouped["start"] 

    df_rttm_grouped.to_csv(output_csv_file_path, index=False, encoding="utf-8")

    return df_rttm_grouped

if __name__ == "__main__":
    audio_file_path = r"D:/Study/GPT_AGENT_2025_BOOK/chap05/audio/싼기타_비싼기타.mp3"
    stt_output_file_path = r"D:/Study/GPT_AGENT_2025_BOOK/chap05/audio/싼기타_비싼기타.csv"
    rttm_file_path = r"D:/Study/GPT_AGENT_2025_BOOK/chap05/audio/싼기타_비싼기타.rttm"
    rttm_csv_file_path = r"D:/Study/GPT_AGENT_2025_BOOK/chap05/audio/싼기타_비싼기타_rttm.csv"
    
    # result, df = whisper_stt(
    #     audio_file_path,
    #     csv_file_path,
    #     )
    
    df_rttm = speaker_diarization(audio_file_path, rttm_file_path, rttm_csv_file_path)
    print(df_rttm)

