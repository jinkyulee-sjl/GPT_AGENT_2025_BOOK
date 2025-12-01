import os
import torch
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# os.environ["PATH"] += os.pathsep + r"D:\Study\GPT_AGENT_2025_BOOK\ffmpeg-7.1.1-full_build-shared\bin  "
# ffmpeg 의 dll 을 사용하려면 ffmpeg shared 를 설치해야 함.
bin_path = r"D:\Study\GPT_AGENT_2025_BOOK\ffmpeg-7.1.1-full_build-shared\bin"
os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")

# python 3.8 이후 버전에서는 os.add_dll_directory()를 사용하여 DLL 경로를 추가해 주지 않으면
# DLL 파일을 자동으로 불러오지 않는다.
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(bin_path)

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
    
if __name__ == "__main__":
    result, df = whisper_stt(
        r"D:/Study/GPT_AGENT_2025_BOOK/chap05/audio/싼기타_비싼기타.mp3",
        r"D:/Study/GPT_AGENT_2025_BOOK/chap05/audio/싼기타_비싼기타.csv",
        )
    
    print(df)
