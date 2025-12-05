
import os
import traceback
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# Load environment variables
load_dotenv(r"d:\Study\GPT_AGENT_2025_BOOK\.env")

token = os.getenv("HF_TOKEN")
print(f"Token found: {bool(token)}")

try:
    print("Attempting to load pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    print("Pipeline loaded successfully!")
except Exception:
    print("Error loading pipeline:")
    traceback.print_exc()
