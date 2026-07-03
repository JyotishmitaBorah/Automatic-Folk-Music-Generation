from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_AUDIO = PROJECT_ROOT / "dataset" / "raw_audio"
PROCESSED_AUDIO = PROJECT_ROOT / "dataset" / "processed_audio"

PROCESSED_AUDIO.mkdir(exist_ok=True)

wav_files = sorted(RAW_AUDIO.glob("MOI_*.wav"))

print(f"Found {len(wav_files)} audio files.\n")

for wav in tqdm(wav_files):

    y, sr = librosa.load(
        wav,
        sr=22050,
        mono=True
    )

    # Trim silence
    y, _ = librosa.effects.trim(
        y,
        top_db=20
    )

    # Normalize volume
    y = librosa.util.normalize(y)

    output_file = PROCESSED_AUDIO / wav.name

    sf.write(
        output_file,
        y,
        sr
    )

print("\nPreprocessing Completed!")
print(f"Processed files saved in:\n{PROCESSED_AUDIO}")