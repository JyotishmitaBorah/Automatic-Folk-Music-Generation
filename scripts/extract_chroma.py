from pathlib import Path
import librosa
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

AUDIO = PROJECT/"dataset"/"processed_audio"
OUTPUT = PROJECT/"dataset"/"features"/"chroma"

OUTPUT.mkdir(parents=True, exist_ok=True)

for file in sorted(AUDIO.glob("MOI_*.wav")):

    y, sr = librosa.load(file, sr=22050)

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    np.save(
        OUTPUT/(file.stem+".npy"),
        chroma
    )

print("Chroma Extraction Complete")