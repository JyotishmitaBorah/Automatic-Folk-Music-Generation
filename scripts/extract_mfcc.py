from pathlib import Path
import librosa
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

AUDIO = PROJECT/"dataset"/"processed_audio"
OUTPUT = PROJECT/"dataset"/"features"/"mfcc"

OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(AUDIO.glob("MOI_*.wav"))

for file in files:

    y, sr = librosa.load(file, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    np.save(
        OUTPUT/(file.stem+".npy"),
        mfcc
    )

print("MFCC Extraction Complete")