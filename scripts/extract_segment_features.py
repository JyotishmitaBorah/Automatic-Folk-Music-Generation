from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent.parent

SEGMENTS = PROJECT/"dataset"/"segments"

MFCC_OUT = PROJECT/"dataset"/"segment_features"/"mfcc"
MEL_OUT = PROJECT/"dataset"/"segment_features"/"mel"

MFCC_OUT.mkdir(parents=True, exist_ok=True)
MEL_OUT.mkdir(parents=True, exist_ok=True)

files = sorted(SEGMENTS.glob("*.wav"))

print(f"Found {len(files)} segments")

for audio in tqdm(files):

    y, sr = librosa.load(audio, sr=22050)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    np.save(
        MFCC_OUT/f"{audio.stem}.npy",
        mfcc
    )

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    np.save(
        MEL_OUT/f"{audio.stem}.npy",
        mel
    )

print("\nFeature Extraction Completed.")