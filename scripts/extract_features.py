from pathlib import Path
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_DIR = PROJECT_ROOT / "dataset" / "processed_audio"
FEATURE_DIR = PROJECT_ROOT / "dataset" / "features"

FEATURE_DIR.mkdir(exist_ok=True)

rows = []

files = sorted(AUDIO_DIR.glob("MOI_*.wav"))

print(f"Found {len(files)} audio files")

for file in tqdm(files):

    y, sr = librosa.load(file, sr=22050)

    duration = librosa.get_duration(y=y, sr=sr)

    rms = np.mean(librosa.feature.rms(y=y))

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = np.asarray(tempo).item()

    rows.append({
        "ID": file.stem,
        "Duration": duration,
        "Tempo": tempo,
        "RMS": float(rms),
        "ZeroCrossingRate": float(zcr),
        "SpectralCentroid": float(centroid),
        "SpectralBandwidth": float(bandwidth),
        "SpectralRolloff": float(rolloff)
    })

df = pd.DataFrame(rows)

df.to_csv(FEATURE_DIR / "audio_features.csv", index=False)

print(df.head())

print("\nFeature Extraction Completed.")