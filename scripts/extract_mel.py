from pathlib import Path
import librosa
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

AUDIO = PROJECT/"dataset"/"processed_audio"
OUTPUT = PROJECT/"dataset"/"features"/"mel"

OUTPUT.mkdir(parents=True, exist_ok=True)

for file in sorted(AUDIO.glob("MOI_*.wav")):

    y, sr = librosa.load(file, sr=22050)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    np.save(
        OUTPUT/(file.stem+".npy"),
        mel_db
    )

print("Mel Spectrogram Extraction Complete")