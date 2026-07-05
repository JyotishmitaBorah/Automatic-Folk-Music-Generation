from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

PROJECT = Path(__file__).resolve().parent.parent

# ----------------------------
# Load Generated MFCC
# ----------------------------

generated = np.load(
    PROJECT / "outputs" / "generated" / "generated_sequence.npy"
)

print("Generated MFCC Shape:", generated.shape)

# ----------------------------
# Parameters
# ----------------------------

SR = 22050
N_MFCC = 13
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# ----------------------------
# Approximate Inverse
# ----------------------------

print("\nConverting MFCC → Mel Spectrogram...")

mel = librosa.feature.inverse.mfcc_to_mel(
    generated,
    n_mels=N_MELS
)

print("Mel Shape:", mel.shape)

print("\nConverting Mel → Audio...")

audio = librosa.feature.inverse.mel_to_audio(
    mel,
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    power=2.0
)

print("Audio Length:", len(audio))

# ----------------------------
# Save Audio
# ----------------------------

OUTPUT = PROJECT / "outputs" / "generated"
OUTPUT.mkdir(parents=True, exist_ok=True)

sf.write(
    OUTPUT / "generated_folk.wav",
    audio,
    SR
)

print("\nGenerated audio saved successfully!")