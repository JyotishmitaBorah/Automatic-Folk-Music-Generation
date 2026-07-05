from pathlib import Path
import sys
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT))

from models.lstm_model import FolkLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

# -------------------------
# Load trained model
# -------------------------

checkpoint = torch.load(
    PROJECT / "models" / "prediction_lstm_best.pth",
    map_location=device
)

model = FolkLSTM().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Prediction Model Loaded!")

# -------------------------
# Load prediction dataset
# -------------------------

data = np.load(
    PROJECT / "dataset" / "training" / "train_dataset_prediction.npz",
    allow_pickle=True
)

X = data["X"]

print("Dataset Loaded!")
print("Shape:", X.shape)

# -------------------------
# Select Seed Sequence
# -------------------------

# -------------------------------------------------
# Select Seed
# -------------------------------------------------

seed = X[0]

generated = seed.copy()

print("Seed Shape :", generated.shape)

window = seed.copy()

# -------------------------------------------------
# Recursive Generation
# -------------------------------------------------

NUM_NEW_FRAMES = 400

model.eval()

with torch.no_grad():

    for i in range(NUM_NEW_FRAMES):

        inp = torch.tensor(
            window,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        prediction = model(inp)

        prediction = prediction.squeeze(0)

        prediction = prediction.cpu().numpy()

        next_frame = prediction[-1]

        generated = np.concatenate(
            [
                generated,
                next_frame.reshape(13,1)
            ],
            axis=1
        )

        window = np.concatenate(
            [
                window[:,1:],
                next_frame.reshape(13,1)
            ],
            axis=1
        )

        if (i+1)%20==0:

            print(f"{i+1} Frames Generated")

print()

print("Generated Shape :", generated.shape)

OUTPUT = PROJECT / "outputs" / "generated"

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    OUTPUT/"generated_sequence.npy",
    generated
)

print("Generated Sequence Saved!")

import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(14,5))

librosa.display.specshow(
    generated,
    x_axis="time",
    sr=22050
)

plt.colorbar()

plt.title("Generated Folk Music Sequence")

plt.tight_layout()

plt.savefig(
    OUTPUT/"generated_sequence.png",
    dpi=300
)

plt.close()

print("Image Saved!")