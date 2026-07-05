from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

# ----------------------------
# Load normalized dataset
# ----------------------------

data = np.load(
    PROJECT / "dataset" / "training" / "train_dataset_normalized.npz",
    allow_pickle=True
)

X = data["X"]

print("Loaded:", X.shape)

# ----------------------------
# Create prediction dataset
# ----------------------------

inputs = []
targets = []

for sample in X:

    # sample shape = (13,431)

    inp = sample[:, :-1]     # Frames 0 ... 429

    tgt = sample[:, 1:]      # Frames 1 ... 430

    inputs.append(inp)
    targets.append(tgt)

inputs = np.array(inputs, dtype=np.float32)
targets = np.array(targets, dtype=np.float32)

print("Input Shape :", inputs.shape)
print("Target Shape:", targets.shape)

# ----------------------------
# Save
# ----------------------------

save_path = PROJECT / "dataset" / "training" / "train_dataset_prediction.npz"

np.savez_compressed(
    save_path,
    X=inputs,
    Y=targets
)

print("\nPrediction Dataset Saved!")
print(save_path)