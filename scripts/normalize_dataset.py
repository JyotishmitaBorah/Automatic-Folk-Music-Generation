from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

input_file = PROJECT / "dataset" / "training" / "train_dataset.npz"
output_file = PROJECT / "dataset" / "training" / "train_dataset_normalized.npz"

data = np.load(input_file, allow_pickle=True)

X = data["X"]
filenames = data["filenames"]

mean = X.mean()
std = X.std()

X_norm = (X - mean) / std

np.savez_compressed(
    output_file,
    X=X_norm,
    filenames=filenames,
    mean=mean,
    std=std
)

print("Normalization Complete!")
print("Mean :", X_norm.mean())
print("Std  :", X_norm.std())
print("Saved to:", output_file)