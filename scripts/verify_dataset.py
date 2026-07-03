from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

data = np.load(
    PROJECT / "dataset" / "training" / "train_dataset.npz",
    allow_pickle=True
)

X = data["X"]
filenames = data["filenames"]

print("Shape:", X.shape)
print("Datatype:", X.dtype)
print("Number of samples:", len(filenames))
print("First sample:", filenames[0])