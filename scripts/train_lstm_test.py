from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

data = np.load(
    PROJECT / "dataset" / "training" / "train_dataset_normalized.npz",
    allow_pickle=True
)

X = data["X"]

print("Minimum :", X.min())
print("Maximum :", X.max())
print("Mean    :", X.mean())
print("Std Dev :", X.std())

print("Dataset Shape:", X.shape)

import torch
from torch.utils.data import Dataset

class FolkDataset(Dataset):

    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x, x

dataset = FolkDataset(X)

print("Dataset Loaded Successfully")
print("Total Samples:", len(dataset))
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

print("\nChecking one batch...")

for x, y in loader:
    print("Input Shape :", x.shape)
    print("Target Shape:", y.shape)
    break



import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT))

from models.lstm_model import FolkLSTM

model = FolkLSTM()

print("\nModel Loaded Successfully!\n")
print(model)


print("\nTesting Forward Pass...")

for x, y in loader:

    output = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)

    break

print("\nCreating Loss Function...")

criterion = torch.nn.MSELoss()

print("Loss Function Ready!")

print("\nCalculating Initial Loss...")

for x, y in loader:

    output = model(x)

    loss = criterion(
        output,
        y.permute(0,2,1)
    )

    print("Initial Loss :", loss.item())

    break