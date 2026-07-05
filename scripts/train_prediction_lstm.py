from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT))

from models.lstm_model import FolkLSTM

# ----------------------------
# Load Prediction Dataset
# ----------------------------
data = np.load(
    PROJECT / "dataset" / "training" / "train_dataset_prediction.npz",
    allow_pickle=True
)

X = data["X"]
Y = data["Y"]

print("Input Shape :", X.shape)
print("Target Shape:", Y.shape)

# ----------------------------
# Dataset
# ----------------------------
class PredictionDataset(Dataset):

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


dataset = PredictionDataset(X, Y)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

print("Dataset Ready!")

# ----------------------------
# Device
# ----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# ----------------------------
# Model
# ----------------------------

model = FolkLSTM().to(device)

print("\nPrediction LSTM Loaded Successfully!\n")
print(model)

print("\nTesting Forward Pass...")

for x, y in loader:

    x = x.to(device)

    output = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)

    break

criterion = nn.MSELoss()

print("\nLoss Function Ready!")

print("\nCalculating Initial Loss...")

for x, y in loader:

    x = x.to(device)
    y = y.to(device)

    output = model(x)

    loss = criterion(
        output,
        y.permute(0,2,1)
    )

    print("Initial Loss:", loss.item())

    break

# ----------------------------
# Optimizer
# ----------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\nOptimizer Ready!")

print("\nRunning One Training Step...")

for x, y in loader:

    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(
        output,
        y.permute(0,2,1)
    )

    loss.backward()

    optimizer.step()

    print("Training Loss:", loss.item())

    break