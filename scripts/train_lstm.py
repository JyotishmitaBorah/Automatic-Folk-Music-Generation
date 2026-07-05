from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt

# ----------------------------
# Project Path
# ----------------------------
PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT))

from models.lstm_model import FolkLSTM

# ----------------------------
# Load Dataset
# ----------------------------
data = np.load(
    PROJECT / "dataset" / "training" / "train_dataset_normalized.npz",
    allow_pickle=True
)

X = data["X"]

print("Dataset Shape:", X.shape)

# ----------------------------
# Dataset Class
# ----------------------------
class FolkDataset(Dataset):

    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x


dataset = FolkDataset(X)

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

print("Using:", device)

# ----------------------------
# Model
# ----------------------------
model = FolkLSTM().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ----------------------------
# Training
# ----------------------------
EPOCHS = 100

loss_history = []

best_loss = float("inf")

MODEL_DIR = PROJECT / "models"
MODEL_DIR.mkdir(exist_ok=True)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

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

        running_loss += loss.item()

    epoch_loss = running_loss / len(loader)

    loss_history.append(epoch_loss)

    print(
        f"Epoch {epoch+1:03d}/{EPOCHS} | Loss = {epoch_loss:.6f}"
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            MODEL_DIR / "folk_lstm.pth"
        )

print("\nTraining Finished!")
print("Best Loss:", best_loss)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

OUTPUT = PROJECT / "outputs"
OUTPUT.mkdir(exist_ok=True)

plt.savefig(
    OUTPUT / "training_loss.png",
    dpi=300
)

plt.show()

np.save(
    OUTPUT / "loss_history.npy",
    np.array(loss_history)
)