from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt

# =====================================================
# Project Path
# =====================================================

PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT))

from models.lstm_model import FolkLSTM

# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# =====================================================
# Load Dataset
# =====================================================

data = np.load(
    PROJECT / "dataset" / "training" / "train_dataset_prediction.npz",
    allow_pickle=True
)

X = data["X"]
Y = data["Y"]

print("Input Shape :", X.shape)
print("Target Shape:", Y.shape)

# =====================================================
# Dataset
# =====================================================

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

print("Dataset Loaded Successfully!")

# =====================================================
# Model
# =====================================================

model = FolkLSTM().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =====================================================
# Paths
# =====================================================

MODEL_DIR = PROJECT / "models"
MODEL_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = PROJECT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT = MODEL_DIR / "prediction_checkpoint.pth"
BEST_MODEL = MODEL_DIR / "prediction_lstm_best.pth"

# =====================================================
# Resume Training
# =====================================================

TOTAL_EPOCHS = 100

start_epoch = 0
best_loss = float("inf")
loss_history = []

if CHECKPOINT.exists():

    print("\nCheckpoint Found!")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_loss = checkpoint["best_loss"]

    loss_history = checkpoint["loss_history"]

    print(f"Resuming from Epoch {start_epoch + 1}")

else:

    print("\nNo checkpoint found.")
    print("Starting from Epoch 1")

# =====================================================
# Training
# =====================================================

print("\nTraining Started...\n")

try:

    for epoch in range(start_epoch, TOTAL_EPOCHS):

        model.train()

        running_loss = 0.0

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(
                output,
                y.permute(0, 2, 1)
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)

        loss_history.append(epoch_loss)

        print(
            f"Epoch {epoch+1:03d}/{TOTAL_EPOCHS} | Loss = {epoch_loss:.6f}"
        )

        # ----------------------------------------
        # Save Best Model
        # ----------------------------------------

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "loss_history": loss_history
                },
                BEST_MODEL
            )

        # ----------------------------------------
        # Save Checkpoint EVERY EPOCH
        # ----------------------------------------

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "loss_history": loss_history
            },
            CHECKPOINT
        )

except KeyboardInterrupt:

    print("\nTraining Interrupted!")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "loss_history": loss_history
        },
        CHECKPOINT
    )

    print("Checkpoint Saved Successfully!")

# =====================================================
# Save Graph
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(loss_history)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Prediction Model Training Loss")

plt.grid(True)

plt.savefig(
    OUTPUT_DIR / "prediction_training_loss.png",
    dpi=300
)

plt.close()

np.save(
    OUTPUT_DIR / "prediction_loss_history.npy",
    np.array(loss_history)
)

print("\nTraining Graph Saved!")

print("\nBest Loss :", best_loss)
print("\nDone!")