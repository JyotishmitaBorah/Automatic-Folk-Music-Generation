
from pathlib import Path
import sys
import json
import time
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT))

from models.symbolic_models import FolkLSTMv2

# ----------------------------------------------------
# Paths
# ----------------------------------------------------

TRAIN_DIR = PROJECT / "dataset" / "training"
MODEL_DIR = PROJECT / "models"
OUTPUT_DIR = PROJECT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------
# Dataset
# ----------------------------------------------------

class SymbolicDataset(Dataset):

    def __init__(self, X, Y):
        self.X = torch.LongTensor(X)
        self.Y = torch.LongTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ----------------------------------------------------
# Training
# ----------------------------------------------------

def train(model,
          train_loader,
          val_loader,
          device,
          epochs,
          lr):

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    best_loss = 9999

    train_losses = []
    val_losses = []

    checkpoint_path = MODEL_DIR / "symbolic_LSTM_checkpoint.pth"
    best_path = MODEL_DIR / "symbolic_LSTM_best.pth"

    for epoch in range(epochs):

        start = time.time()

        # ---------------- TRAIN ----------------

        model.train()

        running = 0

        for x,y in train_loader:

            x=x.to(device)
            y=y.to(device)

            optimizer.zero_grad()

            out,_=model(x)

            loss=criterion(
                out.reshape(-1,out.size(-1)),
                y.reshape(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5
            )

            optimizer.step()

            running+=loss.item()

        train_loss=running/len(train_loader)

        train_losses.append(train_loss)

        # ---------------- VALIDATION ----------------

        model.eval()

        running=0

        with torch.no_grad():

            for x,y in val_loader:

                x=x.to(device)
                y=y.to(device)

                out,_=model(x)

                loss=criterion(
                    out.reshape(-1,out.size(-1)),
                    y.reshape(-1)
                )

                running+=loss.item()

        val_loss=running/len(val_loader)

        val_losses.append(val_loss)

        if val_loss<best_loss:

            best_loss=val_loss

            torch.save(
                {
                    "model_state_dict":model.state_dict(),
                    "loss":best_loss
                },
                best_path
            )

        torch.save(
            {
                "model_state_dict":model.state_dict(),
                "loss":val_loss
            },
            checkpoint_path
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"{time.time()-start:.1f}s"
        )

    return train_losses,val_losses,best_loss


# ----------------------------------------------------
# Main
# ----------------------------------------------------

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--lr",type=float,default=0.001)

    args=parser.parse_args()

    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(device)

    train_data = np.load(TRAIN_DIR/"symbolic_train.npz")
    val_data = np.load(TRAIN_DIR/"symbolic_val.npz")
    X_train = train_data["X"]
    Y_train = train_data["Y"]

    X_val = val_data["X"]
    Y_val = val_data["Y"]

    print(X_train.shape)
    print(X_val.shape)

    with open(TRAIN_DIR/"symbolic_vocab.json") as f:
        vocab=json.load(f)

    vocab_size=len(vocab)

    train_loader=DataLoader(
        SymbolicDataset(X_train,Y_train),
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader=DataLoader(
        SymbolicDataset(X_val,Y_val),
        batch_size=args.batch_size
    )

    model=FolkLSTMv2(vocab_size).to(device)

    train_losses,val_losses,best=train(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr
    )

    plt.figure(figsize=(8,5))

    plt.plot(train_losses,label="Train")

    plt.plot(val_losses,label="Validation")

    plt.legend()

    plt.grid()

    plt.savefig(
        OUTPUT_DIR/"symbolic_training_curve.png",
        dpi=200
    )

    with open(
        OUTPUT_DIR/"training_summary.json",
        "w"
    ) as f:

        json.dump(
            {
                "best_val_loss":float(best),
                "epochs":args.epochs
            },
            f,
            indent=2
        )

    print("\nTraining Finished")
