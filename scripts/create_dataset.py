from pathlib import Path
import numpy as np
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent.parent

MFCC_DIR = PROJECT / "dataset" / "segment_features" / "mfcc"
OUTPUT_DIR = PROJECT / "dataset" / "training"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(MFCC_DIR.glob("*.npy"))

print(f"Found {len(files)} MFCC files")

X = []
filenames = []

TARGET_FRAMES = 431

for file in tqdm(files):

    mfcc = np.load(file)

    # Pad or trim to fixed length
    if mfcc.shape[1] > TARGET_FRAMES:
        mfcc = mfcc[:, :TARGET_FRAMES]

    elif mfcc.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")

    X.append(mfcc.astype(np.float32))
    filenames.append(file.stem)

X = np.array(X)

print("Dataset shape:", X.shape)

np.savez_compressed(
    OUTPUT_DIR / "train_dataset.npz",
    X=X,
    filenames=np.array(filenames)
)

print("Training dataset created successfully.")