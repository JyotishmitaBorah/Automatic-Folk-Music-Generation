from pathlib import Path
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine

PROJECT = Path(__file__).resolve().parent.parent

generated = np.load(
    PROJECT / "outputs" / "generated" / "generated_sequence.npy"
)

original_data = np.load(
    PROJECT / "dataset" / "training" / "train_dataset_prediction.npz",
    allow_pickle=True
)

original = original_data["X"][0]

# Compare only the original-length portion
generated_part = generated[:, :original.shape[1]]

print("Original Shape :", original.shape)
print("Generated Shape:", generated_part.shape)

euclidean_distance = euclidean(
    original.flatten(),
    generated_part.flatten()
)

cosine_similarity = 1 - cosine(
    original.flatten(),
    generated_part.flatten()
)

correlation = np.corrcoef(
    original.flatten(),
    generated_part.flatten()
)[0,1]

print("\n========== Evaluation ==========")
print(f"Euclidean Distance : {euclidean_distance:.4f}")
print(f"Cosine Similarity  : {cosine_similarity:.4f}")
print(f"Correlation        : {correlation:.4f}")