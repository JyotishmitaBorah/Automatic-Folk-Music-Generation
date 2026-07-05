from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent

TRAIN = PROJECT/"dataset"/"training"
GEN = PROJECT/"outputs"/"generated_symbolic"
OUT = PROJECT/"outputs"

# -----------------------
# Original pitches
# -----------------------

train=np.load(TRAIN/"symbolic_train.npz")

X=train["X"]

original=[]

for seq in X:
    for t in seq:
        if t>2:
            original.append(int(t))

# -----------------------
# Generated pitches
# -----------------------

generated=[]

for f in sorted(GEN.glob("sample_*.json")):

    with open(f) as fp:
        data=json.load(fp)

    for n in data["notes"]:

        if n["pitch"]!="REST":
            generated.append(int(n["pitch"]))

# -----------------------
# Histogram
# -----------------------

plt.figure(figsize=(10,5))

plt.hist(
    original,
    bins=40,
    alpha=0.6,
    label="Original"
)

plt.hist(
    generated,
    bins=40,
    alpha=0.6,
    label="Generated"
)

plt.legend()

plt.title("Pitch Distribution Comparison")

plt.xlabel("Pitch")

plt.ylabel("Frequency")

plt.grid(alpha=.3)

plt.tight_layout()

plt.savefig(
    OUT/"pitch_distribution_comparison.png",
    dpi=200
)

plt.close()

# -----------------------
# Statistics
# -----------------------

report={

"original_notes":len(original),

"generated_notes":len(generated),

"original_mean":float(np.mean(original)),

"generated_mean":float(np.mean(generated)),

"original_std":float(np.std(original)),

"generated_std":float(np.std(generated))

}

with open(
    OUT/"comparison_results.json",
    "w"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )

print(report)
