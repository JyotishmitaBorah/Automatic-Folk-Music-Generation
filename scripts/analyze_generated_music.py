from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# =====================================================
# Paths
# =====================================================

PROJECT = Path(__file__).resolve().parent.parent

ORIGINAL_DIR = PROJECT / "dataset" / "note_sequences"
GENERATED_DIR = PROJECT / "outputs" / "generated"

OUTPUT_DIR = PROJECT / "outputs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Original Dataset
# =====================================================

original_notes = []

for file in sorted(ORIGINAL_DIR.glob("*notes.json")):

    with open(file, "r") as f:

        data = json.load(f)

    original_notes.extend(data["notes"])

print("Original notes :", len(original_notes))

# =====================================================
# Load Generated Dataset
# =====================================================

generated_notes = []

for file in sorted(GENERATED_DIR.glob("*.json")):

    with open(file, "r") as f:

        data = json.load(f)

    if "notes" in data:
        generated_notes.extend(data["notes"])

print("Generated notes :", len(generated_notes))

# =====================================================
# Utility Functions
# =====================================================

def get_pitches(notes):

    pitches = []

    for n in notes:

        if n["pitch"] != "REST":
            pitches.append(int(n["pitch"]))

    return pitches


def get_durations(notes):

    return [float(n["duration"]) for n in notes]


def get_intervals(notes):

    pitches = get_pitches(notes)

    intervals = []

    for i in range(1, len(pitches)):

        intervals.append(
            pitches[i] - pitches[i-1]
        )

    return intervals


original_pitch = get_pitches(original_notes)
generated_pitch = get_pitches(generated_notes)

original_duration = get_durations(original_notes)
generated_duration = get_durations(generated_notes)

original_interval = get_intervals(original_notes)
generated_interval = get_intervals(generated_notes)

print("Original pitch count :", len(original_pitch))
print("Generated pitch count :", len(generated_pitch))

# =====================================================
# Pitch Histogram
# =====================================================

plt.figure(figsize=(10,5))

bins = np.arange(
    min(original_pitch + generated_pitch)-1,
    max(original_pitch + generated_pitch)+2
)

plt.hist(
    original_pitch,
    bins=bins,
    alpha=0.6,
    label="Original",
    color="steelblue"
)

plt.hist(
    generated_pitch,
    bins=bins,
    alpha=0.6,
    label="Generated",
    color="tomato"
)

plt.xlabel("Relative Pitch (Semitones)")
plt.ylabel("Count")
plt.title("Pitch Distribution")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR/"pitch_histogram.png",
    dpi=200
)

plt.close()


# =====================================================
# Duration Histogram
# =====================================================

plt.figure(figsize=(10,5))

plt.hist(
    original_duration,
    bins=20,
    alpha=0.6,
    label="Original",
    color="steelblue"
)

plt.hist(
    generated_duration,
    bins=20,
    alpha=0.6,
    label="Generated",
    color="tomato"
)

plt.xlabel("Duration (Seconds)")
plt.ylabel("Count")
plt.title("Duration Distribution")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR/"duration_histogram.png",
    dpi=200
)

plt.close()


# =====================================================
# Interval Histogram
# =====================================================

plt.figure(figsize=(10,5))

bins = np.arange(
    min(original_interval + generated_interval)-1,
    max(original_interval + generated_interval)+2
)

plt.hist(
    original_interval,
    bins=bins,
    alpha=0.6,
    label="Original",
    color="steelblue"
)

plt.hist(
    generated_interval,
    bins=bins,
    alpha=0.6,
    label="Generated",
    color="tomato"
)

plt.xlabel("Pitch Interval (Semitones)")
plt.ylabel("Count")
plt.title("Melodic Interval Distribution")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR/"interval_histogram.png",
    dpi=200
)

plt.close()

print("✓ Pitch Histogram Saved")
print("✓ Duration Histogram Saved")
print("✓ Interval Histogram Saved")

# =====================================================
# Melodic Contour
# =====================================================

plt.figure(figsize=(14,5))

orig_plot = original_pitch[:300]
gen_plot = generated_pitch[:300]

plt.plot(
    orig_plot,
    label="Original",
    linewidth=2,
    alpha=0.8
)

plt.plot(
    gen_plot,
    label="Generated",
    linewidth=2,
    alpha=0.8
)

plt.xlabel("Note Index")
plt.ylabel("Relative Pitch")
plt.title("Melodic Contour Comparison")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR/"melodic_contour.png",
    dpi=200
)

plt.close()

print("✓ Melodic Contour Saved")


# =====================================================
# Transition Matrix
# =====================================================

def transition_matrix(pitches):

    unique = sorted(list(set(pitches)))

    mapping = {
        p:i
        for i,p in enumerate(unique)
    }

    matrix = np.zeros(
        (len(unique),len(unique)),
        dtype=np.int32
    )

    for i in range(len(pitches)-1):

        a = mapping[pitches[i]]
        b = mapping[pitches[i+1]]

        matrix[a,b]+=1

    return matrix,unique


matrix,labels = transition_matrix(generated_pitch)

plt.figure(figsize=(8,8))

plt.imshow(
    matrix,
    cmap="viridis",
    interpolation="nearest"
)

plt.colorbar()

ticks=np.arange(len(labels))

plt.xticks(
    ticks,
    labels,
    rotation=90
)

plt.yticks(
    ticks,
    labels
)

plt.xlabel("Next Pitch")
plt.ylabel("Current Pitch")
plt.title("Generated Melody Transition Matrix")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR/"transition_matrix.png",
    dpi=200
)

plt.close()

print("✓ Transition Matrix Saved")


# =====================================================
# Statistics
# =====================================================

pitch_similarity = len(
    set(original_pitch).intersection(
        set(generated_pitch)
    )
)/len(
    set(original_pitch)
)

duration_difference = abs(
    np.mean(original_duration)
    -
    np.mean(generated_duration)
)

interval_difference = abs(
    np.mean(original_interval)
    -
    np.mean(generated_interval)
)

analysis = {

    "original_notes":len(original_notes),

    "generated_notes":len(generated_notes),

    "original_unique_pitches":len(set(original_pitch)),

    "generated_unique_pitches":len(set(generated_pitch)),

    "pitch_similarity":round(
        float(pitch_similarity),
        4
    ),

    "mean_original_duration":round(
        float(np.mean(original_duration)),
        4
    ),

    "mean_generated_duration":round(
        float(np.mean(generated_duration)),
        4
    ),

    "duration_difference":round(
        float(duration_difference),
        4
    ),

    "mean_original_interval":round(
        float(np.mean(original_interval)),
        4
    ),

    "mean_generated_interval":round(
        float(np.mean(generated_interval)),
        4
    ),

    "interval_difference":round(
        float(interval_difference),
        4
    )
}

with open(
    OUTPUT_DIR/"analysis.json",
    "w"
) as f:

    json.dump(
        analysis,
        f,
        indent=4
    )

print("\n==============================")
print("Analysis Complete")
print("==============================")

print(json.dumps(
    analysis,
    indent=4
))

print("\nFigures saved in:")

print(OUTPUT_DIR)