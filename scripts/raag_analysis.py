from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import Counter

PROJECT = Path(__file__).resolve().parent.parent

# Generated samples are here
INPUT_DIR = PROJECT / "outputs" / "generated_symbolic"

# Save analysis here
OUTPUT_DIR = PROJECT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

json_files = sorted(INPUT_DIR.glob("sample_*.json"))

if len(json_files) == 0:
    print("No generated json files found.")
    exit()

all_pitches = []

for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)

    for note in data["notes"]:
        if note["pitch"] != "REST":
            all_pitches.append(int(note["pitch"]))

print("Total Notes:", len(all_pitches))

pitch_classes = [p % 12 for p in all_pitches]

counter = Counter(pitch_classes)

counts = [counter.get(i, 0) for i in range(12)]

labels = [
    "C","C#","D","D#","E","F",
    "F#","G","G#","A","A#","B"
]

plt.figure(figsize=(8,5))
plt.bar(labels, counts)
plt.title("Pitch Class Distribution")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"pitch_class_distribution.png", dpi=200)
plt.close()

raags = {
    "Bilawal":[0,2,4,5,7,9,11],
    "Kafi":[0,2,3,5,7,9,10],
    "Bhairav":[0,1,4,5,7,8,11],
    "Kalyan":[0,2,4,6,7,9,11],
    "Asavari":[0,2,3,5,7,8,10]
}

generated = set(pitch_classes)

similarity = {}

for name, notes in raags.items():
    similarity[name] = len(generated.intersection(notes))/7

plt.figure(figsize=(8,5))
plt.bar(similarity.keys(), similarity.values())
plt.ylabel("Similarity")
plt.title("Generated Melody vs Raag Profiles")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"raag_similarity.png", dpi=200)
plt.close()

with open(OUTPUT_DIR/"raag_analysis.json","w") as f:
    json.dump(similarity,f,indent=4)

print("\nFinished.")
print(similarity)