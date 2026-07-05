from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parent.parent

OUTPUT = PROJECT / "outputs"
GEN = OUTPUT / "generated_symbolic"

FINAL = PROJECT / "Final_Project_Output"

FINAL.mkdir(exist_ok=True)

# --------------------------------------------------
# Copy analysis files
# --------------------------------------------------

files = [
    "analysis_results.json",
    "comparison_results.json",
    "raag_analysis.json",
    "training_summary.json",
    "final_report.txt",
    "pitch_distribution_comparison.png",
    "pitch_class_distribution.png",
    "raag_similarity.png",
    "symbolic_training_curve.png"
]

for f in files:

    src = OUTPUT / f

    if src.exists():

        shutil.copy(src, FINAL / src.name)

# --------------------------------------------------
# Copy generated files
# --------------------------------------------------

GEN_OUT = FINAL / "Generated_Music"

GEN_OUT.mkdir(exist_ok=True)

for ext in ["*.json","*.mid","*.wav"]:

    for f in GEN.glob(ext):

        shutil.copy(f, GEN_OUT / f.name)

print("="*50)
print("Project packaged successfully.")
print("Folder created:")
print(FINAL)
print("="*50)