from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parent.parent

OUT = PROJECT / "outputs"

report = []

report.append("="*60)
report.append("AUTOMATIC FOLK MUSIC GENERATION")
report.append("="*60)
report.append("")

# ----------------------------
# Training Summary
# ----------------------------

summary = OUT / "training_summary.json"

if summary.exists():
    with open(summary) as f:
        s = json.load(f)

    report.append("TRAINING")
    report.append(f"Epochs : {s['epochs']}")
    report.append(f"Best Validation Loss : {s['best_val_loss']:.4f}")
    report.append("")

# ----------------------------
# Analysis
# ----------------------------

analysis = OUT / "analysis_results.json"

if analysis.exists():
    with open(analysis) as f:
        a = json.load(f)

    report.append("GENERATED MUSIC ANALYSIS")

    for k,v in a.items():
        report.append(f"{k} : {v}")

    report.append("")

# ----------------------------
# Raag
# ----------------------------

raag = OUT / "raag_analysis.json"

if raag.exists():

    with open(raag) as f:
        r = json.load(f)

    report.append("RAAG SIMILARITY")

    for k,v in r.items():
        report.append(f"{k:12s} : {v:.3f}")

    report.append("")

# ----------------------------
# Comparison
# ----------------------------

compare = OUT / "comparison_results.json"

if compare.exists():

    with open(compare) as f:
        c = json.load(f)

    report.append("ORIGINAL vs GENERATED")

    for k,v in c.items():
        report.append(f"{k} : {v}")

report.append("")
report.append("="*60)
report.append("Project Completed Successfully")
report.append("="*60)

with open(OUT/"final_report.txt","w") as f:
    f.write("\n".join(report))

print("Final report created.")