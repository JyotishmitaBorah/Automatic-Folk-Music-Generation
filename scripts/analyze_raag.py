"""
Phase 5: Raag/Svara Analysis & Model Comparison Report
=======================================================
1. Analyzes original melodies using pre-computed chroma/pitch features:
   - Pitch-class histograms (svara distribution)
   - Interval transition matrices
   - Most common motifs (n-grams)
2. Performs the same analysis on generated melodies
3. Computes similarity metrics (KL divergence, cosine similarity)
4. Produces a comprehensive comparison report

Uses ALREADY EXTRACTED chroma features from dataset/features/chroma/
(as specified — no re-extraction from scratch).
"""

from pathlib import Path
import sys
import json
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent

# =====================================================
# Paths
# =====================================================

CHROMA_DIR    = PROJECT / "dataset" / "features" / "chroma"
NOTE_DIR      = PROJECT / "dataset" / "note_sequences"
GEN_DIR       = PROJECT / "outputs" / "generated_symbolic"
OUTPUT_DIR    = PROJECT / "outputs" / "analysis"
TRAIN_DIR     = PROJECT / "dataset" / "training"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Svara names for the 12 semitones relative to Sa
SVARA_NAMES = ["Sa", "Re♭", "Re", "Ga♭", "Ga", "Ma", "Ma♯",
               "Pa", "Dha♭", "Dha", "Ni♭", "Ni"]


# =====================================================
# Analysis Functions
# =====================================================

def pitch_class_histogram(notes, normalize=True):
    """
    Build a 12-bin pitch-class histogram from note events.
    Each note's relative pitch mod 12 maps to a svara.
    Weighted by duration (longer notes contribute more).
    """
    hist = np.zeros(12)
    for note in notes:
        if note["pitch"] == "REST":
            continue
        pc = int(note["pitch"]) % 12
        hist[pc] += note["duration"]

    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()

    return hist


def chroma_to_histogram(chroma_npy, normalize=True):
    """
    Convert a pre-computed chroma feature matrix (12, T) to a
    pitch-class histogram by averaging across time.
    """
    hist = chroma_npy.mean(axis=1)  # average energy per pitch class
    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


def interval_histogram(notes, max_interval=12):
    """
    Build a histogram of melodic intervals (pitch differences between
    consecutive notes). Useful for characterizing melodic movement.
    """
    intervals = []
    prev_pitch = None
    for note in notes:
        if note["pitch"] == "REST":
            prev_pitch = None
            continue
        if prev_pitch is not None:
            interval = int(note["pitch"]) - prev_pitch
            intervals.append(interval)
        prev_pitch = int(note["pitch"])

    # Build histogram from -max_interval to +max_interval
    bins = list(range(-max_interval, max_interval + 1))
    hist = np.zeros(len(bins))
    for iv in intervals:
        clamped = max(-max_interval, min(max_interval, iv))
        idx = clamped + max_interval
        hist[idx] += 1

    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist, bins


def extract_ngrams(notes, n=3):
    """
    Extract pitch n-grams (motifs) from a note sequence.
    Returns a Counter of the most common patterns.
    """
    pitches = [note["pitch"] for note in notes if note["pitch"] != "REST"]
    ngrams = []
    for i in range(len(pitches) - n + 1):
        gram = tuple(pitches[i:i+n])
        ngrams.append(gram)
    return Counter(ngrams)


def kl_divergence(p, q, epsilon=1e-10):
    """
    KL divergence D_KL(P || Q).
    Measures how different Q is from P (original).
    Lower = more similar.
    """
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def transition_matrix(notes, n_classes=12):
    """
    Build a pitch-class transition matrix: T[i][j] = P(next_pc = j | current_pc = i)
    """
    T = np.zeros((n_classes, n_classes))
    prev_pc = None
    for note in notes:
        if note["pitch"] == "REST":
            prev_pc = None
            continue
        pc = int(note["pitch"]) % 12
        if prev_pc is not None:
            T[prev_pc][pc] += 1
        prev_pc = pc

    # Normalize rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums

    return T


# =====================================================
# Main Analysis
# =====================================================

if __name__ == "__main__":
    print("="*60)
    print("Phase 5: Raag/Svara Analysis")
    print("="*60)

    # ----- 1. Analyze Original Melodies -----
    print("\n--- Analyzing Original Melodies ---")

    # Load note sequences
    note_files = sorted(NOTE_DIR.glob("MOI_*_notes.json"))
    print(f"Found {len(note_files)} original note sequences")

    if len(note_files) == 0:
        print("ERROR: No note sequences found. Run extract_note_events.py first.")
        sys.exit(1)

    all_original_notes = []
    for nf in note_files:
        with open(nf, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_original_notes.extend(data["notes"])

    print(f"Total original note events: {len(all_original_notes)}")

    # Pitch-class histogram from note events
    orig_pc_hist = pitch_class_histogram(all_original_notes)

    # Pitch-class histogram from pre-computed chroma features
    chroma_files = sorted(CHROMA_DIR.glob("MOI_*.npy"))
    print(f"Found {len(chroma_files)} chroma feature files")

    orig_chroma_hist = np.zeros(12)
    for cf in chroma_files:
        chroma = np.load(cf)
        orig_chroma_hist += chroma.mean(axis=1)
    orig_chroma_hist = orig_chroma_hist / orig_chroma_hist.sum()

    # Interval histogram
    orig_interval_hist, interval_bins = interval_histogram(all_original_notes)

    # Transition matrix
    orig_transition = transition_matrix(all_original_notes)

    # Common motifs
    orig_motifs_3 = extract_ngrams(all_original_notes, n=3)
    orig_motifs_4 = extract_ngrams(all_original_notes, n=4)

    # ----- 2. Analyze Generated Melodies -----
    print("\n--- Analyzing Generated Melodies ---")

    model_names = ["rnn", "lstm", "transformer"]
    gen_results = {}

    for model_name in model_names:
        model_dir = GEN_DIR / model_name
        if not model_dir.exists():
            print(f"  {model_name}: no generated samples found, skipping")
            continue

        gen_files = sorted(model_dir.glob("*_notes.json"))
        if len(gen_files) == 0:
            print(f"  {model_name}: no note JSONs found, skipping")
            continue

        print(f"  {model_name}: {len(gen_files)} samples")

        all_gen_notes = []
        for gf in gen_files:
            with open(gf, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_gen_notes.extend(data["notes"])

        gen_pc_hist = pitch_class_histogram(all_gen_notes)
        gen_interval_hist, _ = interval_histogram(all_gen_notes)
        gen_transition = transition_matrix(all_gen_notes)
        gen_motifs_3 = extract_ngrams(all_gen_notes, n=3)

        # Compute metrics
        pc_kl = kl_divergence(orig_pc_hist, gen_pc_hist)
        pc_cos = cosine_similarity(orig_pc_hist, gen_pc_hist)
        iv_kl = kl_divergence(orig_interval_hist, gen_interval_hist)
        iv_cos = cosine_similarity(orig_interval_hist, gen_interval_hist)

        # Transition matrix similarity (Frobenius norm of difference)
        trans_diff = np.linalg.norm(orig_transition - gen_transition, 'fro')

        # Motif overlap: fraction of top-20 original motifs found in generated
        top_orig_motifs = set([m for m, _ in orig_motifs_3.most_common(20)])
        gen_motif_set = set(gen_motifs_3.keys())
        motif_overlap = len(top_orig_motifs & gen_motif_set) / max(len(top_orig_motifs), 1)

        gen_results[model_name] = {
            "pc_histogram": gen_pc_hist.tolist(),
            "interval_histogram": gen_interval_hist.tolist(),
            "kl_pitch_class": round(pc_kl, 4),
            "cosine_pitch_class": round(pc_cos, 4),
            "kl_interval": round(iv_kl, 4),
            "cosine_interval": round(iv_cos, 4),
            "transition_frobenius": round(trans_diff, 4),
            "motif_overlap_top20": round(motif_overlap, 4),
            "num_notes": len(all_gen_notes),
            "num_unique_pitches": len(set(n["pitch"] for n in all_gen_notes
                                          if n["pitch"] != "REST"))
        }

        print(f"    PC KL: {pc_kl:.4f}, PC Cos: {pc_cos:.4f}, "
              f"IV KL: {iv_kl:.4f}, Motif: {motif_overlap:.2%}")

    # ----- 3. Load Training Results -----
    training_results_path = PROJECT / "outputs" / "symbolic_training_results.json"
    training_results = {}
    if training_results_path.exists():
        with open(training_results_path, "r") as f:
            tr_list = json.load(f)
        for r in tr_list:
            training_results[r["model"].lower()] = r

    # ----- 4. Create Visualizations -----
    print("\n--- Creating Visualizations ---")

    # 4a. Pitch-class distribution comparison
    fig, axes = plt.subplots(1, len(gen_results) + 1, figsize=(5 * (len(gen_results) + 1), 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Original
    ax = axes[0]
    bars = ax.bar(range(12), orig_pc_hist, color="#4CAF50", alpha=0.8)
    ax.set_xticks(range(12))
    ax.set_xticklabels(SVARA_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_title("Original\n(All Songs)", fontweight="bold")
    ax.set_ylabel("Relative Frequency")
    ax.set_ylim(0, max(orig_pc_hist) * 1.3)

    # Generated
    colors = {"rnn": "#2196F3", "lstm": "#FF9800", "transformer": "#E91E63"}
    for i, (model_name, res) in enumerate(gen_results.items()):
        ax = axes[i + 1]
        hist = np.array(res["pc_histogram"])
        ax.bar(range(12), hist, color=colors.get(model_name, "#999"),
               alpha=0.8)
        ax.bar(range(12), orig_pc_hist, color="#4CAF50", alpha=0.3,
               label="Original")
        ax.set_xticks(range(12))
        ax.set_xticklabels(SVARA_NAMES, rotation=45, ha="right", fontsize=8)
        title = f"{model_name.upper()}\nKL={res['kl_pitch_class']:.3f}"
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylim(0, max(max(hist), max(orig_pc_hist)) * 1.3)

    plt.suptitle("Svara (Pitch-Class) Distribution: Original vs Generated",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "svara_distribution_comparison.png", dpi=200)
    plt.close()

    # 4b. Interval distribution comparison
    fig, axes = plt.subplots(1, len(gen_results) + 1,
                             figsize=(5 * (len(gen_results) + 1), 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    ax = axes[0]
    ax.bar(interval_bins, orig_interval_hist, color="#4CAF50", alpha=0.8)
    ax.set_xlabel("Interval (semitones)")
    ax.set_title("Original", fontweight="bold")
    ax.set_ylabel("Relative Frequency")

    for i, (model_name, res) in enumerate(gen_results.items()):
        ax = axes[i + 1]
        hist = np.array(res["interval_histogram"])
        ax.bar(interval_bins, hist, color=colors.get(model_name, "#999"),
               alpha=0.8)
        ax.bar(interval_bins, orig_interval_hist, color="#4CAF50", alpha=0.3)
        ax.set_xlabel("Interval (semitones)")
        ax.set_title(f"{model_name.upper()}\nKL={res['kl_interval']:.3f}",
                     fontweight="bold")

    plt.suptitle("Melodic Interval Distribution: Original vs Generated",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interval_distribution_comparison.png", dpi=200)
    plt.close()

    # 4c. Transition matrices
    fig, axes = plt.subplots(1, min(len(gen_results) + 1, 4),
                             figsize=(6 * min(len(gen_results) + 1, 4), 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    ax = axes[0]
    im = ax.imshow(orig_transition, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels(SVARA_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(SVARA_NAMES, fontsize=7)
    ax.set_title("Original", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    for i, model_name in enumerate(gen_results.keys()):
        if i + 1 >= len(axes):
            break
        ax = axes[i + 1]
        # Recompute transition for this model
        model_dir = GEN_DIR / model_name
        gen_files = sorted(model_dir.glob("*_notes.json"))
        all_notes = []
        for gf in gen_files:
            with open(gf, "r") as f:
                data = json.load(f)
            all_notes.extend(data["notes"])
        gen_trans = transition_matrix(all_notes)
        im = ax.imshow(gen_trans, cmap="YlOrRd", aspect="equal")
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        ax.set_xticklabels(SVARA_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(SVARA_NAMES, fontsize=7)
        ax.set_title(f"{model_name.upper()}", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Pitch-Class Transition Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transition_matrix_comparison.png", dpi=200)
    plt.close()

    # 4d. Chroma vs Note-based histogram comparison (validation)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(12)
    ax1.bar(x - 0.15, orig_pc_hist, 0.3, label="From Note Events",
            color="#2196F3", alpha=0.8)
    ax1.bar(x + 0.15, orig_chroma_hist, 0.3, label="From Chroma Features",
            color="#FF5722", alpha=0.8)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(SVARA_NAMES, rotation=45, ha="right")
    ax1.set_title("Original Pitch-Class: Note Events vs Chroma", fontweight="bold")
    ax1.set_ylabel("Relative Frequency")
    ax1.legend()

    # Dominant svaras
    sorted_svaras = sorted(enumerate(orig_pc_hist), key=lambda x: -x[1])
    top5 = sorted_svaras[:5]
    ax2.barh([SVARA_NAMES[i] for i, _ in top5],
             [v for _, v in top5], color="#4CAF50")
    ax2.set_title("Top 5 Dominant Svaras\n(Mising Oi Nitom)", fontweight="bold")
    ax2.set_xlabel("Relative Frequency")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "original_svara_analysis.png", dpi=200)
    plt.close()

    # ----- 5. Save Complete Report -----
    print("\n--- Saving Report ---")

    report = {
        "original_analysis": {
            "num_songs": len(note_files),
            "total_notes": len(all_original_notes),
            "pitch_class_histogram": orig_pc_hist.tolist(),
            "pitch_class_histogram_chroma": orig_chroma_hist.tolist(),
            "dominant_svaras": [
                {"svara": SVARA_NAMES[i], "weight": round(v, 4)}
                for i, v in sorted(enumerate(orig_pc_hist), key=lambda x: -x[1])[:5]
            ],
            "top_motifs_3gram": [
                {"motif": list(m), "count": c}
                for m, c in orig_motifs_3.most_common(15)
            ],
            "top_motifs_4gram": [
                {"motif": list(m), "count": c}
                for m, c in orig_motifs_4.most_common(10)
            ]
        },
        "generated_analysis": gen_results,
        "training_metrics": {}
    }

    # Add training metrics
    for model_name in model_names:
        if model_name in training_results:
            tr = training_results[model_name]
            report["training_metrics"][model_name] = {
                "best_val_loss": tr.get("best_val_loss"),
                "best_val_ppl": tr.get("best_val_ppl"),
                "final_train_loss": tr.get("final_train_loss"),
                "num_params": tr.get("num_params"),
                "total_time_sec": tr.get("total_time_sec")
            }

    with open(OUTPUT_DIR / "analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ----- 6. Print Summary Table -----
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print(f"{'='*80}")

    print(f"\n--- Original Corpus ---")
    print(f"Songs: {len(note_files)}")
    print(f"Total notes: {len(all_original_notes)}")
    print(f"Dominant svaras: ", end="")
    for i, v in sorted(enumerate(orig_pc_hist), key=lambda x: -x[1])[:5]:
        print(f"{SVARA_NAMES[i]}({v:.3f}) ", end="")
    print()

    print(f"\n{'Model':<14} {'Val Loss':>9} {'Val PPL':>9} {'PC KL':>8} "
          f"{'PC Cos':>8} {'IV KL':>8} {'Motif%':>8} {'Params':>10}")
    print("-" * 80)

    for model_name in model_names:
        vl = "--"
        vp = "--"
        params = "--"
        if model_name in training_results:
            tr = training_results[model_name]
            vl = f"{tr['best_val_loss']:.4f}"
            vp = f"{tr['best_val_ppl']:.1f}"
            params = f"{tr['num_params']:,}"

        if model_name in gen_results:
            gr = gen_results[model_name]
            print(f"{model_name.upper():<14} {vl:>9} {vp:>9} "
                  f"{gr['kl_pitch_class']:>8.4f} {gr['cosine_pitch_class']:>8.4f} "
                  f"{gr['kl_interval']:>8.4f} {gr['motif_overlap_top20']:>7.1%} "
                  f"{params:>10}")
        else:
            print(f"{model_name.upper():<14} {vl:>9} {vp:>9} {'--':>8} {'--':>8} "
                  f"{'--':>8} {'--':>8} {params:>10}")

    print(f"\n{'='*80}")
    print("Analysis complete! Results saved to:", OUTPUT_DIR)
    print(f"{'='*80}")
