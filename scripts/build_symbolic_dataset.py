"""
Phase 2: Tokenization, Symbolic Augmentation & Dataset Construction
====================================================================
Takes the note-event JSON files from Phase 1 and:
1. Builds a vocabulary of combined (pitch, duration_bucket) tokens
2. Applies symbolic augmentation:
   - Pitch transposition (±1 to ±5 semitones)
   - Tempo scaling (0.85, 0.9, 1.0, 1.1, 1.15)
   - Overlapping fragment extraction (window=64 tokens, stride=32)
3. Tokenizes all augmented sequences
4. Creates train/val split (80/20 stratified by original song)
5. Saves tokenized dataset as .npz files

Design Rationale:
- Combined tokens (e.g. "P3_D4" = pitch +3 semitones, duration bucket 4)
  keep sequence length short (one token per note), which is critical
  for the small Transformer and for keeping training tractable.
- Duration buckets are log-spaced to match perception: humans perceive
  duration ratios, not absolute differences.
- Augmentation is essential: 45 songs yield ~2000-4000 note events total.
  With 11 transpositions × 5 tempo scales × fragmentation, we can reach
  ~50,000+ training sequences, sufficient for small models.
- Fragment length of 64 tokens ≈ 15-30 seconds of melody, enough context
  for the model to learn phrase-level structure.
"""

from pathlib import Path
import json
import numpy as np
from collections import Counter
import random

# =====================================================
# Paths
# =====================================================

PROJECT = Path(__file__).resolve().parent.parent

NOTE_DIR = PROJECT / "dataset" / "note_sequences"
TRAIN_DIR = PROJECT / "dataset" / "training"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Duration Bucketing
# =====================================================

# Log-spaced duration bins (in seconds)
# These roughly correspond to musical subdivisions at ~100 BPM:
# 0.05s (32nd note), 0.1s (16th), 0.15s, 0.2s (8th-ish),
# 0.3s, 0.45s (quarter-ish), 0.7s, 1.0s, 1.5s, 2.0s, 3.0s, 5.0s
DURATION_BINS = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.45,
                          0.70, 1.00, 1.50, 2.00, 3.00, 5.00])
NUM_DUR_BUCKETS = len(DURATION_BINS)

# =====================================================
# Augmentation Parameters
# =====================================================

TRANSPOSITIONS = list(range(-5, 6))  # -5 to +5 semitones (11 variants)
TEMPO_SCALES = [0.85, 0.9, 1.0, 1.1, 1.15]  # 5 tempo variants

# Fragment extraction
FRAGMENT_LENGTH = 64   # tokens per training sequence
FRAGMENT_STRIDE = 32   # overlap = 50%

# Special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
REST_TOKEN = "REST"


def duration_to_bucket(duration_sec):
    """
    Quantize a duration (seconds) to the nearest bucket index.
    Uses log-distance for perceptually uniform bucketing.
    """
    log_dur = np.log(max(duration_sec, 1e-4))
    log_bins = np.log(DURATION_BINS)
    idx = np.argmin(np.abs(log_bins - log_dur))
    return int(idx)


def make_token(pitch, dur_bucket):
    """
    Create a combined token string from pitch and duration bucket.
    Examples: "P3_D4", "REST_D2", "P-2_D7"
    """
    if pitch == "REST" or pitch == REST_TOKEN:
        return f"REST_D{dur_bucket}"
    else:
        return f"P{int(pitch)}_D{dur_bucket}"


def notes_to_tokens(notes):
    """
    Convert a list of note event dicts to a list of token strings.
    """
    tokens = []
    for note in notes:
        dur_bucket = duration_to_bucket(note["duration"])
        token = make_token(note["pitch"], dur_bucket)
        tokens.append(token)
    return tokens


def augment_transpose(notes, semitone_shift):
    """
    Transpose all pitched notes by a given number of semitones.
    REST events are unchanged.
    """
    augmented = []
    for note in notes:
        if note["pitch"] == "REST":
            augmented.append(note.copy())
        else:
            augmented.append({
                "pitch": note["pitch"] + semitone_shift,
                "duration": note["duration"]
            })
    return augmented


def augment_tempo(notes, scale_factor):
    """
    Scale all durations by a factor (e.g., 0.9 = slightly faster).
    """
    return [
        {"pitch": n["pitch"], "duration": n["duration"] * scale_factor}
        for n in notes
    ]


def extract_fragments(token_sequence, length=FRAGMENT_LENGTH, stride=FRAGMENT_STRIDE):
    """
    Extract overlapping fragments from a token sequence.
    Each fragment is a list of `length` tokens.
    Short sequences (< length) are padded.
    """
    fragments = []
    n = len(token_sequence)

    if n == 0:
        return fragments

    if n <= length:
        # Pad short sequences
        padded = token_sequence + [PAD_TOKEN] * (length - n)
        fragments.append(padded)
    else:
        for start in range(0, n - length + 1, stride):
            fragment = token_sequence[start:start + length]
            fragments.append(fragment)
        # Ensure last tokens are covered
        if (n - length) % stride != 0:
            fragment = token_sequence[n - length:n]
            fragments.append(fragment)

    return fragments


def build_vocabulary(all_token_sequences):
    """
    Build a token-to-index vocabulary from all token sequences.
    Includes special tokens: PAD, SOS, EOS.
    """
    counter = Counter()
    for seq in all_token_sequences:
        counter.update(seq)

    # Sort tokens for reproducibility
    sorted_tokens = sorted(counter.keys())

    # Build vocab with special tokens first
    vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
    idx = 3
    for token in sorted_tokens:
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    return vocab


def encode_sequence(token_sequence, vocab):
    """
    Convert a list of token strings to a list of integer indices.
    """
    return [vocab.get(t, vocab[PAD_TOKEN]) for t in token_sequence]


# =====================================================
# Main Pipeline
# =====================================================

if __name__ == "__main__":

    # 1. Load all note sequences
    json_files = sorted(NOTE_DIR.glob("MOI_*_notes.json"))
    print(f"Found {len(json_files)} note sequence files")

    if len(json_files) == 0:
        print("ERROR: No note sequence files found. Run extract_note_events.py first.")
        exit(1)

    song_data = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        song_data.append(data)

    total_notes = sum(len(s["notes"]) for s in song_data)
    print(f"Total note events across all songs: {total_notes}")

    # 2. Apply augmentation and tokenize
    print("\nApplying augmentation...")
    all_fragments = []       # list of token-string lists
    fragment_sources = []    # track which song each fragment came from

    for song in song_data:
        song_id = song["song_id"]
        notes = song["notes"]

        for transp in TRANSPOSITIONS:
            transposed = augment_transpose(notes, transp)

            for tempo_scale in TEMPO_SCALES:
                scaled = augment_tempo(transposed, tempo_scale)
                tokens = notes_to_tokens(scaled)
                fragments = extract_fragments(tokens)

                for frag in fragments:
                    all_fragments.append(frag)
                    fragment_sources.append(song_id)

    print(f"Total augmented fragments: {len(all_fragments)}")

    # 3. Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = build_vocabulary(all_fragments)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Save vocabulary
    vocab_path = TRAIN_DIR / "symbolic_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    # Inverse vocab for decoding
    inv_vocab = {v: k for k, v in vocab.items()}

    # 4. Encode all fragments
    print("\nEncoding fragments...")
    encoded = []
    for frag in all_fragments:
        enc = encode_sequence(frag, vocab)
        encoded.append(enc)

    encoded = np.array(encoded, dtype=np.int64)
    print(f"Encoded dataset shape: {encoded.shape}")

    # 5. Create input/target pairs for next-token prediction
    # Input: tokens[0:N-1], Target: tokens[1:N]
    X = encoded[:, :-1]  # input sequences
    Y = encoded[:, 1:]   # target sequences (shifted by 1)

    print(f"Input shape:  {X.shape}")
    print(f"Target shape: {Y.shape}")

    # 6. Train/Val split (80/20 by song)
    print("\nCreating train/val split...")
    unique_songs = sorted(set(fragment_sources))
    random.seed(42)
    random.shuffle(unique_songs)

    n_val = max(1, len(unique_songs) // 5)  # 20%
    val_songs = set(unique_songs[:n_val])
    train_songs = set(unique_songs[n_val:])

    train_mask = np.array([s not in val_songs for s in fragment_sources])
    val_mask = ~train_mask

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]

    print(f"Train: {X_train.shape[0]} sequences")
    print(f"Val:   {X_val.shape[0]} sequences")
    print(f"Val songs: {sorted(val_songs)}")

    # 7. Save datasets
    np.savez_compressed(
        TRAIN_DIR / "symbolic_train.npz",
        X=X_train, Y=Y_train
    )
    np.savez_compressed(
        TRAIN_DIR / "symbolic_val.npz",
        X=X_val, Y=Y_val
    )

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "seq_length": FRAGMENT_LENGTH - 1,
        "num_train": int(X_train.shape[0]),
        "num_val": int(X_val.shape[0]),
        "fragment_length": FRAGMENT_LENGTH,
        "fragment_stride": FRAGMENT_STRIDE,
        "transpositions": TRANSPOSITIONS,
        "tempo_scales": TEMPO_SCALES,
        "duration_bins": DURATION_BINS.tolist(),
        "val_songs": sorted(val_songs),
        "train_songs": sorted(train_songs),
        "total_notes_original": total_notes
    }

    with open(TRAIN_DIR / "symbolic_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print("Phase 2 Complete: Tokenization & Dataset Construction")
    print(f"{'='*60}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train sequences: {X_train.shape[0]}")
    print(f"Val sequences:   {X_val.shape[0]}")
    print(f"Sequence length: {FRAGMENT_LENGTH - 1}")
    print(f"Saved to: {TRAIN_DIR}")
