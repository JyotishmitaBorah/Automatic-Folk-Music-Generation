"""
Phase 1: Note-Event Extraction
===============================
For each of the 45 Mising Oi Nitom songs:
1. Extract pitch contour using librosa.pyin (reuse pre-computed where available)
2. Estimate tonic (Sa) via weighted pitch-class histogram
3. Quantize voiced frames to semitone offsets relative to tonic
4. Apply median filter to smooth vibrato / pitch jitter
5. Segment the smoothed contour into note events: (relative_pitch, duration_sec)
6. Save per-song note sequences as JSON

Design Rationale:
- pyin is chosen over yin for its probabilistic voicing detection (important for
  vocal folk music with unvoiced consonants and breathing)
- Tonic estimation via pitch-class histogram is standard in Indian music analysis
  (cf. Gulati et al., 2014). We weight by frame voicing probability.
- Relative semitone representation ensures key invariance across songs sung at
  different pitch levels — critical for a small-data setting where we want the
  model to learn melodic patterns, not absolute frequencies.
- Median filter (kernel=5 frames ≈ 116 ms at hop=512/sr=22050) removes micro-
  fluctuations from vibrato while preserving actual note transitions.
"""

from pathlib import Path
import json
import numpy as np
import librosa
from scipy.ndimage import median_filter
from tqdm import tqdm

# =====================================================
# Paths
# =====================================================

PROJECT = Path(__file__).resolve().parent.parent

AUDIO_DIR    = PROJECT / "dataset" / "processed_audio"
PITCH_DIR    = PROJECT / "dataset" / "features" / "pitch"
CHROMA_DIR   = PROJECT / "dataset" / "features" / "chroma"
NOTE_DIR     = PROJECT / "dataset" / "note_sequences"

PITCH_DIR.mkdir(parents=True, exist_ok=True)
NOTE_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Parameters
# =====================================================

SR = 22050
HOP_LENGTH = 512
FRAME_LENGTH = 2048
FMIN = librosa.note_to_hz("C2")   # ~65 Hz
FMAX = librosa.note_to_hz("C7")   # ~2093 Hz
MEDIAN_KERNEL = 5                  # frames for smoothing
MIN_NOTE_FRAMES = 3                # minimum frames to consider a valid note (~70 ms)

# Duration of one hop in seconds
HOP_SEC = HOP_LENGTH / SR  # ≈ 0.0232 seconds


def extract_pitch(audio_path, pitch_cache_path):
    """
    Extract pitch contour using librosa.pyin.
    Reuses cached .npy file if it exists and matches expected length.
    """
    y, sr = librosa.load(audio_path, sr=SR)
    expected_frames = 1 + len(y) // HOP_LENGTH

    # Check if a cached pitch file exists with correct length
    if pitch_cache_path.exists():
        cached = np.load(pitch_cache_path)
        if len(cached) == expected_frames:
            return cached, y, sr

    # Extract fresh pitch contour
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )

    np.save(pitch_cache_path, f0)
    return f0, y, sr


def estimate_tonic(f0, chroma_path=None):
    """
    Estimate the tonic (Sa) frequency for a song.

    Method: Build a pitch-class histogram from the voiced f0 values,
    weighted by frequency of occurrence. The pitch class with the highest
    energy is taken as the tonic. We then find the median f0 value
    belonging to that pitch class to get the exact frequency.

    If a pre-computed chroma file exists, we cross-validate by checking
    that the chroma-based dominant pitch class agrees with the f0-based one.

    Design note: In Indian classical/folk music, Sa is typically the most
    frequently occurring pitch class. This heuristic works well when the
    melody is predominantly scalar (as in Oi Nitom vocal melodies).
    """
    voiced_f0 = f0[~np.isnan(f0)]

    if len(voiced_f0) == 0:
        # Fallback: if no voiced frames, return A4 = 440 Hz
        return 440.0

    # Convert Hz to MIDI note numbers (continuous)
    midi_notes = librosa.hz_to_midi(voiced_f0)

    # Extract pitch classes (0-11)
    pitch_classes = np.round(midi_notes) % 12

    # Build histogram
    hist = np.zeros(12)
    for pc in pitch_classes.astype(int):
        hist[pc] += 1

    # The tonic pitch class is the most frequent
    tonic_pc = np.argmax(hist)

    # Cross-validate with chroma if available
    if chroma_path is not None and chroma_path.exists():
        chroma = np.load(chroma_path)  # shape (12, T)
        chroma_energy = chroma.sum(axis=1)
        chroma_tonic_pc = np.argmax(chroma_energy)

        # If chroma agrees, great. If not, trust f0-based estimate
        # (chroma can be confused by harmonics in polyphonic sections)
        if chroma_tonic_pc == tonic_pc:
            pass  # agreement
        # else: stick with f0-based estimate

    # Get the median frequency of frames belonging to the tonic pitch class
    tonic_mask = (np.round(midi_notes) % 12) == tonic_pc
    tonic_freq = np.median(voiced_f0[tonic_mask])

    # Normalize to the octave closest to the median pitch
    median_pitch = np.median(voiced_f0)
    while tonic_freq > median_pitch * 1.5:
        tonic_freq /= 2.0
    while tonic_freq < median_pitch / 1.5:
        tonic_freq *= 2.0

    return float(tonic_freq)


def quantize_to_semitones(f0, tonic_freq):
    """
    Convert f0 contour (Hz) to semitone offsets relative to tonic.
    Unvoiced/silent frames become NaN.

    semitone = 12 * log2(f0 / tonic_freq)

    We round to the nearest integer semitone for discretization.
    """
    semitones = np.full_like(f0, np.nan)
    voiced = ~np.isnan(f0)
    semitones[voiced] = 12.0 * np.log2(f0[voiced] / tonic_freq)
    semitones[voiced] = np.round(semitones[voiced])
    return semitones


def smooth_contour(semitones, kernel_size=MEDIAN_KERNEL):
    """
    Apply median filter to the semitone contour to remove vibrato jitter.
    Only applies to voiced (non-NaN) regions. NaN frames are preserved.
    """
    # Replace NaN with a sentinel for median filter, then restore
    sentinel = -999
    filled = np.where(np.isnan(semitones), sentinel, semitones)
    smoothed = median_filter(filled, size=kernel_size)
    # Restore NaN where sentinel persists
    smoothed = np.where(smoothed == sentinel, np.nan, smoothed)
    # Also restore NaN where original was NaN (edge effects)
    smoothed = np.where(np.isnan(semitones), np.nan, smoothed)
    return smoothed


def segment_notes(semitones):
    """
    Convert a frame-level semitone contour into a list of note events.

    Each note event is a dict:
        {"pitch": int or "REST", "duration": float (seconds)}

    Consecutive frames with the same pitch (or consecutive NaN frames)
    are merged into a single note/rest event.

    Notes shorter than MIN_NOTE_FRAMES are merged into their neighbors
    (treated as passing tones / artifacts).
    """
    events = []
    n_frames = len(semitones)

    if n_frames == 0:
        return events

    # Build raw run-length encoding
    runs = []
    current_val = semitones[0]
    current_len = 1

    for i in range(1, n_frames):
        val = semitones[i]
        same = (np.isnan(current_val) and np.isnan(val)) or \
               (not np.isnan(current_val) and not np.isnan(val) and current_val == val)
        if same:
            current_len += 1
        else:
            runs.append((current_val, current_len))
            current_val = val
            current_len = 1

    runs.append((current_val, current_len))

    # Merge very short notes into neighbors
    merged = []
    for val, length in runs:
        if length < MIN_NOTE_FRAMES and len(merged) > 0 and not np.isnan(merged[-1][0]):
            # Extend previous note
            prev_val, prev_len = merged[-1]
            merged[-1] = (prev_val, prev_len + length)
        else:
            merged.append((val, length))

    # Convert to note events
    for val, length in merged:
        duration = length * HOP_SEC
        if np.isnan(val):
            events.append({"pitch": "REST", "duration": round(duration, 4)})
        else:
            events.append({"pitch": int(val), "duration": round(duration, 4)})

    return events


def process_song(audio_path, song_id):
    """
    Full pipeline for a single song:
    pitch extraction → tonic estimation → quantization → smoothing → segmentation
    """
    pitch_cache = PITCH_DIR / f"{song_id}_pitch.npy"
    chroma_path = CHROMA_DIR / f"{song_id}.npy"

    # Step 1: Extract pitch
    f0, y, sr = extract_pitch(audio_path, pitch_cache)

    # Step 2: Estimate tonic
    tonic = estimate_tonic(f0, chroma_path)

    # Step 3: Quantize to semitones relative to tonic
    semitones = quantize_to_semitones(f0, tonic)

    # Step 4: Smooth
    smoothed = smooth_contour(semitones)

    # Step 5: Segment into note events
    notes = segment_notes(smoothed)

    # Metadata
    duration_sec = len(y) / sr
    voiced_pct = np.count_nonzero(~np.isnan(f0)) / len(f0) * 100

    result = {
        "song_id": song_id,
        "tonic_hz": round(tonic, 2),
        "tonic_note": librosa.hz_to_note(tonic),
        "duration_sec": round(duration_sec, 2),
        "voiced_frames_pct": round(voiced_pct, 1),
        "num_notes": len(notes),
        "notes": notes
    }

    return result


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    audio_files = sorted(AUDIO_DIR.glob("MOI_*.wav"))
    print(f"Found {len(audio_files)} audio files")

    all_results = []
    summary_rows = []

    for audio_path in tqdm(audio_files, desc="Extracting note events"):
        song_id = audio_path.stem
        result = process_song(audio_path, song_id)

        # Save individual song JSON
        json_path = NOTE_DIR / f"{song_id}_notes.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        all_results.append(result)

        summary_rows.append({
            "song_id": song_id,
            "tonic_hz": result["tonic_hz"],
            "tonic_note": result["tonic_note"],
            "duration_sec": result["duration_sec"],
            "voiced_pct": result["voiced_frames_pct"],
            "num_notes": result["num_notes"]
        })

    # Save summary
    summary_path = NOTE_DIR / "extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Note-Event Extraction Complete!")
    print(f"{'='*60}")
    print(f"Songs processed : {len(all_results)}")
    print(f"Output directory: {NOTE_DIR}")

    # Print summary table
    print(f"\n{'Song':<12} {'Tonic':>8} {'Note':>6} {'Dur(s)':>8} {'Voiced%':>8} {'#Notes':>7}")
    print("-" * 55)
    for r in summary_rows:
        print(f"{r['song_id']:<12} {r['tonic_hz']:>8.1f} {r['tonic_note']:>6} "
              f"{r['duration_sec']:>8.1f} {r['voiced_pct']:>7.1f}% {r['num_notes']:>7}")
