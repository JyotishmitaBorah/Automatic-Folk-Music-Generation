from pathlib import Path
import sys
import json
import argparse

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT))

from models.symbolic_models import FolkLSTMv2

# =====================================================
# Paths
# =====================================================

TRAIN_DIR = PROJECT / "dataset" / "training"
MODEL_DIR = PROJECT / "models"
OUTPUT_DIR = PROJECT / "outputs" / "generated"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Duration bins
# =====================================================

DURATION_BINS = np.array([
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.45,
    0.70,
    1.00,
    1.50,
    2.00,
    3.00,
    5.00
])

# =====================================================
# Decode Token
# =====================================================

def decode_token(token):

    if token in ["<PAD>", "<SOS>", "<EOS>"]:
        return None

    if token.startswith("REST"):

        bucket = int(token.split("_D")[1])

        return {
            "pitch": "REST",
            "duration": float(DURATION_BINS[bucket])
        }

    pitch = int(token.split("_")[0][1:])
    bucket = int(token.split("_D")[1])

    return {
        "pitch": pitch,
        "duration": float(DURATION_BINS[bucket])
    }


def tokens_to_notes(tokens):

    notes = []

    for t in tokens:

        x = decode_token(t)

        if x is not None:
            notes.append(x)

    return notes


# =====================================================
# Generate Sequence
# =====================================================

def generate_sequence(
        model,
        seed,
        inv_vocab,
        device,
        max_new_tokens=200,
        temperature=0.9):

    model.eval()

    generated = seed.copy()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            x = torch.LongTensor([generated[-63:]]).to(device)

            logits, _ = model(x)

            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                1
            ).item()

            if next_token <= 2:
                continue

            generated.append(next_token)

    return [inv_vocab[i] for i in generated]


# =====================================================
# MIDI Writer
# =====================================================

def notes_to_midi(notes, outfile):

    import mido

    mid = mido.MidiFile()

    track = mido.MidiTrack()

    mid.tracks.append(track)

    track.append(
        mido.MetaMessage(
            "set_tempo",
            tempo=mido.bpm2tempo(100)
        )
    )

    track.append(
        mido.Message(
            "program_change",
            program=73,
            time=0
        )
    )

    base_note = 60

    ticks_per_sec = mid.ticks_per_beat * 100 / 60

    for n in notes:

        ticks = max(
            1,
            int(n["duration"] * ticks_per_sec)
        )

        if n["pitch"] == "REST":

            track.append(
                mido.Message(
                    "note_off",
                    note=60,
                    velocity=0,
                    time=ticks
                )
            )

        else:

            note = base_note + n["pitch"]

            note = min(127, max(0, note))

            track.append(
                mido.Message(
                    "note_on",
                    note=note,
                    velocity=90,
                    time=0
                )
            )

            track.append(
                mido.Message(
                    "note_off",
                    note=note,
                    velocity=0,
                    time=ticks
                )
            )

    mid.save(outfile)

# =====================================================
# WAV Writer
# =====================================================

def notes_to_wav(notes, outfile, sr=22050):

    from scipy.io.wavfile import write

    audio = []

    tonic = 261.63

    for n in notes:

        length = max(1, int(sr * n["duration"]))

        t = np.linspace(0, n["duration"], length, endpoint=False)

        if n["pitch"] == "REST":

            wave = np.zeros(length)

        else:

            freq = tonic * (2 ** (n["pitch"] / 12))

            wave = (
                0.60 * np.sin(2*np.pi*freq*t)
                + 0.25 * np.sin(2*np.pi*2*freq*t)
                + 0.15 * np.sin(2*np.pi*3*freq*t)
            )

            fade = min(length//5, int(0.02*sr))

            if fade > 0:
                wave[:fade] *= np.linspace(0,1,fade)
                wave[-fade:] *= np.linspace(1,0,fade)

        audio.append(wave)

    audio = np.concatenate(audio)

    audio /= np.max(np.abs(audio))

    audio *= 0.8

    write(
        outfile,
        sr,
        (audio*32767).astype(np.int16)
    )


# =====================================================
# Load Vocabulary
# =====================================================

with open(TRAIN_DIR/"symbolic_vocab.json") as f:

    vocab = json.load(f)

inv_vocab = {v:k for k,v in vocab.items()}

vocab_size = len(vocab)


# =====================================================
# Load Validation Seeds
# =====================================================

val = np.load(TRAIN_DIR/"symbolic_val.npz")

X_val = val["X"]


# =====================================================
# Load Model
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = FolkLSTMv2(vocab_size).to(device)

ckpt = torch.load(
    MODEL_DIR/"symbolic_LSTM_best.pth",
    map_location=device
)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()

# =====================================================
# Main Generation
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=5
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=180
    )

    args = parser.parse_args()

    print("="*60)
    print("Generating Folk Music")
    print("="*60)

    summary = []

    for sample in range(args.samples):

        print(f"\nSample {sample+1}")

        # ----------------------------
        # Random Seed
        # ----------------------------

        seed = X_val[
            np.random.randint(len(X_val))
        ][:20].tolist()

        seed = [x for x in seed if x != 0]

        print("Seed Length :", len(seed))

        # ----------------------------
        # Generate Tokens
        # ----------------------------

        tokens = generate_sequence(
            model=model,
            seed=seed,
            inv_vocab=inv_vocab,
            device=device,
            max_new_tokens=args.tokens,
            temperature=args.temperature
        )

        notes = tokens_to_notes(tokens)

        duration = sum(
            n["duration"] for n in notes
        )

        print("Generated Notes :", len(notes))
        print("Duration :", round(duration,2),"sec")

        # ----------------------------
        # Save JSON
        # ----------------------------

        json_path = OUTPUT_DIR / f"sample_{sample+1}.json"

        with open(json_path,"w") as f:

            json.dump(
                {
                    "tokens":tokens,
                    "notes":notes
                },
                f,
                indent=2
            )

        # ----------------------------
        # Save MIDI
        # ----------------------------

        midi_path = OUTPUT_DIR / f"sample_{sample+1}.mid"

        notes_to_midi(
            notes,
            str(midi_path)
        )

        # ----------------------------
        # Save WAV
        # ----------------------------

        wav_path = OUTPUT_DIR / f"sample_{sample+1}.wav"

        notes_to_wav(
            notes,
            str(wav_path)
        )

        print("Saved:")
        print(" ",midi_path.name)
        print(" ",wav_path.name)

        summary.append(
            {
                "sample":sample+1,
                "notes":len(notes),
                "duration":duration
            }
        )

    with open(
        OUTPUT_DIR/"generation_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print("\n")
    print("="*60)
    print("Generation Finished Successfully")
    print("="*60)
    print("Outputs saved in:")
    print(OUTPUT_DIR)