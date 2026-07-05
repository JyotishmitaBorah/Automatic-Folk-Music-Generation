from pathlib import Path
import json
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

GEN_DIR = PROJECT / "outputs" / "generated_symbolic"


# ----------------------------------------
# Remove impossible notes
# ----------------------------------------

MIN_PITCH = -12
MAX_PITCH = 24

MIN_DURATION = 0.08
MAX_DURATION = 2.5


def clean_notes(notes):

    cleaned = []

    previous_pitch = None

    for note in notes:

        pitch = note["pitch"]
        duration = float(note["duration"])

        duration = max(MIN_DURATION, min(MAX_DURATION, duration))

        if pitch != "REST":

            pitch = int(pitch)

            pitch = max(MIN_PITCH, min(MAX_PITCH, pitch))

            # avoid huge melodic jumps
            if previous_pitch is not None:

                diff = pitch - previous_pitch

                if abs(diff) > 7:

                    pitch = previous_pitch + np.sign(diff) * 7

            previous_pitch = pitch

        cleaned.append(
            {
                "pitch": pitch,
                "duration": round(duration,3)
            }
        )

    return cleaned


def process_file(json_path):

    with open(json_path,"r") as f:
        data=json.load(f)

    notes=data["notes"]

    cleaned=clean_notes(notes)

    data["notes"]=cleaned

    out=json_path.parent/(json_path.stem+"_clean.json")

    with open(out,"w") as f:
        json.dump(data,f,indent=2)

    print("Saved",out.name)


if __name__=="__main__":

    total=0

    for model in GEN_DIR.iterdir():

        if not model.is_dir():
            continue

        for file in model.glob("sample*_notes.json"):

            process_file(file)
            total+=1

    print()
    print("==============================")
    print("Finished")
    print("Files:",total)