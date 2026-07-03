import csv
import json
from pathlib import Path
import soundfile as sf

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_AUDIO = PROJECT_ROOT / "dataset" / "raw_audio"
METADATA_DIR = PROJECT_ROOT / "dataset" / "metadata"

METADATA_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = METADATA_DIR / "metadata.csv"

header = [
    "ID",
    "File_Name",
    "Title",
    "Singer",
    "Source",
    "YouTube_URL",
    "Duration",
    "Language",
    "Folk_Type",
    "Format",
    "Sample_Rate",
    "Channels",
    "Remarks"
]

rows = []

# Get all wav files
wav_files = sorted(RAW_AUDIO.glob("MOI_*.wav"))

for wav in wav_files:

    # Matching json file
    json_file = wav.with_suffix(".info.json")

    if not json_file.exists():
        print(f"Missing JSON for {wav.name}")
        continue

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio = sf.info(str(wav))

    duration = int(audio.duration)
    minutes = duration // 60
    seconds = duration % 60

    rows.append([
        wav.stem,
        wav.name,
        data.get("title", ""),
        data.get("uploader", "Unknown"),
        "YouTube",
        data.get("webpage_url", ""),
        f"{minutes}:{seconds:02d}",
        "Mising",
        "Mising Oi Nitom",
        "WAV",
        audio.samplerate,
        "Mono" if audio.channels == 1 else "Stereo",
        ""
    ])

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("="*40)
print("Metadata Generated Successfully!")
print(f"Songs Processed : {len(rows)}")
print(f"Saved at : {OUTPUT_CSV}")
print("="*40)