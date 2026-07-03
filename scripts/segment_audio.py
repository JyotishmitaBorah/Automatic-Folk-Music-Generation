from pathlib import Path
from pydub import AudioSegment

PROJECT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT / "dataset" / "processed_audio"
OUTPUT_DIR = PROJECT / "dataset" / "segments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_LENGTH = 10 * 1000  # 10 seconds in milliseconds

files = sorted(INPUT_DIR.glob("MOI_*.wav"))

total_segments = 0

for audio_file in files:

    audio = AudioSegment.from_wav(audio_file)

    total_length = len(audio)

    count = 1

    for start in range(0, total_length, SEGMENT_LENGTH):

        end = start + SEGMENT_LENGTH

        segment = audio[start:end]

        if len(segment) < SEGMENT_LENGTH:
            continue

        filename = f"{audio_file.stem}_seg{count:03d}.wav"

        segment.export(
            OUTPUT_DIR / filename,
            format="wav"
        )

        count += 1
        total_segments += 1

print(f"\nCreated {total_segments} segments.")