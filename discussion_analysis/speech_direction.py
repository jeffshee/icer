import pandas as pd
import numpy as np


def get_directions_num(
    diarization_csv_path: str,
    total_speaker: int
):
    diarization = pd.read_csv(diarization_csv_path)
    directions_num = np.zeros((total_speaker, total_speaker), dtype=np.int)

    rows = diarization
    for i in range(len(rows)):
        row = rows.iloc[i]
        speaker = row["Speaker"].item()

        if i == 0:
            start = None
            end = speaker
        else:
            start = end
            end = speaker
            if start == end:
                continue
            directions_num[start][end] += 1
            print(f"[{i:03d}]: {start} → {end}")
    print(directions_num)
    return directions_num


if __name__ == "__main__":
    diarization_csv_path = "./discussion_analysis/transcript_demo.csv"
    # diarization_csv_path = "/home/icer/Project/dataset/transcript/transcript.csv"

    total_speaker = 6  # speakerの総数

    directions_num = get_directions_num(
        diarization_csv_path,
        total_speaker
    )
