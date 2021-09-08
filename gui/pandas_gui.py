import numpy as np
import pandas as pd
import MeCab
import re
import os
from pandasgui import show

# Override word_cloud font to support Japanese
os.environ["FONT_PATH"] = "GenYoMinJP-Regular.ttf"


def get_word_str(text):
    mecab = MeCab.Tagger()
    parsed = mecab.parse(text)
    lines = parsed.split('\n')
    lines = lines[0:-2]
    nouns_list = []
    adj_list = []
    verbs_list = []

    for line in lines:
        tmp = re.split('\t|,', line)

        # 対象
        if tmp[1] == "名詞":
            nouns_list.append(tmp[0])
        if tmp[1] == "形容詞":
            adj_list.append(tmp[0])
        if tmp[1] == "動詞":
            verbs_list.append(tmp[0])

    return " ".join(nouns_list), " ".join(adj_list), " ".join(verbs_list)


def get_word_cloud(df_transcript: pd.DataFrame):
    word_cloud = []
    for t in df_transcript.Text:
        if t is np.nan:
            word_cloud.append(("", "", ""))
        else:
            word_cloud.append(get_word_str(t))
    # Make a copy and remove "Text", "Start time(ms)", "End time(ms)" columns
    # df_word_cloud = df_transcript.copy()
    df_word_cloud = df_transcript.drop(columns=["Text", "Start time(ms)", "End time(ms)"])
    # Append word cloud result to df
    for i, target in enumerate(["Nouns", "Adj.", "Verbs"]):
        df_word_cloud[f"Word Cloud ({target})"] = [a[i] for a in word_cloud]
    return df_word_cloud


def show_transcript_gui(transcript_csv: str):
    df_transcript = pd.read_csv(transcript_csv)
    df_word_cloud = get_word_cloud(df_transcript)
    show(df_transcript, df_word_cloud)


if __name__ == "__main__":
    dummy_csv = "../output/exp58_s2t/transcript/transcript.csv"
    show_transcript_gui(dummy_csv)
