# -*- coding: utf-8 -*-
import os
import shutil

import pandas as pd
import speech_recognition

from edit_audio import wav_to_transcript, split_wave_per_sentence, mix_audio
from main_util import cleanup_directory
from run_speaker_diarization_v2 import run_speaker_diarization


def pinmic_audio_to_transcript(pinmic_audio_name_list, path_diarization, path_audio, path_result, matching_index_list):
    # speakerの切り替わりのタイミングを取得する
    df_diarization = pd.read_csv("{}result.csv".format(path_diarization), encoding="shift_jis", header=0, usecols=["time(ms)", "speaker class"])
    for speaker_class in range(len(df_diarization["speaker class"].unique())):
        df_diarization_sorted = df_diarization.copy()
        df_diarization_sorted.sort_values(by=["time(ms)"], ascending=True, inplace=True)
        df_diarization_sorted_diff = df_diarization_sorted.copy()
        df_diarization_sorted_diff["speaker class"] = df_diarization_sorted_diff["speaker class"].diff()
        df_split_ms = df_diarization_sorted_diff[df_diarization_sorted_diff["speaker class"] != 0]

    cleanup_directory(path_result + "selected_wav/")
    for (pinmic_audio_name, matching_index) in zip(pinmic_audio_name_list, matching_index_list):
        # 音声を文章ごとに分割
        split_wave_per_sentence(df_diarization_sorted, df_split_ms["time(ms)"].values.tolist(), path_audio + pinmic_audio_name + ".wav", path_result + "{}/split_wave/".format(pinmic_audio_name))

        # 対象人物の音声のみ移動
        wav_name_list = os.listdir(path_result + "{}/split_wave/".format(pinmic_audio_name))
        selected_wav_name_list = [x for x in wav_name_list if "_SpeakerName{}".format(matching_index) in x]
        for selected_wav_name in selected_wav_name_list:
            shutil.copy(path_result + "{}/split_wave/".format(pinmic_audio_name) + selected_wav_name,
                        path_result + "selected_wav/" + selected_wav_name)

    # 音声書き起こし
    with speech_recognition.AudioFile(path_audio + pinmic_audio_name + ".wav") as src:
        audio_duration_ms = src.DURATION * 1000
    wav_to_transcript(path_result + "selected_wav/", split_ms_list=df_split_ms["time(ms)"].values.tolist() + [audio_duration_ms], output_path=path_result + "transcript.csv")


if __name__ == '__main__':
    path_result = "pinmic_to_transcript/"
    path_audio = "audio/"
    pinmic_audio_name_list = ["200225_Haga_22_voice1", "200225_Haga_22_voice2", "200225_Haga_22_voice3", "200225_Haga_22_voice4", "200225_Haga_22_voice5"]

    # pinmic_mix_audioの作成
    mixed_audio_name = pinmic_audio_name_list[0].split("pinmic_")[0] + "pinmic_mix"
    mix_audio([path_audio + pinmic_audio_name + ".wav" for pinmic_audio_name in pinmic_audio_name_list], path_audio + mixed_audio_name + ".wav")

    # mix_audioからDiarization
    path_diarization_result = "diarization/{}_{}_{}/".format(mixed_audio_name, 0.5, 0.5)
    run_speaker_diarization(path_audio, mixed_audio_name, path_diarization_result, len(pinmic_audio_name_list))

    # match speaker (manually)
    matching_index_list = []  # Diarization結果のインデックスとpinmic_audio_name_indexを人力で対応させる
    for i in range(len(pinmic_audio_name_list)):
        print("Input the matching index of {} ->".format(pinmic_audio_name_list[i]))
        matching_index_list.append(int(input()))

    cleanup_directory(path_result)
    pinmic_audio_to_transcript(pinmic_audio_name_list, path_diarization_result, path_audio, path_result, matching_index_list)
