# coding: utf-8
import os
import shutil
import subprocess

import pandas as pd
import speech_recognition
from pydub import AudioSegment
from pydub.silence import split_on_silence

from main_util import cleanup_directory
import math


def split_wave_per_sentence(df_diarization, split_ms_list, wave_file_path, result_path):
    cleanup_directory(result_path)
    for i in range(len(split_ms_list)):
        if i + 1 == len(split_ms_list):  # 最後なら
            trim_ms_range = (split_ms_list[i], df_diarization.tail(1)["time(ms)"].values.tolist()[0])
        else:
            trim_ms_range = (split_ms_list[i], split_ms_list[i + 1])
        speaker_index = df_diarization[df_diarization["time(ms)"] == split_ms_list[i]]["speaker class"].values.tolist()[
            0]
        trim_audio(input_audio=wave_file_path,
                    output_audio=result_path + "{}_SpeakerName{}.wav".format(i, speaker_index),
                    trim_ms_range=trim_ms_range)


# 複数音声ファイルを混ぜた音声を出力
def mix_audio(input_audio_list, output_audio):
    base_sound = AudioSegment.from_file(input_audio_list[0], format="wav")  # 音声を読み込み
    for i in range(1, len(input_audio_list)):
        sound1 = AudioSegment.from_file(input_audio_list[i], format="wav")  # 音声を読み込み
        base_sound = base_sound.overlay(sound1)
    base_sound.export(output_audio, format="wav")  # 保存する


# 動画（音有りのもの）から音声のみを取得
def extract_audio(original_video, extracted_audio):
    # Extract audio from input video.
    cmd = "ffmpeg -i {} {} -y".format(original_video, extracted_audio)
    subprocess.call(cmd, shell=True)


# 動画(音無しのもの)に音声を結合
def set_audio(input_video, input_audio, output_video):
    # Add audio to output video.
    cmd = "ffmpeg -i {} -i {} {} -y".format(input_video, input_audio, output_video)
    subprocess.call(cmd, shell=True)


# 音声をトリミング
def trim_audio(input_audio, output_audio, trim_ms_range):
    sound = AudioSegment.from_file(input_audio, format="wav")
    trimmed_sound = sound[trim_ms_range[0]:trim_ms_range[1]]
    trimmed_sound.export(output_audio, format="wav")


# 音声をテキストに変換してcsv形式で保存
def wav_to_transcript(wav_path, split_ms_list, output_path=None):
    wav_name_list = os.listdir(wav_path)
    wav_name_list.sort()

    r = speech_recognition.Recognizer()
    txt_list = []

    for i, wav_name in enumerate(wav_name_list):
        with speech_recognition.AudioFile(wav_path + wav_name) as src:
            audio = r.record(src)
            print(wav_name)
            try:
                txt = r.recognize_google(audio, language='ja-JP')
                print(txt)
            except speech_recognition.UnknownValueError or TimeoutError:  # 何も聞き取れないとき
                txt = None
                print("Error")
        txt_list.append([int(wav_name.split("_")[0]), wav_name.split("SpeakerName")[1].split(".")[0], txt,
                        split_ms_list[int(wav_name.split("_")[0])],
                        split_ms_list[int(wav_name.split("_")[0]) + 1] - 1])

    if output_path:
        df = pd.DataFrame(txt_list, columns=["Order", "Speaker", "Text", "Start time(ms)", "End time(ms)"])
        df = df.sort_values("Order")
        df.to_csv(output_path, index=False, encoding="utf_8_sig", header=True)


# 指定長以上の音声を分割
def wav_to_shortwav(wav_path, max_duration=10.0):  # max_duration[s] を最大長にする
    wav_name_list = os.listdir(wav_path)
    wav_name_list = sorted(wav_name_list)

    short_wave_path = wav_path.replace("split_wave/", "split_short_wave/")
    cleanup_directory(short_wave_path)

    for wav_name in wav_name_list:
        with speech_recognition.AudioFile(wav_path + wav_name) as src:
            if src.DURATION > max_duration:
                for i in range(int(src.DURATION // max_duration + 1)):
                    if i == src.DURATION // max_duration:
                        trim_ms_range = (i * 1000 * max_duration, 1000 * src.DURATION)
                    else:
                        trim_ms_range = (i * 1000 * max_duration, (i + 1) * 1000 * max_duration)
                    trim_audio(wav_path + wav_name, short_wave_path + wav_name.replace(".wav", "_{}.wav".format(i)), trim_ms_range)
            else:
                shutil.copy(wav_path + wav_name, short_wave_path + wav_name)


# 音声をテキストに変換
def wav_to_txt(wav_path):
    r = speech_recognition.Recognizer()
    # r.energy_threshold = 2000
    # r.dynamic_energy_threshold = False
    # with speech_recognition.AudioFile('audio/200225_Haga_22.wav') as src:
    #     r.adjust_for_ambient_noise(src)
    with speech_recognition.AudioFile(wav_path) as src:
        audio = r.record(src)
        print(wav_path)
        try:
            txt = r.recognize_google(audio, language='ja-JP')
            print(txt)
        except speech_recognition.UnknownValueError:  # 何も聞き取れないとき
            print("Error")


def m4a_to_wav(m4a_path):
    cmd = "ffmpeg -i {} {}".format(m4a_path, m4a_path.replace(".m4a", ".wav"))
    subprocess.call(cmd, shell=True)


def naive_split_wav_to_transcript(wav_path, split_ms_list, output_path=None):
    wav_name_list = os.listdir(wav_path)
    wav_name_list.sort()
    txt_list = []

    for i, wav_name in enumerate(wav_name_list):
        txt = naive_split_wav_to_txt(wav_path + wav_name)
        txt_list.append([int(wav_name.split("_")[0]), wav_name.split("SpeakerName")[1].split(".")[0], txt,
                        split_ms_list[int(wav_name.split("_")[0])],
                        split_ms_list[int(wav_name.split("_")[0]) + 1] - 1])

    if output_path:
        df = pd.DataFrame(txt_list, columns=["Order", "Speaker", "Text", "Start time(ms)", "End time(ms)"])
        df = df.sort_values("Order")
        df.to_csv(output_path, index=False, encoding="utf_8_sig", header=True)


def naive_split_wav_to_txt(wav_path):
    chunks_path = 'chunks'
    cleanup_directory(chunks_path)
    transcript_l = []
    full_audio = AudioSegment.from_wav(wav_path)
    split_msec = 60 * 1000
    split_num = math.ceil(len(full_audio) / split_msec)

    chunks = []
    for i in range(split_num):
        if i == split_num - 1:  # if last chunks
            chunks.append(full_audio[split_msec * i:-1])
        else:
            chunks.append(full_audio[split_msec * i:split_msec * (i + 1)])

    for i, chunk in enumerate(chunks):
        # Create 0.5 seconds silence chunk
        chunk_silent = AudioSegment.silent(duration=10)
        # add 0.5 sec silence to beginning and
        # end of audio chunk. This is done so that
        # it doesn't seem abruptly sliced.
        audio_chunk = chunk_silent + chunk + chunk_silent

        filename = chunks_path + '/chunk{}.wav'.format(i)
        audio_chunk.export(filename, bitrate='192k', format="wav")

        r = speech_recognition.Recognizer()

        # recognize the chunk
        with speech_recognition.AudioFile(filename) as source:
            audio = r.record(source)
            try:
                txt = r.recognize_google(audio, language='ja_JP')
                # print(i, txt)
                transcript_l.append(txt)
            except speech_recognition.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except speech_recognition.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

    result = ' '.join(transcript_l)
    print(result)
    return result


def silence_based_wav_to_txt(wav_path):
    chunks_path = 'chunks'
    cleanup_directory(chunks_path)
    transcript_l = []
    full_audio = AudioSegment.from_wav(wav_path)
    print('Average dBFS:', full_audio.dBFS)
    chunks = split_on_silence(full_audio, min_silence_len=500, silence_thresh=-35)

    for i, chunk in enumerate(chunks):
        # Create 0.5 seconds silence chunk
        chunk_silent = AudioSegment.silent(duration=10)
        # add 0.5 sec silence to beginning and
        # end of audio chunk. This is done so that
        # it doesn't seem abruptly sliced.
        audio_chunk = chunk_silent + chunk + chunk_silent

        filename = chunks_path + '/chunk{}.wav'.format(i)
        audio_chunk.export(filename, bitrate='192k', format="wav")

        r = speech_recognition.Recognizer()

        # recognize the chunk
        with speech_recognition.AudioFile(filename) as source:
            audio = r.record(source)
            try:
                txt = r.recognize_google(audio, language='ja_JP')
                print(txt)
                transcript_l.append(txt)
            except speech_recognition.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except speech_recognition.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

    print(' '.join(transcript_l))


if __name__ == "__main__":
    # pass
    # naive_split_wav_to_txt("test/split_wave/seg_test.wav")
    # naive_split_wav_to_txt("test/split_wave/38_SpeakerName2.wav")

    # audio_path = 'test/split_wave/'
    # wav_to_shortwav(audio_path)
    # wav_to_transcript('test/split_short_wave/', None)
    # audio_name_list = ["200225_Haga_22_voice1", "200225_Haga_22_voice2", "200225_Haga_22_voice3",
    #                     "200225_Haga_22_voice4", "200225_Haga_22_voice5"]
    # for audio in audio_name_list:
    #     input = "audio/" + audio + ".wav"
    #     output = "audio/" + audio + "_trim.wav"
    #     trim_audio(input, output, [0, 180000])
    wav_to_txt('test/split_wave/0_SpeakerName2.wav')
    wav_to_txt('segmentation/0_SpeakerName2/segment00.wav')
    wav_to_txt('segmentation/0_SpeakerName2/segment00.wav')
