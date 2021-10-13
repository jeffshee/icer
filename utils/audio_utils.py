import subprocess

from pydub import AudioSegment


# 複数音声ファイルを混ぜた音声を出力
def mix_audio(audio_path_list, output_path):
    base_sound = AudioSegment.from_file(audio_path_list[0], format="wav")  # 音声を読み込み
    for i in range(1, len(audio_path_list)):
        sound1 = AudioSegment.from_file(audio_path_list[i], format="wav")  # 音声を読み込み
        base_sound = base_sound.overlay(sound1)
    base_sound.export(output_path, format="wav")  # 保存する


# 動画（音有りのもの）から音声のみを取得
def extract_audio(video_path, output_path):
    # Extract audio from input video.
    cmd = "ffmpeg -y -i {} {}".format(video_path, output_path)
    subprocess.call(cmd, shell=True)


# 動画(音無しのもの)に音声を結合
def set_audio(video_path, audio_path, output_path):
    """
    Add audio to output video.
    :param video_path: Source (Video)
    :param audio_path: Source (Audio) This can be a video, the audio track will be copied to the output.
    :param output_path: Destination (Video)
    :return:
    """
    # Add audio to output video.
    cmd = f"ffmpeg -y -i {video_path} -i {audio_path} -c copy -map 0:0 -map 1:1 -shortest {output_path}"
    subprocess.call(cmd, shell=True)


# 音声をトリミング
def trim_audio(audio_path, output_path, trim_ms_range):
    sound = AudioSegment.from_file(audio_path, format="wav")
    trimmed_sound = sound[trim_ms_range[0]:trim_ms_range[1]]
    trimmed_sound.export(output_path, format="wav")


"""
Deprecated
"""
# def split_wave_per_sentence(df_diarization, split_ms_list, wave_file_path, result_path):
#     cleanup_directory(result_path)
#     for i in range(len(split_ms_list)):
#         if i + 1 == len(split_ms_list):  # 最後なら
#             trim_ms_range = (split_ms_list[i], df_diarization.tail(1)["time(ms)"].values.tolist()[0])
#         else:
#             trim_ms_range = (split_ms_list[i], split_ms_list[i + 1])
#         speaker_index = df_diarization[df_diarization["time(ms)"] == split_ms_list[i]]["speaker class"].values.tolist()[
#             0]
#         trim_audio(input_audio=wave_file_path,
#                    output_audio=result_path + "{}_SpeakerName{}.wav".format(i, speaker_index),
#                    trim_ms_range=trim_ms_range)
#
# # 音声をテキストに変換してcsv形式で保存
# def wav_to_transcript(wav_path, split_ms_list, output_path=None):
#     wav_name_list = os.listdir(wav_path)
#     wav_name_list.sort()
#
#     r = speech_recognition.Recognizer()
#     txt_list = []
#
#     for i, wav_name in enumerate(wav_name_list):
#         with speech_recognition.AudioFile(wav_path + wav_name) as src:
#             audio = r.record(src)
#             print(wav_name)
#             try:
#                 txt = r.recognize_google(audio, language='ja-JP')
#                 print(txt)
#             except Exception as e:
#                 txt = None
#                 print(e.args)
#         txt_list.append([int(wav_name.split("_")[0]), wav_name.split("SpeakerName")[1].split(".")[0], txt,
#                          split_ms_list[int(wav_name.split("_")[0])],
#                          split_ms_list[int(wav_name.split("_")[0]) + 1] - 1])
#
#     if output_path:
#         df = pd.DataFrame(txt_list, columns=["Order", "Speaker", "Text", "Start time(ms)", "End time(ms)"])
#         df = df.sort_values("Order")
#         df.to_csv(output_path, index=False, encoding="utf_8_sig", header=True)
#
#
# # 指定長以上の音声を分割
# def wav_to_shortwav(wav_path, max_duration=10.0):  # max_duration[s] を最大長にする
#     wav_name_list = os.listdir(wav_path)
#     wav_name_list = sorted(wav_name_list)
#
#     short_wave_path = wav_path.replace("split_wave/", "split_short_wave/")
#     cleanup_directory(short_wave_path)
#
#     for wav_name in wav_name_list:
#         with speech_recognition.AudioFile(wav_path + wav_name) as src:
#             if src.DURATION > max_duration:
#                 for i in range(int(src.DURATION // max_duration + 1)):
#                     if i == src.DURATION // max_duration:
#                         trim_ms_range = (i * 1000 * max_duration, 1000 * src.DURATION)
#                     else:
#                         trim_ms_range = (i * 1000 * max_duration, (i + 1) * 1000 * max_duration)
#                     trim_audio(wav_path + wav_name, short_wave_path + wav_name.replace(".wav", "_{}.wav".format(i)),
#                                trim_ms_range)
#             else:
#                 shutil.copy(wav_path + wav_name, short_wave_path + wav_name)
#
#
# # 音声をテキストに変換
# def wav_to_txt(wav_path):
#     r = speech_recognition.Recognizer()
#     with speech_recognition.AudioFile(wav_path) as src:
#         audio = r.record(src)
#         print(wav_path)
#         try:
#             txt = r.recognize_google(audio, language='ja-JP')
#             print(txt)
#         except speech_recognition.UnknownValueError:  # 何も聞き取れないとき
#             print("Error")
#
#
# def m4a_to_wav(m4a_path):
#     cmd = "ffmpeg -y -i {} {}".format(m4a_path, m4a_path.replace(".m4a", ".wav"))
#     subprocess.call(cmd, shell=True)
