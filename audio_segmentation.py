import os

import pandas as pd
import speech_recognition
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter import seg2csv

from edit_audio import trim_audio


def segment_audio(input_audio):
    root = 'segmentation'
    try:
        os.makedirs(root)
    except OSError:
        pass
    dir_name = (input_audio.split('/')[-1]).split('.')[0]

    # Output Directories
    output_dir_wav = root + '/' + dir_name
    output_dir_csv = root + '/csv'
    try:
        os.makedirs(output_dir_wav)
    except OSError:
        pass
    try:
        os.makedirs(output_dir_csv)
    except OSError:
        pass

    # 'smn' は入力信号を音声区間(speeech)、音楽区間(music)、ノイズ区間(noise)にラベル付けしてくれる
    # detect_genderをTrueにすると、男性(male) / 女性(female)のラベルに細分化されるが、処理速度は遅くなる
    seg = Segmenter(vad_engine='smn', detect_gender=False)

    # 区間検出実行（たったこれだけでOK）
    segmentation = seg(input_audio)

    # ('区間ラベル',  区間開始時刻（秒）,  区間終了時刻（秒）)というタプルが
    # リスト化されているのが変数 segmentation
    # print(segmentation)

    # inaSpeechSegmenter単体では分割されたwavを作成してくれないので、
    # pydubのAudioSegmentにお世話になる (ありがたいライブラリ)
    speech_segment_index = 0
    for segment in segmentation:
        # segmentはタプル
        # タプルの第1要素が区間のラベル
        if segment[0] == 'speech':  # 音声区間
            # 区間の開始時刻の単位を秒からミリ秒に変換
            trim_ms_range = (segment[1] * 1000, segment[2] * 1000)

            trim_audio(input_audio, output_dir_wav + '/segment{:02d}.wav'.format(speech_segment_index), trim_ms_range)

            # Insert silence and trim?
            # sound = AudioSegment.from_file(input_audio, format="wav")
            # trimmed_sound = sound[trim_ms_range[0]:trim_ms_range[1]]
            # chunk_silent = AudioSegment.silent(duration=500)
            # audio_chunk = chunk_silent + trimmed_sound + chunk_silent
            # audio_chunk.export(output_dir_wav + '/segment{:02d}.wav'.format(speech_segment_index), format="wav")

            speech_segment_index += 1

    # 区間ごとのラベル,開始時間,終了時間をcsv形式で保存
    seg2csv(segmentation, output_dir_csv + '/{}.csv'.format(dir_name))


def segment_audio_to_transcript(wav_path, split_ms_list, output_path=None):
    # 音声をテキストに変換してcsv形式で保存
    wav_name_list = os.listdir(wav_path)
    wav_name_list.sort()
    txt_list = []
    for i, wav_name in enumerate(wav_name_list):
        print(wav_name)

        segment_audio(wav_path + wav_name)
        segment_list = os.listdir('segmentation/' + wav_name.split('.')[0])
        segment_list.sort()

        transcript_l = []
        for segment in segment_list:
            r = speech_recognition.Recognizer()
            with speech_recognition.AudioFile('segmentation/' + wav_name.split('.')[0] + '/' + segment) as src:
                audio = r.record(src)
                try:
                    txt = r.recognize_google(audio, language='ja-JP')
                    transcript_l.append(txt)
                    print(txt)
                except speech_recognition.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except speech_recognition.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
        txt = ' '.join(transcript_l)
        txt_list.append([int(wav_name.split("_")[0]), wav_name.split("SpeakerName")[1].split(".")[0], txt, None, None])

        # txt_list.append([int(wav_name.split("_")[0]), wav_name.split("SpeakerName")[1].split(".")[0], txt,
        #                  split_ms_list[int(wav_name.split("_")[0])],
        #                  split_ms_list[int(wav_name.split("_")[0]) + 1] - 1])

    if output_path:
        df = pd.DataFrame(txt_list, columns=["Order", "Speaker", "Text", "Start time(ms)", "End time(ms)"])
        df = df.sort_values("Order")
        df.to_csv(output_path, index=False, encoding="utf_8_sig", header=True)


if __name__ == '__main__':
    # input_dir = 'test/split_wave'
    # wav_name_list = os.listdir(input_dir)
    # for wav_name in wav_name_list:
    #     segment_audio(input_dir + '/' + wav_name)
    segment_audio_to_transcript('test/split_wave/', None, output_path='trans.csv')
