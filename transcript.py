import os
import shutil
from os.path import join, dirname, basename
import pandas as pd
import speech_recognition
from pydub import AudioSegment

from edit_audio import mix_audio, trim_audio
from main_util import cleanup_directory

"""
各API使用料金の調査
    CMU Sphinx (works offline)  (日本語対応しない)
    Google Speech Recognition   (Default API keyを使えばほぼ無制限だけど、そもそもそれはテスト目的なのでいつまで使えれか保証できない）
                                    https://github.com/Uberi/speech_recognition/issues/63
    Google Cloud Speech API     (最初の60min以内無料) $0.024/min    https://cloud.google.com/speech-to-text/pricing
    Wit.ai                      不明
    Microsoft Azure Speech
    Microsoft Bing Voice Recognition (Deprecated)
    Houndify API                不明
    IBM Speech to Text          (Lite 毎月500min無料) (Standard) $0.02/min   https://www.ibm.com/cloud/watson-speech-to-text/pricing
    Snowboy Hotword Detection (works offline)

"""

retry_num = 3  # 音声認識が成功するまで試す回数


def transcript(output_dir, audio_path_list, people_num=None):
    """
    The main routine to transcript the audio file
    :param output_dir: The directory that contains all outputted files
    :param audio_path_list: A list of path to the audio files, these audio files will be mixed into one.
        It can be only one audio file, for example: ['/foo/bar/audio.wav'], in this case you must specify people_num.
    :param people_num: people_num for Diarization.
    :return:
    """

    if people_num is None:
        assert len(audio_path_list) > 1
        people_num = len(audio_path_list)

    diarization_dir = join(output_dir, 'diarization')
    split_audio_dir = join(output_dir, 'split')
    segment_audio_dir = join(output_dir, 'segment')

    mixed_audio_name = 'mixed_audio'
    mixed_audio_path = join(output_dir, mixed_audio_name + '.wav')
    transcript_path = join(output_dir, 'transcript.csv')

    cleanup_directory(output_dir)
    cleanup_directory(split_audio_dir)
    cleanup_directory(segment_audio_dir)

    """ 1. Diarization """
    mix_audio(audio_path_list, mixed_audio_path)
    from run_speaker_diarization_v2 import run_speaker_diarization
    run_speaker_diarization(output_dir + os.sep, mixed_audio_name, diarization_dir + os.sep,
                            people_num=people_num, overlap_rate=0.5)

    """ 2. Split audio after Diarization """
    df_diarization_compact = diarization_compact(join(diarization_dir, 'result.csv'))
    split_path_list = split_audio_after_diarization(df_diarization_compact, mixed_audio_path, split_audio_dir)
    split_time = df_diarization_compact['time(ms)'].values.tolist()

    """ 3. Segmentation -> Speech2Text """
    output_csv = []
    for i, split_path in enumerate(sorted(split_path_list)):
        segment_path_list = segment_audio(split_path, segment_audio_dir)
        segment_transcript_list = []
        split_progress = '[{}/{}]'.format(i, len(split_path_list))

        for j, segment_path in enumerate(segment_path_list):
            attempt = 0
            is_success = False
            r = speech_recognition.Recognizer()
            segment_progress = '[{}/{}]'.format(j, len(segment_path_list))

            if AudioSegment.from_file(segment_path).duration_seconds > 60:
                # Segmentの長さが1分以上の場合、最終手段としてsplit_on_silenceを使って音声を分割する
                print('=== Duration>60 sec! Use transcript_split_on_silence as last resort ===')
                split_silence_dir = join(join(output_dir, 'split_silence'), basename(segment_path)[:-4])
                cleanup_directory(split_silence_dir)
                segment_transcript_list.append(transcript_split_on_silence(segment_path, split_silence_dir))
                print(split_progress, segment_progress, segment_path, segment_transcript_list[-1])
            else:
                with speech_recognition.AudioFile(segment_path) as src:
                    audio = r.record(src)
                    while (not is_success) and (attempt < retry_num):
                        # TODO: 使用料金の掛かるAPIが使えれば、Google Speech Recognitionが認識不能の場合でも、他のAPIによる結果で補完することも考えられる。
                        if attempt > 0:
                            print('=== Attempt #{} ==='.format(attempt + 1))
                        try:
                            segment_transcript_list.append(r.recognize_google(audio, language='ja-JP'))
                            is_success = True
                            print(split_progress, segment_progress, segment_path, segment_transcript_list[-1])
                        except speech_recognition.UnknownValueError:
                            segment_transcript_list.append('')
                            print(split_progress, segment_progress, segment_path, 'Could not understand audio')
                        except speech_recognition.RequestError as e:
                            segment_transcript_list.append('')
                            print(split_progress, segment_progress, segment_path, 'RequestError: {}'.format(e))
                        attempt += 1

        origin_filename = basename(split_path)
        order, speaker_class = int(origin_filename.split('_')[0]), int(origin_filename.split('_')[2][:-4])
        output_csv.append([order, speaker_class, '; '.join(segment_transcript_list), split_time[i], split_time[i + 1]])
    df = pd.DataFrame(output_csv, columns=['Order', 'Speaker', 'Text', 'Start time(ms)', 'End time(ms)'])
    df = df.sort_values('Order')
    df.to_csv(transcript_path, index=False, encoding='utf_8_sig', header=True)

    # Remove `split` directory, we don't need them anymore
    shutil.rmtree(split_audio_dir)


def transcript_split_on_silence(input_path, output_dir):
    """
    This is used as the last resort, if the audio is too long (>1 min) even after diarization and segmentation
    The case might be the speaker is giving a long presentation, etc.
    """
    from pydub.silence import split_on_silence
    full_audio = AudioSegment.from_wav(input_path)
    dbfs = full_audio.dBFS
    split_list = split_on_silence(full_audio, min_silence_len=1000, silence_thresh=dbfs - 10, keep_silence=100)
    origin_filename = basename(input_path)[:-4]  # remove extension

    transcript_list = []
    for i, split in enumerate(split_list):
        # Export as file
        output_path = join(output_dir, '{}_{:03d}.wav'.format(origin_filename, i))
        split.export(output_path, format='wav')

        attempt = 0
        is_success = False
        r = speech_recognition.Recognizer()
        with speech_recognition.AudioFile(output_path) as source:
            while (not is_success) and (attempt < retry_num):
                audio = r.record(source)
                try:
                    transcript_list.append(r.recognize_google(audio, language='ja-JP'))
                    is_success = True
                except speech_recognition.UnknownValueError:
                    pass
                except speech_recognition.RequestError:
                    pass
                attempt += 1

    return ' '.join(transcript_list)


def diarization_compact(csv_path):
    df_diarization = pd.read_csv(csv_path, encoding='shift_jis', header=0, usecols=['time(ms)', 'speaker class'])
    df_diarization.sort_values(by=['time(ms)'], ascending=True, inplace=True)
    df_diarization_compact = df_diarization[df_diarization['speaker class'].diff() != 0]
    df_diarization_compact = df_diarization_compact.append(df_diarization.tail(1))
    print(df_diarization_compact)
    return df_diarization_compact


def split_audio_after_diarization(df_diarization_compact, input_path, output_dir):
    split_list = df_diarization_compact['time(ms)'].values.tolist()
    speaker_class_list = df_diarization_compact['speaker class'].values.tolist()
    output_path_list = []
    for i, (split, speaker_class) in enumerate(zip(split_list, speaker_class_list)):
        if i == len(split_list) - 1:
            break
        output_path = join(output_dir, '{:03d}_speaker_{}.wav'.format(i, speaker_class))
        output_path_list.append(output_path)
        trim_ms_range = (split, split_list[i + 1])
        trim_audio(input_audio=input_path, output_audio=output_path, trim_ms_range=trim_ms_range)
    return output_path_list


def segment_audio(input_path, output_dir):
    from inaSpeechSegmenter import Segmenter
    seg = Segmenter(vad_engine='smn', detect_gender=False)
    segmentation = seg(input_path)
    output_path_list = []
    i = 0
    for segment in segmentation:
        if segment[0] == 'speech':  # 音声区間
            trim_ms_range = (segment[1] * 1000, segment[2] * 1000)
            origin_filename = basename(input_path)[:-4]  # remove extension
            output_path = join(output_dir, '{}_seg_{:03d}.wav'.format(origin_filename, i))
            output_path_list.append(output_path)
            trim_audio(input_path, output_path, trim_ms_range)
            i += 1
    return output_path_list


def diarization_pinmic_matching(df_diarization_compact, audio_path_list, mixed_audio_path):
    """
        Auto Matching based on MSE
        行：各Pinmic音声
        列：Diarizationの結果であるspeaker_classに対して、該当範囲において、Pinmic音声とMix音声のMSEを算出
        各行と列でMSEが最小となるものは、Pinmicとspeaker_classを対応付ける
        ※　Segmentationの精度はそこまでよくないので、単純にMSEを計算して比較するのはうまくいかなかった。
    """
    import numpy as np
    people_num = len(audio_path_list)
    mixed_audio_segment = AudioSegment.from_file(mixed_audio_path, format='wav')
    pinmic_audio_segment_list = [AudioSegment.from_file(file, format='wav') for file in audio_path_list]
    matching = np.empty((people_num, people_num))
    for row in range(people_num):
        for col in range(people_num):
            mixed = audio_segment_of_speaker_class(mixed_audio_segment, df_diarization_compact, col)
            pinmic = audio_segment_of_speaker_class(pinmic_audio_segment_list[row], df_diarization_compact, col)
            mse = ((np.array(mixed.get_array_of_samples()) - np.array(pinmic.get_array_of_samples())) ** 2).mean()
            matching[row, col] = mse
    print(matching)
    print(np.argmin(matching, axis=1))


def audio_segment_of_speaker_class(audio_segment, df_diarization_compact, speaker_class):
    import copy
    df = copy.deepcopy(df_diarization_compact)[:-1]
    df['end(ms)'] = df_diarization_compact['time(ms)'].values.tolist()[1:]
    start_list = df[df['speaker class'] == speaker_class]['time(ms)'].values.tolist()
    end_list = df[df['speaker class'] == speaker_class]['end(ms)'].values.tolist()
    return sum([audio_segment[start:end] for start, end in zip(start_list, end_list)])


if __name__ == "__main__":
    transcript('output_exp22',
               ["test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice{}.wav".format(i) for i in range(1, 6)])
    transcript('output_exp23',
               ["test/wave/200225_芳賀先生_実験23/200225_芳賀先生_実験23voice{}.wav".format(i) for i in range(1, 7)])
