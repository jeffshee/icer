import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf

tf.get_logger().setLevel('INFO')

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

# TODO
#  1. 短くなった音声を統合してみる
#  2. Google Speech Recognitionが認識不能の場合でも、他のAPIによる結果で補完することも考えられる

retry_num = 3  # 音声認識が成功するまで試す回数
use_loudness_based_diarization = True
allow_overlapping = False


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
    cleanup_directory(diarization_dir)

    """ 1. Diarization """
    if use_loudness_based_diarization:
        if allow_overlapping:
            df_diarization_compact = loudness_based_diarization_overlap(audio_path_list, diarization_dir)
        else:
            df_diarization_compact = loudness_based_diarization_v2(audio_path_list, diarization_dir)
    else:
        mix_audio(audio_path_list, mixed_audio_path)
        from run_speaker_diarization_v2 import run_speaker_diarization
        run_speaker_diarization(output_dir + os.sep, mixed_audio_name, diarization_dir + os.sep,
                                people_num=people_num, overlap_rate=0.5)
        df_diarization_compact = diarization_compact_v2(join(diarization_dir, 'result.csv'), diarization_dir)

    """ 2. Split audio after Diarization """
    if use_loudness_based_diarization:
        split_path_list, start_time_list, end_time_list = split_audio_after_diarization_v2(df_diarization_compact,
                                                                                           audio_path_list,
                                                                                           split_audio_dir)
    else:
        split_path_list, start_time_list, end_time_list = split_audio_after_diarization_v2(df_diarization_compact,
                                                                                           [mixed_audio_path],
                                                                                           split_audio_dir)

    """ 3. Segmentation -> Speech2Text """
    output_csv = []
    for i, split_path in enumerate(sorted(split_path_list)):
        segment_path_list = segment_audio(split_path, segment_audio_dir)
        if len(segment_path_list) == 0:
            continue
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
                        if attempt > 0:
                            print('=== Attempt #{} ==='.format(attempt + 1))
                        try:
                            segment_transcript_list.append(r.recognize_google(audio, language='ja-JP'))
                            is_success = True
                            print(split_progress, segment_progress, segment_path, segment_transcript_list[-1])
                        except speech_recognition.UnknownValueError:
                            if attempt == retry_num - 1:
                                segment_transcript_list.append('')
                            print(split_progress, segment_progress, segment_path, 'Could not understand audio')
                        except speech_recognition.RequestError as e:
                            if attempt == retry_num - 1:
                                segment_transcript_list.append('')
                            print(split_progress, segment_progress, segment_path, 'RequestError: {}'.format(e))
                        attempt += 1

        origin_filename = basename(split_path)
        order, speaker_class = int(origin_filename.split('_')[0]), int(origin_filename.split('_')[2][:-4])
        output_csv.append(
            [order, get_hms(start_time_list[i]), get_hms(end_time_list[i]), '; '.join(segment_transcript_list),
             speaker_class, start_time_list[i], end_time_list[i]])
    df = pd.DataFrame(output_csv,
                      columns=['Order', 'Start time(HH:MM:SS)', 'End time(HH:MM:SS)', 'Text', 'Speaker',
                               'Start time(ms)', 'End time(ms)'])
    df = df.sort_values('Order')
    df.to_csv(transcript_path, index=False, encoding='utf_8_sig', header=True)

    # Remove `split` directory, we don't need them anymore
    # shutil.rmtree(split_audio_dir)


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


def diarization_compact_v2(csv_path, output_dir=None):
    df_diarization = pd.read_csv(csv_path, encoding='shift_jis', header=0, usecols=['time(ms)', 'speaker class'])
    df_diarization.sort_values(by=['time(ms)'], ascending=True, inplace=True)
    time_list = df_diarization[df_diarization['speaker class'].diff() != 0].append(df_diarization.tail(1))[
        'time(ms)'].tolist()
    start_time_list, end_time_list = time_list[:-1], time_list[1:]
    speaker_class_list = df_diarization[df_diarization['speaker class'].diff() != 0]['speaker class'].tolist()
    df_diarization_compact = pd.DataFrame(list(zip(*[speaker_class_list, start_time_list, end_time_list])),
                                          columns=['Speaker', 'Start time(ms)', 'End time(ms)'])
    if output_dir is not None:
        df_diarization_compact.to_csv(join(output_dir, 'result_compact.csv'), index=False, header=True)
    return df_diarization_compact


def split_audio_after_diarization_v2(df_diarization_compact, audio_path_list, output_dir, min_range=1000):
    speaker_class_list, start_time_list, end_time_list = df_diarization_compact['Speaker'], df_diarization_compact[
        'Start time(ms)'], df_diarization_compact['End time(ms)']
    output_path_list = []
    output_start_time_list = []
    output_end_time_list = []
    for i, (speaker_class, start_time, end_time) in enumerate(zip(speaker_class_list, start_time_list, end_time_list)):
        if end_time - start_time < min_range:
            continue

        output_path = join(output_dir, '{:03d}_speaker_{}.wav'.format(i, speaker_class))
        output_path_list.append(output_path)
        output_start_time_list.append(start_time)
        output_end_time_list.append(end_time)
        if len(audio_path_list) > 1:
            trim_audio(input_audio=audio_path_list[speaker_class], output_audio=output_path,
                       trim_ms_range=[start_time, end_time])
        else:
            trim_audio(input_audio=audio_path_list[0], output_audio=output_path, trim_ms_range=[start_time, end_time])
    return output_path_list, output_start_time_list, output_end_time_list


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


def audio_segment_of_speaker_class(audio_segment, df_diarization_compact, speaker_class):
    import copy
    df = copy.deepcopy(df_diarization_compact)[:-1]
    df['end(ms)'] = df_diarization_compact['time(ms)'].values.tolist()[1:]
    start_list = df[df['speaker class'] == speaker_class]['time(ms)'].values.tolist()
    end_list = df[df['speaker class'] == speaker_class]['end(ms)'].values.tolist()
    return sum([audio_segment[start:end] for start, end in zip(start_list, end_list)])


def loudness_based_diarization_v2(audio_path_list, output_dir=None):
    import numpy as np
    from inaSpeechSegmenter import Segmenter
    audio_list = [AudioSegment.from_file(path, format='wav') for path in audio_path_list]
    seg = Segmenter(vad_engine='smn', detect_gender=False)

    start_time_list = []
    end_time_list = []
    for path in audio_path_list:
        speech_segment = list(filter(lambda x: x[0] == 'speech', seg(path)))
        for interval in speech_segment:
            start_time_list.append(interval[1] * 1000)
            end_time_list.append(interval[2] * 1000)

    checkpoint = sorted(
        list(zip(['start'] * len(start_time_list), start_time_list)) + list(
            zip(['end'] * len(end_time_list), end_time_list)),
        key=lambda x: x[1])

    output_csv = []
    output_csv_compact = []
    count = 0
    for i, (c1, c2) in enumerate(zip(checkpoint, checkpoint[1:])):
        count += 1 if c1[0] == 'start' else -1
        if count == 0:
            continue
        start_time, end_time = c1[1], c2[1]
        dBFS_list = [audio[start_time:end_time].dBFS for audio in audio_list]
        speaker_class = np.argmax(dBFS_list)
        if len(output_csv_compact) > 0 and speaker_class == output_csv_compact[-1][0]:
            # Combine with previous checkpoint
            output_csv_compact[-1][2] = end_time
        else:
            output_csv_compact.append([speaker_class, start_time, end_time])
        for t in range(int(start_time), int(end_time)):
            output_csv.append([t, speaker_class])
    df = pd.DataFrame(output_csv, columns=['time(ms)', 'speaker class'])
    df_compact = pd.DataFrame(output_csv_compact, columns=['Speaker', 'Start time(ms)', 'End time(ms)'])
    if output_dir is not None:
        df_compact.to_csv(join(output_dir, 'result_compact.csv'), index=False, header=True)
        # Output 'result.csv' similar to diarization for backward compatibility, can be removed if not longer needed
        df.to_csv(join(output_dir, 'result.csv'), index=False, header=True)
    return df_compact


def loudness_based_diarization_overlap(audio_path_list, output_dir=None):
    from inaSpeechSegmenter import Segmenter
    seg = Segmenter(vad_engine='smn', detect_gender=False)

    speech_segment_list = []
    for speaker_class, path in enumerate(audio_path_list):
        speech_segment = list(filter(lambda x: x[0] == 'speech', seg(path)))
        for interval in speech_segment:
            speech_segment_list.append([speaker_class, interval[1] * 1000, interval[2] * 1000])

    speech_segment_list = sorted(speech_segment_list, key=lambda x: x[1])

    df_compact = pd.DataFrame(speech_segment_list, columns=['Speaker', 'Start time(ms)', 'End time(ms)'])
    if output_dir is not None:
        df_compact.to_csv(join(output_dir, 'result_compact.csv'), index=False, header=True)
    return df_compact


def get_hms(ms):
    import datetime
    td = datetime.timedelta(milliseconds=ms)
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)


if __name__ == "__main__":
    # Prepare short ver. for testing
    # for path in ["test/wave/200225_芳賀先生_実験23/200225_芳賀先生_実験23voice{}.wav".format(i) for i in range(1, 7)]:
    #     trim_audio(path, path[:-4] + '_short.wav', [0, 300000])
    transcript('output_test',
               ["test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice{}_short.wav".format(i) for i in range(1, 6)])

    # transcript('output_exp22_loudness_2',
    #            ["test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice{}.wav".format(i) for i in range(1, 6)])
    # transcript('output_exp23_loudness_2',
    #            ["test/wave/200225_芳賀先生_実験23/200225_芳賀先生_実験23voice{}.wav".format(i) for i in range(1, 7)])

    # transcript('output_exp22_loudness_overlap',
    #            ["test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice{}.wav".format(i) for i in range(1, 6)])
