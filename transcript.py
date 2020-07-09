import os
import shutil
from os.path import join, dirname, basename
import pandas as pd
import speech_recognition

from edit_audio import mix_audio, trim_audio
from main_util import cleanup_directory


def transcript_mixed(output_dir, mixed_audio_path, people_num):
    diarization_dir = join(output_dir, 'diarization')
    split_audio_dir = join(output_dir, 'split')
    segment_audio_dir = join(output_dir, 'segment')

    mixed_audio_name = basename(mixed_audio_path)[:-4]  # remove extension
    transcript_path = join(output_dir, 'transcript.csv')

    cleanup_directory(output_dir)
    cleanup_directory(split_audio_dir)
    cleanup_directory(segment_audio_dir)

    """ 1. Diarization """
    from run_speaker_diarization_v2 import run_speaker_diarization
    run_speaker_diarization(dirname(mixed_audio_path) + os.sep, mixed_audio_name, diarization_dir + os.sep,
                            people_num=people_num)

    """ 2. Split audio after Diarization """
    df_diarization_compact = diarization_compact(join(diarization_dir, 'result.csv'))
    split_path_list = split_audio_after_diarization(df_diarization_compact, mixed_audio_path, split_audio_dir)

    """ 3. Segmentation -> Speech2Text """
    output_csv = []
    retry_num = 3
    for i, split_path in enumerate(sorted(split_path_list)):
        segment_path_list = segment_audio(split_path, segment_audio_dir)
        segment_transcript_list = []
        split_progress = '[{}/{}]'.format(i, len(split_path_list))

        for j, segment_path in enumerate(segment_path_list):
            attempt = 0
            is_success = False
            r = speech_recognition.Recognizer()
            segment_progress = '[{}/{}]'.format(j, len(segment_path_list))

            with speech_recognition.AudioFile(segment_path) as src:
                while (not is_success) and (attempt < retry_num):
                    audio = r.record(src)
                    try:
                        segment_transcript_list.append(r.recognize_google(audio, language='ja-JP'))
                        is_success = True
                        print(split_progress, segment_progress, segment_path, segment_transcript_list[-1])
                    except speech_recognition.UnknownValueError:
                        segment_transcript_list.append('Could not understand audio')
                        print(split_progress, segment_progress, segment_path, e)
                    except speech_recognition.RequestError as e:
                        segment_transcript_list.append('RequestError: {}'.format(e))
                        print(split_progress, segment_progress, segment_path, e)
                    attempt += 1

        origin_filename = basename(split_path)
        order, speaker_class = int(origin_filename.split('_')[0]), int(origin_filename.split('_')[2][:-4])
        output_csv.append([order, speaker_class, '; '.join(segment_transcript_list)])

    df = pd.DataFrame(output_csv, columns=['order', 'speaker_class', 'transcript'])
    df = df.sort_values('order')
    df.to_csv(transcript_path, index=False, encoding='utf_8_sig', header=True)

    # Remove `split` directory, we don't need them anymore
    shutil.rmtree(split_audio_dir)


def transcript_pinmic(output_dir, audio_path_list):
    # TODO: Diarization後、スピーチ認識はmixされたものを使っている。これはpinmic ver.を使った方が良いかもしれない。
    diarization_dir = join(output_dir, 'diarization')
    split_audio_dir = join(output_dir, 'split')
    segment_audio_dir = join(output_dir, 'segment')

    mixed_audio_name = 'pinmic-mix'
    mixed_audio_path = join(output_dir, mixed_audio_name + '.wav')
    transcript_path = join(output_dir, 'transcript.csv')

    cleanup_directory(output_dir)
    cleanup_directory(split_audio_dir)
    cleanup_directory(segment_audio_dir)

    """ 1. Diarization """
    mix_audio(audio_path_list, mixed_audio_path)
    from run_speaker_diarization_v2 import run_speaker_diarization
    run_speaker_diarization(output_dir + os.sep, mixed_audio_name, diarization_dir + os.sep,
                            people_num=len(audio_path_list))

    """ 2. Split audio after Diarization """
    df_diarization_compact = diarization_compact(join(diarization_dir, 'result.csv'))
    split_path_list = split_audio_after_diarization(df_diarization_compact, mixed_audio_path, split_audio_dir)

    """ 3. Segmentation -> Speech2Text """
    output_csv = []
    retry_num = 3
    for i, split_path in enumerate(sorted(split_path_list)):
        segment_path_list = segment_audio(split_path, segment_audio_dir)
        segment_transcript_list = []
        split_progress = '[{}/{}]'.format(i, len(split_path_list))

        for j, segment_path in enumerate(segment_path_list):
            attempt = 0
            is_success = False
            r = speech_recognition.Recognizer()
            segment_progress = '[{}/{}]'.format(j, len(segment_path_list))

            with speech_recognition.AudioFile(segment_path) as src:
                while (not is_success) and (attempt < retry_num):
                    audio = r.record(src)
                    try:
                        segment_transcript_list.append(r.recognize_google(audio, language='ja-JP'))
                        is_success = True
                        print(split_progress, segment_progress, segment_path, segment_transcript_list[-1])
                    except speech_recognition.UnknownValueError:
                        segment_transcript_list.append('Could not understand audio')
                        print(split_progress, segment_progress, segment_path, e)
                    except speech_recognition.RequestError as e:
                        segment_transcript_list.append('RequestError: {}'.format(e))
                        print(split_progress, segment_progress, segment_path, e)
                    attempt += 1

        origin_filename = basename(split_path)
        order, speaker_class = int(origin_filename.split('_')[0]), int(origin_filename.split('_')[2][:-4])
        output_csv.append([order, speaker_class, '; '.join(segment_transcript_list)])

    df = pd.DataFrame(output_csv, columns=['order', 'speaker_class', 'transcript'])
    df = df.sort_values('order')
    df.to_csv(transcript_path, index=False, encoding='utf_8_sig', header=True)

    # Remove `split` directory, we don't need them anymore
    shutil.rmtree(split_audio_dir)


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


if __name__ == "__main__":
    transcript_pinmic('output_exp22',
                      ["test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice{}.wav".format(i) for i in range(1, 6)])
    transcript_pinmic('output_exp23',
                      ["test/wave/200225_芳賀先生_実験23/200225_芳賀先生_実験23voice{}.wav".format(i) for i in range(1, 7)])
