import os
from os.path import join, basename
from pprint import pprint

import pandas as pd
import speech_recognition
from pydub import AudioSegment

from edit_audio import mix_audio, trim_audio
from split_audio import optimized_segment_audio
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

config = {
    # NOTE:（　）の中はデフォルト
    'use_run_speaker_diarization': True,  # run_speaker_diarizationを実行するか、loudness-basedの手法を使用するか (True)

    # run_speaker_diarization関連設定
    'use_spectral_cluster': False,  # run_speaker_diarizationを実行する際、SpectralClusteringを使用するか、UIS-RNNを使用するか (False)
    'people_num': None,  # run_speaker_diarizationのパラメータ
    'embedding_per_second': 1.0,  # run_speaker_diarizationのパラメータ
    'overlap_rate': 0.5,  # run_speaker_diarizationのパラメータ
    'use_loudness_after_diarization': True,  # SpectralClusteringやUIS-RNNを行った後、話者判別の結果だけをloudness-basedで差し替える (True)

    # Loudness-based設定
    'allow_overlapping': False,  # 話者間の対話で音声のOverlapping（重なり）が存在すると想定するか、しないか (False)

    # S2T関連設定
    'max_split_duration_sec': 30,  # 音声を分割する際、最大何秒まで許せるか（これより短い音声は統合される） (30)
    's2t_retry_num': 3,  # 音声認識が成功するまで試す回数 (3)
    's2t_use_mixed_audio': True,  # 音声認識する際、Mix音声を使うか、Pinmic音声を使うか (True)

    # Debug関連設定
    'no_clean': False,  # (Debug) Use old result (False)
    'skip_run_speaker_diarization': False,  # (Debug) Skip run_speaker_diarization (False)
    'skip_split': False,  # (Debug) Skip split (False)
    'skip_transcript': False,  # (Debug) Skip transcript (False)

    # Output
    'output_format': 'segment'  # segment|split
}


def transcript(output_dir, audio_path_list, transcript_config=None):
    """
    The main routine to transcript the audio file
    :param output_dir: The directory that contains all outputted files
    :param audio_path_list: A list of path to the audio files, these audio files will be mixed into one.
        It can be only one audio file, for example: ['/foo/bar/audio.wav'], in this case you must specify people_num.
    :param transcript_config: Configuration.
    :return:
    """

    if transcript_config is None:
        transcript_config = config

    if transcript_config['people_num'] is None:
        assert len(audio_path_list) > 1
        transcript_config['people_num'] = len(audio_path_list)

    diarization_dir = join(output_dir, 'diarization')
    split_audio_dir = join(output_dir, 'split')
    segment_audio_dir = join(output_dir, 'segment')

    mixed_audio_name = 'mixed_audio'
    mixed_audio_path = join(output_dir, mixed_audio_name + '.wav')
    transcript_path = join(output_dir, 'transcript.csv')

    if not transcript_config['no_clean']:
        cleanup_directory(output_dir)
        cleanup_directory(split_audio_dir)
        cleanup_directory(segment_audio_dir)
        cleanup_directory(diarization_dir)

    with open(join(output_dir, 'transcript_config.txt'), 'w') as f:
        pprint(transcript_config, stream=f)

    mix_audio(audio_path_list, mixed_audio_path)

    """ 1. Diarization """
    if transcript_config['use_run_speaker_diarization']:
        if not transcript_config['skip_run_speaker_diarization']:
            from run_speaker_diarization_v2 import run_speaker_diarization
            print("\nRunning speaker diarization")
            run_speaker_diarization(output_dir + os.sep, mixed_audio_name, diarization_dir + os.sep,
                                    people_num=transcript_config['people_num'],
                                    embedding_per_second=transcript_config['embedding_per_second'],
                                    overlap_rate=transcript_config['overlap_rate'],
                                    use_spectral_cluster=transcript_config['use_spectral_cluster'])
        df_diarization_compact = diarization_compact(join(diarization_dir, 'result.csv'), diarization_dir)
        if transcript_config['use_loudness_after_diarization']:
            df_diarization_compact = loudness_after_diarization(audio_path_list, df_diarization_compact,
                                                                diarization_dir)
    else:
        if transcript_config['allow_overlapping']:
            # 現在の実験データでは、まだノイズが酷かったため、この方法はうまくいきません。
            df_diarization_compact = speech_segmenter_only_v2(audio_path_list, diarization_dir)
        else:
            df_diarization_compact = loudness_after_speech_segmenter(audio_path_list, diarization_dir)

    """ 2. Split audio after Diarization """
    if transcript_config['skip_split']:
        return
    if transcript_config['s2t_use_mixed_audio']:
        split_path_list, start_time_list, end_time_list = split_audio_after_diarization(df_diarization_compact,
                                                                                        [mixed_audio_path],
                                                                                        split_audio_dir)
    else:
        split_path_list, start_time_list, end_time_list = split_audio_after_diarization(df_diarization_compact,
                                                                                        audio_path_list,
                                                                                        split_audio_dir)
    """ 3. Segmentation -> Speech2Text """
    if transcript_config['skip_transcript']:
        return
    print("\nRunning Speech2Text")
    output_csv = []
    order = 0
    for i, split_path in enumerate(sorted(split_path_list)):
        segment_path_list, segment_split_list = optimized_segment_audio(input_path=split_path,
                                                                        output_dir=segment_audio_dir,
                                                                        max_duration_sec=transcript_config[
                                                                            'max_split_duration_sec'])
        if len(segment_path_list) == 0:
            continue
        segment_transcript_list = []
        split_progress = '[{}/{}]'.format(i, len(split_path_list))

        origin_filename = basename(split_path)
        speaker_class = int(origin_filename.split('_')[2][:-4])

        for j, segment_path in enumerate(segment_path_list):
            attempt = 0
            is_success = False
            r = speech_recognition.Recognizer()
            segment_progress = '[{}/{}]'.format(j, len(segment_path_list))

            with speech_recognition.AudioFile(segment_path) as src:
                audio = r.record(src)
                while (not is_success) and (attempt < transcript_config['s2t_retry_num']):
                    if attempt > 0:
                        print('=== Attempt #{} ==='.format(attempt + 1))
                    try:
                        segment_transcript_list.append(r.recognize_google(audio, language='ja-JP'))
                        is_success = True
                        print(split_progress, segment_progress, segment_path, segment_transcript_list[-1])
                    except speech_recognition.UnknownValueError:
                        if attempt == transcript_config['s2t_retry_num'] - 1:
                            segment_transcript_list.append('')
                        print(split_progress, segment_progress, segment_path, 'Could not understand audio')
                    except speech_recognition.RequestError as e:
                        if attempt == transcript_config['s2t_retry_num'] - 1:
                            segment_transcript_list.append('')
                        print(split_progress, segment_progress, segment_path, 'RequestError: {}'.format(e))
                    attempt += 1
            if transcript_config['output_format'] == 'segment':
                start_time, end_time = start_time_list[i] + segment_split_list[j][0], start_time_list[i] + \
                                       segment_split_list[j][1]
                output_csv.append(
                    [order, get_hms(start_time), get_hms(end_time), segment_transcript_list[-1],
                     speaker_class, int(start_time), int(end_time)])
                order += 1

        if transcript_config['output_format'] == 'split':
            output_csv.append(
                [order, get_hms(start_time_list[i]), get_hms(end_time_list[i]), '; '.join(segment_transcript_list),
                 speaker_class, start_time_list[i], end_time_list[i]])
            order += 1

    df = pd.DataFrame(output_csv,
                      columns=['Order', 'Start time(HH:MM:SS)', 'End time(HH:MM:SS)', 'Text', 'Speaker',
                               'Start time(ms)', 'End time(ms)'])
    df = df.sort_values('Order')
    df.to_csv(transcript_path, index=False, encoding='utf_8_sig', header=True)

    # Remove `split` directory, we don't need them anymore
    # import shutil
    # shutil.rmtree(split_audio_dir)


def diarization_compact(csv_path, output_dir=None):
    df_diarization = pd.read_csv(csv_path, encoding='shift_jis', header=0, usecols=['time(ms)', 'speaker class'])
    df_diarization.sort_values(by=['time(ms)'], ascending=True, inplace=True)
    time_list = df_diarization[df_diarization['speaker class'].diff() != 0].append(df_diarization.tail(1))[
        'time(ms)'].tolist()
    start_time_list, end_time_list = time_list[:-1], time_list[1:]
    speaker_class_list = df_diarization[df_diarization['speaker class'].diff() != 0]['speaker class'].tolist()
    df_diarization_compact = pd.DataFrame(list(zip(
        *[speaker_class_list, start_time_list, end_time_list, [get_hms(t) for t in start_time_list],
          [get_hms(t) for t in end_time_list]])),
        columns=['Speaker', 'Start time(ms)', 'End time(ms)', 'Start time(HH:MM:SS)',
                 'End time(HH:MM:SS)'])
    if output_dir is not None:
        df_diarization_compact.to_csv(join(output_dir, 'result_compact.csv'), index=False, header=True)
    return df_diarization_compact


def split_audio_after_diarization(df_diarization_compact, audio_path_list, output_dir, min_range=1000):
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


def audio_segment_of_speaker_class(audio_segment, df_diarization_compact, speaker_class):
    import copy
    df = copy.deepcopy(df_diarization_compact)[:-1]
    df['end(ms)'] = df_diarization_compact['time(ms)'].values.tolist()[1:]
    start_list = df[df['speaker class'] == speaker_class]['time(ms)'].values.tolist()
    end_list = df[df['speaker class'] == speaker_class]['end(ms)'].values.tolist()
    return sum([audio_segment[start:end] for start, end in zip(start_list, end_list)])


def loudness_after_speech_segmenter(audio_path_list, output_dir=None):
    """
    Speecj SegmenterでSpeech領域を抽出後、重なる領域があれば、各Pinmic音声のLoudnessでどの領域を採用するかを判定する。
    つまり、もし同じTimeframeで複数のPinmic音声にSpeech領域が存在する場合、平均音量の小さい方はノイズであろうと仮定している。
    また、これは話者間の対話で音声のOverlapping（重なり）が存在しないと想定した手法である。
    :param audio_path_list:
    :param output_dir:
    :return:
    """
    from inaSpeechSegmenter import Segmenter
    import numpy as np
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
            output_csv_compact[-1][4] = get_hms(end_time)
        else:
            output_csv_compact.append([speaker_class, start_time, end_time, get_hms(start_time), get_hms(end_time)])
        for t in range(int(start_time), int(end_time)):
            output_csv.append([t, speaker_class])
    df = pd.DataFrame(output_csv, columns=['time(ms)', 'speaker class'])
    df_compact = pd.DataFrame(output_csv_compact,
                              columns=['Speaker', 'Start time(ms)', 'End time(ms)', 'Start time(HH:MM:SS)',
                                       'End time(HH:MM:SS)'])
    if output_dir is not None:
        df_compact.to_csv(join(output_dir, 'result_compact.csv'), index=False, header=True)
        # Output 'result.csv' similar to diarization for backward compatibility, can be removed if not longer needed
        df.to_csv(join(output_dir, 'result.csv'), index=False, header=True)
    return df_compact


def loudness_after_diarization(audio_path_list, df_diarization_compact, output_dir=None):
    """
    Diarization（UIS-RNN）を使えば、話者の入れ替わりのtimeframeが分かる。そこから、各PinmicのLoudnessで話者を判別する？
    :param df_diarization_compact: diarizationの結果
    :param audio_path_list:
    :param output_dir:
    :return:
    """
    import numpy as np
    audio_list = [AudioSegment.from_file(path, format='wav') for path in audio_path_list]
    start_time_list, end_time_list = df_diarization_compact['Start time(ms)'].values.tolist(), df_diarization_compact[
        'End time(ms)'].values.tolist()

    output_csv = []
    output_csv_compact = []
    for i, (start_time, end_time) in enumerate(zip(start_time_list, end_time_list)):
        dBFS_list = [audio[start_time:end_time].dBFS for audio in audio_list]
        speaker_class = np.argmax(dBFS_list)
        if len(output_csv_compact) > 0 and speaker_class == output_csv_compact[-1][0]:
            output_csv_compact[-1][2] = end_time
            output_csv_compact[-1][4] = get_hms(end_time)
        else:
            output_csv_compact.append([speaker_class, start_time, end_time, get_hms(start_time), get_hms(end_time)])
        for t in range(int(start_time), int(end_time)):
            output_csv.append([t, speaker_class])
    df = pd.DataFrame(output_csv, columns=['time(ms)', 'speaker class'])
    df_compact = pd.DataFrame(output_csv_compact,
                              columns=['Speaker', 'Start time(ms)', 'End time(ms)', 'Start time(HH:MM:SS)',
                                       'End time(HH:MM:SS)'])
    if output_dir is not None:
        df_compact.to_csv(join(output_dir, 'result_compact_loudness.csv'), index=False, header=True)
        # Output 'result.csv' similar to diarization for backward compatibility, can be removed if not longer needed
        df.to_csv(join(output_dir, 'result_loudness.csv'), index=False, header=True)
    return df_compact


def get_hms(ms):
    import datetime
    td = datetime.timedelta(milliseconds=ms)
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)


"""
Archived
"""


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


def noise_cancel_test():
    """
    何とかしてPinmicのノイズを取り除こうとしたけど、やはりうまくいかない。元の音声も削られてしまう。
    :return:
    """
    from pydub.silence import detect_silence
    audio = AudioSegment.from_file("test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice2_short.wav")
    dbfs = audio.dBFS
    print(dbfs)
    silence = detect_silence(audio, silence_thresh=dbfs)
    for start, end in silence:
        audio = audio[:start] + AudioSegment.silent(duration=end - start) + audio[end:]
    audio.export("noise_cancel.wav", format='wav')


def speech_segmenter_only(audio_path_list, output_dir=None):
    """
    DiarizationしなくてもそれぞれのPinmic音声のSpeech領域を抽出して、そのままS2Tすればいいはずだけど、
    実際の実験データは周りの人の音声も少し混ざっている。そのままS2Tすると、こういう問題が起きる。例えば、
    ======================================================================================
    話者2	受け入れられるように感じるので それだったらそれに対応してよ 嫌いと言うか分かりそうで 20台よりもさらに 若い世代とかファミリーで行けるような分かりやすさを重視した
    話者0	受け入れられるようにも感じるので それだったら; それに対応してより雷 塗装
    話者3	外に出られるようにも感じるので それだったらそれに対応してより 意外とそうと言うか分かりそうで逃げたよりもさらに 若い 9代とかファミリーで受け入れられてもらえるような分かりやすさを重視した結果
    話者0	20代よりもさらに若い十代とか; ファミリーで受け入れられてもらえるようなそのわかりやすさを重視したっていうか
    ======================================================================================
    本当は誰が喋っているか分からなくなってしまう。
    :param audio_path_list:
    :param output_dir:
    :return:
    """
    from inaSpeechSegmenter import Segmenter
    seg = Segmenter(vad_engine='smn', detect_gender=False)

    speech_segment_list = []
    for speaker_class, path in enumerate(audio_path_list):
        speech_segment = list(filter(lambda x: x[0] == 'speech', seg(path)))
        for interval in speech_segment:
            speech_segment_list.append(
                [speaker_class, interval[1] * 1000, interval[2] * 1000, get_hms(interval[1] * 1000),
                 get_hms(interval[2] * 1000)])

    speech_segment_list = sorted(speech_segment_list, key=lambda x: x[1])

    df_compact = pd.DataFrame(speech_segment_list,
                              columns=['Speaker', 'Start time(ms)', 'End time(ms)', 'Start time(HH:MM:SS)',
                                       'End time(HH:MM:SS)'])
    if output_dir is not None:
        df_compact.to_csv(join(output_dir, 'result_compact.csv'), index=False, header=True)
    return df_compact


def speech_segmenter_only_v2(audio_path_list, output_dir=None):
    """
    上記speech_segmenter_onlyの改良版。各Pinmic音声から抽出されたSpeech領域で完全に重なる領域があれば、それを除去するように改良した。
    それでも、完全に除去できないことがある。例えば、
    ======================================================================================
    話者３　0:07:13	0:07:33	自分も大丈夫三宅 考えた時にどこを狙うかるって考えたらま 20台とかその辺のお金持ってる人を狙うのだとしたらやっぱり 深夜枠のアニメ
    話者４　0:07:14	0:08:12	自分も最初 売上 考えた時にどこを狙うかって考えたら; 20代とかその辺の; お金持ってる層を狙うのだとしたらやっぱり; 深夜枠のアニメ; かなって思ったんですけど; ; それよりはやっぱり その; まさきアニメが; なんとなく; 受け入れられてるようにも感じるので それだったら; それに対応して寄りたい; 早々と言うか; 若いそうで 20台よりもさらに若い10代とか; ファミリーで受け入れられてもらえるようなその; わかりやすさを重視したっていうか; なるべく 万人に受けるようなもの の方が; いいのかな と思います
    ======================================================================================
    この場合、Timeframeが完全に重なっていないため、今回の除去対象外になっている。
    また、これは話者間の対話で音声のOverlapping（重なり）が存在すると想定した手法である。
    :param audio_path_list:
    :param output_dir:
    :return:
    """
    from inaSpeechSegmenter import Segmenter
    seg = Segmenter(vad_engine='smn', detect_gender=False)

    speech_segment_list = []
    for speaker_class, path in enumerate(audio_path_list):
        speech_segment = list(filter(lambda x: x[0] == 'speech', seg(path)))
        for interval in speech_segment:
            speech_segment_list.append((speaker_class, interval[1] * 1000, interval[2] * 1000,
                                        get_hms(interval[1] * 1000), get_hms(interval[2] * 1000)))

    speech_segment_list = sorted(speech_segment_list, key=lambda x: x[1])
    remove_list = []
    for interval in speech_segment_list:
        remove_list += list(
            filter(lambda x: x != interval and is_fully_overlap(x[1:], interval[1:]), speech_segment_list))
    speech_segment_list = list(filter(lambda x: x not in remove_list, speech_segment_list))

    df_compact = pd.DataFrame(speech_segment_list,
                              columns=['Speaker', 'Start time(ms)', 'End time(ms)', 'Start time(HH:MM:SS)',
                                       'End time(HH:MM:SS)'])
    if output_dir is not None:
        df_compact.to_csv(join(output_dir, 'result_compact.csv'), index=False, header=True)
    return df_compact


def is_fully_overlap(range_a, range_b):
    """ Check if range_a is inside of range_b """
    start_a, end_a = range_a
    start_b, end_b = range_b
    return start_b <= start_a and end_a <= end_b


if __name__ == "__main__":
    # Prepare short ver. for testing
    # for path in ["test/wave/200225_芳賀先生_実験23/200225_芳賀先生_実験23voice{}.wav".format(i) for i in range(1, 7)]:
    #     trim_audio(path, path[:-4] + '_short.wav', [0, 300000])
    # variant = '_loudness_30sec'
    # variant = '_uis_rnn_30sec'
    # variant = '_spectral_30sec'
    # variant = ''
    # variant = '_uis_rnn_30sec_loudness'
    # transcript('output_exp22' + variant,
    #            ["test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice{}.wav".format(i) for i in range(1, 6)],
    #            config)
    # transcript('output_exp23' + variant,
    #            ["test/wave/200225_芳賀先生_実験23/200225_芳賀先生_実験23voice{}.wav".format(i) for i in range(1, 7)],
    #            config)
    #
    # variant = '_uis_rnn_30sec_loudness_0.5_0.5'
    # transcript('output_exp22' + variant,
    #            ["test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice{}.wav".format(i) for i in range(1, 6)],
    #            config)
    #
    # config['embedding_per_second'] = 1.0
    # config['overlap_rate'] = 0.5
    # variant = '_uis_rnn_30sec_loudness_1.0_0.5'
    # transcript('output_exp22' + variant,
    #            ["test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice{}.wav".format(i) for i in range(1, 6)],
    #            config)
    #
    # config['embedding_per_second'] = 0.5
    # config['overlap_rate'] = 0.7
    # variant = '_uis_rnn_30sec_loudness_0.5_0.7'
    # transcript('output_exp22' + variant,
    #            ["test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice{}.wav".format(i) for i in range(1, 6)],
    #            config)
    pass
