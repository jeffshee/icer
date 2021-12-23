"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""
# ===========================================
#        Parse the argument
# ===========================================
import argparse
import csv
import os
import multiprocessing

import librosa
import numpy as np
from spectralcluster import SpectralClusterer

from s2t_utils.speaker_diarization_v2.uisrnn import parse_arguments
from s2t_utils.speaker_diarization_v2.uisrnn import uisrnn
# from s2t_utils.speaker_diarization_v2.visualization.viewer import PlotDiar

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--resume', default=r's2t_utils/speaker_diarization_v2/ghostvlad/pretrained/weights.h5', type=str)
parser.add_argument('--data_path', default='4persons', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

args = parser.parse_args()

SAVED_MODEL_NAME = 's2t_utils/speaker_diarization_v2/pretrained/saved_model.uisrnn_benchmark'


def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    timeDict = {}
    timeDict['start'] = int(value[0] + 0.5)
    timeDict['stop'] = int(value[1] + 0.5)
    if (key in speakerSlice):
        speakerSlice[key].append(timeDict)
    else:
        speakerSlice[key] = [timeDict]

    return speakerSlice


def arrangeResult(labels, time_spec_rate):
    # {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, 'stop':100}]}
    lastLabel = labels[0]
    speakerSlice = {}
    j = 0
    for i, label in enumerate(labels):
        if (label == lastLabel):
            continue
        speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate * j, time_spec_rate * i)})
        j = i
        lastLabel = label
    speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate * j, time_spec_rate * (len(labels)))})
    return speakerSlice


def genMap(intervals):  # interval slices to maptable
    slicelen = [sliced[1] - sliced[0] for sliced in intervals.tolist()]
    mapTable = {}  # vad erased time to origin time, only split points
    idx = 0
    for i, sliced in enumerate(intervals.tolist()):
        mapTable[idx] = sliced[0]
        idx += slicelen[i]
    mapTable[sum(slicelen)] = intervals[-1, -1]

    keys = [k for k, _ in mapTable.items()]
    keys.sort()
    return mapTable, keys


def fmtTime(timeInMillisecond):
    millisecond = timeInMillisecond % 1000
    minute = timeInMillisecond // 1000 // 60
    second = (timeInMillisecond - minute * 60 * 1000) // 1000
    time = '{}:{:02d}.{}'.format(minute, second, millisecond)
    return time


# Modified
def load_wav(vid_path, sr, specific_intervals=None):
    wav, _ = librosa.load(vid_path, sr=sr)
    if specific_intervals is None:
        intervals = librosa.effects.split(wav, top_db=20)
    else:
        intervals = specific_intervals
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    return np.array(wav_output), (intervals / sr * 1000).astype(int)


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
    return linear.T


# 0s        1s        2s                  4s                  6s
# |-------------------|-------------------|-------------------|
# |-------------------|
#           |-------------------|
#                     |-------------------|
#                               |-------------------|
# Modified
def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5, overlap_rate=0.5,
              specific_intervals=None):
    wav, intervals = load_wav(path, sr=sr, specific_intervals=specific_intervals)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    spec_len = sr / hop_length / embedding_per_second
    spec_hop_len = spec_len * (1 - overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while (True):  # slide window.
        if (cur_slide + spec_len > time):
            break
        spec_mag = mag_T[:, int(cur_slide + 0.5): int(cur_slide + spec_len + 0.5)]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return utterances_spec, intervals


class Predictor(multiprocessing.Process):
    """
    Make tensorflow/keras task into a Process, so that the resource will be automatically freed when the task is done.
    https://github.com/tensorflow/tensorflow/issues/8220
    The hang was caused by Queue actually, refer below. Use Manager instead.
    https://stackoverflow.com/questions/31708646/process-join-and-queue-dont-work-with-large-numbers
    https://stackoverflow.com/questions/33825054/deadlock-with-multiprocessing-module
    """

    def __init__(self, wav_path, embedding_per_second, overlap_rate, return_dict):
        multiprocessing.Process.__init__(self)
        self.wav_path = wav_path
        self.embedding_per_second = embedding_per_second
        self.overlap_rate = overlap_rate
        self.return_dict = return_dict
        self.gpu_id = args.gpu

    def run(self):
        # Set GPU id before importing tensorflow!!!!!!!!!!!!!
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.gpu_id)
        # Import tensorflow here
        import tensorflow as tf
        if tf.__version__.startswith('2'):
            sess = tf.compat.v1.Session()
        else:
            sess = tf.Session()
        # print('Using device #%s' % self.gpu_id)
        # Task
        import s2t_utils.speaker_diarization_v2.ghostvlad.model as spkModel

        params = {'dim': (257, None, 1),
                  'nfft': 512,
                  'spec_len': 250,
                  'win_length': 400,
                  'hop_length': 160,
                  'n_classes': 5994,
                  'sampling_rate': 16000,
                  'normalize': True,
                  }
        print("Evaluating using spkModel")
        network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                       num_class=params['n_classes'],
                                                       mode='eval', args=args)
        network_eval.load_weights(args.resume, by_name=True)

        specs, intervals = load_data(self.wav_path, embedding_per_second=self.embedding_per_second,
                                     overlap_rate=self.overlap_rate)
        mapTable, keys = genMap(intervals)
        self.return_dict["mapTable"] = mapTable
        self.return_dict["keys"] = keys

        feats = []
        for spec in specs:
            spec = np.expand_dims(np.expand_dims(spec, 0), -1)
            v = network_eval.predict(spec)
            feats += [v]

        feats = np.array(feats)[:, 0, :].astype(float)  # [splits, embedding dim]
        self.return_dict["feats"] = feats
        sess.close()
        return


def main(wav_path, path_result, embedding_per_second=1.0, overlap_rate=0.5, use_spectral_cluster=False, min_clusters=1,
         max_clusters=100):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = Predictor(wav_path, embedding_per_second, overlap_rate, return_dict)
    p.start()
    p.join()
    mapTable = return_dict["mapTable"]
    keys = return_dict["keys"]
    feats = return_dict["feats"]
    print("Done.")

    if use_spectral_cluster:
        print("Evaluating using SpectralClusterer")
        clusterer = SpectralClusterer(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            p_percentile=0.95,
            gaussian_blur_sigma=1)
        predicted_label = clusterer.predict(feats)
    else:
        print("Evaluating using UISRNN")
        model_args, _, inference_args = parse_arguments()
        model_args.observation_dim = 512
        uisrnnModel = uisrnn.UISRNN(model_args)
        uisrnnModel.load(SAVED_MODEL_NAME)
        predicted_label = uisrnnModel.predict(feats, inference_args)

    print("Done. Generating result.")
    time_spec_rate = 1000 * (1.0 / embedding_per_second) * (
            1.0 - overlap_rate)  # speaker embedding every time_spec_rate ms
    speakerSlice = arrangeResult(predicted_label, time_spec_rate)

    for spk, timeDicts in speakerSlice.items():  # time map to orgin wav(contains mute)
        for tid, timeDict in enumerate(timeDicts):
            s = 0
            e = 0
            for i, key in enumerate(keys):
                if (s != 0 and e != 0):
                    break
                if (s == 0 and key > timeDict['start']):
                    offset = timeDict['start'] - keys[i - 1]
                    s = mapTable[keys[i - 1]] + offset
                if (e == 0 and key > timeDict['stop']):
                    offset = timeDict['stop'] - keys[i - 1]
                    e = mapTable[keys[i - 1]] + offset

            speakerSlice[spk][tid]['start'] = s
            speakerSlice[spk][tid]['stop'] = e

    for spk, timeDicts in speakerSlice.items():
        print('========= ' + str(spk) + ' =========')
        for timeDict in timeDicts:
            s = timeDict['start']
            e = timeDict['stop']
            s = fmtTime(s)  # change point moves to the center of the slice
            e = fmtTime(e)
            print(s + ' ==> ' + e)

    os.makedirs(path_result, exist_ok=True)

    # # 結果を画像として保存
    # p = PlotDiar(map=speakerSlice, wav=wav_path, gui=False, size=(25, 6))
    # p.draw()
    # p.plot.savefig("{}result.png".format(path_result))

    # 結果をCSVファイルに保存
    with open("{}result.csv".format(path_result), "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")  # writerオブジェクトの作成 改行記号で行を区切る
        writer.writerow(["time(ms)", "speaker class"])
        for spk, timeDicts in speakerSlice.items():
            for timeDict in timeDicts:
                for frame_ms in range(timeDict['start'], timeDict['stop']):
                    writer.writerow([frame_ms, spk])
