# -*- coding: utf-8 -*-
import multiprocessing
import os
import time

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from os.path import join
import tensorflow as tf
import face_recognition

from capture_face import detect_face, cluster_face
from edit_audio import set_audio
from edit_video import concat_video
from emotion_recognition import cleanup_directory, emotion_recognition
from main_util import concat_csv
from match_speaker import match_speaker
from overlay_video import overlay_all_results
from transcript import transcript

config = {
    # NOTE:（　）の中はデフォルト

    # Run mode
    'run_capture_face': False,  # 動画から顔領域の切り出し
    'run_emotional_recognition': False,  # 切り出した顔画像の表情・頷き・口の開閉認識
    'run_transcript': False,  # Diarization・音声認識
    'run_overlay': True,  # 表情・頷き・発話情報を動画にまとめて可視化

    # capture_face関連設定
    'k_resolution': 3,  # k_resolutionに動画の横幅をリサイズ
    'capture_image_num': 200,  # 0~capture_image_numフレーム目からdetect_faceで顔検出

    # 感情認識関連設定
    'face_matching_tolerance': 0.35,  # 顔認識を行う際の距離の閾値
    'frame_use_rate': 3,  # frame_use_rate フレームに１回emotional_recognition
    'split_video_num': 10,  # split_video_num 並列で動画を処理する．10くらいがメモリ・CPUの限界

    # Debug関連設定
    'no_clean': True,  # (Debug) Use old result (False)
}


def init_tf_v1():
    if tf.__version__.startswith('1'):
        from keras.backend import set_session
        session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        set_session(session)


def run_capture_face(video_path, face_dir, cluster_dir, main_config):
    # 顔画像を切り出す
    cluster_num = detect_face(video_path, face_dir, main_config['k_resolution'], main_config['capture_image_num'])
    # 顔画像をcluster_numクラスタにクラスタリングする
    # TODO maybe set cluster_num to the length of audio_path_list could be better?
    cluster_face(face_dir, cluster_dir, cluster_num=cluster_num)
    print('顔画像切り出し完了．\nPath: {} を確認して，対象外人物のフォルダを削除してください．\n完了した場合，何か入力してください．'.format(
        cluster_dir))
    input()  # 入力待ち


def run_emotional_recognition(video_path, cluster_image_path_list, output_dir, main_config):
    known_faces = []
    for path_list in cluster_image_path_list:
        face_images = [face_recognition.load_image_file(path) for path in path_list]
        face_images_encoded = [face_recognition.face_encodings(x)[0] for x in face_images]
        known_faces.append(face_images_encoded)

    # 感情認識モデルの読み込み
    model_dir = 'model'
    model_name = 'mini_XCEPTION'
    # model = model_from_json(open(join(model_dir, 'model_{}.json'.format(model_name)), 'r').read())
    # model.load_weights(join(model_dir, 'model_{}.h5'.format(model_name)))
    model = None
    # emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')
    emotions = ('Negative', 'Negative', 'Normal', 'Positive', 'Normal', 'Normal', 'Normal')

    # 感情認識（並列処理）
    process_list = []
    split_result_dir_list = []
    split_video_path_list = []
    for split_video_index in range(main_config['split_video_num']):
        split_result_dir = join(output_dir, 'temp_{}'.format(split_video_index))
        split_result_dir_list.append(split_result_dir)
        split_video_path_list.append(join(split_result_dir, 'output.avi'))
        cleanup_directory(split_result_dir)
        process_list.append(multiprocessing.Process(target=emotion_recognition, args=(
            video_path, split_result_dir, main_config['k_resolution'], main_config['frame_use_rate'],
            main_config['face_matching_tolerance'],
            cluster_image_path_list, model, model_name, emotions, split_video_index, main_config['split_video_num'],
            known_faces)))
        process_list[split_video_index].start()
    for process in process_list:
        process.join()

    # 分割して処理した結果を結合
    concat_video(split_video_path_list, join(output_dir, 'output.avi'))
    for j in range(len(cluster_image_path_list)):
        split_csv_path_list = [join(output_dir, 'temp_{}', 'result{}.csv').format(i, j) for i in
                               range(main_config['split_video_num'])]
        concat_csv(split_csv_path_list, join(output_dir, 'result{}.csv'.format(j)))

    # Remove `temp*` directory, we don't need them anymore
    # import shutil
    # for d in split_result_dir_list:
    #     shutil.rmtree(d)


def run_transcript(output_dir, audio_path_list):
    transcript(output_dir, audio_path_list)


def run_overlay(output_dir, cluster_dir, emotion_dir, transcript_dir, main_config):
    clustered_face_list = os.listdir(cluster_dir)

    index_to_name_dict = {-1: 'unknown'}
    index_to_name_dict.update({i: val for i, val in enumerate(clustered_face_list)})
    faces_path = [join(cluster_dir, name, 'closest.PNG') for name in clustered_face_list]

    # Path等の設定
    input_video_path = join(emotion_dir, 'output.avi')
    input_audio_path = join(transcript_dir, 'mixed_audio.wav')
    diarization_result_dir = join(transcript_dir, 'diarization')
    output_video_path = join(output_dir, 'overlay.avi')

    # TODO
    # 議論の可視化動画を作成（並列処理）
    video_name = "video"
    diarization_result_path = join(diarization_result_dir, 'result_loudness.csv')
    transcript_result_path = join(transcript_dir, 'transcript.csv')
    matching_index_list, true_face_num = match_speaker(diarization_result_path, emotion_dir, th_matching=0.0)
    emotions_overlay = ['Negative', 'Normal', 'Positive', 'Unknown']
    overlay_all_results(input_video_path, output_video_path, diarization_result_path, transcript_result_path,
                        matching_index_list,
                        true_face_num, emotion_dir, faces_path, video_name, index_to_name_dict,
                        emotions_overlay, main_config['k_resolution'], main_config['split_video_num'], output_dir)

    # 議論可視化動画と音声を結合
    output_video_with_audio_path = output_video_path.replace('.avi', '_audio.mp4')
    set_audio(input_video=output_video_path, input_audio=input_audio_path, output_video=output_video_with_audio_path)


def main(video_path, audio_path_list, output_dir, main_config=None):
    if main_config is None:
        main_config = config
    init_tf_v1()

    # Directories
    face_dir = join(output_dir, 'face')
    cluster_dir = join(face_dir, 'cluster')
    emotion_dir = join(output_dir, 'emotion')
    transcript_dir = join(output_dir, 'transcript')
    overlay_dir = join(output_dir, 'overlay')

    if not config['no_clean']:
        cleanup_directory(output_dir)  # The root directory for current task

    if config['run_capture_face']:
        start = time.time()
        run_capture_face(video_path, face_dir, cluster_dir, main_config)
        print('run_capture_face elapsed time:', time.time() - start, '[sec]')

    if config['run_emotional_recognition']:
        start = time.time()
        cluster_image_path_list = []
        for cluster in os.listdir(cluster_dir):
            cluster_root = join(cluster_dir, cluster)
            img_list = os.listdir(cluster_root)
            cluster_image_path_list.append([join(cluster_root, img) for img in img_list])
        run_emotional_recognition(video_path, cluster_image_path_list, emotion_dir, main_config)
        print('run_emotional_recognition elapsed time:', time.time() - start, '[sec]')

    if config['run_transcript']:
        start = time.time()
        run_transcript(transcript_dir, audio_path_list)
        print('run_transcript elapsed time:', time.time() - start, '[sec]')

    if config['run_overlay']:
        start = time.time()
        run_overlay(overlay_dir, cluster_dir, emotion_dir, transcript_dir, main_config)
        print('run_overlay elapsed time:', time.time() - start, '[sec]')

    print('Elapsed time:', time.time() - start, '[sec]')


if __name__ == '__main__':
    # Prepare short ver. for testing
    # from edit_video import trim_video
    # from edit_audio import trim_audio
    #
    # input_video_path = 'test/200225_芳賀先生_実験22video.mp4'
    # output_video_path = 'test/200225_芳賀先生_実験22video_10min.mp4'
    # trim_video(input_video_path, ['00:00:00', '00:10:00'], output_video_path)  # 'hh:mm:ss'
    # for path in ["test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice{}.wav".format(i) for i in range(1, 6)]:
    #     trim_audio(path, path[:-4] + '_10min.wav', [0, 10 * 60 * 1000])

    main('test/200225_芳賀先生_実験22video_10min.mp4',
         ["test/wave/200225_芳賀先生_実験22/200225_芳賀先生_実験22voice{}_10min.wav".format(i) for i in range(1, 6)], 'main_test')
