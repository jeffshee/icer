# -*- coding: utf-8 -*-
import datetime
import glob
import multiprocessing
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import face_recognition
from keras.models import model_from_json
from keras.backend.tensorflow_backend import set_session

from capture_face import detect_face, cluster_face
from edit_audio import extract_audio, set_audio, mix_audio
from edit_video import concat_video
from emotion_recognition import cleanup_directory, emotion_recognition
from main_util import concat_csv
from match_speaker import match_speaker
from overlay_video import overlay_all_results
from run_speaker_diarization_v2 import run_speaker_diarization

if __name__ == '__main__':
    # tf v2
    # config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    # session = tf.Session(config=config)
    # set_session(session)
    implement_mode = {
        "detect_face": 1,  # 動画から顔領域の切り出し
        "emotional_recognition": 1,  # 切り出した顔画像の表情・頷き・口の開閉認識
        "diarization": 1,  # 音声から話者識別
        "overlay": 1  # 表情・頷き・発話情報を動画にまとめて可視化
    }

    # video_name_list = [["191031_Haga_Trim"], ["Take02"], ["Take01_copycopy"]][2]
    # 処理する動画のリスト
    # video_name_list = ["200225_Haga_22", "200225_Haga_23"]
    video_name_list = ["expt23", "expt22"]
    # 処理する音声のリスト（use_mix_audioがTrueの場合）
    # audio_name_list = [["200225_Haga_22_voice1", "200225_Haga_22_voice2", "200225_Haga_22_voice3",
    #                     "200225_Haga_22_voice4", "200225_Haga_22_voice5"],
    #                    ["200225_Haga_23_voice1", "200225_Haga_23_voice2", "200225_Haga_23_voice3",
    #                     "200225_Haga_23_voice4", "200225_Haga_23_voice5", "200225_Haga_23_voice6"]]
    # audio_name_list = [["200225_Haga_22_voice1_trim", "200225_Haga_22_voice2_trim", "200225_Haga_22_voice3_trim", "200225_Haga_22_voice4_trim", "200225_Haga_22_voice5_trim"]]
    audio_name_list = [["expt23", "expt22"]]
    # audio_name_list = [["200225_Haga_22_voice1_trim", "200225_Haga_22_voice2_trim", "200225_Haga_22_voice3_trim", "200225_Haga_22_voice4_trim", "200225_Haga_22_voice5_trim"]]
    # audio_name_listの音声の拡張子
    audio_format = ".wav"

    # ハイパーパラメータ
    capture_image_num = 200  # 0~capture_image_numフレーム目からdetect_faceで顔検出
    face_matching_tolerance = 0.35  # 顔認識を行う際の距離の閾値
    frame_use_rate = 3  # frame_use_rate フレームに１回emotional_recognition
    k_resolution = 3  # k_resolutionに動画の横幅をリサイズ
    split_video_num = 10  # split_video_num 並列で動画を処理する．10くらいがメモリ・CPUの限界
    use_mix_audio = True  # 動画から音声を抽出するか、音声をミックスするか

    # PATH
    path_audio = "audio/"
    path_face = "face_image/"
    path_video = "video/"

    for video_index, video_name in enumerate(video_name_list):
        start = time.time()
        print("Video name:", video_name)
        print("Implement mode:", implement_mode, "\n")

        target_video_path = "{}{}".format(path_video, video_name) + ".mp4"
        clustered_face_path = "{}{}/".format(path_face, video_name)

        # 顔画像切り出し
        if implement_mode["detect_face"]:
            cluster_num = detect_face(target_video_path, clustered_face_path, k_resolution,
                                    capture_image_num)  # 顔画像を切り出す
            cluster_face(clustered_face_path, clustered_face_path + "cluster/",
                        cluster_num=cluster_num)  # 顔画像をcluster_numクラスタにクラスタリングする
            print("顔画像切り出し完了．\nPath:{} を確認して，対象外人物のフォルダを削除してください．\n完了した場合，何か入力してください．".format(clustered_face_path + "cluster/"))
            input_str = input()  # 入力待ち
            print(input_str)

        # 解析対象人物の顔画像を読み込む
        target_image_path_list = []
        for i in os.listdir(path_face + video_name + "/cluster/"):
            target_image_path_list.append(glob.glob(path_face + video_name + "/cluster/{}/*".format(i)))
        result_path = "result/{}_{}_{}_{}_{}people/".format(datetime.date.today(), video_name, k_resolution,
                                                            frame_use_rate, len(target_image_path_list))
        target_people_num = len(target_image_path_list)

        # 感情認識
        if implement_mode["emotional_recognition"]:
            # 顔画像のテンプレートを読み込む
            known_faces = []
            for file_path in target_image_path_list:
                face_image = [face_recognition.load_image_file(x) for x in file_path]
                face_image_encoded = [face_recognition.face_encodings(x)[0] for x in face_image]
                known_faces.append(face_image_encoded)

            # 感情認識モデルの読み込み
            path_model = "model/"
            model_name = "mini_XCEPTION"
            model = model_from_json(open("{}model_{}.json".format(path_model, model_name), "r").read())
            model.load_weights('{}model_{}.h5'.format(path_model, model_name))
            emotions = ('Negative', 'Negative', 'Normal', 'Positive', 'Normal', 'Normal',
                        'Normal')  # emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')

            # 感情認識（並列処理）
            cleanup_directory(result_path)
            process_list = []
            for split_video_index in range(split_video_num):
                split_result_path = "{}split_video_{}/".format(result_path, split_video_index)
                cleanup_directory(split_result_path)
                process_list.append(multiprocessing.Process(target=emotion_recognition, args=(
                    target_video_path, split_result_path, k_resolution, frame_use_rate, face_matching_tolerance,
                    target_image_path_list, model, model_name, emotions, split_video_index, split_video_num,
                    known_faces)))
                process_list[split_video_index].start()
            for process in process_list:
                process.join()

            # 分割して処理した結果を結合
            input_video_list = ["{}split_video_{}/output.avi".format(result_path, i) for i in range(split_video_num)]
            concat_video(input_video_list, result_path + "output.avi")
            for j in range(len(target_image_path_list)):
                input_csv_list = ["{}split_video_{}/result{}.csv".format(result_path, i, j) for i in
                                range(split_video_num)]
                concat_csv(input_csv_list, result_path + "result{}.csv".format(j))

        # 話者分離 (Diarization)
        # オリジナルのビデオから音声を抽出
        embedding_per_second = 0.5
        overlap_rate = 0.5
        path_diarization_result = "diarization/{}_{}_{}/".format(video_name, embedding_per_second, overlap_rate)
        if implement_mode["diarization"]:
            if use_mix_audio:
                mixed_audio = "{}{}.wav".format(path_audio, video_name)
                input_audio_list = [path_audio + filename + audio_format for filename in audio_name_list[video_index]]
                mix_audio(input_audio_list, mixed_audio)
            else:
                extracted_audio = "{}{}.wav".format(path_audio, video_name)
                extract_audio(target_video_path, extracted_audio)
            run_speaker_diarization(path_audio, video_name, path_diarization_result, people_num=target_people_num,
                                    embedding_per_second=embedding_per_second, overlap_rate=overlap_rate,
                                    noise_class_num=3)

        # emotion_recognitionとdiarization結果を結合した情報を動画中にまとめて表示
        if implement_mode["overlay"]:
            input_audio = "{}{}.wav".format(path_audio, video_name)
            clustered_face_list = os.listdir(clustered_face_path + "cluster/")

            index_to_name_dict = {-1: "unknown"}
            index_to_name_dict.update({i: val for i, val in enumerate(clustered_face_list)})
            faces_path = ["{}{}/{}/{}".format(clustered_face_path, "cluster", name, "closest.PNG") for name in
                        clustered_face_list]

            # Path等の設定
            input_movie_path = "{}output.avi".format(result_path)
            output_movie_path = "{}output_overlay.avi".format(result_path)

            # 議論の可視化動画を作成（並列処理）
            matching_index_list, true_face_num = match_speaker(path_diarization_result, result_path, th_matching=0.0)
            emotions_overlay = ['Negative', 'Normal', 'Positive', 'Unknown']
            overlay_all_results(input_movie_path, output_movie_path, path_diarization_result, matching_index_list,
                                true_face_num, result_path, faces_path, video_name, index_to_name_dict,
                                emotions_overlay, k_resolution, split_video_num, path_audio)

            # 議論可視化動画と音声を結合
            output_movie_with_audio = output_movie_path.replace(".avi", "_audio.mp4")
            set_audio(input_video=output_movie_path, input_audio=input_audio, output_video=output_movie_with_audio)

        print("elapsed_time:{0}".format(time.time() - start) + "[sec]")
