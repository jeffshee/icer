# -*- coding: utf-8 -*-
# 必要モジュールをインポート
import csv
import datetime
import glob
import multiprocessing
import os
import time
from os.path import join

import cv2
import face_recognition
import numpy as np
from keras.models import model_from_json
from scipy.spatial import distance as dist

from main_util import concat_csv
from deprecated.edit_video import concat_video
from main_util import cleanup_directory


# int型の座標を取得
def get_coords(p1):
    try:
        return int(p1[0][0][0]), int(p1[0][0][1])
    except IndexError:
        return int(p1[0][0]), int(p1[0][1])


# MAR: Mouse Aspect Ratio
# https://medium.freecodecamp.org/smilfie-auto-capture-selfies-by-detecting-a-smile-using-opencv-and-python-8c5cfb6ec197
def calculate_MAR(landmarks):
    A = dist.euclidean(landmarks["top_lip"][2], landmarks["bottom_lip"][4])
    B = dist.euclidean(landmarks["top_lip"][3], landmarks["bottom_lip"][3])
    C = dist.euclidean(landmarks["top_lip"][4], landmarks["bottom_lip"][2])
    L = (A + B + C) / 3  # ABCの平均→縦
    D = dist.euclidean(landmarks["top_lip"][0], landmarks["bottom_lip"][0])  # 横
    mar = L / D
    return mar


def box_label(bgr, x1, y1, x2, y2, label, color):
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)  # 顔を囲む箱
    cv2.rectangle(bgr, (int(x1), int(y1 - 18)), (x2, y1), (255, 255, 255), -1)  # テキスト記入用の白いボックス
    cv2.putText(bgr, label, (x1, int(y1 - 5)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)


def my_compare_faces(known_faces, face_encoding_to_check):
    distance_list = []
    if len(known_faces) == 0:
        return np.empty((0))

    for known_face_list in known_faces:
        distance_know_face_list = list(np.linalg.norm(known_face_list - face_encoding_to_check, axis=1))
        distance_list.append(min(distance_know_face_list))

    return distance_list


# 配列のx値、y値のリスト作成
# b=Trueでx、Falseでy
def get_parts_coordinates(a, b):
    l = list()
    if b == True:
        for w in a:
            l.append(w[0])
    elif b == False:
        for w in a:
            l.append(w[1])
    return l

# TODO
# config = {
#     'model_path': join('model', 'mini_XCEPTION'),
#     'k_resolution': ,
#     'frame_use_rate': ,
#     'face_matching_th': ,
#     'emotions':,
#     'split_video_num':,
# }
def emotion_recognition(target_video_path, path_result, k_resolution, frame_use_rate, face_matching_th,
                        target_image_path_list, model, model_name, emotions, split_video_index, split_video_num,
                        known_faces):
    from keras.models import model_from_json
    model_dir = '../model'
    model_name = 'mini_XCEPTION'
    model = model_from_json(open(join(model_dir, 'model_{}.json'.format(model_name)), 'r').read())
    model.load_weights(join(model_dir, 'model_{}.h5'.format(model_name)))

    input_movie = cv2.VideoCapture(target_video_path)  # 動画を読み込む
    video_frame_rate = input_movie.get(cv2.CAP_PROP_FPS)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画の長さを測る

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画コーデック指定
    original_w = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))  # 動画の幅を測る
    original_h = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 動画の高さを測る
    resize_rate = (1080 * k_resolution) / original_w  # 動画の横幅を3Kに固定
    w = int(original_w * resize_rate)
    h = int(original_h * resize_rate)
    output_movie = cv2.VideoWriter(join(path_result, 'output.avi'), fourcc, video_frame_rate, (w, h))  # 出力する動画の詳細を設定する

    # CSVファイルの最初の列
    csv_saving_list = [[["frame_number", "time(ms)", "prediction", "x", "y", "absolute_x", "relative_x", "x_movement",
                         "absolute_y", "relative_y", "y_movement", "mouse opening", "gesture", "top", "bottom", "left",
                         "right"]] for _ in range(len(target_image_path_list))]

    # 各変数を初期化
    # 前フレームの顔の座標を記録する変数
    rec_face = [[0, 0, 0, 0]] * len(target_image_path_list)
    # 顔の中心を示す変数
    # 顔の中央のオプティカルフロー計算に用いる変数
    p0 = [np.array([0, 0])] * len(target_image_path_list)
    p1 = [np.array([0, 0])] * len(target_image_path_list)
    # 首を動かしたかどうかを判断する変数
    gesture = [0] * len(target_image_path_list)
    # オプティカルフローの移動量
    x_movement = [0] * len(target_image_path_list)
    y_movement = [0] * len(target_image_path_list)
    # 座標
    x = [0] * len(target_image_path_list)
    y = [0] * len(target_image_path_list)
    # 動作判定の表示時間
    gesture_show_frame = 30
    gesture_show = [gesture_show_frame] * len(target_image_path_list)
    # 顔が認識できたかどうか
    face_recog_flag = [False] * len(target_image_path_list)
    # 初めて顔を認識したかどうか
    first_face_recog_flag = [False] * len(target_image_path_list)
    # オプティカルフローの前フレームからの移動量の絶対値
    absolute_x = [0] * len(target_image_path_list)
    absolute_y = [0] * len(target_image_path_list)
    # オプティカルフローの前フレームからの移動量の相対値
    relative_x = [0] * len(target_image_path_list)
    relative_y = [0] * len(target_image_path_list)
    # 目の中心位置座標
    eye_center = [[0, 0]] * len(target_image_path_list)
    # 感情
    emotion = ["Unknown"] * len(target_image_path_list)
    mouse_opening_rate_list = [0] * len(target_image_path_list)

    frame_duration = length // split_video_num
    start_frame = split_video_index * frame_duration
    stop_frame = length if split_video_index == split_video_num - 1 else start_frame + frame_duration
    input_movie.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 動画の開始フレームを設定
    first_flag = True
    for frame_number in range(start_frame, stop_frame):
        # 動画を読み込む
        ret, frame = input_movie.read()
        print("\nWriting frame {} / {}".format(frame_number, length - 1))

        # 動画が読み取れない場合は終了
        if not ret:
            break

        # フレームのサイズを調整する
        frame = cv2.resize(frame, (w, h))

        if frame_number % frame_use_rate == 0 or first_flag:

            # openCVのBGRをRGBに変更
            rgb_frame = frame[:, :, ::-1]

            # 顔領域の検出
            if frame_number % 30 == 0 or first_flag:  # 30フレームに一回のみ画像中の全領域から顔検出
                face_location_list = face_recognition.face_locations(rgb_frame,
                                                                     model="hog")  # model="cnn"にすると検出率は上がるが10倍以上時間がかかる top,right,bottom,leftの順
                if face_location_list:
                    min_top, max_bottom = min([x[0] for x in face_location_list]), max(
                        [x[2] for x in face_location_list])
                    dif_height = max_bottom - min_top
                    min_top = int(max(0, min_top - dif_height * 0.3))
                    max_bottom = int(min(h, max_bottom + dif_height * 0.3))
                else:
                    min_top = 0
                    max_bottom = h
            else:  # 処理時間削減のために領域を絞り込んで顔検出
                rgb_frame_trimmed = rgb_frame[min_top:max_bottom, :]
                face_location_list_trimmed = face_recognition.face_locations(rgb_frame_trimmed,
                                                                             model="hog")  # model="cnn"にすると検出率は上がるが10倍以上時間がかかる top,right,bottom,leftの順?
                face_location_list = []
                for face_location in face_location_list_trimmed:
                    face_location = list(face_location)
                    face_location[0] = face_location[0] + min_top
                    face_location[2] = face_location[2] + min_top
                    face_location_list.append(face_location)
                # print(face_location_list)

            face_encoded_list = face_recognition.face_encodings(rgb_frame, known_face_locations=face_location_list)

            # 入力画像とのマッチング
            face_distance_list = [[] for _ in range(len(known_faces))]
            for face_encoded in face_encoded_list:
                tmp_distance = my_compare_faces(known_faces, face_encoded)
                for i in range(len(face_distance_list)):
                    face_distance_list[i].append(tmp_distance[i])
            # print(face_distance_list)
            face_index_list = [None] * len(face_encoded_list)
            if face_location_list:
                for i in range(len(known_faces)):
                    if min(face_distance_list[i]) <= face_matching_th:
                        face_index = face_distance_list[i].index(min(face_distance_list[i]))
                        if face_index_list[face_index] is None:  # クエリ画像が検出顔のどれともマッチングしていないとき
                            face_index_list[face_index] = i
                        elif face_distance_list[face_index_list[face_index]].index(
                                min(face_distance_list[face_index_list[face_index]])) >= face_distance_list[i].index(
                            min(face_distance_list[i])):  # クエリ画像が検出顔のいずれかとマッチング済みであるとき
                            face_index_list[face_index] = i
            # print("face_index_list", face_index_list)

            # 顔パーツ領域の検出
            for (top, right, bottom, left), face_index in zip(face_location_list, face_index_list):
                if face_index is None:
                    continue

                rec_face[face_index] = [top, bottom, left, right]

                # 顔領域を切り取る
                dst = frame[top:bottom, left:right]

                # 表情認識
                if model_name == "mini_XCEPTION":
                    val = cv2.resize(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), (64, 64)) / 255.0
                    val = val.reshape(1, 64, 64, 1)
                else:
                    exit("Invalid model_name")

                # Solved https://stackoverflow.com/questions/60905801/parallelizing-model-predictions-in-keras-using-multiprocessing-for-python
                predictions = model.predict(val, batch_size=1)  # 学習モデルから表情を特定
                emotion[face_index] = emotions[int(np.argmax(predictions[0]))]
                box_label(frame, left, top, right, bottom, "{}:{}".format(face_index, emotion[face_index]),
                          color=(0, 255, 0))
                face_recog_flag[face_index] = True

                # 顔の部位の座標を検出する
                face_landmarks_list = face_recognition.face_landmarks(dst, face_locations=[
                    (0, dst.shape[1], dst.shape[0], 0)])  # 顔の部位の座標が取れないときは return []
                # 顔の部位を切り取る．画像中に複数顔が存在する場合を想定して返り値がlistになっている．
                for face_landmarks in face_landmarks_list:
                    # 独自の計算式から口の空き具合を計算
                    mouse_opening_rate_list[face_index] = calculate_MAR(face_landmarks)

                    # 目の位置
                    leye_x = get_parts_coordinates(face_landmarks["left_eye"], True)
                    reye_x = get_parts_coordinates(face_landmarks["right_eye"], True)
                    eye_y = get_parts_coordinates(face_landmarks["left_eye"] + face_landmarks["right_eye"], False)
                    topeye = min(eye_y)
                    bottomeye = max(eye_y)
                    lefteye = min(leye_x)
                    righteye = max(reye_x)
                    eye_center[face_index] = left + (righteye + lefteye) // 2, top + (topeye + bottomeye) // 2

                if not first_face_recog_flag[face_index]:
                    # face_center[face_index] = (left + right) / 2, (top + bottom) / 2
                    p0[face_index] = np.array([[eye_center[face_index]]], np.float32)
                    first_face_recog_flag[face_index] = True

            # 顔が読み取れなかった場合 感情＝不明とする
            for face_index, flag in enumerate(face_recog_flag):
                if not flag:
                    emotion[face_index] = "Unknown"

        else:
            for face_index in range(len(face_recog_flag)):
                if emotion[face_index] != "Unknown":
                    box_label(frame, int(rec_face[face_index][2]), int(rec_face[face_index][0]),
                              int(rec_face[face_index][3]), int(rec_face[face_index][1]),
                              "{}:{}".format(face_index, emotion[face_index]), color=(0, 255, 0))

        # 首の動作判定
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for face_index in range(len(known_faces)):
            if first_flag:  # 前フレームがない場合は終了
                break
            elif first_face_recog_flag[face_index]:
                if face_recog_flag[face_index]:
                    p1[face_index] = np.array([[eye_center[face_index]]], np.float32)
                else:
                    # オプティカルフローのパラメータ設定
                    lk_params = dict(winSize=(int(rec_face[face_index][1] - rec_face[face_index][0]),
                                              int(rec_face[face_index][1] - rec_face[face_index][0])), maxLevel=2,
                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    p1[face_index], st, err = cv2.calcOpticalFlowPyrLK(frame_gray_old, frame_gray, p0[face_index], None,
                                                                       **lk_params)

                x[face_index] = get_coords(p0[face_index])[0]
                y[face_index] = get_coords(p0[face_index])[1]

                a, b = get_coords(p0[face_index]), get_coords(p1[face_index])
                absolute_x[face_index] = abs(a[0] - b[0])
                absolute_y[face_index] = abs(a[1] - b[1])
                relative_x[face_index] = a[0] - b[0]
                relative_y[face_index] = a[1] - b[1]
                decay_number = int((rec_face[face_index][1] - rec_face[face_index][0]) * 0.05)
                x_movement[face_index] = max(0, int((x_movement[face_index] + abs(a[0] - b[0])) - decay_number))
                y_movement[face_index] = max(0, int((y_movement[face_index] + abs(a[1] - b[1])) - decay_number))

                gesture_threshold = int((rec_face[face_index][1] - rec_face[face_index][0]) * 0.2)

                if y_movement[face_index] > gesture_threshold:
                    gesture[face_index] = 1
                    y_movement[face_index] = 0
                    gesture_show[face_index] = gesture_show_frame  # number of frames a gesture is shown

                if gesture[face_index] and gesture_show[face_index] > 0:
                    cv2.circle(frame, get_coords(p1[face_index]), 100, (0, 0, 255), 5)  # now nodding
                    gesture_show[face_index] -= 1

                if gesture_show[face_index] <= 0:
                    gesture[face_index] = 0

                # 計算した座標に点を表示
                cv2.circle(frame, get_coords(p1[face_index]), 3, (255, 0, 255), -1)
                cv2.putText(frame, "{}".format(gesture_threshold), (face_index * 100, 30), cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 0), 2)
                cv2.putText(frame, "{}".format(y_movement[face_index]), (face_index * 100, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)

                p0[face_index] = p1[face_index]
            else:
                pass
        frame_gray_old = frame_gray

        # データをリストに追加
        for face_index in range(len(known_faces)):
            csv_saving_list[face_index].append([
                frame_number, int(1000 * frame_number / video_frame_rate),
                emotion[face_index],
                x[face_index], y[face_index],
                absolute_x[face_index], relative_x[face_index], x_movement[face_index],
                absolute_y[face_index], relative_y[face_index], y_movement[face_index],
                mouse_opening_rate_list[face_index],
                gesture[face_index],
                rec_face[face_index][0], rec_face[face_index][1], rec_face[face_index][2], rec_face[face_index][3]
            ])

        face_recog_flag = [False] * len(target_image_path_list)
        output_movie.write(frame)
        first_flag = False

    # 動画保存
    input_movie.release()
    cv2.destroyAllWindows()

    # 結果をCSVファイルに保存
    for i in range(len(target_image_path_list)):
        with open(join(path_result, 'result{}.csv'.format(i)), "w", encoding="Shift_jis") as f:
            writer = csv.writer(f, lineterminator="\n")  # writerオブジェクトの作成 改行記号で行を区切る
            writer.writerows(csv_saving_list[i])  # csvファイルに書き込み


if __name__ == '__main__':
    start = time.time()
    # time.sleep(3600*10)

    # 感情検知モデルの読み込み
    path_model = "../model/"
    model_name = "mini_XCEPTION"
    model = model_from_json(open("{}model_{}.json".format(path_model, model_name), "r").read())
    model.load_weights('{}model_{}.h5'.format(path_model, model_name))
    emotions = ('Negative', 'Negative', 'Normal', 'Positive', 'Normal', 'Normal',
                'Normal')  # emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')
    emotions_color_bgr_dict = {'Negative': (255, 0, 0), 'Normal': (0, 255, 0), 'Positive': (0, 0, 255)}

    # ハイパーパラメータの設定
    frame_use_rate = 3
    k_resolution = 3
    tolerance = 0.35  # 顔認識を行う際の距離の閾値
    path_face = "face_image_auto/"
    path_video = "video/"
    video_name = ["Take01", "Take02", "190130"][0]
    target_image_path_list = []
    for i in os.listdir(path_face + video_name + "/cluster/"):
        target_image_path_list.append(glob.glob(path_face + video_name + "/cluster/{}/*".format(i)))
    target_video_path = "{}{}".format(path_video, video_name) + ".mp4"
    result_path = "result/{}_{}_{}_{}_{}_{}people/".format(datetime.date.today(), model_name, video_name, k_resolution,
                                                           frame_use_rate, len(target_image_path_list))
    # 人の顔画像を読み込む
    known_faces = []
    for file_path in target_image_path_list:
        face_image = [face_recognition.load_image_file(x) for x in file_path]
        face_image_encoded = [face_recognition.face_encodings(x)[0] for x in face_image]
        known_faces.append(face_image_encoded)

    # 動画を複数に分割して並列処理
    split_video_num = 10
    cleanup_directory(result_path)
    process_list = []
    for split_video_index in range(split_video_num):
        split_result_path = "result/{}_{}_{}_{}_{}_{}people/split_video_{}/".format(datetime.date.today(), model_name,
                                                                                    video_name, k_resolution,
                                                                                    frame_use_rate,
                                                                                    len(target_image_path_list),
                                                                                    split_video_index)
        cleanup_directory(split_result_path)
        process_list.append(multiprocessing.Process(target=emotion_recognition, args=(
            target_video_path, split_result_path, k_resolution, frame_use_rate, tolerance, target_image_path_list,
            model,
            model_name, emotions, split_video_index, split_video_num, known_faces, emotions_color_bgr_dict)))
        process_list[split_video_index].start()
    for process in process_list:
        process.join()
    # 分割して処理した動画を結合
    input_video_list = [
        "result/{}_{}_{}_{}_{}_{}people/split_video_{}/output.avi".format(datetime.date.today(), model_name, video_name,
                                                                          k_resolution, frame_use_rate,
                                                                          len(target_image_path_list), i) for i in
        range(split_video_num)]
    concat_video(input_video_list, result_path + "output.avi")
    for j in range(len(target_image_path_list)):
        input_csv_list = [
            "result/{}_{}_{}_{}_{}_{}people/split_video_{}/result{}.csv".format(datetime.date.today(), model_name,
                                                                                video_name, k_resolution,
                                                                                frame_use_rate,
                                                                                len(target_image_path_list), i, j) for i
            in range(split_video_num)]
        concat_csv(input_csv_list, result_path + "result{}.csv".format(j))

    print("elapsed_time:{0}".format(time.time() - start) + "[sec]")
