import pandas as pd
import cv2
import csv
import numpy as np
import pickle
import os
from os.path import join
from keras.models import model_from_json
from ast import literal_eval as make_tuple
import face_recognition
from scipy.spatial import distance as dist

os.environ["CUDA_VISIBLE_DEVICES"]="0"
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


# int型の座標を取得
def get_coords(p1):
    try:
        return int(p1[0][0][0]), int(p1[0][0][1])
    except IndexError:
        return int(p1[0][0]), int(p1[0][1])

def get_parts_coordinates(a, b):
    l = list()
    if b == True:
        for w in a:
            l.append(w[0])
    elif b == False:
        for w in a:
            l.append(w[1])
    return l

def get_video_dimension(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

def get_face_location_from_csv(df, frame_num):
    face_location = df.loc[df["frame"] == frame_num].values.tolist()
    if len(face_location) == 0:
        # frame_num not found in csv
        return None
    return [make_tuple(s) for s in face_location[0][2:]]

def get_eye_location(face_landmarks):
    # 目の位置
    leye_x = get_parts_coordinates(face_landmarks["left_eye"], True)
    reye_x = get_parts_coordinates(face_landmarks["right_eye"], True)
    eye_y = get_parts_coordinates(face_landmarks["left_eye"] + face_landmarks["right_eye"], False)
    topeye = min(eye_y)
    bottomeye = max(eye_y)
    lefteye = min(leye_x)
    righteye = max(reye_x)
    eye_center=  (righteye + lefteye) // 2, (topeye + bottomeye) // 2
    return eye_center

def emotion_recognition_new(target_video_path,k_prame,path_result,emotions,split_video_index,split_video_num,file_path,file_name):
    with open(file_path+file_name, "rb") as f:
        rst = pickle.load(f)
    person_number = len(rst)

    k=k_prame ## how often we read one frame (e.g. 3 frames)
    model_dir = 'model'
    model_name = 'mini_XCEPTION'
    model = model_from_json(open(join(model_dir, 'model_{}.json'.format(model_name)), 'r').read())
    model.load_weights(join(model_dir, 'model_{}.h5'.format(model_name)))##

    input_movie = cv2.VideoCapture(target_video_path)  # 動画を読み込む
    video_frame_rate = input_movie.get(cv2.CAP_PROP_FPS)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画の長さを測る

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画コーデック指定
    original_w = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))  # 動画の幅を測る
    original_h = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 動画の高さを測る
    # resize_rate = (1080 * k_resolution) / original_w  # 動画の横幅を3Kに固定
    # w = int(original_w * resize_rate)
    # h = int(original_h * resize_rate)
    output_movie = cv2.VideoWriter(join(path_result, 'output.avi'), fourcc, video_frame_rate, (original_w, original_h))  # 出力する動画の詳細を設定する

    # CSVファイルの最初の列
    csv_saving_list = [[["frame_number", "time(ms)", "prediction", "x", "y", "absolute_x", "relative_x", "x_movement",
                         "absolute_y", "relative_y", "y_movement", "mouse opening", "gesture", "top", "bottom", "left",
                         "right","gesture_threshold"]] for _ in range(person_number)]
    # 各変数を初期化
    # 前フレームの顔の座標を記録する変数
    rec_face = [[0, 0, 0, 0]] * person_number
    # 顔の中心を示す変数
    # 顔の中央のオプティカルフロー計算に用いる変数
    p0 = [np.array([0, 0])] * person_number
    p1 = [np.array([0, 0])] * person_number
    # 首を動かしたかどうかを判断する変数
    gesture = [0] * person_number
    # オプティカルフローの移動量
    x_movement = [0] * person_number
    y_movement = [0] * person_number
    # 座標
    x = [0] * person_number
    y = [0] * person_number
    # 動作判定の表示時間
    gesture_show_frame = 15
    gesture_show = [gesture_show_frame] * person_number
    # 顔が認識できたかどうか
    face_recog_flag = [False] * person_number
    # 初めて顔を認識したかどうか
    first_face_recog_flag = [False] * person_number
    # オプティカルフローの前フレームからの移動量の絶対値
    absolute_x = [0] * person_number
    absolute_y = [0] * person_number
    # オプティカルフローの前フレームからの移動量の相対値
    relative_x = [0] * person_number
    relative_y = [0] * person_number
    # 目の中心位置座標
    eye_center = [[0, 0]] * person_number
    # 感情
    emotion = ["Unknown"] * person_number
    mouse_opening_rate_list = [0] * person_number
    ##
    top_for_lastframe=[0] * person_number

    frame_duration = length // split_video_num
    start_frame = split_video_index * frame_duration
    stop_frame = length if split_video_index == split_video_num - 1 else start_frame + frame_duration
    input_movie.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 動画の開始フレームを設定
    first_flag = True
    file_strat_frame=0 # the first frame we use
    file_frame_count=0
    # print("0")
    for i in range (0,len(rst[0])):
        if rst[0][i].frame_number>=start_frame:
            file_strat_frame=rst[0][i].frame_number  ## the first frame we use
            file_frame_count=i  ##record the index
            break
    write_flg=False
    gesture_threshold=0 ##tmp
    for frame_number in range(start_frame, stop_frame):
        # 動画を読み込む
        ret, frame = input_movie.read()
        # print("initial frame shape",frame.shape)
        print("\nWriting frame {} / {}".format(frame_number, length - 1))
        # 動画が読み取れない場合は終了
        if not ret:
            break

        # フレームのサイズを調整する
        # frame = cv2.resize(frame, (w, h)) ## no need to modify the size in this new method

        print("frame_num",frame_number,file_strat_frame)

        if frame_number!= file_strat_frame:
            if write_flg:
                for face_index in range(person_number):

                    # cv2.circle(frame, get_coords(p0[face_index]), 3, (255, 0, 255), -1)
                    # cv2.putText(frame, "{}".format(gesture_threshold), (face_index * 100, 30), cv2.FONT_HERSHEY_COMPLEX,
                    #             1.0, (0, 0, 0), 2)
                    # cv2.putText(frame, "{}".format(y_movement[face_index]), (face_index * 100, 60),
                    #         cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)
                    if emotion[face_index] != "Unknown":
                        box_label(frame, int(rec_face[face_index][2]), int(rec_face[face_index][0]),
                                  int(rec_face[face_index][3]), int(rec_face[face_index][1]),
                                  "{}:{}".format(face_index, emotion[face_index]), color=(0, 255, 0))
                        print(frame_number,"draw",face_index)
            output_movie.write(frame)
            continue  ## make sure we use the same frame with the .pt file

        ## emotion recognization and head movement judgement
        else:
            write_flg = True
            for face_index in range(person_number):
                if rst[face_index][file_frame_count].location==None:
                    continue
                else:
                    top, right, bottom, left=rst[face_index][file_frame_count].location
                    top,bottom = top,bottom

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
                    # if (rst[face_index][file_frame_count].is_detected):
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

            ## head movement judgement
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for face_index in range(person_number):
                if first_flag:  # 前フレームがない場合は終了
                    break
                elif first_face_recog_flag[face_index]:
                    p1[face_index] = np.array([[eye_center[face_index]]], np.float32)
                    if face_recog_flag[face_index]: ##
                        p1[face_index] = np.array([[eye_center[face_index]]], np.float32)
                    else: ## copy the codes from perious method  (unclear)
                        # オプティカルフローのパラメータ設定
                        lk_params = dict(winSize=(int(rec_face[face_index][1] - rec_face[face_index][0]),
                                                  int(rec_face[face_index][1] - rec_face[face_index][0])), maxLevel=2,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                        p1[face_index], st, err = cv2.calcOpticalFlowPyrLK(frame_gray_old, frame_gray, p0[face_index],
                                                                           None, **lk_params)
                    x[face_index] = get_coords(p0[face_index])[0]
                    y[face_index] = get_coords(p0[face_index])[1]

                    a, b = get_coords(p0[face_index]), get_coords(p1[face_index])
                    absolute_x[face_index] = abs(a[0] - b[0])
                    absolute_y[face_index] = abs(a[1] - b[1])
                    relative_x[face_index] = a[0] - b[0]
                    relative_y[face_index] = a[1] - b[1]
                    decay_number = int((rec_face[face_index][1] - rec_face[face_index][0]) * 0.08)
                    x_movement[face_index] = max(0, int((x_movement[face_index] + abs(a[0] - b[0])) - decay_number))
                    y_movement[face_index] = max(0, int((y_movement[face_index] + abs(a[1] - b[1])) - decay_number))
                    if y_movement[face_index]!=0:
                        y_movement[face_index]=y_movement[face_index]+decay_number*0.5


                    gesture_threshold = int((rec_face[face_index][1] - rec_face[face_index][0]) * 0.2)

                    if y_movement[face_index] > gesture_threshold :
                        gesture[face_index] = 1
                        y_movement[face_index] = 0
                        gesture_show[face_index] = gesture_show_frame  # number of frames a gesture is shown

                    if gesture[face_index] and gesture_show[face_index] > 0:
                        cv2.circle(frame, get_coords(p1[face_index]), 100, (0, 0, 255), 5)  # now nodding
                        gesture_show[face_index] -= 1 * k

                    if gesture_show[face_index] <= 0:
                        gesture[face_index] = 0

                    # 計算した座標に点を表示
                    cv2.circle(frame, get_coords(p1[face_index]), 3, (255, 0, 255), -1)
                    cv2.putText(frame, "{}".format(gesture_threshold), (face_index * 100, 30), cv2.FONT_HERSHEY_COMPLEX,
                                1.0, (0, 0, 0), 25)
                    cv2.putText(frame, "{}".format(y_movement[face_index]), (face_index * 100, 60),
                                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 25)

                    p0[face_index] = p1[face_index]
                    top_for_lastframe[face_index]=rec_face[face_index][0]
                else:
                    pass
            frame_gray_old = frame_gray

            # データをリストに追加
            for face_index in range(person_number):
                csv_saving_list[face_index].append([
                    frame_number, int(1000 * frame_number / video_frame_rate),
                    emotion[face_index],
                    x[face_index], y[face_index],
                    absolute_x[face_index], relative_x[face_index], x_movement[face_index],
                    absolute_y[face_index], relative_y[face_index], y_movement[face_index],
                    mouse_opening_rate_list[face_index],
                    gesture[face_index],
                    rec_face[face_index][0], rec_face[face_index][1], rec_face[face_index][2], rec_face[face_index][3],
                    gesture_threshold
                ])

            face_recog_flag = [False] * person_number
            output_movie.write(frame)
            first_flag = False
        if(file_frame_count+k_prame)<=len(rst[0]):
            file_strat_frame=rst[0][file_frame_count+k_prame].frame_number
            file_frame_count=file_frame_count+k_prame
        else:
            break

    # 動画保存
    input_movie.release()
    cv2.destroyAllWindows()

    # 結果をCSVファイルに保存
    for i in range(person_number):
        with open(join(path_result, 'result{}.csv'.format(i)), "w", encoding="Shift_jis") as f:
            writer = csv.writer(f, lineterminator="\n")  # writerオブジェクトの作成 改行記号で行を区切る
            writer.writerows(csv_saving_list[i])  # csvファイルに書き込み

# a=emotion_recognition("file/out1.mp4",3,128,"file/detect_face.csv")

if __name__ == '__main__':
    # with open("utils/detect_face2.pt", "rb") as f:
    #     rst = pickle.load(f)
    #     # print(rst[0]["face_locations"])##
    #     # print(rst[0][60])  ##
    #     # print(rst[71])  ##
    #
    #     print(rst[0])
    # #     print(len(rst[0]))

    #
    target_video_path="test_video/out1.mp4"
    path_result="output_0309-4/"
    k_resolution=3
    emotions = ('Negative', 'Negative', 'Normal', 'Positive', 'Normal', 'Normal', 'Normal')
    split_video_index=0
    split_video_num=1
    file_path="utils/"
    file_name="60s-1.pt"
    emotion_recognition_new(target_video_path,3,path_result,emotions,split_video_index,split_video_num,file_path,file_name)
