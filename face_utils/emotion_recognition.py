import csv
import os

from scipy.spatial import distance as dist
from tqdm import tqdm

from utils.video_utils import *
from constants import constants

config = {
    "debug": constants.DEBUG,
    "num_gpu": constants.NUM_GPU
}


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


# int型の座標を取得
def get_coords(p1):
    try:
        return int(p1[0][0][0]), int(p1[0][0][1])
    except IndexError:
        return int(p1[0][0]), int(p1[0][1])


def get_parts_coordinates(a, b):
    coord = list()
    if b:
        for w in a:
            coord.append(w[0])
    else:
        for w in a:
            coord.append(w[1])
    return coord


def get_eye_location(face_landmarks):
    # 目の位置
    leye_x = get_parts_coordinates(face_landmarks["left_eye"], True)
    reye_x = get_parts_coordinates(face_landmarks["right_eye"], True)
    eye_y = get_parts_coordinates(face_landmarks["left_eye"] + face_landmarks["right_eye"], False)
    topeye = min(eye_y)
    bottomeye = max(eye_y)
    lefteye = min(leye_x)
    righteye = max(reye_x)
    eye_center = (righteye + lefteye) // 2, (topeye + bottomeye) // 2
    return eye_center


def emotion_recognition(interpolated_result: dict, video_path: str, output_dir: str, offset=0, parallel_num=1, rank=0,
                        k_frame=3,
                        emotion_label=('Negative', 'Negative', 'Normal', 'Positive', 'Normal', 'Normal', 'Normal')):
    """
    
    :param interpolated_result: Final result from capture_face_ng.py
    :param video_path: Input video path
    :param output_dir: Directory of output CSV
    :param offset:
    :param parallel_num: Previously split_video_num, parameter for multiprocessing
    :param rank: Previously split_video_index, parameter for multiprocessing
    :param k_frame: How often we process 1 frame (default once every 3 frame)
    :param emotion_label: Default ('Negative', 'Negative', 'Normal', 'Positive', 'Normal', 'Normal', 'Normal')
    :return: 
    """
    gpu_count = 3
    if parallel_num <= gpu_count:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        # Disable CUDA otherwise, GPU actually aren't that helpful since we don't do batch processing here
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import face_recognition
    from keras.models import model_from_json

    face_num = len(interpolated_result)  # Previously person_number, number of person

    model_dir = "model"
    if not os.path.isdir(model_dir):
        model_dir = "../model"
    model_name = "mini_XCEPTION"
    with open(os.path.join(model_dir, f"model_{model_name}.json"), 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(os.path.join(model_dir, f"model_{model_name}.h5"))

    video_capture = get_video_capture(video_path)
    video_framerate = get_video_framerate(video_path)
    video_length = get_video_length(video_path)
    divided = video_length // parallel_num
    frame_range = [[i * divided, (i + 1) * divided] for i in range(parallel_num)]
    frame_range[-1][1] = video_length - 1

    start, end = frame_range[rank]
    bar = tqdm(total=end - start, desc="Process#{}".format(rank))

    # CSVファイルの最初の列
    csv_saving_list = [[["frame_number", "time(ms)", "prediction", "x", "y", "absolute_x", "relative_x", "x_movement",
                         "absolute_y", "relative_y", "y_movement", "mouse opening", "gesture", "top", "bottom", "left",
                         "right", "gesture_threshold"]] for _ in range(face_num)]
    # 各変数を初期化
    # 前フレームの顔の座標を記録する変数
    rec_face = [[0, 0, 0, 0]] * face_num
    # 顔の中心を示す変数
    # 顔の中央のオプティカルフロー計算に用いる変数
    p0 = [np.array([0, 0])] * face_num
    p1 = [np.array([0, 0])] * face_num
    # 首を動かしたかどうかを判断する変数
    gesture = [0] * face_num
    # オプティカルフローの移動量
    x_movement = [0] * face_num
    y_movement = [0] * face_num
    # 座標
    x = [0] * face_num
    y = [0] * face_num
    # 動作判定の表示時間
    gesture_show_frame = 15
    gesture_show = [gesture_show_frame] * face_num
    # 顔が認識できたかどうか
    face_recog_flag = [False] * face_num
    # 初めて顔を認識したかどうか
    first_face_recog_flag = [False] * face_num
    # オプティカルフローの前フレームからの移動量の絶対値
    absolute_x = [0] * face_num
    absolute_y = [0] * face_num
    # オプティカルフローの前フレームからの移動量の相対値
    relative_x = [0] * face_num
    relative_y = [0] * face_num
    # 目の中心位置座標
    eye_center = [[0, 0]] * face_num
    # 感情
    emotion = ["Unknown"] * face_num
    mouse_opening_rate_list = [0] * face_num
    ##
    top_for_lastframe = [0] * face_num

    ###
    first_flag = True
    gesture_threshold = 0

    frame_index = start
    set_frame_position(video_capture, start)
    while video_capture.isOpened() and get_frame_position(video_capture) in range(start, end):
        ret, frame = read_video_capture(video_capture, offset)
        # Progress
        bar.update(1)
        bar.refresh()

        if frame_index % k_frame != 0:
            frame_index += 1
            continue

        for face_index in range(face_num):
            if interpolated_result[face_index][frame_index].location is None:
                continue
            else:
                top, right, bottom, left = interpolated_result[face_index][frame_index].location
                rec_face[face_index] = [top, bottom, left, right]
                # 顔領域を切り取る
                dst = frame[top:bottom, left:right]
                # 表情認識
                val = cv2.resize(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), (64, 64)) / 255.0
                val = val.reshape(1, 64, 64, 1)

                predictions = model.predict(val, batch_size=1)  # 学習モデルから表情を特定
                emotion[face_index] = emotion_label[int(np.argmax(predictions[0]))]
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

        # Head movement judgement
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for face_index in range(face_num):
            if first_flag:  # 前フレームがない場合は終了
                break
            elif first_face_recog_flag[face_index]:
                p1[face_index] = np.array([[eye_center[face_index]]], np.float32)
                if face_recog_flag[face_index]:
                    p1[face_index] = np.array([[eye_center[face_index]]], np.float32)
                else:  # copy the codes from previous method  (unclear)
                    # オプティカルフローのパラメータ設定
                    lk_params = dict(winSize=(int(rec_face[face_index][1] - rec_face[face_index][0]),
                                              int(rec_face[face_index][1] - rec_face[face_index][0])),
                                     maxLevel=2,
                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    p1[face_index], st, err = cv2.calcOpticalFlowPyrLK(frame_gray_old, frame_gray,
                                                                       p0[face_index],
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
                if y_movement[face_index] != 0:
                    y_movement[face_index] = y_movement[face_index] + decay_number * 0.5

                gesture_threshold = int((rec_face[face_index][1] - rec_face[face_index][0]) * 0.2)

                if y_movement[face_index] > gesture_threshold:
                    gesture[face_index] = 1
                    y_movement[face_index] = 0
                    gesture_show[face_index] = gesture_show_frame  # number of frames a gesture is shown

                if gesture[face_index] and gesture_show[face_index] > 0:
                    gesture_show[face_index] -= 1 * k_frame

                if gesture_show[face_index] <= 0:
                    gesture[face_index] = 0

                p0[face_index] = p1[face_index]
                top_for_lastframe[face_index] = rec_face[face_index][0]
            else:
                pass
        frame_gray_old = frame_gray

        # データをリストに追加
        for face_index in range(face_num):
            csv_saving_list[face_index].append([
                frame_index, int(1000 * frame_index / video_framerate),
                emotion[face_index],
                x[face_index], y[face_index],
                absolute_x[face_index], relative_x[face_index], x_movement[face_index],
                absolute_y[face_index], relative_y[face_index], y_movement[face_index],
                mouse_opening_rate_list[face_index],
                gesture[face_index],
                rec_face[face_index][0], rec_face[face_index][1], rec_face[face_index][2], rec_face[face_index][3],
                gesture_threshold
            ])

        face_recog_flag = [False] * face_num
        first_flag = False
        frame_index += 1

    # 結果をCSVファイルに保存
    for face_index in range(face_num):
        filename = f"result{face_index}.csv" if parallel_num == 1 else f"result{face_index}_{rank}.csv"
        with open(os.path.join(output_dir, filename), "w", encoding="Shift_jis") as f:
            writer = csv.writer(f, lineterminator="\n")  # writerオブジェクトの作成 改行記号で行を区切る
            writer.writerows(csv_saving_list[face_index])  # csvファイルに書き込み


def emotion_recognition_multiprocess(interpolated_result: dict, video_path: str, output_dir: str, offset=0,
                                     parallel_num=config["num_gpu"],
                                     k_frame=3,
                                     emotion_label=(
                                             'Negative', 'Negative', 'Normal', 'Positive', 'Normal', 'Normal',
                                             'Normal')):
    print("\nEmotion Recognition")
    print("Using", parallel_num, "Process(es)")
    process_list = []
    for i in range(parallel_num):
        kwargs = dict(interpolated_result=interpolated_result,
                      video_path=video_path,
                      output_dir=output_dir,
                      offset=offset,
                      parallel_num=parallel_num,
                      rank=i,
                      k_frame=k_frame,
                      emotion_label=emotion_label
                      )
        p = Process(target=emotion_recognition, kwargs=kwargs)
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    if parallel_num == 1:
        return

    # Concat CSV
    face_num = len(interpolated_result)
    for face_index in range(face_num):
        concat_csv = [["frame_number", "time(ms)", "prediction", "x", "y", "absolute_x", "relative_x", "x_movement",
                       "absolute_y", "relative_y", "y_movement", "mouse opening", "gesture", "top", "bottom",
                       "left", "right", "gesture_threshold"]]

        for rank in range(parallel_num):
            filename = f"result{face_index}_{rank}.csv"
            with open(os.path.join(output_dir, filename), "r", encoding="Shift_jis") as f:
                reader = csv.reader(f, lineterminator="\n")
                concat_csv.extend([row for row in reader][1:])  # Ignore headers
            os.remove(os.path.join(output_dir, filename))

        with open(os.path.join(output_dir, f"result{face_index}.csv"), "w", encoding="Shift_jis") as f:
            writer = csv.writer(f, lineterminator="\n")  # writerオブジェクトの作成 改行記号で行を区切る
            writer.writerows(concat_csv)  # csvファイルに書き込み
