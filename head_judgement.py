import pickle
import csv
import numpy as np
from os.path import join

# int型の座標を取得
def get_coords(p1):
    # print(p1)
    try:
        return int(p1[0][0][0]), int(p1[0][0][1])
    except IndexError:
        return int(p1[0][0]), int(p1[0][1])

def head_judgement(file_path,path_result):
    ### read the .pt file
    # pkl_file = open(file_path, 'rb')
    # data = pickle.load(pkl_file)

    ### read the .csv file
    csvFile = open(file_path, "r")
    result={}
    frist_line=""
    reader = csv.reader(csvFile)
    page_num=0
    for item in reader:
        if reader.line_num == 1:
            frist_line=item
            continue
        result[page_num] = item
        page_num=page_num+1
    # print(frist_line)
    initial_num=0
    for tag in frist_line:
        if tag.find("face")==0:
            initial_num=initial_num+1
    # print("person_number",initial_num)
    data=result

    person_num=initial_num
    ########

    # person_num=len(data[1])-1##人数


    # CSVファイルの最初の列
    csv_saving_list = [[["frame_number", "time(ms)", "prediction", "x", "y", "absolute_x", "relative_x", "x_movement",
                         "absolute_y", "relative_y", "y_movement", "mouse opening", "gesture", "top", "bottom", "left",
                         "right"]] for _ in range(person_num)]


    ##各変数を初期化
    # 前フレームの顔の座標を記録する変数
    rec_face = [[0, 0, 0, 0]] * person_num
    # 顔の中心を示す変数
    # 顔の中央のオプティカルフロー計算に用いる変数
    p0 = [np.array([0, 0])] * person_num
    p1 = [np.array([0, 0])] * person_num
    # 首を動かしたかどうかを判断する変数
    gesture = [0] * person_num
    # オプティカルフローの移動量
    x_movement = [0] * person_num
    y_movement = [0] * person_num
    # 座標
    x = [0] * person_num
    y = [0] * person_num
    # 動作判定の表示時間
    gesture_show_frame = 30
    gesture_show = [gesture_show_frame] * person_num
    # 顔が認識できたかどうか
    face_recog_flag = [False] * person_num
    # 初めて顔を認識したかどうか
    first_face_recog_flag = [False] * person_num
    # オプティカルフローの前フレームからの移動量の絶対値
    absolute_x = [0] * person_num
    absolute_y = [0] * person_num
    # オプティカルフローの前フレームからの移動量の相対値
    relative_x = [0] * person_num
    relative_y = [0] * person_num
    # 目の中心位置座標
    eye_center = [[0, 0]] * person_num
    # 感情
    emotion = ["Unknown"] * person_num
    mouse_opening_rate_list = [0] * person_num

    first_flag = True
    start_frame=0
    last_frame=len(data)-1

    for frame_number in range(start_frame,last_frame):
        for face_index in range(person_num):
            tmp_data=data[frame_number][face_index+2] ## 第三列から顔の座標
            tmp_data=tmp_data.replace(' ', '').split(",")
            if len(tmp_data)<2:
                break
            tmp_data[0]=int(tmp_data[0].replace("(","")) ##各人の顔の座標
            tmp_data[1]=int(tmp_data[1])
            tmp_data[2]=int(tmp_data[2])
            tmp_data[3]=int(tmp_data[3].replace(")",""))

            tmp_data[0],tmp_data[1],tmp_data[2],tmp_data[3]= tmp_data[1],tmp_data[0],tmp_data[3],tmp_data[2]
            # print("tmp_data",tmp_data)
            tmp_eye_inf=data[frame_number][face_index+2+person_num] ## 第(2+n)列から目の座標, nは人数
            tmp_eye_inf=tmp_eye_inf.replace(' ', '').split(",")
            tmp_eye_inf[0]=int(tmp_eye_inf[0].replace("(",""))
            tmp_eye_inf[1]=int(tmp_eye_inf[1].replace(")",""))

            if len(data[frame_number][face_index+2].replace(" ",""))>0: ##顔の座標が存在する
                face_recog_flag[face_index]=True
            rec_face[face_index]=[tmp_data[1],tmp_data[3],tmp_data[2],tmp_data[0]]#[top, bottom, left, right]

            if not first_face_recog_flag[face_index]:  ##最初に認識できた
                p0[face_index] = np.array([[tmp_eye_inf]], np.float32)
                first_face_recog_flag[face_index] = True

        for face_index in range(person_num):
            if first_flag:  # 前フレームがない場合は終了
                break
            elif first_face_recog_flag[face_index]:  ##認識できた場合
                if face_recog_flag[face_index]: ##表情が認識できる ##np.array([[tmp_eye_inf]], np.float32)
                    p1[face_index] = np.array([[tmp_eye_inf]], np.float32)
                else:
                    break
                    # # オプティカルフローのパラメータ設定
                    # lk_params = dict(winSize=(int(rec_face[face_index][1] - rec_face[face_index][0]),
                    #                           int(rec_face[face_index][1] - rec_face[face_index][0])), maxLevel=2,
                    #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    # p1[face_index], st, err = cv2.calcOpticalFlowPyrLK(frame_gray_old, frame_gray, p0[face_index], None,
                    #                                                    **lk_params)
                x[face_index] = get_coords(p0[face_index])[0]
                y[face_index] = get_coords(p0[face_index])[1]


                a, b = get_coords(p0[face_index]), get_coords(p1[face_index]) ##a前の時目の座標 b現在の座標
                absolute_x[face_index] = abs(a[0] - b[0]) ##移動距離
                absolute_y[face_index] = abs(a[1] - b[1])
                relative_x[face_index] = a[0] - b[0]
                relative_y[face_index] = a[1] - b[1]

                decay_number = int((rec_face[face_index][1] - rec_face[face_index][0]) * 0.05)
                x_movement[face_index] = max(0, int((x_movement[face_index] + abs(a[0] - b[0])) - decay_number))
                y_movement[face_index] = max(0, int((y_movement[face_index] + abs(a[1] - b[1])) - decay_number))

                gesture_threshold = int((rec_face[face_index][1] - rec_face[face_index][0]) * 0.2) ##判断の閾値
                if y_movement[face_index] > gesture_threshold:
                    gesture[face_index] = 1
                    y_movement[face_index] = 0
                    gesture_show[face_index] = gesture_show_frame  # number of frames a gesture is shown

                if gesture[face_index] and gesture_show[face_index] > 0:
                    gesture_show[face_index] -= 1

                if gesture_show[face_index] <= 0:
                    gesture[face_index] = 0

                p0[face_index] = p1[face_index] ##目の座標を更新する
            else:
                pass
        ##データをリストに追加
        for face_index in range(person_num):
            csv_saving_list[face_index].append([
                frame_number, int(1000 * frame_number / 30),  ##  1s 30frame
                emotion[face_index],
                x[face_index], y[face_index],
                absolute_x[face_index], relative_x[face_index], x_movement[face_index],
                absolute_y[face_index], relative_y[face_index], y_movement[face_index],
                mouse_opening_rate_list[face_index],
                gesture[face_index],
                rec_face[face_index][0], rec_face[face_index][1], rec_face[face_index][2], rec_face[face_index][3]
            ])

        face_recog_flag = [False] * person_num
        first_flag = False

    # 結果をCSVファイルに保存
    for i in range(person_num):
        with open(join(path_result, 'result{}.csv'.format(i)), "w", encoding="Shift_jis") as f:
            writer = csv.writer(f, lineterminator="\n")  # writerオブジェクトの作成 改行記号で行を区切る
            writer.writerows(csv_saving_list[i])  # csvファイルに書き込み

if __name__ == '__main__':
    x=head_judgement("file/detect_face.csv","file/result/")
