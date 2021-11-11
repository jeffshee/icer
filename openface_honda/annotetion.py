import cv2
import numpy as np
import csv
import itertools

# csv_path = "./pos/anno_2.csv" #アノテーションリスト
csv_name = input("csvファイル名（拡張子なし）:")
csv_path = f"./pos/{csv_name}.csv"
video_path = "./test.mp4" #動画ファイルのパス

first_flag = 1 #最初のステップ（全員をクリックするステップ）中のフラグ
id_positions = [] #idごとの座標
nod_flags = [] #idごとの頷きフラグ
num_people = 0 #人の数

#位置格納関数
def mouse_event(event, x, y, flags, param):
    #グローバル変数を利用
    global first_flag, num_people
    #マウス左ボタン押下時
    if event == cv2.EVENT_LBUTTONDOWN:

        #最初のステップ（全員をクリックするステップ）
        if first_flag:

            if num_people >= 9:
                cv2.putText(paint, "Only 9 people can be selected", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), thickness=2)

            else:
                nod_flags.append(0)
                id_positions.extend([x, y]) 
                num_people += 1

#保存関数
def save_anno(frame):

    #グローバル変数を利用
    global nod_flags, id_positions
    
    # csvに保存
    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        record = [frame]
        record.extend(nod_flags)
        record.extend(id_positions)
        writer.writerow(record)

cap = cv2.VideoCapture(video_path) #動画の読み込み
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #高さ

cv2.namedWindow("window", cv2.WINDOW_NORMAL) #windowの生成
cv2.setMouseCallback("window", mouse_event) #描画関数を設定
frame = 0 #フレーム番号

#プログラム全体終了フラグ
fin_flag = True

#ループ処理
while fin_flag:
    #フレーム取り出し
    gg, img = cap.read()

    #フレームが取り出せた場合
    if gg:

        paint = img.copy() #データのコピー

        #フレーム番号の描画
        frame += 1 #フレーム番号の加算

        #ループ処理
        while True:

            if not first_flag:
                paint = img.copy() #データのコピー

            cv2.putText(paint, str(frame), (0, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2) #フレーム番号の描画

            # 一人ずつ頷きアノテーションの判定
            for id, nod_flag in enumerate(nod_flags, 1):

                # 頷いてる場合
                if nod_flag:
                    cv2.putText(paint, f"{id}:NOD", (100*id, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

                # 頷いていない場合
                else:
                    cv2.putText(paint, f"{id}:X", (100*id, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

            #windowに描画する
            cv2.imshow("window", paint)

            #キー入力を待つ
            key = cv2.waitKey(10)

            #数字キーが押されたら、対応するidの頷きフラグを反転
            if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]:

                for id in range(num_people):
                    if key == ord(str(id+1)):
                        nod_flags[id] = (nod_flags[id] + 1) % 2 # flagを反転
                        break

            #sキーが押されたら,アノテーションの保存,次のフレーム
            if key == ord('s'):

                #最初のステップ（全員をクリックするステップ）を終了させる
                if first_flag:

                    # csvにコラム書き込み
                    with open(csv_path, 'a') as f:
                        writer = csv.writer(f)

                        # [frame, nod_1, nod_2, .., x_1, y_1, x_2, y_2, ...]をコラムに
                        nod_col = [f"nod_{i}" for i in range(num_people)]
                        pos_col = [[f"x_{i}", f"y_{i}"] for i in range(num_people)]
                        pos_col = itertools.chain.from_iterable(pos_col)
                        col = ["frame"]
                        col.extend(nod_col)
                        col.extend(pos_col)

                        writer.writerow(col)

                    first_flag = 0

                # アノテーションの保存
                else:
                    save_anno(frame)
                    break

            #escキーが押されたらプログラム終了
            elif key == 27:
                fin_flag = False

                break
    else: #フレームが取り出せない場合,ループ終了
        break

cv2.destroyAllWindows() #windowを消す