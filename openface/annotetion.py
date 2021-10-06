import cv2
import numpy as np
import csv

img_path = "./pos/" #画像保存先のディレクトリ
csv_path = "./pos/anno.csv" #アノテーションリスト
video_path = "./nod_1.mp4" #動画ファイルのパス

# csvにコラム書き込み
with open(csv_path, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "nod"])

#位置格納用
point = np.empty((0,2), dtype = np.int64) #確定位置
point_drawing = np.empty(0, dtype = np.int64) #暫定位置

nod_flag = 0 #0,1のフラグ
# record = [] #フラグ記録用

#位置格納関数
def mouse_event(event, x, y, flags, param):
    #グローバル変数を利用
    global point, point_drawing, nod_flag, paint
    #マウス左ボタン押下時
    if event == cv2.EVENT_LBUTTONDOWN:
        point = np.append(point, np.array([[x,y]]), axis = 0) #位置格納
        nod_flag = (nod_flag + 1) % 2 # flagを反転

    #マウス左ボタン解放時
    elif event == cv2.EVENT_LBUTTONUP:
        point = np.append(point, np.array([[x,y]]), axis = 0) #位置格納
        point_drawing = np.empty(0, dtype = np.int64) #暫定位置
    #移動操作
    elif event == cv2.EVENT_MOUSEMOVE:
        pt_num = int(len(point))
        if pt_num % 2 == 1:
            point_drawing = [point[pt_num-1],[x,y]] #暫定位置格納

#保存関数
def save_anno(frame):
    #グローバル変数を利用
    global nod_flag
    
    # csvに保存
    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([frame, nod_flag])


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
        #フレーム番号の描画
        frame += 1 #フレーム番号の加算

        #ループ処理
        while True:
            paint = img.copy() #データのコピー

            cv2.putText(paint, str(frame), (0, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2) #フレーム番号の描画

            # 頷いてる場合
            if nod_flag == 1:
                cv2.putText(paint, "NOD", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

            # 頷いていない場合
            else:
                cv2.putText(paint, "X", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

            #矩形座標が決まっているもののみ描画
            for i in range(int(len(point)/2)):
                cv2.rectangle(paint, tuple(point[2*i]), tuple(point[2*i+1]), (0,0,255), thickness=2) #描画

            #暫定位置の描画
            if len(point_drawing) == 2:
                cv2.rectangle(paint, tuple(point_drawing[0]), tuple(point_drawing[1]), (0,0,255), thickness=2) #描画

            #windowに描画する
            cv2.imshow("window", paint)

            #キー入力を待つ
            key = cv2.waitKey(10)
            #sキーが押されたら,アノテーションの保存,次のフレーム
            if key == ord('s'):
                save_anno(frame)
                point = np.empty((0,2), dtype=np.int64)
                break
            #dキーが押されたら,矩形データを削除
            elif key == ord('d'):
                point = np.empty((0,2), dtype=np.int64)
            #rキーが押されたら次のフレーム
            # elif key == ord('r'):
            #     point = np.empty((0,2), dtype=np.int64)
            #     break
            #escキーが押されたらプログラム終了
            elif key == 27:
                fin_flag = False

                break
    else: #フレームが取り出せない場合,ループ終了
        break

cv2.destroyAllWindows() #windowを消す