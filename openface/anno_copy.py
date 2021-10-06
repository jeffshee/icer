import cv2
import numpy as np

img_path = "./pos/" #画像保存先のディレクトリ
poslist = "./pos/poslist.txt" #正解画像リスト
video_path = "./nod_1.mp4" #動画ファイルのパス

#位置格納用
point = np.empty((0,2), dtype = np.int64) #確定位置
point_drawing = np.empty(0, dtype = np.int64) #暫定位置

#位置格納関数
def mouse_event(event, x, y, flags, param):
    #グローバル変数を利用
    global point, point_drawing
    #マウス左ボタン押下時
    if event == cv2.EVENT_LBUTTONDOWN:
        point = np.append(point, np.array([[x,y]]), axis = 0) #位置格納
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
def save_img(img, frame):
    #グローバル変数を利用
    global point
    #画像の保存
    img_name = str(frame) + ".jpg" #画像名
    cv2.imwrite(img_path + img_name, img) #画像の保存

    #poslistの保存
    point = point.reshape((-1,4))
    pt_num = int(len(point)) #矩形数

    pt_txt = ' ' + str(pt_num) #出力用

    for i in range(pt_num):
        pt_txt = pt_txt +  ' ' + str(point[i][0]) + ' ' + str(point[i][1]) + ' ' + str(point[i][2]-point[i][0]) + ' ' + str(point[i][3]-point[i][1])

    txt_data = img_name + pt_txt + '\n' #画像名と共に保存
    with open(poslist, 'a') as txt: #ポジリスト
        txt.write(txt_data)

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
            #sキーが押されたら,データの保存,次のフレーム
            if key == ord('s'):
                save_img(img, frame)
                point = np.empty((0,2), dtype=np.int64)
                break
            #dキーが押されたら,矩形データを削除
            elif key == ord('d'):
                point = np.empty((0,2), dtype=np.int64)
            #rキーが押されたら次のフレーム
            elif key == ord('r'):
                point = np.empty((0,2), dtype=np.int64)
                break
            #escキーが押されたらプログラム終了
            elif key == 27:
                fin_flag = False
                break
    else: #フレームが取り出せない場合,ループ終了
        break

cv2.destroyAllWindows() #windowを消す