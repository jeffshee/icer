##感情認識の結果の可視化
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from PIL import Image, ImageOps, ImageFont, ImageDraw

def add_border(input_image, border, color):
    img = Image.open(input_image)

    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border, fill=color)
    else:
        raise RuntimeError('Border is not an integer or tuple!')

    return bimg

def resize_with_original_aspect(img, base_w, base_h):
    base_ratio = base_w / base_h  # リサイズ画像サイズ縦横比
    img_h, img_w = img.shape[:2]  # 画像サイズ
    img_ratio = img_w / img_h  # 画像サイズ縦横比

    white_img = np.zeros((base_h, base_w, 3), np.uint8)  # 白塗り画像のベース作成
    white_img[:, :] = [255, 255, 255]  # 白塗り

    # 画像リサイズ, 白塗りにオーバーレイ
    if img_ratio > base_ratio:
        h = int(base_w / img_ratio)  # 横から縦を計算
        w = base_w  # 横を合わせる
        resize_img = cv2.resize(img, (w, h))  # リサイズ
    else:
        h = base_h  # 縦を合わせる
        w = int(base_h * img_ratio)  # 縦から横を計算
        resize_img = cv2.resize(img, (w, h))  # リサイズ

    white_img[int(base_h / 2 - h / 2):int(base_h / 2 + h / 2),
    int(base_w / 2 - w / 2):int(base_w / 2 + w / 2)] = resize_img  # オーバーレイ
    resize_img = white_img  # 上書き

    return resize_img

def emo_to_video(face_dir,emo_files_dir,dia_dir,input_video_path,output_movie_path,k_resolution):
    # 出力動画の詳細を設定する
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画コーデック指定
    input_movie = cv2.VideoCapture(input_video_path)  # 動画を読み込む

    video_length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = input_movie.get(cv2.CAP_PROP_FPS)  # 動画のフレームレートを取得
    original_w = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))  # 動画の幅を取得
    original_h = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 動画の高さを取得
    resize_rate = (1080 * k_resolution) / original_w
    embedded_video_width = int(original_w * resize_rate)
    embedded_video_height = int(original_h * resize_rate)
    w_padding = int(embedded_video_width // 2) ##出力動画の幅

    output_movie = cv2.VideoWriter(output_movie_path, fourcc, video_frame_rate,
                                   (w_padding, embedded_video_height))

    face_dir_copy=face_dir
    num_dirs=0
    for _,dirs,_ in os.walk(face_dir_copy):
        for _ in dirs:
            num_dirs=num_dirs+1
    per_num=num_dirs
    ##debug
    print("per_num",per_num)
    return 0

    # diarizationの結果を読み込む
    df_diarization = pd.read_csv(dia_dir, encoding="shift_jis", header=0,
                                 usecols=["time(ms)", "speaker class"])
    df_diarization_sorted = df_diarization.copy()
    df_diarization_sorted.sort_values(by=["time(ms)"], ascending=True, inplace=True)
    df_diarization_sorted_diff = df_diarization_sorted.copy()
    df_diarization_sorted_diff["speaker class"] = df_diarization_sorted_diff["speaker class"].diff()
    df_split_ms = df_diarization_sorted_diff[df_diarization_sorted_diff["speaker class"] != 0]
    buff = df_diarization_sorted[df_diarization_sorted_diff["speaker class"] != 0].copy()
    speaker_swith = {}
    speaker_swith[buff["time(ms)"][0]] = [None, None]
    for prev_t, next_t, prev_speaker, next_speaker in zip(buff["time(ms)"][:-1], buff["time(ms)"][1:],
                                                          buff["speaker class"][:-1], buff["speaker class"][1:]):##从第一位到倒数第二位 从第二位到倒数第一位的数据
        speaker_swith[next_t] = [prev_speaker, next_speaker] ##获取切换时间点前一个人和后一个人的转换顺序

    # 動画として保存
    split_frame_list = [int(split_ms * video_frame_rate / 1000) for split_ms in df_split_ms["time(ms)"]]
    split_frame_list = [0] + split_frame_list + [video_length]

    for split_index,split_frame in enumerate(split_frame_list):
        if split_index==0:
            continue
        talk_start_frame = split_frame_list[split_index - 1]
        talk_end_frame = split_frame_list[split_index]
        # 現在のスピーカーを取得
        time_ms = int((1000 * split_frame) // video_frame_rate)
        current_speaker_series = df_diarization_sorted[df_diarization_sorted["time(ms)"] == time_ms]["speaker class"]
        if current_speaker_series.tolist():
            current_speaker = current_speaker_series.tolist()[0]
        else:
            current_speaker = -1
        #######################
        ### 顔画像と感情を表示 ###
        #######################
        fig, axes = plt.subplots(nrows=per_num, ncols=3, figsize=(20, 20))
        clustered_face_list = os.listdir(face_dir)
        faces_path = [join(face_dir, name, 'closest.PNG') for name in clustered_face_list]

        for face_index, face_path in enumerate(faces_path):
            # 顔を表示
            if face_index == current_speaker:
                img = np.array(add_border(face_path, border=5, color='rgb(255,215,0)')) ##当前讲话人添加边界
            else:
                img = np.array(Image.open(face_path))

            plt.subplot(per_num, 3, (face_index * 3) + 1)
            plt.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,
                            labelright=False, labeltop=False)  # 目盛りの表示を消す
            plt.imshow(img, aspect="equal")
            # plt.title(index_to_name_dict[face_index], loc="center", fontsize=18) ##后续修改

            # 感情のヒストグラムを表示
            df_emotion = pd.read_csv(join(emo_files_dir, "result{}.csv".format(face_index)), encoding="shift_jis",
                                     header=0,
                                     usecols=["prediction"])
            # 1 speech あたりの感情
            emotions = ['Negative', 'Normal', 'Positive', 'Unknown']
            df_emotion_count = df_emotion['prediction'][talk_start_frame:talk_end_frame].value_counts()
            no_emotions = list(set(emotions) - set(df_emotion_count.index.values))
            for no_emotion in no_emotions:
                df_emotion_count[no_emotion] = 0
            df_emotion_count.sort_index(inplace=True)
            title = "Current" if face_index == 0 else None  ##仅添加一次标题
            df_emotion_count.plot(kind="barh", ax=axes[face_index, 1], color=["blue", "green", "red", "gray"],
                                  xticks=[0, (talk_end_frame - talk_start_frame) // 2,
                                          talk_end_frame - talk_start_frame],
                                  xlim=(0, talk_end_frame - talk_start_frame), fontsize=18) ##作图
            axes[face_index, 1].set_title(title, fontsize=18)

            # 累積感情
            df_emotion_count = df_emotion['prediction'][0:talk_end_frame].value_counts()
            no_emotions = list(set(emotions) - set(df_emotion_count.index.values))
            for no_emotion in no_emotions:
                df_emotion_count[no_emotion] = 0
            df_emotion_count.sort_index(inplace=True)
            title = "Total" if face_index == 0 else None
            df_emotion_count.plot(kind="barh", ax=axes[face_index, 2], color=["blue", "green", "red", "gray"],
                                  xticks=[0, (talk_end_frame) // 2, talk_end_frame], xlim=(0, talk_end_frame),
                                  fontsize=18)
            axes[face_index, 2].set_title(title, fontsize=18)
        plt.subplots_adjust(wspace=0.40)  # axe間の余白を調整
        fig.canvas.draw()
        face_and_index_img = np.array(
            fig.canvas.renderer.buffer_rgba())  # im = np.array(fig.canvas.renderer._renderer) # matplotlibが3.1より前の場合
        face_and_index_img = cv2.cvtColor(face_and_index_img, cv2.COLOR_RGBA2BGR)
        face_and_index_img_resized = resize_with_original_aspect(face_and_index_img, w_padding, embedded_video_height)
        plt.close(fig)

        for frame_index in range(talk_end_frame-talk_start_frame):
            frame = cv2.hconcat([frame, face_and_index_img_resized])
            output_movie.write(frame)

    # 動画を保存
    #test
    input_movie.release()
    cv2.destroyAllWindows()



emo_to_video("main_test (copy)/face/cluster/","main_test (copy)/emotion/","main_test (copy)/transcript/diarization/result.csv","main_test (copy)/emotion/output.avi","output/",3)