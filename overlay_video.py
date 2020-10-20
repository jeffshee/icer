# -*- coding: utf-8 -*-
import datetime
import collections
import multiprocessing
import warnings

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFont, ImageDraw

from edit_audio import wav_to_transcript, split_wave_per_sentence
from edit_video import concat_video
from main_util import cleanup_directory, csv_to_text

warnings.simplefilter(action='ignore', category=FutureWarning)

from matplotlib import rcParams  # matplotlibで日本語を表示するための設定
from make_graph import get_graph_fig, get_transition_matrix

from os.path import join


def plot_tabel(df, save_path, colLabels=None):
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic',
                                   'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    if not colLabels:
        colLabels = df.columns
    the_table = ax.table(cellText=df.values,
                         colLabels=colLabels,
                         rowLabels=df.index,
                         loc='center',
                         cellLoc="center",
                         rowLoc="center",
                         colLoc="center",
                         bbox=(0, 0, 1, 1))
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(20)

    plt.savefig(save_path)
    plt.close()


def put_text_in_image_corner(img, text, font_size=1):
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 2, 4)[0]
    cv2.putText(img, " " + text, (0, size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 6)
    cv2.putText(img, " " + text, (0, size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 2)


def box_label(bgr, x1, y1, x2, y2, label, color):
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)  # 顔を囲む箱
    cv2.rectangle(bgr, (int(x1), int(y1 - 18)), (x2, y1), (255, 255, 255), -1)  # テキスト記入用の白いボックス
    cv2.putText(bgr, label, (x1, int(y1 - 5)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)


def add_border(input_image, border, color):
    img = Image.open(input_image)

    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border, fill=color)
    else:
        raise RuntimeError('Border is not an integer or tuple!')

    return bimg


def index_to_name(index_to_name_dict):
    def func(x, pos):
        return index_to_name_dict[x]

    return func


def ms_to_s(x, pos):
    return int(x // 1000)


def del_continual_value(target_list):
    ret_list = []
    last_value = None
    for x in target_list:
        if x != last_value:
            ret_list.append(x)
            last_value = x

    return ret_list


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


def overlay_parallel(input_movie_path, output_movie_path, fourcc, split_frame_list, df_diarization_sorted,
                     video_frame_rate, true_face_num, faces_path, index_to_name_dict, path_emo_rec, emotions, w_padding,
                     embedded_video_height, video_length, matching_index_list, df_diarization, embedded_video_width,
                     h_padding, h_transcript, summary_image, font_path, df_transcript, split_video_num,
                     split_video_index, speaker_switch):
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic',
                                   'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

    font = ImageFont.truetype(font_path, 32)

    input_movie = cv2.VideoCapture(input_movie_path)  # 動画を読み込む
    output_movie = cv2.VideoWriter(output_movie_path, fourcc, video_frame_rate,
                                   (embedded_video_width + w_padding, embedded_video_height + h_padding + h_transcript))

    frame_duration = video_length // split_video_num
    start_frame = split_video_index * frame_duration
    stop_frame = video_length if split_video_index == split_video_num - 1 else start_frame + frame_duration
    input_movie.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 動画の開始フレームを設定

    # 動画をフレーム単位で処理
    for frame_number in range(start_frame, stop_frame):
        time_ms = int((1000 * frame_number) // video_frame_rate)
        for i, split_frame in enumerate(split_frame_list):
            if frame_number < split_frame:
                talk_start_frame = split_frame_list[i - 1]
                talk_end_frame = split_frame_list[i]
                break

        #######################
        ### 話者間の関係性の可視化（グラフ） ###
        #######################
        # sp_switch_key = np.array(list(speaker_switch.keys()))
        # sp_switch = [speaker_switch[k] for k in sp_switch_key[(sp_switch_key <= time_ms).tolist()]]
        # speaker_id = [index_to_name_dict[k] for k in index_to_name_dict.keys() if k >= 0]
        # graph_edge = [[0] * len(speaker_id) for _ in range(len(speaker_id))]
        # for prev_sp, next_sp in sp_switch:
        #     if (prev_sp is None) or (next_sp is None):
        #         continue
        #     graph_edge[next_sp][prev_sp] = graph_edge[next_sp][prev_sp] + 1

        # graph_df = pd.DataFrame(index=speaker_id,
        #                 columns=speaker_id,
        #                 data=graph_edge)

        # print(speaker_switch)
        # print(graph_df)
        graph_df = get_transition_matrix(speaker_switch, index_to_name_dict, time_ms)
        dt_now = datetime.datetime.now()
        graph_fig = get_graph_fig(len(graph_df), graph_df, len(speaker_switch),
                                  'test{}_ID{}.png'.format(dt_now.strftime('%Y-%m-%d-%H-%M-%S'), split_video_index),
                                  add_flag=True)
        graph_fig = cv2.cvtColor(graph_fig, cv2.COLOR_RGBA2BGR)
        graph_fig = resize_with_original_aspect(graph_fig, int(embedded_video_width * 4 / 16), h_padding)

        # 現在のスピーカーを取得
        current_speaker_series = df_diarization_sorted[df_diarization_sorted["time(ms)"] == time_ms]["speaker class"]
        if current_speaker_series.tolist():
            current_speaker = current_speaker_series.tolist()[0]
        else:
            current_speaker = -1

        #######################
        ### 顔画像と感情を表示 ###
        #######################
        fig, axes = plt.subplots(nrows=true_face_num, ncols=3, figsize=(20, 20))
        for face_index, face_path in enumerate(faces_path):
            # 顔を表示
            if face_index == current_speaker:
                img = np.array(add_border(face_path, border=5, color='rgb(255,215,0)'))
            else:
                img = np.array(Image.open(face_path))

            plt.subplot(true_face_num, 3, (face_index * 3) + 1)
            plt.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,
                            labelright=False, labeltop=False)  # 目盛りの表示を消す
            plt.imshow(img, aspect="equal")
            plt.title(index_to_name_dict[face_index], loc="center", fontsize=18)

            # 感情のヒストグラムを表示
            df_emotion = pd.read_csv(join(path_emo_rec, "result{}.csv".format(face_index)), encoding="shift_jis",
                                     header=0,
                                     usecols=["prediction"])
            # 1 speech あたりの感情
            df_emotion_count = df_emotion['prediction'][talk_start_frame:talk_end_frame].value_counts()
            no_emotions = list(set(emotions) - set(df_emotion_count.index.values))
            for no_emotion in no_emotions:
                df_emotion_count[no_emotion] = 0
            df_emotion_count.sort_index(inplace=True)
            title = "Current" if face_index == 0 else None
            df_emotion_count.plot(kind="barh", ax=axes[face_index, 1], color=["blue", "green", "red", "gray"],
                                  xticks=[0, (talk_end_frame - talk_start_frame) // 2,
                                          talk_end_frame - talk_start_frame],
                                  xlim=(0, talk_end_frame - talk_start_frame), fontsize=18)
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

        ###########################
        ### Diarization結果の表示 ###
        ###########################
        # プロットの設定
        plt.xlim(0, video_length)
        if -1 in matching_index_list:  # unknown speakerが含まれている場合
            yticks = [-1] + [y for y in range(true_face_num)]
            ylim = (max(yticks) + 0.5, -1 - 0.5)
        else:
            yticks = [y for y in range(true_face_num)]
            ylim = (max(yticks) + 0.5, 0 - 0.5)

        # speakerを示す点をプロット
        df_diarization_filled = df_diarization.copy()
        for i in range(0, int(df_diarization["time(ms)"].min())):
            df_diarization_filled = df_diarization_filled.append(
                pd.Series([i, -1], index=df_diarization_filled.columns), ignore_index=True)
        for i in range(int(df_diarization["time(ms)"].max()), int((1000 * video_length) // video_frame_rate)):
            df_diarization_filled = df_diarization_filled.append(
                pd.Series([i, -1], index=df_diarization_filled.columns), ignore_index=True)
        df_diarization_filled.plot(kind="scatter", x="time(ms)", y="speaker class", marker="|", figsize=(10, 4),
                                   yticks=yticks, color="black", ylim=ylim)
        # 軸の目盛りを変更
        ax = plt.gca()
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(index_to_name(index_to_name_dict)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ms_to_s))
        ax.set_xlabel("time [s]")
        ax.set_ylabel(None)
        plt.draw()
        # 現時刻を示す縦棒をプロット
        if -1 in matching_index_list:  # unknown speakerが含まれている場合
            y_time_list = list(np.linspace(-1, true_face_num - 1, 50))
        else:
            y_time_list = list(np.linspace(0, true_face_num - 1, 50))
        plt.scatter(x=[time_ms] * len(y_time_list), y=y_time_list, marker="|", color="yellow")
        # gestureが検出された時刻に点をプロット
        for face_index in range(len(faces_path)):
            df_gesture = pd.read_csv(join(path_emo_rec, "result{}.csv".format(face_index)), encoding="shift_jis",
                                     header=0,
                                     usecols=["time(ms)", "gesture"])
            df_gesture_yes = df_gesture[df_gesture["gesture"] == 1]
            plt.scatter(x=df_gesture_yes["time(ms)"], y=[face_index] * len(df_gesture_yes), marker="|", color="red",
                        s=30)
        # matplotlib -> opencv へ画像の形式を変換
        fig = plt.gcf()
        fig.canvas.draw()
        plot_diarization = np.array(
            fig.canvas.renderer.buffer_rgba())  # im = np.array(fig.canvas.renderer._renderer) # matplotlibが3.1より前の場合
        plot_diarization = cv2.cvtColor(plot_diarization, cv2.COLOR_RGBA2BGR)
        plot_diarization_resized = resize_with_original_aspect(plot_diarization, int(embedded_video_width * 6 / 16),
                                                               h_padding)

        plt.close(fig)

        # zoomed diarization
        display_frame = 300
        df_diarization_filled[
            (frame_number + display_frame >= (df_diarization_filled["time(ms)"] * video_frame_rate // 1000)) & (
                    frame_number - display_frame <= (
                    df_diarization_filled["time(ms)"] * video_frame_rate // 1000))].plot(kind="scatter",
                                                                                         x="time(ms)",
                                                                                         y="speaker class",
                                                                                         marker="|",
                                                                                         figsize=(6, 4),
                                                                                         yticks=yticks,
                                                                                         color="black",
                                                                                         ylim=ylim)
        # 軸の目盛りを変更
        ax = plt.gca()
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(index_to_name(index_to_name_dict)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ms_to_s))
        ax.set_xlabel("time [s]")
        ax.set_ylabel(None)
        plt.subplots_adjust(left=0)
        plt.draw()
        # 現時刻を示す縦棒をプロット
        if -1 in matching_index_list:  # unknown speakerが含まれている場合
            y_time_list = list(np.linspace(-1, true_face_num - 1, 50))
        else:
            y_time_list = list(np.linspace(0, true_face_num - 1, 50))
        plt.scatter(x=[time_ms] * len(y_time_list), y=y_time_list, marker="|", color="yellow")
        # matplotlib -> opencv へ画像の形式を変換
        fig = plt.gcf()
        fig.canvas.draw()
        plot_diarization_zoomed = np.array(
            fig.canvas.renderer.buffer_rgba())  # im = np.array(fig.canvas.renderer._renderer) # matplotlibが3.1より前の場合
        plot_diarization_zoomed = cv2.cvtColor(plot_diarization_zoomed, cv2.COLOR_RGBA2BGR)
        plot_diarization_zoomed_resized = resize_with_original_aspect(plot_diarization_zoomed,
                                                                      int(embedded_video_width * 6 / 16), h_padding)
        plt.close(fig)

        #############################
        ### 動画から１フレーム読み込む ###
        #############################
        ret, frame = input_movie.read()
        # 動画が読み取れない場合は終了
        if not ret:
            break

        # ###########################
        # ### 議事録(Minutes)の表示 ###
        # ###########################
        # display_frame = 100
        # df_minutes_extracted = df_minutes[(frame_number >= (df_minutes["ms"] * video_frame_rate // 1000)) & ((df_minutes["ms"] * video_frame_rate // 1000) >= frame_number - display_frame)]
        # if not df_minutes_extracted.empty:
        #     # print(df_minutes_extracted)
        #     for i, x in enumerate(df_minutes_extracted.itertuples()):
        #         df_face_pos = pd.read_csv("{}result{}.csv".format(path_emo_rec, x[1]), encoding="shift_jis", header=0, usecols=["frame_number", "top", "left"])
        #         search_margin = 10
        #         while True:
        #             df_face_pos_extracted = df_face_pos[(df_face_pos["frame_number"] <= frame_number + search_margin) & (df_face_pos["frame_number"] >= frame_number - search_margin)]
        #             df_face_pos_extracted = df_face_pos_extracted[df_face_pos_extracted["top"] != 0]
        #             if df_face_pos_extracted.empty:
        #                 search_margin *= 2
        #             else:
        #                 break
        #         img_pil = Image.fromarray(frame)
        #         draw = ImageDraw.Draw(img_pil)
        #         draw.text((df_face_pos_extracted["left"].values.tolist()[0], df_face_pos_extracted["top"].values.tolist()[0] - 100), "{}: {}".format(x[1], x[3]), font=font, fill=(0, 0, 0))
        #         frame = np.array(img_pil)

        #######################
        ### Transcriptの表示 ###
        #######################
        img_transcript_pil = Image.fromarray(np.full((h_transcript, embedded_video_width, 3), 255, dtype=np.uint8))
        max_char = 100
        df_transcript_extracted = df_transcript[
            (frame_number >= (df_transcript["Start time(ms)"] * video_frame_rate // 1000)) & (
                    (df_transcript["End time(ms)"] * video_frame_rate // 1000) >= frame_number)]

        if not df_transcript_extracted.empty:
            draw_transcript = ImageDraw.Draw(img_transcript_pil)
            start = str(df_transcript_extracted["Start time(HH:MM:SS)"].item())
            end = str(df_transcript_extracted["End time(HH:MM:SS)"].item())
            text = str(df_transcript_extracted["Text"].item())
            display = " ".join([start, end, text])
            for i in range(len(display) // max_char):
                display = display[:(i + 1) * max_char] + "\n" + display[(i + 1) * max_char:]
            draw_transcript.text((20, 60), display, font=font, fill=(0, 0, 0))

        img_transcript = np.array(img_transcript_pil)

        #############################
        ### 画像を結合し動画として保存 ###
        #############################
        plot_diarization_resized = cv2.hconcat([plot_diarization_resized, plot_diarization_zoomed_resized, graph_fig])
        # plot_diarization_resized = cv2.hconcat([plot_diarization_resized, plot_diarization_zoomed_resized])
        put_text_in_image_corner(frame, text="Video")
        put_text_in_image_corner(img_transcript, text="Transcript")
        put_text_in_image_corner(plot_diarization_resized, text="Diarization")
        frame = resize_with_original_aspect(frame, embedded_video_width, embedded_video_height)
        frame[-5:, :, :] = 0
        img_transcript[-5:, :, :] = 0
        frame = cv2.vconcat([frame, img_transcript, plot_diarization_resized])
        face_and_index_img_resized[-5:, :, :] = 0
        put_text_in_image_corner(face_and_index_img_resized, text="Emotion")
        put_text_in_image_corner(summary_image, text="Summary")
        face_and_index_img_resized_v2 = cv2.vconcat([face_and_index_img_resized, summary_image])
        face_and_index_img_resized_v2[:, :5, :] = 0
        frame = cv2.hconcat([frame, face_and_index_img_resized_v2])
        output_movie.write(frame)

    # 動画を保存
    input_movie.release()
    cv2.destroyAllWindows()


def overlay_all_results(input_video_path, output_video_path, diarization_result_path, transcript_result_path,
                        matching_index_list, true_face_num,
                        emotion_dir, faces_path, video_name, index_to_name_dict, emotions, k_resolution,
                        split_video_num, output_dir):
    # df_minutes = pd.read_excel(path_minutes + "{}.xlsx".format(video_name), encoding="shift_jis", header=0, usecols=["name", "ms", "comment"])

    font_path = 'GenYoMinJP-Regular.ttf'

    # diarization結果を口の開閉とのマッチング結果に置き換える
    df_diarization = pd.read_csv(diarization_result_path, encoding="shift_jis", header=0,
                                 usecols=["time(ms)", "speaker class"])
    df_diarization_copy = df_diarization.copy()
    for speaker_class in range(len(df_diarization["speaker class"].unique())):
        df_diarization["speaker class"][df_diarization_copy["speaker class"] == speaker_class] = matching_index_list[
            speaker_class]

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
    h_padding = int(embedded_video_height // 2)
    h_transcript = int(embedded_video_height // 8)
    w_padding = int(embedded_video_width // 2)

    # speakerの切り替わりのタイミングを取得する
    df_diarization_sorted = df_diarization.copy()
    df_diarization_sorted.sort_values(by=["time(ms)"], ascending=True, inplace=True)
    df_diarization_sorted_diff = df_diarization_sorted.copy()
    df_diarization_sorted_diff["speaker class"] = df_diarization_sorted_diff["speaker class"].diff()
    df_split_ms = df_diarization_sorted_diff[df_diarization_sorted_diff["speaker class"] != 0]

    buff = df_diarization_sorted[df_diarization_sorted_diff["speaker class"] != 0].copy()
    speaker_swith = {}
    speaker_swith[buff["time(ms)"][0]] = [None, None]
    for prev_t, next_t, prev_speaker, next_speaker in zip(buff["time(ms)"][:-1], buff["time(ms)"][1:],
                                                          buff["speaker class"][:-1], buff["speaker class"][1:]):
        speaker_swith[next_t] = [prev_speaker, next_speaker]

    # 議論のsummary画像を作成する
    data_summary = [[] for _ in range(true_face_num)]
    columns_summary = ['Num of\nutterances', 'Speech\ntime [s]', "Speech\ndensity [s]", 'Time\noccupancy [%]',
                       "Num of\nnods"]
    new_columns_name = ['発話数', '発話時間 [s]', "発話密度 [s]", '会話占有率 [%]', "頷き回数"]
    num_of_utterances = collections.Counter(del_continual_value(df_diarization_sorted["speaker class"].to_list()))
    speech_time = df_diarization["speaker class"].value_counts() / 1000
    time_occupancy = df_diarization["speaker class"].value_counts(normalize=True)
    for x in range(true_face_num):
        try:
            data_summary[x].append(int(num_of_utterances[x]))
        except KeyError:
            data_summary[x].append(0)

        try:
            data_summary[x].append(int(speech_time[x]))
        except KeyError:
            data_summary[x].append(0)

        try:
            data_summary[x].append(int(speech_time[x] / num_of_utterances[x]))
        except KeyError:
            data_summary[x].append(0)

        try:
            data_summary[x].append(int(time_occupancy[x] * 100))
        except KeyError:
            data_summary[x].append(0)

        df_gesture_tmp = pd.read_csv(join(emotion_dir, "result{}.csv".format(x)), encoding="shift_jis", header=0,
                                     usecols=["gesture"])
        df_gesture_tmp = df_gesture_tmp[df_gesture_tmp.diff()["gesture"] != 0]
        gesture_count = df_gesture_tmp["gesture"].value_counts()
        try:
            data_summary[x].append(int(gesture_count[1]))
        except KeyError:
            data_summary[x].append(0)
    rows_summary = [" {} ".format(index_to_name_dict[index]) for index in range(len(faces_path))]
    df_summary = pd.DataFrame(data_summary, index=rows_summary, columns=columns_summary)
    print(df_summary)
    plot_tabel(df_summary, join(emotion_dir, "summary_{}.png".format(video_name)), colLabels=new_columns_name)
    summary_image = cv2.imread(join(emotion_dir, "summary_{}.png".format(video_name)))
    summary_image = resize_with_original_aspect(summary_image, w_padding, h_padding + h_transcript)

    # 動画として保存
    split_frame_list = [int(split_ms * video_frame_rate / 1000) for split_ms in df_split_ms["time(ms)"]]
    split_frame_list = [0] + split_frame_list + [video_length]
    print("split_frame_list", split_frame_list)

    df_transcript = pd.read_csv(transcript_result_path, encoding="utf_8_sig", header=0,
                                usecols=['Order', 'Start time(HH:MM:SS)', 'End time(HH:MM:SS)', 'Text', 'Speaker',
                                         'Start time(ms)', 'End time(ms)'])
    # 並列処理で動画を作成する
    split_video_num_v2 = split_video_num * 2
    # split_video_num_v2 = split_video_num
    process_list = []

    # https://github.com/opencv/opencv/issues/5150
    multiprocessing.set_start_method('spawn')
    for split_video_index in range(split_video_num_v2):
        # if split_video_index > 0:
        #     continue
        split_result_path = join(output_dir, "split_video_overlay_{}".format(split_video_index))
        cleanup_directory(split_result_path)
        process_list.append(multiprocessing.Process(target=overlay_parallel,
                                                    args=(input_video_path,
                                                          join(output_dir,
                                                               "split_video_overlay_{}".format(split_video_index),
                                                               "output.avi"),
                                                          fourcc, split_frame_list, df_diarization_sorted,
                                                          video_frame_rate, true_face_num,
                                                          faces_path, index_to_name_dict, emotion_dir, emotions,
                                                          w_padding, embedded_video_height, video_length,
                                                          matching_index_list, df_diarization, embedded_video_width,
                                                          h_padding,
                                                          h_transcript, summary_image, font_path, df_transcript,
                                                          split_video_num_v2, split_video_index, speaker_swith)))
        process_list[split_video_index].start()
    for process in process_list:
        process.join()

    # 分割して処理した結果を結合
    input_video_list = [join(output_dir, "split_video_overlay_{}".format(i), "output.avi") for i in
                        range(split_video_num_v2)]
    concat_video(input_video_list, output_video_path)


if __name__ == "__main__":
    input_movie_path = "video/200225_Haga_22.mp4"

    path_diarization = 'diarization/200225_Haga_22_0.5_0.5/'
    df_diarization = pd.read_csv("{}result.csv".format(path_diarization), encoding="shift_jis", header=0,
                                 usecols=["time(ms)", "speaker class"])
    input_movie = cv2.VideoCapture(input_movie_path)  # 動画を読み込む
    video_length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = input_movie.get(cv2.CAP_PROP_FPS)  # 動画のフレームレートを取得

    # speakerの切り替わりのタイミングを取得する
    df_diarization_sorted = df_diarization.copy()
    df_diarization_sorted.sort_values(by=["time(ms)"], ascending=True, inplace=True)
    df_diarization_sorted_diff = df_diarization_sorted.copy()
    df_diarization_sorted_diff["speaker class"] = df_diarization_sorted_diff["speaker class"].diff()
    df_split_ms = df_diarization_sorted_diff[df_diarization_sorted_diff["speaker class"] != 0]

    split_wave_per_sentence(df_diarization_sorted, df_split_ms["time(ms)"].values.tolist(),
                            "audio/200225_Haga_22.wav", "test/split_wave/")
