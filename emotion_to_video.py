##感情認識の結果の可視化
import os
import pandas as pd
import cv2
def emo_to_video(face_dir,emo_files_dir,dia_dir,input_video_path):
    # 出力動画の詳細を設定する
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画コーデック指定
    input_movie = cv2.VideoCapture(input_video_path)  # 動画を読み込む
    video_length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = input_movie.get(cv2.CAP_PROP_FPS)  # 動画のフレームレートを取得

    face_dir_copy=face_dir
    num_dirs=0
    for _,dirs,_ in os.walk(face_dir_copy):
        for _ in dirs:
            num_dirs=num_dirs+1
    per_num=num_dirs

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

    print(split_frame_list)



emo_to_video("main_test (copy)/face/cluster/",1,1,"main_test (copy)/emotion/output.avi")