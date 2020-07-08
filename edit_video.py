# coding: utf-8
import datetime
import os
import subprocess

import math


def calibrate_video(original_video_path, output_video_path, k_resolution, capture_image_num=50, model='hog' use_gpu=False):
    input_movie = cv2.VideoCapture(original_video_path)  # 動画を読み込む
    original_w = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))  # 動画の幅を測る
    original_h = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 動画の高さを測る
    resize_rate = (1080 * k_resolution) / original_w  # 動画の横幅を変更
    w = int(original_w * resize_rate)
    h = int(original_h * resize_rate)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画の長さを測る
    max_face_detection_num = 0
    calibrate_num = 0
    capture_image_num = min(capture_image_num, length)

    for i, frame_number in enumerate(range(0, length, length // capture_image_num)):
        print("frame_number:", frame_number)
        input_movie.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # 動画の開始フレームを設定

        # 動画を読み込む
        ret, frame = input_movie.read()
    
        # フレームのサイズを調整する
        frame = cv2.resize(frame, (w, h))

        # 動画が読み取れない場合は終了
        if not ret:
            break

        # openCVのBGRをRGBに変更
        rgb_frame = frame[:, :, ::-1]

        # 顔検出が最多になる動画の横方向のシフト数を算出
        face_location_list = face_recognition.face_locations(rgb_frame, model=model)  # model="cnn"にすると検出率は上がるが10倍以上時間がかかる top,right,bottom,leftの順
        if i == 0: max_face_detection_num = len(face_location_list)
        base_face_detection_num_in_frame = len(face_location_list)

        for dw in range(w//5, w, w//5):
            face_location_list = face_recognition.face_locations(np.append(rgb_frame[:, dw:, :], rgb_frame[:, :dw, :], axis=1), model="hog")  # model="cnn"にすると検出率は上がるが10倍以上時間がかかる top,right,bottom,leftの順
            face_detection_num_in_frame = len(face_location_list)
            if (face_detection_num_in_frame - base_face_detection_num_in_frame) > 0 and (max_face_detection_num < len(face_location_list)):
                max_face_detection_num = max(max_face_detection_num, len(face_location_list))
                calibrate_num = dw//k_resolution
            print('MAX: {}'.format(max_face_detection_num))
        print('calibrate_num: {}'.format(calibrate_num))

    if calibrate_num != 0:
        if use_gpu: cmd = "ffmpeg -i {} -vcodec h264_nvenc -vf crop={}:{}:{}:{} {}_right.mp4".format(original_video_path, original_w-calibrate_num, original_h, calibrate_num, 0, output_video_path)
        else: cmd = "ffmpeg -i {} -vf crop={}:{}:{}:{} {}_right.mp4".format(original_video_path, original_w-calibrate_num, original_h, calibrate_num, 0, output_video_path)
        proc = subprocess.Popen(cmd, shell=True)
        time.sleep(5)
        if use_gpu: cmd = "ffmpeg -i {} -vcodec h264_nvenc -vf crop={}:{}:{}:{} {}_left.mp4".format(original_video_path, calibrate_num, original_h, 0, 0, output_video_path)
        # if use_gpu: cmd = "ffmpeg -i {} -vcodec h264_nvenc -vf crop={}:{}:{}:{} {}_left.mp4".format(original_video_path, original_w-calibrate_num, original_h, calibrate_num, 0, output_video_path)
        else: cmd = "ffmpeg -i {} -vf crop={}:{}:{}:{} {}_left.mp4".format(original_video_path, calibrate_num, original_h, 0, 0, output_video_path)
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()  # 一番最後の動画のトリミングが終了するまで待つ
        input_video_list = ["{}_left.mp4".format(output_video_path), "{}_right.mp4".format(output_video_path)]
        concat_video_path = "{}_calibrated.mp4".format(output_video_path)
        if use_gpu: cmd = " ffmpeg -i {} -i {} -vcodec h264_nvenc -filter_complex hstack {}".format(input_video_list[0], input_video_list[1], concat_video_path)
        else: cmd = " ffmpeg -i {} -i {} -filter_complex hstack {}".format(input_video_list[0], input_video_list[1], concat_video_path)
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()  # 一番最後の動画のトリミングが終了するまで待つ


def split_video(original_video_path, split_num):  # ms単位で正確には分割できないので，結局使ってない
    video_length_format = get_length(original_video_path)
    video_length_sec = datetime.timedelta(hours=int(video_length_format[0]),
                                        minutes=int(video_length_format[1]),
                                        seconds=int(video_length_format[2].split(".")[0]),
                                        milliseconds=math.modf(float(video_length_format[2]))[
                                                           0] * 1000).total_seconds()
    time_duration_sec = video_length_sec // split_num
    time_duration_format = str(datetime.timedelta(seconds=time_duration_sec)) + ".00000"
    for i in range(split_num):
        output_video_path = original_video_path.replace(".", "_part{}.".format(i))

        start_frame_sec = i * time_duration_sec
        start_frame_format = str(datetime.timedelta(seconds=start_frame_sec)) + ".00000"
        if i == split_num - 1:
            time_duration_sec = video_length_sec - time_duration_sec * (split_num - 1)
            time_duration_format = str(datetime.timedelta(seconds=time_duration_sec))
        print(start_frame_sec, start_frame_sec + time_duration_sec)
        print(start_frame_format, time_duration_format)

        cmd = "ffmpeg -y -ss {} -i {} -ss 0 -t {} -c:v copy -strict -2 -an {}".format(start_frame_format,
                                                                                    original_video_path,
                                                                                    time_duration_format,
                                                                                    output_video_path)
        # cmd = "ffmpeg -y -ss {} -i {} -ss 0 -t {} -c:v copy -c:a copy -async 1 -strict -2 {}".format(start_frame_format, original_video_path, time_duration_format, output_video_path)
        popen = subprocess.Popen(cmd, shell=True)

    popen.wait()  # 一番最後の動画のトリミングが終了するまで待つ


def concat_video(input_video_list, concat_video_path, use_gpu=False):
    with open('input_video_list.txt', 'w') as f:
        for x in input_video_list:
            f.write("file '" + str(x) + "'\n")
    
    if use_gpu: cmd = "ffmpeg -y -safe 0 -f concat -i input_video_list.txt -vcodec h264_nvenc -c:v copy -map 0:v {}".format(concat_video_path)
    else: cmd = "ffmpeg -y -safe 0 -f concat -i input_video_list.txt -c:v copy -map 0:v {}".format(concat_video_path)
    # cmd = "ffmpeg -y -safe 0 -f concat -i input_video_list.txt -c:v copy -c:a copy -map 0:v -map 0:a {}".format(concat_video_path)
    subprocess.call(cmd, shell=True)
    os.remove('input_video_list.txt')


def get_length(filename):
    result = subprocess.Popen(["ffprobe", filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = [x.decode() for x in result.stdout.readlines() if "Duration" in x.decode()][0]
    duration = duration.split("Duration: ")[1]
    duration = duration.split(",")[0]
    duration = duration.split(":")
    return duration


def trim_video(input_video_path, trim_time, use_gpu=False):
    start_time = trim_time[0]  # "00:18:00"
    end_time = trim_time[1]  # "00:37:00"
    output_video_path = input_video_path.replace(".", "_trim.")
    if use_gpu: cmd = "ffmpeg -y -i {} -vcodec h264_nvenc -ss {} -to {} -c:v copy -c:a copy ./video/{}".format(input_video_path, start_time, end_time, output_video_path)
    else: cmd = "ffmpeg -y -i {} -ss {} -to {} -c:v copy -c:a copy ./video/{}".format(input_video_path, start_time, end_time, output_video_path)
    subprocess.Popen(cmd, shell=True)


if __name__ == "__main__":
    # video_name = "Take01"
    # original_video_path = "video/{}.mp4".format(video_name)
    # split_num = 10
    #
    # split_video(original_video_path, split_num)
    #
    # input_video_list = [original_video_path.replace(".", "_part{}.".format(i)) for i in range(split_num)]
    # concat_video_path = original_video_path.replace(".", "_concat.")
    # concat_video(input_video_list, concat_video_path)

    video_name = "video/200225_Haga_22.mp4"
    trim_video(video_name, ["00:00:00", "00:03:00"])
