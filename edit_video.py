# coding: utf-8
import datetime
import os
import subprocess

import math


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


def concat_video(input_video_list, concat_video_path):
    with open('input_video_list.txt', 'w') as f:
        for x in input_video_list:
            f.write("file '" + str(x) + "'\n")

    cmd = "ffmpeg -y -safe 0 -f concat -i input_video_list.txt -c:v copy -map 0:v {}".format(concat_video_path)
    # cmd = "ffmpeg -y -safe 0 -f concat -i input_video_list.txt -c:v copy -c:a copy -map 0:v -map 0:a {}".format(concat_video_path)
    subprocess.call(cmd, shell=True)
    os.remove('input_video_list.txt')


def get_length(filename):
    result = subprocess.Popen(["ffprobe", filename],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = [x.decode() for x in result.stdout.readlines() if "Duration" in x.decode()][0]
    duration = duration.split("Duration: ")[1]
    duration = duration.split(",")[0]
    duration = duration.split(":")
    return duration


def trim_video(input_video_path, trim_time):
    start_time = trim_time[0]  # "00:18:00"
    end_time = trim_time[1]  # "00:37:00"
    output_video_path = input_video_path.replace(".", "_trim.")
    cmd = "ffmpeg -y -i {} -ss {} -to {} -c:v copy -c:a copy {}".format(input_video_path, start_time, end_time,
                                                                        output_video_path)
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
