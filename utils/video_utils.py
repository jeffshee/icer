import cv2
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from face_utils.face import Face
from multiprocessing import Process
import pandas as pd

"""
Video utils (v2)
"""


# def get_frame_position(video_capture: cv2.VideoCapture) -> int:
#     return int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) + 1
#
#
# def set_frame_position(video_capture: cv2.VideoCapture, position: int) -> int:
#     return int(video_capture.set(cv2.CAP_PROP_POS_FRAMES, position - 1))
#
#
# def get_video_dimension(video_capture: cv2.VideoCapture) -> Tuple[int, int]:
#     return int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#
# def get_video_length(video_capture: cv2.VideoCapture) -> int:
#     return int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
#
#
# def get_video_framerate(video_capture: cv2.VideoCapture) -> float:
#     return float(video_capture.get(cv2.CAP_PROP_FPS))

def get_video_capture(video_path: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(video_path)


def img_offset(img: np.ndarray, offset: int = 0):
    img = np.concatenate([img[:, offset:], img[:, :offset]], axis=1)
    return img


def read_video_capture(video_capture: cv2.VideoCapture, offset=0):
    retval, image = video_capture.read()
    if retval:
        image = img_offset(image, offset)
    return retval, image


def get_frame_position(video_capture: cv2.VideoCapture) -> int:
    return int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))


def set_frame_position(video_capture: cv2.VideoCapture, position: int) -> int:
    return int(video_capture.set(cv2.CAP_PROP_POS_FRAMES, position))


def get_video_dimension(video_path: str) -> Tuple[int, int]:
    video_capture = get_video_capture(video_path)
    return int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


def get_video_length(video_path: str) -> int:
    video_capture = get_video_capture(video_path)
    return int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


def get_video_framerate(video_path: str) -> float:
    video_capture = get_video_capture(video_path)
    return float(video_capture.get(cv2.CAP_PROP_FPS))


def concat_video(video_path_list: List[str], output_path: str, console_quiet=True, remove_src=True):
    """
    Concat video (by using ffmpeg)
    Note that videos must have same properties (height, width, codec, etc.),
    as this function doesn't re-encode the videos.
    :param video_path_list: List of video paths
    :param output_path: Output path
    :param console_quiet: Avoid output to console
    :param remove_src: Remove videos in video_path_list when done
    :return: 
    """
    import subprocess
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dirname:
        print("Temporary directory created at", temp_dirname)
        list_path = os.path.join(temp_dirname, "video_path_list.txt")
        with open(list_path, 'w') as f:
            for input_video in video_path_list:
                f.write(f"file {os.path.abspath(input_video)}\n")

        quiet = "-loglevel quiet > /dev/null 2>&1 < /dev/null" if console_quiet else ""
        # -map 0:v is probably a workaround for muted video
        cmd = f"ffmpeg -y -safe 0 -f concat -i {list_path} -c:v copy -map 0:v {output_path} {quiet}"
        subprocess.call(cmd, shell=True)

    if remove_src:
        for input_video in video_path_list:
            os.remove(input_video)


def crop_video(video_path: str, output_path: str, roi, start_time=0, end_time=-1, remove_audio=False,
               console_quiet=True, use_gpu=True):
    """
    Crop video (by using ffmpeg)
    :param video_path: Video path
    :param output_path: Output path
    :param roi: Region to crop
    :param start_time: Start time (Use to trim the output)
    :param end_time: End time (Use to trim the output)
    :param remove_audio: Remove audio track from output
    :param console_quiet: Avoid output to console
    :param use_gpu: Use GPU during the encoding
    :return:
    """

    import time
    import subprocess

    # Trim
    start_time = time.strftime("%H:%M:%S", time.gmtime(start_time))
    end_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
    trim = f"-ss {start_time} -to {end_time}" if end_time != -1 else f"-ss {start_time}"

    # Audio
    audio = "-an" if remove_audio else "-c:a copy"

    x, y, w, h = roi

    codec = "h264_nvenc" if use_gpu else "libx264"
    lossless = "-crf 0"
    quiet = "-loglevel quiet > /dev/null 2>&1 < /dev/null" if console_quiet else ""

    cmd = f"""
            ffmpeg -y -i {video_path} {trim} -filter:v "crop={w}:{h}:{x}:{y}" \
            -c:v {codec} {lossless} {audio} {output_path} {quiet}
            """
    print(cmd)
    subprocess.call(cmd, shell=True)


def trim_video(video_path: str, output_path: str, start_time=0, end_time=-1, remove_audio=False, console_quiet=True):
    """
    Trim video (by using ffmpeg)
    :param video_path: Video path
    :param output_path: Output path
    :param start_time: Start time (Use to trim the output)
    :param end_time: End time (Use to trim the output)
    :param remove_audio: Remove audio track from output
    :param console_quiet: Avoid output to console
    :return:
    """
    import time
    import subprocess

    # Trim
    start_time = time.strftime("%H:%M:%S", time.gmtime(start_time))
    end_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
    trim = f"-ss {start_time} -to {end_time}" if end_time != -1 else f"-ss {start_time}"

    # Audio
    audio = "-an" if remove_audio else "-c:a copy"

    quiet = "-loglevel quiet > /dev/null 2>&1 < /dev/null" if console_quiet else ""
    cmd = f'ffmpeg -y -i {video_path} {trim} -c:v copy {audio} {output_path} {quiet}'
    subprocess.call(cmd, shell=True)


def calibrate_video(video_path: str, output_path: str, offset=0, start_time=0, end_time=-1,
                    remove_audio=False, console_quiet=True, use_gpu=True):
    """
    Calibrate 360-video angle (by using ffmpeg)
    :param video_path: Video path
    :param output_path: Output path
    :param offset: Offset
    :param start_time: Start time (Use to trim the output)
    :param end_time: End time (Use to trim the output)
    :param remove_audio: Remove audio track from output
    :param console_quiet: Avoid output to console
    :param use_gpu: Use GPU during the encoding
    :return:
    """
    import time
    import subprocess

    # Trim
    start_time = time.strftime("%H:%M:%S", time.gmtime(start_time))
    end_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
    trim = f"-ss {start_time} -to {end_time}" if end_time != -1 else f"-ss {start_time}"

    # Audio
    audio = "-an" if remove_audio else "-c:a copy"

    flip = -1 if offset < 0 else 1

    codec = "h264_nvenc" if use_gpu else "libx264"
    lossless = "-crf 0"
    quiet = "-loglevel quiet > /dev/null 2>&1 < /dev/null" if console_quiet else ""

    cmd = f"""
            ffmpeg -y -i {video_path} {trim} -filter_complex \
            "[0:v][0:v]overlay=-{offset}[bg]; \
            [bg][0:v]overlay={flip}*W-{offset},format=yuv420p[out]" \
            -map "[out]" -map 0:a? -c:v {codec} {lossless} {audio} {output_path} {quiet}
            """
    subprocess.call(cmd, shell=True)


def output_video_emotion(interpolated_result: dict, emotion_csv_path_list: list, video_path: str, output_path: str,
                         offset=0, parallel_num=1, rank=0):
    from tqdm import tqdm

    def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            p = (x, y)
            pts.append(p)

        if style == 'dotted':
            for p in pts:
                cv2.circle(img, p, thickness, color, -1)
        else:
            s = pts[0]
            e = pts[0]
            i = 0
            for p in pts:
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(img, s, e, color, thickness)
                i += 1
        return img

    def drawpoly(img, pts, color, thickness=1, style='dotted', ):
        s = pts[0]
        e = pts[0]
        pts.append(pts.pop(0))
        for p in pts:
            s = e
            e = p
            drawline(img, s, e, color, thickness, style)
        return img

    def drawrect(img, pt1, pt2, color, thickness=1, style='dotted', label="TEST", font_color=(255, 255, 255)):
        if style == "line":
            cv2.rectangle(img, pt1, pt2, color, thickness)
        else:
            pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
            drawpoly(img, pts, color, thickness, style)
        if label is not None:
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            text_w, text_h = text_size
            cv2.rectangle(img, (pt1[0], pt1[1] - 16 - text_h), (pt1[0] + text_w, pt1[1] - 16), (0, 0, 0), -1)
            cv2.putText(img, label, (pt1[0], pt1[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, font_color, 2)
        return img

    video_capture = get_video_capture(video_path)
    video_length = get_video_length(video_path)
    divided = video_length // parallel_num  # Frames per process
    frame_range = [[i * divided, (i + 1) * divided] for i in range(parallel_num)]
    frame_range[-1][1] = video_length - 1

    start, end = frame_range[rank]
    bar = tqdm(total=end - start, desc="Process#{}".format(rank))

    cmap = plt.get_cmap("rainbow")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_path, fourcc, get_video_framerate(video_path),
                                   get_video_dimension(video_path))

    # Read emotion recognition CSV
    emotion_df_list = [pd.read_csv(path) for path in emotion_csv_path_list]
    emotion_recognition_k_frame = emotion_df_list[0]["frame_number"][1] - emotion_df_list[0]["frame_number"][0]

    frame_index = start
    set_frame_position(video_capture, start)  # Move position
    while video_capture.isOpened() and get_frame_position(video_capture) in range(start, end):
        ret, frame = read_video_capture(video_capture, offset)
        # Progress
        bar.update(1)
        bar.refresh()

        for key in interpolated_result.keys():
            # Assume fill_blank = True
            face: Face = interpolated_result[key][frame_index]
            df = emotion_df_list[key]
            emotion = \
                df[df["frame_number"] == (frame_index - frame_index % emotion_recognition_k_frame)][
                    "prediction"].values[0]
            gesture = \
                df[df["frame_number"] == (frame_index - frame_index % emotion_recognition_k_frame)][
                    "gesture"].values[0]
            emo_x = df[df["frame_number"] == (frame_index - frame_index % emotion_recognition_k_frame)][
                "x"].values[0]
            emo_y = df[df["frame_number"] == (frame_index - frame_index % emotion_recognition_k_frame)][
                "y"].values[0]
            if face.location is not None:
                top, right, bottom, left = face.location
                # BBox
                p1 = (left, top)
                p2 = (right, bottom)
                if face.is_detected:
                    frame = drawrect(frame, p1, p2, color=np.array(cmap(key / len(interpolated_result))) * 255,
                                     thickness=8, style="line", label=emotion)
                else:
                    frame = drawrect(frame, p1, p2, color=np.array(cmap(key / len(interpolated_result))) * 255,
                                     thickness=8, style="dotted", label=emotion)
                # Draw circle when gesture flag == 1
                if gesture:
                    mid = (np.array(p1) + np.array(p2)) // 2
                    frame = cv2.circle(frame, tuple(mid), radius=100, color=(0, 0, 255), thickness=5)
                # Draw (x,y) from emotion recognition (center of eyes?)
                frame = cv2.circle(frame, (emo_x, emo_y), radius=3, color=(255, 0, 255), thickness=-1)

        video_writer.write(frame)
        frame_index += 1

    video_capture.release()
    video_writer.release()


def output_video_emotion_multiprocess(interpolated_result: dict, emotion_csv_path_list: list, video_path: str,
                                      output_path: str, offset=0, parallel_num=32):
    print(f"\nOutput to {output_path}")
    print("Using", parallel_num, "Process(es)")
    process_list = []
    for i in range(parallel_num):
        kwargs = dict(interpolated_result=interpolated_result,
                      emotion_csv_path_list=emotion_csv_path_list,
                      video_path=video_path,
                      output_path=f"{output_path[:-4]}_{i:02}{output_path[-4:]}",
                      offset=offset,
                      parallel_num=parallel_num,
                      rank=i
                      )
        p = Process(target=output_video_emotion, kwargs=kwargs)
        process_list.append(p)
        p.start()

    for p in process_list:
        # p.start()
        p.join()

    # Concat
    concat_video([f"{output_path[:-4]}_{i:02}{output_path[-4:]}" for i in range(parallel_num)], output_path)


"""
Deprecated
"""
# def output_video_interpolated(interpolated_result: dict, video_path: str, output_path: str):
#     """
#     Output interpolated result from capture_face as video.
#     Legend: (line) = box of detected face, (dotted-line) = interpolated box of undetected face
#     :param video_path: Input video path
#     :param interpolated_result: Result from interpolate_result
#     :param output_path: Output video path
#     """
#
#     def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
#         dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
#         pts = []
#         for i in np.arange(0, dist, gap):
#             r = i / dist
#             x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
#             y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
#             p = (x, y)
#             pts.append(p)
#
#         if style == 'dotted':
#             for p in pts:
#                 cv2.circle(img, p, thickness, color, -1)
#         else:
#             s = pts[0]
#             e = pts[0]
#             i = 0
#             for p in pts:
#                 s = e
#                 e = p
#                 if i % 2 == 1:
#                     cv2.line(img, s, e, color, thickness)
#                 i += 1
#         return img
#
#     def drawpoly(img, pts, color, thickness=1, style='dotted', ):
#         s = pts[0]
#         e = pts[0]
#         pts.append(pts.pop(0))
#         for p in pts:
#             s = e
#             e = p
#             drawline(img, s, e, color, thickness, style)
#         return img
#
#     def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
#         pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
#         drawpoly(img, pts, color, thickness, style)
#         return img
#
#     video_capture = get_video_capture(video_path)
#     cmap = plt.get_cmap("rainbow")
#     fourcc = cv2.VideoWriter_fourcc(*"XVID")
#     video_writer = cv2.VideoWriter(output_path, fourcc, get_video_framerate(video_path),
#                                    get_video_dimension(video_path))
#
#     frame_index = 0
#     while video_capture.isOpened():
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         for key in interpolated_result.keys():
#             # Assume fill_blank = True
#             face: Face = interpolated_result[key][frame_index]
#             if face.location is not None:
#                 top, right, bottom, left = face.location
#                 # BBox
#                 p1 = (left, top)
#                 p2 = (right, bottom)
#                 if face.is_detected:
#                     frame = cv2.rectangle(frame, p1, p2, color=np.array(cmap(key / len(interpolated_result))) * 255,
#                                           thickness=10)
#                 else:
#                     frame = drawrect(frame, p1, p2, color=np.array(cmap(key / len(interpolated_result))) * 255,
#                                      thickness=10, style="dotted")
#
#         video_writer.write(frame)
#         frame_index += 1
#
#     video_capture.release()
#     video_writer.release()
