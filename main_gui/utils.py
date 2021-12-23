from sys import platform
from typing import Tuple

import cv2
import numpy as np

"""
Video
"""


def get_frame_position(video_capture: cv2.VideoCapture) -> int:
    return int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) + 1


def set_frame_position(video_capture: cv2.VideoCapture, position: int) -> int:
    return int(video_capture.set(cv2.CAP_PROP_POS_FRAMES, position - 1))


def get_video_dimension(video_capture: cv2.VideoCapture) -> Tuple[int, int]:
    return int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


def get_video_length(video_capture: cv2.VideoCapture) -> int:
    return int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


def get_video_framerate(video_capture: cv2.VideoCapture) -> float:
    return float(video_capture.get(cv2.CAP_PROP_FPS))


def get_video_writer(output_path: str, framerate: float, dimension: Tuple[int, int]) -> cv2.VideoWriter:
    # FFmpeg: http://ffmpeg.org/doxygen/trunk/isom_8c-source.html
    # ImageJ: Uncompressed palettized 8-bit RGBA (1987535218) fourcc: rawv
    # RGBA
    # fourcc = cv2.VideoWriter_fourcc(*"RGBA")  # Working (kind of)
    # PNG
    # fourcc = cv2.VideoWriter_fourcc(*"png ")  # Working (best compatibility) (Doesn't work on Windows!) (x)
    # Uncompressed RGB
    # fourcc = cv2.VideoWriter_fourcc(*"raw ") # Didn't encode(x)
    # Uncompressed YUV422
    # fourcc = cv2.VideoWriter_fourcc(*"yuv2")  # Encoded but didn't play(x)
    #
    # fourcc = cv2.VideoWriter_fourcc(*"XVID") # Very lossy(x)
    # fourcc = cv2.VideoWriter_fourcc(*"IYUV")  # Working on Windows
    # fourcc = -1
    if platform == "linux" or platform == "linux2":
        fourcc = cv2.VideoWriter_fourcc(*"png ")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"IYUV")
    return cv2.VideoWriter(output_path, fourcc, framerate, dimension)


def img_offset(img: np.ndarray, offset: int = 0):
    img = np.concatenate([img[:, offset:], img[:, :offset]], axis=1)
    return img


def read_video_capture(video_capture: cv2.VideoCapture, offset=0):
    retval, image = video_capture.read()
    if retval:
        image = img_offset(image, offset)
    return retval, image


"""
File
"""


def filename_append(filename: str, append: str, ext_override: str = None):
    basename, ext = filename.rsplit(".", 1)
    if ext_override is not None:
        ext = ext_override
    return f"{basename}-{append}.{ext}"
