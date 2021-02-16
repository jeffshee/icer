import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from utils.face import Face


def get_video_capture(video_path: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(video_path)


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


def output_video(interpolated_result: dict, video_path: str, output_path: str):
    """
    Output interpolated result from capture_face as video.
    Legend: (line) = box of detected face, (dotted-line) = interpolated box of undetected face
    :param video_path: Input video path
    :param interpolated_result: Result from interpolate_result
    :param output_path: Output video path
    """

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

    def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
        pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
        drawpoly(img, pts, color, thickness, style)
        return img

    video_capture = get_video_capture(video_path)
    cmap = plt.get_cmap("rainbow")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_path, fourcc, get_video_framerate(video_path),
                                   get_video_dimension(video_path))

    frame_index = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        for key in interpolated_result.keys():
            # Assume fill_blank = True
            face: Face = interpolated_result[key][frame_index]
            if face.location is not None:
                top, right, bottom, left = face.location
                # BBox
                p1 = (left, top)
                p2 = (right, bottom)
                if face.is_detected:
                    frame = cv2.rectangle(frame, p1, p2, color=np.array(cmap(key / len(interpolated_result))) * 255,
                                          thickness=10)
                else:
                    frame = drawrect(frame, p1, p2, color=np.array(cmap(key / len(interpolated_result))) * 255,
                                     thickness=10, style="dotted")

        video_writer.write(frame)
        frame_index += 1

    video_capture.release()
    video_writer.release()
