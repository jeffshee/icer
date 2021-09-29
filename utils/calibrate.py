import subprocess

from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QSlider

import cv2
import numpy as np
from gui.dialogs import *


def cv2_to_qpixmap(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytes_per_line = 3 * width
    return QPixmap(QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888))


class OffsetGUI(QMainWindow):
    def __init__(self, img: np.ndarray):
        super().__init__()
        self.img = img
        self.img_scale = 0.8
        h, w, c = img.shape

        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setScaledContents(True)
        self.label.setPixmap(
            cv2_to_qpixmap(img).scaled(int(1920 * self.img_scale), int(1080 * self.img_scale), Qt.KeepAspectRatio))
        layout.addWidget(self.label)

        layout.addWidget(QLabel(text="オフセット [終了するには、ENTERキーもしくはSPACEキーを押してください]"))
        slider = QSlider(Qt.Horizontal)
        slider.setMaximum(w)
        slider.setMinimum(-w)
        slider.valueChanged.connect(self.on_slider_value_changed)
        layout.addWidget(slider)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowTitle("360度動画の角度調整")

        self.offset = 0

    def on_slider_value_changed(self, val):
        self.offset = val
        img = img_offset(self.img, val)
        self.label.setPixmap(
            cv2_to_qpixmap(img).scaled(int(1920 * self.img_scale), int(1080 * self.img_scale), Qt.KeepAspectRatio))

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            QApplication.quit()


def img_offset(img: np.ndarray, offset: int = 0):
    img = np.concatenate([img[:, offset:], img[:, :offset]], axis=1)
    return img


def adjust_offset(video_path: str = None):
    """
    Launch a GUI for adjusting the offset
    :param video_path: Video path (Optional)
    :return:
    """
    if not video_path:
        video_path = get_video_path()
    video_capture = cv2.VideoCapture(video_path)
    ret, frame = video_capture.read()
    app = QApplication(sys.argv)
    win = OffsetGUI(frame)
    win.show()
    app.exec_()
    offset = win.offset
    print(f"Offset: {offset}")
    return offset


def calibrate_video_ffmpeg(video_path: str = None, use_gpu=True):
    """
    Output calibrated video (by using ffmpeg)
    :param video_path: Video path (Optional)
    :param use_gpu:
    :return:
    """
    if not video_path:
        video_path = get_video_path()
    output_path = video_path[:-4] + "_calibrate" + video_path[-4:]
    offset = adjust_offset(video_path)
    flip = -1 if offset < 0 else 1
    if use_gpu:
        """GPU"""
        cmd = f"""
                ffmpeg -y -i {video_path} -filter_complex \
                "[0:v][0:v]overlay=-{offset}[bg]; \
                 [bg][0:v]overlay={flip}*W-{offset},format=yuv420p[out]" \
                -map "[out]" -map 0:a? -codec:v h264_nvenc -c:a copy {output_path}
                """
    else:
        """CPU"""
        cmd = f"""
                ffmpeg -i {video_path} -filter_complex \
                "[0:v][0:v]overlay=-{offset}[bg]; \
                 [bg][0:v]overlay={flip}*W-{offset},format=yuv420p[out]" \
                -map "[out]" -map 0:a? -codec:v libx264 -c:a copy {output_path}
                """

    subprocess.call(cmd, shell=True)


class VideoCaptureWithOffset(cv2.VideoCapture):
    """
    VideoCapture with offset
    """

    def __init__(self, offset=0, *args, **kwargs):
        self.offset = offset
        super(VideoCaptureWithOffset, self).__init__(*args, **kwargs)

    def read(self, image=None):
        retval, image = super().read(image)
        if retval:
            image = img_offset(image, self.offset)
        return retval, image


if __name__ == "__main__":
    # adjust_offset()
    calibrate_video_ffmpeg()
