import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QSlider, QApplication, QDialog, \
    QDialogButtonBox

from gui.dialogs import get_video_path
from utils.video_utils import img_offset


def cv2_to_qpixmap(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytes_per_line = 3 * width
    qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    qpixmap = QPixmap.fromImage(qimage)
    return qpixmap


class OffsetDialog(QDialog):
    def __init__(self, img: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("360度動画の角度調整")

        self.img = img
        self.img_scale = 0.8
        h, w, c = img.shape
        self.offset = 0

        layout = QVBoxLayout()
        self.view = QLabel()
        self.view.setScaledContents(True)
        self.view.setPixmap(
            cv2_to_qpixmap(img).scaled(int(1920 * self.img_scale), int(1080 * self.img_scale), Qt.KeepAspectRatio))
        layout.addWidget(self.view)

        self.label = QLabel(text=f"オフセット [現在値:{self.offset}]")
        layout.addWidget(self.label)

        slider = QSlider(Qt.Horizontal)
        slider.setMaximum(w)
        slider.setMinimum(-w)
        slider.valueChanged.connect(self.on_slider_value_changed)
        layout.addWidget(slider)

        # Buttons
        btn = QDialogButtonBox(QDialogButtonBox.Ok)
        btn.accepted.connect(self.accept)
        layout.addWidget(btn)

        self.setLayout(layout)

    def on_slider_value_changed(self, val):
        self.offset = val
        self.label.setText(f"オフセット [現在値:{self.offset}]")
        img = img_offset(self.img, val)
        self.view.setPixmap(
            cv2_to_qpixmap(img).scaled(int(1920 * self.img_scale), int(1080 * self.img_scale), Qt.KeepAspectRatio))

    def get_result(self):
        return self.offset


def adjust_offset_dialog(video_path: str = None):
    if not video_path:
        video_path = get_video_path()
    video_capture = cv2.VideoCapture(video_path)
    _, frame = video_capture.read()
    dlg = OffsetDialog(frame)
    if dlg.exec_():
        offset = dlg.get_result()
        print(f"Offset: {offset}")
        return offset
    return 0


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    video_path = get_video_path()
    for i in range(5):
        print(adjust_offset_dialog(video_path))

"""
Deprecated
"""
# class OffsetGUI(QMainWindow):
#     def __init__(self, img: np.ndarray):
#         super().__init__()
#         self.img = img
#         self.img_scale = 0.8
#         h, w, c = img.shape
#
#         layout = QVBoxLayout()
#         self.view = QLabel()
#         self.view.setScaledContents(True)
#         self.view.setPixmap(
#             cv2_to_qpixmap(img).scaled(int(1920 * self.img_scale), int(1080 * self.img_scale), Qt.KeepAspectRatio))
#         layout.addWidget(self.view)
#
#         self.offset = 0
#         self.label = QLabel(text=f"オフセット [現在値:{self.offset}]")
#         layout.addWidget(self.label)
#
#         slider = QSlider(Qt.Horizontal)
#         slider.setMaximum(w)
#         slider.setMinimum(-w)
#         slider.valueChanged.connect(self.on_slider_value_changed)
#         layout.addWidget(slider)
#
#         widget = QWidget(self)
#         widget.setLayout(layout)
#         self.setCentralWidget(widget)
#         self.setWindowTitle("360度動画の角度調整")
#
#         # Buttons
#         btn_ok = QPushButton("確定")
#         # Default button
#         btn_ok.setDefault(True)
#         btn_ok.setAutoDefault(True)
#         # btn_ok.clicked.connect(lambda _: QApplication.quit())
#         btn_ok.clicked.connect(self.on_click_ok)
#         layout.addWidget(btn_ok)
#
#     def on_click_ok(self):
#         self.hide()
#
#     def on_slider_value_changed(self, val):
#         self.offset = val
#         self.label.setText(f"オフセット [現在値:{self.offset}]")
#         img = img_offset(self.img, val)
#         self.view.setPixmap(
#             cv2_to_qpixmap(img).scaled(int(1920 * self.img_scale), int(1080 * self.img_scale), Qt.KeepAspectRatio))
#
#     def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
#         if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
#             self.hide()
#             # QApplication.quit()

# def adjust_offset(video_path: str = None):
#     """
#     Launch a GUI for adjusting the offset
#     :param video_path: Video path (Optional)
#     :return:
#     """
#     if not video_path:
#         video_path = get_video_path()
#     video_capture = cv2.VideoCapture(video_path)
#     _, frame = video_capture.read()
#     # app = QApplication.instance()
#     # app = QApplication(sys.argv)
#     win = OffsetGUI(frame)
#     win.show()
#     app.exec_()
#     # del app
#     offset = copy.copy(win.offset)
#     print(f"Offset: {offset}")
#     return offset

# class VideoCaptureWithOffset(cv2.VideoCapture):
#     """
#     VideoCapture with offset
#     @bug this cause segfault!
#     """
#
#     def __init__(self, offset=0, *args, **kwargs):
#         self.offset = offset
#         super(VideoCaptureWithOffset, self).__init__(*args, **kwargs)
#
#     def read(self, image=None):
#         retval, image = super().read(image)
#         if retval:
#             image = img_offset(image, self.offset)
#         return retval, image
