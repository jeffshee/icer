import subprocess

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QSlider

from gui.dialogs import *
from utils.video_utils import *


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

        slider = QSlider(Qt.Horizontal)
        slider.setMaximum(w)
        slider.setMinimum(-w)
        slider.valueChanged.connect(self.on_slider_value_changed)
        layout.addWidget(slider)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def on_slider_value_changed(self, val):
        print(val)
        img = img_offset(self.img, val)
        self.label.setPixmap(
            cv2_to_qpixmap(img).scaled(int(1920 * self.img_scale), int(1080 * self.img_scale), Qt.KeepAspectRatio))


def img_offset(img: np.ndarray, offset: int = 0):
    img = np.concatenate([img[:, offset:], img[:, :offset]], axis=1)
    return img


def adjust_offset(video_path: str = None):
    if not video_path:
        video_path = get_video_path()
    video_capture = get_video_capture(video_path)
    ret, frame = video_capture.read()
    app = QApplication(sys.argv)
    win = OffsetGUI(frame)
    win.show()
    app.exec_()


def calibrate_video(video_path: str = None):
    if not video_path:
        video_path = get_video_path()
    output_path = video_path[:-4] + "_calibrate" + video_path[-4:]
    pixel_offset = 50
    # cmd = f"""
    #     ffmpeg -i {video_path} -filter_complex \
    #     "[0:v][0:v]overlay={pixel_offset}:0[bg]; \
    #      [bg][0:v]overlay={pixel_offset}-W,format=yuv420p[out]" \
    #     -map "[out]" -map 0:a -codec:v libx264 -crf 23 -preset medium -c:a copy {output_path}
    #     """
    # cmd = f"""
    #         ffmpeg -i {video_path} -filter_complex \
    #         "[0:v][0:v]overlay={pixel_offset}:0[bg]; \
    #          [bg][0:v]overlay={pixel_offset}-W,format=yuv420p[out]" \
    #         -map "[out]" -map 0:a -codec:v libx264 -c:a copy {output_path}
    #         """
    cmd = f"""
                ffmpeg -y -i {video_path} -filter_complex \
                "[0:v][0:v]overlay={pixel_offset}:0[bg]; \
                 [bg][0:v]overlay={pixel_offset}-W,format=yuv420p[out]" \
                -map "[out]" -map 0:a -codec:v h264_nvenc -c:a copy {output_path}
                """
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    # calibrate_video()
    # img_offset()
    adjust_offset("/home/jeffshee/Developer/icer/datasets/200225_expt22/video.mp4")
