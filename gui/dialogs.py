from PyQt5.QtWidgets import QWidget, QInputDialog, QFileDialog


def get_video_path():
    video_path = ""
    while not video_path:
        video_path = QFileDialog.getOpenFileName(caption="動画を指定してください", filter="Videos (*.mp4 *.avi)")[0]
    return video_path


def get_face_num():
    i, ret = None, False
    while not ret:
        i, ret = QInputDialog.getInt(QWidget(), "REID作成", "人数を入力してください", min=1, max=10)
    return i
