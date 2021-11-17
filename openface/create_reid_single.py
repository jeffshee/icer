import os
import sys
import tempfile
from multiprocessing import Process

from PyQt5.QtWidgets import QApplication

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gui.calibrate import adjust_offset_dialog
from gui.cropper import select_roi_dialog
from gui.dialogs import get_video_path, get_face_num
from utils.video_utils import crop_video, calibrate_video


def create_reid(video_path: str = None, face_num: int = None):
    _ = QApplication(sys.argv)
    if not video_path:
        video_path = get_video_path()
    offset = adjust_offset_dialog(video_path)
    if not face_num:
        face_num = get_face_num()
    with tempfile.TemporaryDirectory() as temp_dirname:
        print("Temporary directory created at", temp_dirname)
        calibrate_path = os.path.join(temp_dirname, "calibrate.mp4")
        calibrate_video(video_path, output_path=calibrate_path, offset=offset, end_time=-1, remove_audio=True)

        output_dir = os.path.join(os.path.dirname(video_path), "reid")
        os.makedirs(output_dir, exist_ok=True)
        post_processes = []
        for i in range(face_num):
            output_path = os.path.join(output_dir, f"reid{i}.mp4")
            roi = select_roi_dialog(video_path, offset, window_title=f"顔の領域の指定({i + 1}人目)", cancelable=False)
            print(f"[{i}] ROI: {roi}")
            kwargs = dict(video_path=calibrate_path,
                          output_path=output_path,
                          roi=roi,
                          console_quiet=False)
            post_processes.append(Process(target=crop_video, kwargs=kwargs))

        for p in post_processes:
            p.start()
        # for p in post_processes:
            p.join()

    QApplication.quit()


if __name__ == "__main__":
    video_path = "/home/icer/Project/openface_dir2/2019-11-01_191031_Haga_3_3_4people/split_video_0/output.avi"
    face_num = 5
    create_reid(video_path, face_num)
