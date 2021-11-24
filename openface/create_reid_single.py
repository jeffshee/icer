import os
import sys
from multiprocessing import Process
import joblib

from PyQt5.QtWidgets import QApplication

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gui.calibrate import adjust_offset_dialog
from gui.cropper import select_roi_dialog
from gui.dialogs import get_video_path, get_face_num
from utils.video_utils import crop_video, calibrate_video


def create_reid(video_path: str = None, face_num: int = None):
    _ = QApplication(sys.argv)
    output_dir = os.path.join(os.path.dirname(video_path), "reid")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "reid_log.txt")
    if not video_path:
        video_path = get_video_path()
    offset = adjust_offset_dialog(video_path)
    with open(log_path, mode='a') as f:
        f.write(f"\noffset: {offset}")

    if not face_num:
        face_num = get_face_num()

    calibrate_path = os.path.join(output_dir, "calibrate.mp4")
    calibrate_video(video_path, output_path=calibrate_path, offset=offset, end_time=-1, remove_audio=True)

    post_processes = []
    rois = []
    for i in range(face_num):
        output_path = os.path.join(output_dir, f"reid{i}.mp4")
        roi = select_roi_dialog(video_path, offset, window_title=f"顔の領域の指定({i + 1}人目)", cancelable=False)
        rois.append(roi)
        print(f"[{i}] ROI: {roi}")
        with open(log_path, mode='a') as f:
            f.write(f"\n[{i}] ROI: {roi}")
        kwargs = dict(video_path=calibrate_path, output_path=output_path, roi=roi, console_quiet=False)
        post_processes.append(Process(target=crop_video, kwargs=kwargs))

    joblib.dump(rois, os.path.join(output_dir, "rois.txt"))

    for p in post_processes:
        p.start()
    # for p in post_processes:  # single-gpuの時はコメントアウト
        p.join()

    QApplication.quit()


if __name__ == "__main__":
    video_path = "/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_0/output.avi"
    face_num = 5
    create_reid(video_path, face_num)
