import os
import tempfile
from multiprocessing import Process

from gui.calibrate import adjust_offset, calibrate_video_ffmpeg
from gui.cropper import select_roi
from gui.dialogs import get_video_path, get_face_num
from utils.video_utils import crop_video


def create_reid(video_path: str = None, face_num: int = None):
    if not video_path:
        video_path = get_video_path()
    if not face_num:
        face_num = get_face_num()
    offset = adjust_offset(video_path)

    with tempfile.TemporaryDirectory() as temp_dirname:
        print("Temporary directory created at", temp_dirname)
        calibrate_path = os.path.join(temp_dirname, "calibrate.mp4")
        calibrate_video_ffmpeg(video_path, output_path=calibrate_path, offset=offset, end_time=10, remove_audio=True)

        output_dir = os.path.join(os.path.dirname(video_path), "reid")
        os.makedirs(output_dir, exist_ok=True)
        post_processes = []
        # TODO unknown Seg Fault, seems from get_roi
        for i in range(face_num):
            output_path = os.path.join(output_dir, f"reid{i}.mp4")
            roi = select_roi(video_path, offset)
            kwargs = dict(video_path=calibrate_path,
                          output_path=output_path,
                          roi=roi,
                          console_quite=False)
            post_processes.append(Process(target=crop_video, kwargs=kwargs))

        for p in post_processes:
            p.start()
            p.join()


if __name__ == "__main__":
    create_reid(face_num=5)
