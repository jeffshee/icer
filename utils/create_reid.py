import os

from gui.dialogs import *
from utils.video_utils import *


def create_reid(video_path: str = None, face_num: int = None):
    if not video_path:
        video_path = get_video_path()
    if not face_num:
        face_num = get_face_num()
    # print(video_path, face_num)

    output_dir = os.path.join(os.path.dirname(video_path), "reid")
    os.makedirs(output_dir, exist_ok=True)
    post_processes = []
    for i in range(face_num):
        output_path = os.path.join(output_dir, f"reid{i}.mp4")
        roi = get_roi(video_path)
        post_processes.append(Process(target=crop_video, args=(video_path, output_path, roi), kwargs=dict(end_time=10)))

    for p in post_processes:
        p.start()
        p.join()


if __name__ == "__main__":
    create_reid()
