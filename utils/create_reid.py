from utils.video_utils import *

if __name__ == "__main__":
    face_num = 6
    video_path = "../datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4"
    roi_list = []
    for i in range(face_num):
        roi_list.append(get_roi(video_path))
    for i, roi in enumerate(roi_list):
        output_path = f"{i}.mp4"
        crop_video(video_path, output_path, roi, end_time=10)
