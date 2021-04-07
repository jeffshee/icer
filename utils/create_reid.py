from utils.video_utils import get_video_capture, crop_video


def get_roi(video_path: str):
    video_capture = get_video_capture(video_path)
    ret, frame = video_capture.read()
    # Custom GUI
    from gui.qt_cropper import selectROI
    roi = selectROI(frame)
    x, y, w, h = int(roi.x()), int(roi.y()), int(roi.width()), int(roi.height())
    roi = (x, y, w, h)
    # OpenCV
    # roi = cv2.selectROI(frame)
    # if roi == (0, 0, 0, 0):
    #     roi = None
    print("ROI:", roi)
    return roi


if __name__ == "__main__":
    video_path = "../datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4"
    output_path = "06.mp4"
    roi = get_roi(video_path)
    crop_video(video_path, output_path, roi, end_time=10)
