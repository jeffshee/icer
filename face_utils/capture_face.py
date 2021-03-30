import pickle
from multiprocessing import Manager
import time
from collections import defaultdict
import os

from utils.video_utils import *

config = {
    "debug": False
}


def get_timestamp():
    from datetime import datetime
    return datetime.today().strftime('%Y%m%d%H%M%S')


def calculate_original_box(top, right, bottom, left, resized_w, resized_h, original_w, original_h, roi=None):
    if roi is not None:
        pad_x, pad_y = roi[0], roi[1]
        return [pad_y + top, pad_x + right, pad_y + bottom, pad_x + left]
    else:
        alpha_top = top / (resized_h - top)
        alpha_bottom = bottom / (resized_h - bottom)

        alpha_right = right / (resized_w - right)
        alpha_left = left / (resized_w - left)

        original_top = alpha_top * original_h / (1 + alpha_top)
        original_bottom = alpha_bottom * original_h / (1 + alpha_bottom)

        original_right = alpha_right * original_w / (1 + alpha_right)
        original_left = alpha_left * original_w / (1 + alpha_left)
        return [round(original_top), round(original_right), round(original_bottom), round(original_left)]


def calculate_box_midpoint(top, right, bottom, left):
    midpoint = np.array([(bottom + top) / 2, (right + left) / 2])
    return midpoint


def detect_face(video_path: str, gpu_index=0, parallel_num=1, k_resolution=3, frame_skip=0, batch_size=8,
                drop_last=True, return_dict=None, roi=None):
    """
    Detect all the faces in video, using CNN and CUDA for high accuracy and performance
    Refer to the example of face_recognition below:
    https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py
    :param video_path: Path to video
    :param parallel_num: Number of parallel jobs (Multiprocessing on GPUs)
    :param gpu_index: Which GPU to use
    :param batch_size: Batch size
    :param drop_last: Drop last incomplete batch.
    Workaround for a dlib bug that raise CUDA OOM error, especially when the last incomplete batch size == 1,
    even there are plenty of RAM still available. Refer:
    https://forums.developer.nvidia.com/t/cudamalloc-out-of-memory-although-the-gpu-memory-is-enough/84327
    :param k_resolution: resize factor, None = original
    :param frame_skip:
    :param return_dict:
    :param roi: ROI where the face recognition is performed

    :return:
    """
    # NOTE: face_recognition MUST NOT imported before dlib, CUDA will fail to initialize otherwise
    import dlib
    dlib.cuda.set_device(gpu_index)
    import face_recognition
    from tqdm import tqdm

    from face_utils.encode_face import batch_face_encodings

    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    if roi is not None:
        original_w, original_h = roi[2], roi[3]
    else:
        original_w, original_h = get_video_dimension(video_path)

    # Resize
    if k_resolution is not None:
        resize_rate = (1080 * k_resolution) / original_w
    else:
        resize_rate = 1
    w = int(original_w * resize_rate)
    h = int(original_h * resize_rate)

    video_length = get_video_length(video_path)

    divided = video_length // parallel_num  # Frames per process
    frame_range = [[i * divided, (i + 1) * divided] for i in range(parallel_num)]
    # Set the rest of the frames to the last process
    # NOTE: Workaround, for some reason OpenCV will fail to read the last frame
    frame_range[-1][1] = video_length - 1

    # The frame range of this process
    start, end = frame_range[gpu_index]

    frame_list = []
    frame_numbers = []

    result = []

    bar = tqdm(total=end - start, desc="GPU#{}".format(gpu_index))

    set_frame_position(video_capture, start)  # Move position
    while video_capture.isOpened() and get_frame_position(video_capture) in range(start, end):
        # Progress
        bar.update(1)
        bar.refresh()

        current_pos = get_frame_position(video_capture)
        next_pos = current_pos + frame_skip + 1

        # Grab a single frame of video
        ret, frame = video_capture.read()
        if frame_skip != 0 and current_pos % frame_skip != 0:
            continue

        # Crop
        if roi is not None:
            frame = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

        # Resize
        if k_resolution is not None:
            frame = cv2.resize(frame, (w, h))

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = frame[:, :, ::-1]

        # Save each frame of the video to a list
        frame_list.append(frame)
        frame_numbers.append(current_pos)

        # Every batch_size frames, batch process the list of frames to find faces
        if len(frame_list) == batch_size or (next_pos >= end and not drop_last):
            batch_of_face_locations = face_recognition.batch_face_locations(frame_list,
                                                                            number_of_times_to_upsample=0,
                                                                            batch_size=len(frame_list))
            batch_of_face_encodings = batch_face_encodings(frame_list, batch_of_face_locations)

            for frame_number_in_batch, (face_locations, face_encodings) in enumerate(
                    zip(batch_of_face_locations, batch_of_face_encodings)):
                temp = []
                for location, encoding in zip(face_locations, face_encodings):
                    temp.append(Face(frame_numbers[frame_number_in_batch],
                                     calculate_original_box(*location, w, h, original_w, original_h, roi),
                                     np.array(encoding)))
                if len(temp) != 0:
                    # Only append if there is any face detected
                    result.append(temp)

            # Clear the frames array to start the next batch
            frame_list = []
            frame_numbers = []

    if return_dict is not None:
        return_dict[gpu_index] = result


# TODO: Have a possibility causing CUDA OOM, need optimization. (Implemented drop_last as workaround)
# Consider https://github.com/1adrianb/face-alignment or https://github.com/jacobgil/dlib_facedetector_pytorch
def detect_face_multiprocess(video_path: str, parallel_num=3, k_resolution=3, frame_skip=3, batch_size=8,
                             roi=None, output_dir=".") -> list:
    # Note: Batch size = 8 is about the limitation of current machine
    print(f"\nProcessing {video_path}")
    print("Using", parallel_num, "GPU(s)")
    process_list = []
    manager = Manager()
    return_dict = manager.dict()

    for i in range(parallel_num):
        if roi is None:
            kwargs = {"video_path": video_path,
                      "gpu_index": i,
                      "parallel_num": parallel_num,
                      "k_resolution": k_resolution,
                      "frame_skip": frame_skip,
                      "batch_size": batch_size,
                      "return_dict": return_dict}
        else:
            four_k_size = 3840 * 2160
            roi_size = int(roi[2] * roi[3])
            # Heuristic for current GPU setting, which 1 GPU can process 3K res * 8 images in 1 batch at the most
            # Calculate the size ratio of the 4K and the ROI, then adjust the heuristic such that it doesn't cause OOM
            heuristic = 0.65
            ratio = four_k_size / roi_size * heuristic
            batch_size = int(batch_size * ratio)
            if batch_size < 8:
                # ROI is too large, fallback
                kwargs = {"video_path": video_path,
                          "gpu_index": i,
                          "parallel_num": parallel_num,
                          "k_resolution": k_resolution,
                          "frame_skip": frame_skip,
                          "batch_size": batch_size,
                          "return_dict": return_dict}
            else:
                # Proceed with the heuristic
                kwargs = {"video_path": video_path,
                          "gpu_index": i,
                          "parallel_num": parallel_num,
                          "k_resolution": None,
                          "frame_skip": frame_skip,
                          "batch_size": int(batch_size * ratio),
                          "return_dict": return_dict,
                          "roi": roi}
            if config["debug"]:
                print(kwargs)
        p = Process(target=detect_face, kwargs=kwargs)
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    # Combine result from multiprocessing
    combined = []
    for i in range(parallel_num):
        combined.extend(return_dict[i])

    # Save result into pickle
    if config["debug"]:
        with open(os.path.join(output_dir, f"{get_timestamp()}_detect_face.pt"), "wb") as f:
            pickle.dump(combined, f)

    return combined


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


def match_result(result_from_detect_face: list, method="cluster_face", **kwargs) -> defaultdict:
    print(f"\nMatching result")
    if method == "cluster_face":
        from face_utils.matching import cluster_face
        return cluster_face(result_from_detect_face, **kwargs)
    elif method == "match_frame":
        from face_utils.matching import match_frame
        return match_frame(result_from_detect_face, **kwargs)
    elif method == "reidentification":
        # Detect face for all single person face video
        assert kwargs["face_video_list"] is not None
        face_video_list = kwargs["face_video_list"]
        face_video_result = []
        for face_video in face_video_list:
            temp = detect_face_multiprocess(face_video, k_resolution=1, batch_size=32)
            face_video_result.append([t[0] for t in temp])
        from face_utils.matching import reidentification
        return reidentification(result_from_detect_face, face_video_result)
    else:
        raise ValueError("Unknown method")


def interpolate_result(result_from_match_result: defaultdict, video_path: str, box_th=0.1, output_dir="."):
    """
    :param result_from_match_result: Result from match_result
    :param video_path: Input video path
    :param box_th: ignore boxes that too far from median
    :return:
    """
    print(f"\nInterpolating result")
    import warnings
    # Ignore weird RuntimeWarning when importing SciPy
    warnings.simplefilter('ignore', RuntimeWarning)
    from scipy.interpolate import interp1d
    warnings.resetwarnings()

    # Miss detected face
    ghosts = []
    original_w, original_h = get_video_dimension(video_path)
    total_frame = get_video_length(video_path)

    for key, face_list in result_from_match_result.items():
        # Interpolate (top,right,bottom,left) for undetected frames
        # Set to None when interpolation is impossible, e.g. before the first or after the last detected frame
        face_list_detected = [face for face in face_list if face.is_detected and face.location is not None]
        face_location_median = np.median([calculate_box_midpoint(*face.location) for face in face_list_detected],
                                         axis=0)
        face_list_filtered = []
        for face in face_list_detected:
            distance_to_median = np.linalg.norm(calculate_box_midpoint(*face.location) - face_location_median)
            if distance_to_median < box_th * original_w:
                face_list_filtered.append(face)
            # else:
            #     print("filtered", face)

        time = np.array([face.frame_number for face in face_list_filtered])
        value_top = np.array([face.location[0] for face in face_list_filtered])
        value_right = np.array([face.location[1] for face in face_list_filtered])
        value_bottom = np.array([face.location[2] for face in face_list_filtered])
        value_left = np.array([face.location[3] for face in face_list_filtered])

        if time.min() == time.max():
            ghosts.append(key)
            continue

        f_top = interp1d(time, value_top)
        f_right = interp1d(time, value_right)
        f_bottom = interp1d(time, value_bottom)
        f_left = interp1d(time, value_left)

        # Interpolation
        interpolated_result = []
        cur_ptr = 0
        for f in range(total_frame):
            if cur_ptr < len(face_list_filtered) and f == face_list_filtered[cur_ptr].frame_number:
                interpolated_result.append(face_list_filtered[cur_ptr])
                cur_ptr += 1
            else:
                fill_face = Face(f, None, None, False)
                if time.min() < f < time.max():
                    fill_face.location = [int(f_top(f)), int(f_right(f)), int(f_bottom(f)), int(f_left(f))]
                interpolated_result.append(fill_face)

        # Swap the result with the interpolated one
        result_from_match_result[key] = interpolated_result

    # Remove ghost from result
    for ghost in ghosts:
        result_from_match_result.pop(ghost)

    # Dump
    with open(os.path.join(output_dir, f"{get_timestamp()}_interpolated_face.pt"), "wb") as f:
        pickle.dump(result_from_match_result, f)

    return result_from_match_result


# def main(video_path: str, output_dir: str, face_num=None, face_video_list=None):
#     """
#     Main routine
#     NOTE, format of interpolated result:
#     Dict{
#         0: [<Face>, <Face>, ...], # Result of person 0
#         1: [<Face>, <Face>, ...], # Result of person 1
#         2: [<Face>, <Face>, ...], # Result of person 2
#         ...
#     }
#     """
#     # Prepare output_dir
#     os.makedirs(output_dir, exist_ok=True)
#     # Capture Face
#     roi = get_roi(video_path)
#     start = time.time()
#     result = interpolate_result(
#         match_result(detect_face_multiprocess(video_path, roi=roi), method="reidentification",
#                      face_video_list=face_video_list), video_path=video_path)
#     # Emotion Recognition
#     from face_utils.emotion_recognition import emotion_recognition_multiprocess
#     emotion_recognition_multiprocess(result, video_path, output_dir)
#     # Output Video
#     output_video_emotion_multiprocess(result, [os.path.join(output_dir, f"output_emo/result{i}.csv") for i in range(6)],
#                                       video_path,
#                                       output_path=os.path.join(output_dir, f"{os.path.basename(video_path)[:-4]}_emotion.avi"))
#     print('capture_face elapsed time:', time.time() - start, '[sec]')

def main(video_path: str, output_dir: str, face_num=None, face_video_list=None):
    """
    Main routine
    :return: interpolated result
    Dict{
        0: [<Face>, <Face>, ...], # Result of person 0
        1: [<Face>, <Face>, ...], # Result of person 1
        2: [<Face>, <Face>, ...], # Result of person 2
        ...
    }
    """
    roi = get_roi(video_path)
    start = time.time()
    result = detect_face_multiprocess(video_path, roi=roi, output_dir=output_dir)
    if face_video_list is not None:
        result = match_result(result, method="reidentification", face_video_list=face_video_list, output_dir=output_dir)
    else:
        face_cluster_dir = os.path.join(output_dir, "face_cluster")
        os.makedirs(face_cluster_dir)
        result = match_result(result, method="cluster_face", face_num=face_num, video_path=video_path,
                              output_dir=face_cluster_dir)
    result = interpolate_result(result, video_path=video_path)
    print('capture_face elapsed time:', time.time() - start, '[sec]')
    return result

# def test():
#     # Prepare short ver. for testing
#     # from edit_video import trim_video
#     # trim_video("../datasets/Videos_new_200929/200221_expt12_video.mp4", ['00:00:00', '00:00:16'], "test.mp4")
#     pass
#
#
# if __name__ == "__main__":
#     main("../datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4", "../output",
#          face_video_list=[f"reid_test/untitled{i}.mp4" for i in range(1, 7)])
