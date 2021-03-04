import pickle
from multiprocessing import Process, Manager
import time
from collections import defaultdict

from utils.video_utils import *

config = {
    "debug": False
}


def get_timestamp():
    from datetime import datetime
    return datetime.today().strftime('%Y%m%d%H%M%S')


def calculate_original_box(x1, y1, x2, y2, resized_w, resized_h, original_w, original_h, roi=None):
    alpha_x1 = x1 / (resized_w - x1)
    alpha_x2 = x2 / (resized_w - x2)

    alpha_y1 = y1 / (resized_h - y1)
    alpha_y2 = y2 / (resized_h - y2)

    original_x1 = alpha_x1 * original_w / (1 + alpha_x1)
    original_x2 = alpha_x2 * original_w / (1 + alpha_x2)

    original_y1 = alpha_y1 * original_h / (1 + alpha_y1)
    original_y2 = alpha_y2 * original_h / (1 + alpha_y2)

    if roi is not None:
        pad_x, pad_y = roi[0], roi[1]
        return [pad_x + round(original_x1), pad_y + round(original_y1), pad_x + round(original_x2),
                pad_y + round(original_y2)]
    else:
        return [round(original_x1), round(original_y1), round(original_x2), round(original_y2)]


def calculate_box_midpoint(top, right, bottom, left):
    midpoint = np.array([(bottom + top) / 2, (right + left) / 2])
    return midpoint


# TODO: Initial analysis to detect a box where participants' faces will be in there mostly, cut the computational cost
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

    :return:
    """
    # NOTE: face_recognition MUST NOT imported before dlib, CUDA will fail to initialize otherwise
    import dlib
    dlib.cuda.set_device(gpu_index)
    import face_recognition
    from tqdm import tqdm

    from utils.encode_face import batch_face_encodings

    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    if roi is not None:
        original_w, original_h = roi[2], roi[3]
    else:
        original_w, original_h = get_video_dimension(video_path)

    # Resize
    if k_resolution is not None:
        resize_rate = (1080 * k_resolution) / original_w
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
                                     calculate_original_box(*location, w, h, original_w, original_h),
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
                             roi=None) -> list:
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
            ratio = four_k_size // roi_size
            kwargs = {"video_path": video_path,
                      "gpu_index": i,
                      "parallel_num": parallel_num,
                      "k_resolution": None,
                      "frame_skip": frame_skip,
                      "batch_size": batch_size * ratio,
                      "return_dict": return_dict}
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
    with open(f"{get_timestamp()}_detect_face.pt", "wb") as f:
        pickle.dump(combined, f)

    return combined


def get_roi(video_path: str):
    video_capture = get_video_capture(video_path)
    ret, frame = video_capture.read()
    roi = cv2.selectROI(frame)
    print("ROI:", roi)
    return roi


def match_result(result_from_detect_face: list, method="cluster_face", **kwargs) -> defaultdict:
    if method == "cluster_face":
        from utils.clustering import cluster_face
        return cluster_face(result_from_detect_face, **kwargs)
    elif method == "match_frame":
        from utils.matching import match_frame
        return match_frame(result_from_detect_face, **kwargs)
    elif method == "reidentification":
        # Detect face for all single person face video
        assert kwargs["face_video_list"] is not None
        face_video_list = kwargs["face_video_list"]
        face_video_result = []
        for face_video in face_video_list:
            temp = detect_face_multiprocess(face_video, k_resolution=1, batch_size=32)
            face_video_result.append([t[0] for t in temp])
        from utils.clustering import reidentification
        return reidentification(result_from_detect_face, face_video_result)
    else:
        raise ValueError("Unknown method")


def interpolate_result(result_from_match_result: defaultdict, video_path: str, box_th=0.1):
    """
    :param result_from_match_result: Result from match_result
    :param video_path: Input video path
    :param box_th: ignore boxes that too far from median
    :return:
    """
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
    with open(f"{get_timestamp()}_interpolated_face.pt", "wb") as f:
        pickle.dump(result_from_match_result, f)

    return result_from_match_result


def main(video_path, face_num):
    """
    Main routine, do detect_face on Multi-GPU,
    then perform frame-face matching or face clustering,
    finally interpolate the result for undetected face.
    :return: The final interpolated result.
    Dict{
        0: [<Face>, <Face>, ...], # Result of person 0
        1: [<Face>, <Face>, ...], # Result of person 1
        2: [<Face>, <Face>, ...], # Result of person 2
        ...
    }
    """
    start = time.time()
    roi = get_roi(video_path)
    result = interpolate_result(
        match_result(detect_face_multiprocess(video_path, roi=roi), video_path=video_path, face_num=face_num),
        video_path=video_path)
    print('capture_face_ng elapsed time:', time.time() - start, '[sec]')
    return result


def test():
    # Prepare short ver. for testing
    # from edit_video import trim_video
    # trim_video("../datasets/Videos_new_200929/200221_expt12_video.mp4", ['00:00:00', '00:00:16'], "test.mp4")

    # Test full run
    # print(main("test.mp4"))
    # print(main("../datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4"))

    # video_path = "../datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4"
    # with open("detect_face_long.pt", "rb") as f:
    #     result_from_detect_face = pickle.load(f)
    # result = interpolate_result(
    #     match_result(result_from_detect_face, video_path=video_path, face_num=6, use_old=True), video_path=video_path)
    # output_video(result, video_path, output_path="test_out_long_cluster_v2.avi")

    video_path = "../datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4"
    with open("detect_face_long.pt", "rb") as f:
        result_from_detect_face = pickle.load(f)
    result = interpolate_result(
        match_result(result_from_detect_face, method="reidentification",
                     face_video_list=[f"reid_test/untitled{i}.mp4" for i in range(1, 7)]), video_path=video_path)
    output_video(result, video_path, output_path="test_out_long_cluster_re_id_v2.avi")


if __name__ == "__main__":
    main("../datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4", 6)
    # test()
