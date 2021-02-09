import pickle
from multiprocessing import Process, Manager

import cv2
import numpy as np

from utils.face import Face
from utils.match_frame import FrameMatcher
from utils.cluster_face_ng import cluster_face
import time


def get_frame_position(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))


def set_frame_position(video_capture, position):
    return int(video_capture.set(cv2.CAP_PROP_POS_FRAMES, position))


def get_video_dimension(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


def get_video_length(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


def calculate_original_box(x1, y1, x2, y2, resized_w, resized_h, original_w, original_h):
    alpha_x1 = x1 / (resized_w - x1)
    alpha_x2 = x2 / (resized_w - x2)

    alpha_y1 = y1 / (resized_h - y1)
    alpha_y2 = y2 / (resized_h - y2)

    original_x1 = alpha_x1 * original_w / (1 + alpha_x1)
    original_x2 = alpha_x2 * original_w / (1 + alpha_x2)

    original_y1 = alpha_y1 * original_h / (1 + alpha_y1)
    original_y2 = alpha_y2 * original_h / (1 + alpha_y2)

    return [round(original_x1), round(original_y1), round(original_x2), round(original_y2)]


# TODO: Initial analysis to detect a box where participants' faces will be in there mostly, cut the computational cost
# TODO: Set a distance threshold, if two boxes are too close, omit one of them (to avoid double detection)
def detect_face(video_path, gpu_index=0, parallel_num=1, k_resolution=3, frame_skip=0, batch_size=8, drop_last=True,
                return_dict=None):
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
    :param k_resolution:
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
    original_w, original_h = get_video_dimension(video_capture)
    resize_rate = (1080 * k_resolution) / original_w  # Resize
    w = int(original_w * resize_rate)
    h = int(original_h * resize_rate)

    video_length = get_video_length(video_capture)
    divided = video_length // parallel_num  # Frames per process
    frame_range = [[i * divided, (i + 1) * divided] for i in range(parallel_num)]
    # Set the rest of the frames to the last process
    # Note: This is actually a workaround, for some reason OpenCV will fail to read the last frame
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

        current_pos = get_frame_position(video_capture)
        next_pos = current_pos + frame_skip + 1

        # Grab a single frame of video
        ret, frame = video_capture.read()
        if frame_skip != 0 and current_pos % frame_skip != 0:
            continue

        # # Bail out when the video file ends
        # if not ret:
        #     break

        # Resize
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


# TODO
# Have a possibility causing CUDA OOM, need optimization
# Maybe there is a bug, OOM always happen when processing the end of the video
def detect_face_multiprocess(video_path, parallel_num=3, k_resolution=3, frame_skip=3, batch_size=8):
    # Note: Batch size = 8 is about the limitation of current machine
    print("Using", parallel_num, "GPU(s)")
    process_list = []
    manager = Manager()
    return_dict = manager.dict()

    for i in range(parallel_num):
        kwargs = {"video_path": video_path,
                  "gpu_index": i,
                  "parallel_num": parallel_num,
                  "k_resolution": k_resolution,
                  "frame_skip": frame_skip,
                  "batch_size": batch_size,
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
    with open("detect_face.pt", "wb") as f:
        pickle.dump(combined, f)

    return combined


def match_result(result_from_detect_face, face_num=None):
    # Calculate max_face_num
    if face_num is None:
        face_num = 0
        for result in result_from_detect_face:
            face_num = max(len(result), face_num)

    matcher = FrameMatcher(face_num)
    for r1, r2 in zip(result_from_detect_face, result_from_detect_face[1:]):
        matcher.match(r1, r2)

    return matcher.get_result()


# TODO
# Set a threshold, if the interpolation is larger than that, ignore
def interpolate_result(result_from_match_result, total_frame=None, fill_blank=True):
    import warnings
    # Ignore weird RuntimeWarning when importing SciPy
    warnings.simplefilter('ignore', RuntimeWarning)
    from scipy.interpolate import interp1d
    warnings.resetwarnings()

    # Miss detected face
    ghosts = []

    # Fill blank
    if fill_blank and total_frame is not None:
        for key, face_list in result_from_match_result.items():
            filled_list = []
            cur_ptr = 0
            for f in range(total_frame):
                if cur_ptr < len(face_list) and f == face_list[cur_ptr].frame_number:
                    filled_list.append(face_list[cur_ptr])
                    cur_ptr += 1
                else:
                    filled_list.append(Face(f, None, None, False))
            # Swap the result with the filled_list
            result_from_match_result[key] = filled_list

    for key, face_list in result_from_match_result.items():
        # Interpolate (top,right,bottom,left) for undetected frames
        # Set to None when interpolation is impossible, e.g. before the first or after the last detected frame
        time = np.array([face.frame_number for face in face_list if face.is_detected])
        value_top = np.array(
            [face.location[0] for face in face_list if face.is_detected and face.location is not None])
        value_right = np.array(
            [face.location[1] for face in face_list if face.is_detected and face.location is not None])
        value_bottom = np.array(
            [face.location[2] for face in face_list if face.is_detected and face.location is not None])
        value_left = np.array(
            [face.location[3] for face in face_list if face.is_detected and face.location is not None])

        if time.min() == time.max():
            ghosts.append(key)
            continue

        f_top = interp1d(time, value_top)
        f_right = interp1d(time, value_right)
        f_bottom = interp1d(time, value_bottom)
        f_left = interp1d(time, value_left)

        for face in face_list:
            if not face.is_detected:
                if time.min() < face.frame_number < time.max():
                    face.location = [int(f_top(face.frame_number)), int(f_right(face.frame_number)),
                                     int(f_bottom(face.frame_number)), int(f_left(face.frame_number))]
                else:
                    face.location = None

    # Remove ghost from result
    for ghost in ghosts:
        result_from_match_result.pop(ghost)

    return result_from_match_result


def output_video(interpolated_result: dict, input_path, output_path):
    def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            p = (x, y)
            pts.append(p)

        if style == 'dotted':
            for p in pts:
                cv2.circle(img, p, thickness, color, -1)
        else:
            s = pts[0]
            e = pts[0]
            i = 0
            for p in pts:
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(img, s, e, color, thickness)
                i += 1
        return img

    def drawpoly(img, pts, color, thickness=1, style='dotted', ):
        s = pts[0]
        e = pts[0]
        pts.append(pts.pop(0))
        for p in pts:
            s = e
            e = p
            drawline(img, s, e, color, thickness, style)
        return img

    def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
        pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
        drawpoly(img, pts, color, thickness, style)
        return img

    video_capture = cv2.VideoCapture(input_path)
    # TODO
    # get the parameter from input
    # colormap
    frame_rate = 30
    width, height = 3840, 2160
    colormap = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    frame_index = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        for key in interpolated_result.keys():
            # Assume fill_blank = True
            face: Face = interpolated_result[key][frame_index]
            if face.location is not None:
                top, right, bottom, left = face.location
                # BBox
                p1 = (left, top)
                p2 = (right, bottom)
                if face.is_detected:
                    frame = cv2.rectangle(frame, p1, p2, color=colormap[key], thickness=10)
                else:
                    frame = drawrect(frame, p1, p2, color=colormap[key], thickness=10, style="dotted")

        video_writer.write(frame)
        frame_index += 1

    video_capture.release()
    video_writer.release()


def main(video_path):
    """
    Main routine, do detect_face on Multi-GPU,
    then perform frame-face matching,
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
    total_frame = get_video_length(cv2.VideoCapture(video_path))
    result = interpolate_result(match_result(detect_face_multiprocess(video_path)), total_frame)
    print('capture_face_ng elapsed time:', time.time() - start, '[sec]')
    return result


def test():
    # Prepare short ver. for testing
    # from edit_video import trim_video
    # trim_video("../datasets/Videos_new_200929/200221_expt12_video.mp4", ['00:00:00', '00:00:16'], "test.mp4")

    # Test full run
    # print(main("test.mp4"))
    # print(main("../datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4"))

    # # Test using pickled result
    # with open("detect_face.pt", "rb") as f:
    #     # with open("detect_face_long.pt", "rb") as f:
    #     result_from_detect_face = pickle.load(f)
    # # total_frame = get_video_length(cv2.VideoCapture("../datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4"))
    # total_frame = get_video_length(cv2.VideoCapture("test.mp4"))
    # result = interpolate_result(match_result(result_from_detect_face),
    #                             total_frame)
    # print(result)
    # output_video(result, "test.mp4", "test_out.avi")
    # # output_video(result, "../datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4", "long_out.avi")

    # with open("detect_face.pt", "rb") as f:
    #     result_from_detect_face = pickle.load(f)
    # total_frame = get_video_length(cv2.VideoCapture("test.mp4"))
    # result = interpolate_result(cluster_face(result_from_detect_face, face_num=3, input_path="test.mp4"), total_frame)
    # print(result)
    # output_video(result, "test.mp4", "test_out_cluster.avi")

    input_path = "../datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4"
    with open("detect_face_long.pt", "rb") as f:
        result_from_detect_face = pickle.load(f)
    total_frame = get_video_length(cv2.VideoCapture(input_path))
    result = interpolate_result(cluster_face(result_from_detect_face, face_num=6, input_path=input_path), total_frame)
    print(result)
    output_video(result, input_path, "test_out_long_cluster.avi")


if __name__ == "__main__":
    test()
