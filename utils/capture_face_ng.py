import pickle
from multiprocessing import Process, Manager

import cv2
import numpy as np

from utils.face import Face
from utils.match_frame import FrameMatcher


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

    return round(original_x1), round(original_y1), round(original_x2), round(original_y2)


# TODO: Initial analysis to detect a box where participants' faces will be in there mostly, cut the computational cost
def detect_face(video_path, gpu_index=0, parallel_num=1, k_resolution=3, frame_skip=0, batch_size=8,
                return_dict=None):
    """
    Detect all the faces in video, using CNN and CUDA for high accuracy and performance
    Refer to the example of face_recognition below:
    https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py
    :param video_path: Path to video
    :param parallel_num: Number of parallel jobs (Multiprocessing on GPUs)
    :param gpu_index: Which GPU to use
    :param batch_size: Batch size
    :param k_resolution:
    :param frame_skip:
    :param return_dict:

    :return:
    """
    import dlib
    dlib.cuda.set_device(gpu_index)  # Call before import face_recognition
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
    # frame_range[-1][1] = video_length - 1

    # The frame range of this process
    start, end = frame_range[gpu_index]

    frame_list = []
    frame_numbers = []

    result = []

    bar = tqdm(total=end - start, desc="GPU#{}".format(gpu_index))

    set_frame_position(video_capture, start)  # Move position
    while video_capture.isOpened() and get_frame_position(video_capture) in range(start, end):
        current_pos = get_frame_position(video_capture)
        next_pos = current_pos + frame_skip + 1

        # Grab a single frame of video
        ret, frame = video_capture.read()
        if frame_skip != 0 and current_pos % frame_skip != 0:
            bar.update(1)
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
        if len(frame_list) == batch_size or next_pos >= end:
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

            bar.update(len(frame_list))
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

    # Debug
    # print(combined)

    return combined


def match_result(result_from_detect_face=None):
    if result_from_detect_face is None:
        with open("detect_face.pt", "rb") as f:
            result_from_detect_face = pickle.load(f)
            print(result_from_detect_face)

    # Calculate max_face_num
    max_face_num = 0
    for result in result_from_detect_face:
        max_face_num = max(len(result), max_face_num)

    matcher = FrameMatcher(max_face_num)
    for r1, r2 in zip(result_from_detect_face, result_from_detect_face[1:]):
        matcher.match(r1, r2)

    # Debug
    # print(matcher.get_result())

    return matcher.get_result()


# # TODO
# def interpolate_result(result_from_match_result):
#     for face_list in result_from_match_result:
#         for face in face_list:
#             if face.is_detected
#
#         loc = np.array(match_result_dict[face_index])
#         flag = np.array(match_result_flag[face_index])
#         loc_true = loc[flag]
#         print(loc_true, len(loc), len(loc_true))


# # TODO
# def recognize_emotion(video_path, k_resolution=3, batch_size=128):
#     df = pd.read_csv("detect_face.csv")
#     # Note convert string to tuple
#     # t = df["face1"][0]
#     # print(make_tuple(t))
#     # print(get_face_location_from_csv("detect_face.csv", 110000))
#     from keras.models import model_from_json
#     model_dir = '../model'
#     model_name = 'mini_XCEPTION'
#     model = model_from_json(open(join(model_dir, 'model_{}.json'.format(model_name)), 'r').read())
#     model.load_weights(join(model_dir, 'model_{}.h5'.format(model_name)))
#
#     # Open video file
#     video_capture = cv2.VideoCapture(video_path)
#     original_w, original_h = get_video_dimension(video_capture)
#     resize_rate = (1080 * k_resolution) / original_w  # Resize
#     w = int(original_w * resize_rate)
#     h = int(original_h * resize_rate)
#
#     frames = []
#     faces = []
#
#     frame_count = 0
#     while video_capture.isOpened():
#         # Grab a single frame of video
#         ret, frame = video_capture.read()
#
#         # Bail out when the video file ends
#         if not ret:
#             break
#
#         # Resize
#         # frame = cv2.resize(frame, (w, h))
#         face_location = get_face_location_from_csv(df, frame_count)
#         for top, right, bottom, left in face_location:
#             face = frame[top:bottom, left:right]
#             import matplotlib.pyplot as plt
#             plt.imshow(face)
#             plt.show()
#             face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (64, 64)) / 255.0
#             faces.append(face)
#
#         frame_count += 1
#         # faces.append()
#
#         # Save each frame of the video to a list
#         frames.append(frame)
#
#         # Every 128 frames (the default batch size), batch process the list of frames to find faces
#         if len(faces) > batch_size:
#             val = np.stack(faces)
#             predictions = model.predict(val)
#             print(predictions)
#             # Clear the frames array to start the next batch
#             faces = []


# # Prepare short ver. for testing
# from edit_video import trim_video
# trim_video("../datasets/Videos_new_200929/200221_expt12_video.mp4", ['00:00:00', '00:00:10'], "test.mp4")

# detect_face_result = detect_face_multiprocess("test.mp4")
# matched_face_locations = match_result(detect_face_result)
# print(matched_face_locations)

matched_face_locations = match_result()
print(matched_face_locations)
# interpolate_result(matched_face_locations)
