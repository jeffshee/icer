import cv2
from multiprocessing import Process, Manager
import pandas as pd
import numpy as np
from os.path import join
from ast import literal_eval as make_tuple


def get_frame_position(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))


def set_frame_position(video_capture, position):
    return int(video_capture.set(cv2.CAP_PROP_POS_FRAMES, position))


def get_video_dimension(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


def get_video_length(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


# TODO: frame_skip, Calculate the bounding box on original resolution, last batch is ditched (bug)
def detect_face(video_path, gpu_index=0, parallel_num=1, k_resolution=3, batch_size=8, return_dict=None):
    """
    Detect all the faces in video, using CNN and CUDA for high accuracy and performance
    Refer to the example of face_recognition below:
    https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py
    :param video_path: Path to video
    :param parallel_num: Number of parallel jobs (Multiprocessing on GPUs)
    :param gpu_index: Which GPU to use
    :param batch_size: Batch size
    :param k_resolution:
    :param return_dict:

    :return:
    """
    import dlib
    dlib.cuda.set_device(gpu_index)  # Call before import face_recognition
    import face_recognition
    from tqdm import tqdm

    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    original_w, original_h = get_video_dimension(video_capture)
    resize_rate = (1080 * k_resolution) / original_w  # Resize
    w = int(original_w * resize_rate)
    h = int(original_h * resize_rate)

    video_length = get_video_length(video_capture)
    divided = video_length // parallel_num  # Frames per process
    frame_range = [[i * divided, (i + 1) * divided] for i in range(parallel_num)]
    frame_range[-1][1] = video_length  # Set the rest of the frames to the last process

    # The frame range of this process
    start, end = frame_range[gpu_index]

    frames = []
    frame_count = start

    result = []

    bar = tqdm(total=end - start, desc="GPU#{}".format(gpu_index))

    set_frame_position(video_capture, start)  # Move position
    while video_capture.isOpened() and get_frame_position(video_capture) in range(start, end):
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Bail out when the video file ends
        if not ret:
            break

        # Resize
        frame = cv2.resize(frame, (w, h))

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = frame[:, :, ::-1]

        # Save each frame of the video to a list
        frame_count += 1
        frames.append(frame)

        # Every 128 frames (the default batch size), batch process the list of frames to find faces
        if len(frames) == batch_size:
            batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)

            # Now let's list all the faces we found in all 128 frames
            for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
                face_locations = sorted(face_locations, key=lambda x: x[1])  # Sort the face with left value
                frame_number = frame_count - batch_size + frame_number_in_batch
                # Note: frame_number, (top, right, bottom, left), ...
                result.append([frame_number, *face_locations])

            bar.update(len(frames))
            # Clear the frames array to start the next batch
            frames = []

    if return_dict is not None:
        return_dict[gpu_index] = result


def detect_face_multiprocess(video_path, parallel_num=3, k_resolution=3, batch_size=8):
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
                  "batch_size": batch_size,
                  "return_dict": return_dict}

        p = Process(target=detect_face, kwargs=kwargs)
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    # Combine result
    result_list = []
    # Squeeze the list
    temp = sorted(return_dict.values(), key=lambda x: x[0])
    max_length = 0
    for i in temp:
        for j in i:
            max_length = max(max_length, len(j))
            result_list.append(j)
    columns = ["frame"] + ["face{}".format(i) for i in range(max_length - 1)]
    df = pd.DataFrame(result_list, columns=columns)
    df.to_csv("detect_face.csv")


def get_face_location_from_csv(df, frame_num):
    face_location = df.loc[df["frame"] == frame_num].values.tolist()
    if len(face_location) == 0:
        # frame_num not found in csv
        return None
    return [make_tuple(s) for s in face_location[0][2:]]


# TODO
def recognize_emotion(video_path, k_resolution=3, batch_size=128):
    df = pd.read_csv("detect_face.csv")
    # Note convert string to tuple
    # t = df["face1"][0]
    # print(make_tuple(t))
    # print(get_face_location_from_csv("detect_face.csv", 110000))
    from keras.models import model_from_json
    model_dir = '../model'
    model_name = 'mini_XCEPTION'
    model = model_from_json(open(join(model_dir, 'model_{}.json'.format(model_name)), 'r').read())
    model.load_weights(join(model_dir, 'model_{}.h5'.format(model_name)))

    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    original_w, original_h = get_video_dimension(video_capture)
    resize_rate = (1080 * k_resolution) / original_w  # Resize
    w = int(original_w * resize_rate)
    h = int(original_h * resize_rate)

    frames = []
    faces = []

    frame_count = 0
    while video_capture.isOpened():
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Bail out when the video file ends
        if not ret:
            break

        # Resize
        frame = cv2.resize(frame, (w, h))
        face_location = get_face_location_from_csv(df, frame_count)
        for top, right, bottom, left in face_location:
            face = frame[top:bottom, left:right]
            face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (64, 64)) / 255.0
            faces.append(face)

        frame_count += 1
        # faces.append()

        # Save each frame of the video to a list
        frames.append(frame)

        # Every 128 frames (the default batch size), batch process the list of frames to find faces
        if len(faces) > batch_size:
            val = np.stack(faces)
            predictions = model.predict(val)
            print(predictions)
            # Clear the frames array to start the next batch
            faces = []


# # Prepare short ver. for testing
# from edit_video import trim_video
# trim_video("../datasets/Videos_new_200929/200221_expt12_video.mp4", ['00:00:00', '00:00:10'], "test.mp4")
# detect_face_multiprocess("test.mp4")
recognize_emotion("test.mp4")
