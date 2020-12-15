import cv2
from multiprocessing import Process, Manager
import pandas as pd
import numpy as np
from os.path import join
from ast import literal_eval as make_tuple
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import shutil
from main_util import cleanup_directory
import os
import numpy as np
from utils.matcher import Matcher


# TODO
# return face landmarks

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


def calculate_box_midpoint(top, right, bottom, left):
    midpoint = np.array([(bottom + top) / 2, (right + left) / 2])
    return midpoint


def calculate_box_distance(box1, box2):
    return np.linalg.norm(calculate_box_midpoint(*box1) - calculate_box_midpoint(*box2))


# TODO
def interpolate_detect_face_result(detect_face_result):
    interpolated = []
    for i in range(len(detect_face_result) - 1):
        pass


def my_compare_faces(known_faces, face_encoding_to_check):
    distance_list = []
    if len(known_faces) == 0:
        return np.empty((0))

    for known_face_list in known_faces:
        distance_know_face_list = list(np.linalg.norm(known_face_list - face_encoding_to_check, axis=1))
        distance_list.append(min(distance_know_face_list))

    return distance_list


# TODO
def match_detect_face_result_easy(video_path):
    # Check only the mid-points
    video_capture = cv2.VideoCapture(video_path)

    with open("detect_face.pt", "rb") as f:
        result = pickle.load(f)

        box = [[]]
        p = np.array([0, 1, 2])
        for r1, r2 in zip(result, result[1:]):
            _, face_boxes1 = r1[0], r1[1:]
            _, face_boxes2 = r2[0], r2[1:]

            r1_csv = []
            for r1_label, box1 in enumerate(face_boxes1):
                distance_r1_to_r2 = []
                for box2 in face_boxes2:
                    distance_r1_to_r2.append(calculate_box_distance(box1, box2))

                distance_r1_to_r2 = np.array(distance_r1_to_r2)
                # get the sort arguments
                distance_argsort = distance_r1_to_r2.argsort()
                # create label point to boxes in r2
                label_list_r2 = np.array(range(len(face_boxes2)))
                # ranks of r1's preference
                ranks = label_list_r2[distance_argsort]
                # label_r1 is the label of current box in r1
                r1_csv.append(np.concatenate([[r1_label], ranks]).astype(str))

            r2_csv = []
            for r2_label, box2 in enumerate(face_boxes2):
                distance_r2_to_r1 = []
                for box1 in face_boxes1:
                    distance_r2_to_r1.append(calculate_box_distance(box2, box1))

                distance_r2_to_r1 = np.array(distance_r2_to_r1)
                # get the sort arguments
                distance_argsort = distance_r2_to_r1.argsort()
                # create label point to boxes in r2
                label_list_r1 = np.array(range(len(face_boxes1)))
                # ranks of r1's preference
                ranks = label_list_r1[distance_argsort]
                # label_r1 is the label of current box in r1
                r2_csv.append(np.concatenate([[r2_label], ranks]).astype(str))

            df_r1 = pd.DataFrame(r1_csv, columns=["id"] + [str(i + 1) for i in range(len(face_boxes2))])
            df_r2 = pd.DataFrame(r2_csv, columns=["id"] + [str(i + 1) for i in range(len(face_boxes1))])
            print(df_r1)
            print("==========")
            print(df_r2)
            if len(face_boxes1) > len(face_boxes2):
                m = Matcher(df_r2, df_r1)
            else:
                m = Matcher(df_r1, df_r2)
            m.match()
            print("Matched pairs:", m.pairs)
            print(m.wives)
            map = [int(m.wives[str(i)]) for i in range(len(face_boxes1))]
            print(p[map])
            print("\n")


# TODO
# use face_distance, choose closest. If collide, check nearest mid point.
def match_detect_face_result(video_path, output_path="../face_cluster", face_matching_th=0.35):
    import face_recognition

    video_capture = cv2.VideoCapture(video_path)

    with open("detect_face.pt", "rb") as f:
        result = pickle.load(f)
        max_face_num = 0
        for r in result:
            max_face_num = max(max_face_num, len(r) - 1)

        for cluster_index in range(max_face_num):
            cleanup_directory(join(output_path, '{}'.format(cluster_index)))

        encoding_list = []
        img_filepath_list = []
        count = 0
        for r in result:
            frame_number, face_boxes = r[0], r[1:]
            # Skip bad frame
            if len(face_boxes) != max_face_num:
                continue
            set_frame_position(video_capture, frame_number)
            ret, frame = video_capture.read()
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            for top, right, bottom, left in face_boxes:
                img_filepath = join(output_path, "{}.png".format(count))
                cv2.imwrite(img_filepath, frame[top:bottom, left:right])
                img_filepath_list.append(img_filepath)
                count += 1

            frame = frame[:, :, ::-1]
            encoding_list.extend(face_recognition.face_encodings(frame, face_boxes))

        encoding_list = np.array(encoding_list)
        model = KMeans(n_clusters=max_face_num)
        model.fit(encoding_list)

        pred_labels = model.labels_

        closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, encoding_list)

        for i, (label, face_image_name) in enumerate(zip(pred_labels, img_filepath_list)):
            if i == closest[label]:
                shutil.move(face_image_name,
                            join(output_path, "{}", "{}.png").format(label, "closest"))
            else:
                shutil.move(face_image_name,
                            join(output_path, "{}", "{}.png").format(label, i))

        # for r in result:
        #     frame_number, face_boxes = r[0], r[1:]
        #     set_frame_position(video_capture, frame_number)
        #     ret, frame = video_capture.read()
        #     frame = frame[:, :, ::-1]
        #     encoding_frame = face_recognition.face_encodings(frame, face_boxes)
        #     pred = model.predict(encoding_frame)

        known_faces = []
        for label in range(max_face_num):
            temp = []
            for i, encoding in enumerate(encoding_list):
                if label == pred_labels[i]:
                    temp.append(encoding)
            known_faces.append(temp)

        # cluster_image_path_list = []
        # for cluster in os.listdir(output_path):
        #     cluster_root = join(output_path, cluster)
        #     img_list = os.listdir(cluster_root)
        #     cluster_image_path_list.append([join(cluster_root, img) for img in img_list])
        #
        # known_faces = []
        # for path_list in cluster_image_path_list:
        #     face_images = [face_recognition.load_image_file(path) for path in path_list]
        #     face_images_encoded = [
        #         face_recognition.face_encodings(x, known_face_locations=[(x.shape[1], 0, 0, x.shape[0])])[0] for x in
        #         face_images]
        #     known_faces.append(face_images_encoded)

        for r in result:
            frame_number, face_boxes = r[0], r[1:]
            set_frame_position(video_capture, frame_number)
            ret, frame = video_capture.read()
            frame = frame[:, :, ::-1]
            face_encoded_list = face_recognition.face_encodings(frame, known_face_locations=face_boxes)

            # 入力画像とのマッチング
            face_distance_list = [[] for _ in range(len(known_faces))]
            for face_encoded in face_encoded_list:
                tmp_distance = my_compare_faces(known_faces, face_encoded)
                for i in range(len(face_distance_list)):
                    face_distance_list[i].append(tmp_distance[i])
            print(face_distance_list)
            face_index_list = [None] * len(face_encoded_list)
            for i in range(len(known_faces)):
                if min(face_distance_list[i]) <= face_matching_th:
                    face_index = face_distance_list[i].index(min(face_distance_list[i]))
                    if face_index_list[face_index] is None:  # クエリ画像が検出顔のどれともマッチングしていないとき
                        face_index_list[face_index] = i
                    elif face_distance_list[face_index_list[face_index]].index(
                            min(face_distance_list[face_index_list[face_index]])) >= face_distance_list[i].index(
                        min(face_distance_list[i])):  # クエリ画像が検出顔のいずれかとマッチング済みであるとき
                        face_index_list[face_index] = i
            print(face_index_list)


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

        # Every 128 frames (the default batch size), batch process the list of frames to find faces
        if len(frame_list) == batch_size or next_pos >= end:
            batch_of_face_locations = face_recognition.batch_face_locations(frame_list,
                                                                            number_of_times_to_upsample=0,
                                                                            batch_size=len(frame_list))

            # Now let's list all the faces we found in all 128 frames
            for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
                face_locations = sorted(face_locations, key=lambda x: x[1])  # Sort the face with left value
                frame_number = frame_numbers[frame_number_in_batch]
                # Note: frame_number, (top, right, bottom, left), ...
                # Scale the box to original resolution
                face_locations = [calculate_original_box(top, right, bottom, left, w, h, original_w, original_h) for
                                  top, right, bottom, left in face_locations]
                result.append([frame_number, *face_locations])

            bar.update(len(frame_list))
            # Clear the frames array to start the next batch
            frame_list = []
            frame_numbers = []

    if return_dict is not None:
        return_dict[gpu_index] = result


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

    # Combine result
    result_list = []
    # Squeeze the list
    temp = sorted(return_dict.values(), key=lambda x: x[0])
    max_length = 0
    for i in temp:
        for j in i:
            max_length = max(max_length, len(j))
            result_list.append(j)

    # Save result into pickle
    with open("detect_face.pt", "wb") as f:
        pickle.dump(result_list, f)
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
        # frame = cv2.resize(frame, (w, h))
        face_location = get_face_location_from_csv(df, frame_count)
        for top, right, bottom, left in face_location:
            face = frame[top:bottom, left:right]
            import matplotlib.pyplot as plt
            plt.imshow(face)
            plt.show()
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
# recognize_emotion("test.mp4")
# match_detect_face_result("test.mp4")
# match_detect_face_result_easy("test.mp4")

def test():
    p = {"face0": [], "face1": [], "face2": []}
    t1 = ["b", "c"]
    t2 = ["A", "B", "C"]
    t1_int = ["0", "1"]
    t2_int = ["0", "1", "2"]
    m = {"0": "1", "1": "2"}
    # p' = []
