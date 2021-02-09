import os
import shutil
from os.path import join
from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from main_util import cleanup_directory
from utils.face import Face


def get_frame_position(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))


def set_frame_position(video_capture, position):
    return int(video_capture.set(cv2.CAP_PROP_POS_FRAMES, position))


def output_face_image(video_capture: cv2.VideoCapture, face: Face, output_path):
    set_frame_position(video_capture, face.frame_number)
    ret, frame = video_capture.read()
    if ret and face.location is not None:
        top, right, bottom, left = face.location
        face_img = frame[top:bottom, left:right]
        cv2.imwrite(output_path, face_img)


def my_compare_faces(known_faces, face_encoding_to_check):
    distance_list = []
    if len(known_faces) == 0:
        return np.empty(0)

    for known_face_list in known_faces:
        distance_know_face_list = np.linalg.norm(known_face_list - face_encoding_to_check, axis=1)
        distance_list.append(np.min(distance_know_face_list))

    return distance_list


def cluster_face(result_from_detect_face, face_num=None, input_path=None, target_cluster_size=1000,
                 output_path="face_cluster", unattended=False):
    if not unattended and input_path is None:
        raise ValueError("not unattended and input_path is None")

    if face_num is None:
        face_num = 0
        for result in result_from_detect_face:
            face_num = max(len(result), face_num)

    video_capture = cv2.VideoCapture(input_path)

    for cluster_index in range(face_num):
        cleanup_directory(join(output_path, '{}'.format(cluster_index)))

    face_image_encoded_list = []
    face_image_path_list = []
    count = 0
    break_flag = False
    for i in range(face_num, 0, -1):
        for r in result_from_detect_face:
            if len(r) == i:
                for face in r:
                    if not unattended:
                        face_image_path = join(output_path, "{}.png".format(count))
                        output_face_image(video_capture, face, face_image_path)
                        face_image_path_list.append(face_image_path)
                    face_image_encoded_list.append(face.encoding)
                    count += 1
                    if count >= target_cluster_size:
                        break_flag = True
                        break
            if break_flag:
                break
        if break_flag:
            break

    # 画像を読み込みエンコード
    face_image_encoded_np = np.array(face_image_encoded_list)

    # 学習(cluster_num種類のグループにクラスタリングする)
    model = KMeans(n_clusters=face_num)
    model.fit(face_image_encoded_np)

    # 学習結果のラベル
    pred_labels = model.labels_
    print("ClusterFace bincount:", np.bincount(pred_labels))

    # Get nearest point to centroid
    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, face_image_encoded_np)

    known_faces = []
    if not unattended:
        # クラスタリング結果ごとにインデックスを付け保存
        for i, (label, face_image_path) in enumerate(zip(pred_labels, face_image_path_list)):
            if i == closest[label]:
                shutil.move(face_image_path,
                            join(output_path, "{}", "closest.png").format(label))
            else:
                shutil.move(face_image_path,
                            join(output_path, "{}", "{}").format(label, os.path.basename(face_image_path)))
        print('顔画像切り出し完了．\nPath: {} を確認して，対象外人物のフォルダを削除してください．\n完了した場合，何か入力してください．'.format(
            os.path.abspath(output_path)))
        # Launch Nautilus (Linux only)
        import subprocess
        subprocess.run(["xdg-open", output_path])
        # 入力待ち
        input()

        """
        known_faces = [[face1_enc1, face1_enc2, ... ], [face2_enc1, face2_enc2, ... ], ... ]
        """
        cluster_dir_path_list = sorted(
            [join(output_path, d) for d in os.listdir(output_path) if os.path.isdir(join(output_path, d))])
        for cluster_dir_path in cluster_dir_path_list:
            idx = sorted([int(f[:-4]) for f in os.listdir(cluster_dir_path) if f != "closest.png"])
            face_images_encoded = face_image_encoded_np[idx]
            known_faces.append(face_images_encoded)
    else:
        for i in range(face_num):
            idx = [j for j in range(len(pred_labels)) if pred_labels[j] == i]
            face_images_encoded = face_image_encoded_np[idx]
            known_faces.append(face_images_encoded)

    result = defaultdict(list)
    for r in result_from_detect_face:
        face_encoded_list = [face.encoding for face in r]
        # 入力画像とのマッチング
        face_matching_th = 0.35
        face_distance_list = [[] for _ in range(len(known_faces))]
        for face_encoded in face_encoded_list:
            tmp_distance = my_compare_faces(known_faces, face_encoded)
            for i in range(len(face_distance_list)):
                face_distance_list[i].append(tmp_distance[i])
        # print(face_distance_list)
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

        for face, idx in zip(r, face_index_list):
            if idx is not None:
                result[idx].append(face)

    print(result)
    return result
