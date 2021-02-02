import os
from os.path import join
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from main_util import cleanup_directory
import face_recognition

import numpy as np
import shutil
import cv2
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


def cluster_face(result_from_detect_face, max_face_num, input_path, output_path="face_cluster"):
    video_capture = cv2.VideoCapture(input_path)

    for cluster_index in range(max_face_num):
        cleanup_directory(join(output_path, '{}'.format(cluster_index)))

    count = 0
    target_cluster_size = 1000
    break_flag = False
    for i in range(max_face_num, 0, -1):
        for r in result_from_detect_face:
            if len(r) == i:
                for face in r:
                    output_face_image(video_capture, face, join(output_path, "{}.png".format(count)))
                    count += 1
                    if count >= target_cluster_size:
                        break_flag = True
                        break
            if break_flag:
                break
        if break_flag:
            break

    # face_image_name_list = os.listdir(target_image_path)
    # face_image_name_list = [f for f in face_image_name_list if
    #                         os.path.isfile(os.path.join(target_image_path, f))]  # ファイル名のみの一覧を取得
    #
    # # 画像を読み込みエンコード
    # face_image_encoded_list = []
    # for face_image_name in face_image_name_list:
    #     face_image = face_recognition.load_image_file(join(target_image_path, face_image_name))
    #     face_image_encoded_list.append(face_recognition.face_encodings(face_image)[0])
    # face_image_encoded_np = np.array(face_image_encoded_list)
    #
    # # 学習(cluster_num種類のグループにクラスタリングする)
    # model = KMeans(n_clusters=max_face_num)
    # model.fit(face_image_encoded_np)
    #
    # # 学習結果のラベル
    # pred_labels = model.labels_
    # print(np.bincount(pred_labels))
    #
    # closest, _ = pairwise_distances_argmin_min(model.cluster_centers_,
    #                                            face_image_encoded_np)  # Get nearest point to centroid
    #
    # # クラスタリング結果ごとにインデックスを付け保存
    # for i, (label, face_image_name) in enumerate(zip(pred_labels, face_image_name_list)):
    #     if i == closest[label]:
    #         shutil.copyfile(join(target_image_path, face_image_name),
    #                         join(output_path, "{}", "{}.PNG").format(label, "closest"))
    #     else:
    #         shutil.copyfile(join(target_image_path, face_image_name),
    #                         join(output_path, "{}", "{}.PNG").format(label, i))
