# -*- coding: utf-8 -*-

import os
from os.path import join
import shutil

import cv2
import face_recognition
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from emotion_recognition import cleanup_directory


def detect_face(target_video_path, output_path, k_resolution, capture_image_num=50, model='hog'):
    cleanup_directory(output_path)

    input_movie = cv2.VideoCapture(target_video_path)  # 動画を読み込む
    original_w = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))  # 動画の幅を測る
    original_h = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 動画の高さを測る
    resize_rate = (1080 * k_resolution) / original_w  # 動画の横幅を変更
    w = int(original_w * resize_rate)
    h = int(original_h * resize_rate)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画の長さを測る

    max_face_detection_num = 0
    capture_image_num = min(capture_image_num, length)

    if model == 'cnn':
        rgb_frames = []
        print('CNN detect face')
        for i, frame_number in enumerate(range(0, length, length // capture_image_num)):
            print("frame_number:", frame_number)
            input_movie.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # 動画の開始フレームを設定

            # 動画を読み込む
            ret, frame = input_movie.read()

            # 動画が読み取れない場合は終了
            if not ret:
                break

            # フレームのサイズを調整する
            frame = cv2.resize(frame, (w, h))

            # openCVのBGRをRGBに変更
            rgb_frames.append(frame[:, :, ::-1])

        # rgb_frames = np.asarray(rgb_frames)
        print(len(rgb_frames))
        face_location_lists = face_recognition.batch_face_locations(rgb_frames, number_of_times_to_upsample=1,
                                                                    batch_size=1)  # model="cnn"にすると検出率は上がるが10倍以上時間がかかる top,right,bottom,leftの順
        print(face_location_lists)
        max_face_detection_num = max(max_face_detection_num, max([len(f_list) for f_list in face_location_lists]))
        for i, (face_location_list, rgb_frame) in enumerate(zip(face_location_lists, rgb_frames)):
            for j, (top, right, bottom, left) in enumerate(face_location_list):
                # 顔領域を切り取る
                dst = frame[top:bottom, left:right]
                rgb_dst = rgb_frame[top:bottom, left:right]
                if face_recognition.face_encodings(rgb_dst):
                    cv2.imwrite(join(output_path, '{}_{}.PNG').format(j, i), dst)

    else:
        for i, frame_number in enumerate(range(0, length, length // capture_image_num)):
            print("frame_number:", frame_number)
            input_movie.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # 動画の開始フレームを設定

            # 動画を読み込む
            ret, frame = input_movie.read()

            # 動画が読み取れない場合は終了
            if not ret:
                break

            # フレームのサイズを調整する
            frame = cv2.resize(frame, (w, h))

            # openCVのBGRをRGBに変更
            rgb_frame = frame[:, :, ::-1]

            # 顔領域の検出
            face_location_list = face_recognition.face_locations(rgb_frame,
                                                                 model="hog")  # model="cnn"にすると検出率は上がるが10倍以上時間がかかる top,right,bottom,leftの順
            max_face_detection_num = max(max_face_detection_num, len(face_location_list))
            for j, (top, right, bottom, left) in enumerate(face_location_list):
                # 顔領域を切り取る
                dst = frame[top:bottom, left:right]
                rgb_dst = rgb_frame[top:bottom, left:right]
                if face_recognition.face_encodings(rgb_dst):
                    cv2.imwrite(join(output_path, '{}_{}.PNG').format(j, i), dst)

    return max_face_detection_num


def cluster_face(target_image_path, output_path, cluster_num):
    for cluster_index in range(cluster_num):
        cleanup_directory(join(output_path, '{}'.format(cluster_index)))

    face_image_name_list = os.listdir(target_image_path)
    face_image_name_list = [f for f in face_image_name_list if
                            os.path.isfile(os.path.join(target_image_path, f))]  # ファイル名のみの一覧を取得

    # 画像を読み込みエンコード
    face_image_encoded_list = []
    for face_image_name in face_image_name_list:
        face_image = face_recognition.load_image_file(join(target_image_path, face_image_name))
        face_image_encoded_list.append(face_recognition.face_encodings(face_image)[0])
    face_image_encoded_np = np.array(face_image_encoded_list)

    # 学習(cluster_num種類のグループにクラスタリングする)
    model = KMeans(n_clusters=cluster_num)
    model.fit(face_image_encoded_np)

    # 学習結果のラベル
    pred_labels = model.labels_
    print(np.bincount(pred_labels))

    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_,
                                               face_image_encoded_np)  # Get nearest point to centroid

    # クラスタリング結果ごとにインデックスを付け保存
    for i, (label, face_image_name) in enumerate(zip(pred_labels, face_image_name_list)):
        if i == closest[label]:
            shutil.copyfile(join(target_image_path, face_image_name),
                            join(output_path, "{}", "{}.PNG").format(label, "closest"))
        else:
            shutil.copyfile(join(target_image_path, face_image_name),
                            join(output_path, "{}", "{}.PNG").format(label, i))


if __name__ == '__main__':
    path_video = "video/"
    video_name = ["Take01", "Take02", "190130"][2]
    target_video_path = "{}{}".format(path_video, video_name) + ".mp4"
    output_path = "face_image_auto/{}/".format(video_name)
    k_resolution = 3
    capture_image_num = 100  # capture_image_numフレームから顔検出を行う
    cluster_num = detect_face(target_video_path, output_path, k_resolution, capture_image_num)

    cluster_face(output_path, output_path + "cluster/", cluster_num=cluster_num)
