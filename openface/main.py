import os
import sys
import pandas as pd
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from openface.create_reid_single import create_reid
from openface.read_single import extract_feature
from openface.nod_detection import train, test_multi


def main():
    input_video_path = "/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_0/output.avi"
    reid_dir = os.path.join(os.path.dirname(input_video_path), "reid")
    face_num = 5

    # 複数人の動画から各人の領域をcrop
    # create_reid(input_video_path, face_num)

    # 各人の動画をopenfaceで処理 (shell scriptで実行: docker上)


    # openfaceで出力された特徴 (顔のrotation) からsliding windowで特徴量抽出
    processed_dir = os.path.join(reid_dir, "processed")
    for i in range(face_num):
        csv_path = os.path.join(processed_dir, f"reid{i}.csv")
        extract_feature(csv_path, processed_dir)

    # ===== SVMモデルのTraining ===== #
    trvl_csv_path = "/home/icer/Project/icer/openface/data/preprocess_araki.csv"
    trvl_df = pd.read_csv(trvl_csv_path, engine='python')
    model, feature_columns, sc = train(trvl_df)
    # joblib.dump(model, os.path.join(processed_dir + "model.sav"))

    # ===== 頷き検出とその結果をビデオに反映 ===== #
    # 各人のROI (crop後のビデオにおけるbounding-box) が入ったリストを取得
    rois = joblib.load(os.path.join(reid_dir, "rois.txt"))

    # 各人の特徴量を読み込み
    test_dfs = []
    for i in range(face_num):
        test_csv_path = os.path.join(processed_dir, f"reid{i}_processed.csv")
        test_df = pd.read_csv(test_csv_path, engine='python')
        test_dfs.append(test_df)

    calibrate_video_path = os.path.join(reid_dir, "calibrate.mp4")  # 検出結果を反映させるビデオのpath
    output_dir = processed_dir
    test_multi(test_dfs, rois, model, sc, feature_columns, calibrate_video_path, output_csv_dir=output_dir, output_vid_dir=output_dir)


if __name__ == '__main__':
    main()
