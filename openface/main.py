import os
import sys
import pandas as pd
import joblib
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from openface.create_reid_single import create_reid
from openface.read_single import extract_feature
from openface.nod_detection import train, test_multi


def main(
    input_video_path: str,
    face_num: int,
    script_path: str,
    train_csv_path: str = "./data/nod_detection_train_data.csv",
    valid_csv_path: str = "./data/nod_detection_valid_data.csv",
    output_dir: str = None,
    cwd: str = './openface_src/',
):
    reid_dir = os.path.join(os.path.dirname(input_video_path), "reid")

    # 複数人の動画から各人の領域をcrop (全体のmainルーチンに移行)
    create_reid(input_video_path, face_num)

    # 各人の動画をopenfaceで処理 (shell scriptで実行: docker&local)
    subprocess.call(f"chmod +x {script_path}".split(" "))
    # vscode実行時にはcwdの位置に注意！
    if os.path.exists(cwd) is False:
        cwd = str(os.getcwd()) + "/openface/openface_src/"
    sp = subprocess.run([f"{script_path} {reid_dir}"], shell=True, encoding='utf-8', cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"executed openface:\n{sp}")

    # openfaceで出力された特徴 (顔のrotation) からsliding windowで特徴量抽出
    processed_dir = os.path.join(reid_dir, "processed")
    for i in range(face_num):
        csv_path = os.path.join(processed_dir, f"reid{i}.csv")
        extract_feature(csv_path, processed_dir)

    # ===== SVMモデルのTraining ===== #
    train_df = pd.read_csv(train_csv_path, engine='python')
    valid_df = pd.read_csv(valid_csv_path, engine='python')
    model, feature_columns, sc = train(train_df=train_df, valid_df=valid_df)

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
    if output_dir is None:
        output_dir = processed_dir
    test_multi(test_dfs, rois, model, sc, feature_columns, calibrate_video_path, output_csv_dir=output_dir, output_vid_dir=output_dir)


if __name__ == '__main__':
    input_video_path = "/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_4/output.avi"
    face_num = 5
    script_path = "/home/icer/Project/icer/openface/run.sh"
    train_csv_path = "/home/icer/Project/icer/openface//data/nod_detection_train_data.csv"
    valid_csv_path = "/home/icer/Project/icer/openface//data/nod_detection_valid_data.csv"
    output_dir = "/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_4/result/"
    main(input_video_path=input_video_path, face_num=face_num, script_path=script_path, train_csv_path=train_csv_path, valid_csv_path=valid_csv_path, output_dir=output_dir)
