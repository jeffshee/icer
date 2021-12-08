import os
import sys
import pandas as pd
import numpy as np
from glob import glob


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from feature_engineering import numericalSort, feature_grouping, get_timedata, trans_pandas, remove_low_confidence_rows


def make_dataset(src_dir, out_dir, feature_dim=11):
    """
    Train/Validationデータ作成用

    複数のcsvファイルを読み込んで、スライディングウィンドウ法により
    約1秒間のopenfaceのデータから集約特徴量生成

    label: [0: 頷きなし, 1: 頷きあり]

    csv名にnodが含まれない場合 (例: other-〇〇.csv) -> labelは 0
    csv名にnodが含まれる場合 (例: nod-〇〇.csv)     -> labelは 1
    """

    outputs_list = np.empty((0, feature_dim * 3))
    # outputs_list = np.empty((0, feature_dim * 3 + 1))
    labels_list = np.empty((0, 1))
    frames_list = np.empty((0, 2))
    timestamps_list = np.empty((0, 2))

    # 出力するDataFrameのカラム
    columns = ["mean", "std", "mad", "max", "min", "energy", "entropy", "iqr", "range", "skewness", "kurtosis"]

    csv_list = sorted(glob(src_dir + "*csv*"), key=numericalSort)

    # 各動画のopenface情報から特徴量抽出
    for csv_path in csv_list:
        input_df = pd.read_csv(csv_path)

        # 動画名に"nod" -> 1, ない -> 0
        if "nod" in csv_path:
            label = 1
        else:
            label = 0
        print(f"csv: {csv_path}, label: {label}")

        # 元のcsvのカラム名の先頭に空白があるため
        columns_dict = {}
        for column_name in input_df.columns:
            columns_dict[column_name] = column_name.lstrip()
        input_df = input_df.rename(columns=columns_dict)

        # successの処理
        print(input_df.shape)
        input_df = remove_low_confidence_rows(input_df)
        print(input_df.shape)

        # スライディング法により特徴量抽出
        outputs = feature_grouping(input_df)
        labels = np.full(len(outputs), int(label))
        labels = labels.reshape(len(labels), 1)

        # 時間情報（フレーム, タイムスタンプ）を各ウィンドウに追加
        frames, timestamps = get_timedata(input_df)

        # pandasに変換
        output_df = trans_pandas(outputs, labels, frames, timestamps, columns)

        # 保存
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]  # 拡張子なしファイル名
        output_df.to_csv(f"{out_dir}{csv_name}_processed.csv", index=False)

        # 追加
        outputs_list = np.concatenate([outputs_list, outputs])
        frames_list = np.concatenate([frames_list, frames])
        timestamps_list = np.concatenate([timestamps_list, timestamps])
        labels_list = np.concatenate([labels_list, labels])

    # pandasに変換
    outputs_df = trans_pandas(outputs_list, labels_list, frames_list, timestamps_list, columns)
    outputs_df.to_csv(f"{out_dir}dataset.csv", index=False)


if __name__ == "__main__":
    INPUT_DIR = "/home/icer/Videos/Webcam/processed_data/"
    OUT_DIR = "/icer/Videos/Webcam/processed_data/"
    make_dataset(INPUT_DIR, OUT_DIR)
