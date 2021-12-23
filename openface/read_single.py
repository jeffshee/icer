import pandas as pd
import numpy as np
import os

from feature_engineering import feature_grouping, get_face_data, get_timedata, remove_low_confidence_rows


def trans_pandas(data, columns_ori, frames, timestamps, faces):
    """ numpy -> pandas """

    # カラム
    columns_x = [s + "_x" for s in columns_ori]
    columns_y = [s + "_y" for s in columns_ori]
    columns_z = [s + "_z" for s in columns_ori]
    columns = columns_x[:]
    columns.extend(columns_y)
    columns.extend(columns_z)
    columns.append("frame_in")
    columns.append("frame_out")
    columns.append("timestamp_in")
    columns.append("timestamp_out")
    columns.append("face_x")
    columns.append("face_y")
    columns.append("face_radius")

    # 時間情報を結合
    data = np.concatenate([data, frames], axis=1)
    data = np.concatenate([data, timestamps], axis=1)

    # 顔情報を結合
    data = np.concatenate([data, faces], axis=1)

    # dataframe作成
    out_df = pd.DataFrame(data=data, columns=columns)

    return out_df


def extract_feature(csv_path, output_dir):
    # csv読み込み
    input_df = pd.read_csv(csv_path)

    # csv名
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]

    # 元のcsvのカラム名の先頭に空白があるため
    columns_dict = {}
    for column_name in input_df.columns:
        columns_dict[column_name] = column_name.lstrip()
    input_df = input_df.rename(columns=columns_dict)
    print(input_df)

    # confidenceの低い行を削除
    input_df = remove_low_confidence_rows(input_df)

    # 特徴量抽出
    columns = ["mean", "std", "mad", "max", "min", "energy", "entropy", "iqr", "range", "skewness", "kurtosis"]

    # スライディング法により特徴量抽出
    outputs = feature_grouping(input_df)

    # 時間情報（フレーム, タイムスタンプ）を各ウィンドウに追加
    frames, timestamps = get_timedata(input_df)

    # 顔情報（顔のx, y座標, 半径）を各ウィンドウに追加
    faces = get_face_data(input_df)

    # pandasに変換
    output_df = trans_pandas(
        data=outputs,
        columns_ori=columns,
        frames=frames,
        timestamps=timestamps,
        faces=faces
    )

    # 保存
    output_df.to_csv(os.path.join(output_dir, f"{csv_name}_processed.csv"), index=False)


if __name__ == "__main__":
    DIR_PATH = "/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_0/reid/processed"
    CSV_PATH = os.path.join(DIR_PATH, "reid0.csv")
    extract_feature(CSV_PATH, DIR_PATH)
    CSV_PATH = os.path.join(DIR_PATH, "reid1.csv")
    extract_feature(CSV_PATH, DIR_PATH)
    CSV_PATH = os.path.join(DIR_PATH, "reid2.csv")
    extract_feature(CSV_PATH, DIR_PATH)
    CSV_PATH = os.path.join(DIR_PATH, "reid3.csv")
    extract_feature(CSV_PATH, DIR_PATH)
    CSV_PATH = os.path.join(DIR_PATH, "reid4.csv")
    extract_feature(CSV_PATH, DIR_PATH)
