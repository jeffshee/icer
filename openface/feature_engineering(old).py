import os
import pandas as pd
import numpy as np
from scipy import stats
from glob import glob


def window_feature(data):
    """
    ウィンドウごとのデータから11次元データを抽出
    """
    features = []
    features.append(np.mean(data))  # 平均値
    features.append(np.std(data))  # 標準偏差
    features.append(stats.median_abs_deviation(data))  # median_absolute_deviation
    features.append(np.max(data))  # max
    features.append(np.min(data))  # min
    features.append(np.mean(data**2))  # energy（二乗平均）
    features.append(stats.entropy(data))  # entropy (負値が含まれると-infになる)
    features.append(stats.iqr(data))  # 四分位範囲
    features.append(np.max(data) - np.min(data))  # 範囲
    features.append(stats.skew(data))  # Skewness
    features.append(stats.kurtosis(data))  # Kurtosis

    return features


def sliding_window(data, label, width=32, overlap=16):
    """
    スライディングウィンドウ法により特徴抽出
    """
    size = len(data)

    # ウィンドウの数：width, width+overlap, width+2*overlap, としていってsizeになるまでの個数
    n_windows = (size - width) // overlap

    # 各ウィンドウの最終位置を格納
    windows = []
    for i in range(n_windows):
        windows.append(width + i * overlap)

    # ウィンドウごとに集約特徴量を抽出
    outputs = []
    for window in windows:
        trans_data = data[window - width:window]
        outputs.append(window_feature(trans_data))

    # 正解ラベル
    labels = np.full(n_windows, int(label))

    return np.array(outputs), labels.reshape(len(labels), 1)


def feature_grouping(df, label):

    # ロー、ピッチ、ヨーを取り出し
    data_x = df[" pose_Rx"].values
    data_y = df[" pose_Ry"].values
    data_z = df[" pose_Rz"].values

    # それぞれスライディングウィンドウ
    outputs_x, labels_x = sliding_window(data_x, label)
    outputs_y, labels_y = sliding_window(data_y, label)
    outputs_z, labels_z = sliding_window(data_z, label)

    # 値同じならok
    print(len(labels_x), len(labels_y), len(labels_z))

    # 合成
    outputs = np.concatenate([outputs_x, outputs_y, outputs_z], axis=1)
    # labels = np.concatenate([labels_x, labels_y, labels_z], axis=1)

    return outputs, labels_x


def get_timedata(df):
    def sliding_window_times(data, width=32, overlap=16):
        """
        スライディングウィンドウ法により特徴抽出
        """
        size = len(data)

        # ウィンドウの数：width, width+overlap, width+2*overlap, としていってsizeになるまでの個数
        n_windows = (size - width) // overlap

        # 各ウィンドウの最終位置を格納
        windows = []
        for i in range(n_windows):
            windows.append(width + i * overlap)

        # ウィンドウごとにtimestamp_in, timestamp_outを取得
        frames, timestamps = [], []
        for window in windows:
            window_data = data[window - width:window]
            frame_in = window_data.loc[:, "frame"].iloc[0]
            frame_out = window_data.loc[:, "frame"].iloc[-1]
            timestamp_in = window_data.loc[:, " timestamp"].iloc[0]
            timestamp_out = window_data.loc[:, " timestamp"].iloc[-1]
            frames.append([frame_in, frame_out])
            timestamps.append([timestamp_in, timestamp_out])

        return np.array(frames), np.array(timestamps)

    # frame_in, frame_out, timestamp_in, timestamp_outを取り出し
    frames, timestamps = sliding_window_times(df)

    return frames, timestamps


def trans_pandas(data, labels, frames, timestamps, columns_ori):
    """ numpy -> pandas """

    # コラム
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
    columns.append("label")

    # dataとlabelを結合
    # data = np.concatenate([data, labels], axis=1)
    data = np.concatenate([data, frames], axis=1)
    data = np.concatenate([data, timestamps], axis=1)
    data = np.concatenate([data, labels], axis=1)

    # dataframe作成
    out_df = pd.DataFrame(data=data, columns=columns)

    return out_df


def make_dataset(dir_path, out_path, feature_dim=11):

    outputs_list = np.empty((0, feature_dim * 3))
    labels_list = np.empty((0, 1))
    frames_list = np.empty((0, 2))
    timestamps_list = np.empty((0, 2))

    # 各動画のopenface情報から特徴量抽出
    for csv_path in glob(dir_path + "*csv*"):
        input_df = pd.read_csv(csv_path)

        # 動画名に"nod" -> True, ない -> False
        if "nod" in csv_path:
            label = True
        else:
            label = False

        print(f"csv: {csv_path}, label: {label}")

        # スライディング法により特徴量抽出
        outputs, labels = feature_grouping(input_df, label)

        frames, timestamps = get_timedata(input_df)

        outputs_list = np.concatenate([outputs_list, outputs])
        labels_list = np.concatenate([labels_list, labels])
        frames_list = np.concatenate([frames_list, frames])
        timestamps_list = np.concatenate([timestamps_list, timestamps])

    # pandasに変換
    columns = ["mean", "std", "mad", "max", "min", "energy", "entropy", "iqr", "range", "skewness", "kurtosis"]
    output_df = trans_pandas(outputs_list, labels_list, frames_list, timestamps_list, columns)

    print(output_df)

    output_df.to_csv(out_path)


def make_test_dataset(csv_path, out_path):
    def get_timedata(df):
        def sliding_window_times(data, width=32, overlap=16):
            """
            スライディングウィンドウ法により特徴抽出
            """
            size = len(data)

            # ウィンドウの数：width, width+overlap, width+2*overlap, としていってsizeになるまでの個数
            n_windows = (size - width) // overlap

            # 各ウィンドウの最終位置を格納
            windows = []
            for i in range(n_windows):
                windows.append(width + i * overlap)

            # ウィンドウごとにtimestamp_in, timestamp_outを取得
            frames, timestamps = [], []
            for window in windows:
                window_data = data[window - width:window]
                frame_in = window_data.loc[:, "frame"].iloc[0]
                frame_out = window_data.loc[:, "frame"].iloc[-1]
                timestamp_in = window_data.loc[:, "timestamp"].iloc[0]
                timestamp_out = window_data.loc[:, "timestamp"].iloc[-1]
                frames.append([frame_in, frame_out])
                timestamps.append([timestamp_in, timestamp_out])

            return np.array(frames), np.array(timestamps)

        # frame_in, frame_out, timestamp_in, timestamp_outを取り出し
        frames, timestamps = sliding_window_times(df)

        return frames, timestamps

    def sliding_window(data, width=32, overlap=16):
        """
        スライディングウィンドウ法により特徴抽出
        """
        size = len(data)

        # ウィンドウの数：width, width+overlap, width+2*overlap, としていってsizeになるまでの個数
        n_windows = (size - width) // overlap

        # 各ウィンドウの最終位置を格納
        windows = []
        for i in range(n_windows):
            windows.append(width + i * overlap)

        # ウィンドウごとに集約特徴量を抽出
        outputs = []
        for window in windows:
            trans_data = data[window - width:window]
            outputs.append(window_feature(trans_data))

        return np.array(outputs)

    def feature_grouping(df):

        # ロー、ピッチ、ヨーを取り出し
        data_x = df["pose_Rx"].values
        data_y = df["pose_Ry"].values
        data_z = df["pose_Rz"].values

        # それぞれスライディングウィンドウ
        outputs_x = sliding_window(data_x)
        outputs_y = sliding_window(data_y)
        outputs_z = sliding_window(data_z)

        # 合成
        outputs = np.concatenate([outputs_x, outputs_y, outputs_z], axis=1)

        return outputs

    def trans_pandas(data, frames, timestamps, columns_ori):
        """ numpy -> pandas """

        # コラム
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

        # dataとlabelを結合
        # data = np.concatenate([data, labels], axis=1)
        data = np.concatenate([data, frames], axis=1)
        data = np.concatenate([data, timestamps], axis=1)

        # dataframe作成
        out_df = pd.DataFrame(data=data, columns=columns)

        return out_df

    input_df = pd.read_csv(csv_path)

    # スライディング法により特徴量抽出
    outputs = feature_grouping(input_df)
    frames, timestamps = get_timedata(input_df)

    # pandasに変換
    columns = ["mean", "std", "mad", "max", "min", "energy", "entropy", "iqr", "range", "skewness", "kurtosis"]
    output_df = trans_pandas(outputs, frames, timestamps, columns)

    print(output_df)

    output_df.to_csv(out_path)


if __name__ == "__main__":

    # DIR_PATH = "/home/icer/Videos/Webcam/processed_data/"
    # OUT_PATH = "/home/icer/Project/icer/openface/data/preprocess_araki.csv"
    # make_dataset(DIR_PATH, OUT_PATH)

    DIR_PATH = "/home/icer/Project/openface_dir/output_vid_dir/splited_test_csv/"
    OUT_DIR = "/home/icer/Project/openface_dir/output_vid_dir/feature_splited_test_csv/"

    for csv_path in glob(DIR_PATH + "*csv*"):
        print(f"csv: {csv_path}")
        fname = os.path.basename(csv_path) 
        make_test_dataset(csv_path, OUT_DIR + fname)
