import pandas as pd
import numpy as np
from scipy import stats
from glob import glob
import re
import os

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def window_feature(data):
    """
    ウィンドウごとのデータから11次元データを抽出
    """
    features = []

    features.append(np.mean(data)) # 平均値
    features.append(np.std(data)) # 標準偏差
    features.append(stats.median_abs_deviation(data)) # median_absolute_deviation
    features.append(np.max(data)) # max
    features.append(np.min(data)) # min
    features.append(np.mean(data**2)) # energy（二乗平均
    normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    features.append(stats.entropy(normed_data)) # entropy
    features.append(stats.iqr(data)) # 四分位範囲
    features.append(np.max(data) - np.min(data)) # 範囲
    features.append(stats.skew(data)) # Skewness 
    features.append(stats.kurtosis(data)) # Kurtosis

    return features

def window_success(success):
    return np.count_nonzero(success)/len(success)

def sliding_window(data, success=False, width=32, overlap=16):
    """
    スライディングウィンドウ法により特徴抽出
    """
    
    size = len(data)

    # ウィンドウの数：width, width+overlap, width+2*overlap, としていってsizeになるまでの個数
    n_windows = (size - width) // overlap
    
    # 各ウィンドウの最終位置を格納
    windows = []
    for i in range(n_windows):
        windows.append(width+i*overlap)
    
    # ウィンドウごとに集約特徴量を抽出
    outputs = []    
    for window in windows:
        trans_data = data[window-width:window]

        if success:
            outputs.append(window_success(trans_data))

        else:
            outputs.append(window_feature(trans_data))
        
    return np.array(outputs)

def feature_grouping(df):
    """
    集約特徴量の作成
    ロー、ピッチ、ヨーごとにスライディングウィンドウ
    """

    # ロー、ピッチ、ヨーを取り出し
    data_x = df["pose_Rx"].values
    data_y = df["pose_Ry"].values
    data_z = df["pose_Rz"].values
    success = df["success"].values

    # それぞれスライディングウィンドウ
    outputs_x = sliding_window(data_x)
    outputs_y = sliding_window(data_y)
    outputs_z = sliding_window(data_z)
    success_rate = sliding_window(success, success=True)

    # 合成
    success_rate = success_rate.reshape(len(success_rate), 1)
    # outputs = np.concatenate([outputs_x, outputs_y, outputs_z], axis=1)
    outputs = np.concatenate([outputs_x, outputs_y, outputs_z, success_rate], axis=1)

    return outputs

def get_timedata(df):
    """
    時間情報（フレーム, タイムスタンプ）を各ウィンドウに追加
    """
    def sliding_window_times(data, width=32, overlap=16):
        
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

def trans_pandas(data, labels, frames, timestamps, columns_ori):
    """ numpy -> pandas """
    
    # コラム
    columns_x = [s+"_x" for s in columns_ori]
    columns_y = [s+"_y" for s in columns_ori]
    columns_z = [s+"_z" for s in columns_ori]
    columns = columns_x[:]
    columns.extend(columns_y)
    columns.extend(columns_z)
    columns.append("success_rate")
    columns.append("frame_in")
    columns.append("frame_out") 	
    columns.append("timestamp_in") 	
    columns.append("timestamp_out") 
    columns.append("label")

    # dataとlabelを結合
    data = np.concatenate([data, frames], axis=1)
    data = np.concatenate([data, timestamps], axis=1)
    data = np.concatenate([data, labels], axis=1)

    # dataframe作成
    out_df = pd.DataFrame(data=data, columns=columns)

    return out_df

def window_check(df, th=0.8, remove_sr=True):
    """
    success_rate(各ウィンドウのsuccessの割合)がth(しきい値)以下ならその行を消す処理
    """

    # th以下の行を削除
    sr = df["success_rate"].values
    index = np.where(sr < th)[0]
    df = df.drop(df.index[index])

    # success_rateコラムも削除
    if remove_sr:
        df = df.drop("success_rate", axis=1)

    return df

def make_dataset(dir_path, out_path, feature_dim=11):
    """
    main
    複数のcsvファイルを読み込んで、スライディングウィンドウ法により
    約1秒間のopenfaceのデータから集約特徴量生成

    label: 1うなづきあり、0うなずきなし
    """

    # outputs_list = np.empty((0, feature_dim*3))
    outputs_list = np.empty((0, feature_dim*3+1))
    labels_list = np.empty((0, 1))
    frames_list = np.empty((0, 2))
    timestamps_list = np.empty((0, 2))

    # 出力するDataFrameのコラム
    columns = ["mean", "std", "mad", "max", "min", "energy", "entropy", "iqr", "range", "skewness", "kurtosis"]

    # 各動画のopenface情報から特徴量抽出
    for csv_path in sorted(glob(dir_path+"*csv*"), key=numericalSort):
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

        # スライディング法により特徴量抽出
        outputs = feature_grouping(input_df)
        labels = np.full(len(outputs), int(label))
        labels = labels.reshape(len(labels), 1)

        # 時間情報（フレーム, タイムスタンプ）を各ウィンドウに追加
        frames, timestamps = get_timedata(input_df)

        # pandasに変換
        output_df = trans_pandas(outputs, labels, frames, timestamps, columns)

        # successの処理
        output_df = window_check(output_df)
        print(output_df.shape)

        # 保存
        csv_name = os.path.splitext(os.path.basename(csv_path))[0] # 拡張子なしファイル名
        output_df.to_csv(f"{out_path}{csv_name}_processed.csv", index=False)

        # 追加
        outputs_list = np.concatenate([outputs_list, outputs])
        frames_list = np.concatenate([frames_list, frames])
        timestamps_list = np.concatenate([timestamps_list, timestamps])
        labels_list = np.concatenate([labels_list, labels])

    # pandasに変換
    outputs_df = trans_pandas(outputs_list, labels_list, frames_list, timestamps_list, columns)

    # successの処理
    outputs_df = window_check(outputs_df)
    print(outputs_df)

    outputs_df.to_csv(f"{out_path}entire.csv", index=False)

if __name__ == "__main__":

    DIR_PATH = "../../movie/processed_data/"
    OUT_PATH = "./csv/"
    make_dataset(DIR_PATH, OUT_PATH)
    