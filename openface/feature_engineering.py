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

def feature_grouping(df, label):
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
    success_rate = sliding_window(success, True)

    # 正解ラベル
    labels = np.full(len(outputs_x), int(label))
    labels = labels.reshape(len(labels), 1)

    # 合成
    success_rate = success_rate.reshape(len(success_rate), 1)
    # outputs = np.concatenate([outputs_x, outputs_y, outputs_z], axis=1)
    outputs = np.concatenate([outputs_x, outputs_y, outputs_z, success_rate], axis=1)

    return outputs, labels

def trans_pandas(data, labels, columns_ori):
    """ numpy -> pandas """
    
    # コラム
    columns_x = [s+"_x" for s in columns_ori]
    columns_y = [s+"_y" for s in columns_ori]
    columns_z = [s+"_z" for s in columns_ori]
    columns = columns_x[:]
    columns.extend(columns_y)
    columns.extend(columns_z)
    columns.append("success_rate")
    columns.append("label")

    # dataとlabelを結合
    data = np.concatenate([data, labels], axis=1)

    # dataframe作成
    out_df = pd.DataFrame(data=data, columns=columns)

    return out_df

def window_check(df, th=0.8):
    """
    success_rate(各ウィンドウのsuccessの割合)がth(しきい値)以下ならその行を消す処理
    """

    # th以下の行を削除
    sr = df["success_rate"].values
    index = np.where(sr < th)[0]
    df = df.drop(df.index[index])

    # success_rateコラムも削除
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

    # 各動画のopenface情報から特徴量抽出
    for csv_path in glob(dir_path+"*csv*"):
        input_df = pd.read_csv(csv_path)

        # 動画名に"nod" -> True, ない -> False
        if "nod" in csv_path:
            label = True
        else:
            label = False

        print(f"csv: {csv_path}, label: {label}")

        # 元のcsvのカラム名の先頭に空白があるため
        columns_dict = {}
        for column_name in input_df.columns:
            columns_dict[column_name] = column_name.lstrip()
        columns_dict
        input_df = input_df.rename(columns=columns_dict)

        # スライディング法により特徴量抽出
        outputs, labels = feature_grouping(input_df, label)

        outputs_list = np.concatenate([outputs_list, outputs])
        labels_list = np.concatenate([labels_list, labels])

    # pandasに変換
    columns = ["mean", "std", "mad", "max", "min", "energy", "entropy", "iqr", "range", "skewness", "kurtosis"]
    output_df = trans_pandas(outputs_list, labels_list, columns)

    # successの処理
    output_df = window_check(output_df)
    print(output_df)

    output_df.to_csv(out_path, index=False)

if __name__ == "__main__":

    DIR_PATH = "../../movie/processed_data/"
    OUT_PATH = "./preprocess_2.csv"
    make_dataset(DIR_PATH, OUT_PATH)