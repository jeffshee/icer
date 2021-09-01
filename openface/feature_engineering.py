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
    features.append(np.mean(data**2)) # energy（二乗平均）
    features.append(stats.entropy(data)) # entropy
    features.append(stats.iqr(data)) # 四分位範囲
    features.append(np.max(data) - np.min(data)) # 範囲
    features.append(stats.skew(data)) # Skewness 
    features.append(stats.kurtosis(data)) # Kurtosis

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
        windows.append(width+i*overlap)
    
    # ウィンドウごとに集約特徴量を抽出
    outputs = []    
    for window in windows:
        trans_data = data[window-width:window]
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

def trans_pandas(data, labels, columns_ori):
    """ numpy -> pandas """
    
    # コラム
    columns_x = [s+"_x" for s in columns_ori]
    columns_y = [s+"_y" for s in columns_ori]
    columns_z = [s+"_z" for s in columns_ori]
    columns = columns_x[:]
    columns.extend(columns_y)
    columns.extend(columns_z)
    columns.append("label")

    # dataとlabelを結合
    # labels = labels.reshape(len(labels), 1)
    data = np.concatenate([data, labels], axis=1)

    # dataframe作成
    out_df = pd.DataFrame(data=data, columns=columns)

    return out_df

def make_dataset(dir_path, out_path, feature_dim=11):

    outputs_list = np.empty((0, feature_dim*3))
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

        # スライディング法により特徴量抽出
        outputs, labels = feature_grouping(input_df, label)

        outputs_list = np.concatenate([outputs_list, outputs])
        labels_list = np.concatenate([labels_list, labels])

    # pandasに変換
    columns = ["mean", "std", "mad", "max", "min", "energy", "entropy", "iqr", "range", "skewness", "kurtosis"]
    output_df = trans_pandas(outputs_list, labels_list, columns)

    print(output_df)

    output_df.to_csv(out_path)

if __name__ == "__main__":

    DIR_PATH = "../../movie/processed_data/"
    OUT_PATH = "./preprocess_1.csv"
    make_dataset(DIR_PATH, OUT_PATH)