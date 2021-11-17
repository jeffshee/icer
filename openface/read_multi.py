import pandas as pd
import numpy as np
import math
import os

from feature_engineering import feature_grouping, get_timedata, window_check

def match_face_id(face_coord, face_id_dict):
    min_face_id = None
    min_coord_diff = (float('inf'), float('inf'))
    min_dist = float('inf')
    for k, v in face_id_dict.items():
        tmp_coord_diff = (abs(v[0] - face_coord[0]), abs(v[1] - face_coord[1]))
        tmp_dist = math.sqrt(tmp_coord_diff[0] ** 2 + tmp_coord_diff[1] ** 2)
        if tmp_dist < min_dist:
            min_face_id = k
            min_coord_diff = tmp_coord_diff
            min_dist = tmp_dist
    return min_face_id, min_dist, min_coord_diff

def devide_faces(df, csv_name, dir_path):
    """
    人（new_face_id）を見分けて分割
    """
    # 抽出
    _df = df[["frame", "face_id", "timestamp", "confidence", "success", "eye_lmk_x_25", "eye_lmk_y_25", "eye_lmk_x_53", "eye_lmk_y_53"]]
    new_df = _df.copy()

    # 眉間の座標
    new_df.loc[:, ['face_coord_x']] = (new_df['eye_lmk_x_25'] + new_df['eye_lmk_x_53']) / 2
    new_df.loc[:, ['face_coord_y']] = (new_df['eye_lmk_y_25'] + new_df['eye_lmk_y_53']) / 2

    face_id_dict = {}
    for index, row in new_df[(new_df['frame'] == 1)].iterrows():
        face_id_dict[index] = (row['face_coord_x'], row['face_coord_y'])
    print(face_id_dict)

    counter = 0
    total_dist = 0
    output_df = new_df.copy()

    for index, row in new_df.iterrows():
        face_coord = (row['face_coord_x'], row['face_coord_y'])
        face_id = row['face_id']
        
        new_face_id, dist, coord_diff = match_face_id(face_coord, face_id_dict)
        output_df.loc[index, 'new_face_id'] = new_face_id
        
        if face_id != new_face_id:
            # print(f"[{index}] not matched!: old_id: {face_id}, new_id: {new_face_id} (dist: {dist}, diff: {coord_diff})")
            # print(row)
            counter += 1
            total_dist += dist
            output_df.loc[index, 'face_id_update'] = True

    print(f"\n====\nmiss match total: {counter}")
    print(f"average dist: {dist / counter}")

    # dfにnew_face_idコラム追加
    output_df["new_face_id"] = output_df["new_face_id"].astype(int)
    df["new_face_id"] = output_df["new_face_id"]

    df.to_csv(f"{dir_path}new_face_id.csv")

    # idごとにcsv分割
    df_dict = {}
    id_list = []
    for id in np.unique(df["new_face_id"]):
        tmp_df = df[df["new_face_id"] == id]
        tmp_df.to_csv(f"{dir_path}{csv_name}_{str(id)}.csv", index=False)

        # 辞書型にdataframeごと保存
        df_dict[str(id)] = tmp_df
        id_list.append(str(id))

    return df_dict, id_list

def trans_pandas(data, columns_ori, frames, timestamps):
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

    # 時間情報を結合
    data = np.concatenate([data, frames], axis=1)
    data = np.concatenate([data, timestamps], axis=1)

    # dataframe作成
    out_df = pd.DataFrame(data=data, columns=columns)

    return out_df

def read_multi(csv_path, out_path):
    """
    複数人の動画からのopenfaceデータ(csv)を人ごとにわけてスライディングウィンドウし、
    別々のcsvファイルにしてout_pathに保存    
    """

    # csv読み込み
    input_df = pd.read_csv(csv_path)

    # csv名
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]

    # 元のcsvのカラム名の先頭に空白があるため
    columns_dict = {}
    for column_name in input_df.columns:
        columns_dict[column_name] = column_name.lstrip()
    input_df = input_df.rename(columns=columns_dict)

    # 人（new_face_id）を見分けてcsvを分割
    df_dict, id_list = devide_faces(input_df, csv_name, out_path)

    # idごとに特徴量抽出
    feature_dict = {}
    columns = ["mean", "std", "mad", "max", "min", "energy", "entropy", "iqr", "range", "skewness", "kurtosis"]

    for id in id_list:

        # スライディング法により特徴量抽出
        outputs = feature_grouping(df_dict[id])
        feature_dict[id] = outputs

        # 時間情報（フレーム, タイムスタンプ）を各ウィンドウに追加
        frames, timestamps = get_timedata(df_dict[id])

        print(f"id {id}: {frames.shape}")

        # pandasに変換
        output_df = trans_pandas(outputs, columns, frames, timestamps)

        # success_rateで間引き
        # output_df = window_check(output_df, th=0.8)
        # print(f"{output_df.shape} -> {output_remove_df.shape}")

        # 保存
        output_df.to_csv(f"{out_path}{csv_name}_{id}_processed.csv", index=False)

if __name__ == "__main__":
    """
    CSV_PATH：処理するcsvファイルのパス
    OUT_PATH: 出力先のパス
    """

    CSV_PATH = "./openface/source/multi/test.csv"
    OUT_PATH = "./openface/preprocessed/multi/"
    read_multi(CSV_PATH, OUT_PATH)