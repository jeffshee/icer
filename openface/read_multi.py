import pandas as pd
import numpy as np
import math
import os

from feature_engineering import feature_grouping, get_face_data, get_timedata, remove_low_confidence_rows


def match_face_id(face_coord, face_id_dict):
    """
    face_coord (tuple): 顔の座標
    face_id_dict (dict): # 基準となる各idの顔座標
    """
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


def divide_faces(df, csv_name, dir_path, id_total=4):
    """
    人（new_face_id）を見分けて分割
    """
    # 抽出
    _df = df[["frame", "face_id", "timestamp", "confidence", "success", "eye_lmk_x_25", "eye_lmk_y_25", "eye_lmk_x_53", "eye_lmk_y_53"]]
    new_df = _df.copy()

    # confidenceの低い行を削除
    new_df = remove_low_confidence_rows(new_df)

    # 眉間の座標
    new_df.loc[:, ['face_coord_x']] = (new_df['eye_lmk_x_25'] + new_df['eye_lmk_x_53']) / 2
    new_df.loc[:, ['face_coord_y']] = (new_df['eye_lmk_y_25'] + new_df['eye_lmk_y_53']) / 2

    # 基準となる各idの顔座標を取得
    face_id_dict = {}
    for frame_idx in range(1, new_df.tail(1)['frame'].item() + 1):  # 1フレームから最終フレームまで
        frame_rows = new_df[(new_df['frame'] == frame_idx)]  # frame_idxの行をすべて抽出
        if (frame_rows['success'] == 1).sum() == id_total:  # 検出に成功している行数がid数あるとき
            for index, row in frame_rows.iterrows():  # 一行ずつ取り出し
                id = int(row['face_id'])
                face_id_dict[id] = (row['face_coord_x'], row['face_coord_y'])
            print(f"frame-{frame_idx}: {face_id_dict}")
            break

    # 不要になったカラムを削除
    new_df = new_df.drop("confidence", axis=1)
    new_df = new_df.drop("success", axis=1)

    counter = 0
    total_dist = 0
    for index, row in new_df.iterrows():
        face_coord = (row['face_coord_x'], row['face_coord_y'])
        face_id = row['face_id']

        new_face_id, dist, coord_diff = match_face_id(face_coord, face_id_dict)
        new_df.loc[index, 'new_face_id'] = int(new_face_id)

        if face_id != new_face_id:
            print(f"[{index}] not matched!: old_id: {face_id}, new_id: {new_face_id} (dist: {dist}, diff: {coord_diff})")
            print(row)
            counter += 1
            total_dist += dist
            new_df.loc[index, 'face_id_update'] = True

    print(f"\n====\nmiss match total: {counter}")
    print(f"average dist: {dist / counter}")

    # dfにnew_face_idカラムを追加
    df["new_face_id"] = new_df["new_face_id"]
    df = df.dropna(axis=0)  # nanのある行を削除

    # idごとにcsv分割
    df_dict = {}
    id_list = []
    for id in np.unique(df["new_face_id"]):
        id = int(id)
        tmp_df = df[df["new_face_id"] == id]
        tmp_df.to_csv(f"{dir_path}{csv_name}_{str(id)}.csv", index=False)

        # 辞書型にdataframeごと保存
        df_dict[str(id)] = tmp_df
        id_list.append(str(id))
    return df_dict, id_list


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


def read_multi(csv_path, dir_path):
    """
    複数人の動画からのopenfaceデータ(csv)を人ごとにわけてスライディングウィンドウし、
    別々のcsvファイルにしてdir_pathに保存
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
    print(input_df)

    # 人（new_face_id）を見分けてcsvを分割
    df_dict, id_list = divide_faces(input_df, csv_name, dir_path)

    # idごとに特徴量抽出
    feature_dict = {}
    columns = ["mean", "std", "mad", "max", "min", "energy", "entropy", "iqr", "range", "skewness", "kurtosis"]

    for id in id_list:
        # 該当のidのdataframeを取得
        individ_df = df_dict[id]

        # スライディング法により特徴量抽出
        outputs = feature_grouping(individ_df)
        feature_dict[id] = outputs

        # 時間情報（フレーム, タイムスタンプ）を各ウィンドウに追加
        frames, timestamps = get_timedata(individ_df)

        # 顔情報（顔のx, y座標, 半径）を各ウィンドウに追加
        faces = get_face_data(individ_df)

        # pandasに変換
        output_df = trans_pandas(
            data=outputs,
            columns_ori=columns,
            frames=frames,
            timestamps=timestamps,
            faces=faces
        )

        # 保存
        output_df.to_csv(f"{dir_path}{csv_name}_{id}_processed.csv", index=False)


if __name__ == "__main__":

    # CSV_PATH = "/home/icer/Project/openface_dir/multi_people_data/multi_people.csv"
    # DIR_PATH = "/home/icer/Project/openface_dir/multi_people_data/processed_csv/"
    # read_multi(CSV_PATH, DIR_PATH)

    CSV_PATH = "/home/icer/Project/openface_dir2/2019-11-01_191031_Haga_3_3_4people/split_video_0/processed/output.csv"
    DIR_PATH = "/home/icer/Project/openface_dir2/2019-11-01_191031_Haga_3_3_4people/split_video_0/processed/"
    read_multi(CSV_PATH, DIR_PATH)
