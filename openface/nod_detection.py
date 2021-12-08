import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import cv2
from typing import Tuple
import shutil
import os


#  ====== 前処理用 ======
def drop_na(df, axis: int = 0, subset: list=None):
    """
    欠損値のある行or列を削除
    axis=0: 行, axis=1: 列
    subset: 削除したいcolumn名のリスト
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    if subset is None:
        dropped_df = df.dropna(axis=axis, how="any")
    else:
        dropped_df = df.dropna(subset=subset)
    return dropped_df


def feature_selection(X, y=None, selected_bool_list: list=None):
    """
    分類に有用な特徴量を選択
    """
    from sklearn.ensemble import RandomForestRegressor
    from boruta import BorutaPy

    if (selected_bool_list is None) and (y is not None):
        # RandomForestRegressorでBorutaを実行
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)
        feat_selector.fit(X.values, y.values)

        # 選択された特徴量を確認
        selected_bool_list = feat_selector.support_
        print('選択された特徴量の数: %d' % np.sum(selected_bool_list))
        print(X.columns[selected_bool_list])

    X_selected = X.iloc[:, selected_bool_list]
    return X_selected, selected_bool_list


#  ====== 後処理用 ======
def output_csv(
    base_df: pd.core.frame.DataFrame,
    pred_result: np.ndarray,
    output_path: str="output_pred.csv",
    mode: str="test"
):
    """
    base_df: predに対応するDataFrame
    pred_result: 判別後の予測結果 (ndarray)
    """
    if mode == "test":
        in_columns = ['frame_in', 'frame_out', 'face_x', 'face_y', 'face_radius']
        out_columns = ['frame', 'face_x', 'face_y', 'face_radius', 'pred']
    else:
        in_columns = ['frame_in', 'frame_out', 'face_x', 'face_y', 'face_radius', 'label']
        out_columns = ['frame', 'label', 'face_x', 'face_y', 'face_radius', 'pred']

    base_df = base_df[in_columns]
    base_df = base_df.assign(pred=pred_result)

    out_df = pd.DataFrame(columns=out_columns)

    for i, (_, row) in enumerate(base_df.iterrows()):
        fin = row['frame_in'].astype(int)
        fout = row['frame_out'].astype(int)

        for frame in range(fin, fout + 1):
            face_x = row['face_x'].astype(int)
            face_y = row['face_y'].astype(int)
            face_radius = row['face_radius'].astype(int)

            if (out_df['frame'] == frame).sum() == 0:
                if mode == "test":
                    out_df = out_df.append({
                        'frame': frame,
                        'face_x': face_x,
                        'face_y': face_y,
                        'face_radius': face_radius,
                        'pred': row['pred'],
                    }, ignore_index=True)
                else:
                    out_df = out_df.append({
                        'frame': frame,
                        'face_x': face_x,
                        'face_y': face_y,
                        'face_radius': face_radius,
                        'label': row['label'],
                        'pred': row['pred'],
                    }, ignore_index=True)
            else:
                if row['pred'] == 1:
                    out_df.loc[out_df['frame'] == frame, 'pred'] += 1

    if output_path is not None:
        out_df.to_csv(output_path, index=False)

    return out_df


def get_video_capture(video_path: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(video_path)


def get_video_length(video_path: str) -> int:
    video_capture = get_video_capture(video_path)
    return int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


def get_frame_position(video_capture: cv2.VideoCapture) -> int:
    return int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))


def set_frame_position(video_capture: cv2.VideoCapture, position: int) -> int:
    return int(video_capture.set(cv2.CAP_PROP_POS_FRAMES, position))


def get_video_dimension(video_path: str) -> Tuple[int, int]:
    video_capture = get_video_capture(video_path)
    return int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


def get_video_framerate(video_path: str) -> float:
    video_capture = get_video_capture(video_path)
    return float(video_capture.get(cv2.CAP_PROP_FPS))


def output_video_nod_multi(nod_df, video_path: str, output_path: str = "output_video_nod"):
    tmp_video_path = os.path.dirname(video_path) + "/tmp_" + os.path.basename(video_path)
    shutil.copy(video_path, tmp_video_path)

    video_capture = get_video_capture(tmp_video_path)
    video_length = get_video_length(tmp_video_path)

    # 並列処理用
    # divided = video_length // parallel_num  # Frames per process
    # frame_range = [[i * divided, (i + 1) * divided] for i in range(parallel_num)]
    # frame_range[-1][1] = video_length - 1
    # start, end = frame_range[rank]

    frame_range = (0, video_length - 1)
    start, end = frame_range

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_path, fourcc, get_video_framerate(tmp_video_path),
                                   get_video_dimension(tmp_video_path))

    frame_index = start
    set_frame_position(video_capture, start)  # Move position
    while video_capture.isOpened() and get_frame_position(video_capture) in range(start, end + 1):
        ret, frame = video_capture.read()

    #     if (nod_df['frame'] == frame_index).sum() > 0:
    #         row = nod_df[nod_df['frame'] == frame_index]
    #         nod = row['pred'].values[0]
    #         texts = {0: "No", 1: "Nod", 2: "Nod!!!"}

    #         # when nod flag == 1 or 2 (1: weakly, 2: strongly)
    #         if nod > 0:
    #             mid = (row['face_x'].values[0], row['face_y'].values[0])
    #             radius = int(row['face_radius'].values[0])
    # #             frame = cv2.circle(frame, mid, radius=radius, color=(0, 0, 255), thickness=int(1 * nod))

    #             text = texts[nod]
    #             font = cv2.FONT_HERSHEY_PLAIN
    #             textsize = cv2.getTextSize(text, font, 1, 2)[0]
    #             coord = (int(mid[0] - (textsize[0] * 1.7)), int(mid[1] - (radius * 2)))
    #             frame = cv2.putText(frame, text, coord, font, 4, (0, 0, 255), 5, cv2.LINE_AA)

        if (nod_df['frame'] == frame_index).sum() > 0:
            row = nod_df[nod_df['frame'] == frame_index]
            nod = row['pred'].values[0]

            if nod > 1:
                mid = (row['face_x'].values[0], row['face_y'].values[0])
                radius = int(row['face_radius'].values[0])
                text = "Nod!!!"
                font = cv2.FONT_HERSHEY_PLAIN
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                coord = (int(mid[0] - (textsize[0] * 1.7)), int(mid[1] - (radius * 2)))
                frame = cv2.putText(frame, text, coord, font, 4, (0, 0, 255), 5, cv2.LINE_AA)

        video_writer.write(frame)
        frame_index += 1

    video_capture.release()
    video_writer.release()
    os.remove(tmp_video_path)


def output_video_nod_merge(nod_df, roi: tuple, video_path: str, output_path: str = "output_video_nod"):
    tmp_video_path = os.path.dirname(video_path) + "/tmp_" + os.path.basename(video_path)
    shutil.copy(video_path, tmp_video_path)

    video_capture = get_video_capture(tmp_video_path)
    video_length = get_video_length(tmp_video_path)

    frame_range = (0, video_length - 1)
    start, end = frame_range

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_path, fourcc, get_video_framerate(tmp_video_path),
                                   get_video_dimension(tmp_video_path))

    frame_index = start
    set_frame_position(video_capture, start)  # Move position
    while video_capture.isOpened() and get_frame_position(video_capture) in range(start, end + 1):
        ret, frame = video_capture.read()

        if (nod_df['frame'] == frame_index).sum() > 0:
            row = nod_df[nod_df['frame'] == frame_index]
            nod = row['pred'].values[0]

            if nod > 1:
                mid = (row['face_x'].values[0], row['face_y'].values[0])
                radius = int(row['face_radius'].values[0])
                text = "Nod!!!"
                font = cv2.FONT_HERSHEY_PLAIN
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                rel_coord = (int(mid[0] - (textsize[0] * 1.7)), int(mid[1] - (radius * 2)))
                coord = (int(roi[0] + rel_coord[0]), int(roi[1] + rel_coord[1]))
                frame = cv2.putText(frame, text, coord, font, 4, (0, 0, 255), 5, cv2.LINE_AA)

        video_writer.write(frame)
        frame_index += 1

    video_capture.release()
    video_writer.release()
    os.remove(tmp_video_path)


# ====== モデル ======
def eval(X, y, model, data_attrib: str="train"):
    pred = model.predict(X)
    accuracy = accuracy_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    print(f"{data_attrib}データに対する正解率： {accuracy: .2f}")
    print(f"{data_attrib}データに対する適合率： {precision: .2f}")
    print(f"{data_attrib}データに対する再現率： {recall: .2f}")
    cm = confusion_matrix(y, pred)
    print(cm)


def train(
    train_df,
    valid_df=None,
    random_state: int=0
):
    """
    頷き検出モデルの訓練

    train_df: trainデータ (pandas.DataFrame)
    valid_df: validationデータ (pandas.DataFrame)
    random_state: SVMカーネル用の乱数
    """
    print("====== Train ======")

    # train/validデータの前処理
    print(f"before drop_na: {len(train_df.columns)}")
    train_df = drop_na(train_df, axis=1)  # 欠損値のある列を削除
    print(f"after drop_na: {len(train_df.columns)}")

    X_train = train_df.iloc[:, 1:-5]  # 特徴量を取得
    y_train = train_df['label'].astype(int)  # ラベルデータを取得

    X_train, _ = feature_selection(X=X_train, y=y_train)  # 特徴量を選択
    feature_columns = X_train.columns

    # データの標準化処理
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)

    # kernel SVMのインスタンスを生成
    model = SVC(kernel='rbf', random_state=random_state)

    # モデルの学習
    model.fit(X_train, y_train)

    # trainデータに対する評価
    eval(X_train, y_train, model, data_attrib="train")

    # validデータに対する評価
    if valid_df is not None:
        # t前処理
        valid_df = drop_na(valid_df, axis=0)  # 欠損のある行を削除
        y_valid = valid_df['label'].astype(int)  # ラベルデータを取得
        X_valid = valid_df.loc[:, feature_columns]  # 訓練に用いた特徴量のみ抽出
        X_valid = sc.transform(X_valid)  # 標準化処理
        eval(X_valid, y_valid, model, data_attrib="valid")

    return model, feature_columns, sc


def test_single(
    test_df,
    model,
    sc,
    feature_columns,
    video_path: str,
    output_csv_dir: str,
    output_vid_dir: str,
    suffix: str,
):
    print("====== Test ======")
    # testデータに対する前処理
    test_df = drop_na(test_df, axis=0)  # 欠損のある行を削除
    X_test = test_df.loc[:, feature_columns]  # 訓練に用いた特徴量のみ抽出
    X_test = sc.transform(X_test)  # 標準化処理

    pred_test = model.predict(X_test)
    output_csv_path = os.path.join(output_csv_dir, f"pred_{suffix}.csv")
    output_csv(test_df, pred_test, output_path=output_csv_path)  # 予測結果をCSVに書き出し

    nod_df = pd.read_csv(output_csv_path)
    output_path = os.path.join(output_vid_dir, "pred_test_{suffix}.mp4")
    output_video_nod_multi(nod_df, video_path, output_path)


def test_multi(
    test_dfs,
    rois,
    model,
    sc,
    feature_columns,
    video_path: str,
    output_csv_dir: str,
    output_vid_dir: str,
):
    print("====== Test ======")

    for i, (test_df, roi) in enumerate(zip(test_dfs, rois)):
        print(f"Processing: {i}")

        # testデータに対する前処理
        test_df = drop_na(test_df, axis=0)  # 欠損のある行を削除
        X_test = test_df.loc[:, feature_columns]  # 訓練に用いた特徴量のみ抽出
        X_test = sc.transform(X_test)  # 標準化処理

        pred_test = model.predict(X_test)

        output_csv_path = os.path.join(output_csv_dir, f"pred_reid{i}.csv")
        output_csv(test_df, pred_test, output_path=output_csv_path)  # 予測結果をCSVに書き出し

        nod_df = pd.read_csv(output_csv_path)
        output_path = os.path.join(output_vid_dir, "pred_test.mp4")
        if i == 0:
            output_video_nod_merge(nod_df, roi, video_path, output_path)
        else:
            output_video_nod_merge(nod_df, roi, output_path, output_path)


if __name__ == '__main__':
    # ===== Train ===== #
    trvl_csv_path = "/home/icer/Project/icer/openface/data/nod_detection_train_data.csv"
    trvl_df = pd.read_csv(trvl_csv_path, engine='python')
    model, feature_columns, sc = train(trvl_df)

    # # ===== Test (single) ===== #
    # name_list = ["reid0", "reid1", "reid2", "reid3", "reid4"]

    # for name in name_list:
    #     test_csv_path = f"/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_0/reid/processed/{name}_processed.csv"
    #     test_df = pd.read_csv(test_csv_path, engine='python')

    #     video_path = f"/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_0/reid/{name}.mp4"
    #     output_dir = "/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_0/reid/processed"
    #     suffix = name
    #     test_single(test_df, model, sc, feature_columns, video_path, output_csv_dir=output_dir, output_vid_dir=output_dir, suffix=suffix)

    # ===== Test (multi) ===== #
    name_list = ["reid0", "reid1", "reid2", "reid3", "reid4"]
    roi_list = [(407, 661, 389, 388), (802, 683, 350, 347), (1333, 705, 389, 390), (1725, 724, 401, 383), (2176, 744, 316, 335)]

    test_dfs = []
    for name in name_list:
        test_csv_path = f"/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_0/reid/processed/{name}_processed.csv"
        test_df = pd.read_csv(test_csv_path, engine='python')
        test_dfs.append(test_df)

    video_path = "/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_0/reid/calibrate.mp4"
    output_dir = "/home/icer/Project/openface_dir2/2020-01-23_Take01_copycopy_3_3_5people/split_video_0/reid/processed"
    test_multi(test_dfs, roi_list, model, sc, feature_columns, video_path, output_csv_dir=output_dir, output_vid_dir=output_dir)
