from openface.read_multi import read_multi
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # 入力データ読み込み（複数人のスライディングウィンドウcsv）
    input_1 = pd.read_csv("csv/multi_people_1_processed.csv")
    input_2 = pd.read_csv("csv/multi_people_2_processed.csv")
    input_3 = pd.read_csv("csv/multi_people_3_processed.csv")
    input_4 = pd.read_csv("csv/multi_people_4_processed.csv")

    # アノテーションデータ読み込み
    anno_1 = pd.read_csv("pos/anno_1.csv")
    anno_2 = pd.read_csv("pos/anno_2.csv")
    anno_3 = pd.read_csv("pos/anno_3.csv")
    anno_4 = pd.read_csv("pos/anno_4.csv")

    # モデル



    # 推定


    # 