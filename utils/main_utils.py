import os
import shutil

import pandas as pd


def cleanup_directory(name):
    if os.path.isdir(name):
        shutil.rmtree(name)
    os.makedirs(name)


def csv_to_text(csv_path, output_path, specific_column, specific_index=None):
    df_csv = pd.read_csv(csv_path, encoding="utf_8_sig", header=0, index_col=False)
    if specific_index:
        for i in df_csv[specific_index].unique():
            df_csv_extracted = df_csv[df_csv[specific_index] == i]
            df_csv_extracted.to_csv(output_path.replace(".txt", "_{}.txt".format(i)), sep=' ', index=False,
                                    header=False, columns=specific_column, encoding="utf_8_sig")
    df_csv.to_csv(output_path, sep=' ', index=False, header=False, columns=specific_column, encoding="utf_8_sig")


def concat_csv(input_csv_list, result_path):
    df_concat = pd.read_csv(input_csv_list[0])

    for i in range(1, len(input_csv_list)):
        df_part = pd.read_csv(input_csv_list[i])
        df_concat = pd.concat([df_concat, df_part], ignore_index=True)

    df_concat.to_csv(result_path, index=False, encoding="utf-8")
