import os

import pandas as pd
from os.path import join

##说话人分类结果路径  情感路径
def match_speaker(diarization_result_path, emotion_dir, th_matching=0.0):
    df_diarization = pd.read_csv(diarization_result_path, encoding="shift_jis", header=0,
                                 usecols=["time(ms)", "speaker class"]) ##读取csv文件 第一行为列索引 返回times, speaker class中的数据
    matching_similarity_list = [[] for _ in range(len(df_diarization["speaker class"].unique()))]##有几个人建几个[]
    true_face_num = len([x for x in os.listdir(emotion_dir) if ".csv" and "result" in x])##实际人数
    # result0から順番にspeaker区切りごとの平均口開閉度を計算する
    for face_index in range(true_face_num):
        df_emotion = pd.read_csv(join(emotion_dir, "result{}.csv".format(face_index)),
                                 encoding="shift_jis", header=0, usecols=["time(ms)", "mouse opening"])
        # merge 2 data frames
        df_merged = pd.merge(df_emotion, df_diarization, on="time(ms)")
        # if face_index == 0:
        #     df_merged["speaker class"].plot(kind="line", figsize=(50, 5))
        #     plt.show()
        #     plt.close()
        df_merged.dropna(how="any")  # 欠損値を削除

        # 標準化 ##除去空值 补充值后进行标准化
        df_merged = df_merged[df_merged["mouse opening"] != 0]  # 口の開閉が取得できていない部分を削除する
        df_merged = df_merged[df_merged.diff().abs()["mouse opening"] != 0]  # 補完されている部分を削除する ##如果值与先前的值相同则删除
        df_merged["mouse opening"] = (df_merged["mouse opening"] - df_merged["mouse opening"].mean(skipna=True)) / \
                                     df_merged["mouse opening"].std(skipna=True)

        # # plot
        # df_merged["mouse opening"].plot(kind="line", figsize=(50, 5))
        # df_merged["speaker class"].plot(kind="line", figsize=(50, 5))
        # plt.show()
        # plt.close()

        # ある人のspeaker classごとのmouse openingの平均値の最大値を求める
        mean_mouse_opening_in_each_speaker_class_list = []
        for speaker_class in range(len(df_diarization["speaker class"].unique())): ##有几个对话人分几类
            df_merged_extracted1 = df_merged[df_merged["speaker class"] == speaker_class]
            mean_mouse_opening_in_each_speaker_class_list.append(  ##添加了每个人开口频率的平均值
                df_merged_extracted1["mouse opening"].mean() if not pd.isnull(
                    df_merged_extracted1["mouse opening"].mean()) else -100)
        max_mean_mouse_opening = max(mean_mouse_opening_in_each_speaker_class_list) ##计算平均值的最大值

        ##上面的循环代码的意思是在每个人的表情识别结果中计算每个人说话的时候的开口频率的最大值(例如在1号表情文件中,记录0-5号说话人的不同时段中开口频率的均值的最大值)

        # 平均口開閉度（類似度）を計算する
        for speaker_class in range(len(df_diarization["speaker class"].unique())):
            df_merged_extracted1 = df_merged[df_merged["speaker class"] == speaker_class]
            max_rate = (df_merged_extracted1["mouse opening"].mean() / max_mean_mouse_opening)  # 最大値からの変化率
            # matching_similarity_list[speaker_class].append(df_merged_extracted1["mouse opening"].mean() * max_rate)
            # matching_similarity_list[speaker_class].append(df_merged_extracted1["mouse opening"].mean())
            matching_similarity_list[speaker_class].append(
                df_merged_extracted1["mouse opening"].diff().abs().mean() * max_rate)

    for speaker_index, similarity in enumerate(matching_similarity_list):
        print("{}th speaker".format(speaker_index), similarity)

    matching_index_list = [
        matching_similarity.index(max(matching_similarity)) if max(matching_similarity) >= th_matching else -1 for
        matching_similarity in matching_similarity_list]
    print("matching_index_list", matching_index_list)
    return matching_index_list, true_face_num


if __name__ == "__main__":
    path_diarization = "E:/speaker_diarization_v2/result/2019-11-05_191031_Haga_Trim_1.0_0.5/"
    path_emo_rec = "E:/emotion_recognition/result/2019-11-05_191031_Haga_Trim_3_3_4people/"

    # path_diarization = "E:/speaker_diarization_v2/result/2019-09-24_Take01_0.5_0.5/"
    # path_emo_rec = "E:/emotion_recognition/result/2019-09-30_Take01_3_3_5people/"

    # path_emo_rec = "E:/emotion_recognition/result/2019-09-14_190130_3_3_8people/"
    # path_diarization = "E:/speaker_diarization_v2/result/2019-09-13_190130_2.0_0.5/"

    match_speaker(path_diarization, path_emo_rec, th_matching=0.0)
