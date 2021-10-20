import numpy as np
import pandas as pd
from pandas import Series

# data = [[55197, [963, 3281, 1077, 3168], True],
#         [55197, [1023, 1618, 1102, 1539], True],
#         [55197, [1032, 2694, 1146, 2580], True],
#         [55197, [1063, 1050, 1142, 971], True]]
#
# data2 = []
# for i1, i2, i3 in data:
#     data2.append(dict(frame_number=i1, location=i2, is_detected=i3))
#
# # df_face = pd.DataFrame(data, columns=["frame_number", "location", "is_detected"])
# # print(df_face)
#
# df_face2 = pd.DataFrame(data2)
# df_face2 = df_face2.append([dict(frame_number=200, location=[22, 22, 22, 22], is_detected=False),
#                             dict(frame_number=200, location=[22, 22, 22, 22], is_detected=False)])
# print(df_face2)
#
# print(df_face2[df_face2["frame_number"] == 55197])


class FaceDataFrameWrapper:
    def __init__(self):
        self.face_df = pd.DataFrame(columns=["frame_number", "location", "is_detected", "encoding"])

    def append_array(self, array: np.ndarray):
        self.face_df = self.face_df.append(array)

    def append(self, frame_number, location, is_detected, encoding):
        self.face_df = self.face_df.append(
            [dict(frame_number=frame_number, location=location, is_detected=is_detected, encoding=encoding)])

    def write_csv(self, output_path):
        self.face_df.to_csv(output_path)

    def read_csv(self, csv_path):
        self.face_df = pd.read_csv(csv_path)

    def get_data(self, frame_number):
        return self.face_df[self.face_df["frame_number"] == frame_number]

    def concat(self, wrappers):
        self.face_df = pd.concat([wrapper.face_df for wrapper in wrappers])

