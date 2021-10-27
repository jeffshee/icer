import numpy as np
import pandas as pd

from face_utils.face import Face


class FaceDataFrameWrapper:
    def __init__(self, csv_path=None):
        self.face_df = None
        if csv_path:
            self.read_csv(csv_path)
        else:
            self.face_df = pd.DataFrame(columns=["frame_number", "location", "is_detected", "encoding"])

    def append_array(self, array: np.ndarray):
        self.face_df = self.face_df.append(array, ignore_index=True)

    def append(self, frame_number, location: np.ndarray, encoding: np.ndarray, is_detected):
        self.face_df = self.face_df.append(
            [dict(frame_number=frame_number, location=location, is_detected=is_detected, encoding=encoding)],
            ignore_index=True)

    def write_csv(self, output_path):
        self.face_df.to_csv(output_path)

    def read_csv(self, csv_path):
        def converter(instr):
            return np.fromstring(instr[1:-1], sep=' ')

        def converter_int(instr):
            return np.fromstring(instr[1:-1], sep=' ', dtype=int)

        self.face_df = pd.read_csv(csv_path, converters={"location": converter_int, "encoding": converter}, index_col=0)

    def get_data(self, frame_number):
        return self.face_df[self.face_df["frame_number"] == frame_number]

    def concat(self, wrappers):
        self.face_df = pd.concat([wrapper.face_df for wrapper in wrappers], ignore_index=True)

    def get_old_format(self):
        frame_numbers = np.unique(self.face_df["frame_number"])
        ret = []
        for frame_number in frame_numbers:
            ret_frame = []
            data = self.get_data(frame_number)
            for _, row in data.iterrows():
                location = row["location"]
                encoding = row["encoding"]
                is_detected = row["is_detected"]
                ret_frame.append(Face(frame_number, location, encoding, is_detected))
            ret.append(ret_frame)
        return ret


# dummy = FaceDataFrameWrapper("../output/exp22/face_capture/result0.csv")
# old = dummy.get_old_format()
# print(old)

# import os
# # Combine CSV
# face_df_wrapper = FaceDataFrameWrapper()
# csv_path_list = ["../output/exp22/face_capture/result0.csv", "../output/exp22/face_capture/result0.csv"]
# face_df_wrapper.concat([FaceDataFrameWrapper(csv_path) for csv_path in csv_path_list])
# face_df_wrapper.write_csv("test.csv")