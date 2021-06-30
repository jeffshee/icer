from collections import defaultdict
import os
import pandas as pd


"""
the face id (in case of multiple faces),
there is no guarantee that this is consistent across frames
in case of FaceLandmarkVidMulti, especially in longer sequences
"""


class FormatOpenface(object):
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        assert os.path.isfile(self.csv_path), f"{self.csv_path} does not exists"
        self.df = pd.read_csv(self.csv_path)

    def get_df(self):
        return self.df

    def preprocess_df(self, df):
        """
        Warning: columns' name other than [frame] has blank infront
        """
        # get rid of space infront
        columns_dict = {}
        for column_name in df.columns:
            columns_dict[column_name] = column_name.lstrip()
        columns_dict
        df = df.rename(columns=columns_dict)

        # remove unnecessary columns
        _df = df[["frame", "face_id", "timestamp", "confidence", "success", "eye_lmk_x_25", "eye_lmk_y_25", "eye_lmk_x_53", "eye_lmk_y_53"]]
        new_df = _df.copy()

        # get gace coordinate to fix face_id
        new_df.loc[:, ['face_coord_x']] = (new_df[' eye_lmk_x_25'] + new_df[' eye_lmk_x_53']) / 2
        new_df.loc[:, ['face_coord_y']] = (new_df[' eye_lmk_y_25'] + new_df[' eye_lmk_y_53']) / 2

        return new_df

    def get_anker_face_id(self, df):
        face_id_dict = {}
        for index, row in df[(df['frame'] == 1)].iterrows():
            face_id_dict[index] = (row['face_coord_x'], row['face_coord_y'])
        return face_id_dict

    # To fix above issue
    def match_face_id(self, df):
        # Check the consistency of face_id from face coordinates

        # get mean cordinates from Landmarks locations in 2D (x_0, x_1, ... x_66, x_67, y_0,...y_67)
        # ex. (x_0 + ... + x_67 / 68, y_0 + ... + y_67 / 68)
        return df


if __name__ == "__main__":
    csv_path = "/home/icer/Project/openface_dir/output_vid_dir"
    df = pd.read_csv(csv_path)
    new_df = match_face_id(df)