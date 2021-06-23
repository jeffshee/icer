from collections import defaultdict
import pandas as pd


"""
the face id (in case of multiple faces),
there is no guarantee that this is consistent across frames
in case of FaceLandmarkVidMulti, especially in longer sequences
"""


# To fix above issue
def match_face_id(df):
    # Check the consistency of face_id from face coordinates

    # get mean cordinates from Landmarks locations in 2D (x_0, x_1, ... x_66, x_67, y_0,...y_67)
    # ex. (x_0 + ... + x_67 / 68, y_0 + ... + y_67 / 68)
    return df


if __name__ == "__main__":
    csv_path = "/home/icer/Project/openface_dir/output_vid_dir"
    df = pd.read_csv(csv_path)
    new_df = match_face_id(df)