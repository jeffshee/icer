from matching import Player
from matching.games import StableMarriage
import numpy as np
from face_recognition import face_encodings, face_distance
from collections import defaultdict

"""
TODO
1. add "valid/invalid" flag (done)
    valid: box detected by detect_face
    invalid: not detected by detect_face, value is taken from previous frame
2. to capture_face_ng
3. interpolation
"""


class frame_matcer:
    def __init__(self, max_num):
        self.max_num = max_num
        self.dict = defaultdict()
        self.flag = defaultdict()
        for i in range(max_num):
            self.dict[i] = []
            self.flag[i] = []

    def calculate_box_midpoint(self, top, right, bottom, left):
        midpoint = np.array([(bottom + top) / 2, (right + left) / 2])
        return midpoint

    def calculate_box_distance(self, box1, box2):
        if box1 is None or box2 is None:
            return np.inf
        return np.linalg.norm(self.calculate_box_midpoint(*box1) - self.calculate_box_midpoint(*box2))

    def match(self, img_frame_1, box_list_1, img_frame_2, box_list_2):
        # Convert color for face_recognition
        img_frame_1 = img_frame_1[:, :, ::-1]
        img_frame_2 = img_frame_2[:, :, ::-1]

        # Face encodings
        box_list_1_encodings = face_encodings(img_frame_1, known_face_locations=box_list_1)
        box_list_2_encodings = face_encodings(img_frame_2, known_face_locations=box_list_2)

        # Append None for not detected box
        last_box_list = list(zip(*self.dict.values()))
        if len(last_box_list) != 0:
            box_list_1 = last_box_list[-1]
        else:
            box_list_1 += [None] * (self.max_num - len(box_list_1))
        box_list_2 += [None] * (self.max_num - len(box_list_2))

        # Create player object
        box_list_1_players = np.array([Player(box1) for box1 in box_list_1])
        box_list_2_players = np.array([Player(box2) for box2 in box_list_2])

        # Calculate preference for each element of box_1 to box_2, based on box distance
        for i, box1 in enumerate(box_list_1):
            distances = np.array([self.calculate_box_distance(box1, box2) for box2 in box_list_2])
            argsort = distances.argsort()
            preference = box_list_2_players[argsort]
            box_list_1_players[i].set_prefs(list(preference))

        # Calculate preference for each element of box_2 to box_1, based on face embedding distance
        for i in range(self.max_num):
            if i < len(box_list_2_encodings):
                box2_encodings = box_list_2_encodings[i]
                distances = face_distance(box_list_1_encodings, box2_encodings)
                distances = np.append(distances, [np.inf] * (self.max_num - len(box_list_1_encodings)))
            else:
                distances = np.array([np.inf] * self.max_num)
            argsort = distances.argsort()
            preference = box_list_1_players[argsort]
            box_list_2_players[i].set_prefs(list(preference))

        game = StableMarriage(list(box_list_1_players), list(box_list_2_players))
        solution = game.solve()
        keys = solution.keys()

        for key in keys:
            for i, dict_item in enumerate(self.dict.values()):
                if len(dict_item) == 0:
                    # First time
                    dict_item.extend([key.name, solution[key].name])
                    self.flag[i].extend([True, True])
                    break
                elif dict_item[-1] == key.name:
                    next_item = solution[key].name
                    if next_item is None:
                        dict_item.append(key.name)
                        self.flag[i].append([False])
                    else:
                        dict_item.append(next_item)
                        self.flag[i].append([True])
                    break

    def get_result(self):
        return self.dict, self.flag


import pickle
import cv2

video_capture = cv2.VideoCapture("test.mp4")
matcher = frame_matcer(3)


def set_frame_position(video_capture, position):
    return int(video_capture.set(cv2.CAP_PROP_POS_FRAMES, position))


with open("detect_face.pt", "rb") as f:
    result = pickle.load(f)

    for r1, r2 in zip(result, result[1:]):
        frame_number_1, face_boxes1 = r1[0], r1[1:]
        frame_number_2, face_boxes2 = r2[0], r2[1:]
        set_frame_position(video_capture, frame_number_1)
        _, img_frame_1 = video_capture.read()
        set_frame_position(video_capture, frame_number_2)
        _, img_frame_2 = video_capture.read()
        matcher.match(img_frame_1, face_boxes1, img_frame_2, face_boxes2)

    print(matcher.get_result())
