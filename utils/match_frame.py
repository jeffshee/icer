from matching import Player
from matching.games import StableMarriage
import numpy as np
from collections import defaultdict

"""
TODO
1. add "valid/invalid" flag (done)
    valid: box detected by detect_face
    invalid: not detected by detect_face, value is taken from previous frame
2. to capture_face_ng 
3. interpolation
"""


def calculate_box_midpoint(top, right, bottom, left):
    midpoint = np.array([(bottom + top) / 2, (right + left) / 2])
    return midpoint


def calculate_box_distance(box1, box2):
    if box1 is None or box2 is None:
        return np.inf
    return np.linalg.norm(calculate_box_midpoint(*box1) - calculate_box_midpoint(*box2))


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty(0)

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


class FrameMatcher:
    def __init__(self, max_face_num):
        self.max_face_num = max_face_num
        self.frame_number = defaultdict()
        self.dict = defaultdict()
        self.flag = defaultdict()
        for i in range(max_face_num):
            self.frame_number[i] = []
            self.dict[i] = []
            self.flag[i] = []

    def match(self, frame_number_1, box_list_1, box_list_1_encodings, frame_number_2, box_list_2, box_list_2_encodings):
        accum_box_list = list(zip(*self.dict.values()))
        if len(accum_box_list) > 0:
            # Set box_list_1 as last_box_list, should not contain None
            last_box_list = accum_box_list[-1]
            box_list_1 = last_box_list
        else:
            # First time
            # Append None for not detected box
            box_list_1 += [None] * (self.max_face_num - len(box_list_1))

        # Append None for not detected box
        box_list_2 += [None] * (self.max_face_num - len(box_list_2))

        # Create player object
        box_list_1_players = np.array([Player(box1) for box1 in box_list_1])
        box_list_2_players = np.array([Player(box2) for box2 in box_list_2])

        # Calculate preference for each element of box_1 to box_2, based on box distance
        for i, box1 in enumerate(box_list_1):
            distances = np.array([calculate_box_distance(box1, box2) for box2 in box_list_2])
            argsort = distances.argsort()
            preference = box_list_2_players[argsort]
            box_list_1_players[i].set_prefs(list(preference))

        # Calculate preference for each element of box_2 to box_1, based on face embedding distance
        for i in range(self.max_face_num):
            if i < len(box_list_2_encodings):
                box2_encodings = box_list_2_encodings[i]
                distances = face_distance(box_list_1_encodings, box2_encodings)
                distances = np.append(distances, [np.inf] * (self.max_face_num - len(box_list_1_encodings)))
            else:
                distances = np.array([np.inf] * self.max_face_num)
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
                    self.frame_number[i].extend([frame_number_1, frame_number_2])
                    break
                elif dict_item[-1] == key.name: # the last item
                    # if len(self.frame_number[0])==44:
                    #     print("dd")
                    next_item = solution[key].name
                    self.frame_number[i].append(frame_number_2)
                    if next_item is None:
                        dict_item.append(key.name)
                        self.flag[i].append(False)
                    else:
                        dict_item.append(next_item)
                        self.flag[i].append(True)
                    break

    def get_result(self):
        return self.dict, self.flag


def test():
    import pickle

    matcher = FrameMatcher(max_face_num=3)

    with open("detect_face.pt", "rb") as f:
        result = pickle.load(f)[5:]
        c = 0
        for r1, r2 in zip(result, result[1:]):
            matcher.match(r1["frame_number"], r1["face_locations"], r1["face_encodings"], r2["frame_number"], r2["face_locations"],
                          r2["face_encodings"])
            print(r1["frame_number"], r1["face_locations"], r2["frame_number"], r2["face_locations"])
            # print(c)
            c+=1

        print(matcher.get_result())
        print(len(matcher.get_result()[0][0]))


if __name__ == "__main__":
    test()
