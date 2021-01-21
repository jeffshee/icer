from matching import Player
from matching.games import StableMarriage
import numpy as np
from collections import defaultdict

from utils.face import Face


class FrameMatcher:
    def __init__(self, max_face_num):
        self.max_face_num = max_face_num
        self.is_first_round = True
        self.result = defaultdict()

    def match(self, face_list_1, face_list_2):
        if len(face_list_1) == 0 or len(face_list_2) == 0:
            raise ValueError("len(face_list_1) == 0 or len(face_list_2) == 0")
        frame_number_1, frame_number_2 = face_list_1[0].frame_number, face_list_2[0].frame_number

        if self.is_first_round:
            # Initialize
            # Fill empty with placeholder
            face_list_1 += [Face(frame_number_1, None, None, False)] * (self.max_face_num - len(face_list_1))
            for i in range(self.max_face_num):
                self.result[i] = [face_list_1[i]]
            self.is_first_round = False
        else:
            # Use last result instead
            face_list_1 = [self.result[i][-1] for i in range(self.max_face_num)]

        # Fill empty with placeholder
        face_list_2 += [Face(frame_number_2, None, None, False)] * (self.max_face_num - len(face_list_2))

        # Create player object
        face_list_1_players = np.array([Player(face) for face in face_list_1])
        face_list_2_players = np.array([Player(face) for face in face_list_2])

        # Calculate preference of face_1 -> face_2, based on box distance
        for i in range(self.max_face_num):
            face_1 = face_list_1[i]
            distances = np.array([face_1.box_distance(face_2) for face_2 in face_list_2])
            argsort = distances.argsort()
            preference = face_list_2_players[argsort]
            face_list_1_players[i].set_prefs(list(preference))

        # Calculate preference of face_2 -> face_1, based on face distance
        for i in range(self.max_face_num):
            face_2 = face_list_2[i]
            distances = np.array([face_2.face_distance(face_1) for face_1 in face_list_1])
            argsort = distances.argsort()
            preference = face_list_1_players[argsort]
            face_list_2_players[i].set_prefs(list(preference))

        game = StableMarriage(list(face_list_1_players), list(face_list_2_players))
        solution = game.solve()  # {(a:A),(b:B),(c:C),}
        keys = solution.keys()  # [a,b,c]

        for i in range(self.max_face_num):
            last_result = self.result[i][-1]  # get last result of person i
            next_result = None
            for key in keys:
                if key.name == last_result:
                    next_result = solution[key].name

            if not next_result.is_detected:
                # If next result is a placeholder, replace the location and encoding with last result
                next_result.location = last_result.location
                next_result.encoding = last_result.encoding
            self.result[i].append(next_result)

    def get_result(self):
        return self.result
