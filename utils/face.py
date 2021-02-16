import numpy as np
import uuid


def calculate_box_midpoint(top, right, bottom, left):
    midpoint = np.array([(bottom + top) / 2, (right + left) / 2])
    return midpoint


class Face:
    def __init__(self, frame_number: int, location: np.array, encoding: np.array, is_detected=True):
        self.__id = uuid.uuid4()
        self.frame_number = frame_number
        self.location = location
        self.encoding = encoding
        self.is_detected = is_detected

    def __repr__(self):
        return "\n<Face frame_number:%s location:%s is_detected:%s>" % (
            self.frame_number, self.location, self.is_detected)

    def __str__(self):
        return "frame_number:%s location:%s is_detected:%s" % (self.frame_number, self.location, self.is_detected)

    def box_distance(self, face_to_compare):
        if face_to_compare is None or self.location is None or face_to_compare.location is None:
            return np.inf
        return np.linalg.norm(
            calculate_box_midpoint(*self.location) - calculate_box_midpoint(*face_to_compare.location))

    def face_distance(self, face_to_compare):
        if face_to_compare is None or self.encoding is None or face_to_compare.encoding is None:
            return np.inf
        return np.linalg.norm(self.encoding - face_to_compare.encoding)

    def __eq__(self, other):
        if isinstance(other, Face):
            return self.frame_number == other.frame_number and self.location == other.location
        return NotImplemented

    def __hash__(self):
        return hash(self.__id)
