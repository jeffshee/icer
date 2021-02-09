from face_recognition.api import _raw_face_landmarks
from face_recognition.api import face_encoder
import dlib


def _to_dlib_full_object_detections(full_object_detection_list):
    output = dlib.full_object_detections()
    for item in full_object_detection_list:
        output.append(item)
    return output


def batch_face_encodings(batch_face_image, batch_known_face_locations=None, num_jitters=1, model="small"):
    if batch_known_face_locations is None:
        batch_known_face_locations = [None] * len(batch_face_image)
    if len(batch_face_image) != len(batch_known_face_locations):
        raise ValueError("len(batch_face_image) != len(batch_known_face_locations)")
    batch_size = len(batch_face_image)
    batch_raw_landmarks = [
        _raw_face_landmarks(face_image=batch_face_image[i],
                            face_locations=batch_known_face_locations[i],
                            model=model) for i in range(batch_size)]
    # Convert type for dlib
    batch_raw_landmarks = [_to_dlib_full_object_detections(item) for item in batch_raw_landmarks]

    # Output size: [batch_size, face_num, 128]
    return face_encoder.compute_face_descriptor(batch_face_image, batch_raw_landmarks, num_jitters)


def test():
    import face_recognition
    known_obama_image = face_recognition.load_image_file("obama.jpeg")
    known_biden_image = face_recognition.load_image_file("biden.jpeg")
    print(batch_face_encodings([known_obama_image, known_biden_image]))


if __name__ == "__main__":
    test()
