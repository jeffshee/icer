import os
import time
import datetime

config = {
    # Run mode
    "run_capture_face": True,  # 動画から顔領域の切り出し
    "run_emotion_recognition": True,  # 切り出した顔画像の表情・頷き・口の開閉認識
    "run_transcript": True,  # Diarization・音声認識
    "run_overlay": True,  # 表情・頷き・発話情報を動画にまとめて可視化

    "capture_face_pt_path": None  # Pickle to load when run_capture_face is false
}


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        if "log_time" in kwargs:
            name = kwargs.get("log_name", method.__name__.upper())
            kwargs["log_time"][name] = int((te - ts) * 1000)
        else:
            print(f"{method.__name__} elapsed time: {datetime.timedelta(seconds=(te - ts))} (H:MM:SS)")
        return result

    return timed


@timeit
def run_capture_face(kwargs):
    from face_utils.capture_face import main
    return main(**kwargs)


@timeit
def run_emotion_recognition(kwargs):
    from face_utils.emotion_recognition import emotion_recognition_multiprocess
    from utils.video_utils import output_video_emotion_multiprocess
    # Emotion Recognition
    emotion_recognition_multiprocess(**kwargs)
    # Output Video
    capture_face_result = kwargs["interpolated_result"]
    video_path = kwargs["video_path"]
    video_filename = f"{os.path.basename(video_path)[:-4]}_emotion.avi"
    output_dir = kwargs["output_dir"]
    output_video_emotion_multiprocess(capture_face_result,
                                      sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir)]),
                                      video_path, output_path=os.path.join(output_dir, video_filename))


@timeit
def run_transcript(kwargs):
    from transcript import transcript
    transcript(**kwargs)


@timeit
def main(video_path: str, output_dir: str, audio_path_list: list, face_num=None, face_video_list=None):
    capture_face_result = None
    if config["run_capture_face"]:
        capture_face_dir = os.path.join(output_dir, "face_capture")
        os.makedirs(capture_face_dir, exist_ok=True)
        capture_face_result = run_capture_face(
            {"video_path": video_path,
             "output_dir": capture_face_dir,
             "face_num": face_num,
             "face_video_list": face_video_list
             }
        )
    else:
        import pickle
        if config["capture_face_pt_path"] is not None:
            with open(config["capture_face_pt_path"], "rb") as f:
                capture_face_result = pickle.load(f)

    if config["run_emotion_recognition"]:
        assert capture_face_result is not None
        emotion_dir = os.path.join(output_dir, "emotion")
        os.makedirs(emotion_dir, exist_ok=True)
        run_emotion_recognition(
            {"interpolated_result": capture_face_result,
             "video_path": video_path,
             "output_dir": emotion_dir
             }
        )

    if config["run_transcript"]:
        transcript_dir = os.path.join(output_dir, "transcript")
        os.makedirs(transcript_dir, exist_ok=True)
        run_transcript(
            {"output_dir": transcript_dir,
             "audio_path_list": audio_path_list
             }
        )


if __name__ == "__main__":
    main_kwargs = {
        "video_path": "datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23video.mp4",
        "output_dir": "output",
        "audio_path_list": [f"datasets/200225_芳賀先生_実験23/200225_芳賀先生_実験23voice{i}.mp4" for i in range(1, 7)],
        "face_num": 6,
        "face_video_list": [f"datasets/200225_芳賀先生_実験23/reid/untitled{i}.mp4" for i in range(1, 7)]
    }
    main(**main_kwargs)
