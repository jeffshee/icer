import os
import time
import datetime

config = {
    # Run mode
    "run_capture_face": False,  # 動画から顔領域の切り出し
    "run_emotion_recognition": False,  # 切り出した顔画像の表情・頷き・口の開閉認識
    "run_transcript": False,  # Diarization・音声認識
    "run_overlay": True,  # 表情・頷き・発話情報を動画にまとめて可視化

    "capture_face_pt_path": None,  # Pickle to load when run_capture_face is false
    "tensorflow_log_level": str(2)
}

"""
Issue:
1. Python multiprocessing error 'ForkAwareLocal' object has no attribute 'connection'
https://stackoverflow.com/questions/60795412/python-multiprocessing-error-forkawarelocal-object-has-no-attribute-connectio
"""

os.environ["TF_CPP_MIN_LOG_LEVEL"] = config["tensorflow_log_level"]
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


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
def run_capture_face(**kwargs):
    from face_utils.capture_face import main
    return main(**kwargs)


@timeit
def run_emotion_recognition(**kwargs):
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
def run_transcript(**kwargs):
    from transcript import transcript
    transcript(**kwargs)


@timeit
def run_overlay(**kwargs):
    from gui_araki.qt_overlay import main_overlay
    main_overlay(**kwargs)


@timeit
def main(video_path: str, output_dir: str, audio_path_list: list, face_num=None, face_video_list=None):
    capture_face_result = None
    if config["run_capture_face"]:
        capture_face_dir = os.path.join(output_dir, "face_capture")
        os.makedirs(capture_face_dir, exist_ok=True)
        capture_face_result = run_capture_face(
            **{"video_path": video_path,
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
            **{"interpolated_result": capture_face_result,
               "video_path": video_path,
               "output_dir": emotion_dir
               }
        )

    if config["run_transcript"]:
        transcript_dir = os.path.join(output_dir, "transcript")
        os.makedirs(transcript_dir, exist_ok=True)
        run_transcript(
            **{"output_dir": transcript_dir,
               "audio_path_list": audio_path_list
               }
        )

    if config["run_overlay"]:
        emotion_dir = os.path.join(output_dir, "emotion")
        transcript_dir = os.path.join(output_dir, "transcript")
        diarization_dir = os.path.join(output_dir, "transcript/diarization")  # might be changed
        transcript_csv_path = os.path.join(transcript_dir, "transcript.csv")
        diarization_csv_path = transcript_csv_path
        emotion_csv_path_list = [
            os.path.join(emotion_dir, f"result{face_index}.csv") for face_index in range(face_num)
        ]
        run_overlay(
            **{"video_path": video_path,
               "emotion_dir": emotion_dir,
               "diarization_dir": diarization_dir,
               "transcript_csv_path": transcript_csv_path,
               "diarization_csv_path": diarization_csv_path,
               "emotion_csv_path_list": emotion_csv_path_list,
               }
        )


if __name__ == "__main__":
    # main_kwargs = {
    #     "video_path": "datasets/test/test_video.mp4",
    #     "output_dir": "output",
    #     "audio_path_list": ["datasets/test/test_voice_{:02d}.wav".format(i) for i in range(1, 7)],
    #     "face_num": 6,
    #     "face_video_list": ["datasets/test/reid/reid_{:02d}.mp4".format(i) for i in range(1, 7)]
    # }

    main_kwargs = {
        "video_path": "datasets/test2/200221_expt12_video.mp4",
        "output_dir": "output2",
        "audio_path_list": ["datasets/test2/200221_expt12_voice1{}.wav".format(i) for i in range(1, 4)],
        "face_num": 3,
        "face_video_list": ["datasets/test2/reid/{}.mp4".format(i) for i in range(0, 3)]
    }
    main(**main_kwargs)
