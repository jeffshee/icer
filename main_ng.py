import datetime
import os
import sys
import time

from PyQt5.QtWidgets import QApplication

from gui.calibrate import adjust_offset_dialog
from gui.cropper import select_roi_dialog
from gui.dialogs import get_face_num

config = {
    # Run mode
    "run_capture_face": True,  # 動画から顔領域の切り出し
    "run_emotion_recognition": True,  # 切り出した顔画像の表情・頷き・口の開閉認識
    "run_transcript": True,  # Diarization・音声認識
    "run_overlay": True,  # 表情・頷き・発話情報を動画にまとめて可視化

    "capture_face_pt_path": None,  # Pickle to load when run_capture_face is false
    "tensorflow_log_level": str(2)
}

"""
Issue:
1. Python multiprocessing error 'ForkAwareLocal' object has no attribute 'connection'
https://stackoverflow.com/questions/60795412/python-multiprocessing-error-forkawarelocal-object-has-no-attribute-connectio
Perhaps you were using Pycharm with Run with Python console option checked in Run/Debug Configuration. Try uncheck that.
2. Multiprocessing fail
GPU#1: 100%|██████████| 20624/20624 [15:31<00:00, 28.07it/s]Traceback (most recent call last):
  File "/home/jeffshee/Developer/icer/main_ng.py", line 216, in <module>
    main(**main_kwargs)
  File "/home/jeffshee/Developer/icer/main_ng.py", line 42, in timed
    result = method(*args, **kwargs)
  File "/home/jeffshee/Developer/icer/main_ng.py", line 123, in main
    capture_face_result = run_capture_face(**kwargs)
  File "/home/jeffshee/Developer/icer/main_ng.py", line 42, in timed
    result = method(*args, **kwargs)
  File "/home/jeffshee/Developer/icer/main_ng.py", line 57, in run_capture_face
    return main(**kwargs)
  File "/home/jeffshee/Developer/icer/face_utils/capture_face.py", line 340, in main
    result = detect_face_multiprocess(video_path, roi=roi, offset=offset, output_dir=output_dir)
  File "/home/jeffshee/Developer/icer/face_utils/capture_face.py", line 224, in detect_face_multiprocess
    combined.extend(return_dict[i])
  File "<string>", line 2, in __getitem__
  File "/home/jeffshee/anaconda3/envs/icer/lib/python3.8/multiprocessing/managers.py", line 850, in _callmethod
    raise convert_to_error(kind, result)
KeyError: 0

Perhaps write the result to multiple csv is better
"""

os.environ["TF_CPP_MIN_LOG_LEVEL"] = config["tensorflow_log_level"]
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from utils.audio_utils import set_audio


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
    temp_filename = f"{os.path.basename(video_path)[:-4]}_temp.avi"
    output_dir = kwargs["output_dir"]
    offset = kwargs["offset"]

    output_kwargs = dict(interpolated_result=capture_face_result,
                         emotion_csv_path_list=sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir)]),
                         video_path=video_path,
                         output_path=os.path.join(output_dir, temp_filename),
                         offset=offset
                         )
    output_video_emotion_multiprocess(**output_kwargs)
    set_audio(os.path.join(output_dir, temp_filename), video_path, os.path.join(output_dir, video_filename))
    # Remove temp
    os.remove(temp_filename)


@timeit
def run_transcript(**kwargs):
    from s2t_utils.transcript import transcript
    transcript(**kwargs)


def run_overlay(**kwargs):
    from gui.qt_overlay import main_overlay
    # TODO probably need to run as another process
    main_overlay(**kwargs)


@timeit
def main(video_path: str, output_dir: str, audio_path_list: list, face_num=None, face_video_list=None):
    app = QApplication(sys.argv)

    # Preprocess
    offset = 0
    roi = None
    if config["run_capture_face"] or config["run_emotion_recognition"]:
        offset = adjust_offset_dialog(video_path)
        if not face_num:
            face_num = get_face_num()
    if config["run_capture_face"]:
        roi = select_roi_dialog(video_path, offset, window_title="顔検出を行う領域の指定")

    capture_face_result = None
    if config["run_capture_face"]:
        capture_face_dir = os.path.join(output_dir, "face_capture")
        os.makedirs(capture_face_dir, exist_ok=True)
        kwargs = dict(video_path=video_path,
                      output_dir=capture_face_dir,
                      roi=roi,
                      offset=offset,
                      face_num=face_num,
                      face_video_list=face_video_list,
                      )
        capture_face_result = run_capture_face(**kwargs)
    else:
        import pickle
        if config["capture_face_pt_path"]:
            with open(config["capture_face_pt_path"], "rb") as f:
                capture_face_result = pickle.load(f)

    if config["run_emotion_recognition"]:
        assert capture_face_result
        emotion_dir = os.path.join(output_dir, "emotion")
        os.makedirs(emotion_dir, exist_ok=True)
        kwargs = dict(interpolated_result=capture_face_result,
                      video_path=video_path,
                      output_dir=output_dir,
                      offset=offset
                      )
        run_emotion_recognition(**kwargs)

    if config["run_transcript"]:
        transcript_dir = os.path.join(output_dir, "transcript")
        os.makedirs(transcript_dir, exist_ok=True)
        kwargs = dict(output_dir=transcript_dir,
                      audio_path_list=audio_path_list
                      )
        run_transcript(**kwargs)

    if config["run_overlay"]:
        run_overlay(output_dir=output_dir)


# TODO add file check
if __name__ == "__main__":
    # show select input directory dialog
    # input_dir = str(QFileDialog.getExistingDirectory(None, "Select Input Directory"))
    # structure of input directory
    # -- reid
    #    -- *.avi / *.mp4 (reid videos)
    # -- *.avi / *.mp4 (MTG video)
    # -- *.wav (pin-mic voices)

    # expt12 s2t only
    # main_kwargs = {
    #     "video_path": "datasets/200221_expt12/video.mp4",
    #     "output_dir": "output/exp12_s2t",
    #     "audio_path_list": ["datasets/200221_expt12/voice{}.wav".format(i) for i in range(1, 4)],
    #     "face_num": 3,
    #     "face_video_list": []
    # }

    # expt22 s2t only
    # main_kwargs = {
    #     "video_path": "datasets/200225_expt22/video.mp4",
    #     "output_dir": "output/exp22_s2t",
    #     "audio_path_list": ["datasets/200225_expt22/voice{}.wav".format(i) for i in range(1, 6)],
    #     "face_num": 5,
    #     "face_video_list": []
    # }

    # expt58 s2t only
    # main_kwargs = {
    #     "video_path": "datasets/200309_expt58/video.mp4",
    #     "output_dir": "output/exp58_s2t",
    #     "audio_path_list": ["datasets/200309_expt58/voice{}.wav".format(i) for i in range(1, 4)],
    #     "face_num": 3,
    #     "face_video_list": []
    # }

    # expt12
    # main_kwargs = {
    #     "video_path": "datasets/200221_expt12/video.mp4",
    #     "output_dir": "output/exp12_debug",
    #     "audio_path_list": ["datasets/200221_expt12/voice{}.wav".format(i) for i in range(1, 4)],
    #     "face_num": 3,
    #     "face_video_list": ["datasets/200221_expt12/reid/reid{}.mp4".format(i) for i in range(1, 4)]
    # }

    # # expt22
    # main_kwargs = {
    #     "video_path": "datasets/200225_expt22/video.mp4",
    #     "output_dir": "output/exp22",
    #     "audio_path_list": ["datasets/200225_expt22/voice{}.wav".format(i) for i in range(1, 6)],
    #     "face_num": 5,
    #     "face_video_list": ["datasets/200225_expt22/reid/reid{}.mp4".format(i) for i in range(1, 6)]
    # }

    # expt22-30sec
    main_kwargs = {
        "video_path": "datasets/200225_expt22/video-30sec.mp4",
        "output_dir": "output/exp22",
        "audio_path_list": ["datasets/200225_expt22/voice{}.wav".format(i) for i in range(1, 6)],
        "face_num": 5,
        "face_video_list": ["datasets/200225_expt22/reid/reid{}.mp4".format(i) for i in range(1, 6)]
    }

    # # sound only 5 min
    # main_kwargs = {
    #     "video_path": None,
    #     "output_dir": "output/sound_only",
    #     "audio_path_list": ["datasets/sound_only_5min/voice{}_5min.wav".format(i) for i in range(1, 4)],
    #     "face_num": 3,
    #     "face_video_list": None
    # }
    main(**main_kwargs)
