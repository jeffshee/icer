import datetime
import os
import sys
import time

from PyQt5.QtWidgets import QApplication

# OpenCV2+PyQt5 issue workaround for Linux
# https://forum.qt.io/topic/119109/using-pyqt5-with-opencv-python-cv2-causes-error-could-not-load-qt-platform-plugin-xcb-even-though-it-was-found/21
from cv2.version import ci_build, headless

ci_and_not_headless = ci_build and not headless
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    os.environ.pop("QT_QPA_FONTDIR")

from constants import constants

config = {
    "run_capture_face": constants.RUN_CAPTURE_FACE,  # 動画から顔領域の切り出し
    "run_emotion_recognition": constants.RUN_EMOTION_RECOGNITION,  # 切り出した顔画像の表情・頷き・口の開閉認識
    "run_transcript": constants.RUN_TRANSCRIPT,  # Diarization・音声認識
    "run_overlay": constants.RUN_OVERLAY,  # 表情・頷き・発話情報を動画にまとめて可視化
    "tensorflow_log_level": constants.TF_LOG_LEVEL
}

# Filter warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(config["tensorflow_log_level"])

import warnings

warnings.filterwarnings("ignore", category=Warning)

# Set multiprocessing start method to "spawn" (to avoid bug)
# Some GUI will freeze if not set to spawn, Linux default is fork
# This method should only be called once
import multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

from gui.calibrate import adjust_offset_dialog
from gui.cropper import select_roi_dialog
from gui.dialogs import get_face_num
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

    # Get file list, keep only csv files
    emotion_csv_path_list = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".csv"),
                                          [os.path.join(output_dir, f) for f in os.listdir(output_dir)]))
    output_kwargs = dict(interpolated_result=capture_face_result,
                         emotion_csv_path_list=emotion_csv_path_list,
                         video_path=video_path,
                         output_path=os.path.join(output_dir, temp_filename),
                         offset=offset
                         )
    output_video_emotion_multiprocess(**output_kwargs)
    set_audio(os.path.join(output_dir, temp_filename), video_path, os.path.join(output_dir, video_filename))
    # Remove temp
    os.remove(os.path.join(output_dir, temp_filename))


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

    if config["run_emotion_recognition"]:
        assert capture_face_result
        emotion_dir = os.path.join(output_dir, "emotion")
        os.makedirs(emotion_dir, exist_ok=True)
        kwargs = dict(interpolated_result=capture_face_result,
                      video_path=video_path,
                      output_dir=emotion_dir,
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
    main_kwargs = {
        "video_path": "datasets/200221_expt12/video.mp4",
        "output_dir": "output/exp12",
        "audio_path_list": ["datasets/200221_expt12/voice{}.wav".format(i) for i in range(1, 4)],
        "face_num": 3,
        "face_video_list": ["datasets/200221_expt12/reid/reid{}.mp4".format(i) for i in range(1, 4)]
    }

    # # expt22
    # main_kwargs = {
    #     "video_path": "datasets/200225_expt22/video.mp4",
    #     "output_dir": "output/exp22",
    #     "audio_path_list": ["datasets/200225_expt22/voice{}.wav".format(i) for i in range(1, 6)],
    #     "face_num": 5,
    #     "face_video_list": ["datasets/200225_expt22/reid/reid{}.mp4".format(i) for i in range(1, 6)]
    # }

    # # expt22-30sec
    # main_kwargs = {
    #     "video_path": "datasets/200225_expt22/video-30sec.mp4",
    #     "output_dir": "output/exp22",
    #     "audio_path_list": ["datasets/200225_expt22/voice{}.wav".format(i) for i in range(1, 6)],
    #     "face_num": 5,
    #     "face_video_list": ["datasets/200225_expt22/reid/reid{}.mp4".format(i) for i in range(1, 6)]
    # }

    # # sound only 5 min
    # main_kwargs = {
    #     "video_path": None,
    #     "output_dir": "output/sound_only",
    #     "audio_path_list": ["datasets/sound_only_5min/voice{}_5min.wav".format(i) for i in range(1, 4)],
    #     "face_num": 3,
    #     "face_video_list": None
    # }
    main(**main_kwargs)
