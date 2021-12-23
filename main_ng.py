import datetime
import os
import sys
import tempfile
import time
import json

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
# Ignore Pydub ResourceWarning
# https://github.com/jiaaro/pydub/issues/214
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Set multiprocessing start method to "spawn" (to avoid bug)
# Some GUI will freeze if not set to spawn, Linux default is fork
# This method should only be called once
import multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

from main_gui.main import main as main_gui
# from gui.calibrate import adjust_offset_dialog
# from gui.cropper import select_roi_dialog
# from gui.dialogs import get_face_num
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
    main_overlay(**kwargs)


def create_face_video(video_path, output_path_list, reid_roi_list, offset=0):
    from utils.video_utils import crop_video, calibrate_video

    with tempfile.TemporaryDirectory() as temp_dirname:
        print("Temporary directory created at", temp_dirname)
        calibrate_path = os.path.join(temp_dirname, "calibrate.mp4")
        calibrate_video(video_path, output_path=calibrate_path, offset=offset, end_time=30, remove_audio=True)

        for output_path, roi in zip(output_path_list, reid_roi_list):
            kwargs = dict(video_path=calibrate_path,
                          output_path=output_path,
                          roi=roi,
                          console_quiet=False,
                          use_gpu=False)  # Don't use GPU when multi-processing!
            crop_video(**kwargs)


def get_outer_roi(reid_roi_list):
    import numpy as np
    rois = np.array(reid_roi_list)
    # (x,y,w,h) -> (x1,y1,x2,y2) for each roi
    cord = np.concatenate([rois[:, :2], rois[:, :2] + rois[:, 2:]])
    # Outer ROI
    (x, y), (w, h) = np.min(cord, axis=0), np.max(cord, axis=0) - np.min(cord, axis=0)
    return x, y, w, h


def get_timestamp():
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d-%H%M")


@timeit
def main(video_path: str = None, output_dir: str = None, audio_path_list: list = None, face_num=None,
         face_video_list=None):
    QApplication(sys.argv)

    """
    Params example:
    {'offset': -433,
 'result': {'Person01': {'audio_path': '/home/jeffshee/Developers/icer-gui/data/200225_芳賀先生_実験22voice1.wav',
                         'rgb': (255, 127, 14),
                         'roi': (296, 842, 381, 455)},
            'Person02': {'audio_path': '/home/jeffshee/Developers/icer-gui/data/200225_芳賀先生_実験22voice2.wav',
                         'rgb': (44, 160, 44),
                         'roi': (728, 948, 315, 379)},
            'Person03': {'audio_path': '/home/jeffshee/Developers/icer-gui/data/200225_芳賀先生_実験22voice3.wav',
                         'rgb': (214, 39, 40),
                         'roi': (1230, 934, 317, 325)},
            'Person04': {'audio_path': '/home/jeffshee/Developers/icer-gui/data/200225_芳賀先生_実験22voice4.wav',
                         'rgb': (148, 103, 189),
                         'roi': (2246, 854, 507, 543)},
            'Person05': {'audio_path': '/home/jeffshee/Developers/icer-gui/data/200225_芳賀先生_実験22voice5.wav',
                         'rgb': (140, 86, 75),
                         'roi': (2838, 806, 425, 519)}},
 'video_path': '/home/jeffshee/Developers/icer-gui/data/200225_芳賀先生_実験22video.mp4'}
 
    Note: ROI is in (x,y,w,h) format
    """
    params = main_gui()
    video_path = params["video_path"]
    basename = os.path.basename(params["video_path"]).split(".")[0]
    output_dir = os.path.join("output", f"{get_timestamp()}_{basename}")
    os.makedirs(output_dir, exist_ok=True)
    # Dump params into json
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=3)

    offset = params["offset"]
    result = params["result"]
    face_num = len(result)
    audio_path_list = []
    face_video_list = []
    reid_roi_list = []

    reid_dir = os.path.join(output_dir, "reid")
    os.makedirs(reid_dir, exist_ok=True)
    for name, item in result.items():
        audio_path_list.append(item["audio_path"])
        reid_roi_list.append(item["roi"])
        face_video_list.append(os.path.join(reid_dir, f"{name}.mp4"))

    create_face_video(video_path, face_video_list, reid_roi_list, offset)
    roi = get_outer_roi(reid_roi_list)

    # # Preprocess (Old)
    # offset = 0
    # roi = None
    # if config["run_capture_face"] or config["run_emotion_recognition"]:
    #     offset = adjust_offset_dialog(video_path)
    #     if not face_num:
    #         face_num = get_face_num()
    # if config["run_capture_face"]:
    #     roi = select_roi_dialog(video_path, offset, window_title="顔検出を行う領域の指定")

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


if __name__ == "__main__":
    # structure of input directory
    # -- *.avi / *.mp4 (MTG video)
    # -- *.wav (pin-mic voices)

    main()
