import collections
import datetime
import os
from typing import List
import time

import matplotlib
import numpy as np
import pandas as pd
import pyqtgraph as pg
import vlc
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QWidget, QFrame, QSlider, QHBoxLayout, QPushButton, \
    QVBoxLayout, QLabel, QScrollArea
from pyqtgraph.Qt import QtGui
from pyqtgraph.dockarea import *
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget

# Disable VLC error messages
os.environ['VLC_VERBOSE'] = '-1'

config = {
    "win_title": "Overlay",
    "win_size": (1920, 1080),
    "plt_font_size": 12,
    "update_ui_interval": 20,
    "default_volume": 100
}


def current_milli_time():
    return round(time.time() * 1000)


class Slider(QSlider):
    def __init__(self, *args, **kwargs):
        super(Slider, self).__init__(*args, **kwargs)

    # Override default mouse press behaviour
    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        if e.button() == Qt.LeftButton:
            e.accept()
            x = e.pos().x()
            value = (self.maximum() - self.minimum()) * x / self.width() + self.minimum()
            self.setValue(int(value))
            # Emit signal
            self.sliderMoved.emit(int(value))


class VLCWidget(QFrame):
    def __init__(self):
        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()
        self._media = None
        super(VLCWidget, self).__init__()

    @property
    def media(self):
        return self._media

    @media.setter
    def media(self, video_path: str):
        self._media = self.instance.media_new(video_path)
        self.media_player.set_media(self._media)
        self.media_player.set_xwindow(self.winId())

    @property
    def volume(self):
        return self.media_player.audio_get_volume()

    @volume.setter
    def volume(self, volume):
        self.media_player.audio_set_volume(volume)

    @property
    def position(self):
        return self.media_player.get_position()

    @position.setter
    def position(self, position):
        self.media_player.set_position(position)

    @property
    def duration(self):
        return self.media.get_duration()

    @property
    def is_playing(self):
        return self.media_player.is_playing()

    def play(self):
        self.media_player.play()

    def pause(self):
        self.media_player.pause()

    def stop(self):
        self.media_player.stop()

    def add_audio_track(self, audio):
        self.media_player.add_slave(vlc.MediaSlaveType(1), audio, True)


class VLCTimeKeeper:
    def __init__(self, vlc_widget: VLCWidget):
        self._vlc_widget = vlc_widget
        self._prev_sys_time = None
        self._prev_vlc_time = None
        self._accum_delta = 0

    @property
    def precise_cur_time(self):
        position = self._vlc_widget.position
        duration = self._vlc_widget.duration
        cur_time = duration * position  # in ms
        # Workaround for vlc low pooling rate
        cur_sys_time = current_milli_time()

        if self._prev_sys_time is None:
            self._prev_sys_time = cur_sys_time
        if self._prev_vlc_time is None:
            self._prev_vlc_time = cur_time

        if self._vlc_widget.is_playing:
            self._accum_delta += cur_sys_time - self._prev_sys_time
        if self._prev_vlc_time != cur_time:
            self._accum_delta = 0
        self._prev_sys_time = cur_sys_time
        self._prev_vlc_time = cur_time
        cur_time += self._accum_delta
        return cur_time


class VLCControl(QWidget):
    def __init__(self, vlc_widgets: List[VLCWidget]):
        super(VLCControl, self).__init__()
        self.is_paused = False
        self.vlc_widgets = vlc_widgets

        self.slider_position = Slider(Qt.Horizontal, self)
        self.slider_position.setToolTip("Position")
        self.slider_position.setMaximum(1000)
        self.slider_position.sliderMoved.connect(self.set_position)

        self.hbox = QHBoxLayout()
        self.button_play = QPushButton("Play")
        self.hbox.addWidget(self.button_play)
        self.button_play.clicked.connect(self.play_pause)

        self.button_stop = QPushButton("Stop")
        self.hbox.addWidget(self.button_stop)
        self.button_stop.clicked.connect(self.stop)

        self.hbox.addStretch(1)
        self.hbox.addWidget(QLabel("Volume"))
        self.slider_volume = Slider(Qt.Horizontal, self)
        self.slider_volume.setMaximum(100)
        self.slider_volume.setValue(vlc_widgets[0].volume)
        self.slider_volume.setToolTip("Volume")
        self.hbox.addWidget(self.slider_volume)
        self.slider_volume.valueChanged.connect(self.set_volume)

        self.hbox2 = QHBoxLayout()
        self.label_time = QLabel()
        self.hbox2.addWidget(self.label_time)
        self.hbox2.addWidget(self.slider_position)

        self.vboxlayout = QVBoxLayout()
        self.vboxlayout.addLayout(self.hbox2)
        self.vboxlayout.addLayout(self.hbox)
        self.setLayout(self.vboxlayout)

        self.timer = QTimer(self)
        self.timer.setInterval(config["update_ui_interval"])
        self.timer.timeout.connect(self.update_ui)
        self.timekeeper = VLCTimeKeeper(vlc_widgets[0])
        self.play_pause()

    def set_position(self, position):
        """Set the position"""
        # setting the position to where the slider was dragged
        for vlc_widget in self.vlc_widgets:
            vlc_widget.position = position / 1000.0
        # the vlc MediaPlayer needs a float value between 0 and 1, Qt
        # uses integer variables, so you need a factor; the higher the
        # factor, the more precise are the results
        # (1000 should be enough)

    def play_pause(self, *args):
        """Toggle play/pause status"""
        if self.vlc_widgets[0].is_playing:
            for vlc_widget in self.vlc_widgets:
                vlc_widget.pause()
            self.button_play.setText("Play")
            self.is_paused = True
        else:
            if self.vlc_widgets[0].play() == -1:
                print("No file to play")
                return
            for vlc_widget in self.vlc_widgets:
                vlc_widget.play()
            self.button_play.setText("Pause")
            self.timer.start()
            self.is_paused = False

    def stop(self, *args):
        """Stop player"""
        for vlc_widget in self.vlc_widgets:
            vlc_widget.stop()
        self.button_play.setText("Play")

    def set_volume(self, volume):
        """Set the volume"""
        for vlc_widget in self.vlc_widgets:
            vlc_widget.volume = volume

    def update_ui(self):
        """updates the user interface"""
        cur_time = self.timekeeper.precise_cur_time
        position = cur_time / self.vlc_widgets[0].duration

        self.slider_position.setValue(int(position * 1000))
        if cur_time > 0:
            self.label_time.setText(str(datetime.timedelta(seconds=cur_time / 1000)).split(".")[0])
        else:
            self.label_time.setText("0:00:00")

        if not self.vlc_widgets[0].is_playing:
            # no need to call this function if nothing is played
            self.timer.stop()
            if not self.is_paused:
                # after the video finished, the play button stills shows
                # "Pause", not the desired behavior of a media player
                # this will fix it
                self.stop()


class TranscriptWidget(QScrollArea):
    def __init__(self, vlc_widget: VLCWidget, transcript_csv: str, speaker_num: int, name_list: list = None, **kwargs):
        super().__init__()
        self.vlc_widget = vlc_widget
        self.transcript = pd.read_csv(transcript_csv)

        if name_list is None:
            name_list = [f"Speaker {i}" for i in range(speaker_num)]
        assert len(name_list) == speaker_num
        self._name_list = name_list

        self.setWidgetResizable(True)

        content = QWidget()
        self.setWidget(content)
        vbox = QVBoxLayout(content)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label.setWordWrap(True)
        vbox.addWidget(self.label)

        self.timer = QTimer(self)
        self.timer.setInterval(config["update_ui_interval"])
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()
        self.timekeeper = VLCTimeKeeper(vlc_widget)

    def update_ui(self):
        cur_time = self.timekeeper.precise_cur_time

        row = self.transcript[
            (self.transcript["Start time(ms)"] <= cur_time) & (cur_time < self.transcript["End time(ms)"])]
        if len(row) == 0:
            self.label.setText("")
        else:
            speaker, text = self._name_list[row["Speaker"].item()], row["Text"].item()
            if text == np.nan:
                text = ""
            self.label.setText(f"<b>{speaker}</b>: {text}")


class DiarizationWidget(QWidget):
    def __init__(self, vlc_widget: VLCWidget, transcript_csv: str, speaker_num: int, **kwargs):
        super().__init__()
        self.vlc_widget = vlc_widget
        self.mpl_widget = MatplotlibWidget()
        self.mpl_widget.toolbar.hide()
        self.diarization = pd.read_csv(transcript_csv)
        self.x_margin = 10.0  # second
        self.y_num = speaker_num  # num of speakers
        self.y_margin = 0.25
        self.fontsize = config["plt_font_size"]
        self.init_flag = True  # for update_ui

        # settings of matplotlib graph
        self.ax = self.mpl_widget.getFigure().add_subplot(111)

        self.timer = QTimer(self)
        self.timer.setInterval(config["update_ui_interval"])
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()
        self.timekeeper = VLCTimeKeeper(vlc_widget)
        hbox = QHBoxLayout()
        hbox.addWidget(self.mpl_widget)
        self.setLayout(hbox)

    def update_ui(self):
        # cur_time = self.get_current_time()  # in ms
        cur_time = self.timekeeper.precise_cur_time

        if cur_time >= 0.0:  # to prevent xlim turning into minus values
            self.x_lim = [self.ms_to_s(cur_time) - self.x_margin, self.ms_to_s(cur_time) + self.x_margin]
            self.y_lim = [0 - self.y_margin, (self.y_num - 1) + self.y_margin]
            self.ax.set_xlim(self.x_lim)
            self.ax.set_ylim(self.y_lim)

            if self.init_flag:
                # Fixed tick frequency
                # https://stackoverflow.com/questions/12608788/changing-the-tick-frequency-on-x-or-y-axis-in-matplotlib
                tick_spacing = 2
                self.ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacing))
                # self.ax.get_xaxis().set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                self.ax.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                self.ax.set_xlabel("Time [s]", fontsize=self.fontsize)
                self.ax.set_ylabel("Speaker ID", fontsize=self.fontsize)

                self.ax.tick_params(axis='x', labelsize=self.fontsize)
                self.ax.tick_params(axis='y', labelsize=self.fontsize)

                # plot current time bar
                self.ax_line = self.ax.axvline(self.ms_to_s(cur_time), color='blue', linestyle='dashed', linewidth=3)
                self.ax_text = self.ax.text(self.ms_to_s(cur_time), self.y_lim[1], "Current Time", va='bottom',
                                            ha='center', fontsize=self.fontsize, color="blue", weight='bold')

                # plot all diarizations
                rows = self.diarization
                for i in range(len(rows)):
                    row = rows.iloc[i]
                    self.plot_diarization(row)

                self.init_flag = False
            else:
                # update current time bar
                self.ax_line.set_xdata([self.ms_to_s(cur_time)])
                self.ax_text.set_x(self.ms_to_s(cur_time))

            self.mpl_widget.draw()

    def ms_to_s(self, x):
        return x / 1000

    def plot_diarization(self, row):
        speaker = row["Speaker"].item()
        start_time, end_time = row["Start time(ms)"].item(), row["End time(ms)"].item()
        start_time_s, end_time_s = self.ms_to_s(start_time), self.ms_to_s(end_time)
        self.ax.plot([start_time_s, end_time_s], [speaker, speaker], color="black", linewidth=4.0, zorder=1)


class OverviewDiarizationWidget(QWidget):
    def __init__(self, vlc_widget: VLCWidget, transcript_csv: str, emotion_csv_list: list, speaker_num: int, **kwargs):
        super().__init__()
        self.vlc_widget = vlc_widget
        self.mpl_widget = MatplotlibWidget()
        self.mpl_widget.toolbar.hide()
        self.diarization = pd.read_csv(transcript_csv)
        self.video_begin_time = 0  # video's begin time (ms)
        self.video_end_time = self.vlc_widget.duration  # video's end time (ms)
        self.y_num = speaker_num  # num of speakers
        self.y_margin = 0.25
        self.fontsize = config["plt_font_size"]
        self.init_flag = True

        # get each person's gesture
        self.emotion_list = []
        for emotion_csv in emotion_csv_list:
            emotion = pd.read_csv(emotion_csv)
            self.emotion_list.append(emotion)

        # get gesture list
        self.gesture_x, self.gesture_y = self.get_gesture(self.emotion_list)

        # settings of matplotlib graph
        self.ax = self.mpl_widget.getFigure().add_subplot(111)

        self.timer = QTimer(self)
        self.timer.setInterval(config["update_ui_interval"])
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()
        self.timekeeper = VLCTimeKeeper(vlc_widget)
        hbox = QHBoxLayout()
        hbox.addWidget(self.mpl_widget)
        self.setLayout(hbox)

    def update_ui(self):
        cur_time = self.timekeeper.precise_cur_time

        if cur_time >= 0.0:  # to prevent xlim turning into minus values
            if self.init_flag:
                self.ax.get_xaxis().set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                self.ax.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                self.ax.set_xlabel("Time [s]", fontsize=self.fontsize)
                self.ax.set_ylabel("Speaker ID", fontsize=self.fontsize)

                self.x_lim = [self.ms_to_s(self.video_begin_time), self.ms_to_s(self.video_end_time)]
                self.y_lim = [0 - self.y_margin, (self.y_num - 1) + self.y_margin]
                self.ax.set_xlim(self.x_lim)
                self.ax.set_ylim(self.y_lim)
                self.ax.tick_params(axis='x', labelsize=self.fontsize)
                self.ax.tick_params(axis='y', labelsize=self.fontsize)

                # plot current time bar
                self.ax_line = self.ax.axvline(self.ms_to_s(cur_time), color='blue', linestyle='dashed', linewidth=3)
                self.ax_text = self.ax.text(self.ms_to_s(cur_time), self.y_lim[1], "Current Time", va='bottom',
                                            ha='center', fontsize=self.fontsize, color="blue", weight='bold')

                # plot all diarizations
                rows = self.diarization
                for i in range(len(rows)):
                    row = rows.iloc[i]
                    self.plot_diarization(row)

                # plot gesture
                self.ax.scatter(self.gesture_x, self.gesture_y, c='red', marker='o', zorder=2)

                self.init_flag = False
            else:
                # update current time bar
                self.ax_line.set_xdata([self.ms_to_s(cur_time)])
                self.ax_text.set_x(self.ms_to_s(cur_time))

            self.mpl_widget.draw()

    def ms_to_s(self, x):
        return x / 1000

    def plot_diarization(self, row):
        speaker = row["Speaker"].item()
        start_time, end_time = row["Start time(ms)"].item(), row["End time(ms)"].item()
        start_time_s, end_time_s = self.ms_to_s(start_time), self.ms_to_s(end_time)
        self.ax.plot([start_time_s, end_time_s], [speaker, speaker], color="black", linewidth=4.0, zorder=1)

    def get_gesture(self, emotion_list: list):
        gesture_x, gesture_y = [], []
        for speaker_id, emotion_rows in enumerate(emotion_list):
            for i in range(len(emotion_rows)):
                emotion_row = emotion_rows.iloc[i]
                if emotion_row["gesture"] == 1:
                    time = emotion_row["time(ms)"].item()
                    time_s = self.ms_to_s(time)
                    gesture_x.append(time_s), gesture_y.append(speaker_id)
        return gesture_x, gesture_y


class DataFrameWidget(pg.TableWidget):
    """
    A widget to display pandas dataframe
    """

    def __init__(self, dataframe, *args, **kwds):
        super().__init__(*args, **kwds)
        data = []
        for index, row in dataframe.iterrows():
            data.append(row.to_dict())
        self.setData(data)


def del_continual_value(target_list):
    ret_list = []
    last_value = None
    for x in target_list:
        if x != last_value:
            ret_list.append(x)
            last_value = x
    return ret_list


def create_summary(emotion_csv_list: list, transcript_csv: str, speaker_num: int, name_list: list = None, **kwargs):
    new_columns_name = ['話者', '発話数', '発話時間 [s]', "発話密度 [s]", '会話占有率 [%]', "頷き回数"]
    df_diarization = pd.read_csv(transcript_csv)

    if name_list is None:
        name_list = [f"Speaker {i}" for i in range(speaker_num)]
    assert len(name_list) == speaker_num

    num_of_utterances = collections.Counter(del_continual_value(df_diarization["Speaker"].to_list()))
    num_of_utterances = np.array([num_of_utterances.get(i, 0) for i in range(speaker_num)])

    speech_time = []
    for i in range(speaker_num):
        cols = df_diarization[df_diarization["Speaker"] == i]
        speech_time.append((cols["End time(ms)"] - cols["Start time(ms)"]).sum() / 1000)
    speech_time = np.array(speech_time)

    with np.errstate(divide='ignore', invalid='ignore'):
        # Ignore divide by zero warning
        speech_density = speech_time / num_of_utterances
    total_time = np.sum(speech_time)
    time_occupancy = speech_time / total_time * 100

    gesture_count = []
    for emotion_csv in emotion_csv_list:
        df_gesture_tmp = pd.read_csv(emotion_csv, encoding="shift_jis", header=0, usecols=["gesture"])
        df_gesture_tmp = df_gesture_tmp[df_gesture_tmp.diff()["gesture"] != 0]
        gesture_count.append(df_gesture_tmp["gesture"].value_counts().get(1, 0))
    gesture_count = np.array(gesture_count)

    data_summary = [_ for _ in
                    zip(name_list, num_of_utterances, speech_time, speech_density, time_occupancy, gesture_count)]
    df_summary = pd.DataFrame(data_summary, columns=new_columns_name)
    return df_summary


def main_overlay(output_dir: str):
    # read dir according to the specific structure
    # -- emotion
    #    -- result*.csv
    #    -- *.avi / *.mp4
    # -- transcript
    #    -- diarization

    emotion_dir = os.path.join(output_dir, "emotion")
    transcript_dir = os.path.join(output_dir, "transcript")
    transcript_csv_path = os.path.join(transcript_dir, "transcript.csv")
    emotion_dir_files = sorted([os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir)])
    emotion_csv_path_list = list(filter(lambda x: x.endswith(".csv"), emotion_dir_files))
    video_path = list(filter(lambda x: x.endswith(".avi") or x.endswith(".mp4"), emotion_dir_files))[0]
    speaker_num = len(emotion_csv_path_list)

    # initialize qt_app
    app = pg.mkQApp(config["win_title"])
    win = QtGui.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(*config["win_size"])
    win.setWindowTitle(config["win_title"])

    # initialize each dock
    d1 = Dock("Emotion", size=(1600, 900))
    d2 = Dock("Control", size=(1600, 100))
    d3 = Dock("Transcript", size=(1600, 100))
    d4 = Dock("Summary", size=(800, 400))
    d5 = Dock("Diarization", size=(1600, 500))
    d6 = Dock("OverviewDiarization", size=(1600, 500))

    # set dock's position
    area.addDock(d1, 'left')
    area.addDock(d2, 'bottom', d1)
    area.addDock(d4, 'right', d1)
    area.addDock(d3, 'bottom', d1)
    area.addDock(d5, 'top', d2)
    area.addDock(d6, 'right', d5)

    # settings for video
    vlc_widget_list = []
    vlc_widget1 = VLCWidget()
    vlc_widget_list.append(vlc_widget1)
    d1.addWidget(vlc_widget1)
    vlc_widget1.media = video_path
    # set default volume
    vlc_widget1.volume = config["default_volume"]
    vlc_widget1.play()

    # make widgets for each dock
    common_kwargs = dict(emotion_csv_list=emotion_csv_path_list,
                         transcript_csv=transcript_csv_path,
                         speaker_num=speaker_num,
                         name_list=None)

    d2.addWidget(VLCControl(vlc_widget_list))
    d3.addWidget(TranscriptWidget(vlc_widget1, **common_kwargs))
    d4.addWidget(DataFrameWidget(create_summary(**common_kwargs)))
    d5.addWidget(DiarizationWidget(vlc_widget1, **common_kwargs))
    d6.addWidget(OverviewDiarizationWidget(vlc_widget1, **common_kwargs))

    # run displaying
    win.showMaximized()
    pg.mkQApp().exec_()

    # Exit when window is destroyed
    exit(0)


if __name__ == '__main__':
    main_overlay("../output/test")
