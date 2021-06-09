import os
import datetime
from typing import List
from PyQt5 import QtWidgets

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from numpy.testing._private.utils import KnownFailureTest
import pyqtgraph as pg
import vlc
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QFrame, QSlider, QHBoxLayout, QPushButton, \
    QVBoxLayout, QLabel, QScrollArea, QSizePolicy
from pyqtgraph.Qt import QtGui
from pyqtgraph.dockarea import *
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
import pandas as pd
from PIL import Image
import collections
from emotion_to_video import emo_to_video


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


class VLCControl(QWidget):
    def __init__(self, vlc_widgets: List[VLCWidget]):
        super(VLCControl, self).__init__()
        self.is_paused = False
        self.vlc_widgets = vlc_widgets

        self.slider_position = QSlider(Qt.Horizontal, self)
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
        self.slider_volume = QSlider(Qt.Horizontal, self)
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
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.update_ui)

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
        # setting the slider to the desired position
        position = self.vlc_widgets[0].position
        duration = self.vlc_widgets[0].duration
        cur_time = duration * position  # in ms
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
    def __init__(self, vlc_widget: VLCWidget, transcript_csv: str):
        super().__init__()
        self.vlc_widget = vlc_widget
        self.transcript = pd.read_csv(transcript_csv)

        self.setWidgetResizable(True)

        content = QWidget()
        self.setWidget(content)
        vbox = QVBoxLayout(content)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label.setWordWrap(True)
        vbox.addWidget(self.label)

        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()

    def update_ui(self):
        position = self.vlc_widget.position
        duration = self.vlc_widget.duration
        cur_time = duration * position  # in ms
        row = self.transcript[
            (self.transcript["Start time(ms)"] <= cur_time) & (cur_time < self.transcript["End time(ms)"])]
        if len(row) == 0:
            self.label.setText("")
        else:
            speaker, text = row["Speaker"].item(), row["Text"].item()
            if text == np.nan:
                text = ""
            self.label.setText(f"<b>{speaker}</b>: {text}")


class DiarizationWidget(QWidget):
    def __init__(self, vlc_widget: VLCWidget, diarization_csv: str, speaker_num: int = 6):
        super().__init__()
        self.vlc_widget = vlc_widget
        self.mpl_widget = MatplotlibWidget()
        self.mpl_widget.toolbar.hide()
        self.diarization = pd.read_csv(diarization_csv)
        self.x_margin = 10.0  # second
        self.y_num = speaker_num  # num of speakers (need to think)
        self.y_margin = 0.25
        self.fontsize = 16
        self.init_flag = True  # for update_ui

        # settings of matplotlib graph
        self.ax = self.mpl_widget.getFigure().add_subplot(111)

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()
        hbox = QHBoxLayout()
        hbox.addWidget(self.mpl_widget)
        self.setLayout(hbox)

    def update_ui(self):
        cur_time = self.get_current_time()  # in ms
        if cur_time >= 0.0:  # to prevent xlim turning into minus values
            self.x_lim = [self.ms_to_s(cur_time) - self.x_margin, self.ms_to_s(cur_time) + self.x_margin]
            self.y_lim = [0 - self.y_margin, (self.y_num - 1) + self.y_margin]
            self.ax.set_xlim(self.x_lim)
            self.ax.set_ylim(self.y_lim)

            if self.init_flag:
                self.ax.get_xaxis().set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                self.ax.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                self.ax.set_xlabel("Time [s]", fontsize=self.fontsize)
                self.ax.set_ylabel("Speaker ID", fontsize=self.fontsize)

                self.ax.tick_params(axis='x', labelsize=self.fontsize)
                self.ax.tick_params(axis='y', labelsize=self.fontsize)

                # plot current time bar
                self.ax_line = self.ax.axvline(self.ms_to_s(cur_time), color='blue', linestyle='dashed', linewidth=3)
                self.ax_text = self.ax.text(self.ms_to_s(cur_time), self.y_lim[1], "Current Time", va='bottom', ha='center', fontsize=self.fontsize, color="blue", weight='bold')

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

    def get_current_time(self):
        position = self.vlc_widget.position
        duration = self.vlc_widget.duration
        cur_time = duration * position  # in ms
        return cur_time

    def plot_diarization(self, row):
        speaker = row["Speaker"].item()
        start_time, end_time = row["Start time(ms)"].item(), row["End time(ms)"].item()
        start_time_s, end_time_s = self.ms_to_s(start_time), self.ms_to_s(end_time)
        self.ax.plot([start_time_s, end_time_s], [speaker, speaker], color="black", linewidth=4.0, zorder=1)


class OverviewDiarizationWidget(QWidget):
    def __init__(self, vlc_widget: VLCWidget, diarization_csv: str, emotion_csv_list: list, speaker_num: int = 6):
        super().__init__()
        self.vlc_widget = vlc_widget
        self.mpl_widget = MatplotlibWidget()
        self.mpl_widget.toolbar.hide()
        self.diarization = pd.read_csv(diarization_csv)
        self.video_begin_time = 0  # video's begin time (ms)
        self.video_end_time = self.vlc_widget.duration  # video's end time (ms)
        self.y_num = speaker_num  # num of speakers (need to think)
        self.y_margin = 0.25
        self.fontsize = 16
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
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()
        hbox = QHBoxLayout()
        hbox.addWidget(self.mpl_widget)
        self.setLayout(hbox)

    def update_ui(self):
        cur_time = self.get_current_time()  # in ms
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
                self.ax_text = self.ax.text(self.ms_to_s(cur_time), self.y_lim[1], "Current Time", va='bottom', ha='center', fontsize=self.fontsize, color="blue", weight='bold')

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

    def get_current_time(self):
        position = self.vlc_widget.position
        duration = self.vlc_widget.duration
        cur_time = duration * position  # in ms
        return cur_time

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


def create_summary(emotion_dir, diarization_dir):
    result_list = sorted([os.path.join(emotion_dir, d) for d in os.listdir(emotion_dir) if d.endswith(".csv")])
    data_summary = [[] for _ in range(len(result_list))]
    new_columns_name = ['発話数', '発話時間 [s]', "発話密度 [s]", '会話占有率 [%]', "頷き回数"]

    if os.path.isfile(os.path.join(diarization_dir, "result_loudness.csv")):
        diarization_csv = os.path.join(diarization_dir, "result_loudness.csv")
    elif os.path.isfile(os.path.join(diarization_dir, "result.csv")):
        diarization_csv = os.path.join(diarization_dir, "result.csv")
    else:
        raise OSError("Diarization result not found")
    df_diarization = pd.read_csv(diarization_csv)

    num_of_utterances = collections.Counter(del_continual_value(df_diarization["speaker class"].to_list()))
    speech_time = df_diarization["speaker class"].value_counts() / 1000
    time_occupancy = df_diarization["speaker class"].value_counts(normalize=True)
    for x in range(len(result_list)):
        try:
            data_summary[x].append(int(num_of_utterances[x]))
        except KeyError:
            data_summary[x].append(0)

        try:
            data_summary[x].append(int(speech_time[x]))
        except KeyError:
            data_summary[x].append(0)

        try:
            data_summary[x].append(int(speech_time[x] / num_of_utterances[x]))
        except KeyError:
            data_summary[x].append(0)

        try:
            data_summary[x].append(int(time_occupancy[x] * 100))
        except KeyError:
            data_summary[x].append(0)

        df_gesture_tmp = pd.read_csv(os.path.join(emotion_dir, f"result{x}.csv"), encoding="shift_jis", header=0,
                                     usecols=["gesture"])
        df_gesture_tmp = df_gesture_tmp[df_gesture_tmp.diff()["gesture"] != 0]
        gesture_count = df_gesture_tmp["gesture"].value_counts()
        try:
            data_summary[x].append(int(gesture_count[1]))
        except KeyError:
            data_summary[x].append(0)
    # rows_summary = [f"{index_to_name_dict[index]}" for index in range(len(result_list))]
    rows_summary = [f"{index}" for index in range(len(result_list))]
    df_summary = pd.DataFrame(data_summary, index=rows_summary, columns=new_columns_name)
    return df_summary

##



dataset_dir = "/home/user/icer/Project/dataset/"
# dataset_dir = "/home/icer/Project/dataset"
# data_dir = "./gui_araki/data/"
data_dir = "../gui_araki/data/"

app = pg.mkQApp("Overlay")
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1920, 1080)
win.setWindowTitle("Overlay")

d1 = Dock("Emotion", size=(1600, 900))
d2 = Dock("Control", size=(1600, 100))
d3 = Dock("Transcript", size=(1600, 100))
d7 = Dock("Diarization", size=(1600, 500))
d8 = Dock("OverviewDiarization", size=(1600, 500))

# d4 = Dock("Dock4 (tabbed) - Plot", size=(500, 200))
d5 = Dock("Summary", size=(800, 400))
# d6 = Dock("Dock6 (tabbed) - Plot", size=(500, 200))
d6 = Dock("Emotion_Statistics", size=(800,800)) ##


area.addDock(d1, 'left')
area.addDock(d2, 'bottom', d1)
# area.addDock(d3, 'right', d2)
# area.addDock(d4, 'right')
area.addDock(d5, 'right', d1)
# area.addDock(d6, 'top', d4)

area.addDock(d3, 'bottom', d1)
area.addDock(d7, 'top', d2)
area.addDock(d8, 'right', d7)

area.addDock(d6, 'right',d1)##

vlc_widget_list = []
vlc_widget1 = VLCWidget()
vlc_widget_list.append(vlc_widget1)
d1.addWidget(vlc_widget1)
vlc_widget1.media = data_dir + "test_video_emotion.avi"
vlc_widget1.play()

# vlc_widget2 = VLCWidget()
# d3.addWidget(vlc_widget2)
# vlc_widget_list.append(vlc_widget2)
# vlc_widget2.media = "low_fps.mp4"
# vlc_widget2.play()

d2.addWidget(VLCControl(vlc_widget_list))
d3.addWidget(TranscriptWidget(vlc_widget1, transcript_csv=data_dir + "transcript.csv"))

# summary = pg.ImageView()
# img = np.array(Image.open("summary_dummy.png")).T
# summary.setImage(img)
# summary.ui.histogram.hide()
# summary.ui.roiBtn.hide()
# summary.ui.roiPlot.hide()
# summary.ui.menuBtn.hide()
# d5.addWidget(summary)

summary = DataFrameWidget(create_summary(dataset_dir + "/emotion", dataset_dir + "/transcript/diarization"))
d5.addWidget(summary)
# w5 = pg.ImageView()
# w5.setImage(np.random.normal(size=(100, 100)))
# d5.addWidget(w5)
#
# w6 = pg.PlotWidget(title="Dock 6 plot")
# w6.plot(np.random.normal(size=100))
# d6.addWidget(w6)

d7.addWidget(DiarizationWidget(vlc_widget1, diarization_csv=data_dir + "transcript.csv"))

emotion_csv_list = [
    dataset_dir + "emotion/" + "result0.csv",
    dataset_dir + "emotion/" + "result1.csv",
    dataset_dir + "emotion/" + "result2.csv",
    dataset_dir + "emotion/" + "result3.csv",
    dataset_dir + "emotion/" + "result4.csv"
]
d8.addWidget(OverviewDiarizationWidget(vlc_widget1, diarization_csv=data_dir + "transcript.csv", emotion_csv_list=emotion_csv_list))

##
vlc_widget2 = VLCWidget()
vlc_widget_list.append(vlc_widget2)
d6.addWidget(vlc_widget2)
vlc_widget2.media = data_dir + emo_to_video("/home/user/icer/icer/main_test (copy)/face/cluster/","/home/user/icer/icer/main_test (copy)/emotion/","/home/user/icer/icer/main_test (copy)/transcript/diarization/result.csv","/home/user/icer/icer/main_test (copy)/emotion/output.avi","/home/user/icer/icer/outputs/test3.avi",3)
# vlc_widget2.media="/home/user/icer/icer/outputs/test3.avi"
vlc_widget2.play()


win.showMaximized()

if __name__ == '__main__':
    pg.mkQApp().exec_()
