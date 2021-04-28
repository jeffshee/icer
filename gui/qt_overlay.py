import datetime
from typing import List

import numpy as np
import pyqtgraph as pg
import vlc
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QFrame, QSlider, QHBoxLayout, QPushButton, \
    QVBoxLayout, QLabel, QScrollArea, QSizePolicy
from pyqtgraph.Qt import QtGui
from pyqtgraph.dockarea import *
import pandas as pd
from PIL import Image


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


class SummaryWidget(pg.TableWidget):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        data = np.array([
            (1, 1.6, 'x'),
            (3, 5.4, 'y'),
            (8, 12.5, 'z'),
            (443, 1e-12, 'w'),
        ], dtype=[('Column 1', int), ('Column 2', float), ('Column 3', object)])
        # TODO convert dataframe
        # for index, row in data.iterrows():
        #     print(row)
        # data = pd.read_csv("transcript.csv")
        self.setData(data)


app = pg.mkQApp("Overlay")
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1920, 1080)
win.setWindowTitle("Overlay")

d1 = Dock("Emotion", size=(1600, 900))
d2 = Dock("Control", size=(1600, 100))
d3 = Dock("Transcript", size=(1600, 100))

# d4 = Dock("Dock4 (tabbed) - Plot", size=(500, 200))
d5 = Dock("Summary", size=(800, 400))
# d6 = Dock("Dock6 (tabbed) - Plot", size=(500, 200))

area.addDock(d1, 'left')
area.addDock(d2, 'bottom', d1)
area.addDock(d3, 'right', d2)
# area.addDock(d4, 'right')
area.addDock(d5, 'right', d1)
# area.addDock(d6, 'top', d4)

vlc_widget_list = []
vlc_widget1 = VLCWidget()
vlc_widget_list.append(vlc_widget1)
d1.addWidget(vlc_widget1)
vlc_widget1.media = "test_video_emotion.avi"
vlc_widget1.play()

# vlc_widget2 = VLCWidget()
# d3.addWidget(vlc_widget2)
# vlc_widget_list.append(vlc_widget2)
# vlc_widget2.media = "low_fps.mp4"
# vlc_widget2.play()

d2.addWidget(VLCControl(vlc_widget_list))
d3.addWidget(TranscriptWidget(vlc_widget1, transcript_csv="transcript.csv"))

# summary = pg.ImageView()
# img = np.array(Image.open("summary_dummy.png")).T
# summary.setImage(img)
# summary.ui.histogram.hide()
# summary.ui.roiBtn.hide()
# summary.ui.roiPlot.hide()
# summary.ui.menuBtn.hide()
# d5.addWidget(summary)

summary = SummaryWidget()
d5.addWidget(summary)
# w5 = pg.ImageView()
# w5.setImage(np.random.normal(size=(100, 100)))
# d5.addWidget(w5)
#
# w6 = pg.PlotWidget(title="Dock 6 plot")
# w6.plot(np.random.normal(size=100))
# d6.addWidget(w6)

win.showMaximized()

if __name__ == '__main__':
    pg.mkQApp().exec_()
