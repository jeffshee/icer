# -*- coding: utf-8 -*-
"""
This example demonstrates the use of pyqtgraph's dock widget system.

The dockarea system allows the design of user interfaces which can be rearranged by
the user at runtime. Docks can be moved, resized, stacked, and torn out of the main
window. This is similar in principle to the docking system built into Qt, but
offers a more deterministic dock placement API (in Qt it is very difficult to
programatically generate complex dock arrangements). Additionally, Qt's docks are
designed to be used as small panels around the outer edge of a window. Pyqtgraph's
docks were created with the notion that the entire window (or any portion of it)
would consist of dockable components.

"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.console
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QMainWindow, QWidget, QFrame, QSlider, QHBoxLayout, QPushButton, \
    QVBoxLayout, QAction, QFileDialog, QApplication

from pyqtgraph.dockarea import *

app = pg.mkQApp("Overlay")
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1000, 500)
win.setWindowTitle("Overlay")

d1 = Dock("Emotion", size=(1600, 900))
d2 = Dock("Dock2 - Console", size=(500, 300), closable=True)
d3 = Dock("Control", size=(1, 1))
d4 = Dock("Dock4 (tabbed) - Plot", size=(500, 200))
d5 = Dock("Dock5 - Image", size=(500, 200))
d6 = Dock("Dock6 (tabbed) - Plot", size=(500, 200))
area.addDock(d1,
             'left')  ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
area.addDock(d2, 'right')  ## place d2 at right edge of dock area
area.addDock(d3, 'bottom', d1)  ## place d3 at bottom edge of d1
area.addDock(d4, 'right')  ## place d4 at right edge of dock area
area.addDock(d5, 'left', d1)  ## place d5 at left edge of d1
area.addDock(d6, 'top', d4)  ## place d5 at top edge of d4

## Test ability to move docks programatically after they have been placed
area.moveDock(d4, 'top', d2)  ## move d4 to top edge of d2
area.moveDock(d6, 'above', d4)  ## move d6 to stack on top of d4
area.moveDock(d5, 'top', d2)  ## move d5 to top edge of d2

## Add widgets into each dock

## first dock gets save/restore buttons
import vlc

vlc_widgets = []


class VLCWidget(QFrame):
    def __init__(self):
        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()
        super(VLCWidget, self).__init__()

    def set_media(self, video_path):
        self.media_player.set_media(self.instance.media_new(video_path))
        self.media_player.set_xwindow(self.winId())

    def play(self):
        self.media_player.play()

    def pause(self):
        self.media_player.pause()

    def stop(self):
        self.media_player.stop()


class VLCControl(QWidget):
    def __init__(self):
        super(VLCControl, self).__init__()
        self.slider_position = QSlider(Qt.Horizontal, self)
        self.slider_position.setToolTip("Position")
        self.slider_position.setMaximum(1000)
        self.slider_position.sliderMoved.connect(set_position)

        self.hbox = QHBoxLayout()
        self.button_play = QPushButton("Play")
        self.hbox.addWidget(self.button_play)
        self.button_play.clicked.connect(play_pause)

        self.button_stop = QPushButton("Stop")
        self.hbox.addWidget(self.button_stop)
        self.button_stop.clicked.connect(stop)

        self.hbox.addStretch(1)
        self.slider_volume = QSlider(Qt.Horizontal, self)
        self.slider_volume.setMaximum(100)
        self.slider_volume.setValue(0)
        self.slider_volume.setToolTip("Volume")
        self.hbox.addWidget(self.slider_volume)
        self.slider_volume.valueChanged.connect(set_volume)

        self.vboxlayout = QVBoxLayout()
        self.vboxlayout.addWidget(self.slider_position)
        self.vboxlayout.addLayout(self.hbox)
        self.setLayout(self.vboxlayout)


def set_position(*args):
    print("set_position")


def play_pause(*args):
    print("play_pause")


def stop(*args):
    print("stop")


def set_volume(*args):
    print("set_volume")


vlc_widget1 = VLCWidget()
vlc_widgets.append(vlc_widget1)
d1.addWidget(vlc_widget1)
vlc_widget1.set_media("test_video_emotion.avi")
vlc_widget1.play()

vlc_widget2 = VLCWidget()
d2.addWidget(vlc_widget2)
vlc_widgets.append(vlc_widget2)
vlc_widget2.set_media("low_fps.mp4")
vlc_widget2.play()

d3.addWidget(VLCControl())

w4 = pg.PlotWidget(title="Dock 4 plot")
w4.plot(np.random.normal(size=100))
d4.addWidget(w4)

w5 = pg.ImageView()
w5.setImage(np.random.normal(size=(100, 100)))
d5.addWidget(w5)

w6 = pg.PlotWidget(title="Dock 6 plot")
w6.plot(np.random.normal(size=100))
d6.addWidget(w6)

win.show()

if __name__ == '__main__':
    pg.mkQApp().exec_()
