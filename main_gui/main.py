import os
import sys
from multiprocessing import freeze_support
from pprint import pprint

# OpenCV2+PyQt5 issue workaround for Linux
# https://forum.qt.io/topic/119109/using-pyqt5-with-opencv-python-cv2-causes-error-could-not-load-qt-platform-plugin-xcb-even-though-it-was-found/21
from PyQt5 import QtGui
from cv2.version import ci_build, headless

from main_gui.utils import *

ci_and_not_headless = ci_build and not headless
if sys.platform.startswith("linux") and ci_and_not_headless:
    try:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
        os.environ.pop("QT_QPA_FONTDIR")
    except KeyError:
        pass
import numpy as np
from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush, QImage, QIcon, QKeySequence
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QInputDialog, \
    QHBoxLayout, QGroupBox, QListWidgetItem, \
    QSlider, QScrollArea, QListWidget, QAbstractItemView, QPushButton, QFileDialog, QMessageBox, QAction

final_result = None


def cm_tab10(i: int):
    # Colormap tab10 taken from matplotlib
    tab10 = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
             (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
             (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
             (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
             (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0),
             (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
             (0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0),
             (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
             (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0),
             (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0)]
    return tab10[i % 10]


# # Zoom in/out with scrollbar adjustment
# # Reference:
# # https://docs.huihoo.com/qt/4.5/widgets-imageviewer.html
# def adjust_scrollbar(scrollbar: QScrollBar, factor):
#     scrollbar.setValue(int(factor * scrollbar.value() + (factor - 1) * scrollbar.pageStep() / 2))

class RoiWidget(QLabel):
    RoiChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._begin = QPoint()
        self._end = QPoint()
        self._roi_dict = dict()
        self._cur_key = None

        self._cm_count = 0

    def add(self, name) -> bool:
        # Name existed, return false as failed
        if name in self._roi_dict:
            return False

        # Assign new color
        self._cm_count += 1
        r, g, b, _ = cm_tab10(self._cm_count)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)

        # Create new entry
        self._roi_dict[name] = dict(rgb=(r, g, b), roi=QRect())
        self._cur_key = name
        print(f"[ROI] add", name)
        return True

    def switch_to(self, name) -> bool:
        # Name not found, return false as failed
        if name not in self._roi_dict:
            return False
        print(f"[ROI] switch_to", name)
        self._cur_key = name
        self.RoiChanged.emit()
        return True

    def remove(self, name):
        # Name not found, return false as failed
        if name not in self._roi_dict.keys():
            return False

        # Remove entry from dict
        del self._roi_dict[name]
        print(f"[ROI] remove", name)
        self._cur_key = None
        self.RoiChanged.emit()

        # Redraw
        self.update()
        return True

    def get_current(self):
        return self._cur_key

    def get_roi(self, name):
        if name in self._roi_dict:
            return self._roi_dict[name]["roi"]
        return QRect()

    def get_rgb(self, name):
        if name in self._roi_dict:
            return self._roi_dict[name]["rgb"]
        return None

    def get_result(self):
        """
        :return: Format: (x,y,w,h)
        """
        return self._roi_dict

    def update_roi(self):
        # Update current ROI
        cur_item = self._roi_dict[self._cur_key]
        # Normalized (Ensure that no negative width and height)
        rect_roi = QRect(self._begin, self._end).normalized()
        # Ensure in-bound, get intersected with the image's rect
        # TODO upstream the bugfix
        rect_roi = rect_roi.intersected(self.pixmap().rect())
        cur_item["roi"] = rect_roi
        # Debug
        # print("[ROI]", self._roi_dict[self._cur_key])
        self.RoiChanged.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._roi_dict:
            # Get current scaling
            scale = self.width() / self.pixmap().width()
            # Draw all ROIs
            painter = QPainter(self)
            for name, item in self._roi_dict.items():
                rgb, roi = item["rgb"], item["roi"]
                brush = QBrush(QColor(*rgb, 100))
                painter.setBrush(brush)
                roi_scaled = QRect(int(roi.x() * scale), int(roi.y() * scale),
                                   int(roi.width() * scale), int(roi.height() * scale))
                painter.drawRect(roi_scaled)
                painter.drawText(roi_scaled.topLeft(), name)

    def mousePressEvent(self, event):
        if self._roi_dict:
            # Get current scaling
            scale = self.width() / self.pixmap().width()
            self._begin = event.pos() / scale
            self._end = event.pos() / scale
            self.update_roi()
            self.update()

    def mouseMoveEvent(self, event):
        if self._roi_dict:
            # Get current scaling
            scale = self.width() / self.pixmap().width()
            self._end = event.pos() / scale
            self.update_roi()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._roi_dict:
            # Get current scaling
            scale = self.width() / self.pixmap().width()
            self._end = event.pos() / scale
            self.update_roi()
            self.update()


def cv2_to_qpixmap(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytes_per_line = 3 * width
    qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    qpixmap = QPixmap.fromImage(qimage)
    return qpixmap


class ZoomScrollArea(QScrollArea):
    def __init__(self, scale_min=0.5, scale_max=2.0, step=0.1):
        super().__init__()
        self.scale_factor_min = scale_min
        self.scale_factor_max = scale_max
        self.scale_factor_step = step
        self._scale_factor = 0.5

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.modifiers() == Qt.ControlModifier:
            prev_scale_factor = self._scale_factor
            if event.angleDelta().y() > 0:
                self._scale_factor += self.scale_factor_step
            else:
                self._scale_factor -= self.scale_factor_step
            self._scale_factor = max(min(self._scale_factor, self.scale_factor_max), self.scale_factor_min)
            # print("[Zoom] {:.1f}".format(self._scale_factor))
            cur_size = self.widget().size()
            original_size = cur_size / prev_scale_factor
            new_size = original_size * self._scale_factor
            """
            Reference: https://stackoverflow.com/questions/3725342/zooming-in-out-on-a-mouser-point/32269574
            QPointF ScrollbarPos = QPointF(horizontalScrollBar()->value(), verticalScrollBar()->value());
            QPointF DeltaToPos = e->posF() / OldScale - widget()->pos() / OldScale;
            QPointF Delta = DeltaToPos * NewScale - DeltaToPos * OldScale;
            """
            scrollbar_pos = QPoint(self.horizontalScrollBar().value(), self.verticalScrollBar().value())
            delta_to_pos = event.pos() / prev_scale_factor - self.widget().pos() / prev_scale_factor
            delta = delta_to_pos * self._scale_factor - delta_to_pos * prev_scale_factor
            self.widget().resize(new_size)
            # Adjust Scrollbar
            self.horizontalScrollBar().setValue(scrollbar_pos.x() + delta.x())
            self.verticalScrollBar().setValue(scrollbar_pos.y() + delta.y())
        else:
            super().wheelEvent(event)

    def get_scale_factor(self):
        return self._scale_factor


class ControlPanel(QVBoxLayout):
    def __init__(self):
        super().__init__()
        # Audio
        self.audio_button = QPushButton("Browse")
        self.audio_path_label = QLabel("Not selected")
        self.group_box_audio = QGroupBox(f"PinMic Audio")
        hbox = QHBoxLayout()
        hbox.addWidget(self.audio_button)
        hbox.addWidget(self.audio_path_label, stretch=2)
        self.group_box_audio.setLayout(hbox)
        self.addWidget(self.group_box_audio)

        # Offset
        self.offset_slider = QSlider(Qt.Horizontal)
        self.group_box_offset = QGroupBox(f"Offset {self.offset_slider.value()}")
        hbox = QHBoxLayout()
        hbox.addWidget(self.offset_slider)
        self.group_box_offset.setLayout(hbox)
        self.addWidget(self.group_box_offset)

        # Frame
        self.frame_slider = QSlider(Qt.Horizontal)
        self.group_box_frame = QGroupBox(f"Frame {self.frame_slider.value()}")
        hbox = QHBoxLayout()
        hbox.addWidget(self.frame_slider)
        self.group_box_frame.setLayout(hbox)
        self.addWidget(self.group_box_frame)

    def enable_controls(self, enabled):
        self.group_box_audio.setEnabled(enabled)


class RoiPanel(QVBoxLayout):
    def __init__(self):
        super().__init__()
        # List
        self.roi_selector = QListWidget()
        self.roi_selector.setSelectionMode(QAbstractItemView.SingleSelection)

        # Buttons
        hbox_button = QHBoxLayout()
        self.button_add = QPushButton("Add ROI <Ctrl+A>")
        hbox_button.addWidget(self.button_add)
        self.button_remove = QPushButton("Remove ROI <Ctrl+R>")
        hbox_button.addWidget(self.button_remove)

        vbox = QVBoxLayout()
        vbox.addWidget(self.roi_selector)
        vbox.addLayout(hbox_button)
        group_box_roi_selection = QGroupBox("ROI Selection")
        group_box_roi_selection.setLayout(vbox)
        self.addWidget(group_box_roi_selection)


class MainWindow(QMainWindow):
    def __init__(self, video_path: str):
        super().__init__()
        self.setWindowTitle(f"ROIs selector \"{os.path.realpath(video_path)}\"")
        self.video_path = video_path
        self._video_capture = cv2.VideoCapture(video_path)
        self._video_length = get_video_length(self._video_capture)
        self._video_dimension = get_video_dimension(self._video_capture)
        self._frame = None
        self._frame_pos = 0
        self._offset = 0
        self._audio_path = None
        self.person_count = 0

        self._result_dict = dict()
        self.src_scale_factor = 1.0

        self.layout = QVBoxLayout()

        hbox_display = QHBoxLayout()
        # Source image
        self.roi_widget = RoiWidget()
        self.roi_widget.setScaledContents(True)
        self.src_scroll_area = ZoomScrollArea()
        self.src_scroll_area.setWidget(self.roi_widget)

        group_box_src = QGroupBox("Source Image <Ctrl+Scroll to zoom-in/out>")
        hbox = QHBoxLayout()
        hbox.addWidget(self.src_scroll_area)
        group_box_src.setLayout(hbox)
        hbox_display.addWidget(group_box_src)
        self.layout.addLayout(hbox_display, stretch=5)

        # Panels
        hbox_roi_ctrl = QHBoxLayout()
        self.roi_panel = RoiPanel()
        hbox_roi_ctrl.addLayout(self.roi_panel)
        self.control_panel = ControlPanel()
        hbox_roi_ctrl.addLayout(self.control_panel)
        self.layout.addLayout(hbox_roi_ctrl, stretch=1)

        # End selection button
        self.button_ok = QPushButton("End selection")
        self.layout.addWidget(self.button_ok)

        # Shortcuts
        action = QAction("Add ROI", self)
        action.triggered.connect(self.on_clicked_add)
        action.setShortcut(QKeySequence("Ctrl+A"))
        self.addAction(action)

        action = QAction("Remove ROI", self)
        action.triggered.connect(self.on_clicked_remove)
        action.setShortcut(QKeySequence("Ctrl+R"))
        self.addAction(action)

        # Callbacks
        self.roi_panel.button_add.clicked.connect(self.on_clicked_add)
        self.roi_panel.button_remove.clicked.connect(self.on_clicked_remove)
        self.roi_panel.roi_selector.itemSelectionChanged.connect(self.on_item_selection_changed)
        self.control_panel.audio_button.clicked.connect(self.on_clicked_audio)
        self.control_panel.offset_slider.valueChanged.connect(self.on_value_changed_offset)
        self.control_panel.frame_slider.valueChanged.connect(self.on_value_changed_frame)
        self.button_ok.clicked.connect(self.on_clicked_end)

        # Defaults
        self.control_panel.frame_slider.setMaximum(self._video_length)
        self.control_panel.offset_slider.setRange(self._video_dimension[0] // 2 * -1, self._video_dimension[0] // 2)
        self.control_panel.enable_controls(False)
        self._load_source()

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        self.showMaximized()

    def _new_current(self):
        current_name = self.roi_widget.get_current()
        if current_name:
            current = dict(audio_path=None)
            self._result_dict[current_name] = current
            print("[GUI] new_current")
            pprint(self._result_dict[current_name])

    def _save_current(self):
        current_name = self.roi_widget.get_current()
        if current_name:
            current = dict(audio_path=self._audio_path)
            self._result_dict[current_name] = current
            print("[GUI] save_current")
            pprint(self._result_dict[current_name])

    def _load_current(self):
        current_name = self.roi_widget.get_current()
        if current_name:
            saved = self._result_dict[current_name]
            self._audio_path = saved["audio_path"]
            # Refresh
            if self._audio_path is None:
                self.control_panel.audio_path_label.setText("Not selected")
            else:
                self.control_panel.audio_path_label.setText(self._audio_path)
            print("[GUI] load_current")

    def _remove_current(self):
        current_name = self.roi_widget.get_current()
        if current_name:
            print("[GUI] remove_current")
            pprint(self._result_dict[current_name])
            del self._result_dict[current_name]

    def _load_source(self):
        set_frame_position(self._video_capture, self._frame_pos)
        ret, frame = read_video_capture(self._video_capture, self._offset)
        if ret:
            self._frame = frame
            h, w, c = frame.shape
            self.roi_widget.setPixmap(cv2_to_qpixmap(frame))
            new_size = QSize(w, h) * self.src_scroll_area.get_scale_factor()
            self.roi_widget.resize(new_size)
        else:
            self.roi_widget.clear()

    def on_value_changed_frame(self, val):
        self.control_panel.group_box_frame.setTitle(f"Frame {val}")
        self._frame_pos = val
        self._load_source()

    def on_value_changed_offset(self, val):
        self.control_panel.group_box_offset.setTitle(f"Offset {val}")
        self._offset = val
        self._load_source()

    def on_clicked_audio(self):
        audio_path = get_audio_path()
        if audio_path:
            self._audio_path = audio_path
            self.control_panel.audio_path_label.setText(audio_path)
            self._save_current()

    def on_clicked_end(self):
        global final_result
        gui_result = self.get_result()
        if gui_result:
            print("[GUI] end_selection")
            pprint(gui_result)
            final_result = gui_result
            QApplication.quit()

    def on_clicked_add(self):
        name, ret = QInputDialog.getText(self, "Add ROI", "Enter a name:", text=f"Person{(self.person_count + 1):02}")
        if ret:
            if name:
                success = self.roi_widget.add(name)
                if success:
                    self._new_current()
                    pixmap = QPixmap(16, 16)
                    pixmap.fill(QColor().fromRgb(*self.roi_widget.get_rgb(name)))
                    icon = QIcon(pixmap)
                    list_item = QListWidgetItem(icon, name)
                    self.roi_panel.roi_selector.addItem(list_item)
                    self.roi_panel.roi_selector.setCurrentItem(list_item)
                    self.control_panel.enable_controls(True)
                    self.person_count += 1
                else:
                    QMessageBox.critical(self, "Error", "Name already exists!")
            else:
                QMessageBox.critical(self, "Error", "Invalid name!")

    def on_clicked_remove(self):
        self._remove_current()
        cur_item = self.roi_panel.roi_selector.currentItem()
        if cur_item:
            name = cur_item.text()
            row = self.roi_panel.roi_selector.row(cur_item)
            success = self.roi_widget.remove(name)
            if success:
                self.roi_panel.roi_selector.takeItem(row)
                self.person_count -= 1
                if self.roi_panel.roi_selector.currentItem() is None:
                    # Last item
                    self.control_panel.enable_controls(False)

    def on_item_selection_changed(self):
        cur_item = self.roi_panel.roi_selector.currentItem()
        if cur_item:
            name = cur_item.text()
            success = self.roi_widget.switch_to(name)
            if success:
                self._load_current()

    def get_result(self):
        ret_roi = self.roi_widget.get_result()
        ret_param = self._result_dict
        assert ret_roi.keys() == ret_param.keys()
        combined = dict()
        for key in ret_roi.keys():
            combined[key] = {**ret_roi[key], **ret_param[key]}
            # Convert ROI to tuple
            roi = combined[key]["roi"]
            combined[key]["roi"] = roi.x(), roi.y(), roi.width(), roi.height()
        return dict(video_path=self.video_path, offset=self._offset, result=combined)


def get_video_path():
    video_path = QFileDialog.getOpenFileName(caption="Open Video", filter="Videos (*.mp4 *.avi)")[0]
    return video_path


def get_audio_path():
    audio_path = QFileDialog.getOpenFileName(caption="Open Audio", filter="Audios (*.mp3 *.wav)")[0]
    return audio_path


def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    freeze_support()
    video_path = get_video_path()
    if video_path:
        print(f"[Video] {video_path}")
        window = MainWindow(video_path)
        window.show()
        app.exec()
    return final_result


if __name__ == "__main__":
    main()
