import faulthandler
import sys

import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QBrush, QColor, QPen, QPixmap, QPainterPath, QPainter, QImage
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsItem, QGraphicsPathItem, QApplication, QGraphicsView, \
    QGraphicsScene, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout

from gui.dialogs import get_video_path
from utils.video_utils import get_video_capture_with_offset

faulthandler.enable()


class HandleItem(QGraphicsRectItem):
    def __init__(self, position_flags, parent):
        QGraphicsRectItem.__init__(self, -20, -20, 40, 40, parent)
        self._positionFlags = position_flags

        self.setBrush(QBrush(QColor(81, 168, 220, 200)))
        self.setPen(QPen(QColor(0, 0, 0, 255), 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        self.setFlag(self.ItemIsMovable)
        self.setFlag(self.ItemSendsGeometryChanges)

    def positionflags(self):
        return self._positionFlags

    def itemChange(self, change, value):
        retVal = value
        if change == self.ItemPositionChange:
            retVal = self.restrictPosition(value)
        elif change == self.ItemPositionHasChanged:
            pos = value
            if self.positionflags() == SizeGripItem.TopLeft:
                self.parentItem().setTopLeft(pos)
            elif self.positionflags() == SizeGripItem.Top:
                self.parentItem().setTop(pos.y())
            elif self.positionflags() == SizeGripItem.TopRight:
                self.parentItem().setTopRight(pos)
            elif self.positionflags() == SizeGripItem.Right:
                self.parentItem().setRight(pos.x())
            elif self.positionflags() == SizeGripItem.BottomRight:
                self.parentItem().setBottomRight(pos)
            elif self.positionflags() == SizeGripItem.Bottom:
                self.parentItem().setBottom(pos.y())
            elif self.positionflags() == SizeGripItem.BottomLeft:
                self.parentItem().setBottomLeft(pos)
            elif self.positionflags() == SizeGripItem.Left:
                self.parentItem().setLeft(pos.x())
        return retVal

    def restrictPosition(self, newPos):
        retVal = self.pos()

        if self.positionflags() & SizeGripItem.Top or self.positionflags() & SizeGripItem.Bottom:
            retVal.setY(newPos.y())

        if self.positionflags() & SizeGripItem.Left or self.positionflags() & SizeGripItem.Right:
            retVal.setX(newPos.x())

        if self.positionflags() & SizeGripItem.Top and retVal.y() > self.parentItem()._rect.bottom():
            retVal.setY(self.parentItem()._rect.bottom())

        elif self.positionflags() & SizeGripItem.Bottom and retVal.y() < self.parentItem()._rect.top():
            retVal.setY(self.parentItem()._rect.top())

        if self.positionflags() & SizeGripItem.Left and retVal.x() > self.parentItem()._rect.right():
            retVal.setX(self.parentItem()._rect.right())

        elif self.positionflags() & SizeGripItem.Right and retVal.x() < self.parentItem()._rect.left():
            retVal.setX(self.parentItem()._rect.left())

        return retVal


class SizeGripItem(QGraphicsItem):
    Top = 0x01
    Bottom = 0x1 << 1
    Left = 0x1 << 2
    Right = 0x1 << 3
    TopLeft = Top | Left
    BottomLeft = Bottom | Left
    TopRight = Top | Right
    BottomRight = Bottom | Right

    handleCursors = {
        TopLeft: Qt.SizeFDiagCursor,
        Top: Qt.SizeVerCursor,
        TopRight: Qt.SizeBDiagCursor,
        Left: Qt.SizeHorCursor,
        Right: Qt.SizeHorCursor,
        BottomLeft: Qt.SizeBDiagCursor,
        Bottom: Qt.SizeVerCursor,
        BottomRight: Qt.SizeFDiagCursor,
    }

    def __init__(self, parent):
        QGraphicsItem.__init__(self, parent)
        self._handleItems = []

        self._rect = QRectF(0, 0, 0, 0)
        if self.parentItem():
            self._rect = self.parentItem().rect()

        for flag in (self.TopLeft, self.Top, self.TopRight, self.Right,
                     self.BottomRight, self.Bottom, self.BottomLeft, self.Left):
            handle = HandleItem(flag, self)
            handle.setCursor(self.handleCursors[flag])
            self._handleItems.append(handle)

        self.updateHandleItemPositions()

    def boundingRect(self):
        if self.parentItem():
            return self._rect
        else:
            return QRectF(0, 0, 0, 0)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(127, 127, 127), 2.0, Qt.DashLine))
        painter.drawRect(self._rect)

    def doResize(self):
        self.parentItem().setRect(self._rect)
        self.updateHandleItemPositions()

    def updateHandleItemPositions(self):
        for item in self._handleItems:
            item.setFlag(QGraphicsItem.ItemSendsGeometryChanges, False)

            if item.positionflags() == self.TopLeft:
                item.setPos(self._rect.topLeft())
            elif item.positionflags() == self.Top:
                item.setPos(self._rect.left() + self._rect.width() / 2 - 1,
                            self._rect.top())
            elif item.positionflags() == self.TopRight:
                item.setPos(self._rect.topRight())
            elif item.positionflags() == self.Right:
                item.setPos(self._rect.right(),
                            self._rect.top() + self._rect.height() / 2 - 1)
            elif item.positionflags() == self.BottomRight:
                item.setPos(self._rect.bottomRight())
            elif item.positionflags() == self.Bottom:
                item.setPos(self._rect.left() + self._rect.width() / 2 - 1,
                            self._rect.bottom())
            elif item.positionflags() == self.BottomLeft:
                item.setPos(self._rect.bottomLeft())
            elif item.positionflags() == self.Left:
                item.setPos(self._rect.left(),
                            self._rect.top() + self._rect.height() / 2 - 1)
            item.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

    def setTop(self, v):
        self._rect.setTop(v)
        self.doResize()

    def setRight(self, v):
        self._rect.setRight(v)
        self.doResize()

    def setBottom(self, v):
        self._rect.setBottom(v)
        self.doResize()

    def setLeft(self, v):
        self._rect.setLeft(v)
        self.doResize()

    def setTopLeft(self, v):
        self._rect.setTopLeft(v)
        self.doResize()

    def setTopRight(self, v):
        self._rect.setTopRight(v)
        self.doResize()

    def setBottomRight(self, v):
        self._rect.setBottomRight(v)
        self.doResize()

    def setBottomLeft(self, v):
        self._rect.setBottomLeft(v)
        self.doResize()


class CropItem(QGraphicsPathItem):
    def __init__(self, parent):
        QGraphicsPathItem.__init__(self, parent)
        self.extern_rect = parent.boundingRect()
        self.intern_rect = QRectF(0, 0, self.extern_rect.width() / 2, self.extern_rect.height() / 2)
        self.intern_rect.moveCenter(self.extern_rect.center())
        self.setBrush(QColor(10, 100, 100, 100))
        self.setPen(QPen(Qt.NoPen))
        SizeGripItem(self)
        self.create_path()

    def create_path(self):
        self._path = QPainterPath()
        self._path.addRect(self.extern_rect)
        self._path.moveTo(self.intern_rect.topLeft())
        self._path.addRect(self.intern_rect)
        self.setPath(self._path)

    def rect(self):
        return self.intern_rect

    def setRect(self, rect):
        self._intern = rect
        self.create_path()


def cv2_to_qpixmap(img: np.ndarray):
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytes_per_line = 3 * width
    qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    qpixmap = QPixmap.fromImage(qimage)
    return qpixmap


class CropperGUI(QMainWindow):
    def __init__(self, img: np.ndarray, window_title=None, cancelable=True):
        super().__init__()
        self.img = img
        self.img_scale = 0.8
        self.h, self.w, self.c = img.shape

        # --view
        # ----scene
        # ------pixmapItem
        layout = QVBoxLayout()
        view = QGraphicsView()
        scene = QGraphicsScene()
        view.setScene(scene)

        pixmapItem = scene.addPixmap(cv2_to_qpixmap(img))
        self.cropItem = CropItem(pixmapItem)

        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        view.setFixedSize(int(1920 * self.img_scale), int(1080 * self.img_scale))
        view.fitInView(0, 0, int(1920 * self.img_scale), int(1080 * self.img_scale), Qt.KeepAspectRatio)
        # view.fitInView(QRectF(QApplication.desktop().availableGeometry(-1)), Qt.KeepAspectRatio)
        # view.setFixedSize(view.size())

        layout.addWidget(view)

        # Buttons
        btn_cancel = QPushButton("指定しない")
        btn_cancel.clicked.connect(lambda _: QApplication.quit())
        btn_ok = QPushButton("確定")
        # Default button
        btn_ok.setDefault(True)
        btn_ok.setAutoDefault(True)
        btn_ok.clicked.connect(self.on_clicked_ok)

        btn_layout = QHBoxLayout()
        # btn_layout.setAlignment(Qt.AlignRight)
        if cancelable:
            btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_ok)

        layout.addLayout(btn_layout)
        widget = QWidget(self)
        widget.setLayout(layout)

        self.setCentralWidget(widget)
        self.setWindowTitle(window_title if window_title else "動画の切取り領域の指定")

        self.roi = None

    def _get_roi(self):
        self.roi = roi = self.cropItem.rect()
        roi_x, roi_y, roi_w, roi_h = int(roi.x()), int(roi.y()), int(roi.width()), int(roi.height())
        # In-bound check
        roi_x, roi_y = max(0, min(roi_x, self.w)), max(0, min(roi_y, self.h))
        roi_w, roi_h = max(0, min(roi_w, self.w - roi_x)), max(0, min(self.h, self.h - roi_y))
        return roi_x, roi_y, roi_w, roi_h

    def on_clicked_ok(self):
        self.roi = self._get_roi()
        QApplication.quit()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.roi = self._get_roi()
            QApplication.quit()


def select_roi(video_path: str = None, offset=0):
    """
    Launch a GUI for selecting the ROI
    :param video_path: Video path (Optional)
    :param offset: Offset
    :return:
    """

    if not video_path:
        video_path = get_video_path()
    video_capture = get_video_capture_with_offset(video_path, offset)
    app = QApplication(sys.argv)
    _, frame = video_capture.read()
    win = CropperGUI(frame)
    win.show()
    app.exec_()
    roi = win.roi
    print(f"ROI: {roi}")
    return roi


if __name__ == "__main__":
    select_roi()

"""
Deprecated
"""
# class MainWindow(QGraphicsView):
#     def __init__(self):
#         super().__init__()
#
#     def keyPressEvent(self, event):
#         global enter_flag
#         if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
#             enter_flag = True
#             QApplication.quit()


# enter_flag = False
#
#
# def selectROI(img, title="ROI Selector", message=None):
#     # HIDPI support
#     os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
#
#     app = QApplication(sys.argv)
#     app.setApplicationName(title)
#
#     view = MainWindow()
#     scene = QGraphicsScene()
#     view.setScene(scene)
#
#     height, width, channel = img.shape
#     bytesPerLine = 3 * width
#     qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
#
#     pixmapItem = scene.addPixmap(QPixmap.fromImage(qImg))
#     cropItem = CropItem(pixmapItem)
#
#     # view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
#     view.fitInView(QRectF(QApplication.desktop().availableGeometry(-1)), Qt.KeepAspectRatio)
#     view.show()
#     view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#     view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#     view.setFixedSize(view.size())
#
#     # Show message
#     if message is not None:
#         msg = QMessageBox()
#         msg.setIcon(QMessageBox.Information)
#         msg.setText(message)
#         msg.setStandardButtons(QMessageBox.Ok)
#         msg.exec_()
#
#     app.exec_()
#     if enter_flag:
#         roi = cropItem.rect()
#         x, y, w, h = int(roi.x()), int(roi.y()), int(roi.width()), int(roi.height())
#         # In-bound check
#         x, y = max(0, min(x, width)), max(0, min(y, height))
#         w, h = max(0, min(w, width - x)), max(0, min(h, height - y))
#         return x, y, w, h
#     else:
#         return int(0), int(0), width, height
