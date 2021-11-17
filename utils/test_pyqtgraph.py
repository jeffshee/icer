import PyQt5.QtWidgets as pq
import pyqtgraph as pg


class MainWindow(pq.QWidget):
    def __init__(self):
        pq.QWidget.__init__(self)
        self.setFixedSize(500, 300)
        self.layout = pq.QGridLayout()

        self.button_manual = pq.QPushButton('Draw')
        self.button_manual.clicked.connect(self.draw_manual)

        self.layout.addWidget(self.button_manual)
        self.setLayout(self.layout)

        self.show()

    def draw_manual(self):
        y = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]
        x = range(1, len(y) + 1)  # when you change y values, x will automatically have the correct x values
        dialog = Dialog(x, y)  # create the dialog ...
        dialog.exec_()  # ... and show it


class Dialog(pq.QDialog):
    def __init__(self, x, y):
        pq.QDialog.__init__(self)
        self.layout = pq.QHBoxLayout()  # create a layout to add the plotwidget to
        self.graphWidget = pg.PlotWidget()
        self.layout.addWidget(self.graphWidget)  # add the widget to the layout
        self.setLayout(self.layout)  # and set the layout on the dialog
        self.graphWidget.plot(x, y)


if __name__ == "__main__":  # if run as a program, ad not imported ...
    app = pq.QApplication([])  # create ONE application (in your original code you had two, thats why it crashed)
    win = MainWindow()  # create the main window
    app.exec_()  # run the app

