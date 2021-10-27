import sys
import os
from PyQt5.Qt import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication
from pyvis import network as net
import networkx as nx


class PyVisWidget(QWebEngineView):
    def __init__(self, kwargs,matrix,networkx_graph=None ):
        super().__init__()
        html_path = "networkx.html"
        g = net.Network(directed=False)
        for i in range(kwargs["speak_num"]):
            g.add_node(i)

        for i in range(len(matrix)):
            for j in range(len(matrix[0]) - i - 1):
                # print(i, j + i + 1)
                if matrix[i][j + i + 1] != 0:
                    g.add_edge(i, j + i + 1, value=matrix[i][j + i + 1])
        g.write_html(html_path)
        self.load(QUrl.fromLocalFile(os.path.abspath(html_path)))

class Window(QMainWindow):
    def __init__(self, kwargs,matrix,networkx_graph=None,title="PyVis Viewer"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(0, 0, 800, 800)
        self.setCentralWidget(PyVisWidget(kwargs,matrix,networkx_graph))
        self.show()


def show_pyvis(kwargs,matrix,networkx_graph=None, title="PyVis Viewer"):
    app = QApplication(sys.argv)
    window = Window(kwargs,matrix,networkx_graph, title)
    sys.exit(app.exec_())


if __name__ == "__main__":
    show_pyvis()
