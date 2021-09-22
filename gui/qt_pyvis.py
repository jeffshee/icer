import sys
import os
from PyQt5.Qt import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication
from pyvis import network as net
import networkx as nx


class PyVisWidget(QWebEngineView):
    def __init__(self, networkx_graph=None):
        super().__init__()
        html_path = "networkx.html"
        if networkx_graph is None:
            # Generate dummy network graph
            networkx_graph = nx.complete_graph(5)
        # Render HTML file
        g = net.Network()
        g.from_nx(networkx_graph)
        g.write_html(html_path)
        self.load(QUrl.fromLocalFile(os.path.abspath(html_path)))


class Window(QMainWindow):
    def __init__(self, networkx_graph=None, title="PyVis Viewer"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(0, 0, 800, 800)
        self.setCentralWidget(PyVisWidget(networkx_graph))
        self.show()


def show_pyvis(networkx_graph=None, title="PyVis Viewer"):
    app = QApplication(sys.argv)
    window = Window(networkx_graph, title)
    sys.exit(app.exec_())


if __name__ == "__main__":
    show_pyvis()
