import shutil
import sys
import os
import tempfile

from PyQt5.Qt import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication
from pyvis import network as net
import networkx as nx


class PyVisWidget(QWebEngineView):
    def __init__(self, networkx_graph=None):
        super().__init__()
        self.temp_dirname = tempfile.mkdtemp()
        print("Temporary directory created at", self.temp_dirname)
        html_path = os.path.join(self.temp_dirname, "networkx.html")
        if networkx_graph is None:
            # Generate dummy network graph
            networkx_graph = nx.complete_graph(8)
        # Render HTML file
        g = net.Network()
        g.from_nx(networkx_graph)
        g.write_html(html_path)
        self.load(QUrl.fromLocalFile(os.path.abspath(html_path)))

    def clean(self):
        # Remove temp directory
        if os.path.exists(self.temp_dirname):
            shutil.rmtree(self.temp_dirname)


class PyVisWindow(QMainWindow):
    def __init__(self, networkx_graph=None, title="PyVis Viewer"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(0, 0, 520, 520)
        self.widget = PyVisWidget(networkx_graph)
        self.setCentralWidget(self.widget)
        self.show()

    def clean(self):
        self.widget.clean()


def show_pyvis(networkx_graph=None, title="PyVis Viewer"):
    app = QApplication.instance() or QApplication(sys.argv)
    window = PyVisWindow(networkx_graph, title)
    app.exec_()
    window.clean()


if __name__ == "__main__":
    show_pyvis()
