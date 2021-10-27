import sys
import os
from PyQt5.Qt import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication
from pyvis import network as net
import networkx as nx


class PyVisWidget(QWebEngineView):
    def __init__(self, kwargs,networkx_graph=None ):
        super().__init__()
        html_path = "networkx.html"
        if networkx_graph is None:
            # Generate dummy network graph
            networkx_graph = nx.complete_graph(5)
        g = net.Network(directed=True)
        for i in range(kwargs["speaker_num"]):
            if i == kwargs["speaker"]:
                g.add_node(i, color="r")
            else:
                g.add_node(i)
        # g.add_edge(kwargs.s1,kwargs.s2)
        g.write_html(html_path)
        self.load(QUrl.fromLocalFile(os.path.abspath(html_path)))

class Window(QMainWindow):
    def __init__(self, kwargs,networkx_graph=None,title="PyVis Viewer"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(0, 0, 800, 800)
        self.setCentralWidget(PyVisWidget(kwargs,networkx_graph))
        self.show()


def show_pyvis(kwargs,networkx_graph=None, title="PyVis Viewer"):
    app = QApplication(sys.argv)
    window = Window(kwargs,networkx_graph, title)
    sys.exit(app.exec_())


if __name__ == "__main__":
    show_pyvis()
