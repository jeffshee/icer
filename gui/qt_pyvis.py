import shutil
import sys
import os
import tempfile

from PyQt5.Qt import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication
from pyvis import network as net
import networkx as nx


def arrange_graph_view(g):
    for i, node in enumerate(g.nodes):
        g.nodes[i]['group'] = node['id']
        g.nodes[i]['physics'] = False

    sum_weight = sum([edge['weight'] for edge in g.edges])
    for i, edge in enumerate(g.edges):
        g.edges[i]['width'] = abs(edge['weight'])
        g.edges[i]['label'] = \
            f"{(edge['weight'] / sum_weight) * 100:.3f}%"
    g.set_edge_smooth('continuous')
    # g.show_buttons(filter_=['nodes', 'edges'])
    g.barnes_hut(spring_length=500, overlap=1)
    g.set_options("""
        var options = {
        "nodes": {
            "font": {
            "size": 12,
            "strokeWidth": 1
            },
            "shadow": {
            "enabled": true
            }
        },
        "edges": {
            "arrowStrikethrough": false,
            "color": {
            "inherit": true
            },
            "font": {
            "size": 10,
            "strokeWidth": 1,
            "align": "middle"
            },
            "smooth": {
            "type": "continuous",
            "forceDirection": "none"
            }
        }
        }
    """)
    return g


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
        # From araki
        g = arrange_graph_view(g)
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
