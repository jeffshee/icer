import sys
import os
from PyQt5.Qt import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication
from pyvis import network as net
import networkx as nx

html_path = "example.html"
g = net.Network()
nxg = nx.complete_graph(5)
g.from_nx(nxg)
g.write_html(html_path)

app = QApplication(sys.argv)
web = QWebEngineView()
web.load(QUrl.fromLocalFile(os.path.abspath(html_path)))
web.show()
sys.exit(app.exec_())
