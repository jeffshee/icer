#必要なモジュールのインポート
import sys
import os
import pandas as pd
import numpy as np
from PIL import Image
import datetime

import cv2
import matplotlib.pyplot as plt
from pygraphviz import *
import networkx as nx


def get_transition_matrix(speaker_switch, index_to_name_dict, time_ms):
    sp_switch_key = np.array(list(speaker_switch.keys()))
    sp_switch = [speaker_switch[k] for k in sp_switch_key[(sp_switch_key <= time_ms).tolist()]]
    speaker_id = [index_to_name_dict[k] for k in index_to_name_dict.keys() if k >= 0]
    graph_edge = [[0] * len(speaker_id) for _ in range(len(speaker_id))]
    for prev_sp, next_sp in sp_switch:
        if (prev_sp is None) or (next_sp is None):
            continue
        if (prev_sp == -1) or (next_sp == -1):
            continue
        graph_edge[next_sp][prev_sp] = graph_edge[next_sp][prev_sp] + 1

    graph_df = pd.DataFrame(index=speaker_id,
                    columns=speaker_id,
                    data=graph_edge)
    return graph_df

def get_graph_fig(n_nodes, df, max_node, out_dpath, add_flag=False):
    mapping = {}
    c_mapping = {}
    col2node = {}
    idx2node = {}
    for i, (idx, col) in enumerate(zip(df.index, df.columns)):
        mapping[i] = idx
        if add_flag:
            mapping[i] += '\nOut: {}'.format(df[col].sum())  
        col2node[col] = mapping[i]
        idx2node[idx] = mapping[i]
        c_mapping[i] = "#{:02}0000".format(i%16)
    h = nx.fast_gnp_random_graph(n_nodes, 1, 1, directed=True)
    h = nx.relabel_nodes(h, mapping)
    g = nx.nx_agraph.to_agraph(h)
    for i in df.index:
        for c in df.columns:
            if i == c:
                continue
            e = g.get_edge(col2node[c], idx2node[i])
            if df[c][i] == 0:
                e.attr['color'] = None
                e.attr['label'] = ''
            else:
                e.attr['color'] = 'black'
                e.attr['label'] = df[c][i]
    
    for i, c in enumerate(df.columns):
        n = g.get_node(col2node[c])
        # n.attr['pos'] = "10,{}!".format(10+10*i)
        s_node = df[c].sum()
        n.attr['fillcolor'] = c_mapping[i]
        n.attr['height'] = "%s"%(s_node/max_node+1.0)
        n.attr['width'] = "%s"%(s_node/max_node+1.0)

    g.draw(out_dpath, prog='circo')
    g = np.array(Image.open(out_dpath))
    os.remove(out_dpath)
    return g


if __name__ == '__main__':
    df = pd.DataFrame(index=[str(i) for i in range(5)],
                columns=[str(i) for i in range(5)],
                data=[[0,1,0,2,0],
                    [2,0,0,0,3],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [1,0,0,0,0]])
    print(df)
    dt_now = datetime.datetime.now()
    g = make_graph(len(df), df, 9, 'test{}.png'.format(dt_now.strftime('%Y-%m-%d-%H-%M-%S')))
    cv2.imwrite('test.png', g)