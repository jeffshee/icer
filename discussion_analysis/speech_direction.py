import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# 発言方向とその回数のndarrayを取得
def get_directions_num(
    diarization_csv_path: str,
    total_speaker: int
):
    diarization = pd.read_csv(diarization_csv_path)
    directions_num = np.zeros((total_speaker, total_speaker), dtype=np.int)

    rows = diarization
    for i in range(len(rows)):
        row = rows.iloc[i]
        speaker = row["Speaker"].item()

        if i == 0:
            start = None
            end = speaker
        else:
            start = end
            end = speaker
            if start == end:
                continue
            directions_num[start][end] += 1
            # print(f"[{i:03d}]: {start} → {end}")
    # print(directions_num)
    return directions_num


# overlay用
def get_dialog_direction_networkx(
    diarization_csv_path: str,
    total_speaker: int
):
    directions_num_nd = get_directions_num(
        diarization_csv_path,
        total_speaker
    )

    df_directions_num_nd = pd.DataFrame(directions_num_nd)
    # edgeの重みが0のものをマスク
    df_directions_num_nd = df_directions_num_nd.mask(df_directions_num_nd == 0)

    # エッジリストを生成
    edge_lists = df_directions_num_nd.stack().reset_index().apply(tuple, axis=1).values
    # float型→int型
    for i in range(edge_lists.shape[0]):
        edge_lists[i] = tuple(map(int, edge_lists[i]))

    G = nx.MultiDiGraph()
    G.add_weighted_edges_from(edge_lists)
    return G


# matplotlibに描画用
def draw_speaker_direction_network(
    directions_num_nd,
    is_percent_label: bool = True
):
    df_directions_num_nd = pd.DataFrame(directions_num_nd)
    # edgeの重みが0のものをマスク
    df_directions_num_nd = df_directions_num_nd.mask(df_directions_num_nd == 0)

    # エッジリストを生成
    edge_lists = df_directions_num_nd.stack().reset_index().apply(tuple, axis=1).values
    # float型→int型
    for i in range(edge_lists.shape[0]):
        edge_lists[i] = tuple(map(int, edge_lists[i]))

    G = nx.MultiDiGraph()
    G.add_weighted_edges_from(edge_lists)
    pos = nx.circular_layout(G)

    # weight に応じてラインの太さを調整
    line_width = [d['weight'] for _, _, d in G.edges(data=True)]

    # nodeとそれを始点とするedgeごとに色指定
    cmap = plt.get_cmap('Paired')
    node_color = [cmap(node) for node in G.nodes]
    edge_color = [cmap(edge[0]) for edge in G.edges]

    # networkを描画
    nx.draw_networkx(
        G, pos=pos, node_size=800, node_color=node_color, edge_color=edge_color, width=line_width, connectionstyle="arc3,rad=0.1"
    )

    # edgeのラベルを指定
    if is_percent_label:
        sum_weight = np.sum(directions_num_nd)
        edge_labels = {(u, v): f"{(d['weight'] / sum_weight)*100:.2f}%" for u, v, d in G.edges(data=True)}
    else:
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    # edgeのラベルを描画
    nx.draw_networkx_edge_labels(
        G, pos, label_pos=0.75, edge_labels=edge_labels, font_size=7, font_weight='bold', bbox=dict(alpha=0.), rotate=False
    )

    plt.show()


if __name__ == "__main__":
    diarization_csv_path = "./discussion_analysis/transcript_demo.csv"
    # diarization_csv_path = "/home/icer/Project/dataset/transcript/transcript.csv"

    total_speaker = 6  # speakerの総数

    directions_num_nd = get_directions_num(
        diarization_csv_path,
        total_speaker
    )
