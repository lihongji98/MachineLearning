import numpy as np
import seaborn as sns
import math
import itertools
import scipy as sp
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.datasets import Planetoid, ZINC, GNNBenchmarkDataset

from torch_geometric.utils import to_dense_adj
from torch.nn import Embedding

import pdb

import matplotlib.cm as cm

from typing import Mapping, Tuple, Sequence, List
import colorama

import scipy.linalg
from scipy.linalg import block_diag

import networkx as nx
import matplotlib.pyplot as plt

from mycolorpy import colorlist as mcp
from torch_scatter import scatter_mean, scatter_max, scatter_sum

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj


class CoraDataset(object):
    def __init__(self):
        super(CoraDataset, self).__init__()
        cora_pyg = Planetoid(root='/tmp/Cora', name='Cora', split="full")
        self.cora_data = cora_pyg[0]
        self.train_mask = self.cora_data.train_mask
        self.valid_mask = self.cora_data.val_mask
        self.test_mask = self.cora_data.test_mask

    def train_val_test_split(self):
        train_x = self.cora_data.x[self.cora_data.train_mask]
        train_y = self.cora_data.y[self.cora_data.train_mask]

        valid_x = self.cora_data.x[self.cora_data.val_mask]
        valid_y = self.cora_data.y[self.cora_data.val_mask]

        test_x = self.cora_data.x[self.cora_data.test_mask]
        test_y = self.cora_data.y[self.cora_data.test_mask]
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def get_fullx(self):
        return self.cora_data.x

    def get_adjacency_matrix(self):
        adj = to_dense_adj(self.cora_data.edge_index)[0]
        return adj


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, A):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A

        # ============ YOUR CODE HERE =============
        # Compute symmetric norm
        D = torch.diag(torch.sum(A, dim=1))
        D_sqrt_inv = torch.sqrt(torch.inverse(D))
        A = A + torch.eye(A.size()[0])
        DA = torch.einsum('ij, jk -> ik', D_sqrt_inv, A)
        DAD = torch.einsum('ik, kl -> il', DA, D)
        self.adj_norm = DAD

        # + Simple linear transformation and non-linear activation
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.relu = nn.ReLU()
        # =========================================

    def forward(self, x):
        # ============ YOUR CODE HERE =============
        scaled_x = torch.einsum('ib, bn -> in', self.adj_norm, x)
        x = self.relu(self.linear(scaled_x))
        x = torch.mean(x, dim=1)
        # =========================================
        return x


def draw_one_graph(ax, edges, label=None, node_emb=None, layout=None, special_color=False):
    graph = nx.Graph()
    edges = zip(edges[0], edges[1])
    graph.add_edges_from(edges)
    node_pos = layout(graph)
    # add colors according to node embeding
    if (node_emb is not None) or special_color:
        color_map = []
        node_list = [node[0] for node in graph.nodes(data=True)]
        for i, node in enumerate(node_list):
            # just ignore this branch
            if special_color:
                if len(node_list) == 3:
                    crt_color = (1, 0, 0)
                elif len(node_list) == 5:
                    crt_color = (0, 1, 0)
                elif len(node_list) == 4:
                    crt_color = (1, 1, 0)
                else:
                    special_list = [(1, 0, 0)] * 3 + [(0, 1, 0)] * 5 + [(1, 1, 0)] * 4
                    crt_color = special_list[i]
            else:
                crt_node_emb = node_emb[node]
                # map float number (node embeding) to a color
                crt_color = cm.gist_rainbow(crt_node_emb, bytes=True)
                crt_color = (crt_color[0] / 255.0, crt_color[1] / 255.0, crt_color[2] / 255.0, crt_color[3] / 255.0)
            color_map.append(crt_color)

        nx.draw_networkx_nodes(graph, node_pos, node_color=color_map,
                               nodelist=node_list, ax=ax)
        nx.draw_networkx_edges(graph, node_pos, ax=ax)
        nx.draw_networkx_labels(graph, node_pos, ax=ax)
    else:
        nx.draw_networkx(graph, node_pos, ax=ax)


def gallery(graphs, labels=None, node_emb=None, special_color=False, max_graphs=4, max_fig_size=(40, 10),
            layout=nx.layout.kamada_kawai_layout):
    num_graphs = min(len(graphs), max_graphs)
    ff, axes = plt.subplots(1, num_graphs,
                            figsize=max_fig_size,
                            subplot_kw={'xticks': [], 'yticks': []})
    if num_graphs == 1:
        axes = [axes]
    if node_emb is None:
        node_emb = num_graphs * [None]
    if labels is None:
        labels = num_graphs * [" "]

    for i in range(num_graphs):
        draw_one_graph(axes[i], graphs[i].edge_index.numpy(), labels[i], node_emb[i], layout, special_color)
        if labels[i] != " ":
            axes[i].set_title(f"Target: {labels[i]}", fontsize=28)
        axes[i].set_axis_off()
    plt.show()


def hash_node_embedings(node_emb):
    """
    This function is a basic, non-bijective one for visualising the embedings.
    Please use it for guidance, not as a mathematical proof in Part 3.
    It is used just for educational/visualisation purpose.
    You are free to change it with whatever suits you best.
    Hash the tensor representing nodes' features
    to a number in [0,1] used to represent a color

    Args:
      node_emb: list of num_graphs arrays, each of dim (num_nodes x num_feats)
    Returns:
      list of num_graphs arrays in [0,1], each of dim (num_nodes)
    """
    chunk_size_graph = [x.shape[0] for x in node_emb]
    start_idx_graph = [0] + list(itertools.accumulate(chunk_size_graph))[:-1]

    node_emb_flatten = np.concatenate(node_emb).mean(-1)

    min_emb = node_emb_flatten.min()
    max_emb = node_emb_flatten.max()
    node_emb_flatten = (node_emb_flatten - min_emb) / (max_emb - min_emb + +0.00001)

    # split in graphs again according to (start_idx_graph, chunk_size_graph)
    node_emb_hashed = [node_emb_flatten[i:i + l] for (i, l) in zip(start_idx_graph, chunk_size_graph)]
    return node_emb_hashed


def update_stats(training_stats, epoch_stats):
    """ Store metrics along the training
    Args:
      epoch_stats: dict containg metrics about one epoch
      training_stats: dict containing lists of metrics along training
    Returns:
      updated training_stats
    """
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key, val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats


def plot_stats(training_stats, figsize=(5, 5), name=""):
    """ Create one plot for each metric stored in training_stats
    """
    stats_names = [key[6:] for key in training_stats.keys() if key.startswith('train_')]
    f, ax = plt.subplots(len(stats_names), 1, figsize=figsize)
    if len(stats_names) == 1:
        ax = np.array([ax])
    for key, axx in zip(stats_names, ax.reshape(-1, )):
        axx.plot(
            training_stats['epoch'],
            training_stats[f'train_{key}'],
            label=f"Training {key}")
        axx.plot(
            training_stats['epoch'],
            training_stats[f'val_{key}'],
            label=f"Validation {key}")
        axx.set_xlabel("Training epoch")
        axx.set_ylabel(key)
        axx.legend()
    plt.title(name)


def get_color_coded_str(i, color):
    return "\033[3{}m{}\033[0m".format(int(color), int(i))


def print_color_numpy(map, list_graphs):
    """ print matrix map in color according to list_graphs
    """
    list_blocks = []
    for i, graph in enumerate(list_graphs):
        block_i = (i + 1) * np.ones((graph.num_nodes, graph.num_nodes))
        list_blocks += [block_i]
    block_color = block_diag(*list_blocks)

    map_modified = np.vectorize(get_color_coded_str)(map, block_color)
    print("\n".join([" ".join(["{}"] * map.shape[0])] * map.shape[1]).format(
        *[x for y in map_modified.tolist() for x in y]))
