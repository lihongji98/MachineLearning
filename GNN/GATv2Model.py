import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import softmax as sparse_softmax
from torch_geometric.nn.inits import glorot, zeros


class GATv2Layer(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 heads=1, concat=True, negative_slope=0.2, dropout=0.0,
                 add_self_loop=True, edge_dim=None, bias=True, fill_value='mean'):
        super().__init__(node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loop
        self.fill_value = fill_value
        self.edge_dim = edge_dim

        self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels, heads * out_channels, bias=bias)

        self.attention_layer = nn.Parameter(torch.empty(1, heads, out_channels), requires_grad=True)

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)

        if bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * out_channels), requires_grad=True)
        elif bias and not concat:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        if self.edge_dim is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        glorot(self.attention_layer)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        H, C = self.heads, self.out_channels
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)

        if self.add_self_loops:
            num_nodes = min(x_l.size(0), x_r.size(0))
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr,
                                                   fill_value=self.fill_value, num_nodes=num_nodes)

        if edge_attr is None:
            edge_attr = torch.zeros(size=(edge_index.size(1), 1))
        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        out = out.mean(dim=1) if not self.concat else out.view(-1, self.heads * self.out_channels)

        if self.bias is not None:
            out += self.bias

        return out

    def edge_update(self, x_j, x_i, edge_attr, index, ptr, dim_size):
        x = x_i + x_j

        if edge_attr is not None and self.edge_dim is not None:
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x += edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.attention_layer).sum(dim=-1)
        alpha = sparse_softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def message(self, x_j, alpha):
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
