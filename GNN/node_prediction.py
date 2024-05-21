import torch
import torch_geometric
from torch import nn
from torch_geometric.utils import to_dense_adj

from GNN.GATv2Model import GATv2Layer


class GATv2Model(nn.Module):
    def __init__(self, in_channels=1433, dim_embed=32, num_layer=2, num_class=7):
        super().__init__()

        self.lin_x_project = nn.Linear(in_channels, dim_embed)

        self.conv_layers = nn.ModuleList()

        for _ in range(num_layer):
            self.conv_layers.append(GATv2Layer(in_channels=dim_embed, out_channels=4, heads=8))

        self.lin_pred = nn.Linear(dim_embed, num_class)

    def forward(self, data):
        h = self.lin_x_project(data.x)

        for conv in self.conv_layers:
            h = h + conv(h, data.edge_index)

        out = self.lin_pred(h)
        return out


def train(model, data, num_epochs, use_edge_index=False):
    if not use_edge_index:
        # Create the adjacency matrix
        # Important: add self-edges so a node depends on itself
        adj = to_dense_adj(data.edge_index)[0]
        adj += torch.eye(adj.shape[0])
    else:
        # Directly use edge_index, ignore this branch for now
        adj = data.edge_index

    # Set up the loss and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # A utility function to compute the accuracy
    def get_acc(outs, y, mask):
        return (outs[mask].argmax(dim=1) == y[mask]).sum().float() / mask.sum()

    best_acc_val = -1
    for epoch in range(num_epochs):
        # Zero grads -> forward pass -> compute loss -> backprop
        optimizer.zero_grad()
        outs = model(data)
        loss = loss_fn(outs[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Compute accuracies, print only if this is the best result so far
        acc_val = get_acc(outs, data.y, data.val_mask)
        acc_test = get_acc(outs, data.y, data.test_mask)
        if acc_val > best_acc_val:
            best_acc_val = acc_val
        print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Val: {acc_val:.3f} | Test: {acc_test:.3f}')


if __name__ == "__main__":
    dataset = torch_geometric.datasets.Planetoid(root='/', name='Cora')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print()
    data = dataset[0]
    print(data)
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Number of validation nodes: {data.val_mask.sum()}')
    print(f'Number of test nodes: {data.test_mask.sum()}')
    print()

    model = GATv2Model(num_layer=3)

    train(model, data, num_epochs=100)
