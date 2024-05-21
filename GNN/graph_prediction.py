import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from GNN.GATv2Model import GATv2Layer
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


class GATv2Model(nn.Module):
    def __init__(self, in_channels=7, dim_embed=256, num_layer=2, num_class=1):
        super().__init__()

        self.lin_x_project = nn.Linear(in_channels, dim_embed)
        self.pool = global_mean_pool

        self.conv_layers = nn.ModuleList()

        for _ in range(num_layer):
            self.conv_layers.append(GATv2Layer(in_channels=dim_embed, out_channels=32, heads=8))

        self.lin_pred = nn.Linear(dim_embed, num_class)

    def forward(self, data):
        h = self.lin_x_project(data.x)

        for conv in self.conv_layers:
            h = h + conv(h, data.edge_index)

        h_graph = self.pool(h, data.batch)
        out = self.lin_pred(h_graph)

        return out


def train_graph_classification(model, train_loader, test_loader, num_epochs):
    # Set up the loss and the optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # A utility function to compute the accuracy
    def get_acc(model, loader):
        n_total = 0
        n_ok = 0
        for data in loader:
            outs = model(data).squeeze()
            n_ok += ((outs>0) == data.y).sum().item()
            n_total += data.y.shape[0]
        return n_ok/n_total

    for epoch in range(num_epochs):
        for data in train_loader:
            optimizer.zero_grad()
            outs = model(data).squeeze()
            loss = loss_fn(outs, data.y.float())
            loss.backward()
            optimizer.step()

        # Compute accuracies
        acc_train = get_acc(model, train_loader)
        acc_test = get_acc(model, test_loader)
        print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Train: {acc_train:.3f} | Test: {acc_test:.3f}')


if __name__ == "__main__":
    dataset = TUDataset(root='./', name='MUTAG')

    print({dataset})
    print(f'Number of graphs: {len(dataset)}')  # New!
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print()

    # Inspect the first graph
    print('First graph:')
    data = dataset[0]
    print(data)
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')

    dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    data = train_dataset[0]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = GATv2Model()

    train_graph_classification(model, train_loader, test_loader, num_epochs=100)


