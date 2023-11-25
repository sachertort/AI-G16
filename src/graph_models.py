import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GAT, GCN, BatchNorm
from torch_geometric.data import Data

class ConversationGAT(nn.Module):
    def __init__(self):
        super(ConversationGAT, self).__init__()
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=128,
                            num_layers=2,
                            dropout=0.5,
                            bidirectional=True,
                            batch_first=True)

        self.gat = GAT(in_channels=256,
                       hidden_channels=128,
                       num_layers=2,
                       out_channels=64,
                       norm=BatchNorm(128),
                       heads=8,
                       dropout=0.0,
                       concat=True,
                       add_self_loops=False)

        self.norm2 = BatchNorm(64)

        self.output = nn.Linear(128, 2)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x, (hidden, cell) = self.lstm(x)
        x = self.gat(x=x,
                     edge_index=edge_index,
                     edge_attr=edge_attr[edge_index[0]]+edge_attr[edge_index[1]])
        x = F.relu(self.norm2(x))

        x = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        x = self.output(x)
        return x
    

class ConversationGCN(nn.Module):
    def __init__(self):
        super(ConversationGCN, self).__init__()
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=128,
                            num_layers=2,
                            dropout=0.5,
                            bidirectional=True,
                            batch_first=True)

        self.gcn = GCN(in_channels=256,
                       hidden_channels=128,
                       num_layers=2,
                       out_channels=64,
                       norm=BatchNorm(128),
                       dropout=0.0,
                       add_self_loops=False)

        self.norm2 = BatchNorm(64)

        self.output = nn.Linear(128, 2)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        x, (hidden, cell) = self.lstm(x)
        x = self.gcn(x=x,
                     edge_index=edge_index)
        x = F.relu(self.norm2(x))

        x = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        x = self.output(x)
        return x
