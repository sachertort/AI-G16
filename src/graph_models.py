import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from torch_geometric.data import Data


class CauseExtractor(nn.Module):
    def __init__(self):
        """
        Initializes the CauseExtractor module.
        """
        super(CauseExtractor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=768, hidden_size=128, num_layers=3, dropout=0.7, bidirectional=True, batch_first=True
        )
        self.hidden = nn.Linear(537, 64)
        self.norm = BatchNorm(64)
        self.output = nn.Linear(64, 2)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Performs forward pass of the CauseExtractor module.

        Args:
            data (Data): Input data containing x, edge_index, edge_attr, and emotions.

        Returns:
            torch.Tensor: Output tensor after passing through the module.
        """
        x, edge_index, edge_attr, emotions = data.x, data.edge_index, data.edge_attr, data.emotions
        x, (hidden, cell) = self.lstm(x)
        x = torch.cat([x[edge_index[0]], edge_attr[edge_index[0]], x[edge_index[1]],  edge_attr[edge_index[1]], emotions[edge_index[1]]], dim=1)
        x = self.hidden(x)
        x = F.relu(self.norm(x))
        x = self.output(x)
        return x
