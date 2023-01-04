import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.build_layers()

    def build_layers(self):
        self.l1 = nn.Linear(28*28, 500)
        self.l2 = nn.Linear(500, 100)
        self.l3 = nn.Linear(100, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        return x
