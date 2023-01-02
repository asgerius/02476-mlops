from __future__ import annotations

import os

import numpy as np
import torch
from pelutils import log

from utils import device


def load_data(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    test = np.load(os.path.join(path, "test.npz"))
    test_x = torch.from_numpy(test["images"]).to(device).float().reshape(-1, 28 * 28)
    test_y = torch.from_numpy(test["labels"]).to(device)

    trains_x = list()
    trains_y = list()
    for i in range(5):
        train = np.load(os.path.join(path, f"train_{i}.npz"))
        trains_x.append(torch.from_numpy(train["images"]).to(device))
        trains_y.append(torch.from_numpy(train["labels"]).to(device))

    train_x = torch.cat(trains_x).float().reshape(-1, 28 * 28)
    train_y = torch.cat(trains_y)

    return train_x, train_y, test_x, test_y
