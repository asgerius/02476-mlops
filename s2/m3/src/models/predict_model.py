from __future__ import annotations

import torch
import torch.nn as nn
from pelutils import TT, log

from src.data import load_data
from src.models import MnistModel


def eval(model: MnistModel, test_x: torch.Tensor, test_y: torch.Tensor) -> tuple[float, float]:
    TT.profile("Eval")
    model.eval()
    pred = model(test_x)
    loss = nn.CrossEntropyLoss()(pred, test_y)
    pred_index = pred.argmax(dim=1)
    acc = (pred_index == test_y).float().mean()

    model.train()
    TT.end_profile()

    return loss.item(), acc

if __name__ == "__main__":
    with log.log_errors:
        log.configure("predict-mnist.log")
        train_x, train_y, test_x, test_y = load_data("data/external/corruptmnist")
        log.section("Evaluting saved model")
        model = MnistModel()
        model.load_state_dict(torch.load("models/mnist-model.pt"))
        test_loss, test_acc = eval(model, test_x, test_y)
        log("Test loss: %.6f" % test_loss, "Test acc: %.2f %%" % (100 * test_acc))
