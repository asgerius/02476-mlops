from __future__ import annotations

from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pelutils import TT, log

from src import device
from src.data import load_data
from src.models import MnistModel
from src.models.predict_model import eval


def train(lr: float, batch_size: int, train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor):

    model = MnistModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5
    num_batches = len(train_x) // batch_size
    for i in range(epochs):
        TT.profile("Epoch")
        log("Epoch %i / %i" % (i + 1, epochs))
        order = torch.randperm(len(train_x))
        train_x = train_x[order]
        train_y = train_y[order]
        losses = list()
        for j in range(num_batches):
            tx = train_x[j*batch_size:(j+1)*batch_size]
            ty = train_y[j*batch_size:(j+1)*batch_size]
            pred = model(tx)
            loss = loss_fn(pred, ty)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        log("Mean training loss: %.6f"  % np.mean(losses))
        TT.end_profile()

        test_loss, test_acc = eval(model, test_x, test_y)
        log("Test loss: %.6f" % test_loss, "Test acc: %.2f %%" % (100 * test_acc))

    log("Save model")
    torch.save(model.state_dict(), "models/mnist-model.pt")


if __name__ == "__main__":

    with log.log_errors:
        log.configure("mnist.log")

        parser = ArgumentParser()
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--batch_size", type=int, default=100)
        args = parser.parse_args()

        log("Load data")
        train_x, train_y, test_x, test_y = load_data("data/external/corruptmnist")
        log(
            f"Train images: {train_x.shape}",
            f"Train labels: {train_y.shape}",
            f"Test images:  {test_x.shape}",
            f"Test labels:  {test_y.shape}",
        )

        log.section("Training new model")
        train(args.lr, args.batch_size, train_x, train_y, test_x, test_y)
