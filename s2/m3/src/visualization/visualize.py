import matplotlib.pyplot as plt
import pelutils.ds.plots as plots

from src.data import load_data

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_data("data/external/corruptmnist")
    with plots.Figure("reports/figures/examples-test.png"):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(test_x[i].view(28, 28).numpy())
