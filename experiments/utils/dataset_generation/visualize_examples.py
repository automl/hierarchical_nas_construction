import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.facecolor": (1.0, 1.0, 1.0, 1)})


def image_normalization(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def disable_ax_ticks(ax):
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)


def show_mnist_examples(x, y, save_path):
    for i in range(3):
        fig = plt.figure(constrained_layout=True, figsize=(12, 9), dpi=100)
        gs = fig.add_gridspec(3, 4)
        main_ax = fig.add_subplot(gs[:3, :3])
        fig.suptitle(f"{y[i]}")
        main_ax.imshow(image_normalization(np.moveaxis(x[i], 0, -1)))
        disable_ax_ticks(main_ax)

        for j in range(3):
            c_ax = fig.add_subplot(gs[j, -1])
            subimage = x[i].copy()
            subimage[:j] = 0
            subimage[j + 1 :] = 0
            subimage[j] = subimage[j] - subimage[j].min()
            c_ax.imshow(image_normalization(np.moveaxis(subimage, 0, -1)))
            disable_ax_ticks(c_ax)
        plt.savefig(save_path[:-4] + str(i) + save_path[-4:])
        plt.close()


def show_fashionMNIST_examples(x, y, save_path):
    plt.figure(figsize=(9, 9), dpi=100)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title("y={}".format(y[i]))
        plt.imshow(image_normalization(x[i][0]), cmap="gray")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def show_CIFARTile(x, y, save_path):
    plt.figure(figsize=(9, 9), dpi=100)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(f"Tile Classes={y[i]}")
        plt.imshow(np.moveaxis(image_normalization(x[i]), 0, -1))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_data(path, mode="train"):
    if not os.path.isdir(path):
        raise ValueError(f"Path {path} is no valid directory!")
    if mode not in ["train", "valid", "test"]:
        raise ValueError(f"Type {mode} does not exist in {path}: {os.listdir(path)}")

    full_path_x = os.path.join(path, f"{mode}_x.npy")
    full_path_y = os.path.join(path, f"{mode}_y.npy")

    return np.load(full_path_x), np.load(full_path_y)
