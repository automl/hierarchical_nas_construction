import collections
import os

import numpy as np
from torchvision import datasets, transforms


def train_test_filter(op, combs, ratio):
    paths = {}
    for i, j, k in combs:
        val = op(i, j, k)
        if val not in paths:
            paths[val] = []
        paths[val].append((i, j, k))
    train_combs, test_combs = [], []

    for val, paths in paths.items():
        idxs = np.arange(len(paths))
        if len(paths) > 1:
            train_idxs = np.random.choice(
                idxs, size=int(len(paths) * ratio), replace=False
            )
            train_combs += [paths[i] for i in train_idxs]
            test_combs += [paths[i] for i in idxs if i not in train_idxs]
    return train_combs, test_combs


def generate_examples(op, combs, weights, nums, n):
    x, y, metainfo = [], [], []
    for n1, n2, n3 in combs:
        mod = op(n1, n2, n3)
        r_idxs = np.arange(len(nums[n1]))
        g_idxs = np.arange(len(nums[n2]))
        b_idxs = np.arange(len(nums[n3]))

        for _ in range(int(n * weights[mod])):
            r = nums[n1][np.random.choice(r_idxs)]
            g = nums[n2][np.random.choice(g_idxs)]
            b = nums[n3][np.random.choice(b_idxs)]
            x.append(np.vstack([r, g, b]))
            y.append(mod)
            metainfo.append([n1, n2, n3])
    return x, y, metainfo


def proc_weights(op, combs):
    weights = collections.Counter(op(i, j, k) for i, j, k in combs)
    weights = {k: sum(weights.values()) / v for k, v in weights.items()}
    return {k: v / min(weights.values()) for k, v in weights.items()}


def generate_data(op, n, lb, ub):
    download = "MNIST" not in os.listdir("raw_data")
    train_data = datasets.MNIST(
        "raw_data/MNIST",
        train=True,
        download=download,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    test_data = datasets.MNIST(
        "raw_data/MNIST",
        train=False,
        download=download,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    train_nums = {i: [] for i in range(10)}
    test_nums = {i: [] for i in range(10)}

    for image, number in train_data:
        train_nums[number].append(image)
    for image, number in test_data:
        test_nums[number].append(image)

    combs = [
        (i, j, k)
        for i in range(10)
        for j in range(10)
        for k in range(10)
        if lb <= op(i, j, k) <= ub
    ]
    train_combs, test_combs = train_test_filter(op, combs, 0.75)
    train_weights = proc_weights(op, train_combs)
    test_weights = proc_weights(op, test_combs)

    train_n = n
    test_n = int(0.33 * train_n)
    train_x, train_y, metainfo = generate_examples(
        op, combs, train_weights, train_nums, train_n
    )
    test_x, test_y, _ = generate_examples(op, combs, test_weights, test_nums, test_n)

    train_x, train_y = (
        np.array(train_x, dtype=np.float32).squeeze(),
        np.array(train_y).squeeze(),
    )
    test_x, test_y = (
        np.array(test_x, dtype=np.float32).squeeze(),
        np.array(test_y).squeeze(),
    )

    train_shuff = np.arange(len(train_y))
    np.random.shuffle(train_shuff)
    test_shuff = np.arange(len(test_y))
    np.random.shuffle(test_shuff)
    train_x, train_y, metainfo = (
        train_x[train_shuff],
        train_y[train_shuff],
        [metainfo[i] for i in train_shuff],
    )
    test_x, test_y = test_x[test_shuff], test_y[test_shuff]
    return (train_x[:60000], train_y[:60000], metainfo[:60000]), (
        test_x[:10000],
        test_y[:10000],
    )


def load_multnist_data():
    op = lambda i, j, k: (i * j * k) % 10
    return generate_data(op, 20, 0, 9)


def load_addnist_data():
    op = lambda i, j, k: (i + j + k) - 1
    return generate_data(op, 200, 0, 19)
