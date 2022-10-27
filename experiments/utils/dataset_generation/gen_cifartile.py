import os

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def generate_n(n, class_idx, class_dict, data):
    xs, ys, metainfo = [], [], []

    for i in range(n):
        print("\r{}/{}".format(i, n), end="")
        n_classes = np.random.choice([1, 2, 3, 4])

        images = []
        if n_classes == 1:
            c = np.random.choice(class_idx, 1, replace=False)
            classes = c
            images += list(np.random.choice(class_dict[c[0]], size=4, replace=False))
        elif n_classes == 2:
            classes = np.random.choice(class_idx, 2, replace=False)
            for c in classes:
                images += list(np.random.choice(class_dict[c], size=2, replace=False))

        elif n_classes == 3:
            classes = np.random.choice(class_idx, 3, replace=False)
            images += list(
                np.random.choice(class_dict[classes[0]], size=2, replace=False)
            )
            for c in classes[1:]:
                images += list(np.random.choice(class_dict[c], size=1, replace=False))
        else:
            classes = np.random.choice(class_idx, 4, replace=False)
            for c in classes:
                images += list(np.random.choice(class_dict[c], size=1, replace=False))

        np.random.shuffle(images)
        metainfo.append(classes)
        out = np.zeros((3, 64, 64))
        out[:, :32, :32] = data[images[0]][0]
        out[:, :32, 32:] = data[images[1]][0]
        out[:, 32:, :32] = data[images[2]][0]
        out[:, 32:, 32:] = data[images[3]][0]
        xs.append(out)
        ys.append(n_classes - 1)

    xs = np.array(xs).astype(np.float32)
    ys = np.array(ys).astype(np.long)
    return xs, ys, metainfo


def load_cifartile_data(metainfo=False):
    data_path = os.getcwd() + "/raw_data/"
    dataset = "CIFAR"
    download = dataset not in os.listdir(data_path)

    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]

    train_data = datasets.CIFAR10(
        data_path + dataset,
        train=True,
        download=download,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        ),
    )
    test_data = datasets.CIFAR10(
        data_path + dataset,
        train=False,
        download=download,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        ),
    )

    train_class_dict = {}
    for i, (_, y) in enumerate(train_data):
        if y not in train_class_dict:
            train_class_dict[y] = []
        train_class_dict[y].append(i)

    test_class_dict = {}
    for i, (_, y) in enumerate(test_data):
        if y not in test_class_dict:
            test_class_dict[y] = []
        test_class_dict[y].append(i)

    class_idx = np.arange(10)

    if metainfo:
        train_x, train_y, metainfo = generate_n(
            600, class_idx, train_class_dict, train_data
        )
        test_x, test_y, _ = generate_n(100, class_idx, test_class_dict, test_data)

        return [train_x, train_y], [test_x, test_y], metainfo
    else:
        train_x, train_y, metainfo = generate_n(
            60000, class_idx, train_class_dict, train_data
        )
        test_x, test_y, _ = generate_n(10000, class_idx, test_class_dict, test_data)

        return [train_x, train_y], [test_x, test_y]
