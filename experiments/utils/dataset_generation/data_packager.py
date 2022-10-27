import argparse
import collections
import os
import random
import shutil

import numpy as np
import torch
from gen_cifartile import load_cifartile_data
from gen_gutenberg import load_gutenberg
from gen_language_data import load_language_data
from gen_multnist_data import load_addnist_data, load_multnist_data
from sklearn.model_selection import train_test_split as tts
from torchvision import datasets, transforms
from visualize_examples import (
    load_data,
    show_CIFARTile,
    show_fashionMNIST_examples,
    show_mnist_examples,
)

parser = argparse.ArgumentParser(description="Data packager")
parser.add_argument(
    "--base_path",
    default=os.getcwd(),
    type=str,
    help="Where to save generated datasets",
    required=True,
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="Seed for dataset generation",
    required=False,
)
parser.add_argument(
    "--save_image_path",
    default=os.getcwd(),
    type=str,
    help="Where to save examples of dataset",
    required=False,
)
args = parser.parse_args()


# convert a list of tensors into a list of np arrays
def tlist_to_numpy(tlist):
    return [x.numpy() for x in tlist]


def process_torch_dataset(name, location, verbose=True, return_data=False):
    # load various datasets, put into respective dirs
    # pylint: disable=W0632
    if name == "MultNIST":
        (train_x, train_y, _), (test_x, test_y) = load_multnist_data()
    elif name == "AddNIST":
        (train_x, train_y, _), (test_x, test_y) = load_addnist_data()
    elif name == "Language":
        (train_x, train_y), (test_x, test_y) = load_language_data(
            metainfo=False, verbose=False
        )
    elif name == "Gutenberg":
        (train_x, train_y), (test_x, test_y) = load_gutenberg()
    elif name == "CIFARTile":
        (train_x, train_y), (test_x, test_y) = load_cifartile_data()
    elif name == "FashionMNIST":
        download = name not in os.listdir("raw_data")
        train_data = datasets.FashionMNIST(
            "raw_data/" + name,
            train=True,
            download=download,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        test_data = datasets.FashionMNIST(
            "raw_data/" + name,
            train=False,
            download=download,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        train_x, train_y = zip(*train_data)
        test_x, test_y = zip(*test_data)
        train_x = tlist_to_numpy(train_x)
        test_x = tlist_to_numpy(test_x)
        train_x, train_y = np.stack(train_x), np.array(train_y)
        test_x, test_y = np.stack(test_x), np.array(test_y)
    else:
        raise ValueError("Invalid dataset name!")
    # pylint: enable=W0632

    # split train data into train and valid
    train_x, valid_x, train_y, valid_y = tts(
        train_x, train_y, train_size=45000, test_size=15000
    )

    # print out stats of label distribution across classes
    def sort_counter(c):
        return sorted(list(c.items()), key=lambda x: x[0])

    if verbose:
        print("=== {} ===".format(name))
        print("Train Bal:", sort_counter(collections.Counter(train_y)))
        print("Valid Bal:", sort_counter(collections.Counter(valid_y)))
        print("Test Bal:", sort_counter(collections.Counter(test_y)))

    # randomly shuffle arrays
    train_shuff = np.arange(len(train_y))
    valid_shuff = np.arange(len(valid_y))
    test_shuff = np.arange(len(test_y))
    np.random.shuffle(train_shuff)
    np.random.shuffle(valid_shuff)
    np.random.shuffle(test_shuff)
    train_x, train_y = train_x[train_shuff], train_y[train_shuff]
    valid_x, valid_y = valid_x[valid_shuff], valid_y[valid_shuff]
    test_x, test_y = test_x[test_shuff], test_y[test_shuff]

    # print out data shapes of each split
    if verbose:
        print(
            "{} |  Train: {}, {} | Valid: {}, {} | Test: {}, {} |".format(
                name,
                train_x.shape,
                train_y.shape,
                valid_x.shape,
                valid_y.shape,
                test_x.shape,
                test_y.shape,
            )
        )

    # name and tag datasets
    dataset_path = os.path.join(location, name)

    if return_data:
        return [train_x, train_y], [valid_x, valid_y], [test_x, test_y]

    if os.path.isdir(dataset_path):
        shutil.rmtree(dataset_path)

    os.mkdir(dataset_path)
    np.save(dataset_path + "/train_x.npy", train_x, allow_pickle=False)
    np.save(dataset_path + "/train_y.npy", train_y, allow_pickle=False)
    np.save(dataset_path + "/valid_x.npy", valid_x, allow_pickle=False)
    np.save(dataset_path + "/valid_y.npy", valid_y, allow_pickle=False)
    np.save(dataset_path + "/test_x.npy", test_x, allow_pickle=False)
    np.save(dataset_path + "/test_y.npy", test_y, allow_pickle=False)
    print(f"Processed dataset {name}")


if __name__ == "__main__":
    base_path = args.base_path
    # set seed for reproducibility
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    save_image_path = args.save_image_path

    # load and save development datasets
    if not os.path.isdir("raw_data"):
        os.mkdir("raw_data")
    if not os.path.isdir(save_image_path):
        os.mkdir(save_image_path)
    for dataset in ["AddNIST", "FashionMNIST", "MultNIST", "CIFARTile"]:
        process_torch_dataset(dataset, location=base_path)
        full_path = os.path.join(base_path, dataset)
        x, y = load_data(full_path)
        if dataset == "AddNIST" or dataset == "MultNIST":
            show_mnist_examples(x, y, save_image_path + f"/{dataset}.png")
        elif dataset == "FashionMNIST":
            show_fashionMNIST_examples(x, y, save_image_path + f"/{dataset}.png")
        elif dataset == "CIFARTile":
            show_CIFARTile(x, y, save_image_path + f"/{dataset}.png")
        else:
            print("WARNING: No visualizer for examples!")
    shutil.rmtree("raw_data")
