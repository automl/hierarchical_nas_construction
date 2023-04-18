import argparse
import json
import random
from functools import partial
from typing import Union

import numpy as np
import torch
from benchmarks.objectives.addNIST import AddNISTObjective
from benchmarks.objectives.cifarTile import CifarTileObjective
from benchmarks.objectives.hierarchical_nb201 import NB201Pipeline
from benchmarks.search_spaces.hierarchical_nb201.graph import NB201Spaces
from nas_201_api import NASBench201API
from neps.search_spaces.search_space import SearchSpace
from path import Path

ObjectiveMapping = {
    "nb201_addNIST": AddNISTObjective,
    "nb201_cifarTile": CifarTileObjective,
    "nb201_cifar10": partial(NB201Pipeline, dataset="cifar10"),
    "nb201_cifar100": partial(NB201Pipeline, dataset="cifar100"),
    "nb201_ImageNet16-120": partial(NB201Pipeline, dataset="ImageNet16-120"),
}


def get_genotype(path_to_genotypes: Union[str, Path]) -> str:
    with open(Path(path_to_genotypes) / "genotypes.txt") as f:
        data = f.readlines()
    return data[-1][:-1]


def genotype_to_identifier(genotype: str):
    replace_map = {
        "nor_conv_3x3": "conv3x3",
        "nor_conv_1x1": "conv1x1",
        "skip_connect": "id",
        "none": "zero",
        "avg_pool_3x3": "avg_pool",
    }
    identifier = "(CELL Cell"
    for src_node_connections in genotype.split("+"):
        for key in src_node_connections[1:-1].split("|"):
            identifier += f" (OPS {replace_map[key.split('~')[0]]})"
    return identifier + ")"


def distill(result):
    result = result.split("\n")
    cifar10 = result[5].replace(" ", "").split(":")
    cifar100 = result[7].replace(" ", "").split(":")
    imagenet16 = result[9].replace(" ", "").split(":")

    cifar10_train = float(cifar10[1].strip(",test")[-7:-2].strip("="))
    cifar10_test = float(cifar10[2][-7:-2].strip("="))
    cifar100_train = float(cifar100[1].strip(",valid")[-7:-2].strip("="))
    cifar100_valid = float(cifar100[2].strip(",test")[-7:-2].strip("="))
    cifar100_test = float(cifar100[3][-7:-2].strip("="))
    imagenet16_train = float(imagenet16[1].strip(",valid")[-7:-2].strip("="))
    imagenet16_valid = float(imagenet16[2].strip(",test")[-7:-2].strip("="))
    imagenet16_test = float(imagenet16[3][-7:-2].strip("="))

    return (
        cifar10_train,
        cifar10_test,
        cifar100_train,
        cifar100_valid,
        cifar100_test,
        imagenet16_train,
        imagenet16_valid,
        imagenet16_test,
    )


parser = argparse.ArgumentParser("DARTS evaluation on cell-based nb201")
parser.add_argument("--working_directory", type=str, help="where data should be saved")
parser.add_argument(
    "--data_path", type=str, default="datapath", help="location of the data corpus"
)
parser.add_argument("--api_path", type=str, default="", help="location of the api data")
parser.add_argument(
    "--objective",
    type=str,
    default="cifar10",
    help="choose dataset",
    choices=["cifar10", "cifar100", "ImageNet16-120", "cifarTile", "addNIST"],
)
args = parser.parse_args()

splits = args.working_directory.split("/")
search_space = splits[-4]
dataset = splits[-3]
method = splits[-2]
args.seed = int(splits[-1])

genotype = get_genotype(args.working_directory)

run_pipeline_fn = ObjectiveMapping[dataset](
    data_path=args.data_path, seed=args.seed, eval_mode=True
)

if hasattr(run_pipeline_fn, "set_seed"):
    run_pipeline_fn.set_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)

search_space = NB201Spaces(
    space=search_space[6:-4], dataset=dataset[6:], adjust_params=False
)

if not isinstance(search_space, dict) and not isinstance(search_space, SearchSpace):
    search_space = {"architecture": search_space}

# read in best config
best_config = SearchSpace(**search_space)
identifier = genotype_to_identifier(genotype)
best_config.load_from({"architecture": identifier})
model = best_config["architecture"].to_pytorch()

# evaluate
results = run_pipeline_fn("", "", architecture=model)

if args.objective in ["cifar10", "cifar100", "ImageNet16-120"] and args.api_path:
    api = NASBench201API(args.api_path)
    result = api.query_by_arch(genotype, hp="200")
    print(result)
    (
        cifar10_train,
        cifar10_test,
        cifar100_train,
        cifar100_valid,
        cifar100_test,
        imagenet16_train,
        imagenet16_valid,
        imagenet16_test,
    ) = distill(result)
    results["api"] = {
        "cifar10-test": cifar10_test,
        "cifar100-valid": cifar100_valid,
        "cifar100-test": cifar100_test,
        "ImageNet16-120-valid": imagenet16_valid,
        "ImageNet16-120-test": imagenet16_test,
    }

print(results)
with open(Path(args.working_directory) / "best_config_eval.json", "w") as f:
    json.dump(results, f, indent=4)
