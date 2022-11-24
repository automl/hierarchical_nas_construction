import argparse
import collections
import json
import os
import random
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from benchmarks.objectives.addNIST import AddNISTObjective
from benchmarks.objectives.cifarTile import CifarTileObjective
from benchmarks.objectives.hierarchical_nb201 import NB201Pipeline
from benchmarks.search_spaces.hierarchical_nb201.graph import (
    NB201_HIERARCHIES_CONSIDERED,
    NB201Spaces,
)
from neps.optimizers.bayesian_optimization.kernels import GraphKernelMapping
from neps.optimizers.bayesian_optimization.models.gp_hierarchy import (
    ComprehensiveGPHierarchy,
)
from neps.search_spaces.search_space import SearchSpace
from scipy import stats

ConfigResult = collections.namedtuple("ConfigResult", ["config", "result"])


SearchSpaceMapping = {
    "nb201": NB201Spaces,
}
hierarchies_considered_in_search_space = {**NB201_HIERARCHIES_CONSIDERED}
ObjectiveMapping = {
    "nb201_addNIST": AddNISTObjective,
    "nb201_cifarTile": CifarTileObjective,
    "nb201_cifar10": partial(NB201Pipeline, dataset="cifar10"),
    "nb201_cifar100": partial(NB201Pipeline, dataset="cifar100"),
    "nb201_ImageNet16-120": partial(NB201Pipeline, dataset="ImageNet16-120"),
}


def read_data(
    working_directory: Path,
    ylog: bool = True,
    debug_mode: bool = False,
    rs_only: bool = False,
):
    configs, y = [], []
    for f in os.listdir(working_directory):
        if not os.path.isdir(working_directory / f):
            continue
        for s in os.listdir(working_directory / f):
            if (
                "bug" in s
                or (rs_only and "random_search" != f)
                or ("fixed_1_none" in str(working_directory) and "gp_evo" in f)
                or not os.path.isdir(working_directory / f / s)
                or "naswot" in f
            ):
                continue
            results_dir = working_directory / f / s / "results"
            print(results_dir)
            previous_results = dict()
            for config_dir in results_dir.iterdir():
                config_id = config_dir.name[len("config_") :]
                result_file = config_dir / "result.yaml"
                config_file = config_dir / "config.yaml"
                if result_file.exists():
                    with result_file.open("rb") as results_file_stream:
                        result = yaml.safe_load(results_file_stream)
                    with config_file.open("rb") as config_file_stream:
                        config = yaml.safe_load(config_file_stream)
                    previous_results[config_id] = ConfigResult(config, result)

            configs += [previous_results[str(i)].config for i in range(1, 101)]
            if "cifar10" in str(results_dir) or "ImageNet" in str(results_dir):
                y += [
                    1-previous_results[str(i)].result["info_dict"]["x-valid_1"]/100
                    for i in range(1, 101)
                ]
            else:
                y += [
                    previous_results[str(i)].result["info_dict"]["val_score"]
                    for i in range(1, 101)
                ]

            if debug_mode:
                return configs, np.log(y)

    if ylog:
        y = np.log(y)
    return configs, y


parser = argparse.ArgumentParser(description="Surrogate experiment")
parser.add_argument("--working_directory", help="path to data")
parser.add_argument(
    "--search_space",
    default="nb201",
    help="The benchmark dataset to run the experiments.",
    # choices=SearchSpaceMapping.keys(),
)
parser.add_argument(
    "--objective",
    default="nb201_cifar10",
    help="The benchmark dataset to run the experiments.",
    choices=ObjectiveMapping.keys(),
)
parser.add_argument(
    "--surrogate_model",
    default="gp_hierarchical",
    choices=["gp", "gp_hierarchical"],
)
parser.add_argument("--n_train", type=int, default=100)
parser.add_argument(
    "--log", action="store_true", help="Whether to report the results in log scale"
)
parser.add_argument("--seeds", nargs="*", default=list(range(20)))
parser.add_argument(
    "--rs_only",
    action="store_true",
)
parser.add_argument(
    "--DEBUG",
    action="store_true",
)

args = parser.parse_args()
assert os.path.isdir(args.working_directory)
args.working_directory = Path(args.working_directory)

if not args.DEBUG:
    if args.rs_only:
        json_path = (
            args.working_directory
            / f"rsOnly_surrogate_{args.surrogate_model}_{args.n_train}.json"
        )
    else:
        json_path = (
            args.working_directory
            / f"surrogate_{args.surrogate_model}_{args.n_train}.json"
        )
    if os.path.isfile(json_path):
        os.remove(json_path)

idx = args.search_space.find("_")
dataset = args.objective[args.objective.find("_") + 1 :]
search_space = SearchSpaceMapping[args.search_space[:idx]](
    space=args.search_space[idx + 1 :], dataset=dataset,
)
search_space = SearchSpace(**{"architecture": search_space})

configs, y = read_data(
    args.working_directory, ylog=args.log, debug_mode=args.DEBUG, rs_only=args.rs_only
)
all_dict = {}
n_train = args.n_train
for seed in args.seeds:
    indices = list(range(len(configs)))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.shuffle(indices)
    # assert n_train + 500 <= len(configs)
    train_indices = indices[:n_train]
    if args.rs_only:
        test_indices = indices[-100:]  # stable 500 config test set
    else:
        test_indices = indices[-500:]  # stable 500 config test set
    x_train, y_train = [], []
    for idx in train_indices:
        copied_search_space = deepcopy(search_space)
        copied_search_space.load_from(configs[idx])
        x_train.append(copied_search_space)
        y_train.append(y[idx])
    x_test, y_test = [], []
    for idx in test_indices:
        copied_search_space = deepcopy(search_space)
        copied_search_space.load_from(configs[idx])
        x_test.append(copied_search_space)
        y_test.append(y[idx])

    if args.surrogate_model == "gp_hierarchical":
        hierarchy_considered = hierarchies_considered_in_search_space[args.search_space]
        graph_kernels = ["wl"] * (len(hierarchy_considered) + 1)
        wl_h = [2, 1] + [2] * len(hierarchy_considered)
        graph_kernels = [
            GraphKernelMapping[kernel](
                h=wl_h[j],
                oa=False,
                se_kernel=None,
            )
            for j, kernel in enumerate(graph_kernels)
        ]
        surrogate_model = ComprehensiveGPHierarchy
        surrogate_model_args = {
            "graph_kernels": graph_kernels,
            "hp_kernels": [],
            "verbose": False,
            "hierarchy_consider": hierarchy_considered,
            "d_graph_features": 0,
            "vectorial_features": None,
        }
    elif args.surrogate_model == "gp":
        hierarchy_considered = []
        graph_kernels = ["wl"]
        wl_h = [2]
        graph_kernels = [
            GraphKernelMapping[kernel](
                h=wl_h[j],
                oa=False,
                se_kernel=None,
            )
            for j, kernel in enumerate(graph_kernels)
        ]
        surrogate_model = ComprehensiveGPHierarchy
        surrogate_model_args = {
            "graph_kernels": graph_kernels,
            "hp_kernels": [],
            "verbose": False,
            "hierarchy_consider": hierarchy_considered,
            "d_graph_features": 0,
            "vectorial_features": None,
        }

    surrogate_model = surrogate_model(**surrogate_model_args)
    # train surrogate
    surrogate_model.fit(train_x=x_train, train_y=y_train)
    # & evaluate
    y_pred, y_pred_var = surrogate_model.predict(x_test)
    y_pred, y_pred_var = y_pred.cpu().detach().numpy(), y_pred_var.cpu().detach().numpy()

    # ====== evaluate regression performance ======
    pearson = stats.pearsonr(y_test, y_pred)[0]
    spearman = stats.spearmanr(y_test, y_pred)[0]
    kendalltau = stats.kendalltau(y_test, y_pred)[0]
    y_pred_std = np.sqrt(y_pred_var)
    nll = -np.mean(stats.norm.logpdf(np.array(y_test), loc=y_pred, scale=y_pred_std))

    print(
        f"seed={seed}, n_test={len(y_test)}: pearson={pearson :.3f}, spearman={spearman :.3f}, kendalltau={kendalltau :.3f}, NLL={nll}"
    )

    all_dict["pearson"] = float(pearson)
    all_dict["spearman"] = float(spearman)
    all_dict["kendalltau"] = float(kendalltau)
    all_dict["nll"] = float(nll)
    all_dict["y_pred"] = [float(y_) for y_ in y_pred]
    all_dict["y_test"] = [float(y_) for y_ in y_test]

    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        data[seed] = all_dict
    else:
        data = {}
        data[seed] = all_dict

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
