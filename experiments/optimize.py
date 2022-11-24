import argparse
import json
import logging
import os
import random
import time
from functools import partial

import neps
import numpy as np
import torch
from hierarchical_nas_benchmarks.objectives.cifar_activation import CIFAR10ActivationObjective
from benchmarks.objectives.addNIST import AddNISTObjective
from benchmarks.objectives.cifarTile import CifarTileObjective
from benchmarks.objectives.hierarchical_nb201 import NB201Pipeline
from hierarchical_nas_benchmarks.search_spaces.activation_function_search.graph import ActivationSpace
from benchmarks.search_spaces.hierarchical_nb201.graph import (
    NB201_HIERARCHIES_CONSIDERED,
    NB201Spaces,
)
from neps.optimizers import SearcherMapping
from neps.optimizers.bayesian_optimization.acquisition_functions import AcquisitionMapping
from neps.optimizers.bayesian_optimization.acquisition_samplers import (
    AcquisitionSamplerMapping,
)
from neps.optimizers.bayesian_optimization.kernels import GraphKernelMapping
from neps.optimizers.bayesian_optimization.models.gp_hierarchy import (
    ComprehensiveGPHierarchy,
)
from neps.search_spaces.search_space import SearchSpace
from path import Path

SearchSpaceMapping = {
    "nb201": NB201Spaces,
    "act": partial(ActivationSpace, base_architecture="resnet20"),
}

hierarchies_considered_in_search_space = {**NB201_HIERARCHIES_CONSIDERED}
hierarchies_considered_in_search_space["act_cifar10"] = [0, 1, 2]


def run_debug_pipeline(architecture):
    start = time.time()
    model = architecture.to_pytorch()
    number_of_params = sum(p.numel() for p in model.parameters())
    y = abs(1.5e7 - number_of_params)
    end = time.time()
    return {
        "loss": y,
        "info_dict": {
            "test_score": y,
            "train_time": end - start,
        },
    }


ObjectiveMapping = {
    "act_cifar10": partial(CIFAR10ActivationObjective, dataset="cifar10"),
    "nb201_addNIST": AddNISTObjective,
    "nb201_cifarTile": CifarTileObjective,
    "nb201_cifar10": partial(NB201Pipeline, dataset="cifar10"),
    "nb201_cifar100": partial(NB201Pipeline, dataset="cifar100"),
    "nb201_ImageNet16-120": partial(NB201Pipeline, dataset="ImageNet16-120"),
    "debug": run_debug_pipeline,
}

parser = argparse.ArgumentParser(description="Experiment runner")
parser.add_argument(
    "--search_space",
    default="nb201",
    help="The benchmark dataset to run the experiments.",
)
parser.add_argument(
    "--objective",
    default="nb201_cifar10",
    help="The benchmark dataset to run the experiments.",
    choices=ObjectiveMapping.keys(),
)
parser.add_argument(
    "--n_init", type=int, default=10, help="number of initialising points"
)
parser.add_argument(
    "--max_evaluations_total", type=int, default=100, help="number of evaluations"
)
parser.add_argument(
    "-ps",
    "--pool_size",
    type=int,
    default=200,
    help="number of candidates generated at each iteration",
)
parser.add_argument(
    "--pool_strategy",
    default="evolution",
    help="the pool generation strategy. Options: random," "mutation",
    choices=AcquisitionSamplerMapping.keys(),
)
parser.add_argument(
    "--p_self_crossover",
    default=0.5,
    type=float,
    help="Self crossover probability",
)
parser.add_argument(
    "-s",
    "--searcher",
    default="bayesian_optimization",
    choices=SearcherMapping.keys(),
)
parser.add_argument(
    "--surrogate_model",
    default="gp_hierarchical",
    choices=["gp", "gp_hierarchical"],
)
parser.add_argument(
    "-a",
    "--acquisition",
    default="EI",
    help="the acquisition function for the BO algorithm.",
    choices=AcquisitionMapping.keys(),
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--no_isomorphism",
    action="store_true",
    help="Whether to allow mutation to return" "isomorphic architectures",
)
parser.add_argument(
    "--maximum_noise",
    default=0.01,
    type=float,
    help="The maximum amount of GP jitter noise variance",
)
parser.add_argument(
    "--log", action="store_true", help="Whether to report the results in log scale"
)
parser.add_argument(
    "--data_path",
    default="data/",
    help="Path to data dir.",
)
parser.add_argument(
    "--random_interleave_prob",
    default=0.0,
    type=float,
    help="Probability to interleave random samples",
)
parser.add_argument(
    "--working_directory",
    default=os.path.dirname(os.path.realpath(__file__)) + "/working_dir",
    help="working directory",
)
parser.add_argument(
    "--asynchronous_parallel",
    action="store_true",
    help="Run asynchronous parallel mode",
)

args = parser.parse_args()

args.working_directory = os.path.join(args.working_directory, args.searcher)
if "bayesian_optimization" in args.searcher:
    args.working_directory += f"_{args.surrogate_model}"
    args.working_directory += f"_{args.pool_strategy}"
    args.working_directory += f"_pool{args.pool_size}"
args.working_directory = os.path.join(args.working_directory, f"{args.seed}")

working_dir = Path(args.working_directory)
working_dir.makedirs_p()
with open(working_dir / "args.json", "w") as f:
    json.dump(args.__dict__, f, indent=4)

if "nb201_" in args.objective or "act_" in args.objective:
    run_pipeline_fn = ObjectiveMapping[args.objective](
        data_path=args.data_path, seed=args.seed, log_scale=args.log
    )
    idx = args.search_space.find("_")
    dataset = args.objective[args.objective.find("_") + 1 :]
    search_space_key = args.search_space[:idx]
    if args.surrogate_model == "gp_hierarchical" or args.surrogate_model == "gp":
        if "nb201_" in args.objective:
            search_space = SearchSpaceMapping[search_space_key](
                space=args.search_space[idx + 1 :], dataset=dataset,
            )
        elif "act_" in args.objective:
            search_space = SearchSpaceMapping[search_space_key](dataset=dataset)
    else:
        if "nb201_" in args.objective:
            search_space = SearchSpaceMapping[search_space_key](
                space=args.search_space[idx + 1 :],
                dataset=dataset,
                return_graph_per_hierarchy=False,
            )
        elif "act_" in args.objective:
            search_space = SearchSpaceMapping[search_space_key](dataset=dataset, return_graph_per_hierarchy=False)
elif "debug" == args.objective:
    run_pipeline_fn = ObjectiveMapping[args.objective]
    idx = args.search_space.find("_")
    dataset = args.objective[args.objective.find("_") + 1 :]
    search_space = SearchSpaceMapping[args.search_space[:idx]](
        space=args.search_space[idx + 1 :], dataset="cifar10"
    )
else:
    raise NotImplementedError(f"Objective {args.objective} not implemented")

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


if args.seed is not None:
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

logging.basicConfig(level=logging.INFO)

if not isinstance(search_space, dict) and not isinstance(search_space, SearchSpace):
    search_space = {"architecture": search_space}

if args.searcher == "bayesian_optimization":
    patience = 10 if "fixed_1_none" in args.search_space else 100
    _ = neps.run(
        run_pipeline=run_pipeline_fn,
        pipeline_space=search_space,
        working_directory=args.working_directory,
        max_evaluations_total=args.max_evaluations_total,
        searcher=args.searcher,
        acquisition=args.acquisition,
        acquisition_sampler=args.pool_strategy,
        surrogate_model=surrogate_model,
        surrogate_model_args=surrogate_model_args,
        patience=patience,
    )
else:
    _ = neps.run(
        run_pipeline=run_pipeline_fn,
        pipeline_space=search_space,
        working_directory=args.working_directory,
        max_evaluations_total=args.max_evaluations_total,
        searcher=args.searcher,
    )
