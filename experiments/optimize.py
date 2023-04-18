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
from benchmarks.objectives.addNIST import AddNISTObjective
from benchmarks.objectives.cifar_activation import (
    CIFAR10ActivationObjective,
)
from benchmarks.objectives.cifarTile import CifarTileObjective
from benchmarks.objectives.hierarchical_nb201 import NB201Pipeline
from benchmarks.search_spaces.activation_function_search.graph import (
    ActivationSpace,
)
from benchmarks.search_spaces.hierarchical_nb201.graph import (
    NB201_HIERARCHIES_CONSIDERED,
    NB201Spaces,
)
from neps.optimizers import SearcherMapping
from neps.optimizers.bayesian_optimization.acquisition_functions import AcquisitionMapping
from neps.optimizers.bayesian_optimization.acquisition_samplers import (
    AcquisitionSamplerMapping,
    EvolutionSampler,
)
from neps.optimizers.bayesian_optimization.kernels import GraphKernelMapping
from neps.optimizers.bayesian_optimization.models.gp_hierarchy import (
    ComprehensiveGPHierarchy,
)
from neps.search_spaces.search_space import SearchSpace
from path import Path
from experiments.zero_cost_rank_correlation import ZeroCost, evaluate
from benchmarks.search_spaces.darts_cnn.graph import DARTSSpace
from benchmarks.objectives.darts_cnn import DARTSCnn

SearchSpaceMapping = {
    "nb201": NB201Spaces,
    "act": partial(ActivationSpace, base_architecture="resnet20"),
    "darts": DARTSSpace,
}

hierarchies_considered_in_search_space = {**NB201_HIERARCHIES_CONSIDERED}
hierarchies_considered_in_search_space["act_cifar10"] = [0, 1, 2]
hierarchies_considered_in_search_space["darts"] = [0]

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
    "nb201_addNIST": AddNISTObjective,
    "nb201_cifarTile": CifarTileObjective,
    "nb201_cifar10": partial(NB201Pipeline, dataset="cifar10"),
    "nb201_cifar100": partial(NB201Pipeline, dataset="cifar100"),
    "nb201_ImageNet16-120": partial(NB201Pipeline, dataset="ImageNet16-120"),
    "act_cifar10": partial(CIFAR10ActivationObjective, dataset="cifar10"),
    "act_cifar100": partial(CIFAR10ActivationObjective, dataset="cifar100"),
    "darts": DARTSCnn,
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
    "--max_evaluations_total", type=int, default=150, help="number of evaluations"
)
parser.add_argument(
    "-ps",
    "--pool_size",
    type=int,
    default=200,
    help="number of candidates generated at each iteration",
)
parser.add_argument(
    "-ms",
    "--mutate_size",
    type=int,
    default=200,
    help="number of candidates mutated at each iteration",
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
    default="gpwl_hierarchical",
    choices=["gpwl", "gpwl_hierarchical", "gp_nasbot"],
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
parser.add_argument(
    "--adjust_params",
    default=None,
    choices=[None, "max"],
    help="Adjust nof params of models",
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
    if args.adjust_params is not None and "nb201_" in args.objective:
        assert args.adjust_params in ["max"]
        pipeline_space = dict(
            architecture=SearchSpaceMapping[search_space_key](
            space="fixed_1_none", dataset=dataset, adjust_params=None
        ),
        )
        pipeline_space = SearchSpace(**pipeline_space)
        if args.adjust_params == "max":
            identifier = "(CELL Cell (OPS conv3x3) (OPS conv3x3) (OPS conv3x3) (OPS conv3x3) (OPS conv3x3) (OPS conv3x3))"
        else:
            raise NotImplementedError
        pipeline_space.load_from({"architecture": identifier})
        model = pipeline_space.hyperparameters["architecture"].to_pytorch()
        if dataset in ["cifar10", "cifar100"]:
            _ = model(torch.rand(1, 3, 32, 32))
        elif dataset == "ImageNet16-120":
            _ = model(torch.rand(1, 3, 16, 16))
        elif dataset == "addNIST":
            _ = model(torch.rand(1, 3, 28, 28))
        elif dataset == "cifarTile":
            _ = model(torch.rand(1, 3, 64, 64))
        else:
            raise NotImplementedError
        args.adjust_params = sum(p.numel() for p in model.parameters())
    if (
        args.surrogate_model == "gpwl_hierarchical"
        or args.surrogate_model == "gpwl"
        or "gp_nasbot" == args.surrogate_model
    ):
        if "nb201_" in args.objective:
            search_space = SearchSpaceMapping[search_space_key](
                space=args.search_space[idx + 1 :], dataset=dataset, adjust_params=args.adjust_params
            )
        elif "act_" in args.objective:
            search_space = SearchSpaceMapping[search_space_key](dataset=dataset)
    else:
        if "nb201_" in args.objective:
            search_space = SearchSpaceMapping[search_space_key](
                space=args.search_space[idx + 1 :],
                dataset=dataset,
                return_graph_per_hierarchy=False,
                adjust_params=args.adjust_params,
            )
        elif "act_" in args.objective:
            search_space = SearchSpaceMapping[search_space_key](
                dataset=dataset, return_graph_per_hierarchy=False
            )
elif "darts" == args.objective:
    run_pipeline_fn = ObjectiveMapping[args.objective](
        data_path=args.data_path, seed=args.seed, log_scale=args.log
    )
    search_space = dict(
        normal=SearchSpaceMapping["darts"](),
        reduce=SearchSpaceMapping["darts"](),
    )
    args.pool_strategy = partial(EvolutionSampler, p_crossover=0.0, patience=10)
elif "debug" == args.objective:
    run_pipeline_fn = ObjectiveMapping[args.objective]
    idx = args.search_space.find("_")
    dataset = args.objective[args.objective.find("_") + 1 :]
    search_space = SearchSpaceMapping[args.search_space[:idx]](
        space=args.search_space[idx + 1 :], dataset="cifar10"
    )
else:
    raise NotImplementedError(f"Objective {args.objective} not implemented")

if args.surrogate_model == "gpwl_hierarchical":
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
elif args.surrogate_model == "gpwl":
    hierarchy_considered = None if args.objective == "darts" else []
    if args.objective == "darts":
        graph_kernels = ["wl", "wl"]
        wl_h = [2, 2]
    else:
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
elif "gp_nasbot" == args.surrogate_model:
    hierarchy_considered = []
    graph_kernels = ["nasbot"]
    if "nb201_variable_multi_multi" == args.search_space:
        include_op_list = [
            "id",
            "zero",
            "avg_pool",
            "conv3x3o",
            "conv1x1o",
            "dconv3x3o",
            "batch",
            "instance",
            "layer",
            "relu",
            "hardswish",
            "mish",
            "resBlock",
        ]
        exclude_op_list = ["input", "output"]
    else:
        raise NotImplementedError
    graph_kernels = [
        GraphKernelMapping[kernel](
            include_op_list=include_op_list, exclude_op_list=exclude_op_list
        )
        for kernel in graph_kernels
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
else:
    raise NotImplementedError

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
        initial_design_size=args.n_init,
        patience=patience,
    )
elif args.searcher == "assisted_regularized_evolution":
    zc_proxy = ZeroCost(
        method_type="nwot", n_classes=run_pipeline_fn.num_classes, loss_fn=None
    )
    extract_model = lambda x: x["architecture"].to_pytorch()
    zc_proxy_evaluation = partial(evaluate, zc_proxy=zc_proxy, loader=run_pipeline_fn.get_train_loader(), extract_model=extract_model)
    _ = neps.run(
        run_pipeline=run_pipeline_fn,
        pipeline_space=search_space,
        working_directory=args.working_directory,
        max_evaluations_total=args.max_evaluations_total,
        searcher=args.searcher,
        assisted_zero_cost_proxy=zc_proxy_evaluation,
        assisted_init_population_dir=Path(args.working_directory) / "assisted_init_population",
        initial_design_size=args.n_init,
    )
else:
    _ = neps.run(
        run_pipeline=run_pipeline_fn,
        pipeline_space=search_space,
        working_directory=args.working_directory,
        max_evaluations_total=args.max_evaluations_total,
        searcher=args.searcher,
        initial_design_size=args.n_init,
    )
