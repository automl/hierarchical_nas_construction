import argparse
import collections
import copy
import json
import math
import os
import random
import time
import types
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from hierarchical_nas_benchmarks.evaluation.utils import get_train_val_test_loaders
from hierarchical_nas_benchmarks.objectives.addNIST import AddNISTObjective
from hierarchical_nas_benchmarks.objectives.cifarTile import CifarTileObjective
from hierarchical_nas_benchmarks.objectives.hierarchical_nb201 import (
    NB201Pipeline,
    get_dataloaders,
)
from hierarchical_nas_benchmarks.search_spaces.hierarchical_nb201.graph import (
    NB201_HIERARCHIES_CONSIDERED,
    NB201Spaces,
)
from neps.search_spaces.graph_grammar.graph import Graph
from neps.search_spaces.search_space import SearchSpace
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader

from hierarchical_nas_experiments.zero_cost_proxies_utils import calc_measure
from hierarchical_nas_experiments.zero_cost_proxies_utils.model_stats import (
    get_model_stats,
)
from hierarchical_nas_experiments.zero_cost_proxies_utils.p_utils import (
    get_some_data,
    get_some_data_grasp,
)

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


def no_op(self, x):  # pylint: disable=unused-argument
    return x


def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn is False:
        for l in net.modules():
            if isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm1d):
                l.forward = types.MethodType(no_op, l)
    return net


def find_measures_arrays(
    net_orig,
    trainloader,
    dataload_info,
    device,
    measure_names=None,
    loss_fn=F.cross_entropy,
):
    if measure_names is None:
        raise Exception("No zero-cost proxy provided")

    dataload, num_imgs_or_batches, num_classes = dataload_info

    if not hasattr(net_orig, "get_prunable_copy"):
        net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

    # move to cpu to free up mem
    torch.cuda.empty_cache()
    net_orig = net_orig.cpu()
    torch.cuda.empty_cache()

    # given 1 minibatch of data
    if dataload == "random":
        inputs, targets = get_some_data(
            trainloader, num_batches=num_imgs_or_batches, device=device
        )
    elif dataload == "grasp":
        inputs, targets = get_some_data_grasp(
            trainloader,
            num_classes,
            samples_per_class=num_imgs_or_batches,
            device=device,
        )
    else:
        raise NotImplementedError(f"dataload {dataload} is not supported")

    done, ds = False, 1
    measure_values = {}

    while not done:
        try:
            for measure_name in measure_names:
                if measure_name not in measure_values:
                    val = calc_measure(
                        measure_name,
                        net_orig,
                        device,
                        inputs,
                        targets,
                        loss_fn=loss_fn,
                        split_data=ds,
                    )
                    measure_values[measure_name] = val

            done = True
        except RuntimeError as e:
            if "out of memory" in str(e):
                done = False
                if ds == inputs.shape[0] // 2:
                    raise ValueError(
                        f"Can't split data anymore, but still unable to run. Something is wrong"
                    )
                ds += 1
                while inputs.shape[0] % ds != 0:
                    ds += 1
                torch.cuda.empty_cache()
                print(f"Caught CUDA OOM, retrying with data split into {ds} parts")
            else:
                raise e

    net_orig = net_orig.to(device).train()
    return measure_values


def find_measures(
    net_orig,  # neural network
    dataloader,  # a data loader (typically for training data)
    dataload_info,  # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
    device,  # GPU/CPU device used
    loss_fn,  # loss function to use within the zero-cost metrics
    measure_names=None,  # an array of measure names to compute, if left blank, all measures are computed by default
    measures_arr=None,
):

    # Given a neural net
    # and some information about the input data (dataloader)
    # and loss function (loss_fn)
    # this function returns an array of zero-cost proxy metrics.

    def sum_arr(arr):
        sum = 0.0  # pylint: disable=redefined-builtin
        for i, _ in enumerate(arr):
            sum += torch.sum(arr[i])
        return sum.item()

    if measure_names[0] in ["flops", "params"]:
        data_iterator = iter(dataloader)
        x, _ = next(data_iterator)
        x_shape = list(x.shape)
        x_shape[0] = 1  # to prevent overflow

        model_stats = get_model_stats(
            net_orig, input_tensor_shape=x_shape, clone_model=True
        )

        if measure_names[0] == "flops":
            measure_score = float(model_stats.Flops) / 1e6  # megaflops
        else:
            measure_score = float(model_stats.parameters) / 1e6  # megaparams
        return measure_score

    if measures_arr is None:
        measures_arr = find_measures_arrays(
            net_orig,
            dataloader,
            dataload_info,
            device,
            loss_fn=loss_fn,
            measure_names=measure_names,
        )

    for k, v in measures_arr.items():
        if k == "jacov" or k == "epe_nas" or k == "nwot" or k == "zen":
            measure_score = v
        else:
            measure_score = sum_arr(v)
    return measure_score


class Predictor:
    def __init__(self, ss_type=None):
        self.ss_type = ss_type

    def set_ss_type(self, ss_type):
        self.ss_type = ss_type

    def pre_process(self):
        """
        This is called at the beginning of the NAS algorithm, before any
        architectures have been queried. Use it for any kind of one-time set up.
        """

    def fit(self, xtrain, ytrain) -> None:  # pylint: disable=unused-argument
        """
        This can be called any number of times during the NAS algorithm.
        input: list of architectures, list of architecture accuracies
        output: none
        """

    def query(self, graph: Graph, dataloader: DataLoader) -> float:
        """Predict the score of the given model
        Args:
            graph       : Model to score
            dataloader  : DataLoader for the task to predict. E.g., if the task is to
                          predict the score of a model for classification on CIFAR10 dataset,
                          a CIFAR10 Dataloader is passed here.
        Returns:
            Score of the model. Higher the score, higher the model is ranked.
        """


class ZeroCost(Predictor):
    def __init__(  # pylint: disable=super-init-not-called
        self, method_type="jacov", n_classes: int = 10, loss_fn=None
    ):  # pylint: disable=non-parent-init-called
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.method_type = method_type
        self.dataload = "random"
        self.num_imgs_or_batches = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.loss_fn = loss_fn

    def query(self, graph, dataloader=None, info=None):  # pylint: disable=unused-argument
        # loss_fn = graph.get_loss_fn()
        loss_fn = self.loss_fn()

        # n_classes = graph.num_classes
        n_classes = self.n_classes
        score = find_measures(
            net_orig=graph,
            dataloader=dataloader,
            dataload_info=(self.dataload, self.num_imgs_or_batches, n_classes),
            device=self.device,
            loss_fn=loss_fn,
            measure_names=[self.method_type],
        )

        if math.isnan(score) or math.isinf(score):
            score = -1e8

        if self.method_type == "synflow":
            if score == 0.0:
                return score

            score = math.log(score) if score > 0 else -math.log(-score)

        return score


features = None


class ModelBeforeGlobalAvgPool(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

        def hook(
            module, input, output
        ):  # pylint: disable=unused-argument,redefined-builtin
            global features  # pylint: disable=global-statement
            features = output.clone()

        self.net.module_list[-1].op[1].register_forward_hook(hook)

    def forward(self, x):
        global features  # pylint: disable=global-variable-not-assigned
        _ = self.net(x)
        return features


def evaluate(zc_proxy, x_graphs, loader):
    zc_proxy.pre_process()
    test_pred = []
    for graph in x_graphs:
        if "zen" == zc_proxy.method_type:
            pred = zc_proxy.query(ModelBeforeGlobalAvgPool(deepcopy(graph)), loader)
        else:
            pred = zc_proxy.query(graph, loader)
        if float("-inf") == pred:
            pred = -1e9
        elif float("inf") == pred:
            pred = 1e9
        test_pred.append(pred)
    return np.array(test_pred)


def read_data(
    working_directory: Path,
    ylog: bool = False,
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
                    previous_results[str(i)].result["info_dict"]["x-valid_1"] / 100
                    for i in range(1, 101)
                ]
            else:
                y += [
                    1 - previous_results[str(i)].result["info_dict"]["val_score"]
                    for i in range(1, 101)
                ]

            if debug_mode:
                if ylog:
                    y = np.log(y)
                return configs[:20], y[:20]

    if ylog:
        y = np.log(y)
    return configs, y


if __name__ == "__main__":
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
    parser.add_argument("--data_path", help="path to dataset data")
    parser.add_argument(
        "--top",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--zc_method",
        default=None,
        choices=[
            None,
            "plain",
            "grasp",
            "fisher",
            "epe_nas",
            "grad_norm",
            "snip",
            "synflow",
            "l2_norm",
            "params",
            "zen",
            "jacov",
            "flops",
            "nwot",
        ],
    )
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument(
        "--DEBUG",
        action="store_true",
    )
    args = parser.parse_args()
    assert os.path.isdir(args.working_directory)
    args.working_directory = Path(args.working_directory)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)

    if not args.DEBUG and args.zc_method is None:
        json_path = args.working_directory / "zero_cost_proxy2.json"
        # if os.path.isfile(json_path):
        #     os.remove(json_path)
    elif not args.DEBUG and args.zc_method is not None:
        if args.top != 1.0:
            json_path = (
                args.working_directory
                / f"zero_cost_top_{int(args.top*100)}_{args.zc_method}.json"
            )
        else:
            json_path = args.working_directory / f"zero_cost_{args.zc_method}.json"

    idx = args.search_space.find("_")
    dataset = args.objective[args.objective.find("_") + 1 :]
    search_space = SearchSpaceMapping[args.search_space[:idx]](
        space=args.search_space[idx + 1 :], dataset=dataset, adjust_params=False
    )
    search_space = SearchSpace(**{"architecture": search_space})

    # these are used by the zero-cost methods
    batch_size = 64
    cutout = False
    cutout_length = 16
    cutout_prob = 1.0
    train_portion = 0.7
    if dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
        config, train_loader, ValLoaders = get_dataloaders(
            dataset,
            args.data_path,
            epochs=200,
            gradient_accumulations=1,
            workers=0,
            use_trivial_augment=False,
            eval_mode=False,  # TODO check
        )
        # val_loader["x-valid"]
        if "cifar10" == dataset:
            n_classes = 10
            # test_loader = ValLoaders["ori-test"]
        elif "cifar100" == dataset:
            # test_loader = ValLoaders["ori-test"]
            n_classes = 100
        elif "ImageNet16-120" == dataset:
            # test_loader = ValLoaders["ori-test"]
            n_classes = 120
    elif dataset in ["addNIST", "cifarTile"]:
        train_loader, valid_loader, test_loader = get_train_val_test_loaders(
            dataset=dataset,
            data=args.data_path,
            batch_size=batch_size,
            eval_mode=True,
        )
        if "addNIST" == dataset:
            n_classes = 20
        elif "cifarTile" == dataset:
            n_classes = 4
    else:
        raise NotImplementedError
    loss_fn = nn.CrossEntropyLoss

    configs, y_test = read_data(
        args.working_directory, ylog=False, debug_mode=args.DEBUG, rs_only=False
    )

    if args.top != 1.0:
        assert 0 < args.top < 1
        idx_sort = np.argsort(y_test)
        cutoff = int(len(y_test) // (1 / (1 - args.top)))
        configs = [configs[idx] for idx in idx_sort[cutoff:]]
        y_test = [y_test[idx] for idx in idx_sort[cutoff:]]

    x = []
    for idx, config_id in enumerate(configs):
        copied_search_space = deepcopy(search_space)
        copied_search_space.load_from(config_id)
        model = copied_search_space.hyperparameters["architecture"].to_pytorch()
        x.append(model)

    results = {}
    # works: plain, epe_nas, grad_norm, snip, l2_norm, params, jacov, flops, nwot, zen, synflow, grasp
    # does not work:
    # warnings: fisher (?)
    if args.zc_method is None:
        zc_methods = [
            "plain",
            "grasp",
            "fisher",
            "epe_nas",
            "grad_norm",
            "snip",
            "synflow",
            "l2_norm",
            "params",
            "zen",
            "jacov",
            "flops",
            "nwot",
        ]
    else:
        zc_methods = [args.zc_method]
    for method_type in zc_methods:
        # for method_type in ["fisher"]:
        if not args.DEBUG and os.path.isfile(json_path) and args.zc_method is None:
            with open(json_path) as fp:
                prev_results = json.load(fp)
            if method_type in prev_results:
                continue
            results = prev_results

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(args.seed)

        zc_proxy = ZeroCost(method_type=method_type, n_classes=n_classes, loss_fn=loss_fn)

        start = time.time()
        y_pred = evaluate(zc_proxy=zc_proxy, x_graphs=x, loader=train_loader)
        end = time.time()

        # ====== evaluate regression performance ======
        pearson = stats.pearsonr(y_test, y_pred)[0]
        spearman = stats.spearmanr(y_test, y_pred)[0]
        kendalltau = stats.kendalltau(y_test, y_pred)[0]
        results[method_type] = {
            "pearson": float(pearson),
            "spearman": float(spearman),
            "kendalltau": float(kendalltau),
            "time": end - start,
        }

        print(method_type, pearson, spearman, kendalltau)

        if not args.DEBUG:
            if args.zc_method is None:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
            else:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(results[method_type], f, indent=2)
