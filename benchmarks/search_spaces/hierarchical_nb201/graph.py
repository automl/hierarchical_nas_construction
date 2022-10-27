import os
from copy import deepcopy
from functools import partial

import neps
import neps.search_spaces.graph_grammar.primitives as ops
import neps.search_spaces.graph_grammar.topologies as topos
import networkx as nx
import numpy as np
from neps.search_spaces.graph_grammar.api import FunctionParameter
from neps.search_spaces.graph_grammar.graph import Graph
from neps.search_spaces.graph_grammar.utils import get_edge_lists_of_topologies
from neps.search_spaces.search_space import SearchSpace
from path import Path
from torch import nn

import benchmarks.search_spaces.hierarchical_nb201.primitives as nb201_ops
from benchmarks.search_spaces.hierarchical_nb201.topologies import (
    Diamond3,
    NASBench201Cell,
    Residual3,
)

DIR_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

PRIMITIVES = {
    # macro level
    "resBlock": {"op": nb201_ops.ResNetBasicblock, "stride": 2},
    "Linear1": partial(topos.LinearNEdge, number_of_edges=1),
    "Linear3": partial(topos.LinearNEdge, number_of_edges=3),
    "Linear4": partial(topos.LinearNEdge, number_of_edges=4),
    "Residual3": Residual3,
    "Diamond3": Diamond3,
    "Linear2": partial(topos.LinearNEdge, number_of_edges=2),
    "Residual2": topos.Residual,
    "Diamond2": topos.Diamond,
    # cell level
    "id": ops.Identity(),
    "zero": ops.Zero(stride=1),
    "conv3x3": {
        "op": nb201_ops.ReLUConvBN,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "affine": True,
    },
    "conv1x1": {
        "op": nb201_ops.ReLUConvBN,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "affine": True,
    },
    "avg_pool": {"op": nb201_ops.POOLING, "mode": "avg", "stride": 1, "affine": True},
    "Cell": NASBench201Cell,
    # conv block level
    "conv3x3o": {
        "op": nb201_ops.Conv,
        "kernel_size": 3,
    },
    "conv1x1o": {"op": nb201_ops.Conv, "kernel_size": 1},
    "dconv3x3o": {
        "op": nb201_ops.DepthwiseConv,
        "kernel_size": 3,
    },
    "batch": {"op": nb201_ops.Normalization, "norm_type": "batch_norm"},
    "instance": {
        "op": nb201_ops.Normalization,
        "norm_type": "instance_norm",
    },
    "layer": {"op": nb201_ops.Normalization, "norm_type": "layer_norm"},
    "relu": {"op": nb201_ops.Activation, "act_type": "relu"},
    "hardswish": {"op": nb201_ops.Activation, "act_type": "hardswish"},
    "mish": {"op": nb201_ops.Activation, "act_type": "mish"},
}


def get_nof_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build(
    graph: Graph, n_classes: int = 10,
):
    in_channels = 3
    base_channels = 16
    min_channels = 1
    # max_channels = np.inf
    out_channels_factor = 4  # 64

    def _create_architecture(_graph: Graph, _channels: int, return_model: bool = False):
        in_node = [n for n in _graph.nodes if _graph.in_degree(n) == 0][0]
        for n in nx.topological_sort(_graph):
            for pred in _graph.predecessors(n):
                e = (pred, n)
                if pred == in_node:
                    channels = _channels
                else:
                    pred_pred = list(_graph.predecessors(pred))[0]
                    channels = _graph.edges[(pred_pred, pred)]["C_out"]
                if _graph.edges[e]["op_name"] == "ResNetBasicblock":
                    _graph.edges[e].update({"C_in": channels, "C_out": channels * 2})
                else:
                    _graph.edges[e].update({"C_in": channels, "C_out": channels})

        in_node = [n for n in _graph.nodes if _graph.in_degree(n) == 0][0]
        out_node = [n for n in _graph.nodes if _graph.out_degree(n) == 0][0]
        max_node_label = max(_graph.nodes())
        _graph.add_nodes_from([max_node_label + 1, max_node_label + 2])
        _graph.add_edge(max_node_label + 1, in_node)
        _graph.edges[max_node_label + 1, in_node].update(
            {
                "op": ops.Stem(_channels, C_in=in_channels),
                "op_name": "Stem",
            }
        )
        _graph.add_nodes_from([out_node, max_node_label + 2])
        _graph.add_edge(out_node, max_node_label + 2)

        _graph.edges[out_node, max_node_label + 2].update(
            {
                "op": ops.Sequential(
                    nn.BatchNorm2d(_channels * out_channels_factor),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(_channels * out_channels_factor, n_classes),
                ),
                "op_name": "Out",
            }
        )
        if return_model:
            _graph.compile()
            _graph.update_op_names()
            return _graph._to_pytorch()  # pylint: disable=protected-access

    _create_architecture(graph, base_channels)
    return None


def constraints(
    edge_list: list,
    none_operation: str,
):
    def _compute_pred_succ(edge_list: list):
        nodes = np.unique(edge_list)
        pred = {node: [] for node in nodes}
        succ = {node: [] for node in nodes}
        for u, v in edge_list:
            succ[u].append(v)
            pred[v].append(u)
        return {"pred": pred, "succ": succ}

    def _constraints(
        topology: str,
        current_derivation: list = None,
        pred_succ_mapping: dict = None,
        edge_list: dict = None,
        src_sink_map: dict = None,
        none_operation: str = None,
    ):
        if topology not in edge_list.keys():
            return None
        if current_derivation is None:
            return [None] * len(edge_list[topology])

        current_pred_succ_mapping = deepcopy(pred_succ_mapping[topology])
        for i, d in enumerate(current_derivation):
            if d is None:
                continue
            if (
                d.count("(") == 1 and d.count(")") == 1 and none_operation in d
            ) or none_operation == d:
                u, v = edge_list[topology][i]
                current_pred_succ_mapping["pred"][v].remove(u)
                current_pred_succ_mapping["succ"][u].remove(v)

        edge_idx = current_derivation.index(None)
        cur_edge_list = [
            (u, v)
            for i, (u, v) in enumerate(edge_list[topology])
            if u in current_pred_succ_mapping["succ"]
            and v in current_pred_succ_mapping["succ"][u]
            and v in current_pred_succ_mapping["pred"]
            and u in current_pred_succ_mapping["pred"][v]
            and i != edge_idx  # what would happen if edge would be set to zero -> remove
        ]
        graph = nx.DiGraph()
        graph.add_edges_from(cur_edge_list)
        return (
            not (len(graph) == 0 or graph.number_of_edges() == 0)
            and src_sink_map[topology][0] in graph.nodes
            and src_sink_map[topology][1] in graph.nodes
            and nx.has_path(graph, src_sink_map[topology][0], src_sink_map[topology][1])
        )

    src_sink_map = {}
    for k, v in edge_list.items():
        first_nodes = {e[0] for e in v}
        second_nodes = {e[1] for e in v}
        (src_node,) = first_nodes - second_nodes
        (sink_node,) = second_nodes - first_nodes
        src_sink_map[k] = (src_node, sink_node)
    return partial(
        _constraints,
        edge_list=edge_list,
        src_sink_map=src_sink_map,
        pred_succ_mapping={k: _compute_pred_succ(v) for k, v in edge_list.items()},
        none_operation=none_operation,
    )


class NB201Spaces:
    def __new__(
        cls,
        space: str,
        dataset: str,
        return_graph_per_hierarchy: bool = True,
    ):
        repetitive_kwargs = {}
        if space == "fixed_1_none":  # => original NB201 space
            grammar_files = [
                "macro_fixed_repetitive.cfg",
                "cell.cfg",
            ]
            productions = [
                cls._read_grammar(grammar_file) for grammar_file in grammar_files
            ]
            repetitive_kwargs = {
                "fixed_macro_grammar": True,
                "terminal_to_sublanguage_map": {
                    "SharedCell": "CELL",
                },
            }
        elif space == "variable_multi_multi":  # => fully hierarchical
            grammar_files = ["macro.cfg", "cell_flexible.cfg", "conv_block.cfg"]
            productions = cls._filter_productions(
                "".join(
                    [cls._read_grammar(grammar_file) for grammar_file in grammar_files]
                )
            )
        else:
            raise NotImplementedError(f"Space {space} is not implemented")

        if dataset == "cifar10":
            build_fn = partial(
                build, n_classes=10,
            )
        elif dataset == "cifar100":
            build_fn = partial(
                build, n_classes=100,
            )
        elif dataset == "ImageNet16-120":
            build_fn = partial(
                build, n_classes=120,
            )
        elif dataset == "addNIST":
            build_fn = partial(
                build, n_classes=20,
            )
        elif dataset == "cifarTile":
            build_fn = partial(
                build, n_classes=4,
            )
        else:
            raise NotImplementedError(f"Dataset {dataset} is not implemented")

        constraints_kwargs = {
            "constraints": constraints(
                edge_list=get_edge_lists_of_topologies(PRIMITIVES), none_operation="zero"
            ),
            "none_operation": "zero",
        }

        return FunctionParameter(
            set_recursive_attribute=build_fn,
            old_build_api=True,
            name=f"hierarchical_nb201_{dataset}_{space}",
            structure=productions,
            primitives=PRIMITIVES,
            return_graph_per_hierarchy=return_graph_per_hierarchy,
            constraint_kwargs=constraints_kwargs,
            **repetitive_kwargs,
        )

    @staticmethod
    def _read_grammar(grammar_file: str) -> str:
        with open(Path(DIR_PATH) / "grammars" / grammar_file) as f:
            productions = f.read()
        return productions

    @staticmethod
    def _filter_productions(productions: str) -> str:
        filtered_productions = ""
        already_seen_productions = []
        for prod in productions.split("\n"):
            if prod != "" and prod not in already_seen_productions:
                filtered_productions += prod + "\n"
                already_seen_productions.append(prod)
        return filtered_productions


NB201_HIERARCHIES_CONSIDERED = {
    "nb201_fixed_1_none": [],
    "nb201_variable_multi_multi": [0, 1, 2, 3, 4],
}

if __name__ == "__main__":
    import logging
    import math
    import time

    # pylint: disable=ungrouped-imports
    from neps.optimizers.bayesian_optimization.kernels import GraphKernelMapping
    from neps.optimizers.bayesian_optimization.models.gp_hierarchy import (
        ComprehensiveGPHierarchy,
    )

    # pylint: enable=ungrouped-imports

    def run_pipeline(architecture):
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

    # Information to search spaces
    for space in [
        "fixed_1_none",
        "variable_multi_multi",
    ]:
        pipeline_space = dict(
            architecture=NB201Spaces(space=space, dataset="cifar10"),
        )
        pipeline_space = SearchSpace(**pipeline_space)

        sampled_pipeline_space = pipeline_space.sample(patience=3)
        _ = sampled_pipeline_space.hyperparameters["architecture"].to_pytorch()
        mutated_sampled_pipeline = sampled_pipeline_space.mutate()
        sampled_pipeline_space2 = pipeline_space.sample(patience=3)
        crossover_sampled_pipeline = sampled_pipeline_space.crossover(
            sampled_pipeline_space2
        )

        value = sampled_pipeline_space.hyperparameters["architecture"].value
        if len(value) == 1:
            value = value[0]
        hierarchy_graphs = value[1]
        print(
            space,
            math.log10(pipeline_space.hyperparameters["architecture"].search_space_size),
            len(hierarchy_graphs),
        )

    # test ops
    pipeline_space = dict(
        architecture=NB201Spaces(space="variable_multi_multi", dataset="cifar10"),
    )
    pipeline_space = SearchSpace(**pipeline_space)
    pipeline_space = pipeline_space.sample()
    _ = run_pipeline(pipeline_space.hyperparameters["architecture"])
    print(math.log10(pipeline_space.hyperparameters["architecture"].search_space_size))
    sampled_pipeline_space = pipeline_space.sample()
    _ = sampled_pipeline_space.hyperparameters["architecture"].to_pytorch()
    sampled_pipeline_space.mutate()
    pipeline_space2 = pipeline_space.copy()
    sampled_pipeline_space2 = pipeline_space2.sample()
    sampled_pipeline_space.crossover(sampled_pipeline_space2)

    hierarchy_considered = NB201_HIERARCHIES_CONSIDERED["variable_multi_multi"]
    graph_kernels = ["wl"] * (len(hierarchy_considered) + 1)
    wl_h = [2, 1] + [2] * (len(hierarchy_considered) - 1)
    graph_kernels = [
        GraphKernelMapping[kernel](
            h=wl_h[j],
            oa=False,
            se_kernel=None,
        )
        for j, kernel in enumerate(graph_kernels)
    ]
    surrogate_model = ComprehensiveGPHierarchy(
        graph_kernels=graph_kernels,
        hp_kernels=[],
        verbose=False,
        hierarchy_consider=hierarchy_considered,
        d_graph_features=0,  # set to 0 if not using additional graph topological features
        vectorial_features=pipeline_space.get_vectorial_dim()
        if hasattr(pipeline_space, "get_vectorial_dim")
        else None,
    )

    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        working_directory="results/hierarchical_architecture_example_new",
        max_evaluations_total=20,
        acquisition_sampler="evolution",
        surrogate_model=surrogate_model,
    )

    previous_results, pending_configs = neps.status(
        "results/hierarchical_architecture_example_new"
    )
