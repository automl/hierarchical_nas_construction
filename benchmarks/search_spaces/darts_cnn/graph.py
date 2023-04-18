import itertools
import os
from copy import deepcopy

from neps.search_spaces.graph_grammar.api import FunctionParameter
from neps.search_spaces.graph_grammar.cfg_variants.constrained_cfg import Constraint
from path import Path

import benchmarks.search_spaces.darts_cnn.primitives as darts_primitives
from benchmarks.search_spaces.darts_cnn.topologies import DARTSCell

DIR_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

PRIMITIVES = {
    # macro level
    # cell level
    "sep_conv_3x3": {
        "op": darts_primitives.SepConv,
        "kernel_size": 3,
    },
    "sep_conv_5x5": {
        "op": darts_primitives.SepConv,
        "kernel_size": 5,
    },
    "dil_conv_3x3": {
        "op": darts_primitives.DilConv,
        "kernel_size": 3,
    },
    "dil_conv_5x5": {
        "op": darts_primitives.DilConv,
        "kernel_size": 5,
    },
    "avg_pool_3x3": {
        "op": darts_primitives.Pooling,
        "pool_type": "avg",
    },
    "max_pool_3x3": {
        "op": darts_primitives.Pooling,
        "pool_type": "max",
    },
    "skip_connect": {
        "op": darts_primitives.SkipConnect,
    },
    "none": {
        "op": darts_primitives.Zero,
    },
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "DARTS": DARTSCell,
    # conv block level
}


def build(
    darts_cell,
):
    darts_id = darts_cell.id
    darts_id_op_splitted = darts_id.split("OP ")
    ops = [split[: split.find(")")] for split in darts_id_op_splitted[1:]]
    darts_id_node_splitted = darts_id.split("IN")
    in_nodes = [int(split[2]) for split in darts_id_node_splitted[1:]]
    return [(op, in_node) for op, in_node in zip(ops, in_nodes)]


class DARTSSpace:
    def __new__(
        cls,
        dataset: str = "cifar10",
        return_graph_per_hierarchy: bool = False,
    ):
        productions = cls._read_grammar("cell.cfg")

        constraints_kwargs = {
            "constraints": darts_constraint(),
        }

        return FunctionParameter(
            set_recursive_attribute=build,
            old_build_api=True,
            name=f"darts_{dataset}",
            structure=productions,
            primitives=PRIMITIVES,
            return_graph_per_hierarchy=return_graph_per_hierarchy,
            new_graph_repr_func=True,
            constraint_kwargs=constraints_kwargs,
            prior=None,
            identity_op=[],
        )

    @staticmethod
    def _read_grammar(grammar_file: str) -> str:
        with open(Path(DIR_PATH) / grammar_file) as f:
            productions = f.read()
        return productions


def darts_constraint():
    class DARTSConstraint(Constraint):
        def __init__(self, current_derivation: str = None) -> None:
            super().__init__(current_derivation)

        @staticmethod
        def initialize_constraints(topology: str):
            if topology == "DARTS":
                return DARTSConstraint(current_derivation=[None] * 16)
            return None

        def get_not_allowed_productions(self, productions: list):
            idx = self.current_derivation.index(None)
            if idx % 2 == 0:
                return []
            else:
                if idx % 4 == 1:
                    return []
                else:
                    other_val = self.current_derivation[idx - 2].split(" ")[-1][:-1]
                    return [
                        production
                        for production in productions
                        if production.rhs()[0] == other_val
                    ]

        def update_context(self, new_part: str):
            idx = self.current_derivation.index(None)
            self.current_derivation[idx] = new_part

        @staticmethod
        def get_all_potential_productions(production):
            if str(production.lhs()) == "CELL":
                productions = []
                for i1_edge1, i1_edge2 in itertools.combinations("12", 2):
                    for i2_edge1, i2_edge2 in itertools.combinations("123", 2):
                        for i3_edge1, i3_edge2 in itertools.combinations("1234", 2):
                            for i4_edge1, i4_edge2 in itertools.combinations("12345", 2):
                                in_edges = [
                                    i1_edge1,
                                    i1_edge2,
                                    i2_edge1,
                                    i2_edge2,
                                    i3_edge1,
                                    i3_edge2,
                                    i4_edge1,
                                    i4_edge2,
                                ]
                                new_production = deepcopy(production)
                                indices_of_in_edges = [
                                    i
                                    for i, r in enumerate(new_production.rhs())
                                    if "IN" in str(r)
                                ]
                                new_production._rhs = tuple(  # pylint: disable=protected-access
                                    r
                                    if not i in indices_of_in_edges
                                    else in_edges[indices_of_in_edges.index(i)]
                                    for i, r in enumerate(new_production.rhs())
                                )
                                productions.append(new_production)
                return productions
            return [production]

        @staticmethod
        def mutate_not_allowed_productions(
            nonterminal: str, before: str, after: str, possible_productions: list
        ):
            if "IN" not in nonterminal:
                return []

            if nonterminal in before:
                in_node_already_used = before[before.find(nonterminal) + 4]
            elif nonterminal in after:
                in_node_already_used = after[after.find(nonterminal) + 4]
            else:
                raise NotImplementedError
            return [
                production
                for production in possible_productions
                if production.rhs()[0] == in_node_already_used
            ]

    return DARTSConstraint()


if __name__ == "__main__":
    import math

    # pylint: disable=ungrouped-imports
    from neps.search_spaces.search_space import SearchSpace

    from hierarchical_nas_benchmarks.search_spaces.darts_cnn.genotypes import (
        DrNAS_cifar10,
        Genotype,
    )
    from hierarchical_nas_benchmarks.search_spaces.darts_cnn.model import NetworkCIFAR
    from hierarchical_nas_benchmarks.search_spaces.darts_cnn.visualize import (
        plot,
        plot_from_graph,
    )

    # pylint: enable=ungrouped-imports

    DIR_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

    pipeline_space = dict(
        normal=DARTSSpace(return_graph_per_hierarchy=False),
        reduce=DARTSSpace(return_graph_per_hierarchy=False),
    )
    pipeline_space = SearchSpace(**pipeline_space)
    print(
        "benchmark",
        pipeline_space.hyperparameters["normal"].search_space_size,
        math.log10(pipeline_space.hyperparameters["normal"].search_space_size),
    )

    pipeline_space = pipeline_space.sample()
    _ = pipeline_space.mutate()

    # DrNAS cells
    pipeline_space.load_from(
        {
            "normal": "(CELL DARTS (OP sep_conv_3x3) (IN1 0) (OP sep_conv_5x5) (IN1 1) (OP sep_conv_3x3) (IN2 1) (OP sep_conv_3x3) (IN2 2) (OP skip_connect) (IN3 0) (OP sep_conv_3x3) (IN3 1) (OP sep_conv_3x3) (IN4 2) (OP dil_conv_5x5) (IN4 3))",
            "reduce": "(CELL DARTS (OP max_pool_3x3) (IN1 0) (OP sep_conv_5x5) (IN1 1) (OP dil_conv_5x5) (IN2 2) (OP sep_conv_5x5) (IN2 1) (OP sep_conv_5x5) (IN3 1) (OP dil_conv_5x5) (IN3 3) (OP skip_connect) (IN4 4) (OP sep_conv_5x5) (IN4 1))",
        }
    )

    normal = pipeline_space.hyperparameters["normal"].to_pytorch()
    reduce = pipeline_space.hyperparameters["reduce"].to_pytorch()
    assert normal == DrNAS_cifar10.normal
    assert reduce == DrNAS_cifar10.reduce

    hp_values = pipeline_space.get_normalized_hp_categories()
    plot_from_graph(hp_values["graphs"][0], DIR_PATH / "own_drnas_normal")
    plot_from_graph(hp_values["graphs"][1], DIR_PATH / "own_drnas_reduce")

    genotype = Genotype(
        normal=normal, normal_concat=range(2, 6), reduce=reduce, reduce_concat=range(2, 6)
    )
    plot(genotype=genotype.normal, filename=DIR_PATH / "drnas_normal")
    plot(genotype=genotype.reduce, filename=DIR_PATH / "drnas_reduce")

    # BANAMAS cells
    pipeline_space.load_from(
        {
            "normal": "(CELL DARTS (OP skip_connect) (IN1 1) (OP sep_conv_3x3) (IN1 0) (OP sep_conv_3x3) (IN2 1) (OP max_pool_3x3) (IN2 0) (OP sep_conv_5x5) (IN3 1) (OP sep_conv_3x3) (IN3 0) (OP dil_conv_5x5) (IN4 2) (OP sep_conv_3x3) (IN4 1))",
            "reduce": "(CELL DARTS (OP skip_connect) (IN1 1) (OP sep_conv_3x3) (IN1 0) (OP sep_conv_3x3) (IN2 1) (OP max_pool_3x3) (IN2 0) (OP sep_conv_5x5) (IN3 1) (OP sep_conv_3x3) (IN3 0) (OP dil_conv_5x5) (IN4 2) (OP sep_conv_3x3) (IN4 1))",
        }
    )
    hp_values = pipeline_space.get_normalized_hp_categories()
    plot_from_graph(hp_values["graphs"][0], DIR_PATH / "own_bananas")
    normal = pipeline_space.hyperparameters["normal"].to_pytorch()
    reduce = pipeline_space.hyperparameters["reduce"].to_pytorch()
    genotype = Genotype(
        normal=normal, normal_concat=range(2, 6), reduce=reduce, reduce_concat=range(2, 6)
    )
    plot(genotype=genotype.normal, filename=DIR_PATH / "bananas")

    model = NetworkCIFAR(
        C=36, num_classes=10, layers=20, auxiliary=True, genotype=genotype
    )
