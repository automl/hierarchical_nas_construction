from copy import deepcopy
import os
from path import Path
from functools import partial
import inspect

import neps.search_spaces.graph_grammar.topologies as topos
from neps.search_spaces.graph_grammar.api import FunctionParameter
from neps.search_spaces.graph_grammar.graph import Graph
from torchvision import models
import torch
from torch import nn

import benchmarks.search_spaces.activation_function_search.unary_operations as UnaryOp
from benchmarks.search_spaces.activation_function_search.stacking import Stacking
import benchmarks.search_spaces.activation_function_search.kvary_operations as BinaryOp
from benchmarks.search_spaces.activation_function_search.topologies import BinaryTopo
import benchmarks.search_spaces.activation_function_search.cifar_models as cifar_models

DIR_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

PRIMITIVES = {
    "UnaryTopo": partial(topos.LinearNEdge, number_of_edges=1),
    "BinaryTopo": BinaryTopo,

    # unary ops
    "id": UnaryOp.Identity(),
    "neg": UnaryOp.Negate(),
    "abs": UnaryOp.Absolute(),
    "square": UnaryOp.Square(),
    "cubic": UnaryOp.Cubic(),
    "square_root": UnaryOp.SquareRoot(),
    "mconst": UnaryOp.MultConst,
    "aconst": UnaryOp.AddConst,
    "log": UnaryOp.Log(),
    "exp": UnaryOp.Exp(),
    "sin": UnaryOp.Sin(),
    "cos": UnaryOp.Cos(),
    "sinh": UnaryOp.Sinh(),
    "cosh": UnaryOp.Cosh(),
    "tanh": UnaryOp.Tanh(),
    "asinh": UnaryOp.aSinh(),
    "atanh": UnaryOp.aTanh(),
    "sinc": UnaryOp.Sinc(),
    "umax": UnaryOp.UnaryMax(),
    "umin": UnaryOp.UnaryMin(),
    "sigmoid": UnaryOp.Sigmoid(),
    "logexp": UnaryOp.LogExp(),
    "gaussian": UnaryOp.Gaussian(),
    "erf": UnaryOp.Erf(),
    "const": UnaryOp.Constant,

    # binary ops
    "add": BinaryOp.Addition(),
    "multi": BinaryOp.Multiplication(),
    "sub": BinaryOp.Subtraction(),
    "div": BinaryOp.Division(),
    "bmax": BinaryOp.BinaryMax(),
    "bmin": BinaryOp.BinaryMin(),
    "bsigmoid": BinaryOp.SigmoidMult(),
    "bgaussian_sq": BinaryOp.BinaryGaussianSquare,
    "bgaussian_abs": BinaryOp.BinaryGaussianAbs,
    "wavg": BinaryOp.WeightedAvg,
}

def set_comb_op(node, **kwargs):
    node[1]["comb_op"] = Stacking()

pytorch_activation_functions = [act[1] for act in inspect.getmembers(nn.modules.activation, inspect.isclass) if not (act[1] == nn.Module or act[1] == torch.Tensor or act[1] == nn.Parameter)]

def build(activation_function: Graph, base_architecture: str = "resnet20", num_classes: int = 10):
    def replace_activation_functions(base_module: nn.Module, activation_function: Graph, channels: int = -1):
        for name, module in base_module.named_children():
            if hasattr(module, "out_channels") and module.out_channels > 0:
                channels = module.out_channels
            elif hasattr(module, "num_features") and module.num_features > 0:
                channels = module.num_features

            if any(isinstance(module, act) for act in pytorch_activation_functions):
                activation_function_copied = deepcopy(activation_function)
                for _, _, data in activation_function_copied.edges(data=True):
                    if hasattr(data["op"], "trainable") and data["op"].trainable:
                        data.update({"op": data["op"](channels)})
                activation_function_copied.compile()
                activation_function_copied.update_op_names()
                activation_function_copied = activation_function_copied._to_pytorch()  # pylint: disable=protected-access
                setattr(base_module, name, activation_function_copied)
            elif isinstance(module, nn.Module):
                new_module = replace_activation_functions(base_module=module, activation_function=activation_function, channels=channels)
                setattr(base_module, name, new_module)
        return base_module


    print(base_architecture)
    if hasattr(cifar_models, base_architecture):
        base_model = getattr(cifar_models, base_architecture)(num_classes=num_classes)
    elif hasattr(models, base_architecture):
        base_model = getattr(models, base_architecture)(pretrained=False, num_classes=num_classes)
    else:
        raise NotImplementedError(f"Model {base_architecture} is not implemented!")

    # set stacking as combo op
    activation_function.update_nodes(
        update_func=lambda node, in_edges, out_edges: set_comb_op(
            node, **{"a": in_edges, "b": out_edges}
        ),
        single_instances=False,
    )

    model = replace_activation_functions(base_module=base_model, activation_function=activation_function)

    return model

class ActivationSpace:
    def __new__(cls, base_architecture:str="resnet20", dataset: str ="cifar10", return_graph_per_hierarchy: bool = True):
        assert hasattr(cifar_models, base_architecture) or hasattr(models, base_architecture)

        if dataset == "cifar10":
            build_fn = partial(build, base_architecture=base_architecture, num_classes=10)
        elif dataset == "cifar100":
            build_fn = partial(build, base_architecture=base_architecture, num_classes=100)
        else:
            raise NotImplementedError(f"Dataset {dataset} is not supported")

        productions = cls._read_grammar("grammar.cfg")

        return FunctionParameter(
            set_recursive_attribute=build_fn,
            old_build_api=True,
            name=f"activation_{dataset}_{base_architecture}",
            structure=productions,
            primitives=PRIMITIVES,
            return_graph_per_hierarchy=return_graph_per_hierarchy,
            constraint_kwargs=None,
            prior=None,
        )

    @staticmethod
    def _read_grammar(grammar_file: str) -> str:
        with open(Path(DIR_PATH) / grammar_file) as f:
            productions = f.read()
        return productions

if __name__ == "__main__":
    from neps.search_spaces.search_space import SearchSpace
    import math

    pipeline_space = dict(
        architecture=ActivationSpace(base_architecture="resnet20"),
    )
    pipeline_space = SearchSpace(**pipeline_space)
    print(
        "benchmark",
        math.log10(pipeline_space.hyperparameters["architecture"].search_space_size),
    )

    pipeline_space.load({
        "architecture": "(L2 UnaryTopo (L1 UnaryTopo (umax)))"
    })
    print(pipeline_space["architecture"].id)

    model = pipeline_space.hyperparameters["architecture"].to_pytorch()
    print(model)
