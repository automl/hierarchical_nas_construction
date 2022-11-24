import torch

from neps.search_spaces.graph_grammar.primitives import AbstractPrimitive

class Stacking(AbstractPrimitive):
    def __init__(
        self, **kwargs
    ):  # pylint:disable=W0613
        super().__init__(locals())

    def forward(self, x):  # pylint: disable=W0613
        return torch.stack(x, dim=0)

    @staticmethod
    def get_embedded_ops():
        return None
