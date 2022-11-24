from turtle import forward
import torch

from neps.search_spaces.graph_grammar.primitives import AbstractPrimitive

EPS = 1e-6

class BinaryOperation(AbstractPrimitive):
    trainable = False

    def __init__(
        self, **kwargs
    ):  # pylint:disable=W0613
        super().__init__(locals())

    def forward(self, x):  # pylint: disable=W0613
        raise NotImplementedError

    @staticmethod
    def get_embedded_ops():
        return None

class TrainableBinaryOperation(AbstractPrimitive):
    trainable = True

    def __init__(
        self, in_channels, **kwargs
    ):  # pylint:disable=W0613
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x):  # pylint: disable=W0613
        raise NotImplementedError

    @staticmethod
    def get_embedded_ops():
        return None

class Addition(BinaryOperation):
    def forward(self, x):
        return torch.sum(x, dim=0)

class Multiplication(BinaryOperation):
    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=0)
        return x1 * x2

class Subtraction(BinaryOperation):
    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=0)
        return x1 - x2

class Division(BinaryOperation):
    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=0)
        return x1 / (x2 + EPS)

class BinaryMax(BinaryOperation):
    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=0)
        return torch.maximum(x1, x2)

class BinaryMin(BinaryOperation):
    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=0)
        return torch.minimum(x1, x2)

class SigmoidMult(BinaryOperation):
    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=0)
        return torch.sigmoid(x1) * x2

class BinaryGaussianSquare(TrainableBinaryOperation):
    get_op_name = "BinaryGaussianSquare"

    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=0)
        return torch.exp(-self.beta*torch.square(x1-x2))

class BinaryGaussianAbs(TrainableBinaryOperation):
    get_op_name = "BinaryGaussianAbs"

    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=0)
        return torch.exp(-self.beta*torch.abs(x1-x2))

class WeightedAvg(TrainableBinaryOperation):
    get_op_name = "WeightedAvg"

    def forward(self, x):
        x1, x2 = torch.unbind(x, dim=0)
        return self.beta * x1 + (1-self.beta) * x2
