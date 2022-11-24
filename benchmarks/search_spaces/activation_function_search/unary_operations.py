import torch

from neps.search_spaces.graph_grammar.primitives import AbstractPrimitive

EPS = 1e-6

class UnaryOperation(AbstractPrimitive):
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

class TrainableUnaryOperation(UnaryOperation):
    trainable = True

    def __init__(self, in_channels, **kwargs):
        super().__init__(**kwargs)
        self.beta = torch.nn.Parameter(torch.ones(in_channels, 1, 1))


class Identity(UnaryOperation):
    def forward(self, x):
        return x

class Negate(UnaryOperation):
    def forward(self, x):
        return -x

class Absolute(UnaryOperation):
    def forward(self, x):
        return torch.abs(x)

class Square(UnaryOperation):
    def forward(self, x):
        return x ** 2

class Cubic(UnaryOperation):
    def forward(self, x):
        return x ** 3

class SquareRoot(UnaryOperation):
    def forward(self, x):
        return torch.sqrt(x)

class MultConst(TrainableUnaryOperation):
    get_op_name = "multConst"

    def forward(self, x):
        return self.beta * x

class AddConst(TrainableUnaryOperation):
    get_op_name = "addConst"

    def forward(self, x):
        return self.beta + x

class Log(UnaryOperation):
    def forward(self, x):
        return torch.log(torch.abs(x) + EPS)

class Exp(UnaryOperation):
    def forward(self, x):
        return torch.exp(x)

class Sin(UnaryOperation):
    def forward(self, x):
        return torch.sin(x)

class Cos(UnaryOperation):
    def forward(self, x):
        return torch.cos(x)

class Sinh(UnaryOperation):
    def forward(self, x):
        return torch.sinh(x)

class Cosh(UnaryOperation):
    def forward(self, x):
        return torch.cosh(x)

class Tanh(UnaryOperation):
    def forward(self, x):
        return torch.tanh(x)

class aSinh(UnaryOperation):
    def forward(self, x):
        return torch.asinh(x)

class aTanh(UnaryOperation):
    def forward(self, x):
        return torch.atanh(x)

class Sinc(UnaryOperation):
    def forward(self, x):
        return torch.sinc(x)

class UnaryMax(UnaryOperation):
    def forward(self, x):
        return torch.maximum(x, torch.zeros_like(x))

class UnaryMin(UnaryOperation):
    def forward(self, x):
        return torch.minimum(x, torch.zeros_like(x))

class Sigmoid(UnaryOperation):
    def forward(self, x):
        return torch.sigmoid(x)

class LogExp(UnaryOperation):
    def forward(self, x):
        return torch.log(1 + torch.exp(x))

class Gaussian(UnaryOperation):
    def forward(self, x):
        return torch.exp(-x**2)

class Erf(UnaryOperation):
    def forward(self, x):
        return torch.erf(x)

class Constant(TrainableUnaryOperation):
    get_op_name = "constant"

    def forward(self, x):
        return torch.ones_like(x) * self.beta
