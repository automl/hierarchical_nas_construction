import torch
from neps.search_spaces.graph_grammar.primitives import AbstractPrimitive
from torch import nn


class DARTSAbstractPrimitive(AbstractPrimitive):
    def __init__(self, C: int, stride: int, affine: bool = True):
        super().__init__(locals())
        self.C = C
        self.stride = stride
        self.affine = affine

    def forward(self, x):
        raise NotImplementedError


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class SkipConnect(DARTSAbstractPrimitive):
    def __init__(self, C: int, stride: int, affine: bool = True):
        super().__init__(C, stride, affine)

        self.identity = (
            nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine)
        )

    def forward(self, x):
        return self.identity(x)

    @property
    def get_op_name(self):
        return "skip_connect"


class Zero(DARTSAbstractPrimitive):
    def __init__(self, C: int, stride: int, affine: bool = True):
        super().__init__(C, stride, affine)

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)

    @property
    def get_op_name(self):
        return "none"


class Pooling(DARTSAbstractPrimitive):
    def __init__(self, pool_type: str, C: int, stride, affine: bool = True):
        super().__init__(C, stride, affine)

        assert pool_type in ["avg", "max"]
        self.pool_type = pool_type
        if "avg" == pool_type:
            self.pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif "max" == pool_type:
            self.pool = nn.MaxPool2d(3, stride=stride, padding=1)

    def forward(self, x):
        return self.pool(x)

    @property
    def get_op_name(self):
        return f"{self.pool_type}_pool_3x3"


class SepConv(DARTSAbstractPrimitive):
    def __init__(self, kernel_size: int, C: int, stride: int, affine: bool = True):
        super().__init__(C, stride, affine)
        C_in = C_out = C
        padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    @property
    def get_op_name(self):
        return f"sep_conv_{self.kernel_size}x{self.kernel_size}"


class DilConv(DARTSAbstractPrimitive):
    def __init__(self, kernel_size: int, C: int, stride: int, affine: bool = True):
        super().__init__(C, stride, affine)
        C_in = C_out = C
        padding = (kernel_size // 2) * 2
        dilation = 2
        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    @property
    def get_op_name(self):
        return f"dil_conv_{self.kernel_size}x{self.kernel_size}"


class Concat(AbstractPrimitive):
    """
    Implementation of the channel-wise concatination.
    """

    def __init__(self):
        super().__init__(locals())

    def forward(self, x):  # pylint: disable=no-self-use
        """
        Expecting a list of input tensors. Stacking them channel-wise.
        """
        x = torch.cat(x, dim=1)
        return x


class Unbinder(AbstractPrimitive):
    def __init__(self, idx):
        super().__init__(locals())
        self.idx = idx

    def forward(self, x):
        return torch.unbind(x, dim=0)[self.idx]


class Stacking(AbstractPrimitive):
    def __init__(self, **kwargs):  # pylint: disable=W0613
        super().__init__(locals())

    def forward(self, x):  # pylint: disable=no-self-use
        return torch.stack(x, dim=0)
