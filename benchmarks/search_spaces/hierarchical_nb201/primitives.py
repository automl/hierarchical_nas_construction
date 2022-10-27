from neps.search_spaces.graph_grammar.primitives import AbstractPrimitive
from torch import nn


class ResNetBasicblock(AbstractPrimitive):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        stride: int,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__(locals())
        assert stride == 1 or stride == 2, f"invalid stride {stride}"
        self.conv_a = ReLUConvBN(
            C_in, C_out, 3, stride, 1, 1, affine, track_running_stats
        )
        self.conv_b = ReLUConvBN(C_out, C_out, 3, 1, 1, 1, affine, track_running_stats)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False),
            )
        elif C_in != C_out:
            self.downsample = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        else:
            self.downsample = None
        self.in_dim = C_in
        self.out_dim = C_out
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class ReLUConvBN(AbstractPrimitive):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super().__init__(locals())
        kernel_size = int(kernel_size)
        stride = int(stride)

        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += f"{self.kernel_size}x{self.kernel_size}"
        return op_name


class POOLING(AbstractPrimitive):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        stride: int,
        mode: str,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__(locals())
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        self.mode = mode
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError(f"Invalid mode={mode} in POOLING")

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)

    @property
    def get_op_name(self):
        return f"{self.mode}pool"


class Conv(AbstractPrimitive):
    def __init__(
        self, C_in: int, C_out: int, kernel_size: int, stride: int = 1, bias: bool = False
    ):
        super().__init__(locals())
        pad = 0 if stride == 1 and kernel_size == 1 else 1
        self.conv = nn.Conv2d(
            C_in, C_out, kernel_size, stride=stride, padding=pad, bias=bias
        )

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv(AbstractPrimitive):
    def __init__(
        self,
        C_in: int,
        C_out: int,  # pylint: disable=W0613
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__(locals())
        self.conv = nn.Conv2d(
            C_in,
            C_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=C_in,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class Normalization(AbstractPrimitive):
    def __init__(
        self,
        C_out: int,
        norm_type: str,
        affine: bool = True,
        **kwargs,  # pylint: disable=W0613
    ):
        super().__init__(locals())
        self.norm_type = norm_type
        self.affine = affine
        if norm_type == "batch_norm":
            self.norm = nn.BatchNorm2d(C_out, affine=affine)
        elif norm_type == "layer_norm":
            self.norm = None
        elif norm_type == "instance_norm":
            self.norm = nn.InstanceNorm2d(C_out, affine=affine)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.norm_type == "layer_norm" and self.norm is None:
            self.norm = nn.LayerNorm(x.shape[1:], elementwise_affine=self.affine)
            if x.is_cuda:
                self.norm = self.norm.cuda()
        return self.norm(x)


class Activation(AbstractPrimitive):
    def __init__(self, C_out: int, act_type: str, **kwargs):  # pylint: disable=W0613
        super().__init__(locals())
        self.act_type = act_type
        if act_type == "relu":
            self.act = nn.ReLU(inplace=False)
        elif act_type == "gelu":
            self.act = nn.GELU()
        elif act_type == "silu":
            self.act = nn.SiLU(inplace=False)
        elif act_type == "hardswish":
            self.act = nn.Hardswish(inplace=False)
        elif act_type == "mish":
            self.act = nn.Mish(inplace=False)
        else:
            raise NotImplementedError

    def forward(self, x):  # pylint: disable=W0613
        return self.act(x)
