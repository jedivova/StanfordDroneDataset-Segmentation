from typing import Dict, List, Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        result = x * torch.sigmoid(x)
        return result


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    $$ mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))) $$
    Shape:
        - Input: (N, _) where ``_`` means, any number of additional
          dimensions
        - Output: (N, _), same shape as the input
    Examples:
        >>> mish = Mish()
        >>> input = torch.randn(2)
        >>> output = mish(input)
    """

    def __init__(self, inplace=True):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        result = x * torch.tanh(F.softplus(x))
        return result


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # print(x.size())
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SCSEModule(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 4):
        super().__init__()
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_se = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_se = x * self.channel_se(x)
        spatial_se = x * self.spatial_se(x)
        # print('channel_se',channel_se.shape , 'spatial_se', spatial_se.shape)
        return channel_se + spatial_se


class Identity(nn.Module):
    def __init__(self, num_channels: int):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def conv_block(
        out_channels: int,
        convolution: callable,
        norm_layer: callable = nn.BatchNorm2d,
        attention: callable = None,
        activation: callable = nn.LeakyReLU,
        attention_params: dict = None,
        activation_params: dict = None,
) -> nn.Module:
    args = [convolution]

    if norm_layer is not None:
        args.append(norm_layer(out_channels))

    if attention is not None:
        attention_params = attention_params \
            if attention_params is not None else {}
        args.append(attention(**attention_params))

    if activation is not None:
        activation_params = activation_params \
            if activation_params is not None else {}
        args.append(activation(**activation_params))

    result = nn.Sequential(*args)

    return result


def conv_1x1(
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        groups: int = 1,
        conv_layer: callable = nn.Conv2d,
        norm_layer: callable = nn.BatchNorm2d,
        attention: callable = None,
        activation: callable = nn.LeakyReLU,
        bias: bool = True,
        attention_params: dict = None,
        activation_params: dict = None,
) -> nn.Module:
    convolution = conv_layer(
        in_channels, out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=dilation,
        groups=groups,
        bias=bias
    )

    result = conv_block(
        out_channels,
        convolution,
        norm_layer=norm_layer,
        attention=attention,
        activation=activation,
        attention_params=attention_params,
        activation_params=activation_params
    )

    return result


def conv_3x3(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        conv_layer: callable = nn.Conv2d,
        norm_layer: callable = nn.BatchNorm2d,
        attention: callable = None,
        activation: callable = nn.LeakyReLU,
        bias: bool = True,
        attention_params: dict = None,
        activation_params: dict = None,
) -> nn.Module:
    convolution = conv_layer(
        in_channels, out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        dilation=dilation,
        groups=groups,
        bias=bias
    )

    result = conv_block(
        out_channels,
        convolution,
        norm_layer=norm_layer,
        attention=attention,
        activation=activation,
        attention_params=attention_params,
        activation_params=activation_params
    )

    return result


def conv_5x5(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        conv_layer: callable = nn.Conv2d,
        norm_layer: callable = nn.BatchNorm2d,
        attention: callable = None,
        activation: callable = nn.LeakyReLU,
        bias: bool = True,
        attention_params: dict = None,
        activation_params: dict = None,
) -> nn.Module:
    convolution = conv_layer(
        in_channels, out_channels,
        kernel_size=5,
        stride=stride,
        padding=2,
        dilation=dilation,
        groups=groups,
        bias=bias
    )

    result = conv_block(
        out_channels,
        convolution,
        norm_layer=norm_layer,
        attention=attention,
        activation=activation,
        attention_params=attention_params,
        activation_params=activation_params
    )

    return result


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


activation_dict = {
    "LeakyReLU": nn.LeakyReLU,
    "Mish": Mish,
    "Hswish": Hswish,
}

attentions_dict = {
    "Identity": Identity,
    "SCSEModule": SCSEModule,
}


class MobileBottleneck(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            expansion_channels: int,
            activation_name: str = "LeakyReLU",
            attention_name: str = "Identity",
    ):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]

        if kernel_size == 3:
            middle_conv = conv_3x3
        elif kernel_size == 5:
            middle_conv = conv_5x5
        else:
            raise ValueError(f"Invalid kernel size {kernel_size}. Should be either 3 or 5.")

        self.use_residual = stride == 1 and in_channels == out_channels

        activation = activation_dict[activation_name]
        attention = attentions_dict[attention_name]

        self.block = nn.Sequential(
            conv_1x1(
                in_channels, expansion_channels,
                activation=activation,
                activation_params={"inplace": True}
            ),
            middle_conv(
                expansion_channels, expansion_channels,
                stride=stride,
                groups=expansion_channels,
                attention=attention,
                activation=activation,
                activation_params={"inplace": True},
                attention_params={"num_channels": expansion_channels}
            ),
            conv_1x1(
                expansion_channels, out_channels,
                activation=None
            )
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)


class ClassificationBlock(nn.Module):
    def __init__(self, last_channel, exp_channels_num, num_classes):
        super(ClassificationBlock, self).__init__()
        self.se = SCSEModule(last_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(last_channel, exp_channels_num, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels_num)
        self.act1 = Mish(inplace=True)

        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.Linear(exp_channels_num, num_classes)

    def forward(self, x):
        # print()
        out = self.se(x)
        # print('.se', out.shape)
        out = self.avgpool(out)
        # print('.avgpool', out.shape)

        out = self.conv1(out)
        # print('.conv1', out.shape)
        out = self.bn1(out)
        out = self.act1(out)
        # print('.act1', out.shape)

        # flatten for input to fully-connected layer
        out = out.view(out.size(0), -1)
        # print('.view', out.shape)
        out = self.fc(self.dropout(out))
        # print('.fc', out.shape)
        return out


class MobileNetV3(nn.Module):
    def __init__(
            self,
            num_classes: int = None,
            width_multiplier: float = 1.0,
            mode: str = "small",
            in_channels=3
    ):
        super(MobileNetV3, self).__init__()
        input_channel = 16

        # ! NOTE stage
        # [kernel_size, expansion_channels_before_multiplier, channels_before_multiplier, attention, activation, stride]
        if mode == "large":
            self.stage_list = [
                [
                    [3, 16, 16, "Identity", "LeakyReLU", 1],
                    [3, 64, 24, "Identity", "LeakyReLU", 2],
                    [3, 72, 24, "Identity", "LeakyReLU", 1],
                ],
                [
                    [5, 72, 40, "SCSEModule", "LeakyReLU", 2],
                    [5, 120, 40, "SCSEModule", "LeakyReLU", 1],
                    [5, 120, 40, "SCSEModule", "LeakyReLU", 1],
                ],
                [
                    [3, 240, 80, "Identity", "Mish", 2],
                    [3, 200, 80, "Identity", "Mish", 1],
                    [3, 184, 80, "Identity", "Mish", 1],
                    [3, 184, 80, "Identity", "Mish", 1],
                    [3, 480, 112, "SCSEModule", "Mish", 1],
                    [3, 672, 112, "SCSEModule", "Mish", 1],
                ],
                [
                    [5, 672, 160, "SCSEModule", "Mish", 2],
                    [5, 960, 160, "SCSEModule", "Mish", 1],
                    [5, 960, 160, "SCSEModule", "Mish", 1],
                ]
            ]
        elif mode == "small":
            self.stage_list = [
                [
                    [3, 16, 16, "SCSEModule", "LeakyReLU", 2],
                ],
                [
                    [3, 72, 24, "Identity", "LeakyReLU", 2],
                    [3, 88, 24, "Identity", "LeakyReLU", 1],
                ],
                [
                    [5, 96, 40, "SCSEModule", "Mish", 2],
                    [5, 240, 40, "SCSEModule", "Mish", 1],
                    [5, 240, 40, "SCSEModule", "Mish", 1],
                    [5, 120, 48, "SCSEModule", "Mish", 1],
                    [5, 144, 48, "SCSEModule", "Mish", 1],
                ],
                [
                    [5, 288, 96, "SCSEModule", "Mish", 2],
                    [5, 576, 96, "SCSEModule", "Mish", 1],
                    [5, 576, 96, "SCSEModule", "Mish", 1],
                ]
            ]

        # if input_size is None: input_size = [224, 224]
        # # building first layer
        # assert (input_size[0] % 32 == 0) and (input_size[1] % 32 == 0)
        self.first_conv = conv_3x3(in_channels, input_channel, 2, activation=Mish)

        # building mobile blocks
        self.modules_array = []
        for stage in self.stage_list:
            tmp_feats = []
            for k, exp, c, se, nl, s in stage:
                output_channel = make_divisible(c * width_multiplier)
                exp_channel = make_divisible(exp * width_multiplier)
                tmp_feats.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, nl, se))
                input_channel = output_channel
            self.modules_array.append(tmp_feats)

        if mode == "large":
            last_channel = make_divisible(960 * width_multiplier)
        elif mode == "small":
            last_channel = make_divisible(576 * width_multiplier)

        self.last_conv = conv_1x1(input_channel, last_channel, activation=Mish)
        self.last_channel = last_channel

        # make it nn.Sequential
        self.features = nn.ModuleList([nn.Sequential(*x) for x in self.modules_array])

        self.num_classes = num_classes
        if self.num_classes is not None:
            exp_channels_num = 1280 if width_multiplier <= 1 else make_divisible(1280 * width_multiplier, 8)
            self.classifier = ClassificationBlock(last_channel, exp_channels_num, num_classes)
        else:
            self.classifier = None

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs, *_ = x.shape
        out0 = self.first_conv(x)
        out0 = self.features[0](out0)
        out1 = self.features[1](out0)
        out2 = self.features[2](out1)
        out3 = self.features[3](out2)
        out3 = self.last_conv(out3)

        result = dict(
            out0=out0,
            out1=out1,
            out2=out2,
            out3=out3,
            features=out3
        )
        _classes = self.classifier(result['features'])
        return _classes

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)["logits"]
        return logits

    def _initialize_weights(self) -> None:
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def dice_loss(pred, target, eps=1e-10):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + eps) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + eps)))

    return loss.mean()


def calc_loss(pred, target, bce_weight=0.5):
    # bce = F.binary_cross_entropy_with_logits(pred, target)
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


def set_all_seeds():
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_printoptions(precision=10)