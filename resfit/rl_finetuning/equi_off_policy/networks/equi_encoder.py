import torch
from escnn import gspaces, nn

from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import (
    RegularInnerBatchNorm,
    apply_deltaortho_init,
)

ENCODER_NORMS = ("none", "inner_bn")


def conv_norm(kind: str, field_type: nn.FieldType) -> list[torch.nn.Module]:
    """The normalization after a conv, as a list to splice into a Sequential.

    Empty for "none", so the encoder every run before 2026-09-29 used keeps its
    module indices. InnerBatchNorm normalizes each regular field with statistics
    shared by its channels, so it is equivariant. Before 2026-10-01 this was
    escnn's InnerBatchNorm, which computes the same but syncs; no run used it.
    """
    if kind == "none":
        return []
    if kind == "inner_bn":
        return [RegularInnerBatchNorm(field_type)]
    raise ValueError(f"unknown encoder norm {kind!r}; expected one of {ENCODER_NORMS}")


class EquiResBlock(torch.nn.Module):
    """relu(norm(conv2(relu(norm(conv1(x))))) + skip(x)); every conv is bias-free."""

    def __init__(
        self,
        group: gspaces.GSpace2D,
        input_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        initialize: bool = True,
        norm: str = "none",
    ):
        super(EquiResBlock, self).__init__()
        self.group = group
        rep = self.group.regular_repr

        feat_type_in = nn.FieldType(self.group, input_channels * [rep])
        feat_type_hid = nn.FieldType(self.group, hidden_dim * [rep])
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        # Pre-activation: BN -> ReLU -> Conv (split so x_n can feed the skip)
        self.conv1 = nn.R2Conv(
            feat_type_in,
            feat_type_hid,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            bias=False,
            initialize=initialize,
        )
        self.bn1 = torch.nn.Sequential(*conv_norm(norm, feat_type_hid))
        self.conv2 = nn.R2Conv(
            feat_type_hid,
            feat_type_hid,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
            initialize=initialize,
        )
        self.bn2 = torch.nn.Sequential(*conv_norm(norm, feat_type_hid))

        self.skip = None
        if input_channels != hidden_dim or stride != 1:
            self.skip = torch.nn.Sequential(
                nn.R2Conv(
                    feat_type_in,
                    feat_type_hid,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    initialize=initialize,
                ),
                *conv_norm(norm, feat_type_hid),
            )
        else:
            self.skip = torch.nn.Identity()

    def forward(self, xx):
        out = self.relu(self.bn1(self.conv1(xx)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(xx))


class EquivariantResEncoder76Cyclic(torch.nn.Module):
    """The C_N-equivariant agentview encoder, 76x76 in, a 1x1 map of n_out regular fields out.

    Every conv is bias-free, as in every run so far, and followed by `norm`
    (ENCODER_NORMS). orth_init replaces escnn's He init with delta-orthonormal.
    """

    def __init__(self, obs_channel: int = 2, n_out: int = 128, initialize: bool = True, N=8,
                 norm: str = "none", orth_init: bool = False):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR2(N)

        layers = [
            # 76x76
            nn.R2Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                kernel_size=5,
                padding=0,
                bias=False,
                initialize=initialize,
            ),
            # 72x72
            *conv_norm(norm, nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr])),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr])),
        ]

        layers += [
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=initialize, norm=norm),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=initialize, norm=norm),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 36x36
            EquiResBlock(self.group, n_out // 8, n_out // 4, initialize=initialize, norm=norm),
            EquiResBlock(self.group, n_out // 4, n_out // 4, initialize=initialize, norm=norm),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 18x18
            EquiResBlock(self.group, n_out // 4, n_out // 2, initialize=initialize, norm=norm),
            EquiResBlock(self.group, n_out // 2, n_out // 2, initialize=initialize, norm=norm),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 9x9
            EquiResBlock(self.group, n_out // 2, n_out, initialize=initialize, norm=norm),
            EquiResBlock(self.group, n_out, n_out, initialize=initialize, norm=norm),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 3),
            # 3x3
        ]

        layers += [
            nn.R2Conv(
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                bias=False,
                initialize=initialize,
            ),
            *conv_norm(norm, nn.FieldType(self.group, n_out * [self.group.regular_repr])),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        ]

        self.conv = torch.nn.Sequential(*layers)
        if orth_init:
            apply_deltaortho_init(self.conv, exclude_final_layer=False)

    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)