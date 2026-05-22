import torch
from escnn import gspaces, nn


class EquiResBlock(torch.nn.Module):
    def __init__(
        self,
        group: gspaces.GSpace2D,
        input_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        initialize: bool = True,
        use_norms: bool = True,
    ):
        super(EquiResBlock, self).__init__()
        self.group = group
        self.use_norms = use_norms
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
        # self.bn1 = nn.InnerBatchNorm(feat_type_hid, affine=True, track_running_stats=True)

        # self.bn2 = nn.InnerBatchNorm(feat_type_hid, affine=True, track_running_stats=True)
        self.conv2 = nn.R2Conv(
            feat_type_hid,
            feat_type_hid,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
            initialize=initialize,
        )

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
                # nn.InnerBatchNorm(feat_type_hid, affine=True, track_running_stats=True)
            )
        else:
            self.skip = torch.nn.Identity()

    def forward(self, xx):
        # identity = xx if self.skip is None else self.skip(xx)   # always from xx
        # out = self.relu1(self.bn1(self.conv1(xx)))
        # out = self.bn2(self.conv2(out))
        # out = out + identity
        # return self.relu2(out)

        out = self.relu(self.conv1(xx))
        out = self.conv2(out)
        return self.relu(out + self.skip(xx))


class EquivariantResEncoder76Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 2, n_out: int = 128, initialize: bool = True, N=8, use_norms: bool = True):
        super().__init__()
        self.obs_channel = obs_channel
        self.use_norms = use_norms
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
            # nn.InnerBatchNorm(
            #     nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
            #     affine=True, track_running_stats=True,
            # ),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr])),
        ]

        layers += [
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=initialize, use_norms=use_norms),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=initialize, use_norms=use_norms),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 36x36
            EquiResBlock(self.group, n_out // 8, n_out // 4, initialize=initialize, use_norms=use_norms),
            EquiResBlock(self.group, n_out // 4, n_out // 4, initialize=initialize, use_norms=use_norms),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 18x18
            EquiResBlock(self.group, n_out // 4, n_out // 2, initialize=initialize, use_norms=use_norms),
            EquiResBlock(self.group, n_out // 2, n_out // 2, initialize=initialize, use_norms=use_norms),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 9x9
            EquiResBlock(self.group, n_out // 2, n_out, initialize=initialize, use_norms=use_norms),
            EquiResBlock(self.group, n_out, n_out, initialize=initialize, use_norms=use_norms),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 3),
            # 3x3
        ]

        layers += [
            nn.R2Conv(
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                bias=not use_norms,
                initialize=initialize,
            ),
            # nn.InnerBatchNorm(
            #     nn.FieldType(self.group, n_out * [self.group.regular_repr]),
            #     affine=True, track_running_stats=True,
            # ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        ]

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)