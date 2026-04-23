from escnn import gspaces, nn
import torch

class EquiResBlock(torch.nn.Module):
    def __init__(
        self,
        group: gspaces.GSpace2D,
        input_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        initialize: bool = True,
    ):
        super(EquiResBlock, self).__init__()
        self.group = group
        rep = self.group.regular_repr

        feat_type_in = nn.FieldType(self.group, input_channels * [rep])
        feat_type_hid = nn.FieldType(self.group, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_in,
                feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                initialize=initialize,
            ),
            nn.ReLU(feat_type_hid, inplace=True),
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_hid,
                feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                initialize=initialize,
            ),
        )
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim or stride != 1:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=1, stride=stride, bias=False, initialize=initialize),
            )

    def forward(self, xx: nn.GeometricTensor) -> nn.GeometricTensor:
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out

class EquivariantVoxelEncoder58Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 4, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR3(N)
        self.conv = torch.nn.Sequential(
            # 58
            nn.R3Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            # 56
            nn.ReLU(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), 2),
            # 28
            nn.R3Conv(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 14
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            # 12
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 6
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 3
            nn.R3Conv(
                nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )
    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)
    
class EquivariantVoxelEncoder64Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 4, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR3(N)
        self.conv = torch.nn.Sequential(
            # 64
            nn.R3Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), 2),
            # 32
            nn.R3Conv(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 16
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 8
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            # 6
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 3
            nn.R3Conv(
                nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )
    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)
    
class EquivariantVoxelEncoder16Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 4, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR3(N)
        self.conv = torch.nn.Sequential(
            # 16
            nn.R3Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 8
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            # 6
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 3
            nn.R3Conv(
                nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )
    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)


########
# From Dian's equivairant RL paper
########

class EquivariantEncoder128(torch.nn.Module):
    """
    Equivariant Encoder. The input is a trivial representation with obs_channel channels.
    The output is a regular representation with n_out channels
    """
    def __init__(self, group, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = group
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        return self.conv(geo)

class EquivariantEncoder128Dihedral(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.d4_act = gspaces.Fliprot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.d4_act, obs_channel * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        return self.conv(geo)

class EquivariantEncoder128SO2(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.so2 = gspaces.rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.so2.irrep(0), self.so2.irrep(1), self.so2.irrep(2), self.so2.irrep(3)]
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.so2, obs_channel * [self.so2.trivial_repr]),
                      nn.FieldType(self.so2, n_out // 8 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 8 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out // 8 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 8 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out // 8 * self.repr), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.so2, n_out // 8 * self.repr),
                      nn.FieldType(self.so2, n_out // 4 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 4 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out // 4 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 4 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out // 4 * self.repr), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.so2, n_out // 4 * self.repr),
                      nn.FieldType(self.so2, n_out // 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 2 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out // 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 2 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out // 2 * self.repr), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.so2, n_out // 2 * self.repr),
                      nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out * self.repr), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.so2, n_out * self.repr),
                      nn.FieldType(self.so2, n_out * 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * 2 * self.repr),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * 2 * self.repr)),

            nn.R2Conv(nn.FieldType(self.so2, n_out * 2 * self.repr),
                      nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out * self.repr), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.so2, n_out * self.repr),
                      nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr)),
            # 1x1
        )

    def forward(self, geo):
        return self.conv(geo)

class EquivariantEncoder128O2(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.o2 = gspaces.Fliprot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.o2, obs_channel * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 8 * self.repr), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.o2, n_out // 8 * self.repr),
                      nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 4 * self.repr), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.o2, n_out // 4 * self.repr),
                      nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 2 * self.repr), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.o2, n_out // 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr)),

            nn.R2Conv(nn.FieldType(self.o2, n_out * 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            # 1x1
        )

    def forward(self, geo):
        return self.conv(geo)

class EquivariantResEncoder76Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 2, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 76x76
            nn.R2Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                kernel_size=5,
                padding=0,
                initialize=initialize,
            ),
            # 72x72
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=True),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 36x36
            EquiResBlock(self.group, n_out // 8, n_out // 4, initialize=True),
            EquiResBlock(self.group, n_out // 4, n_out // 4, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 18x18
            EquiResBlock(self.group, n_out // 4, n_out // 2, initialize=True),
            EquiResBlock(self.group, n_out // 2, n_out // 2, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 9x9
            EquiResBlock(self.group, n_out // 2, n_out, initialize=True),
            EquiResBlock(self.group, n_out, n_out, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 3),
            # 3x3
            nn.R2Conv(
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)

    # def forward(self, x) -> nn.GeometricTensor:
    #     import time
    #     torch.cuda.synchronize()
    #     t1 = time.perf_counter()
        
    #     if type(x) is torch.Tensor:
    #         x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        
    #     torch.cuda.synchronize()
    #     t2 = time.perf_counter()

    #     res = self.conv(x)

    #     torch.cuda.synchronize()
    #     t3 = time.perf_counter()

    #     print(f"forward pass: {t3-t2:.4f}s")
    #     print(f"type convert: {t2-t1:.4f}s")

    #     return res