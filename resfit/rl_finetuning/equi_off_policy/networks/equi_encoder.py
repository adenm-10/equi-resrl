import torch
from escnn import gspaces, nn

from resfit.rl_finetuning.equi_off_policy.common_utils.field_norm import FieldNorm


def _zero_field_norm_affine(fn: FieldNorm) -> None:
    """SkipInit: zero FieldNorm's affine scale and bias. Equivariance-safe."""
    with torch.no_grad():
        for p in fn.parameters():
            p.zero_()


# ---------------------------------------------------------------------------
# Stable EquiResBlock.
#
#   y = ReLU( skip(x) + FieldNorm2( R2Conv2( ReLU( FieldNorm1( R2Conv1(x) ) ) ) ) )
#
# At init, FieldNorm2's affine params are 0 → the residual branch outputs
# zero → every block is identity-via-skip at step 0. Equivariance preserved
# (zero is equivariant). Skip projection (when channels change) gets its own
# FieldNorm so both sides of the add have matched scale.
#
# All hidden field types are regular_repr, so FieldNorm is well-defined
# (regular_repr contains a trivial component but isn't all-trivial).
# ---------------------------------------------------------------------------

class EquiResBlock(torch.nn.Module):
    def __init__(
        self,
        group: gspaces.GSpace2D,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        initialize: bool = True,
    ):
        super().__init__()
        self.group = group
        rep = group.regular_repr
        ft_in  = nn.FieldType(group, in_channels * [rep])
        ft_hid = nn.FieldType(group, hidden_dim  * [rep])
        pad = (kernel_size - 1) // 2

        # Main branch: R2Conv (no bias) → FieldNorm → ReLU → R2Conv → FieldNorm
        # bias=False because FieldNorm absorbs the bias on the trivial subspace.
        self.conv1 = nn.R2Conv(ft_in, ft_hid, kernel_size=kernel_size,
                                padding=pad, stride=stride,
                                bias=False, initialize=initialize)
        self.norm1 = FieldNorm(ft_hid)
        self.conv2 = nn.R2Conv(ft_hid, ft_hid, kernel_size=kernel_size,
                                padding=pad,
                                bias=False, initialize=initialize)
        self.norm2 = FieldNorm(ft_hid)
        self.relu  = nn.ReLU(ft_hid, inplace=True)

        # Skip path. When channels/stride change, project + normalize so both
        # sides of the residual sum have matched per-sample scale. When the
        # types match, identity (no parameters, no scale change).
        if in_channels != hidden_dim or stride != 1:
            self.skip_conv = nn.R2Conv(ft_in, ft_hid, kernel_size=1,
                                        stride=stride, bias=False,
                                        initialize=initialize)
            self.skip_norm = FieldNorm(ft_hid)
        else:
            self.skip_conv = None
            self.skip_norm = None

        # SkipInit: zero norm2 so the residual branch contributes nothing at
        # init. norm1 and skip_norm keep their default unit-scale init.
        if initialize:
            _zero_field_norm_affine(self.norm2)

    def forward(self, xx: nn.GeometricTensor) -> nn.GeometricTensor:
        residual = xx
        out = self.relu(self.norm1(self.conv1(xx)))
        out = self.norm2(self.conv2(out))
        if self.skip_conv is not None:
            residual = self.skip_norm(self.skip_conv(residual))
        return self.relu(out + residual)


# ---------------------------------------------------------------------------
# EquivariantResEncoder76Cyclic — stable, always-on FieldNorm, tail-normed.
#
# Tail FieldNorm over the regular_repr c4 field at 1x1 spatial guarantees the
# encoder output is unit-scale by construction at every training step,
# decoupling downstream Linear layers from the encoder's internal dynamics.
# ---------------------------------------------------------------------------

class EquivariantResEncoder76Cyclic(torch.nn.Module):
    def __init__(
        self,
        obs_channel: int = 2,
        n_out: int = 128,
        initialize: bool = True,
        N: int = 8,
    ):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR2(N)

        ft_c1 = nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr])
        ft_c2 = nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr])
        ft_c3 = nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr])
        ft_c4 = nn.FieldType(self.group, n_out      * [self.group.regular_repr])
        ft_in = nn.FieldType(self.group, obs_channel * [self.group.trivial_repr])

        def _rb(in_ch, out_ch):
            return EquiResBlock(self.group, in_ch, out_ch,
                                initialize=initialize)

        self._stem = [
            nn.R2Conv(ft_in, ft_c1, kernel_size=5, padding=0,
                        bias=False, initialize=initialize),
            FieldNorm(ft_c1),
            nn.ReLU(ft_c1, inplace=True),
        ]

        self._tail = [
            nn.R2Conv(ft_c4, ft_c4, kernel_size=3, padding=0,
                        bias=False, initialize=initialize),
            FieldNorm(ft_c4),
            # nn.ReLU(ft_c4, inplace=True),
        ]

        self.conv = torch.nn.Sequential(
            
            # 76 → 72
            *self._stem,
            _rb(n_out // 8, n_out // 8),
            # _rb(n_out // 8, n_out // 8),
            nn.PointwiseMaxPool(ft_c1, 2),
            
            # 36
            _rb(n_out // 8, n_out // 4),
            # _rb(n_out // 4, n_out // 4),
            nn.PointwiseMaxPool(ft_c2, 2),
            
            # 18
            _rb(n_out // 4, n_out // 2),
            # _rb(n_out // 2, n_out // 2),
            nn.PointwiseMaxPool(ft_c3, 2),
            
            # 9
            _rb(n_out // 2, n_out),
            # _rb(n_out,      n_out),
            nn.PointwiseMaxPool(ft_c4, 3),
            
            # 3 → 1, with FieldNorm guaranteeing O(1) feature scale
            *self._tail,
        )

    def forward(self, x) -> nn.GeometricTensor:
        if isinstance(x, torch.Tensor):
            x = nn.GeometricTensor(
                x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr])
            )
        return self.conv(x)