import torch
from torch import nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with a sensible group count.

    Targets up to 32 groups, falls back to the largest divisor of
    ``num_channels`` if 32 doesn't divide. Works for any width including
    very thin stems (e.g. 16ch -> 16 groups, equivalent to InstanceNorm).
    """
    g = min(max_groups, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


def _kaiming_init_conv(conv: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def _init_norm(norm: nn.GroupNorm, *, zero_scale: bool = False) -> None:
    if zero_scale:
        nn.init.zeros_(norm.weight)
    else:
        nn.init.ones_(norm.weight)
    nn.init.zeros_(norm.bias)


# ---------------------------------------------------------------------------
# Stable residual block.
#
#   y = ReLU( skip(x) + GN2( Conv2( ReLU( GN1( Conv1(x) ) ) ) ) )
#
# At init, GN2's affine scale is 0, so the residual branch contributes nothing
# and y = ReLU(skip(x)). Every block is identity-via-skip at step 0; the
# network grows nonlinearity into its blocks as training progresses.
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.norm1 = _make_gn(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.norm2 = _make_gn(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                _make_gn(out_channels),
            )
        else:
            self.skip = nn.Identity()

        # Initialization
        _kaiming_init_conv(self.conv1)
        _kaiming_init_conv(self.conv2)
        _init_norm(self.norm1, zero_scale=False)
        _init_norm(self.norm2, zero_scale=True)        # SkipInit
        if isinstance(self.skip, nn.Sequential):
            _kaiming_init_conv(self.skip[0])
            _init_norm(self.skip[1], zero_scale=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + self.skip(x))


# ---------------------------------------------------------------------------
# ResEncoder76: stable residual conv encoder for 76x76 RGB inputs.
#
# Spatial pipeline (preserved from the original):
#   76 -> (5x5 stem, valid)            -> 72
#      -> (2x ResBlock + maxpool/2)    -> 36
#      -> (2x ResBlock + maxpool/2)    -> 18
#      -> (2x ResBlock + maxpool/2)    ->  9
#      -> (2x ResBlock + maxpool/3)    ->  3
#      -> (3x3 tail, valid)            ->  1
#
# Output: (B, n_out * N, 1, 1).  Tail GN ensures the per-sample feature
# vector is unit-scale by construction at every step of training.
# ---------------------------------------------------------------------------

class ResEncoder76(nn.Module):
    def __init__(self, obs_channel: int = 3, n_out: int = 128, N: int = 8):
        super().__init__()
        c1 = max((n_out // 8) * N, 1)
        c2 = max((n_out // 4) * N, 1)
        c3 = max((n_out // 2) * N, 1)
        c4 = n_out * N
        self.out_channels = c4

        # Stem: 76 -> 72
        self.stem = nn.Sequential(
            nn.Conv2d(obs_channel, c1, kernel_size=5, padding=0, bias=False),
            _make_gn(c1),
            nn.ReLU(inplace=True),
        )
        _kaiming_init_conv(self.stem[0])
        _init_norm(self.stem[1])

        self.stage1 = nn.Sequential(
            ResBlock(c1, c1),
            # ResBlock(c1, c1),
            nn.MaxPool2d(2),     # 72 -> 36
        )
        self.stage2 = nn.Sequential(
            ResBlock(c1, c2),
            # ResBlock(c2, c2),
            nn.MaxPool2d(2),     # 36 -> 18
        )
        self.stage3 = nn.Sequential(
            ResBlock(c2, c3),
            # ResBlock(c3, c3),
            nn.MaxPool2d(2),     # 18 -> 9
        )
        self.stage4 = nn.Sequential(
            ResBlock(c3, c4),
            # ResBlock(c4, c4),
            nn.MaxPool2d(3),     # 9 -> 3
        )

        # Tail: 3 -> 1.  GN over c4 channels at 1x1 spatial = LayerNorm on
        # the flattened feature vector. Downstream Linear sees an O(1) signal.
        self.tail = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=0, bias=False),
            _make_gn(c4),
            # nn.ReLU(inplace=True),
        )
        _kaiming_init_conv(self.tail[0])
        _init_norm(self.tail[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.tail(x)
        return x

def _gn_groups(channels: int, max_groups: int = 32) -> int:
    """Largest divisor of `channels` that is <= max_groups."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# Norm-equipped ResBlock
#   conv → GN → ReLU → conv → GN → + skip → ReLU
#   Skip projection (1x1 conv + GN) when channels change or stride != 1
#   Bias removed from convs since GN provides per-channel affine
# ---------------------------------------------------------------------------

class NormResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      padding=pad, stride=stride, bias=False)
        self.norm1 = torch.nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size,
                                      padding=pad, bias=False)
        self.norm2 = torch.nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.relu  = torch.nn.ReLU(inplace=True)

        self.skip = None
        if in_channels != out_channels or stride != 1:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1,
                                 stride=stride, bias=False),
                torch.nn.GroupNorm(_gn_groups(out_channels), out_channels),
            )

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + identity
        return self.relu(out)


# ---------------------------------------------------------------------------
# In-hand ResNet encoder (no equivariance, GroupNorm throughout)
#
# Same spatial trace as ResEncoder76 (76 → 72 → 36 → 18 → 9 → 3 → 1).
# Output is always [B, n_out, 1, 1].
# A final LayerNorm on the flattened feature stabilizes the
# encoder→head interface (what the working ViT baseline effectively had).
# ---------------------------------------------------------------------------

class ResEncoder76InHand(torch.nn.Module):
    def __init__(
        self,
        obs_channel: int = 3,
        channels: tuple = (16, 32, 64, 128),
        n_out: int = 128,
        blocks_per_stage: int = 1,
        out_layernorm: bool = True,
    ):
        super().__init__()
        assert len(channels) == 4, "channels must be (c1, c2, c3, c4)"
        c1, c2, c3, c4 = channels

        def stage(c_in, c_out, n_blocks, pool_k):
            layers = [NormResBlock(c_in, c_out)]
            for _ in range(n_blocks - 1):
                layers.append(NormResBlock(c_out, c_out))
            layers.append(torch.nn.MaxPool2d(pool_k))
            return layers

        seq = [
            torch.nn.Conv2d(obs_channel, c1, kernel_size=5, padding=0, bias=False),
            torch.nn.GroupNorm(_gn_groups(c1), c1),
            torch.nn.ReLU(inplace=True),
        ]
        seq += stage(c1, c1, blocks_per_stage, pool_k=2)   # 72 → 36
        seq += stage(c1, c2, blocks_per_stage, pool_k=2)   # 36 → 18
        seq += stage(c2, c3, blocks_per_stage, pool_k=2)   # 18 → 9
        seq += stage(c3, c4, blocks_per_stage, pool_k=3)   # 9  → 3
        seq += [
            torch.nn.Conv2d(c4, n_out, kernel_size=3, padding=0, bias=False),
            # torch.nn.GroupNorm(_gn_groups(n_out), n_out),
            # torch.nn.ReLU(inplace=True),
        ]
        self.conv = torch.nn.Sequential(*seq)
        self.out_ln = torch.nn.LayerNorm(n_out) if out_layernorm else torch.nn.Identity()
        self.n_out = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)              # [B, n_out, 1, 1]
        h = h.flatten(1)              # [B, n_out]
        return self.out_ln(h)         # apply LN on the flat feature