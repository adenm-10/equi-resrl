import torch


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
        # self.norm1 = torch.nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size,
                                      padding=pad, bias=False)
        # self.norm2 = torch.nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.relu  = torch.nn.ReLU(inplace=True)

        self.skip = None
        if in_channels != out_channels or stride != 1:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1,
                                 stride=stride, bias=False),
                # torch.nn.GroupNorm(_gn_groups(out_channels), out_channels),
            )

    def forward(self, x):
        # identity = x if self.skip is None else self.skip(x)
        # out = self.relu(self.norm1(self.conv1(x)))
        # out = self.norm2(self.conv2(out))
        # out = out + identity
        # return self.relu(out)

        identity = x if self.skip is None else self.skip(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        return self.relu(out)


# ---------------------------------------------------------------------------
# In-hand ResNet encoder (no equivariance, GroupNorm throughout)
#
# Same spatial trace as the agentview encoder (76 → 72 → 36 → 18 → 9 → 3 → 1).
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
            # torch.nn.GroupNorm(_gn_groups(c1), c1),
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
        # self.out_ln = torch.nn.LayerNorm(n_out)
        self.out_ln = torch.nn.Identity()
        self.n_out = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)              # [B, n_out, 1, 1]
        h = h.flatten(1)              # [B, n_out]
        # return h
        return self.out_ln(h)         # apply LN on the flat feature