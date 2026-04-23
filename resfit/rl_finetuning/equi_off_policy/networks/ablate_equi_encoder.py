import torch

# ---------------------------------------------------------------------------
# Non-equivariant ResBlock (mirrors EquiResBlock exactly)
#   conv3x3 → ReLU → conv3x3 → + skip → ReLU
#   1x1 skip projection when channels change or stride != 1
#   No batch norm (matches equivariant original)
# ---------------------------------------------------------------------------

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                            padding=pad, stride=stride),
            torch.nn.ReLU(inplace=True),
        )
        self.layer2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size,
                                       padding=pad)
        self.relu = torch.nn.ReLU(inplace=True)

        self.upscale = None
        if in_channels != out_channels or stride != 1:
            self.upscale = torch.nn.Conv2d(in_channels, out_channels, 1,
                                            stride=stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        return self.relu(out)


# ---------------------------------------------------------------------------
# Non-equivariant ResEncoder76 (mirrors EquivariantResEncoder76Cyclic)
#
# Channel mapping:  equivariant  n reps of C_N regular_repr  →  n*N raw channels
#   n_out//8 * N,  n_out//4 * N,  n_out//2 * N,  n_out * N
#   (default N=8, n_out=128 → 128, 256, 512, 1024)
# ---------------------------------------------------------------------------

class ResEncoder76(torch.nn.Module):
    def __init__(self, obs_channel: int = 3, n_out: int = 128, N: int = 8):
        super().__init__()
        self.obs_channel = obs_channel
        c1 = (n_out // 8) * N   # 128
        c2 = (n_out // 4) * N   # 256
        c3 = (n_out // 2) * N   # 512
        c4 = n_out * N          # 1024

        self.conv = torch.nn.Sequential(
            # 76x76 → 72x72
            torch.nn.Conv2d(obs_channel, c1, kernel_size=5, padding=0),
            torch.nn.ReLU(inplace=True),
            ResBlock(c1, c1),
            ResBlock(c1, c1),
            torch.nn.MaxPool2d(2),
            # 36x36
            ResBlock(c1, c2),
            ResBlock(c2, c2),
            torch.nn.MaxPool2d(2),
            # 18x18
            ResBlock(c2, c3),
            ResBlock(c3, c3),
            torch.nn.MaxPool2d(2),
            # 9x9
            ResBlock(c3, c4),
            ResBlock(c4, c4),
            torch.nn.MaxPool2d(3),
            # 3x3
            torch.nn.Conv2d(c4, c4, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            # 1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
