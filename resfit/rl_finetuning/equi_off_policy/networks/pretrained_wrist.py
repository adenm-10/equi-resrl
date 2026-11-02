"""The wrist sweep's pretrained wrist encoders (equivariance.wrist_encoder=resnet18 or resnet18_fa; EXPERIMENTS.md,
2026-11-01 CODE, "the ResNet arms"; the practical guide's frame-averaged pretrained encoder,
docs/papers/practica_guide_for_sym_diffpo.pdf).

Phi is torchvision's ImageNet ResNet-18 (IMAGENET1K_V1, cached in ~/.cache/torch) less its classifier: 512 pooled
features. It is frozen (agent.freeze_encoder=true, which q_agent.check_families requires) and always in eval mode, so
its batch norms keep ImageNet's statistics; it runs without gradients, in bf16 on the GPU. The image arrives in
[-1, 1] (ResObsEnc) and is mapped to ImageNet's normalization.

resnet18_fa frame-averages it over C_N: Psi(x) = 1/N sum_k rho(k) Phi(R_k^-1 x). R_k turns the crop by 2 pi k / N
about its centre in torch.rot90's direction (R_{N/4} is rot90 once), after the crop is masked to its inscribed disk so
that no turn moves a corner in or out. rho(k) shifts each of the 512 outputs' 512 / N regular fields by k, as
tests/group_action.py's act_regular does. Then
Psi(R_k x) = rho(k) Psi(x): exactly at right angles, where the resampling is exact, and to the resampling's error at
the others (tests/test_wrist_sweep.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PretrainedWristEncoder(torch.nn.Module):
    n_out = 512

    def __init__(self, frame_average: bool, size: int = 76, N: int = 8):
        super().__init__()
        if frame_average and self.n_out % N:
            raise ValueError(f"frame averaging over C{N} reads the 512 features as regular fields of {N}")
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        self.net = net.eval()
        self.frame_average, self.N = frame_average, N
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        if frame_average:
            # R_k^-1's grid: affine_grid maps each output pixel to where it samples, so R_k^-1 x samples x at R_k p,
            # and [[c, -s], [s, c]] is a turn by +theta in rot90's direction (tests/test_wrist_sweep.py checks it).
            mats = []
            for k in range(N):
                c, s = math.cos(2 * math.pi * k / N), math.sin(2 * math.pi * k / N)
                mats.append([[c, s, 0.0], [-s, c, 0.0]])
            grids = F.affine_grid(torch.tensor(mats), [N, 3, size, size], align_corners=False)
            centre = (torch.arange(size) + 0.5 - size / 2) / (size / 2)
            disk = (centre.view(-1, 1) ** 2 + centre.view(1, -1) ** 2 <= 1.0).float().view(1, 1, size, size)
            self.register_buffer("grids", grids, persistent=False)
            self.register_buffer("disk", disk, persistent=False)

    def train(self, mode: bool = True):
        """The ResNet stays in eval mode whatever the agent's: its batch norms keep ImageNet's statistics."""
        super().train(mode)
        self.net.eval()
        return self

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        x = ((x + 1) / 2 - self.mean) / self.std
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            return self.net(x).float()

    def forward_map(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Its features, [B, 512], and no 9x9 map (the critic's window reads the plain wrist CNN's alone)."""
        if not self.frame_average:
            return self._phi(x), None
        B, N = x.shape[0], self.N
        x = x * self.disk
        turned = F.grid_sample(x.repeat(N, 1, 1, 1), self.grids.repeat_interleave(B, dim=0), mode="bilinear",
                               padding_mode="zeros", align_corners=False)  # [N*B]: R_0^-1 x for every sample, ...
        f = self._phi(turned).view(N, B, self.n_out // N, N)
        return torch.stack([f[k].roll(k, dims=-1) for k in range(N)]).mean(0).reshape(B, self.n_out), None
