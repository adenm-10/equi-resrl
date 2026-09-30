"""Local fixes to escnn 1.0.11 that stop the equivariant update from stalling the CPU.

Neither fix changes a computed value: tests/test_sync_free.py checks that each one gives exactly
what escnn gives, and tests/test_regression_agent.py that the agent trains bit for bit as before.
What they change is how often the CPU has to wait for the GPU.

Some background. The CPU queues GPU work and moves on, so while the GPU computes one layer the
CPU can already be queueing the next. Anything that needs a GPU value on the CPU, such as
`.item()`, `.cpu()`, a copy from ordinary host memory, or a GPU scalar used as a slice bound,
makes the CPU stop and wait until the GPU has finished everything queued so far. Before
2026-09-30 one environment step's learning did that 487 times, 257 of them in the two places
fixed here, and spent roughly as much time waiting as computing. docs/ARCHITECTURE.md, "Why the
update used to stall", has the breakdown.

Importing this module applies fix 1 to the whole process. `equi_rl_utils` imports it, and the
encoder, actor and critic all import `equi_rl_utils`, so every escnn network built here has it.

1. Basis-expansion modules hash by identity.
   escnn gives `BlocksBasisExpansion` and `SingleBlockBasisExpansion`, which are
   `torch.nn.Module`s, a `__hash__` and `__eq__` that compare their sampled bases, with
   `allclose` and `.cpu()` copies. torch keeps a set of the modules it has seen whenever it walks
   a network: in `parameters()`, `children()`, `modules()`, `train()` and `eval()`. So every such
   walk compared bases. That cost 24 ms per `enc.parameters()`, 12 ms per critic soft update,
   and 21 ms for the train/eval switch around every `act()`. torch expects modules to hash by
   identity, and this restores that.

   Why nothing changes: escnn already makes layers with the same basis share one expansion
   instance (its `_stored_filters` cache is keyed on the basis, not on the module). So walking by
   identity visits the same modules as walking by value, except where two distinct instances
   compare equal. That happens only for buffers: `named_buffers()` now lists 28 index buffers it
   used to skip. `parameters()` and `state_dict()` are unchanged, in order, and nothing in this
   repo calls `buffers()` or `named_buffers()`.

2. `GroupPooling` reads its slice bounds as Python ints.
   escnn stores where each block of fields starts and ends as a LongTensor buffer and slices
   with its elements, `x[:, idx[0]:idx[1]]`. On the GPU each bound is a GPU scalar, so each
   slice made the CPU wait: 4 times per critic head per forward. The subclass below reads the
   bounds once, at construction, while the buffers are still on the CPU.
"""

from __future__ import annotations

import torch
from escnn import nn as enn
from escnn.nn import GeometricTensor
from escnn.nn.modules.basismanager.basisexpansion_blocks import BlocksBasisExpansion
from escnn.nn.modules.basismanager.basisexpansion_singleblock import SingleBlockBasisExpansion

# The escnn release these fixes were written against. They rely on its internals (the classes
# above, and GroupPooling's `_contiguous` and `in_indices_*` buffers), so a different release
# needs them re-checked: tests/test_sync_free.py fails until this is updated.
ESCNN_VERSION = "1.0.11"


def _hash_basis_expansions_by_identity() -> None:
    for cls in (BlocksBasisExpansion, SingleBlockBasisExpansion):
        cls.__hash__ = object.__hash__
        cls.__eq__ = object.__eq__


_hash_basis_expansions_by_identity()


class GroupPooling(enn.GroupPooling):
    """escnn's `GroupPooling`, with its slice bounds read once as Python ints.

    The pooling is escnn's: the max over each field's channels, which is what makes the critic's
    Q value invariant (docs/STANDARDS.md, rule 2.2). The buffers, and so the `state_dict`, are
    escnn's too, so checkpoints load either way.

    Only contiguous blocks of fields are supported. That is always the case for the critic,
    which pools `hidden_dim` regular fields, and escnn's other path, which indexes with a tensor,
    is not needed.
    """

    def __init__(self, in_type: enn.FieldType, **kwargs):
        super().__init__(in_type, **kwargs)
        if not all(self._contiguous.values()):
            raise ValueError(
                f"GroupPooling here needs each field size to sit in one contiguous block of "
                f"channels; {in_type} does not. Use escnn.nn.GroupPooling for that case."
            )
        # Per field size s: the input channels [in_start, in_end) holding those fields, and the
        # output channels [out_start, out_end) their maxima go to. The buffers are still on the
        # CPU here, so reading them costs nothing.
        self._bounds = {
            s: (*getattr(self, f"in_indices_{s}").tolist(), *getattr(self, f"out_indices_{s}").tolist())
            for s in self._contiguous
        }

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        assert input.type == self.in_type

        x = input.tensor
        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]
        output = torch.empty(self.evaluate_output_shape(x.shape), device=x.device, dtype=torch.float)

        for size, (in_start, in_end, out_start, out_end) in self._bounds.items():
            # [B, fields * size, ...] -> [B, fields, size, ...], then the max over each field.
            fields = x[:, in_start:in_end, ...].view(batch_size, -1, size, *spatial_shape)
            output[:, out_start:out_end, ...] = torch.max(fields, 2).values

        return GeometricTensor(output, self.out_type, input.coords)
