import torch
import torch.nn as tnn
from torch.nn import Parameter, Module
from torch import distributions as pyd

import escnn.nn as enn
from escnn.nn.init import deltaorthonormal_init
from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.nn import GeometricTensor
from escnn.nn import FieldType

from collections import defaultdict
from abc import ABC, abstractmethod

from typing import Dict, Callable, List, Tuple, Any, Union
import numpy as np



class HLGaussLoss(tnn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float | None = None):
        super().__init__()
        if sigma is None:
            # Use the recommended sigma from the paper
            sigma = 0.75 * (max_value - min_value) / num_bins

        self.sigma = sigma
        # Create bin edges (num_bins + 1 points)
        bin_edges = torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32)
        self.register_buffer("bin_edges", bin_edges)

        # Bin centers for expectation computation
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.register_buffer("bin_centers", bin_centers)

    def _log1mexp(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log(1 - exp(-|x|)) in numerically stable way."""
        x = torch.abs(x)
        return torch.where(x < math.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

    def _log_sub_exp(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log(exp(max(x,y)) - exp(min(x,y))) in numerically stable way."""
        larger = torch.maximum(x, y)
        smaller = torch.minimum(x, y)
        return larger + self._log1mexp(torch.maximum(larger - smaller, torch.zeros_like(larger)))

    def _log_ndtr(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log(Φ(x)) where Φ is standard normal CDF, numerically stable."""
        # For x > 6, use asymptotic expansion
        # For x < -6, use log(Φ(x)) ≈ -x²/2 - log(√(2π)) - log(-x)
        # For -6 ≤ x ≤ 6, use log(0.5 * (1 + erf(x/√2)))

        ndtr_vals = torch.zeros_like(x)

        # For large positive x (> 6): log(Φ(x)) ≈ log(1 - Φ(-x)) ≈ log(1 - e^(-x²/2 - log(√(2π)) + log(-x)))
        large_pos = x > 6
        if torch.any(large_pos):
            ndtr_vals[large_pos] = torch.log(torch.special.ndtr(x[large_pos]))

        # For large negative x (< -6): log(Φ(x)) ≈ -x²/2 - log(√(2π)) - log(-x)
        large_neg = x < -6
        if torch.any(large_neg):
            x_neg = x[large_neg]
            ndtr_vals[large_neg] = -x_neg * x_neg / 2 - math.log(math.sqrt(2 * math.pi)) - torch.log(-x_neg)

        # For moderate x (-6 ≤ x ≤ 6): use direct computation
        moderate = (~large_pos) & (~large_neg)
        if torch.any(moderate):
            x_mod = x[moderate]
            # Use erfc for better precision when x is negative
            erfc_val = torch.special.erfc(-x_mod / math.sqrt(2))
            ndtr_vals[moderate] = torch.log(0.5 * erfc_val)

        return ndtr_vals

    def _normal_cdf_log_difference(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log(Φ(x) - Φ(y)) where Φ is standard normal CDF."""
        # When x >= y >= 0, compute log(Φ(-y) - Φ(-x)) for better precision
        is_y_positive = y >= 0
        x_hat = torch.where(is_y_positive, -y, x)
        y_hat = torch.where(is_y_positive, -x, y)

        log_ndtr_x = self._log_ndtr(x_hat)
        log_ndtr_y = self._log_ndtr(y_hat)

        return self._log_sub_exp(log_ndtr_x, log_ndtr_y)

    def _target_to_probs(self, target):
        target = target.unsqueeze(-1)  # [batch] -> [batch, 1]

        # Compute normalized distances to bin edges
        norm_factor = math.sqrt(2) * self.sigma
        upper_edges = (self.bin_edges[1:].unsqueeze(0) - target) / norm_factor
        lower_edges = (self.bin_edges[:-1].unsqueeze(0) - target) / norm_factor

        # Compute log probabilities for each bin
        bin_log_probs = self._normal_cdf_log_difference(upper_edges, lower_edges)

        # Compute normalization factor (log of total probability mass)
        log_z = self._normal_cdf_log_difference(
            (self.bin_edges[-1] - target) / norm_factor, (self.bin_edges[0] - target) / norm_factor
        )

        # Convert to probabilities
        probs = torch.exp(bin_log_probs - log_z.unsqueeze(-1))

        # Safety check: if anything goes wrong, fall back to uniform
        valid_mask = torch.isfinite(probs).all(dim=-1, keepdim=True)
        uniform_probs = torch.ones_like(probs) / probs.shape[-1]
        return torch.where(valid_mask, probs, uniform_probs)

    def forward(self, logits, target):
        tgt_probs = self._target_to_probs(target.detach())
        # Manual cross-entropy for soft labels: -sum(p * log_softmax(logits))
        log_probs = functional.log_softmax(logits, dim=-1)
        return -(tgt_probs * log_probs).sum(dim=-1).mean()

    def forward_batched(self, logits_batch, target):
        """
        Compute HLGaussLoss for multiple critic heads in a batched manner.
        This exactly replicates the original (buggy) behavior but in a vectorized way.

        Args:
            logits_batch: [num_heads, batch_size, num_bins] - logits from all critic heads
            target: [batch_size] - target values

        Returns:
            loss: scalar - average loss across all heads and batch
        """
        num_heads = logits_batch.shape[0]
        batch_size = target.shape[0]

        # Get the buggy target probabilities (same for all heads)
        tgt_probs_buggy = self._target_to_probs(target.detach())  # [batch_size, batch_size, num_bins]

        # Compute log softmax for all heads
        log_probs = functional.log_softmax(logits_batch, dim=-1)  # [num_heads, batch_size, num_bins]

        # Expand tgt_probs_buggy to all heads: [B, B, bins] -> [heads, B, B, bins]
        tgt_probs_expanded = tgt_probs_buggy.unsqueeze(0).expand(num_heads, -1, -1, -1)

        # Expand log_probs to match: [num_heads, batch_size, num_bins] -> [num_heads, batch_size, batch_size, num_bins]
        log_probs_expanded = log_probs.unsqueeze(2).expand(-1, -1, batch_size, -1)

        # Multiply: [num_heads, batch_size, batch_size, num_bins]
        product = tgt_probs_expanded * log_probs_expanded

        # Sum over bins: [num_heads, batch_size, batch_size]
        summed = product.sum(dim=-1)

        # Mean over all dimensions except heads, then mean over heads
        # This matches the original: -summed_h.mean() for each head, then average across heads
        losses_per_head = -summed.view(num_heads, -1).mean(dim=-1)  # [num_heads]

        return losses_per_head.mean()


class C51Loss(torch.nn.Module):
    def __init__(self, v_min: float, v_max: float, num_atoms: int):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

        # Create support for value distribution
        support = torch.linspace(v_min, v_max, num_atoms, dtype=torch.float32)
        self.register_buffer("support", support)

        # Delta z for projection
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    def project_distribution(
        self, next_distribution: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """Project Bellman update onto categorical support."""
        batch_size = rewards.size(0)

        # Compute target values for each atom: r + gamma * (1 - done) * support
        target_support = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)

        # Clamp target support to valid range
        target_support = torch.clamp(target_support, self.v_min, self.v_max)

        # Compute indices and interpolation weights for projection
        b = (target_support - self.v_min) / self.delta_z
        lower = b.floor().long()  # Lower bound indices
        upper = b.ceil().long()  # Upper bound indices

        # Handle edge cases
        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (self.num_atoms - 1)) * (lower == upper)] += 1

        # Project probabilities onto support
        target_distribution = torch.zeros_like(next_distribution)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, dtype=torch.long, device=target_distribution.device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
        )

        # Lower bound projection
        target_distribution.view(-1).index_add_(
            0, (lower + offset).view(-1), (next_distribution * (upper.float() - b)).view(-1)
        )
        # Upper bound projection
        target_distribution.view(-1).index_add_(
            0, (upper + offset).view(-1), (next_distribution * (b - lower.float())).view(-1)
        )

        return target_distribution

    def forward(self, current_logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        """Compute C51 loss using cross-entropy between current and target distributions."""
        # Convert logits to log probabilities
        current_log_probs = functional.log_softmax(current_logits, dim=-1)

        # Compute cross-entropy loss
        return -(target_distribution * current_log_probs).sum(dim=-1).mean()

    def logits_to_q_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert categorical logits to Q-value using expectation over support."""
        probs = functional.softmax(logits, dim=-1)
        return (probs * self.support).sum(dim=-1, keepdim=True)

    def forward_batched(self, current_logits_batch: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        """
        Compute C51 loss for multiple critic heads in a batched manner.

        Args:
            current_logits_batch: [num_heads, batch_size, num_atoms] - logits from all critic heads
            target_distribution: [batch_size, num_atoms] - target distribution (same for all heads)

        Returns:
            loss: scalar - average loss across all heads and batch
        """
        # Convert logits to log probabilities for all heads
        current_log_probs = functional.log_softmax(current_logits_batch, dim=-1)

        # Expand target distribution to match number of heads
        num_heads = current_logits_batch.shape[0]
        target_expanded = target_distribution.unsqueeze(0).expand(num_heads, -1, -1)

        # Compute cross-entropy loss for all heads at once
        losses = -(target_expanded * current_log_probs).sum(dim=-1)  # [num_heads, batch_size]
        return losses.mean()  # Average over all heads and batch



class EquivariantModule(Module, ABC):

    def __init__(self):
        r"""
        Abstract base class for all equivariant modules.
        
        An :class:`~EquivariantModule` is a subclass of :class:`torch.nn.Module`.
        It follows that any subclass of :class:`~EquivariantModule` needs to implement the
        :meth:`~escnn.nn.EquivariantModule.forward` method.
        With respect to a general :class:`torch.nn.Module`, an *equivariant module* implements a *typed* function as
        both its input and its output are associated with specific :class:`~escnn.nn.FieldType` s.
        Therefore, usually, the inputs and the outputs of an *equivariant module* are not just instances of
        :class:`torch.Tensor` but :class:`~escnn.nn.GeometricTensor` s.
        
        As a subclass of :class:`torch.nn.Module`, it supports most of the commonly used methods (e.g.
        :meth:`torch.nn.Module.to`, :meth:`torch.nn.Module.cuda`, :meth:`torch.nn.Module.train` or
        :meth:`torch.nn.Module.eval`)
        
        Many equivariant modules implement a :meth:`~escnn.nn.EquivariantModule.export` method which converts the module
        to *eval* mode and returns a pure PyTorch implementation of it.
        This can be used after training to efficiently deploy the model without, for instance, the overhead of the
        automatic type checking performed by all the modules in this library.
        
        .. warning ::
            
            Not all modules implement this feature yet.
            If the :meth:`~escnn.nn.EquivariantModule.export` method is called in a module which does not implement it
            yet, a :class:`NotImplementedError` is raised.
            Check the documentation of each individual module to understand if the method is implemented.
        
        Attributes:
            ~.in_type (FieldType): type of the :class:`~escnn.nn.GeometricTensor` expected as input
            ~.out_type (FieldType): type of the :class:`~escnn.nn.GeometricTensor` returned as output
        
        """
        super(EquivariantModule, self).__init__()
        
        # FieldType: type of the :class:`~escnn.nn.GeometricTensor` expected as input
        self.in_type = None
        
        # FieldType: type of the :class:`~escnn.nn.GeometricTensor` returned as output
        self.out_type = None

    @abstractmethod
    def forward(self, *input):
        pass
    
    @abstractmethod
    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        r"""
        Compute the shape the output tensor which would be generated by this module when a tensor with shape
        ``input_shape`` is provided as input.
        
        Args:
            input_shape (tuple): shape of the input tensor

        Returns:
            shape of the output tensor
            
        """
        pass

    
    def check_equivariance(self, atol: float = 1e-7, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        r"""
        
        Method that automatically tests the equivariance of the current module.
        The default implementation of this method relies on :meth:`escnn.nn.GeometricTensor.transform` and uses the
        the group elements in :attr:`~escnn.nn.FieldType.testing_elements`.
        
        This method can be overwritten for custom tests.
        
        Returns:
            a list containing containing for each testing element a pair with that element and the corresponding
            equivariance error
        
        """
    
        c = self.in_type.size
    
        x = torch.randn(3, c, *[10]*self.in_type.gspace.dimensionality)
    
        x = GeometricTensor(x, self.in_type)
        
        errors = []
    
        # for el in self.out_type.testing_elements:
        for _ in range(20):
            el = self.in_type.gspace.fibergroup.sample()
            print(el)
            
            out1 = self(x).transform(el).tensor.detach().numpy()
            out2 = self(x.transform(el)).tensor.detach().numpy()
        
            errs = out1 - out2
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert np.allclose(out1, out2, atol=atol, rtol=rtol), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'\
                    .format(el, errs.max(), errs.mean(), errs.var())
            
            errors.append((el, errs.mean()))
        
        return errors

    
    def export(self):
        r"""
        Export recursively each submodule to a normal PyTorch module and set to "eval" mode.
        
        .. warning ::
            
            Not all modules implement this feature yet.
            If the :meth:`~escnn.nn.EquivariantModule.export` method is called in a module which does not implement it
            yet, a :class:`NotImplementedError` is raised.
            Check the documentation of each individual module to understand if the method is implemented.
        
        .. warning ::
            Since most modules do not use the `coords` attribute of the input :class:`~escnn.nn.GeometricTensor`,
            once converted, they will only expect `tensor` but not `coords` in input.
            There is no standard behavior for modules that explicitly use `coords`, so check their specific
            documentation.
            
        """

        raise NotImplementedError(
            'Conversion of equivariant module {} into PyTorch module is not supported yet'.format(self.__class__)
        )


class FieldNorm(EquivariantModule):

    def __init__(self,
                 in_type: FieldType,
                 eps: float = 1e-05,
                 affine: bool = True,
                 ):
        r"""

        Normalization module which normalizes each field individually.
        The statistics are only computed over the channels within a single field (not over the batch dimension or
        the spatial dimensions).
        Moreover, this layer does not track running statistics and uses only the current input, so it behaves similarly
        at train and eval time.

        For each individual field, the mean is given by the projection on the subspaces transforming under the trivial
        representation while the variance is the squared norm of the field, after the mean has been subtracted.

        If ``affine = True``, a single scale is learnt per input field and the bias is applied only to the
        trivial irreps (this scale and bias are shared over the spatial dimensions in order to preserve equivariance).

        .. warning::
            If a field is only containing trivial irreps, this layer will just set its values to zero and, possibly,
            replace them with a learnable bias if ``affine = True``.


        Args:
            in_type (FieldType): the input field type
            eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
            affine (bool, optional): if ``True``, this module has learnable affine parameters. Default: ``True``

        """

        assert isinstance(in_type.gspace, GSpace)

        super(FieldNorm, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        self.affine = affine

        self._nfields = None

        # group fields by their type and
        #   - check if fields of the same type are contiguous
        #   - retrieve the indices of the fields

        # number of fields of each type
        self._nfields = defaultdict(int)

        # indices of the channels corresponding to fields belonging to each group
        _indices = defaultdict(lambda: [])

        # whether each group of fields is contiguous or not
        self._contiguous = {}

        ntrivials = 0
        position = 0
        last_field = None
        for i, r in enumerate(self.in_type.representations):

            for irr in r.irreps:
                if self.in_type.fibergroup.irrep(*irr).is_trivial():
                    ntrivials += 1

            if r.name != last_field:
                if not r.name in self._contiguous:
                    self._contiguous[r.name] = True
                else:
                    self._contiguous[r.name] = False

            last_field = r.name
            _indices[r.name] += list(range(position, position + r.size))
            self._nfields[r.name] += 1
            position += r.size

        for name, contiguous in self._contiguous.items():

            if contiguous:
                # for contiguous fields, only the first and last indices are kept
                _indices[name] = [min(_indices[name]), max(_indices[name]) + 1]
                setattr(self, f"{self._escape_name(name)}_indices", _indices[name])
            else:
                # otherwise, transform the list of indices into a tensor
                _indices[name] = torch.LongTensor(_indices[name])

                # register the indices tensors as parameters of this module
                self.register_buffer(f"{self._escape_name(name)}_indices", _indices[name])

        # store the size of each field type
        self._sizes = []

        # store for each field type the indices of the trivial irreps in it
        self._trivial_idxs = {}

        # store for each field type the sizes and the indices of all its irreps, grouped by their size
        self._irreps_sizes = {}

        self._has_trivial = {}

        # for each different representation in the input type
        for r in self.in_type._unique_representations:
            p = 0
            trivials = []

            # mask containing the location of the trivial irreps in the irrep decomposition of the representation
            S = np.zeros((r.size, r.size))

            # find all trivial irreps occurring in the representation
            for i, irr in enumerate(r.irreps):
                irr = self.in_type.fibergroup.irrep(*irr)
                if irr.is_trivial():
                    trivials.append(p)
                    S[p, p] = 1.

                p += irr.size

            name = r.name
            self._sizes.append((name, r.size))

            self._has_trivial[name] = len(trivials) > 0

            if self._has_trivial[name]:
                # averaging matrix which computes the expectation of a input vector, i.e. projects it in the trivial
                # subspace by masking out all non-trivial irreps
                P = r.change_of_basis @ S @ r.change_of_basis_inv
                self.register_buffer(f'{self._escape_name(name)}_avg', torch.tensor(P, dtype=torch.float))

                Q = torch.tensor(r.change_of_basis, dtype=torch.float)[:, trivials]
                self.register_buffer(f'{self._escape_name(name)}_change_of_basis', Q)

            if self.affine:
                # scale all dimensions of the same field by the same weight
                weight = Parameter(torch.ones((self._nfields[r.name], 1)), requires_grad=True)
                self.register_parameter(f'{self._escape_name(name)}_weight', weight)
                if self._has_trivial[name]:
                    # the bias is applied only to the trivial channels
                    bias = Parameter(torch.zeros((self._nfields[r.name], len(trivials))), requires_grad=True)
                    self.register_parameter(f'{self._escape_name(name)}_bias', bias)

        self.eps = eps

    def reset_parameters(self):
        if self.affine:
            for name, size in self._sizes:
                weight = getattr(self, f"{self._escape_name(name)}_weight")
                weight.data.fill_(1)
                if hasattr(self, f"{self._escape_name(name)}_bias"):
                    bias = getattr(self, f"{self._escape_name(name)}_bias")
                    bias.data.fill_(0)

    def reset_running_stats(self):
        pass

    def _estimate_stats(self, slice, name: str):

        if self._has_trivial[name]:
            P = getattr(self, f'{self._escape_name(name)}_avg')

            # compute the mean
            means = torch.einsum(
                'ij,bcj...->bci...',
                P,
                slice
                # slice.detach()
            )
            centered = slice - means
        else:
            means = None
            centered = slice

        # Center the data and compute the variance
        # N.B.: we implicitly assume the dimensions to be iid,
        # i.e. the covariance matrix is a scalar multiple of the identity
        # vars = centered.var(dim=2, unbiased=False, keepdim=True).detach()
        vars = (centered**2).mean(dim=2, keepdim=True)
        # vars = (centered**2).mean(dim=2, keepdim=True).detach()

        return means, vars

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        Normalize the input feature map

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]
        spatial_dims = input.shape[2:]

        output = torch.empty_like(input)

        # iterate through all field types
        for name, size in self._sizes:
            indices = getattr(self, f"{self._escape_name(name)}_indices")

            if self._contiguous[name]:
                slice = input[:, indices[0]:indices[1], ...]
            else:
                slice = input[:, indices, ...]

            slice = slice.view(b, -1, size, *spatial_dims)

            means, vars = self._estimate_stats(slice, name)

            if self._has_trivial[name]:
                # center data by subtracting the mean
                slice = slice - means

            # normalize dividing by the std and multiply by the new scale
            if self.affine:
                weight = getattr(self, f"{self._escape_name(name)}_weight").view(1, self._nfields[name], 1, *(1,)*len(spatial_dims))
            else:
                weight = 1.

            # compute the scalar multipliers needed
            scales = weight / (vars + self.eps).sqrt()
            # scales[vars < self.eps] = 0

            # print(name, size, indices, self._has_trivial[name])
            # print(slice.shape, scales.shape)
            # if not self.training:
            #     np.set_printoptions(precision=5, suppress=True, threshold=1000000, linewidth=3000)
            #     print(scales.detach().cpu().numpy().reshape(scales.shape[0], -1).T)

            # scale features
            slice = slice * scales

            # shift the features with the learnable bias
            if self.affine and self._has_trivial[name]:
                bias = getattr(self, f"{self._escape_name(name)}_bias")
                Q = getattr(self, f'{self._escape_name(name)}_change_of_basis')
                slice = slice + torch.einsum(
                    'ij,cj->ci',
                    Q,
                    bias
                ).view(1, bias.shape[0], Q.shape[0], *(1,) * len(spatial_dims))

            # needed for PyTorch's adaptive mixed precision
            slice = slice.to(output.dtype)

            if not self._contiguous[name]:
                output[:, indices, ...] = slice.view(b, -1, *spatial_dims)
            else:
                output[:, indices[0]:indices[1], ...] = slice.view(b, -1, *spatial_dims)

        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords)


    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) > 1, input_shape
        assert input_shape[1] == self.in_type.size, input_shape

        return (input_shape[0], self.out_type.size, *input_shape[2:])

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # return super(NormBatchNorm, self).check_equivariance(atol=atol, rtol=rtol)
        pass

    def _escape_name(self, name: str):
        return name.replace('.', '^')

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')

        main_str = self._get_name() + '('
        if len(extra_lines) == 1:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(extra_lines) + '\n'

        main_str += ')'
        return main_str

    def extra_repr(self):
        return '{in_type}, eps={eps}, affine={affine}' \
            .format(**self.__dict__)

    def export(self):
        raise NotImplementedError()

"""Equivariant initialization utilities.

Drop-in replacement for the non-equivariant `utils.orth_weight_init` /
`utils.apply_initialization_to_network` helpers, operating in the
equivariant basis-coefficient space instead of raw weight matrices.
"""


def apply_deltaortho_init(network, *, exclude_final_layer: bool = False):
    """Apply deltaorthonormal_init to every escnn.nn.Linear in *network*.

    Args:
        network: an nn.Module (or nn.Sequential) containing escnn.nn.Linear layers.
        exclude_final_layer: if True, skip the last escnn.nn.Linear found
                             (useful when the final layer gets its own scaling).
    """
    linears = [m for m in network.modules() if isinstance(m, enn.Linear)]

    if not linears:
        return

    targets = linears[:-1] if exclude_final_layer else linears

    for module in targets:
        deltaorthonormal_init(module.weights.data, module.basisexpansion)
        if module.bias is not None:
            module.bias.data.zero_()


def scale_final_equi_layer(network, scale: float):
    """Scale the weights (and bias) of the last escnn.nn.Linear in *network*."""
    for module in reversed(list(network.modules())):
        if isinstance(module, enn.Linear):
            with torch.no_grad():
                module.weights.mul_(scale)
                if module.bias is not None:
                    module.bias.mul_(scale)
            return

def clip_action_norm(action, max_norm):
    assert max_norm > 0
    assert action.dim() == 2 and action.size(1) == 7

    ee_action = action[:, :6]
    gripper_action = action[:, 6:]

    ee_action_norm = ee_action.norm(dim=1).unsqueeze(1)
    ee_action = ee_action / ee_action_norm
    assert (ee_action.norm(dim=1).min() - 1).abs() <= 1e-5
    scale = ee_action_norm.clamp(max=max_norm)
    ee_action = ee_action * scale
    action = torch.cat([ee_action, gripper_action], dim=1)
    return action  # noqa: RET504


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6, max_action_norm: float = -1):
        if isinstance(scale, float):
            scale = torch.ones_like(loc) * scale

        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.max_action_norm = max_action_norm

    def sample(self, clip=None, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size()
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = equi_clip(eps, self.action_layout, bound=clip)   # was: clamp(-clip, clip)
        x = self.loc + eps
        x = self._equi_clamp(x)                                     # was: _clamp via clamp(low, high)
        if self.max_action_norm > 0:
            x = equi_clip(x, self.action_layout, bound=self.max_action_norm)
        return x

    def _equi_clamp(self, x):
        """Equivariant straight-through clip to [low, high]."""
        assert self.low == -self.high, "equi_clamp assumes symmetric bounds"
        bound = self.high - self.eps
        clamped = equi_clip(x, self.action_layout, bound=bound)
        return x - x.detach() + clamped.detach()


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [float(g) for g in match.groups()]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
            return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

# ==========================
# For Normalizer
# ==========================

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

class DictOfTensorMixin(torch.nn.Module):
    def __init__(self, params_dict=None):
        super().__init__()
        if params_dict is None:
            params_dict = tnn.ParameterDict()
        self.params_dict = params_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = tnn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        def load_dict(state_dict, prefix):
            out_dict = tnn.ParameterDict()
            for key, value in state_dict.items():
                value: torch.Tensor
                if key.startswith(prefix):
                    param_keys = key[len(prefix):].split('.')[1:]
                    # if len(param_keys) == 0:
                    #     import pdb; pdb.set_trace()
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + 'params_dict')
        self.params_dict.requires_grad_(False)
        return 

def slice_gt(gt: enn.GeometricTensor, start: int, end: int,
             out_type: enn.FieldType) -> enn.GeometricTensor:
    """Slice channels [start:end] from a GeometricTensor and rewrap with out_type.

    Caller is responsible for ensuring the slice's representation list
    matches out_type — this is a thin view operation, not a check.
    """
    return enn.GeometricTensor(gt.tensor[:, start:end].contiguous(), out_type)


def cat_gts(gts: list[enn.GeometricTensor]) -> enn.GeometricTensor:
    """Concatenate GeometricTensors along the channel dim.

    Builds the output FieldType by concatenating the input rep lists in order.
    All inputs must share the same gspace.
    """
    group = gts[0].type.gspace
    reps = [r for gt in gts for r in gt.type.representations]
    out_type = enn.FieldType(group, reps)
    return enn.GeometricTensor(torch.cat([gt.tensor for gt in gts], dim=1), out_type)


def equi_clip(x: torch.Tensor, layout, bound: float = 1.0) -> torch.Tensor:
    """Per-irrep equivariant clip.

    For each (slice, kind) entry in layout:
      - "trivial": L_inf clamp to [-bound, +bound]    (per-component)
      - "irrep1":  L_2 projection onto disk of radius bound

    The L_2 projection on irrep(1) pairs commutes with the SO(2)/C_N action
    on those pairs, so the overall map is equivariant.

    Args:
        x: tensor of shape [..., action_dim].
        layout: list of (slice, str) tuples; str in {"trivial", "irrep1"}.
        bound: positive scalar bound (radius for irrep1, half-range for trivial).

    Returns:
        Tensor with the same shape as x.
    """
    assert bound > 0, f"bound must be positive, got {bound}"
    out = torch.empty_like(x)
    for sl, kind in layout:
        chunk = x[..., sl]
        if kind == "trivial":
            out[..., sl] = chunk.clamp(-bound, bound)
        elif kind == "irrep1":
            norm = chunk.norm(dim=-1, keepdim=True)
            scale = (bound / norm.clamp(min=1e-8)).clamp(max=1.0)
            out[..., sl] = chunk * scale
        else:
            raise ValueError(f"Unknown rep kind in layout: {kind!r}")
    return out






