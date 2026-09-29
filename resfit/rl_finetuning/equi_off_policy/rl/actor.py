import torch
from escnn import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import *


class Actor(torch.nn.Module):
    """The equivariant residual actor: pi(g.s) = g.pi(s).

    visual projection -> trunk projection -> one hidden layer -> Linear to the
    action representation, each hidden layer followed by `norm` and a ReLU.
    Every Linear maps between escnn field types, which is what makes the map
    equivariant; `norm` (HIDDEN_NORMS) and equi_tanh preserve it.
    """

    def __init__(self,
                 group,
                 vis_ih_type: nn.FieldType,
                 prop_type:   nn.FieldType,
                 action_type: nn.FieldType,
                 action_layout,
                 hidden_dim: int,
                 action_shape,
                 num_layers: int,
                 dropout: float,
                 norm: str,
                 use_orth_init: bool,
                 last_layer_scale,
                 action_scale: float,
                 use_visual_proj: bool,
                 residual_squash: bool,
                 residual_actor: bool = True):
        super().__init__()
        assert residual_actor, "Non-residual mode dropped — actor is residual-only"
        self.residual_actor = True
        self.group = group

        self.vis_ih_type   = vis_ih_type
        self.prop_type     = prop_type
        self.action_type   = action_type
        self.action_layout = action_layout

        self.dropout          = dropout
        self.use_orth_init    = use_orth_init
        self.last_layer_scale = last_layer_scale
        self.action_scale     = action_scale
        self.residual_squash  = residual_squash

        # ---- Visual projection; Identity feeds the encoder features straight to the trunk ----
        if use_visual_proj:
            self.vp_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
            self.visual_proj = torch.nn.Sequential(
                nn.Linear(vis_ih_type, self.vp_type, initialize=True),
                *hidden_norm(norm, self.vp_type),
                nn.ReLU(self.vp_type),
            )
        else:
            self.vp_type = vis_ih_type
            self.visual_proj = torch.nn.Identity()

        # ---- Trunk: (vp + prop + base_action) ----
        trunk_in_type        = self.vp_type + prop_type + action_type
        self.trunk_proj_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        self.input_proj = torch.nn.Sequential(
            nn.Linear(trunk_in_type, self.trunk_proj_type, initialize=True),
            *hidden_norm(norm, self.trunk_proj_type),
            nn.ReLU(self.trunk_proj_type),
        )

        # ---- Policy MLP (trunk_out + prop + base_action), one hidden layer ----
        # num_layers is not read: equivariance.num_actor_layers is dead (ARCHITECTURE.md).
        pol_in_type     = self.trunk_proj_type + prop_type + action_type
        pol_hidden_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        pol_out_type    = nn.FieldType(group, action_shape)
        self.policy = torch.nn.Sequential(
            nn.Linear(pol_in_type, pol_hidden_type, initialize=True),
            *hidden_norm(norm, pol_hidden_type),
            nn.ReLU(pol_hidden_type, inplace=False),
            nn.Linear(pol_hidden_type, pol_out_type, initialize=True),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        if self.use_orth_init:
            apply_deltaortho_init(self.visual_proj, exclude_final_layer=False)
            apply_deltaortho_init(self.input_proj,  exclude_final_layer=False)
            apply_deltaortho_init(self.policy,      exclude_final_layer=False)

        if self.last_layer_scale is not None:
            scale_final_equi_layer(self.policy, self.last_layer_scale)

    def forward(self, feat: nn.GeometricTensor, std=None):
        assert isinstance(feat, nn.GeometricTensor)

        n          = feat.tensor.shape[1]
        vis_ih_end = self.vis_ih_type.size
        prop_end   = vis_ih_end + self.prop_type.size

        vis_ih      = slice_gt(feat, 0,          vis_ih_end, self.vis_ih_type)
        prop        = slice_gt(feat, vis_ih_end, prop_end,   self.prop_type)
        base_action = slice_gt(feat, prop_end,   n,          self.action_type)

        v        = self.visual_proj(vis_ih)
        trunk_in = cat_gts([v, prop, base_action])

        z         = self.input_proj(trunk_in)
        policy_in = cat_gts([z, prop, base_action])

        mu: torch.Tensor = self.policy(policy_in).tensor
        if self.residual_squash:
            mu = equi_tanh(mu, self.action_layout)
        scaled_mu = mu * self.action_scale

        if std is None:
            return scaled_mu

        noise   = torch.randn_like(scaled_mu) * std
        noisy   = scaled_mu + noise
        clipped = equi_clip(noisy, self.action_layout, bound=1.0)
        return clipped
