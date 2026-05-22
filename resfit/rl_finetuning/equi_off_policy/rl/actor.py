import torch
from escnn import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.equi_off_policy.rl.equi_rl_utils import *


class Actor(torch.nn.Module):
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
                 use_norms: bool,
                 use_orth_init: bool,
                 last_layer_scale,
                 action_scale: float,
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
        self.use_norms        = use_norms
        self.use_orth_init    = use_orth_init
        self.last_layer_scale = last_layer_scale
        self.action_scale     = action_scale

        # delta-ortho overrides escnn's default init; otherwise let escnn init.
        # escnn_init = not use_orth_init
        escnn_init = True

        # ---- Visual projection ----
        self.vp_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        vp_layers = [nn.Linear(vis_ih_type, self.vp_type, initialize=escnn_init)]
        if use_norms:
            vp_layers.append(nn.IIDBatchNorm1d(self.vp_type, track_running_stats=True))
        vp_layers.append(nn.ReLU(self.vp_type))
        self.visual_proj = torch.nn.Sequential(*vp_layers)

        # ---- Trunk: (vp + prop + base_action) ----
        trunk_in_type        = self.vp_type + prop_type + action_type
        self.trunk_proj_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        ip_layers = [nn.Linear(trunk_in_type, self.trunk_proj_type, initialize=escnn_init)]
        if use_norms:
            ip_layers.append(nn.IIDBatchNorm1d(self.trunk_proj_type, track_running_stats=True))
        ip_layers.append(nn.ReLU(self.trunk_proj_type))
        self.input_proj = torch.nn.Sequential(*ip_layers)

        # ---- Policy MLP (trunk_out + prop + base_action) ----
        pol_in_type     = self.trunk_proj_type + prop_type + action_type
        pol_hidden_type = nn.FieldType(group, hidden_dim * [group.regular_repr])
        pol_out_type    = nn.FieldType(group, action_shape)
        current_in = pol_in_type
        
        layers = []
        layers.append(nn.Linear(current_in, pol_hidden_type, initialize=escnn_init))
        layers.append(nn.IIDBatchNorm1d(pol_hidden_type, track_running_stats=True))
        layers.append(nn.ReLU(pol_hidden_type, inplace=False))
        layers.append(nn.Linear(pol_hidden_type, pol_out_type, initialize=escnn_init))
        # layers.append(nn.NormNonLinearity(pol_out_type, function='squash', bias=False))

        # layers = []
        # for _ in range(num_layers):
        #     layers.append(nn.Linear(current_in, pol_hidden_type, initialize=escnn_init))
        #     if use_norms:
        #         layers.append(nn.IIDBatchNorm1d(pol_hidden_type, track_running_stats=True))
        #     if dropout > 0:
        #         layers.append(nn.FieldDropout(pol_hidden_type, p=dropout))
        #     layers.append(nn.ReLU(pol_hidden_type, inplace=True))
        #     current_in = pol_hidden_type
        # layers.append(nn.Linear(current_in, pol_out_type, initialize=escnn_init))
        
        self.policy = torch.nn.Sequential(*layers)
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
        scaled_mu = mu * self.action_scale

        if std is None:
            return scaled_mu

        noise   = torch.randn_like(scaled_mu) * std
        noisy   = scaled_mu + noise
        clipped = equi_clip(noisy, self.action_layout, bound=1.0)
        return clipped