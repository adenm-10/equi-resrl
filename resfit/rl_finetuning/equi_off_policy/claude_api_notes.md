Let me trace every place `agent` is touched in the residual TD3 script to give you the complete API contract.

## Required Interface

**Constructor:**
```python
agent = QAgent(
    obs_shape=(img_c, img_h, img_w),    # e.g. (3, 84, 84)
    prop_shape=(lowdim_dim,),            # proprioceptive state dim
    action_dim=action_dim,               # action space dimension
    rl_cameras=image_keys,               # list of camera key strings
    cfg=cfg.agent,                       # agent config dataclass
    residual_actor=True,                 # enables residual mode
)
```

**Action selection:**
```python
with torch.no_grad(), utils.eval_mode(agent):
    action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)
```

Where `obs` is a dict with keys like `{"observation.state": Tensor, "observation.base_action": Tensor, "<camera_key>": Tensor}`. All tensors are on GPU with batch dimension `(num_envs, ...)`. Images are float (converted from uint8 by the replay buffer or environment). The method must return a `(num_envs, action_dim)` tensor — the **residual** action. `eval_mode=True` is used during evaluation (no exploration noise). `cpu=False` means return on GPU.

**Update step:**
```python
metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)
```

Where `batch` is a `TensorDict` on GPU with this structure:

```
batch: TensorDict, shape (batch_size,)
├── "obs": TensorDict
│   ├── "observation.state": (B, lowdim_dim)
│   ├── "observation.base_action": (B, action_dim)
│   └── "<camera_key>": (B, C, H, W)  # uint8 in buffer, but .to(device) may keep uint8
├── "action": (B, action_dim)          # the combined (base+residual) action that was executed
├── "next": TensorDict
│   ├── "obs": TensorDict (same structure as obs)
│   ├── "done": (B,) bool
│   └── "reward": (B,) float
├── "nonterminal": (B,) bool           # added by MultiStepTransform
└── "_priority": (B,) float            # only if using PER
```

The returned `metrics` dict must include at minimum:

```python
{
    "train/critic_loss": float,
    "train/critic_qt": float,          # mean target Q value
    # When update_actor=True:
    "train/actor_loss_base": float,
    "train/actor_grad_norm": float,     # optional but logged if present
    "train/actor_l2_penalty": float,    # optional but logged if present
    # Internal data (prefixed with _ so they're filtered from wandb):
    "_td_errors": Tensor (B,),          # required if using PER
    "_actions": Tensor (B, action_dim), # used for residual magnitude histograms
    "_target_q": Tensor (B,),           # used for Q-value histograms
}
```

**Optimizer access (for LR warmup):**
```python
agent.actor_opt.param_groups[0]["lr"]   # read and written to directly
```

The training loop manually sets the LR on `agent.actor_opt` during warmup, so this optimizer must exist as an attribute.

**Compatibility with `utils.eval_mode`:**
```python
with utils.eval_mode(agent):
    ...
```

This is likely a context manager that calls `.eval()` on internal modules and restores `.train()` on exit. Your agent needs to work with this pattern — typically means having standard `nn.Module` submodules that respond to `.train()` / `.eval()`.

## What You Can Ignore

The training script doesn't call `.to(device)` on the agent after construction — `QAgent` presumably handles device placement internally in `__init__`. It also never calls `.save()` or `.load()` on the agent (unlike the RLPD script which uses `save_checkpoint`). There's no `step_lr_schedulers()` call either — LR is managed externally.

## Key Design Consideration for Equivariance

The `observation.base_action` in the obs dict is the base policy's proposed action. In your equivariant agent, you need to decide how this transforms under your group. Since it's a delta action (as we discussed), the xy components should transform under the same representation as `action` under Cₙ, while z/θ/gripper components are invariant. Your equivariant encoder and actor need to handle the fact that `observation.state` and `observation.base_action` live in different representation spaces — proprioceptive state may have mixed equivariant/invariant components, while base_action has the same decomposition as the output action.