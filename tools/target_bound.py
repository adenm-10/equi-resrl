"""The largest TD target a dataset's demos can give, through the trainer's own buffer and n-step transform.

    PYTHONPATH=$PWD python tools/target_bound.py local/robomimic-mh-square-rerendered [--n-step 3] [--gamma 0.99]

Pairs each episode's frames as `train_residual_td3.py`'s `_populate_offline_buffer` does (the
transition t -> t+1 carries frame t+1's `next.done` as its done and as its reward), adds them one
at a time to a `TensorDictPrioritizedReplayBuffer` with the trainer's `MultiStepTransform`, and reads
every stored transition back. The agent's target is R_n + gamma^k * nonterminal * Q', so with every
Q' at most 1 the target is at most 1 iff R_n <= 1 everywhere and R_n == 0 wherever the transition
bootstraps. That is the condition under which `agent.clip_q_target_to_reward_range` (a clamp to
[0, 1]) never cuts a correct target. Online the same holds by construction: the env's reward is
robosuite's sparse 0/1, and a success terminates the episode (`dexmg.py`).

Only `next.done`, `episode_index` and `frame_index` are read; observations are stood in for by
frame indices.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer

from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="LeRobot dataset name, e.g. local/robomimic-mh-square-rerendered")
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.dataset}")
    frames = pd.concat([
        pd.read_parquet(path, columns=["next.done", "episode_index", "frame_index"])
        for path in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    ]).sort_values(["episode_index", "frame_index"])

    rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=len(frames)), alpha=0.6, beta=0.4, eps=1e-6,
        priority_key="_priority", transform=MultiStepTransform(n_steps=args.n_step, gamma=args.gamma),
        batch_size=1,
    )
    previous: dict[int, int] = {}
    added = 0
    for episode, frame, done in zip(frames["episode_index"], frames["frame_index"], frames["next.done"]):
        if episode in previous:
            rb.add(TensorDict({
                "obs": TensorDict({"frame": torch.tensor(previous[episode])}, batch_size=[]),
                "action": torch.zeros(1),
                "next": TensorDict({
                    "obs": TensorDict({"frame": torch.tensor(int(frame))}, batch_size=[]),
                    "done": torch.tensor(bool(done)),
                    "reward": torch.tensor(float(done)),
                }, batch_size=[]),
                "_priority": torch.tensor(10.0),
            }, batch_size=[]).unsqueeze(0))
            added += 1
        previous[episode] = int(frame)

    stored = rb.storage[: len(rb)]
    reward = stored["next", "reward"].flatten()
    nonterminal = stored["nonterminal"].flatten()
    largest = (reward + stored["gamma"].flatten() * nonterminal.float()).max().item()
    done_per_episode = frames.groupby("episode_index")["next.done"].sum().value_counts().to_dict()
    print(f"{args.dataset}: {frames['episode_index'].nunique()} episodes, done frames per episode "
          f"{dict(sorted(done_per_episode.items()))}")
    print(f"  {added} transitions added, {len(rb)} stored (the transform holds back the last n-1)")
    print(f"  largest n-step reward {reward.max().item():.4f}; rewarded and bootstrapping: "
          f"{int(((reward > 0) & nonterminal).sum())}")
    print(f"  largest target with every Q' = 1: {largest:.4f}")
    print(f"  n-step rewards seen: {sorted({round(x, 4) for x in reward.tolist()})}")


if __name__ == "__main__":
    main()
