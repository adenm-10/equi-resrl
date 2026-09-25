#!/usr/bin/env bash

# Launch ACT BC training for the two-arm BoxCleanup task from DexMimicGen.
# ACT to match the ResFiT authors' own launchers for every simulation task.
# Dataset: ankile/dexmg-two-arm-box-cleanup

# Two deviations from the older launchers: save_freq 10000, because every save
# writes a 341 MB checkpoint dir and a new wandb artifact that are never cleaned
# up; and num_workers 8, because the dataloader was 98% of step time at default.

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-box-cleanup \
    --policy act \
    --batch_size 256 \
    --num_workers 8 \
    --wandb_project dexmg-boxcleanup-bc \
    --wandb_enable \
    --eval_env TwoArmBoxCleanup \
    --rollout_freq 1000 \
    --steps 50000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 10000
