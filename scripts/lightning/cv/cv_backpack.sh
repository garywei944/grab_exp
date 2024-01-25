#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

mkdir -p logs/sam

for seed in 0; do
  for norm in 'bn'; do
    for da in none; do
      for bt in mean; do
        for rho in 0.1; do
          sbatch --gres=gpu:a6000:1 -p desa \
            scripts/lightning/cv/cv_backpack.job \
            python experiments/lightning/cv/cv_backpack_sam.py \
            --seed $seed \
            --data.train_batch_size 128 \
            --data.data_augmentation $da \
            --model.model_name resnet \
            --model.norm $norm \
            -bt $bt \
            --model.rho $rho
          # -ckpt 'sam-cifar10/uo3wy3tt/checkpoints/epoch=73-step=28934.ckpt' \
          # --model.rho $rho
          # done
        done
      done
    done
  done
done
