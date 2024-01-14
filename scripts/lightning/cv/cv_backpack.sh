#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

mkdir -p logs/sam

for seed in 0; do
  for norm in 'bn'; do
    for da in basic; do
      # for rho in 0.2 0.3; do
      # sbatch --gres=gpu:1 \
      #   scripts/lightning/cv/sam.job \
      #   python experiments/lightning/cv/cv.py \
      #   --seed $seed \
      #   --data.train_batch_size 128 \
      #   --data.data_augmentation $da \
      #   --model.model_name resnet \
      #   --model.norm $norm
      sbatch --gres=gpu:a6000:1 \
        scripts/lightning/cv/cv_backpack.job \
        python experiments/lightning/cv/cv_backpack.py \
        --seed $seed \
        --data.train_batch_size 128 \
        --data.data_augmentation $da \
        --model.model_name resnet \
        --model.norm $norm \
        -ckpt 'sam-cifar10/uo3wy3tt/checkpoints/epoch=73-step=28934.ckpt' \
        -bt 'pair'
      # --model.rho $rho
      # done
    done
  done
done
