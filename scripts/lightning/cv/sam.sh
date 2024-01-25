#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

mkdir -p logs/sam

for seed in 1 2; do
  for norm in 'bn'; do
    for da in basic; do
      for rho in 0.1; do
        # sbatch --gres=gpu:1 \
        #   scripts/lightning/cv/sam.job \
        #   python experiments/lightning/cv/cv.py \
        #   --seed $seed \
        #   --data.train_batch_size 128 \
        #   --data.data_augmentation $da \
        #   --model.model_name resnet \
        #   --model.norm $norm
        sbatch --gres=gpu:2080ti:4 \
          scripts/lightning/cv/sam.job \
          python experiments/lightning/cv/cv_sam.py \
          --seed $seed \
          --data.train_batch_size 32 \
          --data.data_augmentation $da \
          --model.model_name resnet \
          --model.norm $norm \
          --model.rho $rho
      done
    done
  done
done
