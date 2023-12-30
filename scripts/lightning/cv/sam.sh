#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

makedir -p logs/sam

for seed in 42; do
  sbatch --gres=gpu:4 \
    scripts/lightning/cv/sam.job \
    python lightning_exp/cv/cv.py \
    --seed $seed \
    --data.train_batch_size 64 \
    --data.data_augmentation none
done
# done
