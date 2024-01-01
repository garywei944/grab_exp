#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

mkdir -p logs/sam

for seed in 0; do
  # sbatch --gres=gpu:4 \
  #   scripts/lightning/cv/sam.job \
  #   python lightning_exp/cv/cv.py \
  #   --seed $seed \
  #   --data.train_batch_size 64 \
  #   --data.data_augmentation basic
  # sbatch --gres=gpu:4 \
  #   scripts/lightning/cv/sam.job \
  #   python lightning_exp/cv/cv.py \
  #   --seed $seed \
  #   --data.train_batch_size 64 \
  #   --data.data_augmentation none
  sbatch --gres=gpu:4 \
    scripts/lightning/cv/sam.job \
    python lightning_exp/cv/cv_sam.py \
    --seed $seed \
    --data.train_batch_size 64 \
    --data.data_augmentation basic
  # sbatch --gres=gpu:4 \
  #   scripts/lightning/cv/sam.job \
  #   python lightning_exp/cv/cv_sam.py \
  #   --seed $seed \
  #   --data.train_batch_size 64 \
  #   --data.data_augmentation none
done
# done
