#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

mkdir -p logs/sam

for seed in 0; do
  # sbatch --gres=gpu:1 \
  #   scripts/lightning/cv/sam.job \
  #   python experiments/lightning/cv/cv.py \
  #   --seed $seed \
  #   --data.train_batch_size 256 \
  #   --data.data_augmentation basic \
  #   --model.norm gn
  sbatch --gres=gpu:2080ti:4 \
    scripts/lightning/cv/sam.job \
    python experiments/lightning/cv/cv_sam.py \
    --seed $seed \
    --data.train_batch_size 64 \
    --data.data_augmentation none \
    --model.norm gn
  # sbatch --gres=gpu:a6000 \
  #   scripts/lightning/cv/sam.job \
  #   python experiments/lightning/cv/cv.py \
  #   --seed $seed \
  #   --data.train_batch_size 256 \
  #   --data.data_augmentation none \
  #   --model.norm gn
  # sbatch --gres=gpu:a6000 \
  #   scripts/lightning/cv/sam.job \
  #   python experiments/lightning/cv/cv_sam.py \
  #   --seed $seed \
  #   --data.train_batch_size 256 \
  #   --data.data_augmentation basic \
  #   --model.norm gn
  # sbatch --gres=gpu:a6000 \
  #   scripts/lightning/cv/sam.job \
  #   python experiments/lightning/cv/cv_sam.py \
  #   --seed $seed \
  #   --data.train_batch_size 256 \
  #   --data.data_augmentation none \
  #   --model.norm gn
done
# done
