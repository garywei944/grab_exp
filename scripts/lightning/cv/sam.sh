#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

mkdir -p logs/sam

for seed in 0; do
  for norm in 'bn' 'gn'; do
    for da in basic none; do
      sbatch --gres=gpu:1 \
        scripts/lightning/cv/sam.job \
        python experiments/lightning/cv/cv.py \
        --seed $seed \
        --data.train_batch_size 128 \
        --data.data_augmentation $da \
        --model.model_name resnet \
        --model.norm $norm
      sbatch --gres=gpu:1 \
        scripts/lightning/cv/sam.job \
        python experiments/lightning/cv/cv_sam.py \
        --seed $seed \
        --data.train_batch_size 128 \
        --data.data_augmentation $da \
        --model.model_name resnet \
        --model.norm $norm \
        --model.rho 0.1
    done
  done
done
