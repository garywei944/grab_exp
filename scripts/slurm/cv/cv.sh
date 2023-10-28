#!/usr/bin/env bash

# MNIST best lr sgd 0.01 wd 0.01 momentum 0

DATASET=cifar10
mkdir -p logs/$DATASET

for seed in 1 2 3; do
  for lr in 0.01; do
    for wd in 0.01; do
      for balance in batch; do
        sbatch -J $DATASET scripts/slurm/cv/cv.job \
          -d $DATASET \
          -model lenet \
          -bt $balance \
          --seed $seed \
          --random_first_epoch 1 \
          --epochs 100 \
          -opt adam \
          -lr $lr \
          -wd $wd \
          -b 16 \
          -eb 64 \
          --report_grads 1 \
          --cpu_herding \
          --log_first_step
      done
    done
  done
done
