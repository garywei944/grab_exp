#!/usr/bin/env bash

# MNIST best lr sgd 0.01 wd 0.01 momentum 0

DATASET=mnist
mkdir -p logs/$DATASET

for seed in 42; do
  for lr in 0.1 0.01 0.001; do
    for wd in 0.01 0.0001; do
      for balance in rr; do
        sbatch -J $DATASET scripts/slurm/cv/cv.job \
          -d $DATASET \
          -model lr \
          -bt $balance \
          --seed $seed \
          --random_first_epoch 1 \
          --epochs 100 \
          -opt sgd \
          -lr $lr \
          -m 0 \
          -wd $wd \
          -b 64 \
          -eb 1024 \
          --report_grads 0 \
          --cpu_herding \
          --log_first_step 0
      done
    done
  done
done

# # CIFAR10 best lr sgd 0.01 wd 0.01 momentum 0

# DATASET=cifar10
# mkdir -p logs/$DATASET

# for seed in 1 2 3 4 5; do
#   for lr in 0.0001; do
#     for wd in 0.01; do
#       for balance in mean rr so; do
#         sbatch -J $DATASET scripts/slurm/cv/cv.job \
#           -d $DATASET \
#           -model lenet \
#           -bt $balance \
#           --seed $seed \
#           --random_first_epoch 1 \
#           --epochs 100 \
#           -opt adamw \
#           -lr $lr \
#           -wd $wd \
#           -b 16 \
#           -eb 1024 \
#           --report_grads 0 \
#           --cpu_herding \
#           --log_first_step 1
#       done
#     done
#   done
# done
