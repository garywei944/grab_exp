#!/usr/bin/env bash

# MNIST best lr sgd 0.1 wd 0.01 momentum 0

# DATASET=mnist
# mkdir -p logs/$DATASET

# for seed in 1 2 3 4 5; do
#   for lr in 0.1; do
#     for wd in 0.0001; do
#       for balance in mean; do
#         sbatch -J $DATASET scripts/slurm/cv/cv_batch_grad.job \
#           -d $DATASET \
#           -model lr \
#           -bt $balance \
#           --seed $seed \
#           --random_first_epoch 1 \
#           --epochs 100 \
#           -opt sgd \
#           -lr $lr \
#           -wd $wd \
#           -m 0 \
#           -b 64 \
#           -eb 1024 \
#           --report_grads 0 \
#           --cpu_herding \
#           --log_first_step 0
#       done
#     done
#   done
# done

# # CIFAR10 best lr sgd 0.01 wd 0.01 momentum 0

DATASET=cifar10
mkdir -p logs/$DATASET

for seed in 1 2 3 4 5; do
  for lr in 0.01; do
    for wd in 0.01; do
      for balance in mean; do
        sbatch -J $DATASET scripts/slurm/cv/cv_batch_grad.job \
          -d $DATASET \
          -model lenet \
          -bt $balance \
          --seed $seed \
          --random_first_epoch 1 \
          --epochs 100 \
          -opt sgd \
          -lr $lr \
          -wd $wd \
          -m 0 \
          -b 64 \
          -eb 1024 \
          --report_grads 0 \
          --cpu_herding \
          --log_first_step 0
      done
    done
  done
done
