#!/usr/bin/env bash

mkdir -p logs/cifar10

for seed in 1 2 3; do
  for balance in ema; do
    for lr in 0.01; do
      for wd in 0.01; do
        sbatch scripts/slurm/cv/cifar10.job lenet $balance $seed \
          --random_first_epoch 0 \
          --record_grads 1 \
          --optimizer sgd \
          --momentum 0 \
          --learning_rate $lr \
          --weight_decay $wd \
          --per_device_train_batch_size 16 \
          --per_device_eval_batch_size 64 \
          --cpu_herding
      done
    done
  done
done
# --order_path checkpoints/mnist/minnet/mean/mnist_minnet_mean_sgd_lr_0.01_wd_0.0001_b_64_seed_1_100_orders.pt5
