#!/usr/bin/env bash

for seed in 1 2 3; do
  for balance in rr; do
    for lr in 0.1 0.01 0.001 0.0001; do
      for wd in 0 0.01 0.0001; do
        sbatch scripts/slurm/fashion_mnist.job minnet $balance $seed \
          --learning_rate $lr \
          --weight_decay $wd \
          --per_device_train_batch_size 16 \
          --per_device_eval_batch_size 256 \
          --no_record_grads \
          --no_record_orders
      done
    done
  done
done
          # --order_path checkpoints/mnist/minnet/mean/mnist_minnet_mean_sgd_lr_0.01_wd_0.0001_b_64_seed_1_100_orders.pt
