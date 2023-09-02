#!/usr/bin/env bash

mkdir -p logs/mnist

for seed in 1 2 3; do
  for balance in ntk-fixed; do
    for lr in 0.1; do
      for wd in 0; do
        for centered_feature_map in '' '--centered_feature_map'; do
          sbatch scripts/slurm/mnist.job lr $balance $seed \
            --no_random_first_epoch \
            --optimizer sgd \
            --momentum 0 \
            --learning_rate $lr \
            --weight_decay $wd \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 1024 \
            --cpu_herding \
            $centered_feature_map
        done
      done
    done
  done
done
# --order_path checkpoints/mnist/minnet/mean/mnist_minnet_mean_sgd_lr_0.01_wd_0.0001_b_64_seed_1_100_orders.pt
