#!/usr/bin/env bash

# best lr sgd 0.01 wd 0.01 momentum 0

DATASET=cifar10
mkdir -p logs/$DATASET

for seed in 1 2 3; do
  for lr in 0.01; do
    for wd in 0.01; do
      for balance in recursive; do
        for order in 2 3; do
          sbatch -J $DATASET scripts/slurm/cv/cv.job $DATASET lenet $balance $seed \
            --random_first_epoch 0 \
            --optimizer sgd \
            --momentum 0 \
            --learning_rate $lr \
            --weight_decay $wd \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 64 \
            --order_path lenet_cifar10_64_mean_epoch_0_rounds_30.pt \
            --record_grads 1 \
            --cpu_herding \
            --logging_first_step \
            --random_projection kron \
            --kron_order $order
        done
      done
    done
  done
done

# --cpu_herding
