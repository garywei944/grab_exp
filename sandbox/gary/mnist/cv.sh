#!/bin/bash

cmd="python -u experiments/cv/cv.py \
  --dataset cifar10 \
  --model_name lenet \
  --balance_type mean \
  --optimizer sgd \
  --seed 42 \
  --num_train_epochs 10 \
  --output_dir checkpoints \
  --dataloader_num_workers 1 \
  --no_random_first_epoch \
  --learning_rate 0.01 \
  --weight_decay 0.01 \
  --momentum 0 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 1024 \
  --resnet_depth 2 \
  --centered_feature_map \
  --cpu_herding \
  --random_projection kron"
#  --record_grads 1 \

echo $cmd
eval $cmd

#  --no_record_grads \
#  --largest_eig \
#  --no_record_grads \
