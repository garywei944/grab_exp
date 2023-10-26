#!/bin/bash

cmd="python -u experiments/cv/cv.py \
  --dataset mnist \
  --model_name lr \
  --balance_type swr \
  --optimizer adam \
  --seed 42 \
  --random_first_epoch 1 \
  -e 10 \
  --output_dir checkpoints \
  --dataloader_num_workers 1 \
  --learning_rate 0.001 \
  --weight_decay 0.01 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 1024 \
  --resnet_depth 2 \
  --centered_feature_map \
  --cpu_herding \
  --random_projection none"
#  --record_grads 1 \

echo $cmd
eval $cmd

#  --no_record_grads \
#  --largest_eig \
#  --no_record_grads \
