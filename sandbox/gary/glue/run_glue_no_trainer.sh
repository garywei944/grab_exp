#!/bin/bash

export TASK_NAME=mrpc

python run_glue_no_trainer.py \
  --model_name_or_path t5-small \
  --task_name $TASK_NAME \
  --learning_rate 1e-3 \
  --max_length 128 \
  --max_train_steps 250000 \
  --per_device_train_batch_size 128 \
  --output_dir /tmp/$TASK_NAME/
