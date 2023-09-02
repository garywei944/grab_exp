#!/bin/bash

mkdir -p logs/mlm

for seed in 42; do
  for balance in rr; do
    # for lr in 0.0005 0.0002 0.0001 0.00005 0.00002; do
    for lr in 0.0005 0.0002 0.; do
      for wd in 0 0.01 0.001 0.0001; do
        #sbatch scripts/slurm/nlp/mlm.job Skylion007/openwebtext train roberta $balance $seed \
        sbatch scripts/slurm/nlp/mlm.job wikitext wikitext-2-raw-v1 roberta $balance $seed \
          --learning_rate $lr \
          --weight_decay $wd \
          --num_train_epochs 1 \
          --per_device_train_batch_size 1 \
          --per_device_eval_batch_size 1 \
          --gradient_accumulation_steps 128 \
          --balance_type rr
      done
    done
  done
done