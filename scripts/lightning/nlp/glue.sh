#!/bin/bash

for seed in 42; do
  #  for balance in rr mean recursive; do
  # for task in cola sst2 mrpc stsb qqp mnli qnli rte; do
  for task in mrpc; do
  # for task in cola; do
    # sbatch --gres=gpu:2080ti:4 \
    #   scripts/lightning/nlp/glue.job \
      # torchrun --nproc-per-node=4 \
    #   lightning_exp/nlp/glue.py \
    #   --model.model_name_or_path google/t5-v1_1-small \
    #   --model.task_name $task \
    #   --seed $seed \
    #   --data.train_batch_size 32 \
    #   --data.eval_batch_size 64 \
    #   -T 25000 \
    #   -vi 100

    # t5 small
    sbatch --gres=gpu:a5000:4 \
      scripts/lightning/nlp/glue.job \
      torchrun --nproc-per-node=4 \
      lightning_exp/nlp/glue.py \
      --model.model_name_or_path google/t5-v1_1-small \
      --model.task_name $task \
      --seed $seed \
      --data.train_batch_size 32 \
      --data.eval_batch_size 64 \
      -T 250000 \
      -vi 1000

    # small SAM
    # sbatch --gres=gpu:a6000:1 \
    #   scripts/lightning/nlp/glue.job \
    #   python lightning_exp/nlp/glue_sam.py \
    #   --model.model_name_or_path google/t5-v1_1-small \
    #   --model.task_name $task \
    #   --seed $seed \
    #   --data.train_batch_size 128 \
    #   --data.eval_batch_size 256 \
    #   -T 25000 \
    #   -vi 100 \
    #   --model.rho 0.05

    # t5 base
    # sbatch --gres=gpu:a6000:2 \
    #   scripts/lightning/nlp/glue.job \
    #   python lightning_exp/nlp/glue_sam.py \
    #   --model.model_name_or_path google/t5-v1_1-base \
    #   --model.task_name $task \
    #   --seed $seed \
    #   --data.train_batch_size 64 \
    #   --data.eval_batch_size 128 \
    #   -T 25000 \
    #   -vi 100

  done
done
