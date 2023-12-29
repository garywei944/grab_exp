#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

for seed in 42; do
  #  for balance in rr mean recursive; do
  # for task in cola sst2 mrpc stsb qqp mnli qnli rte; do
  # for task in cola sst2 stsb qqp mnli qnli rte; do
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
  # torchrun --nproc-per-node=4 \
  sbatch --gres=gpu:a5000:4 \
    scripts/lightning/nlp/glue.job \
    python lightning_exp/nlp/glue_t5.py \
    --model.model_name_or_path google/t5-v1_1-small \
    --model.learning_rate 1e-3 \
    --model.optimizer adafactor \
    --seed $seed \
    --data.train_batch_size 32 \
    -T 250000 \
    --model.warmup_steps 1000 \
    --data.max_length 512 \
    --data.pad_to_max_length false

  # small SAM
  # sbatch --gres=gpu:2080ti:8 \
  #   scripts/lightning/nlp/glue.job \
  #   python lightning_exp/nlp/glue_sam.py \
  #   --model.model_name_or_path google/t5-v1_1-small \
  #   --model.learning_rate 1e-3 \
  #   --model.optimizer adafactor \
  #   --seed $seed \
  #   --data.train_batch_size 16 \
  #   -T 250000 \
  #   --model.warmup_steps 1000 \
  #   --data.max_length 512 \
  #   --data.pad_to_max_length false \
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
# done
