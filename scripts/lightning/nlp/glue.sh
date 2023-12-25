#!/bin/bash

for seed in 42; do
  #  for balance in rr mean recursive; do
  # cola
  sbatch scripts/lightning/glue.job \
    --model.model_name_or_path google/t5-v1_1-small \
    --model.task_name cola \
    --seed $seed

  #  done
done
