#!/bin/bash
for seed in 1 2 3; do
  for balance in rr mean recursive; do
    # cola
    sbatch scripts/slurm/lora_glue.job cola roberta-base $balance $seed \
      --learning_rate 0.0004 --lora_r 8 --lora_alpha 16 --weight_decay 0.1 \
      --num_train_epochs 80
    # mnli
    sbatch scripts/slurm/lora_glue.job mnli roberta-base $balance $seed \
      --learning_rate 0.0005 --lora_r 8 --lora_alpha 16 --weight_decay 0.1 \
      --num_train_epochs 30
    # mrpc
    sbatch scripts/slurm/lora_glue.job mrpc roberta-base $balance $seed \
      --learning_rate 0.0005 --lora_r 8 --lora_alpha 16 --weight_decay 0.1 \
      --num_train_epochs 30
    # qnli
    sbatch scripts/slurm/lora_glue.job qnli roberta-base $balance $seed \
      --learning_rate 0.0005 --lora_r 8 --lora_alpha 16 --weight_decay 0.1 \
      --num_train_epochs 25
    # qqp
    sbatch scripts/slurm/lora_glue.job qqp roberta-base $balance $seed \
      --learning_rate 0.0005 --lora_r 8 --lora_alpha 16 --weight_decay 0.1 \
      --num_train_epochs 30
    # rte
    sbatch scripts/slurm/lora_glue.job rte roberta-base $balance $seed \
      --learning_rate 0.0005 --lora_r 8 --lora_alpha 16 --weight_decay 0.1 \
      --num_train_epochs 80
    # sst2
    sbatch scripts/slurm/lora_glue.job sst2 roberta-base $balance $seed \
      --learning_rate 0.0005 --lora_r 8 --lora_alpha 16 --weight_decay 0.1 \
      --num_train_epochs 60
    # stsb
    sbatch scripts/slurm/lora_glue.job stsb roberta-base $balance $seed \
      --learning_rate 0.0004 --lora_r 8 --lora_alpha 16 --weight_decay 0.1 \
      --num_train_epochs 40

  done
done
