#!/bin/bash

mkdir -p logs/clm

dataset=wikitext
dataset_config=wikitext-103-raw-v1
config=gpt2
balance=mean
seq_len=128

for seed in 42; do
  for balance in mean rr; do
    for lr in 0.001; do
      for wd in 0.0001; do
#        for order in 2 3 4 5; do
          output_dir=checkpoints/gpt2/$dataset/$balance/$seed
          sbatch scripts/slurm/nlp/clm_sam.job \
            --output_dir $output_dir \
            --dataset_name $dataset \
            --dataset_config_name $dataset_config \
            --config_name $config \
            --config_overrides n_embd=128,n_ctx=128,n_layer=4,n_head=2,n_positions=$seq_len,summary_first_dropout=0,attn_pdrop=0,resid_pdrop=0,embd_pdrop=0 \
            --block_size $seq_len \
            --tokenizer_name $config \
            --seed $seed \
            --balance_type $balance \
            --random_first_epoch 1 \
            --save_strategy steps \
            --save_steps 50000 \
            --learning_rate $lr \
            --weight_decay $wd \
            --adam_beta1 0.9 \
            --warmup_steps 2000 \
            --num_train_epochs 200 \
            --max_steps 400000 \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 256 \
            --max_train_samples 320000 \
            --wandb 1
#            --random_projection kron
#            --kron_order $order
#        done
      done
    done
  done
done
