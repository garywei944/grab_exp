#!/bin/bash
#SBATCH -J clm                           # Job name
#SBATCH -o logs/clm/%j.out               # output file (%j expands to jobID)
#SBATCH -e logs/clm/%j.err               # error log file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 8                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=64g                            # server memory requested (per node)
#SBATCH -t 12:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,desa                     # Request partition
#SBATCH --gres=gpu:a6000:1                # Type/number of GPUs needed

mkdir -p logs/clm

dataset=wikitext
dataset_config=wikitext-103-raw-v1
config=gpt2
seed=42
balance=mean
seq_len=128
output_dir=checkpoints/gpt2/$dataset/$balance/$seed

CMD="python experiments/nlp/clm/clm.py \
  --output_dir $output_dir \
  --dataset_name $dataset \
  --dataset_config_name $dataset_config \
  --config_name $config \
  --config_overrides n_embd=128,n_ctx=128,n_layer=4,n_head=2,n_positions=$seq_len,summary_first_dropout=0,attn_pdrop=0,resid_pdrop=0,embd_pdrop=0 \
  --block_size $seq_len \
  --tokenizer_name $config \
  --seed $seed \
  --balance_type $balance \
  --save_strategy steps \
  --save_steps 25000 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --adam_beta1 0.9 \
  --warmup_ratio 0.3 \
  --num_train_epochs 100 \
  --max_steps 400000 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 256 \
  --max_train_samples 320000 \
  --use_wandb"

echo $CMD
eval $CMD
