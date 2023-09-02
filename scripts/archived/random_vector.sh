#!/bin/bash

for seed in 1 2 3; do
    for balance in mean rr recursive; do
        python experiments/random_vector/random_vector.py \
        $balance 50000 62000 27.7 23 0.2 0.5 -t float32 -s $seed -d cpu \
        -b 1000 --wandb -e 15
        python experiments/random_vector/random_vector.py \
        $balance 50000 62000 27.7 23 0.2 0.5 -t float32 -s $seed -d cpu \
        -b 1000 --wandb --norm -e 15
        python experiments/random_vector/random_vector.py \
        $balance 50000 62000 27.7 23 0.2 0.5 -t float32 -s $seed -d cpu \
        -b 1000 --wandb --pi -e 15
    done
done
