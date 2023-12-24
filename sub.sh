#!/bin/bash
#SBATCH -o logs/%j.out               # output file (%j expands to jobID)
#SBATCH -e logs/%j.err               # error log file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node 2                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=64g                            # server memory requested (per node)
#SBATCH -t 08:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,desa                     # Request partition
#SBATCH --gres=gpu:2080ti:2                  # Type/number of GPUs needed

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
srun python lightning_exp/nlp/glue.py
