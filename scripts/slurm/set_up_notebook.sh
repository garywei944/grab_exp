#!/bin/bash
#SBATCH -J jupyter                           # Job name
#SBATCH -o logs/jupyter_%j.out               # output file (%j expands to jobID)
#SBATCH -e logs/jupyter_%j.err               # error log file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 16                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=64g                            # server memory requested (per node)
#SBATCH -t 0-4:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,desa                 # Request partition
#SBATCH --gres=gpu:a6000:1                   # Type/number of GPUs needed

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)

# print tunneling instructions jupyter-log
echo -e "
For more info and how to connect from windows,
   see https://docs.ycrc.yale.edu/clusters-at-yale/guides/jupyter/

MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@g2-login.coecis.cornell.edu

Windows MobaXterm info
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: g2-login.coecis.cornell.edu
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
# uncomment the following two lines to use your conda environment called notebook_env
# module load miniconda
# source activate notebook_env

# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
jupyter-notebook --no-browser --port=${port} --ip=${node}
