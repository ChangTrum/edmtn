#!/usr/bin/env bash
# Phase C1 — multi-GPU hpc solve validation, c1 / 4×A800 (one rank per GPU).
#   sbatch --parsable cluster/cutensornet_multigpu.sbatch
#SBATCH --job-name=cutn-mgpu
#SBATCH --partition=normal
#SBATCH --nodelist=c1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:25:00
#SBATCH --output=/home/user/cchen1/edmtn/cutensornet_multigpu_%j.out
set -uo pipefail

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate edmtn-gpu
cd /home/user/cchen1/edmtn

SP=/home/user/cchen1/.conda/envs/edmtn-gpu/lib/python3.11/site-packages
export CUTENSORNET_COMM_LIB="$SP/cuquantum/distributed_interfaces/libcutensornet_distributed_interface_mpi.so"
export LD_PRELOAD="/home/apps/mpich/lib/libmpi.so"   # MPICH+Python symbol-visibility fix

echo "host: $(hostname)"; nvidia-smi --query-gpu=index,name --format=csv,noheader
srun --mpi=pmi2 env PYTHONPATH=src python examples/track2/cutensornet_multigpu.py
