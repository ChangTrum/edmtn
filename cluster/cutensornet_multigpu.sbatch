#!/usr/bin/env bash
# LEGACY RECIPE (Phase-C1 multi-GPU validation, kept for reference). Its `srun --mpi=pmi2` launch
# matched the cluster as configured in 2026-06 and is NOT currently working:
# the site's CUDA-aware MPI stack has since regressed. Use
# cluster/test_gpu_hpc.sbatch for the current test recipe + status record.
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
