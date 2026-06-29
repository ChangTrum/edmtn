#!/usr/bin/env bash
# Phase C0 — cuTensorNet multi-GPU (MPI) distributed contraction de-risk, c1 / 4×A800.
#   sbatch --parsable cluster/cutensornet_mpi.sbatch
# Prereq: env `edmtn-gpu` has mpi4py + the built MPI wrapper
#   (cuquantum/distributed_interfaces/libcutensornet_distributed_interface_mpi.so).
#SBATCH --job-name=cutn-mpi
#SBATCH --partition=normal
#SBATCH --nodelist=c1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --output=/home/user/cchen1/edmtn/cutensornet_mpi_%j.out
set -uo pipefail

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate edmtn-gpu
cd /home/user/cchen1/edmtn

SP=/home/user/cchen1/.conda/envs/edmtn-gpu/lib/python3.11/site-packages
export CUTENSORNET_COMM_LIB="$SP/cuquantum/distributed_interfaces/libcutensornet_distributed_interface_mpi.so"
# MPICH + Python: preload the MPI lib so its symbols are in the global scope
# (else distributed_reset_configuration can fail; see cuQuantum issue #31)
export LD_PRELOAD="/home/apps/mpich/lib/libmpi.so"

echo "host: $(hostname)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "MPICH CUDA-aware? (look for cuda/ucx):"; mpichversion 2>/dev/null | grep -iE 'cuda|ucx|device' || echo "  (mpichversion gave nothing)"
echo "COMM_LIB=$CUTENSORNET_COMM_LIB"; [ -f "$CUTENSORNET_COMM_LIB" ] && echo "  wrapper present" || echo "  wrapper MISSING"

srun --mpi=pmi2 python examples/track2/cutensornet_mpi_probe.py
