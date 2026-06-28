#!/usr/bin/env bash
# Track 2 (HPC) Phase A0/A1 — cuTensorNet install/interop + 2D assembler GPU sanity.
# Submit from the cluster login node (shared home), c1 / single A800:
#   sbatch --parsable cluster/track2_sanity.sbatch
# Prereq: env `edmtn-gpu` has `cuquantum-python-cu12` installed (done in the A0 setup).
#SBATCH --job-name=t2-sanity
#SBATCH --partition=normal
#SBATCH --nodelist=c1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/user/cchen1/edmtn/track2_sanity_%j.out
set -euo pipefail

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate edmtn-gpu
cd /home/user/cchen1/edmtn

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

PYTHONPATH=src python examples/track2_cutensornet_sanity.py
