#!/usr/bin/env bash
# HPC track (cuQuantum 2D contraction) GPU validation — c1 / single A800.
#   sbatch --parsable cluster/cutensornet_sanity.sbatch
# Prereq: env `edmtn-gpu` has `cuquantum-python-cu12` installed.
#SBATCH --job-name=cutn-val
#SBATCH --partition=normal
#SBATCH --nodelist=c1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/home/user/cchen1/edmtn/cutensornet_sanity_%j.out
set -euo pipefail

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate edmtn-gpu
cd /home/user/cchen1/edmtn

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

PYTHONPATH=src python examples/track2/cutensornet_sanity.py
