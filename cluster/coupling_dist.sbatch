#!/usr/bin/env bash
# Coupling-distribution scaling study (Gaudin, Track 1 / CuPy fold) — c1 / single A800.
#   sbatch --parsable cluster/coupling_dist.sbatch
# Track 1 only (compressed fold): direct + cold rSVD, cutoff 1e-6 rel, max_bond 1024.
# The fold runs on the A800; the per-bond subspace diagnostic runs on host CPU.
#SBATCH --job-name=coupdist
#SBATCH --partition=normal
#SBATCH --nodelist=c1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/cchen1/edmtn/coupling_dist_%j.out
set -euo pipefail

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate edmtn-gpu
cd /home/user/cchen1/edmtn

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== GPU smoke (validate the CuPy fold path) ==="
PYTHONPATH=src python examples/studies/coupling_distributions.py --smoke --device gpu \
    --name coupling_smoke_gpu

echo "=== real run: K=49, T=3, eps=0.2, Track-1 CuPy ==="
PYTHONPATH=src python examples/studies/coupling_distributions.py --K 49 --T 3 --eps 0.2 \
    --seeds 4 --device gpu --name coupling_dist_K49_gpu
