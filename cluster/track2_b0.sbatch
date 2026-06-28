#!/usr/bin/env bash
# Track 2 Phase B0 probe — exact/approximate contraction surface on c1 / single A800.
#   sbatch --parsable cluster/track2_b0.sbatch
#SBATCH --job-name=t2-b0
#SBATCH --partition=normal
#SBATCH --nodelist=c1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/home/user/cchen1/edmtn/track2_b0_%j.out
set -euo pipefail

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate edmtn-gpu
cd /home/user/cchen1/edmtn

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHONPATH=src python examples/track2_b0_probe.py
