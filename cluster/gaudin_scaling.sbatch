#!/usr/bin/env bash
# Gaudin recompute -- stage G1 (K=49 anchor + capacity measurement) on c1, single A800.
#   sbatch --parsable cluster/gaudin_scaling.sbatch
#
# Protocol = the archived (voided-line) obsmap settings, re-derived cleanly:
#   K=49, T=3, eps=0.1, order 2, cutoff 1e-8 (rel), max_bond 500,
#   direct + exact + quimb canon, f64, no preset, L0=2, tail_frac 0.4,
#   eta_floor 1e-6 (math eta fits only).
# Acceptance gates are PRE-REGISTERED in gaudin_scaling.py: fit gate (tail alpha in
# [0.97, 1.02], R2 >= 0.95, >= 8 tail points, on math eta_rms + physical dP) AND quality
# gate (no cap_hit, no non-finite values).  An --anchor run exits 3 on gate FAIL (also
# when a previously FAILed result is skip-reloaded) and 4 on a reuse mismatch; under
# `set -e` that STOPS this queue -- "out of gate -> stop and report", enforced by rc.
#
# STRICTLY SEQUENTIAL: one configuration at a time on one GPU -- no background '&'
# (user directive 2026-07-19; supersedes the voided observable.sbatch parallel matrix).
# Completed configurations are skipped on resubmit (per-config JSON marks completion).
# c1 note: a heavy CPU job (3x64 threads) shares the node -- request only 32 of the
# remaining CPUs and let SLURM place us; do NOT use --exclusive.
#SBATCH --job-name=gaudinG1
#SBATCH --partition=normal
#SBATCH --nodelist=c1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/cchen1/edmtn/gaudin_g1_%j.out
set -euo pipefail

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate edmtn-gpu
cd /home/user/cchen1/edmtn
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-n/a}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
git rev-parse HEAD

# One A800 on c1 is occupied by another user (user notice 2026-07-19).  SLURM's --gres
# should hand us a free card, but if that occupancy bypassed SLURM the busy card could be
# allocated to us -- probe the ALLOCATED card and refuse to contend rather than share it.
IDX="${CUDA_VISIBLE_DEVICES:-0}"; IDX="${IDX%%,*}"
IFS=', ' read -r MEM UTIL <<< "$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
  --format=csv,noheader,nounits -i "$IDX")"
echo "allocated GPU: index=$IDX mem_used=${MEM}MiB util=${UTIL}%"
if [ "$MEM" -gt 1024 ] 2>/dev/null || [ "$UTIL" -gt 10 ] 2>/dev/null; then
  echo "ERROR: allocated GPU $IDX appears BUSY (mem=${MEM}MiB, util=${UTIL}%) --" \
       "refusing to contend with the existing job; re-check c1 GPU occupancy" >&2
  exit 1
fi

RUN="python examples/research/gaudin_scaling.py"
COMMON="--K 49 --g 1.0 --T 3 --eps 0.1 --order 2 --cutoff 1e-8 --max-bond 500 \
  --method direct --decomp exact --canon quimb --device gpu \
  --L0 2 --tail-frac 0.4 --eta-floor 1e-6"

echo "=== GPU smoke (validate the CuPy streaming path) ==="
PYTHONPATH=src $RUN --smoke --device gpu --name g1_smoke_gpu --force

echo "=== G1 anchor arms (sequential) ==="
PYTHONPATH=src $RUN $COMMON --coupling linear                --anchor --name g1_linear
PYTHONPATH=src $RUN $COMMON --coupling exp    --beta 0.1     --anchor --name g1_exp_b0.1
for s in 0 1 2 3; do
  PYTHONPATH=src $RUN $COMMON --coupling random --seed "$s"  --anchor --name "g1_random_s$s"
done
echo "=== fine-eps arm (time-step sensitivity; the previously floor-affected case) ==="
PYTHONPATH=src $RUN $COMMON --coupling linear --eps 0.05     --anchor --name g1_linear_fine

echo "=== coarse vs fine eps comparison (report-only, no convergence verdict) ==="
PYTHONPATH=src $RUN --compare g1_linear,g1_linear_fine --name g1_eps_compare

echo "=== random: per-seed + pooled fits ==="
PYTHONPATH=src $RUN --pool g1_random_s0,g1_random_s1,g1_random_s2,g1_random_s3 \
  --name g1_random_pooled

echo "G1 complete"

# ---------------------------------------------------------------------------
# G2 -- scale-up.  STATUS UPDATE 2026-07-19: EXECUTED after user authorization, per the
# precondition protocol below -- capacity probe job 48319 (K=24 uncapped: natural Dmax
# FALLS with finer eps at fixed T: 643@60 -> 311@120 -> 222@240 sites), then production
# job 48320 (K=100/T=6/eps=0.05, max_bond=500 pure safety cap, NON-anchor; one-off
# sbatch at cluster ~/g2_production.sbatch).  Results + full story:
# docs/research/gaudin-scaling-study.md.  The original shelved-candidate record follows.
# Candidate (NOT pre-approved): K=100, T=6, eps=0.05, cutoff 1e-8, max_bond TBD.
# Preconditions before ANY G2 submission:
#   1. G1 complete and accepted; per-L SUMMARIES live in progress.jsonl + JSON per_L,
#      the FULL per-site bond profiles in the final NPZ (bonds_L*), and GPU peaks are
#      10 ms SAMPLED peaks -- keep a safety margin when projecting capacity.
#   2. A capacity PROBE toward the target grid (e.g. K=24 at T=6/eps=0.05 first),
#      projecting the next fold_raw tensor + compression workspace from the measured
#      per-site D_left/D_right, D_a=4 and complex128 -- NOT from Dmax alone.
#      Measured wall (docs/benchmarks/gpu-scaling-benchmark.md): K=24, eps=0.2,
#      xi=1e-8 uncapped -> T=3: Dmax 191 / 0.68 GB; T=6: Dmax 643 / 13.29 GB;
#      T=9: OOM (>80 GB). A fixed K=100/T=6/eps=0.05 CANNOT be assumed to fit one card.
#   3. Any max_bond change (e.g. a fallback to 800) is a numerical-protocol change:
#      re-check discarded weight, cap_hit and fit stability -- not just resources.
#   4. Explicit user authorization.
# ---------------------------------------------------------------------------
