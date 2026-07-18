#!/usr/bin/env bash
# P1-14 — real GPU / HPC test execution on c1 / 4×A800 (one rank per GPU).
#   sbatch --parsable cluster/test_gpu_hpc.sbatch
#
# The --require-* flags turn "hardware/stack missing" into a NON-ZERO exit instead of an
# all-skipped false pass (see tests/conftest.py). Three separate pytest invocations run in this
# ONE batch process; the single-GPU sections use device 0 of the 4-GPU allocation, and the
# multigpu section's pytest LAUNCHES the worker srun itself (it does not do multi-GPU inline).
# Split the single-GPU sections into their own 1-GPU sbatch if you prefer a tighter allocation.
#SBATCH --job-name=edmtn-gpu-hpc
#SBATCH --partition=normal
#SBATCH --nodelist=c1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH --output=/home/user/cchen1/edmtn/test_gpu_hpc_%j.out
# fail fast on env setup (conda/cd/paths/nvidia-smi); the pytest steps use `|| rc=1` so a test
# failure is collected rather than aborting the run.
set -euo pipefail

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate edmtn-gpu
cd /home/user/cchen1/edmtn
export PYTHONPATH=src

SP=/home/user/cchen1/.conda/envs/edmtn-gpu/lib/python3.11/site-packages
# The MPICH at /home/apps/mpich has NO launcher (only compiler wrappers -- no mpiexec/mpirun/Hydra)
# and SLURM's srun/pmi2 <-> MPICH PMI regressed (PMI_KVS_Commit -1). The only working MPI stack is
# the SYSTEM OpenMPI (/usr/bin/mpirun + /usr/bin/mpicc), so run the whole multi-GPU worker on it:
# pin mpi4py to its OpenMPI ABI, drop the MPICH LD_PRELOAD, and use an OpenMPI-built comm wrapper.
export MPI4PY_MPIABI=openmpi
OMPI_COMMLIB="$SP/cuquantum/distributed_interfaces/libcutensornet_distributed_interface_openmpi.so"
if [ ! -f "$OMPI_COMMLIB" ]; then
  /usr/bin/mpicc -shared -std=c99 -fPIC -I/usr/local/cuda/include -I"$SP/cuquantum/include" \
    "$SP/cuquantum/distributed_interfaces/cutensornet_distributed_interface_mpi.c" \
    -o "$OMPI_COMMLIB" && echo "built OpenMPI comm-lib" || echo "comm-lib build FAILED (rc=$?)"
fi
export CUTENSORNET_COMM_LIB="$OMPI_COMMLIB"

echo "host: $(hostname)"; nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "-- MPI: OpenMPI (MPI4PY_MPIABI=$MPI4PY_MPIABI, comm-lib=$(basename "$CUTENSORNET_COMM_LIB")) --"

rc=0
# 1) Track 1 single-GPU CPU/CuPy parity (spin-boson + Gaudin). A single-task srun step so the
#    pytest process sees SLURM_NTASKS=1 and the hpc code never enters the size=1 distributed
#    branch (a bare `python` in this --ntasks=4 batch shell would inherit SLURM_NTASKS=4).
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
  python -m pytest -q -m "gpu and not cuquantum and not multigpu" --require-gpu || rc=1
# 2) Track 2 single-GPU cuTensorNet vs NumPy exact (single-task srun -> true non-distributed path)
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
  python -m pytest -q -m "cuquantum and not multigpu" --require-gpu --require-cuquantum || rc=1
# 3) Track 2 multi-GPU: OpenMPI's mpirun launches the 4-rank worker with OpenMPI's OWN PMIx,
#    bypassing SLURM's regressed srun/pmi2. Single node c1, 4 ranks, each picks GPU rank%4
#    (distinct). The worker writes JSON; pytest then validates it.
# OpenMPI here is built with CUDA (MPI ext 'cuda') + UCX PML, but the default run used the
# non-CUDA-aware tuned collective and segfaulted doing MPI_Allreduce on a GPU buffer. Enable
# runtime CUDA support and route through the CUDA-aware UCX PML so the collective handles device
# memory.
MPIRUN=/usr/bin/mpirun
echo "-- launcher: $MPIRUN --"; "$MPIRUN" --version 2>&1 | head -1 || true
RESULT="$PWD/multigpu_result_${SLURM_JOB_ID}.json"
"$MPIRUN" -n 4 --oversubscribe --mca opal_cuda_support 1 --mca pml ucx \
  python tests/hardware/_multigpu_worker.py "$RESULT" || rc=1
EDMTN_MULTIGPU_RESULT="$RESULT" python -m pytest -q -m multigpu \
  --require-gpu --require-cuquantum --require-multigpu=4 || rc=1

echo "OVERALL rc=$rc"
exit $rc
