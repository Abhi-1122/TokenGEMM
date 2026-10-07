#!/usr/bin/env bash
# slurm.sh — run a make target on an RCE compute node (never build on the login node).
#
#   ./slurm.sh csim          ./slurm.sh csynth        ./slurm.sh cosim
#   ./slurm.sh hw_emu        ./slurm.sh hw            (hw takes ~1 h)
#
# It just submits `make <target>` as a Slurm job and tells you where the log is.
# Watch it:  squeue -u $USER      and      tail -f build/logs/<target>-<jobid>.log
set -euo pipefail
cd "$(dirname "$0")"
T="${1:?usage: ./slurm.sh <make target>   (csim | csynth | cosim | hw_emu | hw)}"
case "$T" in hw) TIME=03:00:00 ;; *) TIME=01:00:00 ;; esac

mkdir -p build/logs
sbatch --job-name="gemm-$T" --partition=debug --nodes=1 --ntasks=1 \
       --cpus-per-task=8 --mem=32G --time="$TIME" \
       --mail-type=BEGIN,END,FAIL --mail-user="${MAIL_USER:-garimella.abhishek@research.iiit.ac.in}" \
       --output="build/logs/$T-%j.log" --wrap="make $T"
echo "log: build/logs/$T-<jobid>.log"
