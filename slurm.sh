#!/usr/bin/env bash
# slurm.sh — run a make target on an RCE compute node (never build on the login node).
#
#   ./slurm.sh csim          ./slurm.sh csynth        ./slurm.sh cosim
#   ./slurm.sh hw_emu        ./slurm.sh hw            (hw takes ~3.5 h)
#
# It runs `make <target>` as a Slurm job and tells you where the log is.
# Watch it:  squeue -u $USER      and      tail -f build/logs/<target>-<jobid>.log
# Set MAIL_USER=<address> to get an email when the job starts and ends.
set -euo pipefail
cd "$(dirname "$0")"
T="${1:?usage: ./slurm.sh <make target>   (csim | csynth | cosim | hw_emu | hw)}"
# The hw build peaks at ~17 GB in place/route and OOMs at 32 GB during kernel synthesis.
case "$T" in hw) TIME=06:00:00 MEM=96G ;; *) TIME=01:00:00 MEM=32G ;; esac

mkdir -p build/logs
sbatch --job-name="gemm-$T" --partition=debug --nodes=1 --ntasks=1 \
       --cpus-per-task=8 --mem="$MEM" --time="$TIME" \
       ${MAIL_USER:+--mail-type=BEGIN,END,FAIL --mail-user="$MAIL_USER"} \
       --output="build/logs/$T-%j.log" --wrap="make $T"
echo "log: build/logs/$T-<jobid>.log"
