#!/usr/bin/env bash
# hw-submit.sh — upload your hardware bitstream to the grader (or: make submit).
#
#   HW_TOKEN=<your-token> ./scripts/hw-submit.sh [xclbin]
#
# Submits build/hw/gemm.xclbin by default. Run from an RCE login shell (the grader is on
# node08). A new submission replaces your earlier one if that one is still waiting.
set -euo pipefail
cd "$(dirname "$0")/.."
XCLBIN="${1:-build/hw/gemm.xclbin}"
HW_PORTAL="${HW_PORTAL:-http://node08:18080}"
: "${HW_TOKEN:?set HW_TOKEN to your personal token (from the instructor)}"

[ -f "$XCLBIN" ] || { echo "error: $XCLBIN not found — build it first: ./slurm.sh hw" >&2; exit 1; }
case "$XCLBIN" in *emu*) echo "error: $XCLBIN is an emulation image — submit the hw build" >&2; exit 1;; esac

echo "[submit] uploading $XCLBIN ($(du -h "$XCLBIN" | cut -f1)) ..."
curl -sS -X POST -H "Authorization: Bearer $HW_TOKEN" -H "Content-Type: application/octet-stream" \
     --data-binary @"$XCLBIN" "$HW_PORTAL/api/submit"
echo "[submit] results: http://10.4.25.39:18080/?token=$HW_TOKEN"
