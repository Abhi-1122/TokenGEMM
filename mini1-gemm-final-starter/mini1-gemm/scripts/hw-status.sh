#!/usr/bin/env bash
# hw-status.sh — your grader results from the terminal (the web page has the same data).
#
#   HW_TOKEN=<token> ./scripts/hw-status.sh          # all your submissions
#   HW_TOKEN=<token> ./scripts/hw-status.sh <id>     # one run: scores, story, full log
set -euo pipefail
: "${HW_TOKEN:?set HW_TOKEN to your personal token}"
HW_PORTAL="${HW_PORTAL:-http://node08:18080}"

if [ "$#" -ge 1 ]; then
    curl -sS -H "Authorization: Bearer $HW_TOKEN" "$HW_PORTAL/api/results/$1"
else
    curl -sS -H "Authorization: Bearer $HW_TOKEN" "$HW_PORTAL/api/submissions"
fi
