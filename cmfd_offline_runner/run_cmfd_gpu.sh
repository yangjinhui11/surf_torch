#!/usr/bin/env bash
# Run autonomous offline SURF over a prepared CMFD valid-column case.
set -euo pipefail

: "${SURF_CODE:?Set SURF_CODE to the surf_pytorch repository}"
: "${SURF_CASE:?Set SURF_CASE to a directory containing metadata.json and restartin.nc}"
: "${SURF_DATA:?Set SURF_DATA to the processed CMFD valid-column dataset}"

RUNNER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}

SURF_DEVICE=${SURF_DEVICE:-cuda} \
SURF_PHYSICS_DT=${SURF_PHYSICS_DT:-1800} \
SURF_MAX_STEPS=${SURF_MAX_STEPS:-17520} \
SURF_OUTPUT_STRIDE=${SURF_OUTPUT_STRIDE:-480} \
SURF_FAIL_ON_NONFINITE=${SURF_FAIL_ON_NONFINITE:-1} \
"${PYTHON}" "${RUNNER_DIR}/run_full_domain_offline.py"
