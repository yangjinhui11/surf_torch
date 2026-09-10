#!/bin/bash
# Wait for the decade chain to finish, then run the full comparison
# battery + reproducibility manifest against the fixed run.
set -u
PY=/home/qixiang/.conda/envs/yjh/bin/python
RUNNER=/data/yangjinhui/surf_pytorch/cmfd_offline_runner
EXP=/data/yangjinhui/surf_pytorch/surf_paper/experiments/autonomous_multiyear_2009_2018_torch
CMPOUT=/data/yangjinhui/surf_pytorch/surf_paper/experiments/autonomous_multiyear_2009_2018_compare
CHAIN_PID="${1:-302777}"

cd "$RUNNER"
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 120; done
sleep 30   # let the runner finish its final npz/summary writes

export SURF_TORCH_MULTIYEAR=$EXP/torch_offline_daily.npz
export SURF_FORTRAN_YEARS=/data/yangjinhui/surf_pytorch/fortran_runs/multiyear_2009_2018/years
export SURF_MULTIYEAR_DATA=/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018
export SURF_COMPARE_OUTPUT=$CMPOUT
export SURF_YEARS=2009:2018

$PY compare_multiyear.py          > $CMPOUT/rerun_compare.log 2>&1
echo "compare_multiyear exit=$?"
$PY compare_multiyear_terminal.py > $CMPOUT/rerun_terminal.log 2>&1
echo "compare_terminal exit=$?"
$PY create_multiyear_manifest.py \
    --experiment "$EXP" \
    --fortran-root /data/yangjinhui/surf_pytorch/fortran_runs/multiyear_2009_2018 \
    --data-root /data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018 \
    --source /home/qixiang/yangjinhui/openifs/surf_pytorch \
    --years 2009:2018 > $CMPOUT/rerun_manifest.log 2>&1
echo "manifest exit=$?"
echo BATTERY_DONE
tail -2 /data/yangjinhui/surf_pytorch/surf_paper/experiments/autonomous_multiyear_2009_2018_torch.log
