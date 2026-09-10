#!/bin/bash
# Wait for the validation osmMASTER run to exit, compare its output against
# the archived reference, and launch the 10-year Fortran chain if it passes.
set -u
RUN=/data/yangjinhui/surf_pytorch/fortran_runs/validation_2018_wrap
PY=/home/qixiang/.conda/envs/xichen/bin/python
RUNNER=/data/yangjinhui/surf_pytorch/cmfd_offline_runner
CHAIN_DIR=/data/yangjinhui/surf_pytorch/fortran_runs/multiyear_2009_2018
LOG=$RUN/watchdog.log

echo "[watchdog] start $(date)" >> "$LOG"
# Poll until the validation executable is gone (60 s cadence, 3 h cap).
for i in $(seq 1 180); do
  pgrep -f 'build_offline/bin/osmMASTER' > /dev/null || break
  sleep 60
done
if pgrep -f 'build_offline/bin/osmMASTER' > /dev/null; then
  echo "[watchdog] TIMEOUT after 3 h, aborting" >> "$LOG"
  exit 1
fi
echo "[watchdog] osmMASTER exited $(date)" >> "$LOG"

if [ ! -s "$RUN/o_gg.nc" ]; then
  echo "[watchdog] FAIL: no o_gg.nc produced" >> "$LOG"
  exit 1
fi

cd "$RUNNER"
"$PY" compare_fortran_validation.py > "$RUN/validation_compare.txt" 2>&1
status=$?
echo "[watchdog] comparison exit=$status $(date)" >> "$LOG"
tail -3 "$RUN/validation_compare.txt" >> "$LOG"

if [ $status -eq 0 ] && grep -q "VERDICT: PASS" "$RUN/validation_compare.txt"; then
  mkdir -p "$CHAIN_DIR"
  echo "[watchdog] PASS -> launching 10y chain $(date)" >> "$LOG"
  setsid nohup bash "$RUNNER/run_fortran_multiyear.sh" 2009 2018 \
    > "$CHAIN_DIR/chain.log" 2>&1 < /dev/null &
  echo "[watchdog] chain launched $(date)" >> "$LOG"
else
  echo "[watchdog] comparison FAIL, chain NOT launched" >> "$LOG"
fi
