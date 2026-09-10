#!/usr/bin/env bash
# Chain 10 autonomous OpenIFS offline-driver years over the CMFD 2009-2018
# continuous forcing.  Each year runs osmMASTER restarted from the previous
# year's restartout.nc; the per-year forcing file links its two terminal
# records to the next year's first records (the continuous-file convention
# matched by the PyTorch multiyear runner).
#
# Usage: run_fortran_multiyear.sh [START_YEAR] [END_YEAR]
set -euo pipefail

START=${1:-2009}
END=${2:-2018}
ROOT=/data/yangjinhui/surf_pytorch/fortran_runs/multiyear_${START}_${END}
RUNNER=/data/yangjinhui/surf_pytorch/cmfd_offline_runner
DATA_ROOT=/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018
INIT=/data/yangjinhui/surf_pytorch/surf_paper/experiments/initial_states/native_20d_terminal_complete.nc
EXE=/home/qixiang/yangjinhui/openifs/build_offline/bin/osmMASTER
THREADS=${SURF_FORTRAN_THREADS:-128}
PYTHON=/home/qixiang/.conda/envs/xichen/bin/python

mkdir -p "$ROOT/forcing_cache" "$ROOT/years"
: > "$ROOT/chain_status.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$ROOT/chain_status.log"; }

for ((YEAR=START; YEAR<=END; YEAR++)); do
    RUN="$ROOT/years/$YEAR"
    NEXT=$((YEAR == END ? START : YEAR + 1))
    [ -f "$RUN/o_gg.nc" ] && [ -f "$RUN/restartout.nc" ] && { log "year $YEAR already complete, skip"; continue; }

    NTIME=$($PYTHON -c "import json;print(json.load(open('$DATA_ROOT/cmfd_valid_columns_${YEAR}_dataset/metadata.json'))['ntime'])")
    NSTOP=$((NTIME * 6))
    NDFORC=$((NTIME + 2))
    FORCE="$ROOT/forcing_cache/forcing_${YEAR}.nc"

    # Pre-generate next year's forcing while this year integrates.
    PREFETCH_PID=""
    if [ "$YEAR" -lt "$END" ] && [ ! -f "$ROOT/forcing_cache/forcing_$((YEAR + 1)).nc" ]; then
        NEXT_WRAP=$((YEAR + 2 > END ? START : YEAR + 2))
        log "prefetch: generating forcing for $((YEAR + 1)) (wrap=$NEXT_WRAP) in background"
        $PYTHON "$RUNNER/make_fortran_forcing.py" --year $((YEAR + 1)) --next-records "next:$NEXT_WRAP" \
            --output "$ROOT/forcing_cache/forcing_$((YEAR + 1)).nc" >> "$ROOT/chain_status.log" 2>&1 &
        PREFETCH_PID=$!
    fi

    if [ ! -f "$FORCE" ]; then
        log "year $YEAR: generating forcing (ntime=$NTIME wrap=$NEXT)"
        $PYTHON "$RUNNER/make_fortran_forcing.py" --year "$YEAR" --next-records "next:$NEXT" \
            --output "$FORCE" >> "$ROOT/chain_status.log" 2>&1
    fi

    if [ "$YEAR" -eq "$START" ]; then
        RESTART="$INIT"
    else
        RESTART="$ROOT/years/$((YEAR - 1))/restartout.nc"
    fi
    [ -f "$RESTART" ] || { log "year $YEAR: missing restart $RESTART"; exit 1; }

    log "year $YEAR: setup (NSTOP=$NSTOP NDFORC=$NDFORC)"
    SURF_INIT="$RESTART" bash "$RUNNER/setup_fortran_rundir.sh" "$RUN" "$FORCE" \
        "$NSTOP" 480 "$NSTOP" "$YEAR" "$NDFORC" "$THREADS" >> "$ROOT/chain_status.log" 2>&1
    # restartin must be the chained state, not the shared default.
    ln -sf "$(readlink -f "$RESTART")" "$RUN/restartin.nc"

    log "year $YEAR: integration start"
    STARTED=$SECONDS
    (cd "$RUN" && OMP_NUM_THREADS=$THREADS mpirun --bind-to none -np 1 "$EXE" > fortran.log 2>&1)
    log "year $YEAR: integration done in $((SECONDS - STARTED)) s"

    [ -f "$RUN/restartout.nc" ] || { log "year $YEAR: FAILED, no restartout"; exit 1; }
    [ -n "$PREFETCH_PID" ] && { wait "$PREFETCH_PID" || log "prefetch for $((YEAR + 1)) failed (will regenerate)"; }
    # Free the previous year's forcing; keep only the cache entries still needed.
    PREV_FORCE="$ROOT/forcing_cache/forcing_$((YEAR - 1)).nc"
    [ -f "$PREV_FORCE" ] && rm -f "$PREV_FORCE"
    log "year $YEAR: complete (o_gg.nc + restartout.nc retained)"
done

log "chain $START-$END finished"
