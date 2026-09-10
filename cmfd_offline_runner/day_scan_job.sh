#!/usr/bin/env bash
# usage: day_scan_job.sh GPU TD TN
set -u
GPU=$1; TD=$2; TN=$3
BASE=/data/yangjinhui/surf_pytorch/surf_paper/experiments
OUT=$BASE/scan_tau_20260824
for DAY in 271 281 291 301 311 321 331 341 351 361 91 101 111 121 131 141 151 161 171 181; do
  TAG="td${TD}_tn${TN}_d${DAY}"
  CUDA_VISIBLE_DEVICES=$GPU /home/qixiang/.conda/envs/yjh/bin/python /data/yangjinhui/surf_pytorch/cmfd_offline_runner/modis_slab_asym_debug.py \
    --case $BASE/modis_validation_1y_2018_spunup_20260822 \
    --data /data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset \
    --source /data/yangjinhui/surf_pytorch --runner /data/yangjinhui/surf_pytorch/cmfd_offline_runner \
    --state-archive $BASE/modis_validation_1y_2018_spunup_20260822/torch_offline_daily.npz \
    --day $DAY --tau-day-hours $TD --tau-night-hours $TN \
    --output $OUT/${TAG}.json > /dev/null 2>&1
  /home/qixiang/.conda/envs/yjh/bin/python - "$OUT/${TAG}" <<PY
import sys, numpy as np
z = np.load(sys.argv[1] + ".npz")
w = np.abs(z["dt_air"])
fc = w[48:95].max(); bu = w[:48].max()
print("%-20s forecast_max=%.3e buildup_max=%.3e cols>1e3=%d" % ("$(basename ${TAG})", fc, bu, int((w[48:95].max(axis=0)>1e3).sum())), flush=True)
PY
  rm -f $OUT/${TAG}.npz
done
echo DAYSCAN_DONE_TD${TD}_TN${TN}
