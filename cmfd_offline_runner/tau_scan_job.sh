#!/usr/bin/env bash
# usage: tau_scan_job.sh GPU "td1,td2,..." "tn1,tn2,..."
set -u
GPU=$1; IFS="," read -ra TDS <<< "$2"; IFS="," read -ra TNS <<< "$3"
BASE=/data/yangjinhui/surf_pytorch/surf_paper/experiments
OUT=$BASE/scan_tau_20260824
for td in "${TDS[@]}"; do
  for tn in "${TNS[@]}"; do
    tag="td${td}_tn${tn}"
    CUDA_VISIBLE_DEVICES=$GPU /home/qixiang/.conda/envs/yjh/bin/python /data/yangjinhui/surf_pytorch/cmfd_offline_runner/modis_slab_asym_debug.py \
      --case $BASE/modis_validation_1y_2018_spunup_20260822 \
      --data /data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset \
      --source /data/yangjinhui/surf_pytorch --runner /data/yangjinhui/surf_pytorch/cmfd_offline_runner \
      --state-archive $BASE/modis_validation_1y_2018_spunup_20260822/torch_offline_daily.npz \
      --day 131 --tau-day-hours $td --tau-night-hours $tn \
      --output $OUT/${tag}.json > /dev/null 2>&1
    /home/qixiang/.conda/envs/yjh/bin/python - "$OUT/${tag}" <<PY
import sys, numpy as np
z = np.load(sys.argv[1] + ".npz")
w = np.abs(z["dt_air"])
fc = w[48:95].max(); bu = w[:48].max()
n_over = int((w[48:95].max(axis=0) > 1000).sum())
print("%-16s forecast_max=%.3e  buildup_max=%.3e  cols>1e3=%d" % ("$(basename ${tag})", fc, bu, n_over), flush=True)
PY
    rm -f $OUT/${tag}.npz
  done
done
echo SCAN_DONE_GPU_$GPU
