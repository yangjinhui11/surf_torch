#!/bin/bash
# A/B determinism probe: 2 GPUs x {zsn active, neutralized}, 30 days each.
set -e
PY=/home/qixiang/.conda/envs/yjh/bin/python
export SURF_CODE=/home/qixiang/yangjinhui/openifs/surf_pytorch
export SURF_CASE=/tmp/detprobe_case_unused
cd /data/yangjinhui/surf_pytorch/cmfd_offline_runner
OUT=/data/yangjinhui/surf_pytorch/surf_paper/experiments/determinism_probe
mkdir -p "$OUT"
A6000=GPU-d15a1b7d-1b90-6c15-665a-e75a05b5307d
A100=GPU-a3f0c2a1-aa79-be55-a46f-9ade2ab78828
CUDA_VISIBLE_DEVICES=$A6000 $PY determinism_probe.py --days 30 --zsn 9000 --out $OUT/ab_a6000_active.npz  > $OUT/ab_a6000_active.log  2>&1
CUDA_VISIBLE_DEVICES=$A6000 $PY determinism_probe.py --days 30 --zsn 1e30 --out $OUT/ab_a6000_neutral.npz > $OUT/ab_a6000_neutral.log 2>&1
CUDA_VISIBLE_DEVICES=$A100   $PY determinism_probe.py --days 30 --zsn 9000 --out $OUT/ab_a100_active.npz    > $OUT/ab_a100_active.log    2>&1
CUDA_VISIBLE_DEVICES=$A100   $PY determinism_probe.py --days 30 --zsn 1e30 --out $OUT/ab_a100_neutral.npz   > $OUT/ab_a100_neutral.log   2>&1
$PY determinism_compare.py > $OUT/compare_result.txt 2>&1
echo ALL_DONE
