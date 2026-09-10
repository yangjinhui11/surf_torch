#!/bin/bash
# GPU-native (no libm_exact) full-domain 10-year 2009-2018 production run.
set -e
export CUDA_VISIBLE_DEVICES=3
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512SKX AVX512CLX AVX2 FMA3"
export SURF_CASE=/data/yangjinhui/surf_pytorch/surf_paper/experiments/prod_gpunative_10y_2009_2018
export SURF_CODE=/home/qixiang/yangjinhui/openifs/surf_pytorch
export SURF_MULTIYEAR_DATA=/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018
export SURF_INITIAL=/data/yangjinhui/surf_pytorch/surf_paper/experiments/spinup_era5land_2005_2009/restartin_era5land_2009_eq.nc
export SURF_YEARS=2009:2018
export SURF_PHYSICS_DT=1800
export SURF_OUTPUT_STRIDE=480
export SURF_PRELOAD_FORCING=1
/home/qixiang/.conda/envs/yjh/bin/python /data/yangjinhui/surf_pytorch/cmfd_offline_runner/run_torch_multiyear_prod.py 2>&1 | tee $SURF_CASE/run.log
