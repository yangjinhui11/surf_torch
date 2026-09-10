# CMFD offline SURF runner

This directory is the minimal reproducible driver for the full-domain CMFD
offline SURF experiment. It runs the autonomous PyTorch VUPDZ0/VDF/SURF
closure in double precision on a GPU. It is intentionally separate from the
`surf_pytorch` package: the package contains the model, while this directory
contains only CMFD I/O, initial-state loading, and paper diagnostics.

## Contents

- `run_cmfd_gpu.sh`: launch a full-domain GPU integration.
- `run_full_domain_offline.py`: CMFD three-hour forcing interpolation,
  autonomous VUPDZ0/VDF closure, and SURF time stepping.
- `replay_full_domain_controlled.py` and
  `diagnostic_scripts_compare_realcol.py`: state decoding and forcing
  construction used by the runner.
- `compare_autonomous_annual.py`: compute the 10-day sampled metrics and
  day-365 Fortran, PyTorch, and difference maps.
- `create_annual_repro_manifest.py`: hash the annual inputs, outputs,
  diagnostics, source revision, and experiment scripts.
- `benchmark_scaling.py` and `plot_performance_scaling.py`: execute and plot
  repeated CPU/GPU throughput measurements.
- `fortran_guided_initial_state_inversion.py`: retrieve a deliberately
  perturbed single-column initial soil-water state against independent
  Fortran day-10 output and evaluate the held-out day-20 state.

## Prerequisites

Use Python with `torch`, `numpy`, `netCDF4`, and `matplotlib`; the validated
server environment is `/home/qixiang/.conda/envs/xichen/bin/python`. The
model repository must be available locally and supplied through `SURF_CODE`.

`SURF_DATA` is the processed, valid-column CMFD dataset. It must contain
`lat.npy`, `lon.npy`, and the seven forcing arrays: `tair_K.npy`,
`qair_kg_kg.npy`, `psurf_Pa.npy`, `swdown_W_m2.npy`, `lwdown_W_m2.npy`,
`wind_m_s.npy`, and `precip_kg_m2_s.npy`. The experiment uses 3-hour forcing
records and a 1800-s physics time step.

`SURF_CASE` must contain `metadata.json` with `nvalid` and a `restartin.nc`
file. For the paper experiment it points to the common, physically spun-up
initial state `native_20d_terminal_complete.nc`; both Fortran and PyTorch
must use this identical file.

## Run a year on GPU

```bash
export SURF_CODE=/home/qixiang/yangjinhui/openifs/surf_pytorch
export SURF_DATA=/data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset
export SURF_CASE=/data/yangjinhui/surf_pytorch/surf_paper/experiments/my_cmfd_2018
export PYTHON=/home/qixiang/.conda/envs/xichen/bin/python

mkdir -p "$SURF_CASE"
cp /data/yangjinhui/surf_pytorch/surf_paper/experiments/single_layer_30d/torch_full_1y_dt1800_10d_cyclic_fix_validation/metadata.json "$SURF_CASE/"
ln -s /data/yangjinhui/surf_pytorch/surf_paper/experiments/initial_states/native_20d_terminal_complete.nc "$SURF_CASE/restartin.nc"

SURF_OUTPUT_STRIDE=480 ./run_cmfd_gpu.sh
```

`SURF_OUTPUT_STRIDE=480` saves state every ten days; it is a diagnostic
sampling choice and does not alter the 1800-s integration. The output is
`torch_offline_daily.npz` (the filename is retained for compatibility).
Set `SURF_OUTPUT_STRIDE=48` for daily diagnostics when storage permits.
The runner also appends the terminal state when the stride does not divide the
requested number of physical steps.

## GPU performance option

For a long full-domain GPU integration, set `SURF_PRELOAD_FORCING=1` to keep
the seven CMFD forcing arrays on the GPU in their source `float32` precision.
Each record is still promoted to double precision only when interpolated, so
this option does not change the numerical path. It requires approximately
8 GB of additional GPU memory and is disabled by default.

```bash
SURF_PRELOAD_FORCING=1 SURF_OUTPUT_STRIDE=480 ./run_cmfd_gpu.sh
```

The preloaded and eager-forcing paths produce bitwise-identical final
prognostic arrays in the validated benchmark. Formal throughput values are
measured separately with output disabled, rather than from a production run
that also writes diagnostics.

For batch-size experiments, set `SURF_COLUMN_COUNT=N` to integrate the
first `N` valid CMFD columns. It cannot be used together with
`SURF_SUBSET_INDEX`. The formal three-repeat, 240-step CPU/GPU scaling results
are stored in
`/data/yangjinhui/surf_pytorch/surf_paper/experiments/repro_performance_current_20260806`;
`plot_performance_scaling.py` regenerates the associated figure. These timings
exclude setup and output, and report forcing preload separately.

For a pure integration benchmark, disable state writing and specify the CPU
thread count explicitly:

```bash
SURF_DEVICE=cpu SURF_CPU_THREADS=32 SURF_WRITE_OUTPUT=0 SURF_FAIL_ON_NONFINITE=0 \
  SURF_MAX_STEPS=240 SURF_OUTPUT_STRIDE=480 ./run_cmfd_gpu.sh
```

The reported `integration_seconds` ends immediately after the final model
step and device synchronization. The final state is checked for non-finite
values after timing, so diagnostic transfers and output serialization do not
enter the throughput calculation.

`benchmark_scaling.py` executes repeated isolated CPU and GPU cases and
writes the individual runs, median scaling table, source revision, and
hardware inventory into one artifact directory. For example, on the GPU
server:

```bash
$PYTHON benchmark_scaling.py \
  --source /home/qixiang/yangjinhui/openifs/surf_pytorch \
  --data /data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset \
  --case-template /data/yangjinhui/surf_pytorch/surf_paper/experiments/repro_autonomous_1y_20260806 \
  --output /data/yangjinhui/surf_pytorch/surf_paper/experiments/repro_performance_current_20260806 \
  --python /home/qixiang/.conda/envs/xichen/bin/python \
  --cuda-visible-devices 1 --steps 240 --repeats 3 --cpu-threads 32

$PYTHON plot_performance_scaling.py \
  --summary /data/yangjinhui/surf_pytorch/surf_paper/experiments/repro_performance_current_20260806/performance_results.json \
  --output /data/yangjinhui/surf_pytorch/surf_paper/experiments/repro_performance_current_20260806/surf_cpu_gpu_scaling_current.png
```

## Gradient and regional sensitivity experiments

`gradient_validation.py` runs the deterministic 24-step double-precision
multi-step derivative check and compares reverse-mode gradients with centred
finite differences. Its compact output is retained at
`/data/yangjinhui/surf_pytorch/surf_paper/experiments/gradient_validation`.

`regional_tair_sensitivity.py` evaluates the first 12 physical steps of the
2018 CMFD sequence over the valid columns. It attaches an independent air-
temperature offset to each column and obtains the local terminal skin-
temperature response from one reverse pass. The complete result and map are
retained at
`/data/yangjinhui/surf_pytorch/surf_paper/experiments/regional_tair_sensitivity_2018_12step`.
The result is a local six-hour process diagnostic, not a finite-amplitude
climate sensitivity.

`plot_seasonal_tair_sensitivity.py` summarizes four 6-hour regional
diagnostics sampled from a one-year continuous CMFD integration. Each sampled
slow land state is followed by an unperturbed 24-hour closure adjustment before
the reverse calculation, so omitted fast diagnostic stores in the compact
10-day archive cannot be confused with seasonal land-state effects. The four
output directories and the figure inputs are retained below
`results/seasonal_tair_sensitivity` locally and under
`/data/yangjinhui/surf_pytorch/surf_paper/experiments/seasonal_tair_sensitivity`
on the GPU server.

## Fortran-guided initial-state retrieval

`fortran_guided_initial_state_inversion.py` is a controlled gradient-based
state-estimation prototype. It selects a representative warm, snow-free column
using only the separately produced Fortran day-10 and day-20 fields, adds a
known top-layer soil-water error to the common restart, retrieves that scalar
initial-state correction from the day-10 Fortran target, and evaluates the
day-20 state without using it during the update. It is not an observational
data-assimilation experiment.

```bash
export SURF_CASE=/data/yangjinhui/surf_pytorch/surf_paper/experiments/repro_autonomous_1y_20260806
export SURF_CODE=/home/qixiang/yangjinhui/openifs/surf_pytorch
export SURF_RUNNER=/data/yangjinhui/surf_pytorch/cmfd_offline_runner
export SURF_DATA=/data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset
export SURF_FORTRAN_10D=/data/yangjinhui/surf_pytorch/surf_paper/experiments/single_layer_30d/fortran_full_1y_dt1800_10d_cyclic_fix_validation/o_gg.nc
export SURF_OUTPUT=/data/yangjinhui/surf_pytorch/surf_paper/experiments/fortran_guided_initial_state_inversion

CUDA_VISIBLE_DEVICES=1 $PYTHON fortran_guided_initial_state_inversion.py \
  --case $SURF_CASE --data $SURF_DATA --reference $SURF_FORTRAN_10D \
  --source $SURF_CODE --runner $SURF_RUNNER --output $SURF_OUTPUT \
  --device cuda --target-step 480 --validation-step 960 \
  --initial-bias-kg-m2 5.0 --iterations 5
```

The result directory contains the source revision, the selected longitude and
latitude, every Gauss--Newton update, the day-10 calibration errors, the
withheld day-20 errors, and a compact comparison figure. To redraw only the
figure from an existing summary, add
`--render-existing $SURF_OUTPUT/fortran_guided_initial_state_summary.json` to
the same command; no SURF integration is performed in that mode.

## Reproduce the annual comparison figures and manifest

The comparison script accepts paths explicitly, so no experiment name is
hard-coded. It compares every matching 10-day snapshot and the guaranteed
terminal day-365 snapshot. The terminal maps are separated into
soil-temperature, soil-water, and snow-state figures so every prognostic
component is readable at manuscript scale.

```bash
export SURF_CASE=/data/yangjinhui/surf_pytorch/surf_paper/experiments/repro_autonomous_1y_20260806
export SURF_CODE=/home/qixiang/yangjinhui/openifs/surf_pytorch
export SURF_TORCH_ANNUAL=$SURF_CASE/torch_offline_daily.npz
export SURF_FORTRAN_10D=/data/yangjinhui/surf_pytorch/surf_paper/experiments/single_layer_30d/fortran_full_1y_dt1800_10d_cyclic_fix_validation/o_gg.nc
export SURF_FORTRAN_FINAL=/data/yangjinhui/surf_pytorch/surf_paper/experiments/single_layer_30d/fortran_full_1y_dt1800/o_gg.nc
export SURF_DATA=/data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset
export SURF_COMPARE_OUTPUT=$SURF_CASE/comparison

$PYTHON compare_autonomous_annual.py
$PYTHON create_annual_repro_manifest.py \
  --experiment $SURF_CASE --source $SURF_CODE --data $SURF_DATA \
  --fortran-10d $SURF_FORTRAN_10D --fortran-final $SURF_FORTRAN_FINAL \
  --python $PYTHON --cuda-visible-devices 1
```

The Fortran references are generated by `osmMASTER` using the same
`restartin.nc`, forcing, `TSTEP=1800`, and `NSTOP=17520`. Its 10-day output
uses `NFRPOS=480`; the end state is read from the final Fortran output.
