# MODIS observational validation

This workflow compares the autonomous SURF integration with daily Terra
MODIS climate-model-grid products on the CMFD valid-column domain.

## Products

- MOD11C1.061: daily daytime/nighttime land-surface temperature at 0.05
  degrees, DOI `10.5067/MODIS/MOD11C1.061`.
- MOD10C1.061: daily snow-cover fraction at 0.05 degrees, DOI
  `10.5067/MODIS/MOD10C1.061`.

Earthdata credentials are read by `requests` from `~/.netrc`. The HDF4 files
are processed one day at a time and deleted unless `--keep-raw` is supplied.
The retained NetCDF contains only values sampled at the 97,709 CMFD land
columns, raw quality flags, and documented quality masks.

## Prepare observations

```bash
PY=/home/qixiang/.conda/envs/xichen/bin/python
RUN=/data/yangjinhui/surf_pytorch/cmfd_offline_runner
DATA=/data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset
OBS=/data/yangjinhui/surf_pytorch/surf_paper/observations/modis_cmg

$PY $RUN/prepare_modis_cmg.py \
  --year 2018 \
  --cmfd-data $DATA \
  --output $OBS/modis_cmg_2018_cmfd_columns.nc
```

The strict LST mask retains only the highest-quality retrievals with a stated
error no larger than 1 K. The broader good mask retains produced retrievals
with a stated error no larger than 2 K and is used only as a quality-filter
sensitivity check. Snow validation requires spatial QA 0 or 1 and at least
50 percent clear observations in the CMG cell.

## Run the observation operator

Set the usual annual-run environment and add:

```bash
export SURF_MODIS_TARGETS=$OBS/modis_cmg_2018_cmfd_columns.nc
export SURF_WRITE_OUTPUT=0
$PY $RUN/run_full_domain_offline.py
```

The optional operator writes `surf_modis_observables.nc`. Daytime and
nighttime model LST are sampled at each MOD11C1 cell's UTC view time. The
model LST operator is SURF's native grid-mean radiative skin temperature
(`AvgSurfT`/`radiative_skin_t`), rather than a new aggregation introduced by
the validation code. Snow fraction is sampled at the nominal Terra 10:30
local-solar-time overpass. These diagnostics do not feed back to the
prognostic integration.

## Compare

```bash
$PY $RUN/compare_modis_validation.py \
  --observations $OBS/modis_cmg_2018_cmfd_columns.nc \
  --model $SURF_CASE/surf_modis_observables.nc \
  --forcing-data $DATA \
  --output-dir $SURF_CASE/modis_comparison
```

The comparison reports LST bias, MAE, RMSE, and correlation under both LST
quality masks. Snow diagnostics include fractional bias/RMSE/correlation and
probability of detection, false-alarm ratio, and critical-success index at a
0.5 snow-cover threshold.
