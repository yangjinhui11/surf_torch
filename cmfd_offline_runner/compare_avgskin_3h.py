#!/usr/bin/env python3
"""Compare 3-hourly Fortran AvgSurfT with the Torch avg-skin series."""
import json, sys
import numpy as np
from netCDF4 import Dataset

fortran_path = '/data/yangjinhui/surf_pytorch/fortran_runs/validation_2018_3h/o_gg.nc'
torch_path = ('/data/yangjinhui/surf_pytorch/surf_paper/experiments/'
              'avgskin_1y_2018_3h_20260823/torch_avgskin_3h.npz')
wrap_path = '/data/yangjinhui/surf_pytorch/fortran_runs/validation_2018_wrap/o_gg.nc'
out_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/avgskin_3h_result.json'

d = Dataset(fortran_path)
time_raw = d['time'][:]
fs_all = d['AvgSurfT'][:]
if fs_all.ndim == 3:
    fs_all = fs_all[:, 0, :]
fs_all = np.ma.filled(fs_all, np.nan).astype(np.float64)
nrec = fs_all.shape[0]

dw = Dataset(wrap_path)
fw = np.ma.filled(dw['AvgSurfT'][:], np.nan).astype(np.float64)[:, 0, :]

# Two alignment hypotheses: offset = number of initial records before the
# first window mean.  Record (offset + i) should be the window mean ending
# at step 6*(i+1).  Anchor against the validated 10-day run: mean of the
# eighty 3-hourly windows ending at 480*k must equal 10-day record k.
def anchor_rms(fs, offset):
    worst = 0.0
    for k in (1, 5, 10, 20, 30, 36):
        end = offset + 80 * k
        if end > fs.shape[0]:
            continue
        block = fs[end - 80:end]
        m = np.isfinite(block).all(axis=0) & np.isfinite(fw[k])
        worst = max(worst, float(np.sqrt((((block.mean(axis=0) - fw[k])[m]) ** 2).mean())))
    return worst

hyp = {}
for off in (0, 1):
    try:
        hyp[off] = anchor_rms(fs_all, off)
    except Exception as e:
        hyp[off] = None
offset = 0 if (hyp[1] is None or (hyp[0] is not None and hyp[0] <= hyp[1])) else 1
fs = fs_all[offset:]

z = np.load(torch_path)
steps = z['step']
tsk = z['avg_skin_t'].astype(np.float64)
ncommon = min(fs.shape[0], steps.shape[0])
D = tsk[:ncommon] - fs[:ncommon]
m = np.isfinite(D)
absD = np.abs(D[m])

# Also re-aggregate to 10-day means for a scale reference the paper already uses
D10 = D[:36*80].reshape(36, 80, -1).mean(axis=1) if ncommon >= 36*80 else None

result = {
    'fortran_records_total': int(nrec),
    'fortran_time_first4': [float(t) for t in time_raw[:4]],
    'torch_records': int(steps.shape[0]),
    'compared': int(ncommon),
    'offset_used': offset,
    'anchor_rms_by_offset': {str(k): v for k, v in hyp.items()},
    'bias_K': float(D[m].mean()),
    'rms_K': float(np.sqrt((D[m] ** 2).mean())),
    'max_abs_K': float(absD.max()),
    'p99_abs_K': float(np.percentile(absD, 99)),
    'median_abs_K': float(np.median(absD)),
    'frac_cols_over_0.1K': float((absD > 0.1).mean()),
    'frac_cols_over_1mK': float((absD > 1e-3).mean()),
    'rms_of_10day_means_K': float(np.sqrt(np.nanmean(D10 ** 2))) if D10 is not None else None,
}
print(json.dumps(result, indent=2))
json.dump(result, open(out_path, 'w'), indent=2)
