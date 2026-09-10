#!/usr/bin/env python3
"""Instantaneous AvgSurfT comparison: fortran o_gg record j (time=10800j s)
vs torch radiative_skin_t at step 6j (post-step instantaneous)."""
import json, sys
import numpy as np
from netCDF4 import Dataset

fortran_path = '/data/yangjinhui/surf_pytorch/fortran_runs/validation_2018_3h/o_gg.nc'
torch_path = ('/data/yangjinhui/surf_pytorch/surf_paper/experiments/'
              'avgskin_inst_1y_20260823/torch_avgskin_3h.npz')
out_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/avgskin_inst_result.json'

d = Dataset(fortran_path)
fa = np.ma.filled(d['AvgSurfT'][:, 0, :], np.nan).astype(np.float64)
d.close()
z = np.load(torch_path)
steps = z['step']
tsk = z['avg_skin_t_inst'].astype(np.float64)
assert steps[0] == 6 and steps[-1] == 17520 and len(steps) == 2920

D = tsk - fa[1:2921]
m = np.isfinite(D)
absD = np.abs(D[m])
# breakdown by local hour (CMFD UTC+8): record j covers step 6j, model time 1800*6j s UTC
hours = ((steps.astype(np.float64) * 1800.0 / 3600.0 + 8.0) % 24.0).astype(int)
per_hour = {}
for h in range(24):
    sel = (hours == h) & np.isfinite(D).all(axis=1)
    if sel.sum() > 50:
        Dh = D[sel]
        per_hour[int(h)] = {'n_records': int(sel.sum()),
                            'bias_K': float(Dh.mean()),
                            'rms_K': float(np.sqrt((Dh ** 2).mean()))}
result = {
    'compared_records': int(D.shape[0]),
    'bias_K': float(D[m].mean()),
    'rms_K': float(np.sqrt((D[m] ** 2).mean())),
    'max_abs_K': float(absD.max()),
    'p99_abs_K': float(np.percentile(absD, 99)),
    'median_abs_K': float(np.median(absD)),
    'frac_over_0.1K': float((absD > 0.1).mean()),
    'frac_over_1mK': float((absD > 1e-3).mean()),
    'first_record_bias_rms_K': [float(D[0].mean()), float(np.sqrt((D[0] ** 2).mean()))],
    'per_local_hour': per_hour,
}
print(json.dumps(result, indent=2))
json.dump(result, open(out_path, 'w'), indent=2)
