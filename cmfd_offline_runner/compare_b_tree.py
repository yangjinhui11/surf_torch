#!/usr/bin/env python3
"""B-tree (nhpy_openifs) annual result vs pinned A vs Fortran reference."""
import json, sys
import numpy as np
from netCDF4 import Dataset

B = np.load('/data/yangjinhui/surf_pytorch/surf_paper/experiments/nhpy_b_1y_20260823/torch_offline_daily.npz')
A = np.load('/data/yangjinhui/surf_pytorch/surf_paper/experiments/repro_autonomous_1y_rerun_20260822/torch_offline_daily.npz')
d = Dataset('/data/yangjinhui/surf_pytorch/fortran_runs/validation_2018_wrap/o_gg.nc')
fsoil = np.ma.filled(d['SoilTemp'][:, :, 0, :], np.nan).astype(np.float64)   # (37,4,N)
fsw   = np.ma.filled(d['SoilMoist'][:, :, 0, :], np.nan).astype(np.float64) # (37,4,N)
fswe  = np.ma.filled(d['SWE'][:, 0, :], np.nan).astype(np.float64)          # (37,N)
d.close()
k, fk = 35, 36  # day-360: torch snapshot index 35 (step 17280) vs fortran record 36
res = {'B_steps_tail': [int(B['step'][-2]), int(B['step'][-1])],
       'A_steps_tail': [int(A['step'][-2]), int(A['step'][-1])]}
for key, ln in (('soil_t', 0), ('soil_t', 3), ('soil_w', 3)):
    b = B[key][k][:, ln].astype(np.float64); a = A[key][k][:, ln].astype(np.float64)
    res[f'B_vs_A_{key}_L{ln+1}_day360'] = {
        'bitwise_frac': float((b == a).mean()),
        'rms': float(np.sqrt(np.mean((b - a) ** 2))),
        'max': float(np.max(np.abs(b - a)))}
bswe = B['snow_w'][k].astype(np.float64); aswe = A['snow_w'][k].astype(np.float64)
res['B_vs_A_swe_day360'] = {'bitwise_frac': float((bswe == aswe).mean()),
    'rms': float(np.sqrt(np.mean((bswe - aswe) ** 2))), 'max': float(np.max(np.abs(bswe - aswe)))}
perm = aswe > 1000.0
res['perm_snow_cols'] = int(perm.sum())
if perm.sum() > 0:
    res['B_vs_A_swe_day360_perm'] = {'rms': float(np.sqrt(np.mean((bswe[perm] - aswe[perm]) ** 2))),
                                      'max': float(np.max(np.abs(bswe[perm] - aswe[perm])))}
res['B_vs_A_swe_day360_nonperm'] = {'rms': float(np.sqrt(np.mean((bswe[~perm] - aswe[~perm]) ** 2))),
                                    'max': float(np.max(np.abs(bswe[~perm] - aswe[~perm])))}
def stats(bt, ft):
    dd = bt - ft
    mm = np.isfinite(dd)
    rmsf = np.sqrt(np.mean(ft[mm] ** 2))
    return {'rms': float(np.sqrt(np.mean(dd[mm] ** 2))), 'rel_pct': float(np.sqrt(np.mean(dd[mm] ** 2)) / rmsf * 100)}
res['B_vs_F_soil_t_L1'] = stats(B['soil_t'][k][:, 0], fsoil[fk][0])
res['B_vs_F_soil_t_L4'] = stats(B['soil_t'][k][:, 3], fsoil[fk][3])
res['B_vs_F_soil_w_L4'] = stats(B['soil_w'][k][:, 3], fsw[fk][3])
res['B_vs_F_swe'] = stats(bswe, fswe[fk])
print(json.dumps(res, indent=2))
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else '/tmp/b_tree_result.json', 'w'), indent=2)
