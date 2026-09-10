#!/usr/bin/env python3
"""Merge the 10-year chain's 2017 terminal restartout into a complete restartin.

The Fortran restartout lacks template-only static fields (glacierMask,
nlevsn, multi-layer snow mirrors, ...).  Start from the native 20-day
adjusted restart as the structural template, then overwrite every shared
variable with the chain state; single-layer snow mirrors are rebuilt from
the chain's 2-D snow fields.
"""
import shutil
import sys

import numpy as np
from netCDF4 import Dataset

TEMPLATE = ("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
            "initial_states/native_20d_terminal_complete.nc")
CHAIN = ("/data/yangjinhui/surf_pytorch/fortran_runs/multiyear_2009_2018/"
         "years/2017/restartout.nc")
OUT = sys.argv[1]

shutil.copyfile(TEMPLATE, OUT)
replaced, mirrors, kept = [], [], []

with Dataset(CHAIN) as src, Dataset(OUT, "a") as dst:
    for name, var in src.variables.items():
        if name not in dst.variables:
            print("skip (not in template):", name)
            continue
        if var.shape != dst[name].shape:
            # tile-dimension mismatch (Fortran 9 slots vs template 10):
            # copy the overlapping leading tiles; the loader only reads the
            # first 8 (NTILE_TORCH), so trailing slots are inert layout.
            if var.ndim == dst[name].ndim and var.shape[1:] == dst[name].shape[1:]:
                n_overlap = min(var.shape[0], dst[name].shape[0])
                dst[name][:n_overlap] = np.asarray(var[:n_overlap])
                replaced.append(name + "(leading-%d)" % n_overlap)
            else:
                print("shape mismatch, keep template:", name, var.shape,
                      dst[name].shape)
            continue
        data = np.asarray(var[:])
        dst[name][:] = data
        replaced.append(name)
    # single-layer snow mirrors consistent with the chain 2-D fields
    n = dst["lat"].shape[0]
    for mirror, base in (("SWEML", "SWE"), ("SnowTML", "SnowT"),
                         ("snowdensML", "snowdens")):
        if mirror in dst.variables and base in src.variables:
            dst[mirror][0, 0, :] = np.asarray(src[base][:])[0, :]
            mirrors.append(mirror)
    if "slwML" in dst.variables:
        dst["slwML"][:] = 0.0
        mirrors.append("slwML=0")

with Dataset(OUT) as d, Dataset(CHAIN) as c:
    for k in ("SoilTemp", "SoilMoist", "SWE", "SnowT", "snowdens",
              "SAlbedo", "CanopInt", "AvgSurfT"):
        a = np.asarray(d[k][:], np.float64)
        b = np.asarray(c[k][:], np.float64)
        assert a.shape == b.shape and np.array_equal(a, b), k
        assert np.isfinite(a).all(), k
    print("replaced %d vars: %s" % (len(replaced), ",".join(replaced)))
    print("mirrors:", mirrors)
