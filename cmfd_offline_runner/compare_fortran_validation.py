#!/usr/bin/env python
"""Compare the reproduced 2018 validation Fortran run against the archived
reference experiment.  Both are Fortran osmMASTER integrations of the same
namelist/forcing/restart, so the expectation is bitwise equality; a small
tolerance fallback covers OpenMP reduction-order effects.

Prints a per-variable table and ends with a single VERDICT line that the
watchdog script greps for.
"""
import sys
import numpy as np
from netCDF4 import Dataset

NEW = "/data/yangjinhui/surf_pytorch/fortran_runs/validation_2018_wrap/o_gg.nc"
REF = ("/data/yangjinhui/surf_pytorch/surf_paper/experiments/"
       "fortran_reference_1y_2018/interval_10d/o_gg.nc")

BITWISE_PASS_REL = 0.0
TOLERANCE_REL = 1e-8


def load(path):
    nc = Dataset(path, "r")
    nc.set_auto_mask(False)
    return nc


def main():
    new, ref = load(NEW), load(REF)
    verdicts = []
    rows = []
    for name, var in sorted(new.variables.items()):
        if name not in ref.variables:
            rows.append((name, "MISSING-IN-REF", "", ""))
            verdicts.append("FAIL")
            continue
        a = np.asarray(var[...], dtype=np.float64)
        b = np.asarray(ref.variables[name][...], dtype=np.float64)
        if a.shape != b.shape:
            rows.append((name, "SHAPE %s vs %s" % (a.shape, b.shape), "", ""))
            verdicts.append("FAIL")
            continue
        if np.array_equal(a, b):
            rows.append((name, "bitwise-identical", "0", "0"))
            verdicts.append("PASS")
            continue
        denom = max(np.abs(b).max(), 1e-300)
        maxabs = float(np.abs(a - b).max())
        rel = maxabs / denom
        ok = rel <= TOLERANCE_REL
        rows.append((name, "differs", "%.3e" % maxabs, "%.3e" % rel))
        verdicts.append("PASS" if ok else "FAIL")
    for name in sorted(set(ref.variables) - set(new.variables)):
        rows.append((name, "MISSING-IN-NEW", "", ""))
        verdicts.append("FAIL")

    print("%-24s %-22s %-12s %-12s" % ("variable", "status", "max|d|", "rel"))
    for r in rows:
        print("%-24s %-22s %-12s %-12s" % r)

    n_fail = verdicts.count("FAIL")
    n_pass = verdicts.count("PASS")
    print("\nsummary: %d pass, %d fail (tolerance rel<=%g)"
          % (n_pass, n_fail, TOLERANCE_REL))
    print("VERDICT: " + ("PASS" if n_fail == 0 else "FAIL"))
    new.close()
    ref.close()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
