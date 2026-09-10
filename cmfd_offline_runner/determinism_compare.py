#!/usr/bin/env python3
"""Bitwise-compare determinism-probe npz dumps."""

from __future__ import annotations

import sys

import numpy as np


def compare(path_a: str, path_b: str, label: str) -> None:
    a = np.load(path_a)
    b = np.load(path_b)
    any_diff = False
    print(f"=== {label} ===")
    for name in a.files:
        if name.startswith("__"):
            continue
        x, y = a[name], b[name]
        if x.shape != y.shape:
            print(f"  {name:16s} SHAPE MISMATCH {x.shape} vs {y.shape}")
            any_diff = True
            continue
        ndiff = int(np.count_nonzero(x.view(np.uint64) != y.view(np.uint64))) if x.dtype == np.float64 \
            else int(np.count_nonzero(x != y))
        if ndiff:
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            denom = np.maximum(np.abs(x.astype(np.float64)), 1e-30)
            print(f"  {name:16s} DIFF bits in {ndiff:9d}/{x.size:9d} elems  "
                  f"maxabs={d.max():.3e}  maxrel={(d / denom).max():.3e}")
            any_diff = True
        else:
            print(f"  {name:16s} identical")
    print(f"=== {label}: {'DIFFERENT' if any_diff else 'BIT-IDENTICAL'} ===\n")


if __name__ == "__main__":
    base = "/data/yangjinhui/surf_pytorch/surf_paper/experiments/determinism_probe"
    compare(f"{base}/ab_a6000_active.npz", f"{base}/ab_a6000_neutral.npz",
            "A6000: zsn active vs neutralized (no-op test of the fix)")
    compare(f"{base}/ab_a6000_active.npz", f"{base}/ab_a100_active.npz",
            "A6000 vs A100: same code (GPU-architecture test)")
