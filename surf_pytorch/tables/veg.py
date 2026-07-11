"""Vegetation parameter table (NVTYPES = 20 surface cover classes).

Faithful translation of ``SUSVEG`` (``surf/module/susveg_mod.F90``).
Index 0..20 with a 0-sentinel that mirrors type 8 (bare soil / desert),
exactly as the Fortran does::

    YDVEG%RVCOV(0) = YDVEG%RVCOV(8)
    ... likewise for every column ...

This means an uninitialised veg-type field gracefully falls back to
desert, which the rest of the code assumes.

Columns provided here (matching the assignments in SUSVEG):
  rvcov    vegetation cover fraction
  rvlai    leaf area index
  rvrsmin  minimum stomatal resistance        [s/m]
  rvhstr   humidity stress coeff (already scaled by ZCONV)
  rvz0m    roughness length momentum          [m]
  rvz0h    roughness length heat              [m]
  rvtrsr   transmission of net solar radiation through veg
  rvlamsk  unstable skin-layer conductivity
  rvlamsks stable   skin-layer conductivity

Notes
-----
* ``RVHSTR`` is multiplied by ``ZCONV = 1013.25*(RETV+1)`` at the end of
  SUSVEG.  We do the same so consumers see the final scaled value.
* ``RVZ0H`` is derived from ``RVZ0M`` with per-type /100, /10 or == z0m;
  we hard-wire the post-assignment values (lines 341-360).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..constants import RETV


# --------------------------------------------------------------------------- #
# Per-type values (index 1..20).  Sentinels at index 0 mirror type 8.
# --------------------------------------------------------------------------- #
_RVCOV = np.array([
    0.9,   # 1  Crops, Mixed Farming
    0.85,  # 2  Short Grass
    0.9,   # 3  Evergreen Needleleaf Trees
    0.9,   # 4  Deciduous Needleleaf Trees
    0.9,   # 5  Deciduous Broadleaf Trees
    0.99,  # 6  Evergreen Broadleaf Trees
    0.7,   # 7  Tall Grass
    0.0,   # 8  Desert
    0.5,   # 9  Tundra
    0.9,   # 10 Irrigated Crops
    0.1,   # 11 Semidesert
    0.0,   # 12 Ice Caps and Glaciers
    0.6,   # 13 Bogs and Marshes
    0.0,   # 14 Inland Water
    0.0,   # 15 Ocean
    0.5,   # 16 Evergreen Shrubs
    0.5,   # 17 Deciduous Shrubs
    0.9,   # 18 Mixed Forest/woodland
    0.9,   # 19 Interrupted Forest
    0.6,   # 20 Water and Land Mixtures
], dtype=np.float64)

_RVLAI = np.array([
    3.0, 2.0, 5.0, 5.0, 5.0, 6.0, 2.0, 0.5, 1.0, 3.0,
    0.5, 0.0, 4.0, 0.0, 0.0, 3.0, 1.5, 5.0, 2.5, 4.0,
], dtype=np.float64)

_RVRSMIN = np.array([
    100.0, 100.0, 250.0, 250.0, 175.0, 240.0, 100.0, 250.0, 80.0, 100.0,
    150.0, 0.0, 240.0, 0.0, 0.0, 225.0, 225.0, 250.0, 175.0, 150.0,
], dtype=np.float64)

# RVHSTR pre-scaling (susveg lines 289-308); scaled below.
_RVHSTR_RAW = np.array([
    0.0, 0.0, 0.03, 0.03, 0.03, 0.03, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.03, 0.0,
], dtype=np.float64)

_RVZ0M = np.array([
    0.25,   # 1
    0.1,    # 2
    2.00,   # 3
    2.00,   # 4
    2.00,   # 5
    2.00,   # 6
    0.47,   # 7
    0.013,  # 8
    0.034,  # 9
    0.5,    # 10
    0.17,   # 11
    0.0013, # 12
    0.5,    # 13
    0.0001, # 14
    0.0001, # 15
    0.1,    # 16
    0.25,   # 17
    2.00,   # 18
    1.1,    # 19
    0.02,   # 20
], dtype=np.float64)

# RVZ0H: derived from RVZ0M with per-type denominator (susveg 341-360).
# Build it explicitly to keep full precision.
_Z0H_DENOM = np.array([
    100, 100, 1, 1, 1, 1, 100, 100, 100, 100,
    100, 10, 100, 10, 10, 100, 100, 1, 100, 10,
], dtype=np.float64)
_RVZ0H = _RVZ0M / _Z0H_DENOM

_RVTRSR = np.array([
    0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.05, 0.00, 0.05, 0.05,
    0.05, 0.00, 0.05, 0.00, 0.00, 0.05, 0.05, 0.03, 0.03, 0.00,
], dtype=np.float64)

# RVLAMSK / RVLAMSKS are identical in SUSVEG; keep one array + alias.
_LARGE = 1.0e10
_RVLAMSK = np.array([
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 15.0, 10.0, 10.0,
    10.0, 58.0, 10.0, _LARGE, _LARGE, 10.0, 10.0, 10.0, 10.0, _LARGE,
], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Root profile (SRFROOTFR, Zeng et al. 1998, susveg line 243).
# --------------------------------------------------------------------------- #
# Per-vegetation-type Beta-distribution parameters (ZAA, ZBB) transcribed
# from srfrootfr_mod.F90 lines 123-130.  Types with ZAA <= 0.01 get a
# zero profile and are then overridden to all-roots-in-layer-1 (lines
# 169-180); we replicate that override so the table is directly usable.
# Indexing convention: arrays are 1..20; index 0 (water sentinel) is
# prepended as the type-8 mirror at build time.
# --------------------------------------------------------------------------- #
# Farquhar per-PFT tables (SUFARQUHAR, sufarquhar_mod.F90:317-331).
# Each table is (20, 2): column 1 = C3, column 2 = C4.  Defaults come from
# SUOPTSURF (suoptsurf.F90:164-354).  Only PFTs 1, 2, 7 have explicit C4
# values; the rest use the C3 value for both columns (the Fortran assigns
# RVCMAX25(7,:)=... with a single rhs, filling both).  Sentinel at index 0
# mirrors type 8, applied at build time (see _build_farquhar_tables).
# --------------------------------------------------------------------------- #
# RVCMAX25 -- maximum carboxylation rate at 25 degC (umol CO2 m-2 s-1).
_RF_RVCMAX25 = np.array([
    [49.0, 70.5],   # 1  Crops (C3 / C4)
    [48.7, 40.3],   # 2  Short grass (C3 / C4-as-PFT7)
    [62.2, 62.2],   # 3  Evergreen needleleaf
    [35.0, 35.0],   # 4  Deciduous needleleaf
    [41.9, 41.9],   # 5  Deciduous broadleaf
    [33.9, 33.9],   # 6  Evergreen broadleaf
    [40.3, 40.3],   # 7  Tall grass (C4)
    [50.0, 50.0],   # 8  Desert
    [70.0, 70.0],   # 9  Tundra
    [60.0, 60.0],   # 10 Irrigated Crops
    [50.0, 50.0],   # 11 Semidesert
    [50.0, 50.0],   # 12 Ice caps and glaciers
    [48.6, 48.6],   # 13 Bogs and Marshes
    [50.0, 50.0],   # 14 Inland water
    [50.0, 50.0],   # 15 Ocean
    [58.4, 58.4],   # 16 Evergreen Shrubs
    [44.5, 44.5],   # 17 Deciduous Shrubs
    [42.2, 42.2],   # 18 Mixed Forest/woodland
    [50.0, 50.0],   # 19 Interrupted Forest
    [50.0, 50.0],   # 20 Water and Land Mixtures
], dtype=np.float64)

# RHUMREL -- relative humidity threshold parameter for Ball-Berry (unitless).
_RF_RHUMREL = np.array([
    [4.0, 2.06],    # 1
    [3.32, 3.52],   # 2
    [2.93, 2.93],   # 3
    [1.0, 1.0],     # 4
    [3.25, 3.25],   # 5
    [4.0, 4.0],     # 6
    [3.52, 3.52],   # 7
    [1.0, 1.0],     # 8
    [4.0, 4.0],     # 9
    [1.0, 1.0],     # 10
    [1.0, 1.0],     # 11
    [1.0, 1.0],     # 12
    [4.0, 4.0],     # 13
    [1.0, 1.0],     # 14
    [1.0, 1.0],     # 15
    [2.63, 2.63],   # 16
    [4.0, 4.0],     # 17
    [1.73, 1.73],   # 18
    [1.0, 1.0],     # 19
    [1.0, 1.0],     # 20
], dtype=np.float64)

# RA1 -- Ball-Berry intercept slope parameter g0 (unitless, mol-context).
_RF_RA1 = np.array([
    [0.946, 0.864],  # 1
    [0.888, 0.668],  # 2
    [0.68, 0.68],    # 3
    [0.85, 0.85],    # 4
    [0.9775, 0.9775],# 5
    [0.9775, 0.9775],# 6
    [0.668, 0.668],  # 7
    [0.85, 0.85],    # 8
    [0.9775, 0.9775],# 9
    [0.85, 0.85],    # 10
    [0.72, 0.72],    # 11
    [0.8, 0.8],      # 12
    [0.919, 0.919],  # 13
    [0.85, 0.85],    # 14
    [0.85, 0.85],    # 15
    [0.864, 0.864],  # 16
    [0.9775, 0.9775],# 17
    [0.722, 0.722],  # 18
    [0.85, 0.85],    # 19
    [0.85, 0.85],    # 20
], dtype=np.float64)

# RB1 -- Ball-Berry slope parameter g1 (unitless).
_RF_RB1 = np.array([
    [0.13, 0.168],   # 1
    [0.117, 0.21],   # 2
    [0.168, 0.168],  # 3
    [0.14, 0.14],    # 4
    [0.112, 0.112],  # 5
    [0.138, 0.138],  # 6
    [0.21, 0.21],    # 7
    [0.14, 0.14],    # 8
    [0.112, 0.112],  # 9
    [0.14, 0.14],    # 10
    [0.2, 0.2],      # 11
    [0.14, 0.14],    # 12
    [0.126, 0.126],  # 13
    [0.14, 0.14],    # 14
    [0.14, 0.14],    # 15
    [0.206, 0.206],  # 16
    [0.112, 0.112],  # 17
    [0.153, 0.153],  # 18
    [0.14, 0.14],    # 19
    [0.14, 0.14],    # 20
], dtype=np.float64)

# RG0 -- residual stomatal conductance at zero irradiance
# (mol CO2 m-2 s-1 bar-1).
_RF_RG0 = np.array([
    [0.00699, 0.0219],   # 1
    [0.00664, 0.017],    # 2
    [0.00533, 0.00533],  # 3
    [0.00625, 0.00625],  # 4
    [0.007696, 0.007696],# 5
    [0.007838189, 0.007838189], # 6
    [0.017, 0.017],      # 7
    [0.00625, 0.00625],  # 8
    [0.008125, 0.008125],# 9
    [0.00625, 0.00625],  # 10
    [0.01875, 0.01875],  # 11
    [0.00625, 0.00625],  # 12
    [0.00739, 0.00739],  # 13
    [0.00625, 0.00625],  # 14
    [0.00625, 0.00625],  # 15
    [0.0142, 0.0142],    # 16
    [0.00773, 0.00773],  # 17
    [0.00543, 0.00543],  # 18
    [0.00625, 0.00625],  # 19
    [0.00625, 0.00625],  # 20
], dtype=np.float64)

# RGM25 -- mesophyll conductance at 25 degC (mol m-2 s-1 bar-1).
# Not used for C4; col 2 holds the default 0.4 except PFT 2 (=0.657).
_RF_RGM25 = np.array([
    [0.513, 0.4],   # 1
    [0.657, 0.657], # 2
    [0.1, 0.1],     # 3
    [0.4, 0.4],     # 4
    [0.719, 0.719], # 5
    [0.3868752, 0.3868752], # 6
    [0.4, 0.4],     # 7
    [0.4, 0.4],     # 8
    [0.8, 0.8],     # 9
    [0.4, 0.4],     # 10
    [0.4, 0.4],     # 11
    [0.4, 0.4],     # 12
    [0.59, 0.59],   # 13
    [0.4, 0.4],     # 14
    [0.4, 0.4],     # 15
    [0.4, 0.4],     # 16
    [0.724, 0.724], # 17
    [0.404, 0.404], # 18
    [0.4, 0.4],     # 19
    [0.4, 0.4],     # 20
], dtype=np.float64)

# RE_VCMAX -- activation energy for Vcmax (J mol-1).
_RF_RE_VCMAX = np.array([
    [72801.3, 80090.2],   # 1
    [50059.1, 47110.0],   # 2
    [59880.2, 59880.2],   # 3
    [71513.0, 71513.0],   # 4
    [92966.9, 92966.9],   # 5
    [67706.09, 67706.09], # 6
    [47110.0, 47110.0],   # 7
    [71513.0, 71513.0],   # 8
    [50059.1, 50059.1],   # 9
    [71513.0, 71513.0],   # 10
    [67300.0, 67300.0],   # 11
    [71513.0, 71513.0],   # 12
    [72906.6, 72906.6],   # 13
    [71513.0, 71513.0],   # 14
    [71513.0, 71513.0],   # 15
    [47110.0, 47110.0],   # 16
    [50059.1, 50059.1],   # 17
    [67899.3, 67899.3],   # 18
    [71513.0, 71513.0],   # 19
    [71513.0, 71513.0],   # 20
], dtype=np.float64)

# RE_JMAX -- activation energy for Jmax (J mol-1).
_RF_RE_JMAX = np.array([
    [64849.2, 101270.0],  # 1
    [41459.5, 54530.0],   # 2
    [35048.2, 35048.2],   # 3
    [49884.0, 49884.0],   # 4
    [34918.8, 34918.8],   # 5
    [34918.8, 34918.8],   # 6
    [54530.0, 54530.0],   # 7
    [49884.0, 49884.0],   # 8
    [34918.8, 34918.8],   # 9
    [49884.0, 49884.0],   # 10
    [77900.0, 77900.0],   # 11
    [49884.0, 49884.0],   # 12
    [34918.8, 34918.8],   # 13
    [49884.0, 49884.0],   # 14
    [49884.0, 49884.0],   # 15
    [54530.0, 54530.0],   # 16
    [52755.6, 52755.6],   # 17
    [34918.8, 34918.8],   # 18
    [49884.0, 49884.0],   # 19
    [49884.0, 49884.0],   # 20
], dtype=np.float64)

# KCO2TYP -> ZCTYPE mapping (farquhar_mod.F90:443-446):
# ZCTYPE=1 (C3) by default; ZCTYPE=2 (C4) only when KCO2TYP==4.  KCO2TYP is a
# per-grid ecoclimap-derived field (surfbc_ctl_mod.F90:182), NOT a fixed
# veg-type lookup.  NCO2TYP here stores the C3/C4 pathway index (1=C3, 2=C4)
# per veg type as a *PFT default* used when the runtime KCO2TYP is not
# supplied -- derived from the BATS/ECOCLIMAP matching table in
# sufarquhar_mod.F90:267-287.  Only PFT 7 (tall grass) is C4 by default.
_RF_NCO2TYP = np.array([
    1,  # 1  Crops (C3)
    1,  # 2  Short grass (C3)
    1,  # 3  Evergreen needleleaf (C3)
    1,  # 4  Deciduous needleleaf (C3)
    1,  # 5  Deciduous broadleaf (C3)
    1,  # 6  Evergreen broadleaf (C3)
    2,  # 7  Tall grass (C4)
    1,  # 8  Desert (sentinel)
    1,  # 9  Tundra (C3)
    1,  # 10 Irrigated Crops (C3)
    1,  # 11 Semidesert (C3)
    1,  # 12 Ice caps (sentinel)
    1,  # 13 Bogs and Marshes (C3)
    1,  # 14 Inland water (sentinel)
    1,  # 15 Ocean (sentinel)
    1,  # 16 Evergreen Shrubs (C3 -- note: BATS comment says C4 but
       #    default tables here treat it as C3)
    1,  # 17 Deciduous Shrubs (C3)
    1,  # 18 Mixed Forest (C3)
    1,  # 19 Interrupted Forest (C3)
    1,  # 20 Water and Land Mixtures (C3)
], dtype=np.int64)

# All eight per-PFT tables, in registration order.
_FARQUHAR_TABLES = (
    ("rvmax25",  _RF_RVCMAX25),
    ("rhumrel",  _RF_RHUMREL),
    ("ra1",      _RF_RA1),
    ("rb1",      _RF_RB1),
    ("rg0",      _RF_RG0),
    ("rgm25",    _RF_RGM25),
    ("re_vcmax", _RF_RE_VCMAX),
    ("re_jmax",  _RF_RE_JMAX),
)


def _build_farquhar_tables() -> dict:
    """Return per-PFT (21, 2) arrays indexed 0..20 (0 mirrors type 8).

    Mirrors the sentinel convention used by :func:`_build_table`: index 0
    copies the type-8 (desert) row so an uninitialised veg type falls back
    to desert, exactly as SUSVEG does for every other column.
    """
    out: dict = {}
    for name, arr in _FARQUHAR_TABLES:
        full = np.zeros((arr.shape[0] + 1, arr.shape[1]), dtype=np.float64)
        full[1:, :] = arr
        full[0, :] = full[8, :]   # sentinel mirrors desert (type 8)
        out[name] = full
    # nco2typ: (21,) long, sentinel at 0 mirrors type 8.
    nco2 = np.zeros(_RF_NCO2TYP.shape[0] + 1, dtype=np.int64)
    nco2[1:] = _RF_NCO2TYP
    nco2[0] = nco2[8]
    out["nco2typ"] = nco2
    return out


_RVROOTFR_AA = np.array([
    5.558, 10.739, 6.706, 7.066, 5.990,
    7.344, 8.235, 4.372, 8.992, 5.558,
    4.372, 0.0, 7.344, 0.0, 0.0,
    6.326, 6.326, 4.453, 4.453, 0.0,
], dtype=np.float64)
_RVROOTFR_BB = np.array([
    2.614, 2.608, 2.175, 1.953, 1.955,
    1.303, 1.627, 0.978, 8.992, 2.614,
    0.978, 0.0, 1.303, 0.0, 0.0,
    1.567, 1.567, 1.631, 1.631, 0.0,
], dtype=np.float64)


def _compute_root_fractions(
    rdaw: np.ndarray, n_soil: int = 4,
) -> np.ndarray:
    """Reproduce SRFROOTFR (``srfrootfr_mod.F90``).

    Args
    ----
    rdaw: soil-layer thicknesses (m), length ``n_soil``.
    n_soil: number of soil layers.

    Returns
    -------
    ``(n_soil, 21)`` array of per-(layer, veg-type) root fractions,
    indexed 0..20 (index 0 mirrors type 8 -- desert -- matching the
    sentinel convention used everywhere else in this table).

    The Zeng et al. 1998 integrated root fraction is::

        F(z) = 1 - 0.5*(exp(-A*z) + exp(-B*z))

    and the per-layer fraction is ``F(z_top) - F(z_bot)`` (with the top
    layer anchored at z=0 and the bottom layer closing the sum to 1).
    The "funny" classes 8, 12, 14, 15, 20 (and any with A<=0.01) are
    overridden to all-roots-in-layer-1 (lines 169-180).
    """
    nv = 20
    aa = _RVROOTFR_AA
    bb = _RVROOTFR_BB

    # Cumulative layer depths (ZISD).
    zisd = np.zeros(n_soil, dtype=np.float64)
    zisd[0] = rdaw[0]
    for j in range(1, n_soil):
        zisd[j] = zisd[j - 1] + rdaw[j]

    prootf = np.zeros((n_soil, nv), dtype=np.float64)
    for jt in range(nv):
        if aa[jt] <= 0.01:
            # Zero profile -- override applied below.
            continue
        # Layer 1: F(z1) = 1 - 0.5*(exp(-A*z1) + exp(-B*z1)).
        prootf[0, jt] = 1.0 - 0.5 * (
            np.exp(-aa[jt] * zisd[0]) + np.exp(-bb[jt] * zisd[0])
        )
        # Layers 2..n_soil-1: F(z_top) - F(z_bot).
        for j in range(1, n_soil - 1):
            prootf[j, jt] = 0.5 * (
                np.exp(-aa[jt] * zisd[j - 1]) + np.exp(-bb[jt] * zisd[j - 1])
                - np.exp(-aa[jt] * zisd[j]) - np.exp(-bb[jt] * zisd[j])
            )
        # Bottom layer closes the sum.
        prootf[n_soil - 1, jt] = 1.0 - prootf[: n_soil - 1, jt].sum()

    # Override "funny" classes (8, 12, 14, 15, 20 -> 1-indexed) to all-
    # roots-in-layer-1 (lines 169-180).  Convert to 0-indexed: 7, 11, 13, 14, 19.
    for jt in (7, 11, 13, 14, 19):
        prootf[:, jt] = 0.0
        prootf[0, jt] = 1.0

    # Prepend the water sentinel (index 0) mirroring type 8 (desert) --
    # matching the convention used by every other column in this table
    # and the upstream assignment RVROOTSA(:,0) = RVROOTSA(:,8).
    out = np.zeros((n_soil, nv + 1), dtype=np.float64)
    out[:, 1:] = prootf
    out[:, 0] = prootf[:, 7]
    return out


def _build_table() -> dict:
    """Return per-column arrays indexed 0..20 (0 mirrors type 8)."""
    cols = {
        "rvcov": _RVCOV,
        "rvlai": _RVLAI,
        "rvrsmin": _RVRSMIN,
        "rvhstr": _RVHSTR_RAW,
        "rvz0m": _RVZ0M,
        "rvz0h": _RVZ0H,
        "rvtrsr": _RVTRSR,
        "rvlamsk": _RVLAMSK,
        "rvlamsks": _RVLAMSK,
    }

    zconv = 1013.25 * (RETV + 1.0)

    out: dict = {}
    for name, arr in cols.items():
        full = np.zeros(arr.shape[0] + 1, dtype=np.float64)
        full[1:] = arr
        full[0] = full[8]  # sentinel mirrors desert (type 8)
        if name == "rvhstr":
            full = full * zconv
        out[name] = full
    return out


@dataclass(frozen=True)
class VegConstants:
    """Class-agnostic vegetation constants (susveg lines 98-99, 250, 363-368)."""

    rvinter: float = 0.5     # interception efficiency
    rcepsw: float = 1.0e-3   # humidity-stress floor
    rlhaero: float = 15.0    # aerodynamic skin conductivity (unstable)
    rlhaeros: float = 10.0   # aerodynamic skin conductivity (stable)
    rcvc: float = 240.0      # simplified-veg coupling
    rvlt: float = 4.0
    rvrad: float = 0.5
    repsr: float = 1.0e-10
    repevap: float = 1.0e-10


@dataclass(frozen=True)
class FarquharConstants:
    """Hardcoded Farquhar model constants (sufarquhar_mod.F90:84-258).

    These are the *class-agnostic* scalars consumed by the FARQUHAR body
    (farquhar_mod.F90 ASSOCIATE list, lines 351-380).  The per-PFT tables
    (RVCMAX25 etc.) live in :class:`VegTable` buffers, not here, because
    they vary by vegetation type.  Values are transcribed verbatim from
    SUFARQUHAR; tuples are ``(C3, C4)`` over the ``NPATH`` dimension.

    The four gas/units constants (rmair, rmco2, rrdf, rcondctmin) come from
    YOS_AGS / sucotwo_mod.F90 rather than SUFARQUHAR, but are consumed by
    the same FARQUHAR + cotworestress chain (farquhar_mod.F90:435,1237;
    cotworestress_mod.F90:427), so they are grouped here for a single
    source of truth.
    """

    # Integer dims (sufarquhar_mod.F90:63-65).
    nlai: int = 20            # number of LAI levels
    nlai1: int = 21           # NLAI + 1
    npath: int = 2            # photosynthetic pathways (C3, C4)

    # Kattge & Knorr (2007) acclimation linear-regression coefficients.
    # (a + b*Tgrowth) for the Jmax25/Vcmax25 ratio and the entropy terms.
    rarJV: tuple = (2.59, 1.715)      # a of Jmax25/Vcmax25 ratio
    rbrJV: tuple = (-0.035, 0.0)      # b of Jmax25/Vcmax25 ratio
    rasJ: tuple = (659.7, 630.0)      # a of entropy term for Jmax
    rbsJ: tuple = (-0.75, 0.0)        # b of entropy term for Jmax
    rasV: tuple = (668.39, 641.64)    # a of entropy term for Vcmax
    rbsV: tuple = (-1.07, 0.0)        # b of entropy term for Vcmax

    # Activation / deactivation energies (J mol-1).
    rd_vcmax: tuple = (200000.0, 192000.0)   # deactivation for Vcmax
    rd_jmax: tuple = (200000.0, 192000.0)    # deactivation for Jmax
    re_gamma_star: float = 37830.0           # activation for gamma_star
    re_gm: tuple = (49600.0, -9999.0)        # activation for gm
    rs_gm: tuple = (1400.0, -9999.0)         # entropy for gm
    rs_jmax: tuple = (650.0, 630.0)          # entropy for Jmax
    rd_gm: tuple = (437400.0, -9999.0)       # deactivation for gm
    re_kmc: float = 79430.0                  # activation for KmC
    re_kmo: float = 36380.0                  # activation for KmO
    re_rd: float = 46390.0                   # activation for Rd
    re_sco: float = -24460.0                 # activation for Sco

    # Reference values at 25 degC (ubar unless noted).
    rgamma_star25: float = 42.75             # CO2 compensation point (ubar)
    rkmc25: tuple = (404.9, 650.0)           # Michaelis-Menten KmC (ubar)
    rkmo25: tuple = (278400.0, 450000.0)     # Michaelis-Menten KmO (ubar)
    rsco25: tuple = (2800.0, 2590.0)         # CO2/O2 specificity (bar bar-1)

    # Water-stress multipliers (all unity = unstressed by default).
    rstress_gm: float = 1.0
    rstress_gs: float = 1.0
    rstress_vcmax: float = 1.0

    # Photosynthesis temperature limits (degC).
    rtphoto_max: float = 55.0
    rtphoto_min: float = -4.0

    # Canopy / LAI layering.
    rlai_level_depth: float = 0.15           # depth per LAI level (m^2 m^-2)
    rlaimax: float = 12.0                    # max LAI for splitting (m^2 m^-2)
    rext_coeff: float = 0.5                  # Monsi&Saeki extinction coeff

    # Units / reference scalars.
    rgb_ref: float = 1.0 / 25.0             # leaf boundary-layer resistance (s m-1)
    rtp_00: float = 273.15                   # 0 degC in K
    rpb_std: float = 1013.0                  # standard pressure (hPa)
    rrg_to_par: float = 0.5                  # radiation -> PAR conversion
    rw_to_mol: float = 4.6                   # W -> mmol * RG_TO_PAR
    ralpha_ll: float = 0.3                   # light-limited conversion eff
    rtheta: float = 0.7                      # convexity of J-irradiance curve

    # C4-specific parameters (Yin et al. 2009, Table 2).
    rfpsir: float = 0.4                      # PSII e- partitioned to C4 cycle
    rfq: float = 1.0                         # Q-cycle electron fraction
    rfpseudo: float = 0.1                    # pseudocyclic transport frac
    rh_protons: float = 4.0                  # protons per ATP (mol mol-1)
    rgbs: float = 0.003                      # bundle-sheath conductance
    roi: float = 210000.0                    # intercellular O2 (ubar)
    ralpha: float = 0.1                      # PSII activity in bundle sheath
    rkp: float = 0.7                         # PEP carboxylase efficiency

    # Diffusivity / molar conversions.
    rtetens_1: float = 0.622                 # M(H2O)/M(dry air)
    rtetens_2: float = 0.378                 # 1 - RTETENS_1
    rref_temp: float = 298.0                 # reference temperature (K)
    rmol_to_m_1: float = 0.0244              # mol/m^2/s -> m/s (gs)
    rratio_h2o_to_co2: float = 1.6           # diffusivity ratio H2O/CO2
    rn_vert_att: float = 0.7                 # N vertical attenuation

    # Downregulation CO2 scalars (sufarquhar_mod.F90:84-85).
    rdownregulation_co2_baselevel: float = 380.0
    rdownregulation_co2_minimum: float = 280.0

    # Sentinel / time constants.
    rundef: float = -9999.0
    rundef_sechiba: float = 1.0e20
    rone_month: float = 60.0 * 60.0 * 24.0 * 30.0   # one month (s)
    rtau_temp_air_month: float = 60.0 * 60.0 * 24.0 * 20.0  # monthly-T relax (s)

    # Gas constants consumed by FARQUHAR + cotworestress (YOS_AGS /
    # sucotwo_mod.F90), grouped here for a single source of truth.
    # NOTE: these are the kg/mol values from sucotwo_mod.F90:155-156
    # (RMAIR=28.970e-3, RMCO2=44.011e-3), NOT the g/mol RMCO2=44.0095
    # in sucst.F90:150 -- which feeds a different (dtforc) code path.
    rmair: float = 28.970e-3        # molar mass dry air (kg mol-1)
    rmco2: float = 44.011e-3        # molar mass CO2 (kg mol-1)
    rrdf: float = 0.111111          # Rd = An * RRDCF scaling
    rcondctmin: float = 0.0002      # min conductance floor (cotworestress:427)


class VegTable(nn.Module):
    """Per-type vegetation LUT + class-agnostic scalars.

    Same conventions as :class:`~surf_pytorch.tables.soil.SoilTable`:
    leading sentinel at index 0 (mirrors desert), buffers by default,
    optional learnable override per column.
    """

    PER_TYPE_FIELDS = (
        "rvcov", "rvlai", "rvrsmin", "rvhstr",
        "rvz0m", "rvz0h", "rvtrsr", "rvlamsk", "rvlamsks",
    )

    # Names of the per-PFT Farquhar tables (each (21, 2): C3 col 0, C4 col 1).
    # Registered as buffers OUTSIDE the PER_TYPE_FIELDS loop (same pattern as
    # rvrootsa) because they carry a 2-D (type, pathway) shape, not the flat
    # (21,) shape of the Jarvis columns.
    FARQUHAR_FIELDS = (
        "rvmax25", "rhumrel", "ra1", "rb1", "rg0", "rgm25", "re_vcmax", "re_jmax",
    )

    def __init__(
        self,
        *,
        consts: VegConstants | None = None,
        farquhar_consts: FarquharConstants | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
        learnable=None,
        n_soil: int = 4,
        soil_thickness: tuple = (0.07, 0.21, 0.72, 1.89),
    ) -> None:
        super().__init__()
        table = _build_table()
        self.consts = consts or VegConstants()
        self.farquhar_consts = farquhar_consts or FarquharConstants()
        self.dtype = dtype
        self.device = device
        self.n_soil = n_soil

        learnable = learnable or {}
        for name in self.PER_TYPE_FIELDS:
            arr = torch.as_tensor(table[name], dtype=dtype, device=device)
            if learnable.get(name, False):
                self.register_parameter(name, nn.Parameter(arr))
            else:
                self.register_buffer(name, arr)

        # RVROOTSA (per-(layer,veg-type) root fractions from SRFROOTFR).
        # Stored as a (n_soil, 21) buffer indexed 0..20; gather over the
        # veg-type axis then transpose to (Ngrid, n_soil) for the soil-
        # moisture root-extraction path.  Never learnable -- the root
        # profile is a fixed geometric property of each veg type.
        rdaw = np.asarray(soil_thickness[:n_soil], dtype=np.float64)
        roots = _compute_root_fractions(rdaw, n_soil=n_soil)
        self.register_buffer(
            "rvrootsa", torch.as_tensor(roots, dtype=dtype, device=device)
        )

        # Farquhar per-PFT tables (SUFARQUHAR, defaults from SUOPTSURF).
        # Each is (21, 2): col 0 = C3, col 1 = C4, index 0..20 with the
        # type-8 sentinel mirroring desert -- identical convention to the
        # Jarvis columns above.  Never learnable by default: the Farquhar
        # parameters are physically calibrated LUT values, not optimizable
        # weights.  ``nco2typ`` is a (21,) long buffer giving the default
        # C3(=1)/C4(=2) pathway index per veg type.
        ftables = _build_farquhar_tables()
        for name in self.FARQUHAR_FIELDS:
            arr = torch.as_tensor(ftables[name], dtype=dtype, device=device)
            if learnable.get(name, False):
                self.register_parameter(name, nn.Parameter(arr))
            else:
                self.register_buffer(name, arr)
        self.register_buffer(
            "nco2typ",
            torch.as_tensor(ftables["nco2typ"], dtype=torch.long, device=device),
        )

    def gather(self, name: str, vegtype: torch.Tensor) -> torch.Tensor:
        """Index a per-type column by a long ``vegtype`` tensor (0..20)."""
        lut = getattr(self, name)
        idx = vegtype.long().detach()
        return lut[idx]

    def gather_roots(self, vegtype: torch.Tensor) -> torch.Tensor:
        """Per-(grid, layer) root fraction profile for the given veg type.

        Args
        ----
        vegtype: ``(Ngrid,)`` long tensor of vegetation types (0..20).

        Returns
        -------
        ``(Ngrid, n_soil)`` tensor of root fractions.  Indexing matches
        the Fortran ``RVROOTSA(JK, ITYP)`` (transposed to grid-major).
        """
        idx = vegtype.long().detach()                 # (Ngrid,)
        # rvrootsa is (n_soil, 21); select columns then transpose.
        roots = self.rvrootsa.index_select(1, idx)    # (n_soil, Ngrid)
        return roots.t().contiguous()                 # (Ngrid, n_soil)

    def gather_farquhar(
        self, name: str, vegtype: torch.Tensor,
    ) -> torch.Tensor:
        """Index a per-PFT Farquhar (21, 2) table by veg type.

        Args
        ----
        name: one of :attr:`FARQUHAR_FIELDS` (e.g. ``"rvmax25"``).
        vegtype: ``(Ngrid,)`` long tensor of vegetation types (0..20).

        Returns
        -------
        ``(Ngrid, 2)`` tensor -- column 0 is the C3 value, column 1 is the
        C4 value, for the requested veg type at each grid point.  The
        caller selects the active column with the per-grid pathway index
        (``nco2typ`` gathered via :meth:`gather`).
        """
        lut = getattr(self, name)                     # (21, 2)
        idx = vegtype.long().detach()                 # (Ngrid,)
        return lut.index_select(0, idx)               # (Ngrid, 2)


__all__ = ["VegConstants", "FarquharConstants", "VegTable"]
