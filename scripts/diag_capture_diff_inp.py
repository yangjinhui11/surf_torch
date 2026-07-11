"""Capture forward()'s DiffusionInputs and run both the PyTorch port and
the Fortran reference VDFDIFH on them.

Logic
-----
tests/test_vdfdifh_ref.py already proves that, given identical
DiffusionInputs, the PyTorch VDFDIFH1S wrapper matches the Fortran
reference to ULP (1e-9).  So if the coupled Fortran run produces
Δskin=-6.04 K while the PyTorch forward() produces -4.55 K (Cd=1e-3) or
-3.61 K (VEXCS), the divergence CANNOT be inside SURFSEB -- it must be in
the DiffusionInputs that forward() ASSEMBLES (rhoc_u, plamsk, alphas,
dqsdT, radiation, tskrad, ...).

This script confirms that closure and dumps the actual per-tile values
of every DiffusionInputs field forward() builds at step 0, so we can see
which field is the outlier vs what the coupled driver would feed.

Run
---
    python scripts/diag_capture_diff_inp.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from surf_pytorch import LandSurface, SurfConfig, Static  # noqa: E402
from surf_pytorch.constants import PhysicalConstants  # noqa: E402
from surf_pytorch.forcing import build_idealized_diurnal, IdealizedDiurnalConfig  # noqa: E402
from surf_pytorch.physics import diffusion as diff_mod  # noqa: E402
from surf_pytorch.tables.veg import VegTable  # noqa: E402

try:
    from offline_driver.surf_ref import SurfRef
    HAVE_FT = SurfRef.is_available()
except Exception as _e:
    SurfRef = None
    HAVE_FT = False
    print(f"[warn] Fortran reference unavailable: {_e}")

FORTRAN_DELTA = -6.038
STEP0_INPUT_SKIN = 288.0


def _matching_config() -> IdealizedDiurnalConfig:
    return IdealizedDiurnalConfig(
        n_grid=1, dt_seconds=3600.0, n_steps=48,
        lat_deg=45.0, lon_deg=0.0, day_of_year=172,
        tair_mean=290.0, tair_amp=8.0, wind_speed=5.0, rh=0.7,
        precip_events=[(10, 14, 1e-4)],
        soil_t_init=285.0, soil_w_init=0.30, skin_t_init=288.0, sst_init=290.0,
    )


def _gm(fld, frac):
    a = fld.detach().cpu().numpy()
    fr = frac.detach().cpu().numpy()
    if a.ndim == 2:
        return float(((a * fr).sum(-1) / fr.sum(-1)).reshape(-1)[0])
    return float(np.asarray(a).reshape(-1)[0])


def _with_z0(static, veg_table, dtype, device):
    from dataclasses import replace as _replace
    z0m = veg_table.gather("rvz0m", static.veg_type).to(dtype).to(device)
    z0h = veg_table.gather("rvz0h", static.veg_type).to(dtype).to(device)
    return _replace(static, z0m=z0m, z0h=z0h, z0q=z0h.clone())


def run_one(static, label, capture):
    """Run forward(); if capture is a dict, stash the DiffusionInputs there."""
    cfg = _matching_config()
    fl, _base, init = build_idealized_diurnal(cfg)
    forcing = fl[0]
    dtype = torch.float64
    device = torch.device("cpu")
    model = LandSurface(SurfConfig(dtype=dtype, device=device),
                        consts=PhysicalConstants())

    import surf_pytorch.model as model_mod
    orig = model_mod.compute_vdfdifh1s
    captured = {}

    def spy(inp, consts, config):
        captured["inp"] = inp
        return orig(inp, consts, config)

    model_mod.compute_vdfdifh1s = spy
    try:
        with torch.no_grad():
            new_state, diag = model(init, forcing, static)
    finally:
        model_mod.compute_vdfdifh1s = orig
    if capture is not None:
        capture["inp"] = captured["inp"]
        capture["frac"] = static.tile_frac_base
    frac = static.tile_frac_base
    gm_in = _gm(init.skin_t, frac)
    gm_out = _gm(new_state.skin_t, frac)
    return {
        "label": label,
        "exchange": "VEXCS" if static.z0m is not None else "Cd=1e-3",
        "gm_in": gm_in, "gm_out": gm_out, "delta": gm_out - gm_in,
    }


def _field_summary(inp, frac):
    """Print a compact summary of the DiffusionInputs fields forward built."""
    fields_2d = ["pcfqti", "sskm1m", "tskm1m", "qskm1m", "dqsdT",
                 "alphal", "alphas", "pssrflti", "plamsk"]
    fields_1d = ["pslrfl", "ptskrad", "pemis", "psst", "ptsm1m", "ptice",
                 "ptsnow", "ptmst"]
    for name in fields_2d:
        v = getattr(inp, name, None)
        if v is None:
            continue
        gm = _gm(v, frac)
        rng = (float(v.min()), float(v.max()))
        print(f"    {name:<10} (per-tile) gridmean={gm:+.5f}  "
              f"range=[{rng[0]:+.4g},{rng[1]:+.4g}]")
    for name in fields_1d:
        v = getattr(inp, name, None)
        if v is None:
            continue
        val = float(v.reshape(-1)[0])
        print(f"    {name:<10} (scalar)  = {val:+.5f}")


def main() -> int:
    dtype = torch.float64
    device = torch.device("cpu")
    cfg = _matching_config()
    _fl, static_neutral, _init = build_idealized_diurnal(cfg)
    veg_table = VegTable(dtype=dtype, device=device)
    static_vexcs = _with_z0(static_neutral, veg_table, dtype, device)

    print("=" * 74)
    print("Capture forward()'s DiffusionInputs; run port vs Fortran on them")
    print("=" * 74)
    print(f"reference: Fortran coupled Δskin = {FORTRAN_DELTA:+.3f} K")
    print()

    cap = {}
    r = run_one(static_vexcs, "forward() (VEXCS)", cap)
    inp = cap["inp"]
    frac = cap["frac"]

    print(f"  {r['label']:<28} Δskin={r['delta']:+7.3f}  "
          f"(forward itself)")
    print()

    # Re-run the SAME inp through both VDFDIFH implementations.
    consts = PhysicalConstants.__new__(PhysicalConstants)
    # Build the bundle the way the model did.
    model = LandSurface(SurfConfig(dtype=dtype, device=device),
                        consts=PhysicalConstants())
    cb = model.consts
    cfg_obj = model.config

    print("DiffusionInputs that forward() assembled (step 0):")
    _field_summary(inp, frac)
    print()

    out_port = diff_mod.vdfdifh1s(inp, cb, cfg_obj)
    gm_port = _gm(out_port.ptsktip1, frac)
    print(f"  port  vdfdifh1s(inp) -> Δskin = {gm_port - r['gm_in']:+7.3f}  "
          f"  H={_gm(out_port.pssh,frac):+.3f} "
          f"STR={_gm(out_port.pstr,frac):+.3f} "
          f"G0={_gm(out_port.pg0,frac):+.3f}")

    if HAVE_FT:
        ref = SurfRef()
        out_ft = ref.vdfdifh(inp, model.veg_table, cb, cfg_obj)
        gm_ft = _gm(out_ft.ptsktip1, frac)
        print(f"  Fort SurRef.vdfdifh(inp) -> Δskin = {gm_ft - r['gm_in']:+7.3f}  "
              f"  H={_gm(out_ft.pssh,frac):+.3f} "
              f"STR={_gm(out_ft.pstr,frac):+.3f} "
              f"G0={_gm(out_ft.pg0,frac):+.3f}")
        d = abs(gm_port - gm_ft)
        print()
        print(f"  |port Δskin - Fortran Δskin| (same inputs) = {d:.3e} K")
        if d < 1e-6:
            print("  => CONFIRMED: with identical DiffusionInputs the two agree.")
            print("     The skin-T gap is ENTIRELY in how forward() assembles")
            print("     its inputs (rhoc_u / plamsk / alphas / radiation /")
            print("     tskrad), NOT in SURFSEB itself.")
            print()
            print("  Next: compare these captured fields against what the")
            print("  coupled Fortran driver feeds SURFSEB. Suspects, by")
            print("  likely impact for a dry night step (SW=0):")
            print("    1. rhoc_u / rhoc_q  (VUPDZ0 blends veg cover per-tile;")
            print("       the single-veg-type gather here may over/under-state)")
            print("    2. plamsk / tskrad  (skin conductivity, radiation base T)")
            print("    3. alphas/alphal   (stomatal closure -- forward uses")
            print("       Jarvis, the ULP test used alphas=1)")
        else:
            print("  => Inputs agree but outputs differ -- unexpected; the")
            print("     SURFSEB port may have a regime-specific bug.")
    else:
        print("  [Fortran reference unavailable]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
