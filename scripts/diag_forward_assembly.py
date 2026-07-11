"""P1.6-4a: why does forward() still give -7.98K when forced to -117.6 W/m^2?

diag_fortran_exact Case A (vdfdifh1s called directly with Fortran-exact
DiffusionInputs) closes to -0.0003K.  But forward() with fr_th=-117.6
gives -7.98K vs Fortran -6.04K (residual -1.94K).  So the divergence is
in how forward()'s `_build_diffusion_inputs` ASSEMBLES the inputs vs the
Fortran-exact set.

This script captures the DiffusionInputs that forward() actually builds
(with fr_th forced to -117.6), then prints a field-by-field diff against
the Fortran-exact set (diag_fortran_exact Case A), so we can see which
field carries the -1.94K.
"""
from __future__ import annotations
import os, sys
from dataclasses import replace
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.constants import PhysicalConstants
from surf_pytorch.forcing import build_idealized_diurnal, IdealizedDiurnalConfig
from surf_pytorch.physics import diffusion as diff_mod
import surf_pytorch.model as model_mod

# Fortran-exact values (P1.6-3i dump) -- mirror diag_fortran_exact.py.
FT_PFRTI   = [0.0, 0.0, 0.0, 0.27, 0.0, 0.63, 0.0, 0.10]
FT_RHOCHU  = [0.044719607, 0.060980532, 0.506147340, 0.524166311,
              0.424271351, 0.524166311, 0.515117437, 0.049125006]
FT_PSLRFL  = -116.870129161
FT_PEMIS   = 0.966682658
FT_PTSKRAD = 288.0
FT_PTSKTI  = 288.0
FT_PTSPHY  = 3600.0


def _capture_forward_inp(fr_th_wm2: float):
    """Run forward() once with fr_th forced and harvest the DiffusionInputs."""
    cfg = IdealizedDiurnalConfig(
        n_grid=1, dt_seconds=3600.0, n_steps=48,
        lat_deg=45.0, lon_deg=0.0, day_of_year=172,
        tair_mean=290.0, tair_amp=8.0, wind_speed=5.0, rh=0.7,
        precip_events=[(10, 14, 1e-4)],
        soil_t_init=285.0, soil_w_init=0.30,
        skin_t_init=288.0, sst_init=290.0,
    )
    fl, static, init = build_idealized_diurnal(cfg)
    f0 = fl[0]
    dt = float(f0.dt.item())
    f0_fr = type(f0)(
        **{**f0.__dict__, "fr_th": torch.full_like(f0.fr_th, fr_th_wm2 * dt)},
    )
    model = LandSurface(SurfConfig(dtype=torch.float64, device=torch.device('cpu')),
                        consts=PhysicalConstants())
    cap = {}
    orig = model_mod.compute_vdfdifh1s
    def spy(inp, consts, config):
        cap['inp'] = inp
        return orig(inp, consts, config)
    model_mod.compute_vdfdifh1s = spy
    try:
        with torch.no_grad():
            model(init, f0_fr, static)
    finally:
        model_mod.compute_vdfdifh1s = orig
    return cap['inp'], model.consts, model.config, cfg


def main():
    inp, consts, config, cfg = _capture_forward_inp(FT_PSLRFL)
    rg = float(consts.g.item())
    cp = float(consts.cpd.item())
    dtype = torch.float64; device = torch.device('cpu')

    print("P1.6-4a: forward()-assembled DiffusionInputs vs Fortran truth")
    print("=" * 70)
    print(f"  forced fr_th = {FT_PSLRFL} W/m^2 (Fortran PSLRFL)")
    print()

    # --- Field-by-field dump of what forward() built ---
    print("--- key fields forward() built (gridpoint 0) ---")
    print(f"  pemis        = {inp.pemis[0].item():.6f}   (Fortran {FT_PEMIS:.6f})")
    print(f"  pslrfl       = {inp.pslrfl[0].item():.4f}   (Fortran {FT_PSLRFL:.4f})")
    print(f"  ptskrad      = {inp.ptskrad[0].item():.4f}   (Fortran {FT_PTSKRAD:.4f})")
    print(f"  ptmst        = {inp.ptmst.item():.4f}   (Fortran {FT_PTSPHY:.4f})")
    pcfhti = inp.pcfhti[0].cpu().numpy()
    rscale = rg * FT_PTSPHY * diff_mod.RVDIFTS
    rhochu_port = pcfhti / rscale
    print(f"  rhoChU (port)= {np.array2string(rhochu_port, precision=4)}")
    print(f"  rhoChU (FT)  = {np.array2string(np.array(FT_RHOCHU), precision=4)}")
    print(f"  pfrti (port) = {np.array2string(inp.pfrti[0].cpu().numpy(), precision=3)}")
    print(f"  pfrti (FT)   = {np.array2string(np.array(FT_PFRTI), precision=3)}")
    print(f"  tskm1m       = {inp.tskm1m[0,3].item():.4f} (tile4)  (Fortran {FT_PTSKTI})")

    # --- Baseline: run the captured forward()-inp directly ---
    with torch.no_grad():
        out = diff_mod.vdfdifh1s(inp, consts, config)
    frac = np.array(FT_PFRTI)
    sk = out.ptsktip1[0].cpu().numpy()
    gm = float((sk * frac).sum() / frac.sum())
    print()
    print(f"forward()-inp (as-is)        gridmean Δskin = {gm - FT_PTSKTI:+.4f} K")

    # --- Swap ONE field at a time to the Fortran-exact value, see the effect ---
    rscale = rg * FT_PTSPHY * diff_mod.RVDIFTS
    pcfhti_ft = torch.tensor([[v * rscale for v in FT_RHOCHU]], dtype=dtype, device=device)

    swaps = [
        ("rhoChU -> per-tile FT",  dict(pcfhti=pcfhti_ft, pcfqti=pcfhti_ft.clone())),
        ("pemis  -> 0.9667",       dict(pemis=torch.tensor([FT_PEMIS], dtype=dtype, device=device))),
        ("pslrfl already -116.87", dict()),
        ("ptskrad -> 288",         dict(ptskrad=torch.tensor([FT_PTSKRAD], dtype=dtype, device=device))),
        ("pfrti  -> FT frac",      dict(pfrti=torch.tensor([FT_PFRTI], dtype=dtype, device=device))),
    ]

    print()
    print(f"{'swap':<26} {'gridmean Δskin':>14} {'Δ vs baseline':>14}")
    print("-" * 58)
    base_gm = gm
    cumulative = dict()
    for name, fields in swaps:
        if not fields:
            print(f"{name:<26} {gm - FT_PTSKTI:+14.4f} {'(baseline)':>14}")
            continue
        test = replace(inp, **fields)
        with torch.no_grad():
            o = diff_mod.vdfdifh1s(test, consts, config)
        s = o.ptsktip1[0].cpu().numpy()
        g = float((s * frac).sum() / frac.sum())
        print(f"{name:<26} {g - FT_PTSKTI:+14.4f} {g - base_gm:+14.4f}")

    # --- The all-in Fortran-exact swap (should match Case A -0.0003K) ---
    allfields = dict(
        pcfhti=pcfhti_ft, pcfqti=pcfhti_ft.clone(),
        pemis=torch.tensor([FT_PEMIS], dtype=dtype, device=device),
        ptskrad=torch.tensor([FT_PTSKRAD], dtype=dtype, device=device),
        pfrti=torch.tensor([FT_PFRTI], dtype=dtype, device=device),
    )
    test = replace(inp, **allfields)
    with torch.no_grad():
        o = diff_mod.vdfdifh1s(test, consts, config)
    s = o.ptsktip1[0].cpu().numpy()
    g = float((s * frac).sum() / frac.sum())
    print(f"{'ALL Fortran-exact':<26} {g - FT_PTSKTI:+14.4f} {g - base_gm:+14.4f}  (target -6.038)")


if __name__ == "__main__":
    main()
