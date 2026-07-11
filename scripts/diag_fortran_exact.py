"""P1.6-3j: drive vdfdifh1s with the EXACT Fortran step-0 inputs.

The P1.6-3i instrument dump (from the rebuilt native osmMASTER_coupled run
with --no-pyoifs) gave the actual values the Fortran feeds SURFSEB at
gridpoint 1, step 0::

    PTMST  PSLRFL  PEMIS  PTSKRAD =  3600  -116.87013  0.9666827  288.000
     JT  PFRTI   rhoChU    rhoCqU    PTSRF   PTSKTI  PLAMSK
      4  0.27    0.524166  0.524166  285.0   288.0   10.0
      6  0.63    0.524166  0.524166  285.0   288.0   10.0
      8  0.10    0.049125  0.049125  285.0   288.0   15.0

Two things stand out vs the port:

  1. **rhoChU is PER-TILE** -- the Fortran's VDFEXCS computes a separate
     exchange coefficient per tile (PCFHTI(JL,JT)).  Bare soil (tile 8) has
     rhoChU = 0.049, ~10x smaller than the vegetated tiles (0.524).  The
     port's forward() broadcasts a single grid-mean Cd across all tiles.
  2. **PEMIS = 0.9667**, not the port's 0.99; **PSLRFL = -116.87**.

This script builds a DiffusionInputs from those exact values and runs the
port's vdfdifh1s.  If the grid-mean skin-T delta matches the Fortran's
-6.038 K, the per-tile exchange coefficient is the confirmed root cause.
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


# Exact Fortran step-0 dump (P1.6-3i).  Only the 3 active tiles carry values;
# inactive tiles are filled with the vegetated rhoChU (irrelevant -- frac=0).
FT_PFRTI   = [0.0, 0.0, 0.0, 0.27, 0.0, 0.63, 0.0, 0.10]
FT_RHOCHU  = [0.044719607, 0.060980532, 0.506147340, 0.524166311,
              0.424271351, 0.524166311, 0.515117437, 0.049125006]
FT_RHOCQU  = FT_RHOCHU   # near-identical to rhoChU in the dump
FT_PTSRF   = 285.0       # land tiles (soil top)
FT_PTSKTI  = 288.0       # skin at t-1
FT_PLAMSK  = [58.0, 58.0, 20.0, 10.0, 50.0, 10.0, 6.499682942, 15.0]
FT_PSLRFL  = -116.870129161
FT_PEMIS   = 0.966682658
FT_PTSKRAD = 288.0
FT_PTSPHY  = 3600.0
# Lowest-level forcing (from the dump): PTM1=282.609, PQM1=5.174e-3.
FT_PT1     = 282.608963740
FT_PQ1     = 0.005173906

# Full SURFSEB input set (P1.6-3i extended dump).  These are the ACTUAL
# values the Fortran's VEVAP/skin-reservoir pipeline produced at step 0.
# Per-tile: PCPTSTI (cp*Tsk), PQSTI (qsat(Tsk)), PDQSTI (dqsat/dT),
# PCAIRTI (alphal), PCSATTI (alphas).  NOTE PCAIRTI==PCSATTI per tile.
FT_PCPTSTI = [291365.569, 289356.151, 289356.151, 289356.151,
              289356.151, 289356.151, 289356.151, 289356.151]
FT_PQSTI   = [0.0120121531, 0.0105607564, 0.0105607564, 0.0105607564,
              0.0105607564, 0.0105607564, 0.00381041263, 0.0105607564]
FT_PDQSTI  = [7.67769908e-4, 6.84997915e-4, 6.84997915e-4, 6.84997915e-4,
              6.84997915e-4, 6.84997915e-4, 2.77396709e-4, 6.84997915e-4]
FT_PCAIRTI = [1.0, 1.0, 0.0937288766, 0.00173711374,
              1.0, 0.00173711374, 1.0, 0.255371052]
FT_PCSATTI = FT_PCAIRTI   # identical in the dump
# Dirichlet boundary (per grid): ZASL=0, ZBSL=283494.728, ZAQL=0, ZBQL=0.00498800656
FT_ZBSL    = 283494.728
FT_ZBQL    = 0.00498800656


def _capture_baseline_inp():
    """Run forward() once to harvest the model's assembled DiffusionInputs
    (qsat/dqsdT/alphas/alphal/etc are hard to reconstruct by hand).  We then
    OVERWRITE the Fortran-truth fields."""
    cfg = IdealizedDiurnalConfig(
        n_grid=1, dt_seconds=3600.0, n_steps=48,
        lat_deg=45.0, lon_deg=0.0, day_of_year=172,
        tair_mean=290.0, tair_amp=8.0, wind_speed=5.0, rh=0.7,
        precip_events=[(10, 14, 1e-4)],
        soil_t_init=285.0, soil_w_init=0.30,
        skin_t_init=288.0, sst_init=290.0,
    )
    fl, static, init = build_idealized_diurnal(cfg)
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
            model(init, fl[0], static)
    finally:
        model_mod.compute_vdfdifh1s = orig
    return cap['inp'], model.consts, model.config, static


def main():
    inp, consts, config, static = _capture_baseline_inp()
    dtype = torch.float64; device = torch.device('cpu')

    # ---- Build the Fortran-exact DiffusionInputs ----
    # Per-tile rhoChU/rhoCqU.  The wrapper expects the RG*PTSPHY*RVDIFTS-
    # rescaled form (PCFHTI = rhoChU * RG*PTSPHY*RVDIFTS), which it inverts
    # internally via ZCOEF1 = 1/(RG*RVDIFTS*PTMST).  So multiply back.
    rg = float(consts.g.item())
    from surf_pytorch.physics.diffusion import RVDIFTS
    rscale = rg * FT_PTSPHY * RVDIFTS
    pcfhti = torch.tensor([[v * rscale for v in FT_RHOCHU]], dtype=dtype, device=device)
    pcfqti = torch.tensor([[v * rscale for v in FT_RHOCQU]], dtype=dtype, device=device)

    ft_inp = replace(
        inp,
        # Exchange coefficients -- per-tile (the key fix).
        pcfhti=pcfhti, pcfqti=pcfqti,
        # Radiation / emissivity -- exact Fortran values.
        pslrfl=torch.tensor([FT_PSLRFL], dtype=dtype, device=device),
        pemis=torch.tensor([FT_PEMIS], dtype=dtype, device=device),
        ptskrad=torch.tensor([FT_PTSKRAD], dtype=dtype, device=device),
        pssrflti=torch.zeros((1, 8), dtype=dtype, device=device),  # night, SW=0
        # Tile fractions -- exact Fortran ZFRTI.
        pfrti=torch.tensor([FT_PFRTI], dtype=dtype, device=device),
        # Skin at t-1 / sub-surface T -- exact.
        tskm1m=torch.full((1, 8), FT_PTSKTI, dtype=dtype, device=device),
        sskm1m=torch.tensor([FT_PCPTSTI], dtype=dtype, device=device),
        qskm1m=torch.tensor([FT_PQSTI], dtype=dtype, device=device),
        dqsdT=torch.tensor([FT_PDQSTI], dtype=dtype, device=device),
        # Moisture closure -- exact Fortran PCAIRTI/PCSATTI (the ROOT CAUSE).
        alphal=torch.tensor([FT_PCAIRTI], dtype=dtype, device=device),
        alphas=torch.tensor([FT_PCSATTI], dtype=dtype, device=device),
        # Dirichlet boundary -- exact Fortran ZBSL/ZBQL.
        pslm1_lev=torch.tensor([FT_ZBSL], dtype=dtype, device=device),
        pqtm1_lev=torch.tensor([FT_ZBQL], dtype=dtype, device=device),
        # VLAMSK per-tile (the dump's PLAMSK column).
        plamsk=torch.tensor([FT_PLAMSK], dtype=dtype, device=device),
        ptmst=torch.tensor(FT_PTSPHY, dtype=dtype, device=device),
    )

    names = ["1:water", "2:ice", "3:wet", "4:dryLow", "5:snowLow",
             "6:dryHigh", "7:snowHigh", "8:bare"]
    frac = np.array(FT_PFRTI)

    print("P1.6-3j: vdfdifh1s with EXACT Fortran step-0 inputs")
    print("=" * 64)
    print(f"  PSLRFL={FT_PSLRFL:.4f}  PEMIS={FT_PEMIS:.6f}  PTSKRAD={FT_PTSKRAD}")
    print(f"  PTM1={FT_PT1:.4f}  PQM1={FT_PQ1:.6e}")
    print(f"  PCAIRTI[4,6,8] = {FT_PCAIRTI[3]}, {FT_PCAIRTI[5]}, {FT_PCAIRTI[7]}")
    print()

    # Case A: Fortran-exact rhoChU (per-tile) + Fortran-exact radiation.
    with torch.no_grad():
        out = diff_mod.vdfdifh1s(ft_inp, consts, config)
    sk = out.ptsktip1[0].cpu().numpy()
    print("Case A: per-tile rhoChU (Fortran exact) + Fortran radiation")
    # Fortran ZTSK per tile (P1.6-3j dump) for direct per-tile comparison.
    FT_ZTSK = [290.0, 281.365568, 281.813760, 282.052761,
               280.302073, 282.052761, 286.064876, 281.142123]
    print(f"{'tile':<12} {'port Tout':>10} {'Fortran ZTSK':>13} {'per-tile diff':>14}")
    for j in range(8):
        d = sk[j] - FT_ZTSK[j]
        flag = "  <-- MISMATCH" if abs(d) > 0.05 else ""
        print(f"{names[j]:<12} {sk[j]:10.4f} {FT_ZTSK[j]:13.4f} {d:+14.4f}{flag}")
    gm = float((sk * frac).sum() / frac.sum())
    print(f"  gridpoint-mean (Fortran frac) Δskin = {gm-FT_PTSKTI:+.4f} K")
    print(f"  Fortran o_gg[1]-o_gg[0]            = {-6.038:+.4f} K")
    print(f"  RESIDUAL gap                       = {gm-FT_PTSKTI-(-6.038):+.4f} K")

    # Case B: isolate the per-tile rhoChU effect -- keep port radiation,
    # swap ONLY the exchange coefficients to per-tile Fortran values.
    ft_b = replace(
        inp,
        pcfhti=pcfhti, pcfqti=pcfqti,
        pfrti=torch.tensor([FT_PFRTI], dtype=dtype, device=device),
        tskm1m=torch.full((1, 8), FT_PTSKTI, dtype=dtype, device=device),
        sskm1m=torch.full((1, 8), consts.cpd.item() * FT_PTSKTI,
                          dtype=dtype, device=device),
        plamsk=torch.tensor([FT_PLAMSK], dtype=dtype, device=device),
        ptmst=torch.tensor(FT_PTSPHY, dtype=dtype, device=device),
    )
    with torch.no_grad():
        out_b = diff_mod.vdfdifh1s(ft_b, consts, config)
    sk_b = out_b.ptsktip1[0].cpu().numpy()
    gm_b = float((sk_b * frac).sum() / frac.sum())
    print()
    print("Case B: per-tile rhoChU + PORT radiation (isolate exch-coeff effect)")
    print(f"  gridpoint-mean Δskin = {gm_b-FT_PTSKTI:+.4f} K  "
          f"(vs port default-broadcast Δskin)")

    # Case C: the current port (grid-mean Cd broadcast) for reference.
    with torch.no_grad():
        out_c = diff_mod.vdfdifh1s(inp, consts, config)
    sk_c = out_c.ptsktip1[0].cpu().numpy()
    gm_c = float((sk_c * frac).sum() / frac.sum())
    print()
    print("Case C: current port (grid-mean Cd broadcast, port radiation)")
    print(f"  gridpoint-mean Δskin = {gm_c-FT_PTSKTI:+.4f} K")


if __name__ == "__main__":
    main()
