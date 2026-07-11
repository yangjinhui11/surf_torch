"""P1.6-3k2: full 8-tile grid-mean solve by hand with Fortran step-0 inputs.

The isolated tile-4 SEB matches Fortran EXACTLY (PTSK=282.053).  But in the
8-tile coupled Case A the port gives tile-4 PTSK=284.04, while the Fortran
gives 282.053.  The grid-mean Sl/Ql solve (seb.py sections 5-6) couples the
tiles via the tile-fraction-weighted Ejs/Ejq coefficients.  This script
reproduces that solve by hand on the Fortran's exact per-tile coefficients
and compares the resulting per-tile PTSK to the Fortran dump -- isolating
whether the divergence is in the coupling algebra.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from surf_pytorch import SurfConfig
from surf_pytorch.constants import PhysicalConstants, ConstantBundle
from surf_pytorch.physics.seb import SebInputs, surf_seb

# ---- Fortran step-0 per-tile inputs (8-tile) ----
FT_PFRTI  = np.array([0.0, 0.0, 0.0, 0.27, 0.0, 0.63, 0.0, 0.10])
FT_RHOCHU = np.array([0.044719607, 0.060980532, 0.506147340, 0.524166311,
                      0.424271351, 0.524166311, 0.515117437, 0.049125006])
FT_PCPTSTI= np.array([291365.569, 289356.151, 289356.151, 289356.151,
                      289356.151, 289356.151, 289356.151, 289356.151])
FT_PQSTI  = np.array([0.0120121531, 0.0105607564, 0.0105607564, 0.0105607564,
                      0.0105607564, 0.0105607564, 0.00381041263, 0.0105607564])
FT_PDQSTI = np.array([7.67769908e-4, 6.84997915e-4, 6.84997915e-4, 6.84997915e-4,
                      6.84997915e-4, 6.84997915e-4, 2.77396709e-4, 6.84997915e-4])
FT_PCAIR  = np.array([1.0, 1.0, 0.0937288766, 0.00173711374,
                      1.0, 0.00173711374, 1.0, 0.255371052])
FT_PCSAT  = FT_PCAIR.copy()
FT_PTSRF  = np.array([290.0, 285.0, 285.0, 285.0, 273.15, 285.0, 273.15, 285.0])
FT_PTSKTI = 288.0
FT_LAMSK  = np.array([1e12, 58.0, 20.0, 10.0, 50.0, 10.0, 6.499682942, 15.0])  # tile1 huge
FT_PSLRFL = -116.870129161
FT_PEMIS  = 0.966682658
FT_PTSKRAD= 288.0
FT_ZBSL   = 283494.728
FT_ZBQL   = 0.00498800656
# Fortran per-tile ZTSK output (truth):
FT_ZTSK   = np.array([290.0, 281.365568, 281.813760, 282.052761,
                      280.302073, 282.052761, 286.064876, 281.142123])
# Fortran per-tile solar transmission RVTRSR (0=none, tile3/4 ->0.05, 5/6->0.03)
# tile indices 0-based: water0 ice1 wet2 dryLow3 snowLow4 dryHigh5 snowHigh6 bare7


def _tile_solar_transmission(jt):
    # mirrors seb.py _tile_solar_transmission (Fortran CASE mapping)
    if jt in (2, 3):    return 0.05   # wet skin, low-veg dry -> RVTRSR(1)
    if jt in (5, 6):    return 0.03   # high-veg dry, high-veg snow -> RVTRSR(3)
    return 0.0


def _tile_lam(jt, rlvtt, rlstt):
    # tile1 water, tiles 4/6 (snow) sublimation; else vaporisation.
    # Port _tile_latent_heat: ice(1), low-snow(4), high-snow(6) -> RLSTT.
    if jt in (1, 4, 6):  return rlstt
    return rlvtt


def main():
    cb = ConstantBundle(PhysicalConstants(), dtype=torch.float64,
                        device=torch.device('cpu'))
    rsigma = float(cb.rsigma.item()); cpd = float(cb.cpd.item())
    rvtmp2 = float(cb.rvtmp2.item()); rlvtt = float(cb.rlvtt.item())
    rlstt = float(cb.rlstt.item())
    delta = rvtmp2
    names = ["1:water","2:ice","3:wet","4:dryLow","5:snowLow","6:dryHigh","7:snowHigh","8:bare"]

    print("P1.6-3k2: full 8-tile grid-mean solve by hand (Fortran step-0 inputs)")
    print("=" * 70)
    print(f"consts: rsigma={rsigma:.6e} cpd={cpd:.4f} rvtmp2={rvtmp2:.6f} rlvtt={rlvtt:.1f} rlstt={rlstt:.1f}")
    print()

    # Per-tile coefficients (seb.py sections 2-4)
    icptm1 = FT_PTSKTI / FT_PCPTSTI
    dlwdt = -4.0 * FT_PEMIS * rsigma * FT_PTSKRAD ** 3
    print(f"dlwdt = {dlwdt:.6f}  (grid-mean, all tiles)")
    print()

    cjs1 = FT_RHOCHU.copy()
    cjs3 = -FT_RHOCHU
    cjq2 = FT_RHOCHU * FT_PCAIR
    cjq3 = -FT_RHOCHU * FT_PCSAT * FT_PDQSTI * icptm1
    cjq4 = -FT_RHOCHU * FT_PCSAT * (FT_PQSTI - FT_PDQSTI * FT_PTSKTI)

    lam = np.array([_tile_lam(j, rlvtt, rlstt) for j in range(8)])
    frsr = np.array([_tile_solar_transmission(j) for j in range(8)])
    coef1 = lam - cpd * FT_PTSKTI * delta

    zz = (dlwdt - FT_LAMSK) * icptm1 + cjs3 + coef1 * cjq3
    zizz = 1.0 / zz
    dss1 = -cjs1 * zizz
    dss2 = -coef1 * cjq2 * zizz
    dss4 = (-0.0 * frsr - FT_PSLRFL + dlwdt * FT_PTSKRAD
            - coef1 * cjq4 - FT_LAMSK * FT_PTSRF) * zizz

    djs1 = cjs1 + cjs3 * dss1
    djs2 = cjs3 * dss2
    djs4 = cjs3 * dss4
    djq1 = cjq3 * dss1
    djq2 = cjq2 + cjq3 * dss2
    djq4 = cjq3 * dss4 + cjq4

    print(f"{'tile':<10} {'rhoChU':>8} {'alphal':>9} {'dss4':>12} {'djq4':>12} {'lam':>10}")
    for j in range(8):
        print(f"{names[j]:<10} {FT_RHOCHU[j]:8.4f} {FT_PCAIR[j]:9.5f} "
              f"{dss4[j]:12.4f} {djq4[j]:12.6e} {lam[j]:10.1f}")
    print()

    # Section 5: tile-mean (Fortran frac weighting)
    frti = FT_PFRTI
    ejs1 = (frti * djs1).sum()
    ejs2 = (frti * djs2).sum()
    ejs4 = (frti * djs4).sum()
    ejq1 = (frti * djq1).sum()
    ejq2 = (frti * djq2).sum()
    ejq4 = (frti * djq4).sum()
    print(f"Ejs1={ejs1:.6e} Ejs2={ejs2:.6e} Ejs4={ejs4:.6e}")
    print(f"Ejq1={ejq1:.6e} Ejq2={ejq2:.6e} Ejq4={ejq4:.6e}")
    print()

    # Section 6: 2x2 solve (asl=aql=0)
    bsl = FT_ZBSL; bql = FT_ZBQL
    zz1 = 1.0
    zz2 = 0.0
    zz3 = bsl * ejs1 + bql * ejs2 + ejs4
    zy1 = 1.0
    zy2 = 0.0
    zy3 = bsl * ejq1 + bql * ejq2 + ejq4
    det = zz1 * zy1 - zz2 * zy2
    jq = (zz3 * zy2 + zz1 * zy3) / det
    js = (zz2 * zy3 + zz3 * zy1) / det
    sl = js + bsl       # asl=0
    ql = jq + bql       # aql=0
    print(f"grid-mean: JS={js:.6e}  JQ={jq:.6e}")
    print(f"           SL={sl:.4f}   QL={ql:.6e}")
    print()

    # Section 7: per-tile skin-T
    pssk = dss1 * sl + dss2 * ql + dss4
    ptsk = pssk * icptm1
    pjq = djq1 * sl + djq2 * ql + djq4
    print(f"{'tile':<10} {'hand PTSK':>11} {'Fortran ZTSK':>13} {'diff':>10}")
    for j in range(8):
        d = ptsk[j] - FT_ZTSK[j]
        flag = "  <-- MISMATCH" if abs(d) > 0.05 else ""
        print(f"{names[j]:<10} {ptsk[j]:11.4f} {FT_ZTSK[j]:13.4f} {d:+10.4f}{flag}")
    gm = float((ptsk * frti).sum() / frti.sum())
    print(f"\ngridpoint-mean PTSK = {gm:.4f} K (Fortran 281.962, diff {gm-281.962:+.4f})")

    # ---- Run the PORT's actual tensor surf_seb on the identical 8-tile inputs
    # to see where the port diverges from this hand-algebra. ----
    print()
    print("=" * 70)
    print("PORT tensor surf_seb on the identical 8-tile inputs:")
    dtype = torch.float64
    def _g(x): return torch.tensor([x], dtype=dtype)
    def _t8(x):
        # Per-tile (1,8) from a length-8 array/list.
        return torch.as_tensor(x, dtype=dtype).reshape(1, 8)
    si = SebInputs(
        sskm1m=_t8(FT_PCPTSTI), tskm1m=torch.full((1,8), FT_PTSKTI, dtype=dtype),
        qskm1m=_t8(FT_PQSTI), dqsdT=_t8(FT_PDQSTI),
        rhochu=_t8(FT_RHOCHU), rhocqu=_t8(FT_RHOCHU),
        alphal=_t8(FT_PCAIR), alphas=_t8(FT_PCSAT),
        ssrfl=torch.zeros((1,8), dtype=dtype),
        slrfl_grid=_g(FT_PSLRFL), tskrad_grid=_g(FT_PTSKRAD), emis_grid=_g(FT_PEMIS),
        tsrf=_t8(FT_PTSRF), lamsk=_t8(FT_LAMSK),
        frti=_t8(FT_PFRTI),
        ktvl_grid=torch.tensor([3], dtype=torch.long),
        ktvh_grid=torch.tensor([3], dtype=torch.long),
        asl_grid=torch.zeros(1, dtype=dtype), bsl_grid=_g(FT_ZBSL),
        aql_grid=torch.zeros(1, dtype=dtype), bql_grid=_g(FT_ZBQL),
        ptmst=torch.tensor(3600.0, dtype=dtype),
        lelwdd=True, lelwtl=False,
    )
    cfg = SurfConfig(dtype=dtype, device=torch.device('cpu'))
    with torch.no_grad():
        so = surf_seb(si, cb, cfg)
    port_ptsk = so.ptsk[0].numpy()
    print(f"{'tile':<10} {'hand PTSK':>11} {'port PTSK':>11} {'Fortran':>10} {'p-h':>9}")
    for j in range(8):
        ph = port_ptsk[j] - ptsk[j]
        flag = " <--" if abs(ph) > 0.05 else ""
        print(f"{names[j]:<10} {ptsk[j]:11.4f} {port_ptsk[j]:11.4f} "
              f"{FT_ZTSK[j]:10.4f} {ph:+9.4f}{flag}")
    pgm = float((port_ptsk * frti).sum() / frti.sum())
    print(f"\nport gridpoint-mean PTSK = {pgm:.4f} (hand {gm:.4f}, Fortran 281.962)")


if __name__ == "__main__":
    main()
