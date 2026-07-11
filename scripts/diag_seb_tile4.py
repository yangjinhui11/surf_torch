"""P1.6-3k: surgical single-tile SEB comparison (tile 4, dry low-veg).

The P1.6-3j SURFSEB-output dump proved that on IDENTICAL inputs the
Fortran's SURFSEB gives tile-4 ZTSK = 282.053 K while the port's surf_seb
gives 284.035 K -- a 2 K/tile discrepancy.  This script reproduces the
port's SEB for tile 4 alone, printing every intermediate coefficient, so
the divergent term can be pinpointed.

Fortran step-0 tile-4 truth (from the instrument dump):
    rhoChU=0.524166  rhoCqU=0.524166  PTSRF=285.0  PTSKTI=288.0  LAMSK=10.0
    PCPTSTI=289356.151  PQSTI=0.0105607564  PDQSTI=6.84998e-4
    PCAIRTI=0.001737114  PCSATTI=0.001737114
    PSLRFL=-116.87013  PEMIS=0.9666827  PTSKRAD=288.0
    ZASL=0  ZBSL=283494.728  ZAQL=0  ZBQL=0.00498800656  PTMST=3600
    => ZTSK=282.053  PTSKTIP1=282.053  PSSH=+59.66  PSLH=-3.41  PSTR=-117.32

We replicate the SURFSEB algebra by hand on these scalars and compare to
both the Fortran output and the port's tensor surf_seb.
"""
from __future__ import annotations
import os, sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from surf_pytorch import SurfConfig
from surf_pytorch.constants import PhysicalConstants, ConstantBundle
from surf_pytorch.physics.seb import SebInputs, surf_seb


# --- Fortran step-0 tile-4 truth -------------------------------------------
RHOCHU = 0.524166311
RHOCQU = 0.524166311
PTSRF  = 285.0
PTSKTI = 288.0
LAMSK  = 10.0
PCPTSTI = 289356.151
PQSTI   = 0.0105607564
PDQSTI  = 6.84997915e-4
PCAIRTI = 0.00173711374
PCSATTI = 0.00173711374
PSLRFL  = -116.870129161
PEMIS   = 0.966682658
PTSKRAD = 288.0
ZBSL    = 283494.728
ZBQL    = 0.00498800656

# Fortran physical constants (yomcst / yoethf)
RSIGMA = 5.67e-8       # placeholder, overwritten below
RCPD   = 1005.0        # placeholder
RVTMP2 = 0.608         # placeholder
RLVTT  = 2.5008e6      # placeholder
RTT    = 273.16

# Fortran output (truth):
FT_ZTSK = 282.052761
FT_PSSH = 59.6607960
FT_PSLH = -3.41310801
FT_PSTR = -117.317721
FT_PG0  = -29.4723891
FT_PJQ  = -1.36480646e-6


def _consts():
    cb = ConstantBundle(PhysicalConstants(), dtype=torch.float64,
                        device=torch.device('cpu'))
    return (float(cb.rsigma.item()), float(cb.cpd.item()),
            float(cb.rvtmp2.item()), float(cb.rlvtt.item()),
            float(cb.rtt.item()), cb)


def main():
    global RSIGMA, RCPD, RVTMP2, RLVTT, RTT
    RSIGMA, RCPD, RVTMP2, RLVTT, RTT, cb = _consts()
    dtype = torch.float64
    cfg = SurfConfig(dtype=dtype, device=torch.device('cpu'))

    print("P1.6-3k: single-tile SEB comparison (tile 4, dry low-veg)")
    print("=" * 66)
    print(f"consts: rsigma={RSIGMA:.6e} cpd={RCPD:.4f} rvtmp2={RVTMP2:.6f} rlvtt={RLVTT:.2f}")
    print()

    # --- Hand-replicated SURFSEB algebra (matches seb.py lines 262-368) ---
    # The Dirichlet boundary: asl=0, bsl=ZBSL, aql=0, bql=ZBQL.
    # Section 2: dlwdt (lelwdd branch)
    dlwdt = -4.0 * PEMIS * RSIGMA * PTSKRAD ** 3
    print(f"dlwdt (port formula)        = {dlwdt:.6f}")

    # Section 3: Js/Jq coefficients
    icptm1 = PTSKTI / PCPTSTI
    cjs1 = RHOCHU
    cjs3 = -RHOCHU
    cjq2 = RHOCQU * PCAIRTI
    cjq3 = -RHOCQU * PCSATTI * PDQSTI * icptm1
    cjq4 = -RHOCQU * PCSATTI * (PQSTI - PDQSTI * PTSKTI)
    print(f"icptm1 = PTSKTI/PCPTSTI     = {icptm1:.9e}")
    print(f"cjq2 = rhoCqU*alphal        = {cjq2:.6e}")
    print(f"cjq3 = -rhoCqU*alphas*dqs*ic= {cjq3:.6e}")
    print(f"cjq4 = -rhoCqU*alphas*(qs-dq*tsk) = {cjq4:.6e}")

    # Section 4: eliminate Ssk
    # lam for tile 4 (dry low veg) = RLVTT (vaporisation)
    lam = RLVTT
    frsr = 0.05   # tile 3/4 solar transmission RVTRSR(1)
    delta = RVTMP2
    coef1 = lam - RCPD * PTSKTI * delta
    zz = (dlwdt - LAMSK) * icptm1 + cjs3 + coef1 * cjq3
    zizz = 1.0 / zz
    print(f"coef1 = lam - cpd*tsk*rvtmp2= {coef1:.6f}")
    print(f"zz = (dlwdt-lamsk)*ic+cjs3+c1*cjq3 = {zz:.6e}")
    print(f"zizz = 1/zz                  = {zizz:.6e}")

    dss1 = -cjs1 * zizz
    dss2 = -coef1 * cjq2 * zizz
    dss4 = (-0.0 * frsr - PSLRFL + dlwdt * PTSKRAD - coef1 * cjq4 - LAMSK * PTSRF) * zizz
    print(f"dss1 = -cjs1*zizz            = {dss1:.9e}")
    print(f"dss2 = -coef1*cjq2*zizz      = {dss2:.9e}")
    print(f"dss4 = (...)*zizz            = {dss4:.9e}")

    # Section 5/6: grid-mean solve.  With asl=aql=0 the 2x2 collapses.
    # zz1=1, zz3 = bsl*dss1 + bql*dss2 + dss4, etc.
    zz1 = 1.0
    zz3 = ZBSL * dss1 + ZBQL * dss2 + dss4
    zy1 = 1.0
    zy3 = ZBSL * dss1 + ZBQL * dss2 + dss4   # ejq1 = frti*djq1, djq1=cjq3*dss1
    # Actually need full tile-mean; with single tile frti=1:
    # ejq1 = djq1 = cjq3*dss1, ejq2 = djq2+cjq3*dss2, ejq4 = cjq3*dss4+cjq4
    djq1 = cjq3 * dss1
    djq2 = cjq2 + cjq3 * dss2
    djq4 = cjq3 * dss4 + cjq4
    zz3_full = ZBSL * dss1 + ZBQL * dss2 + dss4   # ejs4 = dss4 (single tile)
    zy3_full = ZBSL * djq1 + ZBQL * djq2 + djq4
    jq = (zz3_full * 0.0 + zz1 * zy3_full) / zz1   # = zy3_full (zz2=0)
    js = (0.0 * zy3_full + zz3_full * zy1) / zz1   # = zz3_full
    sl = 0.0 * js + ZBSL
    ql = 0.0 * jq + ZBQL

    pssk = dss1 * sl + dss2 * ql + dss4
    ptsk = pssk * icptm1
    pjq = djq1 * sl + djq2 * ql + djq4
    pssh = (js) - RCPD * PTSKTI * delta * pjq   # note: pjs = djs1*sl+... = ?
    # Recompute pjs properly
    djs1 = cjs1 + cjs3 * dss1
    djs4 = cjs3 * dss4
    pjs = djs1 * sl + djs4   # djs2 term needs ql but djs2=cjs3*dss2
    djs2 = cjs3 * dss2
    pjs = djs1 * sl + djs2 * ql + djs4
    pssh = pjs - RCPD * PTSKTI * delta * pjq
    pslh = lam * pjq
    pstr = PSLRFL + dlwdt * (ptsk - PTSKRAD)
    pg0 = LAMSK * (ptsk - PTSRF)

    print()
    print(f"{'term':<28} {'hand-calc':>14} {'Fortran':>14} {'diff':>12}")
    print("-" * 70)
    for name, mine, ft in [
        ("PTSK (skin T)", ptsk, FT_ZTSK),
        ("PSSH (sensible)", pssh, FT_PSSH),
        ("PSLH (latent)", pslh, FT_PSLH),
        ("PSTR (net LW)", pstr, FT_PSTR),
        ("PG0 (ground)", pg0, FT_PG0),
        ("PJQ (moisture)", pjq, FT_PJQ),
    ]:
        print(f"{name:<28} {mine:14.4f} {ft:14.4f} {mine-ft:+12.4f}")

    print()
    print(f"=> hand-computed PTSK = {ptsk:.4f} K (Fortran {FT_ZTSK:.4f}, diff {ptsk-FT_ZTSK:+.4f})")

    # --- Sanity: run the actual port tensor surf_seb on the same scalars ---
    print()
    print("--- port tensor surf_seb on the same single-tile inputs ---")
    def _g(x):
        return torch.tensor([x], dtype=dtype)
    def _t(x):
        return torch.tensor([x], dtype=dtype).unsqueeze(-1)  # (1,1)
    si = SebInputs(
        sskm1m=_t(PCPTSTI), tskm1m=_t(PTSKTI),
        qskm1m=_t(PQSTI), dqsdT=_t(PDQSTI),
        rhochu=_t(RHOCHU), rhocqu=_t(RHOCQU),
        alphal=_t(PCAIRTI), alphas=_t(PCSATTI),
        ssrfl=torch.zeros((1, 1), dtype=dtype),
        slrfl_grid=_g(PSLRFL), tskrad_grid=_g(PTSKRAD), emis_grid=_g(PEMIS),
        tsrf=_t(PTSRF), lamsk=_t(LAMSK),
        frti=torch.ones((1, 1), dtype=dtype),
        ktvl_grid=torch.tensor([3], dtype=torch.long),
        ktvh_grid=torch.tensor([3], dtype=torch.long),
        asl_grid=torch.zeros(1, dtype=dtype), bsl_grid=_g(ZBSL),
        aql_grid=torch.zeros(1, dtype=dtype), bql_grid=_g(ZBQL),
        ptmst=torch.tensor(3600.0, dtype=dtype),
        lelwdd=True, lelwtl=False,
    )
    with torch.no_grad():
        so = surf_seb(si, cb, cfg)
    print(f"  port surf_seb PTSK = {so.ptsk[0,0].item():.4f} K")
    print(f"  port surf_seb PSSH = {so.pssh[0,0].item():.4f} (Fortran {FT_PSSH:.4f})")
    print(f"  port surf_seb PSTR = {so.pstr[0,0].item():.4f} (Fortran {FT_PSTR:.4f})")


if __name__ == "__main__":
    main()
