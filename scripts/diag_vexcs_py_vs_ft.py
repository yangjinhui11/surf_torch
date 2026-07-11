"""VEXCS head-to-head: PyTorch port vs Fortran kernel on identical inputs.

The step-0 diagnostic showed that, with veg_type=3 (z0m=2.0 m) on both
sides, the PyTorch VEXCS port produces rhoc_u ~0.90 while the coupled
Fortran run (same veg type) ends up at Δskin=-6.04 K (implying a much
smaller exchange).  This script isolates ONE layer: it feeds byte-
identical surface-layer inputs to BOTH surf_pytorch.physics.exchange.vexcs
(Python) and offline_driver.surf_ref.SurfRef.vexcs (Fortran kernel) and
prints the resulting (rhoc_u, rhoc_q) and z/L.

If they agree, the VEXCS port is faithful and the gap is elsewhere
(stomata / vlamsk / dqsdT / net-LW convention).  If they disagree, the
port itself has a bug (most likely in the stability-function form, the
roughness-length geometry, or the Richardson-number initial guess).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from surf_pytorch import SurfConfig  # noqa: E402
from surf_pytorch.constants import PhysicalConstants, ConstantBundle  # noqa: E402
from surf_pytorch.tables.veg import VegTable  # noqa: E402
from surf_pytorch.tables.ritbl import RitblTable  # noqa: E402
from surf_pytorch.physics import exchange as exch_mod  # noqa: E402
from surf_pytorch.forcing import build_idealized_diurnal, IdealizedDiurnalConfig  # noqa: E402

try:
    from offline_driver.surf_ref import SurfRef
    HAVE_FT = SurfRef.is_available()
except Exception as _e:
    SurfRef = None
    HAVE_FT = False
    print(f"[warn] Fortran reference unavailable: {_e}")


def _matching_config() -> IdealizedDiurnalConfig:
    return IdealizedDiurnalConfig(
        n_grid=1, dt_seconds=3600.0, n_steps=48,
        lat_deg=45.0, lon_deg=0.0, day_of_year=172,
        tair_mean=290.0, tair_amp=8.0, wind_speed=5.0, rh=0.7,
        precip_events=[(10, 14, 1e-4)],
        soil_t_init=285.0, soil_w_init=0.30, skin_t_init=288.0, sst_init=290.0,
    )


def main() -> int:
    dtype = torch.float64
    device = torch.device("cpu")
    consts = ConstantBundle(PhysicalConstants(), dtype=dtype, device=device)
    veg_table = VegTable(dtype=dtype, device=device)
    ritbl = RitblTable(dtype=dtype, device=device)

    cfg = _matching_config()
    forcing_list, static, init_state = build_idealized_diurnal(cfg)
    f = forcing_list[0]
    ngrid = f.t.shape[0]
    cp = consts.cpd

    # z0 from the veg table (veg_type=3 -> z0m=2.0, z0h=z0m/1=2.0, z0q=z0h).
    z0m = veg_table.gather("rvz0m", static.veg_type).to(dtype).to(device)
    z0h = veg_table.gather("rvz0h", static.veg_type).to(dtype).to(device)
    z0q = z0h.clone()

    # Surface-layer inputs VEXCS consumes (gridpoint values).
    # Skin gridpoint mean (tile-frac weighted): (Ngrid,)
    frac = static.tile_frac_base
    skin_grid = ((init_state.skin_t * frac).sum(-1) / frac.sum(-1)).contiguous()
    pcpts = cp * skin_grid
    from surf_pytorch.utils import qsat, scatter_to_tiles
    qskin = qsat(init_state.skin_t,
                 f.p_surf.unsqueeze(-1).expand(ngrid, 8), consts)
    pqsam = scatter_to_tiles(qskin, frac)

    pcptgzlev = cp * f.t + f.geom1
    zero = torch.zeros(ngrid, dtype=dtype, device=device)
    buom = zero.clone()

    common = dict(
        pu=f.u, pv=f.v, pt=f.t, pq=f.q,
        paphms=f.p_surf, pgeomlev=f.geom1,
        pcptgzlev=pcptgzlev, pcpts=pcpts, pqsam=pqsam,
        pz0mm=z0m, pz0hm=z0h, pz0qm=z0q,
        pbuom=buom, pucurr=zero, pvcurr=zero, pi10fgcv=zero,
    )

    print("=" * 72)
    print("VEXCS head-to-head: PyTorch port vs Fortran kernel")
    print("=" * 72)
    print(f"inputs: u={f.u[0].item():.3f} v={f.v[0].item():.3f} "
          f"|U|={torch.hypot(f.u,f.v)[0].item():.3f} m/s")
    print(f"        t_lev={f.t[0].item():.3f} K  q_lev={f.q[0].item():.4f}")
    print(f"        p_surf={f.p_surf[0].item():.1f} Pa  geom1={f.geom1[0].item():.2f}")
    print(f"        skin(grid)={skin_grid[0].item():.3f} K")
    print(f"        z0m={z0m[0].item():.4f} m  z0h={z0h[0].item():.4f}  z0q={z0q[0].item():.4f}")
    print()

    # --- PyTorch VEXCS ---------------------------------------------------- #
    out_py = exch_mod.vexcs(
        pzdl0=None, consts=consts, ritbl=ritbl, kitt=3, ldinit=True, **common,
    )

    # --- Fortran VEXCS ---------------------------------------------------- #
    out_ft = None
    if HAVE_FT:
        ref = SurfRef()
        out_ft = ref.vexcs(consts=consts, pzdl0=None, kitt=3, ldinit=True,
                           **common)

    def _show(label, out):
        if out is None:
            print(f"  {label:<14} [unavailable]")
            return
        print(f"  {label:<14} rhoc_u(pcfh)={out.pcfh[0].item():+.5f}  "
              f"rhoc_q(pcfq)={out.pcfq[0].item():+.5f}  "
              f"rhoc_m(pcfm)={out.pcfm[0].item():+.5f}")
        print(f"  {'':<14} z/L(pzdl)   ={out.pzdl[0].item():+.5f}  "
              f"kh(pkh)    ={out.pkh[0].item():+.5f} m/s")

    _show("PyTorch VEXCS", out_py)
    _show("Fortran VEXCS", out_ft)
    print()

    if out_ft is not None:
        du = abs(out_py.pcfh[0].item() - out_ft.pcfh[0].item())
        dq = abs(out_py.pcfq[0].item() - out_ft.pcfq[0].item())
        dz = abs(out_py.pzdl[0].item() - out_ft.pzdl[0].item())
        print(f"  |Δ rhoc_u| = {du:.5f}")
        print(f"  |Δ rhoc_q| = {dq:.5f}")
        print(f"  |Δ z/L   | = {dz:.5f}")
        print()
        tol = 1e-4
        if du < tol and dq < tol and dz < tol:
            print("  => VEXCS port MATCHES Fortran bit-for-bit.")
            print("     The exchange-coefficient layer is faithful; the")
            print("     skin-T gap must be in a DIFFERENT layer (vlamsk,")
            print("     dqsdT, stomatal closure, or net-LW convention).")
        else:
            print("  => VEXCS port DISAGREES with Fortran.")
            print("     This is a real porting bug in the Monin-Obukhov /")
            print("     stability-function code -- drill into the divergent")
            print("     term (z/L, PSI functions, or the Richardson loop).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
