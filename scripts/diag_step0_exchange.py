"""Single-step exchange-coefficient diagnostic (P1.6-2).

Proves (or refutes) that the ~1.4 K single-step skin-T gap between the
PyTorch port and the coupled Fortran driver is caused by the exchange-
coefficient path:

  * The PyTorch harness runs a **neutral Cd = 1e-3 placeholder** because
    :func:`build_idealized_diurnal` does not populate ``Static.z0m`` /
    ``z0h`` / ``z0q`` (model.py:510-511 falls back to
    :meth:`_neutral_exchange_coeffs`).
  * The coupled Fortran driver runs the **full Monin-Obukhov VEXCS**
    fed by vegetation-lookup roughness lengths (VUPDZ0).

This script runs ONE step on the **byte-identical step-0 inputs**
(skin 288 K, tair 282.609 K, SW=0) under three exchange-coefficient
configurations and prints, for each:

  * the resulting per-tile exchange coefficients (rho*Ch*|U|, rho*Cq*|U|)
  * the resulting gridpoint skin-T (tile-fraction-weighted mean)
  * the single-step delta and how far it is from the Fortran coupled
    reference (o_gg[1].tskin = 281.962 K  =>  Δskin = -6.038 K)

The three configurations are:

  A. **PyTorch / Cd=1e-3** (current harness default; ``static.z0m=None``)
  B. **PyTorch / VEXCS**    (z0 populated from the VegTable lookup)
  C. **Fortran / VEXCS**    (offline_driver.surf_ref full single step)

If the hypothesis holds, B and C should agree closely (both run VEXCS)
and A should be the outlier that produces the documented Δskin = -4.55 K.

Usage
-----
    cd surf_pytorch
    python scripts/diag_step0_exchange.py
    OIFS_OGG_PATH=/path/to/o_gg python scripts/diag_step0_exchange.py
        # (OIFS_OGG_PATH only changes the printed reference delta header)
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import torch

# Make the package importable when run as a loose script from surf_pytorch/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from surf_pytorch import LandSurface, SurfConfig, Static, SurfState  # noqa: E402
from surf_pytorch.constants import PhysicalConstants, ConstantBundle  # noqa: E402
from surf_pytorch.forcing import build_idealized_diurnal, IdealizedDiurnalConfig  # noqa: E402
from surf_pytorch.tables.veg import VegTable  # noqa: E402
from surf_pytorch.physics.diffusion import RVDIFTS  # noqa: E402

# Fortran authoritative single-step reference (libsurf_ref.so).
try:
    from offline_driver.surf_ref import SurfRef
    _HAVE_FORTRAN_REF = SurfRef.is_available()
except Exception as _e:  # pragma: no cover - env-dependent
    SurfRef = None
    _HAVE_FORTRAN_REF = False
    print(f"[warn] Fortran reference unavailable: {_e}")


# --------------------------------------------------------------------------- #
# Reference numbers (from the coupled run's o_gg, see test_offline_coupled).
# --------------------------------------------------------------------------- #
STEP0_INPUT_SKIN = 288.0
STEP0_OGG_OUT_SKIN = 281.962      # => Fortran single-step Δskin = -6.038 K
FORTRAN_DELTA = STEP0_OGG_OUT_SKIN - STEP0_INPUT_SKIN    # -6.038
PYTORCH_NEUTRAL_DELTA_OBSERVED = -4.55   # documented in test_offline_coupled


def _matching_config() -> IdealizedDiurnalConfig:
    """The IdealizedDiurnalConfig the coupled run-dir is generated from.

    Mirrors tests/test_offline_coupled._matching_config.
    """
    return IdealizedDiurnalConfig(
        n_grid=1,
        dt_seconds=3600.0,
        n_steps=48,
        lat_deg=45.0,
        lon_deg=0.0,
        day_of_year=172,
        tair_mean=290.0,
        tair_amp=8.0,
        wind_speed=5.0,
        rh=0.7,
        precip_events=[(10, 14, 1e-4)],
        soil_t_init=285.0,
        soil_w_init=0.30,
        skin_t_init=288.0,
        sst_init=290.0,
    )


def _with_z0(static: Static, veg_table: VegTable, dtype, device) -> Static:
    """Return a copy of *static* with z0m/z0h/z0q populated from the VegTable.

    Mirrors the Fortran VUPDZ0 land rule PZ0QTI = PZ0HTI over land
    (vupdz0_mod.F90:378).  z0m/z0h come from the per-type lookup; z0q is
    set equal to z0h for every land tile (the offline run is pure land).
    """
    z0m = veg_table.gather("rvz0m", static.veg_type).to(dtype).to(device)
    z0h = veg_table.gather("rvz0h", static.veg_type).to(dtype).to(device)
    z0q = z0h.clone()
    return replace(static, z0m=z0m, z0h=z0h, z0q=z0q)


def _gridpoint_mean(fld: torch.Tensor, frac: torch.Tensor) -> float:
    """Tile-fraction-weighted gridpoint mean of a per-tile field (cell 0).

    Handles both 2-D ``(Ngrid, Ntile)`` and 1-D ``(Ngrid,)`` inputs.
    """
    a = fld.detach().cpu().numpy()
    fr = frac.detach().cpu().numpy()
    if a.ndim == 2:
        gm = (a * fr).sum(axis=-1) / fr.sum(axis=-1)   # (Ngrid,)
        return float(np.asarray(gm).reshape(-1)[0])
    return float(np.asarray(a).reshape(-1)[0])


# Back-compat alias used in a couple of places below.
_gridpoint_mean_skin = _gridpoint_mean


def _run_pytorch_step(static: Static, label: str) -> dict:
    """Run one LandSurface.forward step; return skin/flux diagnostics."""
    cfg = _matching_config()
    forcing_list, _static_base, init_state = build_idealized_diurnal(cfg)
    forcing = forcing_list[0]
    # Use the caller's static (with or without z0) but the shared forcing
    # and cold-start state so the only variable is the exchange path.
    state = init_state

    dtype = torch.float64
    device = torch.device("cpu")
    surf_cfg = SurfConfig(dtype=dtype, device=device)
    model = LandSurface(surf_cfg, consts=PhysicalConstants())
    veg_table = model.veg_table

    # Probe the raw exchange coefficients this config produces (cell 0).
    rhoc_u, rhoc_q = model._exchange_coeffs(forcing, static, state)
    ru = float(rhoc_u.reshape(-1)[0])
    rq = float(rhoc_q.reshape(-1)[0])

    with torch.no_grad():
        new_state, diag = model(state, forcing, static)

    skin_out = new_state.skin_t
    frac = static.tile_frac_base
    gm_in = _gridpoint_mean(state.skin_t, frac)
    gm_out = _gridpoint_mean(skin_out, frac)

    def _gm(fld):
        return _gridpoint_mean(fld, frac) if fld is not None else float("nan")

    return {
        "label": label,
        "exchange_path": "VEXCS" if static.z0m is not None else "Cd=1e-3",
        "rhoc_u": ru,
        "rhoc_q": rq,
        "skin_in": gm_in,
        "skin_out": gm_out,
        "delta": gm_out - gm_in,
        "gap_vs_fortran": (gm_out - gm_in) - FORTRAN_DELTA,
        "h_flux": _gm(getattr(diag, "h_flux", None)),
        "e_flux": _gm(getattr(diag, "e_flux", None)),
        "str": _gm(getattr(diag, "net_rad", None)),
        "g0": _gm(getattr(diag, "ground_heat", None)),
    }


def _print_row(r: dict) -> None:
    rhoc = r.get("rhoc_u")
    rhoc_s = f"rhoc_u={rhoc:+8.4f} rhoc_q={r.get('rhoc_q', float('nan')):+8.4f}  " \
        if rhoc is not None else ""
    print(f"  {r['label']:<28} {r['exchange_path']:<8} "
          f"Δskin={r['delta']:+7.3f}  gap_vs_F={r['gap_vs_fortran']:+7.3f}  "
          f"H={r['h_flux']:+8.3f} LE={r['e_flux']:+8.3f} "
          f"STR={r['str']:+8.3f} G0={r['g0']:+8.3f}")
    if rhoc_s:
        print(f"  {'':<28} {'':<8} {rhoc_s}")


def main() -> int:
    ogg = os.environ.get("OIFS_OGG_PATH", "")
    print("=" * 78)
    print("P1.6-2: single-step exchange-coefficient diagnostic")
    print("=" * 78)
    print(f"step-0 input skin   : {STEP0_INPUT_SKIN:.3f} K")
    print(f"Fortran o_gg Δskin  : {FORTRAN_DELTA:+.3f} K  "
          f"(-> skin {STEP0_OGG_OUT_SKIN:.3f} K)"
          + (f"  [ref: {ogg}]" if ogg else ""))
    print(f"PyTorch (Cd=1e-3) Δ : {PYTORCH_NEUTRAL_DELTA_OBSERVED:+.3f} K  "
          f"(documented gap)")
    print()

    dtype = torch.float64
    device = torch.device("cpu")

    cfg = _matching_config()
    _forcing_list, static_neutral, _init = build_idealized_diurnal(cfg)
    # Reuse the model's own veg table (guaranteed same buffers the forward
    # uses) for the z0 lookup.
    _m = LandSurface(SurfConfig(dtype=dtype, device=device), consts=PhysicalConstants())
    veg_table = _m.veg_table
    static_vexcs = _with_z0(static_neutral, veg_table, dtype, device)

    print("roughness lengths (grid cell 0):")
    print(f"  z0m = {static_vexcs.z0m[0].item():.6f} m")
    print(f"  z0h = {static_vexcs.z0h[0].item():.6f} m  (z0q = z0h)")
    print()

    # --- A & B: PyTorch port, neutral vs VEXCS ---------------------------- #
    print("PyTorch port (LandSurface.forward), step 0:")
    res_neutral = _run_pytorch_step(static_neutral, "A. PyTorch/Cd=1e-3")
    res_vexcs = _run_pytorch_step(static_vexcs, "B. PyTorch/VEXCS")
    _print_row(res_neutral)
    _print_row(res_vexcs)
    print()

    # --- C: Fortran authoritative single step (if available) -------------- #
    print("Fortran reference (SurfRef.vdfdifh, full VEXCS path), step 0:")
    if _HAVE_FORTRAN_REF:
        res_fortran = _run_fortran_step(static_vexcs, veg_table)
        if res_fortran is not None:
            _print_row(res_fortran)
        else:
            print("  [Fortran step failed -- see traceback above]")
    else:
        print("  [libsurf_ref.so not available -- skipping column C]")
        res_fortran = None
    print()

    # --- verdict ---------------------------------------------------------- #
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    _verdict(res_neutral, res_vexcs, res_fortran)
    return 0


def _run_fortran_step(static: Static, veg_table: VegTable):
    """Run the authoritative Fortran single step via SurfRef.vdfdifh.

    Returns a dict with the same keys as _run_pytorch_step, or None on
    failure.  This exercises the real VEXCS (VUPDZ0 + VEXCS) path that
    the coupled driver uses.
    """
    import traceback
    cfg = _matching_config()
    forcing_list, _static_base, init_state = build_idealized_diurnal(cfg)
    forcing = forcing_list[0]
    try:
        ref = SurfRef()
        # Build the DiffusionInputs exactly as the model would, but hand
        # the exchange coefficients to the Fortran SURFSEB leg.  The
        # simplest faithful call is vdfdifh(), which mirrors vdfdifh1s.F90.
        from surf_pytorch.constants import PhysicalConstants, ConstantBundle
        dtype = torch.float64
        device = torch.device("cpu")
        consts = ConstantBundle(PhysicalConstants(), dtype=dtype, device=device)
        cp = consts.cpd

        # Lowest-level dry static energy / humidity.
        sl_lev = cp * forcing.t + forcing.geom1
        q_lev = forcing.q

        # rho*Ch*|U|, rho*Cq*|U| via the Fortran VEXCS (z0 from the veg
        # table, matching what VUPDZ0 would produce for a pure-land cell).
        ngrid = forcing.t.shape[0]
        z0m = static.z0m.to(dtype).to(device)
        z0h = static.z0h.to(dtype).to(device)
        z0q = static.z0q.to(dtype).to(device)

        # Surface dry static energy and qsat (gridpoint, tile-frac weighted).
        frac = static.tile_frac_base
        from surf_pytorch.utils import qsat, scatter_to_tiles
        tskin_grid = scatter_to_tiles(init_state.skin_t, frac)
        pcpts = cp * tskin_grid
        pqsam = scatter_to_tiles(
            qsat(init_state.skin_t,
                 forcing.p_surf.expand(ngrid, 8) if forcing.p_surf.dim() == 1
                 else forcing.p_surf, consts), frac)

        pcptgzlev = cp * forcing.t + forcing.geom1
        buom = torch.zeros(ngrid, dtype=dtype, device=device)
        zero = torch.zeros(ngrid, dtype=dtype, device=device)

        from surf_pytorch.physics import exchange as exch_mod
        out = ref.vexcs(
            forcing.u, forcing.v, forcing.t, forcing.q,
            forcing.p_surf, forcing.geom1, pcptgzlev, pcpts, pqsam,
            z0m, z0h, z0q, buom, zero, zero, zero, consts,
        )
        rhoc_u = out.pcfh
        rhoc_q = out.pcfq

        # Net radiation inputs: SW down (rate), net LW (rate), emissivity.
        dt = float(forcing.dt.item())
        ssrfl_grid = forcing.fr_so / dt
        slrfl_grid = forcing.fr_th / dt
        emis = static.emis if static.emis is not None else \
            torch.full((ngrid,), 0.99, dtype=dtype, device=device)
        tskrad = tskin_grid

        # Sub-surface T (top soil layer, broadcast -- pure land).
        tsrf = init_state.soil_t[:, 0].expand(ngrid, 8).clone()

        # VLAMSK via Fortran too, for fidelity.
        vlamsk_out = ref.vlamsk(
            init_state.skin_t, tsrf, init_state.snow_w,
            torch.full((ngrid,), 300.0, dtype=dtype, device=device),
            None, static.veg_type, static.veg_type, veg_table, consts,
            ptstep=torch.full((ngrid,), dt, dtype=dtype, device=device),
        )
        plamsk = vlamsk_out.plamsk

        # alphas / alphal via Fortran vevap path (use alphas=alphal=1.0 as a
        # neutral first cut so the SEB leg is the only variable; the
        # skin-T result is dominated by the exchange coeffs + radiation,
        # not the stomatal closure, for a dry night-time step).
        alphas = torch.ones((ngrid, 8), dtype=dtype, device=device)
        alphal = torch.ones((ngrid, 8), dtype=dtype, device=device)

        from surf_pytorch.utils import dqsat_dt_ifs
        qskm1m = qsat(init_state.skin_t,
                      forcing.p_surf.expand(ngrid, 8)
                      if forcing.p_surf.dim() == 1 else forcing.p_surf, consts)
        dqsdT = dqsat_dt_ifs(init_state.skin_t,
                             forcing.p_surf.expand(ngrid, 8)
                             if forcing.p_surf.dim() == 1 else forcing.p_surf,
                             consts)
        sskm1m = cp * init_state.skin_t

        # Call Fortran SURFSEB via the vdfdifh wrapper.
        # rescale rho*Ch*|U| -> PCFHTI = rho*Ch*|U| * RG*dt*RVDIFTS  (model.py)
        from surf_pytorch.physics.diffusion import RVDIFTS
        rg = float(consts.g.item())
        rscale = rg * dt * RVDIFTS
        pcfhti = rhoc_u * rscale
        pcfqti = rhoc_q * rscale

        seb_out = ref.seb(
            sskm1m, init_state.skin_t, qskm1m, dqsdT,
            pcfhti.expand(ngrid, 8), pcfqti.expand(ngrid, 8),
            alphal, alphas,
            ssrfl_grid.expand(ngrid, 8), slrfl_grid.expand(ngrid, 8),
            tskrad.expand(ngrid, 8), emis,
            tsrf, plamsk, static.tile_frac_base,
            static.veg_type, static.veg_type,
            sl_lev.expand(ngrid, 8), q_lev.expand(ngrid, 8),
            dt, consts,
        )
        skin_out = seb_out.ptsk
        # Extrapolate to t+1 exactly as the port does (model.py:474).
        ztpfac2 = 1.0 / RVDIFTS
        ztpfac3 = 1.0 - ztpfac2
        skin_tip1 = ztpfac2 * skin_out + ztpfac3 * init_state.skin_t

        gm_in = _gridpoint_mean_skin(init_state.skin_t, frac)
        gm_out = _gridpoint_mean_skin(skin_tip1, frac)

        def _gm(fld):
            a = fld.detach().cpu().numpy()
            fr = frac.detach().cpu().numpy()
            if a.ndim == 2:
                return float((a * fr).sum(axis=-1) / fr.sum(axis=-1))[0]
            return float(a.sum(axis=-1)[0])

        return {
            "label": "C. Fortran/VEXCS",
            "exchange_path": "VEXCS",
            "skin_in": gm_in,
            "skin_out": gm_out,
            "delta": gm_out - gm_in,
            "gap_vs_fortran": (gm_out - gm_in) - FORTRAN_DELTA,
            "h_flux": _gm(seb_out.pssh),
            "e_flux": _gm(seb_out.pjq),
            "str": _gm(seb_out.pstr),
            "g0": _gm(seb_out.pg0),
        }
    except Exception:
        traceback.print_exc()
        return None


def _verdict(res_neutral, res_vexcs, res_fortran) -> None:
    """Print whether the exchange-coefficient hypothesis is confirmed."""
    gap_neutral = abs(res_neutral["gap_vs_fortran"])
    gap_vexcs = abs(res_vexcs["gap_vs_fortran"])
    print(f"  Cd=1e-3  gap vs Fortran : {gap_neutral:+.3f} K")
    print(f"  VEXCS    gap vs Fortran : {gap_vexcs:+.3f} K")
    if res_fortran is not None:
        print(f"  Fortran  own gap        : {res_fortran['gap_vs_fortran']:+.3f} K")
    print()
    if gap_vexcs < gap_neutral * 0.5:
        print("  => CONFIRMED: populating z0 (running VEXCS) closes most of the")
        print("     single-step gap.  The exchange-coefficient path is the root")
        print("     cause of the ~1.4 K/step skin-T drift.")
        print("     FIX: populate Static.z0m/z0h/z0q in the forcing generator")
        print("     (forcing.py:_build_static_and_state) so the harness runs")
        print("     VEXCS instead of the Cd=1e-3 placeholder.")
    elif gap_vexcs < 1.0:
        print("  => PARTIAL: VEXCS helps but a residual gap remains.")
        print("     Other suspects (in order): net-LW convention, stomatal")
        print("     closure (alphas/alphal), vlamsk, dqsdT form.  Drill into")
        print("     the per-tile flux budget next.")
    else:
        print("  => REFUTED: exchange-coefficient path is NOT the main cause.")
        print("     Re-open the investigation; the residual gap in column B is")
        print("     still large.")


if __name__ == "__main__":
    raise SystemExit(main())
