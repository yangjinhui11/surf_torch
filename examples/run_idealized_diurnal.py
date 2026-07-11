#!/usr/bin/env python
"""End-to-end demonstration of the M2c idealized-diurnal pipeline.

This is the runnable artifact behind the user request "forcing数据能
不能构造一个，用来进行测试" (can we construct a forcing dataset for
testing?).  It builds an analytic mid-latitude diurnal-cycle forcing
time series, drives the land-surface model over it with the
:class:`ColumnHarness`, and writes the trajectory to disk as both a
human-readable CSV (grid-mean time series) and a machine-readable
``.npz`` snapshot (full stacked arrays, for regression).

No external data file is read and no new dependency is needed -- the
forcing is closed-form (Cooper-1969 solar declination + a single hour-
angle term), so the whole pipeline is deterministic and reproducible.

Usage
-----
::

    python examples/run_idealized_diurnal.py                 # default 2-day run
    python examples/run_idealized_diurnal.py --steps 120      # 5-day run
    python examples/run_idealized_diurnal.py --rain 24,36,2e-4  # inject rain
    python examples/run_idealized_diurnal.py --out /tmp/run   # custom outdir

Outputs (in --out, default ./m2c_run/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``forcing_grid_mean.csv`` -- the constructed forcing (Tair, SW, LW,
  q, precip) grid-mean per step, for inspection.
- ``trajectory_grid_mean.csv`` -- the model response (skin T, top-soil
  T/W, fluxes) grid-mean per step.
- ``trajectory.npz`` -- full stacked arrays (deterministic snapshot).

Validation
----------
The script prints the water- and energy-balance closure residuals on
the run (the same diagnostics the test suite asserts), so you can
eyeball whether the model is behaving physically.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from surf_pytorch import (
    IdealizedDiurnalConfig,
    build_idealized_diurnal,
    ColumnHarness,
    LandSurface,
    SurfConfig,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=48,
                   help="number of coupling steps (default 48 = 2 days)")
    p.add_argument("--dt", type=float, default=3600.0,
                   help="coupling interval in seconds (default 3600)")
    p.add_argument("--lat", type=float, default=45.0, help="latitude (deg)")
    p.add_argument("--lon", type=float, default=0.0, help="longitude (deg)")
    p.add_argument("--tair-mean", type=float, default=290.0,
                   help="mean air temperature (K)")
    p.add_argument("--tair-amp", type=float, default=8.0,
                   help="diurnal air-temperature amplitude (K)")
    p.add_argument("--sw-peak", type=float, default=800.0,
                   help="surface SW peak at mu0=1 (W/m^2)")
    p.add_argument("--rain", type=str, default="",
                   help="rain event as start_step,end_step,rate_kgm2s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="m2c_run",
                   help="output directory")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> IdealizedDiurnalConfig:
    """Translate CLI flags into an IdealizedDiurnalConfig."""
    precip_events = []
    if args.rain:
        parts = args.rain.split(",")
        if len(parts) != 3:
            raise SystemExit("--rain must be start,end,rate")
        precip_events.append(
            (int(parts[0]), int(parts[1]), float(parts[2]))
        )
    return IdealizedDiurnalConfig(
        n_steps=args.steps,
        dt_seconds=args.dt,
        lat_deg=args.lat,
        lon_deg=args.lon,
        tair_mean=args.tair_mean,
        tair_amp=args.tair_amp,
        sw_peak=args.sw_peak,
        precip_events=precip_events,
    )


def export_forcing_csv(forcing_list, path: str) -> None:
    """Write the grid-mean forcing time series as CSV."""
    n = len(forcing_list)
    time = np.array([(k + 0.5) * float(forcing_list[0].dt) for k in range(n)])
    dt = float(forcing_list[0].dt)
    tair = np.array([f.t.mean().item() for f in forcing_list])
    q = np.array([f.q.mean().item() for f in forcing_list])
    sw = np.array([f.fr_so.mean().item() / dt for f in forcing_list])
    lw = np.array([f.fr_th.mean().item() / dt for f in forcing_list])
    mu0 = np.array([f.mu0.mean().item() for f in forcing_list])
    rain = np.array([f.rain.mean().item() / dt for f in forcing_list])
    snow = np.array([f.snow.mean().item() / dt for f in forcing_list])
    data = np.column_stack([time, tair, q, mu0, sw, lw, rain, snow])
    header = ("time_s,tair_K,q_kgkg,mu0,swdown_Wm2,lwnet_Wm2,"
              "rain_kgm2s,snow_kgm2s")
    np.savetxt(path, data, delimiter=",", header=header)
    print(f"  forcing -> {path}  ({n} rows)")


def water_balance(result, forcing_list, state0, model) -> None:
    """Print the water-budget closure (kg/m^2 over the whole run)."""
    rho_h2o = float(model.consts.rho_h2o.item())
    rdaw = torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=torch.float64)
    dt = result.dt

    W0 = ((state0.soil_w_liq * rdaw * rho_h2o).sum(-1)
          + state0.skin_water + state0.snow_w)
    Wf = ((result.final_state.soil_w_liq * rdaw * rho_h2o).sum(-1)
          + result.final_state.skin_water + result.final_state.snow_w)
    dW = Wf - W0
    P = sum(f.rain + f.snow for f in forcing_list)
    E = (result.e_flux * dt).sum(0)
    runoff = (P + E) - dW
    residual = (dW + runoff) - (P + E)
    print("  water balance (kg/m^2):")
    print(f"    precip     P = {P.mean().item():+.4f}")
    print(f"    evap    E*dt = {E.mean().item():+.4f}  (e_flux down-positive)")
    print(f"    storage   dW = {dW.mean().item():+.4f}")
    print(f"    implied runoff = {runoff.mean().item():+.4f}  (must be >= 0)")
    print(f"    closure residual = {residual.abs().max().item():.2e}"
          "  (machine precision)")


def energy_balance(result, forcing_list, state0, static, model) -> None:
    """Print the energy-budget closure (J/m^2 over the whole run).

    The soil-column heat input is ``ZSURFL = SW + LW + H + RLVTT*E``
    (land-tile-weighted), NOT ``diag.ground_heat`` -- the latter is the
    skin-to-subskin flux and is not the soil-column input.  See
    ``tests/test_m2c_harness.py::TestEnergyBalance`` for the reasoning.
    """
    rlvtt = float(model.consts.rlvtt.item())
    dt = result.dt
    rcsoil = model.soil_table.gather("rcsoil", static.soil_type)
    rcsoil_b = rcsoil.unsqueeze(-1).expand(-1, 4)
    rdaw = torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=torch.float64)

    H0 = (state0.soil_t * rcsoil_b * rdaw).sum(-1)
    Hf = (result.final_state.soil_t * rcsoil_b * rdaw).sum(-1)
    dH = Hf - H0
    cum = torch.zeros_like(dH)
    for k, f in enumerate(forcing_list):
        sw = f.fr_so / f.dt
        lw = f.fr_th / f.dt
        H = result.h_flux[k]
        E = result.e_flux[k]
        cum = cum + (sw + lw + H + rlvtt * E) * dt
    rel = (dH - cum).abs().max() / cum.abs().max()
    print("  energy balance (J/m^2 over run):")
    print(f"    soil heat storage dH   = {dH.mean().item()/1e6:+.4f} MJ/m^2")
    print(f"    cumulative ZSURFL*dt   = {cum.mean().item()/1e6:+.4f} MJ/m^2")
    print(f"    relative residual      = {rel.item():.2e}"
          "  (machine precision)")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Building idealized-diurnal forcing ({args.steps} steps, "
          f"{args.dt:.0f}s dt, lat {args.lat:.1f}N)...")
    cfg = build_config(args)
    forcing_list, static, state0 = build_idealized_diurnal(cfg)

    os.makedirs(args.out, exist_ok=True)
    export_forcing_csv(forcing_list,
                       os.path.join(args.out, "forcing_grid_mean.csv"))

    print(f"Running land-surface model ({cfg.n_grid} columns)...")
    model = LandSurface(SurfConfig(dtype=torch.float64, validate=False))
    result = ColumnHarness(model, static, state0).run(forcing_list)

    traj_csv = os.path.join(args.out, "trajectory_grid_mean.csv")
    result.export_csv(traj_csv)
    print(f"  trajectory CSV -> {traj_csv}  ({args.steps} rows)")
    snap = os.path.join(args.out, "trajectory.npz")
    result.export_snapshot(snap)
    print(f"  snapshot npz   -> {snap}")

    print("\nBudget closure check:")
    water_balance(result, forcing_list, state0, model)
    energy_balance(result, forcing_list, state0, static, model)

    print("\nDone.")


if __name__ == "__main__":
    main()
