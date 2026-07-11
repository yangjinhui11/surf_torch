#!/usr/bin/env python
"""Run the surf_pytorch land-surface model driven by **real** ecLand/ERA5
site forcing.

This is the runnable counterpart to ``tests/test_ecland_driven.py``: it
loads the three NetCDF files for a site (ERA5-Hourly-Tiled forcing +
surfclim static descriptor + surfinit initial state), drives the full
``LandSurface.forward`` over every forcing step, and writes the
per-step trajectory to a CSV.  It also prints the surface energy and
water budget closures so you can eyeball physical health without
assertions.

Usage
-----
::

    # default site CA-001 (Cabauw winter), data auto-resolved from the
    # thirdpaty/ecland-main tutorial tree
    python examples/run_ecland.py

    # pick another bundled site
    python examples/run_ecland.py --site VG-000

    # point at your own site data (three NetCDFs you built yourself)
    python examples/run_ecland.py \
        --met  /path/to/met_era5HT_MY-SITE_2022-2022.nc \
        --clim /path/to/surfclim_MY-SITE_2022-2022.nc \
        --init /path/to/surfinit_MY-SITE_2022-2022.nc

    # Farquhar photosynthesis closure instead of Jarvis
    python examples/run_ecland.py --farquhar

    # custom output directory
    python examples/run_ecland.py --out /tmp/ecland_run

Outputs (in --out, default ./ecland_run/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``trajectory.csv`` -- per-step grid-mean time series: skin-T, soil-T
  (4 layers), soil moisture (4 layers), the four surface fluxes
  (net LW, sensible H, latent LE, ground G), 2 m T/Q, 10 m wind, net
  radiation, plus the forcing itself for context.
- ``forcing_grid_mean.csv`` -- the input forcing as seen by the model
  (after the rate->cumulative conversion), for cross-checking units.

How the NetCDF -> model mapping works
-------------------------------------
The unit/sign conventions are reconciled exactly as in
``tests/test_ecland_driven.py``:

* ERA5/ecLand radiation and precip are **rates** (W/m^2, kg/m^2/s);
  they are multiplied by ``dt`` (default 3600 s) to give the
  cumulative fluxes the port's ``Forcing`` expects.
* ``LWdown`` is gross downwelling; the net LW sent to the model is
  reconstructed as ``LWdown - emis*sigma*Tskin^4``.
* ecLand vegetation / soil type indices are **1-based** and consumed
  directly (no -1 offset).
* The eight base tiles are allocated from the aggregate ``cvl`` /
  ``cvh`` / ``cu`` / ``landsea`` cover fields (a cold-start split;
  water/ice/snow tiles are zero on these land sites).

See the repository README section *Real-data & coupled validation ->
ecLand ERA5 site forcing* for how to build these NetCDFs for a new
site.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# xarray is the only extra dependency; fail with a clear message.
try:
    import xarray
except ImportError:
    sys.exit(
        "xarray is required to read ecLand NetCDF forcing. "
        "Install it with:  pip install xarray netcdf4"
    )

import torch

from surf_pytorch.constants import ConstantBundle, PhysicalConstants
from surf_pytorch.config import SurfConfig, N_BASE_TILES
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.forcing import solar_mu0
from surf_pytorch.model import LandSurface


# --------------------------------------------------------------------------- #
# Default data-root resolution (mirrors tests/test_ecland_driven.py)
# --------------------------------------------------------------------------- #
# Walk up from this examples/ file to the repo root, then into the ecland
# tutorial tree.  This file lives at <repo>/surf_pytorch/examples/run_ecland.py,
# so parents[2] is the repo root and the data resolves to
# <repo>/thirdpaty/ecland-main/tutorials/ecland_input_tutorial
# (the test suite resolves the same path from tests/test_ecland_driven.py).
def _default_ecland_root() -> Path:
    here = Path(__file__).resolve()
    repo = here.parents[2]                              # <repo>
    return repo / "thirdpaty" / "ecland-main" / "tutorials" / "ecland_input_tutorial"


# --------------------------------------------------------------------------- #
# Site loading: NetCDF -> raw arrays  (port of test_ecland_driven._load_site)
# --------------------------------------------------------------------------- #
def load_site(met_path: Path, clim_path: Path,
              init_path: Path) -> Dict[str, object]:
    """Open the three ecLand NetCDFs and return raw arrays.

    Returns a dict with keys ``forcing`` (dict of 1-D rate arrays),
    ``clim`` (static scalars), ``init`` (initial-state arrays),
    ``time`` (datetime64 mid-step stamps), ``lat``/``lon``.
    """
    met = xarray.open_dataset(met_path)
    cli = xarray.open_dataset(clim_path)
    ini = xarray.open_dataset(init_path)

    def _squeeze(arr):
        return np.asarray(arr.values).ravel()

    forcing = {v: _squeeze(met[v]) for v in
               ["Tair", "Qair", "Wind_E", "Wind_N", "PSurf",
                "SWdown", "LWdown", "Rainf", "Snowf"]}
    time = met["time"].values if "time" in met else None

    lat = float(np.asarray(cli["lat"].values).ravel()[0])
    lon = float(np.asarray(cli["lon"].values).ravel()[0])
    clim = {
        "lat": lat, "lon": lon,
        "tvl": int(np.asarray(cli["tvl"].values).ravel()[0]),
        "tvh": int(np.asarray(cli["tvh"].values).ravel()[0]),
        "sotype": int(np.asarray(cli["sotype"].values).ravel()[0]),
        "cvl": float(np.asarray(cli["cvl"].values).ravel()[0]),
        "cvh": float(np.asarray(cli["cvh"].values).ravel()[0]),
        "cu": float(np.asarray(cli["cu"].values).ravel()[0]),
        "landsea": float(np.asarray(cli["landsea"].values).ravel()[0]),
        "geopot": float(np.asarray(cli["geopot"].values).ravel()[0]),
        "z0m": float(np.asarray(cli["z0m"].values).ravel()[0]),
        "lz0h": float(np.asarray(cli["lz0h"].values).ravel()[0]),
        "zuv": float(np.asarray(cli["zuv"].values).ravel()[0]),
        "zphista": float(np.asarray(cli["zphista"].values).ravel()[0]),
    }
    init = {
        "SoilTemp": np.asarray(ini["SoilTemp"].values).ravel(),
        "SoilMoist": np.asarray(ini["SoilMoist"].values).ravel(),
        "AvgSurfT": float(np.asarray(ini["AvgSurfT"].values).ravel()[0]),
        "CanopInt": float(np.asarray(ini["CanopInt"].values).ravel()[0]),
        "SWE": float(np.asarray(ini["SWE"].values).ravel()[0]),
    }
    return {"forcing": forcing, "clim": clim, "init": init,
            "time": time, "lat": lat, "lon": lon}


# --------------------------------------------------------------------------- #
# Mapping: ecLand raw -> surf_pytorch Forcing / Static / SurfState
#           (port of test_ecland_driven._to_inputs / _tile_fractions)
# --------------------------------------------------------------------------- #
_DT_SECONDS = 3600.0            # ecLand tutorial forcing is hourly
_CO2 = 6.0e-4                   # kg/kg (absent from met files)
_WET_SKIN_FRAC = 0.05           # interception-reservoir tile fraction


def _tile_fractions(cvl: float, cvh: float, cu: float,
                    landsea: float, ntile: int = N_BASE_TILES) -> np.ndarray:
    """Allocate the 8 base tile fractions from aggregate cover fields."""
    frac = np.zeros(ntile, dtype=np.float64)
    land = float(landsea)
    wet = _WET_SKIN_FRAC * land
    veg_total = max(cvl + cvh, 1e-12)
    urban = cu * land
    scale = (land - wet - urban) / veg_total if veg_total > 1e-12 else 0.0
    low = cvl * scale
    high = cvh * scale
    bare = land - wet - low - high
    frac[2] = wet
    frac[3] = low
    frac[5] = high
    frac[7] = bare
    return frac


def to_inputs(raw: Dict, consts: ConstantBundle,
              dtype: torch.dtype) -> Tuple[List[Forcing], Static, SurfState]:
    """Map ecLand raw arrays to port (forcing_list, static, state0)."""
    f = raw["forcing"]
    c = raw["clim"]
    i = raw["init"]
    n = f["Tair"].size

    g = float(consts.g.item())
    sigma = float(consts.rsigma.item())
    emis = 0.99                       # surface emissivity (idealized default)

    dt_t = torch.tensor(_DT_SECONDS, dtype=dtype)
    lat_rad = math.radians(c["lat"])
    lon_rad = math.radians(c["lon"])

    tile_frac = _tile_fractions(c["cvl"], c["cvh"], c["cu"], c["landsea"])
    static = Static(
        tile_frac_base=torch.tensor(tile_frac, dtype=dtype).unsqueeze(0),
        land_frac=torch.tensor([c["landsea"]], dtype=dtype),
        orography=torch.tensor([c["geopot"]], dtype=dtype),
        soil_type=torch.tensor([c["sotype"]], dtype=torch.long),
        veg_type=torch.tensor([c["tvl"]], dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.tensor([emis], dtype=dtype),
        z0m=torch.tensor([c["z0m"]], dtype=dtype),
        z0h=torch.tensor([math.exp(c["lz0h"])], dtype=dtype),
        z0q=torch.tensor([math.exp(c["lz0h"])], dtype=dtype),
        lat=torch.tensor([lat_rad], dtype=dtype),
        lon=torch.tensor([lon_rad], dtype=dtype),
    )

    ntile = N_BASE_TILES
    state0 = SurfState(
        skin_t=torch.full((1, ntile), i["AvgSurfT"], dtype=dtype),
        soil_t=torch.tensor(i["SoilTemp"], dtype=dtype).unsqueeze(0),
        soil_w_liq=torch.tensor(i["SoilMoist"], dtype=dtype).unsqueeze(0),
        snow_w=torch.tensor([i["SWE"]], dtype=dtype),
        snow_frac=torch.zeros(1, dtype=dtype),
        skin_water=torch.tensor([i["CanopInt"]], dtype=dtype),
        sst=torch.tensor([i["AvgSurfT"]], dtype=dtype),
    )

    forcing_list: List[Forcing] = []
    for k in range(n):
        sw_down = float(f["SWdown"][k])                       # W/m^2
        lw_down = float(f["LWdown"][k])                       # W/m^2
        lw_up = emis * sigma * i["AvgSurfT"] ** 4
        net_lw = lw_down - lw_up                              # net (down+)

        # mu0 from the calendar stamp (mid-timestep).
        tstamp = raw["time"][k]
        day_of_year = int(np.asarray(tstamp).astype("datetime64[D]")
                          .astype(int) % 365) + 1
        hour_utc = (np.asarray(tstamp).astype("datetime64[h]")
                    .astype(int) % 24) + 0.5
        mu0 = solar_mu0(lat_rad, lon_rad, day_of_year, float(hour_utc))

        forcing_list.append(Forcing(
            u=torch.tensor([float(f["Wind_E"][k])], dtype=dtype),
            v=torch.tensor([float(f["Wind_N"][k])], dtype=dtype),
            t=torch.tensor([float(f["Tair"][k])], dtype=dtype),
            q=torch.tensor([float(f["Qair"][k])], dtype=dtype),
            co2=torch.tensor([_CO2], dtype=dtype),
            p_surf=torch.tensor([float(f["PSurf"][k])], dtype=dtype),
            geom1=torch.tensor([g * c["zphista"]], dtype=dtype),
            fr_so=torch.tensor([sw_down * _DT_SECONDS], dtype=dtype),
            fr_th=torch.tensor([net_lw * _DT_SECONDS], dtype=dtype),
            mu0=torch.tensor([mu0], dtype=dtype),
            rain=torch.tensor([float(f["Rainf"][k]) * _DT_SECONDS], dtype=dtype),
            snow=torch.tensor([float(f["Snowf"][k]) * _DT_SECONDS], dtype=dtype),
            dt=dt_t.clone(),
        ))
    return forcing_list, static, state0


def grid_mean(tile_field: torch.Tensor, frac: torch.Tensor) -> float:
    """Fraction-weighted grid mean of a (Ngrid, Ntile) field."""
    return float((tile_field * frac).sum().item())


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Drive surf_pytorch with real ecLand/ERA5 site forcing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ecland_root = _default_ecland_root()

    # --- data-source arguments --------------------------------------------
    # Three modes:
    #  (a) --site <id> : resolve the three NetCDFs from the bundled ecland
    #      tutorial tree (default).
    #  (b) explicit --met/--clim/--init : point at your own site files.
    #  (c) --ecland-root <dir> : override the tutorial root, still using
    #      the standard met_era5HT_<SITE> / surfclim_<SITE> / surfinit_<SITE>
    #      naming.
    p.add_argument("--site", default="CA-001",
                   help="Site id used in the ecLand filenames "
                        "(met_era5HT_<SITE>_...).")
    p.add_argument("--ecland-root", default=str(ecland_root),
                   help="ecLand tutorial data root "
                        "(.../ecland_input_tutorial). Set this if your data "
                        "lives elsewhere but keeps the standard layout.")
    p.add_argument("--met", default=None,
                   help="Explicit path to the forcing NetCDF "
                        "(overrides --site/--ecland-root).")
    p.add_argument("--clim", default=None,
                   help="Explicit path to the surfclim NetCDF "
                        "(overrides --site/--ecland-root).")
    p.add_argument("--init", default=None,
                   help="Explicit path to the surfinit NetCDF "
                        "(overrides --site/--ecland-root).")

    # --- model options ----------------------------------------------------
    p.add_argument("--farquhar", action="store_true",
                   help="Use Farquhar + Ball-Berry closure (default Jarvis).")
    p.add_argument("--dtype", default="float64",
                   choices=["float32", "float64"],
                   help="Compute dtype.")

    # --- output -----------------------------------------------------------
    p.add_argument("--out", default="./ecland_run",
                   help="Output directory for the trajectory CSV.")
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    """Resolve the three NetCDF paths from args, with a friendly error."""
    if args.met and args.clim and args.init:
        met, clim, init = Path(args.met), Path(args.clim), Path(args.init)
    elif args.met or args.clim or args.init:
        sys.exit("If you pass --met/--clim/--init, you must pass all three.")
    else:
        root = Path(args.ecland_root)
        site = args.site
        met = root / "forcing" / "LS-PRACTICALS" / f"met_era5HT_{site}_2022-2022.nc"
        clim = root / "clim" / "LS-PRACTICALS" / f"surfclim_{site}_2022-2022.nc"
        init = root / "clim" / "LS-PRACTICALS" / f"surfinit_{site}_2022-2022.nc"

    missing = [p for p in (met, clim, init) if not p.exists()]
    if missing:
        sys.exit(
            "Missing ecLand data file(s):\n  " +
            "\n  ".join(str(m) for m in missing) +
            "\n\nEither unpack thirdpaty/ecland-main.zip, or pass explicit "
            "paths via --met/--clim/--init. See the repository README "
            "(section: ecLand ERA5 site forcing) for how to build these "
            "files for a new site."
        )
    return met, clim, init


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    met, clim, init = resolve_paths(args)
    print(f"[run_ecland] site={args.site}")
    print(f"  met : {met}")
    print(f"  clim: {clim}")
    print(f"  init: {init}")

    # --- load + map -------------------------------------------------------
    raw = load_site(met, clim, init)
    consts = ConstantBundle(PhysicalConstants(), dtype=torch.float64)
    forcing_list, static, state0 = to_inputs(raw, consts, dtype)
    frac = static.tile_frac_base                            # (1, ntile)
    nstep = len(forcing_list)
    print(f"[run_ecland] {nstep} hourly steps; lat={raw['lat']:.3f} "
          f"lon={raw['lon']:.3f} tvl={raw['clim']['tvl']} "
          f"sotype={raw['clim']['sotype']}")
    print(f"[run_ecland] closure: "
          f"{'Farquhar' if args.farquhar else 'Jarvis'}  dtype={args.dtype}")

    # --- run --------------------------------------------------------------
    cfg = SurfConfig(dtype=dtype, use_farquhar=args.farquhar, validate=True)
    model = LandSurface(cfg)

    dt = float(forcing_list[0].dt.item())
    rho_w = float(consts.rho_h2o.item())
    soil_depth_sum = float(static.soil_thickness.sum().item())

    traj = []          # per-step trajectory rows
    forcing_rows = []  # per-step forcing rows (model units)
    state = state0
    for k, forcing in enumerate(forcing_list):
        new_state, diag = model(state, forcing, static)

        # ---- forcing as seen by the model (W/m^2 / kg/m^2/s) ----
        sw_down = forcing.fr_so.item() / dt
        net_lw_in = forcing.fr_th.item() / dt
        precip_rate = forcing.rain.item() / dt + forcing.snow.item() / dt
        forcing_rows.append({
            "step": k,
            "Tair_K": float(forcing.t.item()),
            "Qair_kgkg": float(forcing.q.item()),
            "wind_ms": math.hypot(float(forcing.u.item()),
                                  float(forcing.v.item())),
            "PSurf_Pa": float(forcing.p_surf.item()),
            "SWdown_Wm2": sw_down,
            "netLW_in_Wm2": net_lw_in,
            "precip_kgm2s": precip_rate,
            "mu0": float(forcing.mu0.item()),
        })

        # ---- model response (grid mean) ----
        net_lw = grid_mean(diag.str_tile, frac)             # PSTR
        h_flux = float(diag.h_flux.mean().item())           # PSSH
        le = grid_mean(diag.slh_tile, frac)                 # PSLH
        g0 = grid_mean(diag.g0_tile, frac)                  # PG0
        e_flux = float(diag.e_flux.mean().item())           # kg/m^2/s
        net_rad = float(diag.net_rad.mean().item()) if diag.net_rad is not None else float("nan")

        traj.append({
            "step": k,
            "skin_T_K": grid_mean(new_state.skin_t, frac),
            "soil_T_L1_K": float(new_state.soil_t[0, 0].item()),
            "soil_T_L2_K": float(new_state.soil_t[0, 1].item()),
            "soil_T_L3_K": float(new_state.soil_t[0, 2].item()),
            "soil_T_L4_K": float(new_state.soil_t[0, 3].item()),
            "soilW_L1": float(new_state.soil_w_liq[0, 0].item()),
            "soilW_L2": float(new_state.soil_w_liq[0, 1].item()),
            "soilW_L3": float(new_state.soil_w_liq[0, 2].item()),
            "soilW_L4": float(new_state.soil_w_liq[0, 3].item()),
            "netLW_Wm2": net_lw,
            "H_Wm2": h_flux,
            "LE_Wm2": le,
            "G0_Wm2": g0,
            "evap_kgm2s": e_flux,
            "T2m_K": float(diag.t2m.mean().item()) if diag.t2m is not None else float("nan"),
            "Q2m_kgkg": float(diag.q2m.mean().item()) if diag.q2m is not None else float("nan"),
            "netRad_Wm2": net_rad,
        })
        state = new_state

        if (k + 1) % 10 == 0 or k == nstep - 1:
            print(f"  step {k+1:3d}/{nstep}  skin_T={traj[-1]['skin_T_K']:.2f} K "
                  f"H={h_flux:6.1f} LE={le:6.1f} netLW={net_lw:6.1f} G0={g0:6.1f} W/m^2")

    # --- physical-health diagnostics -------------------------------------
    print("\n[run_ecland] physical-health summary:")
    skin = np.array([r["skin_T_K"] for r in traj])
    print(f"  skin_T range: {skin.min():.2f} ... {skin.max():.2f} K "
          f"(diurnal amplitude {np.ptp(skin):.2f} K)")

    # Energy budget on the brightest step.
    k_day = max(range(nstep),
                key=lambda k: forcing_rows[k]["SWdown_Wm2"])
    r = traj[k_day]
    seb = (forcing_rows[k_day]["SWdown_Wm2"] + r["netLW_Wm2"]
           + r["H_Wm2"] + r["LE_Wm2"] - r["G0_Wm2"])
    print(f"  energy balance @ step {k_day} (SW={forcing_rows[k_day]['SWdown_Wm2']:.1f}): "
          f"residual = {seb:+.2f} W/m^2  "
          f"(netLW={r['netLW_Wm2']:+.1f} H={r['H_Wm2']:+.1f} "
          f"LE={r['LE_Wm2']:+.1f} G0={r['G0_Wm2']:+.1f})")

    # Cumulative water budget over the whole run.
    total_precip = sum(r["precip_kgm2s"] for r in forcing_rows) * dt
    total_evap = sum(r["evap_kgm2s"] for r in traj) * dt
    dvol = float((state.soil_w_liq - state0.soil_w_liq).mean().item())
    dsoilw = dvol * soil_depth_sum * rho_w
    water_res = total_precip - total_evap - dsoilw
    print(f"  water balance over {nstep} steps: precip={total_precip:.3f} "
          f"evap={total_evap:.3f} dSoilW={dsoilw:+.3f} "
          f"residual={water_res:+.3f} kg/m^2")

    # --- write CSVs -------------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import csv
    cols = list(traj[0].keys())
    with open(out_dir / "trajectory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(traj)
    fcols = list(forcing_rows[0].keys())
    with open(out_dir / "forcing_grid_mean.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fcols)
        w.writeheader()
        w.writerows(forcing_rows)

    print(f"\n[run_ecland] wrote:")
    print(f"  {out_dir / 'trajectory.csv'}")
    print(f"  {out_dir / 'forcing_grid_mean.csv'}")


if __name__ == "__main__":
    main()
