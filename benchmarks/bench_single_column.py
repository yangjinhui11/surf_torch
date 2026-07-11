#!/usr/bin/env python3
"""Single-column throughput benchmark: surf_pytorch vs Fortran ecLand.

Measures simulated-days-per-wall-second for a 365-day integration
on a single ERA5 site, with Fortran reference and multiple PyTorch
backends (CPU eager, CPU compile, GPU eager, GPU compile).

Usage:
    python benchmarks/bench_single_column.py
    python benchmarks/bench_single_column.py --site CA-001 --nday 365 --repeat 3
"""

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]   # surf_pytorch/
OPENIFS = REPO.parent                         # openifs/
ECLAND = OPENIFS / "thirdpaty" / "ecland-main"
ECLAND_BIN = ECLAND / "build" / "bin" / "ecland-master-dp"
INPUT_ROOT = ECLAND / "tutorials" / "ecland_input_tutorial"

sys.path.insert(0, str(REPO))

from surf_pytorch.constants import ConstantBundle, PhysicalConstants
from surf_pytorch.config import SurfConfig, N_BASE_TILES
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.forcing import solar_mu0
from surf_pytorch.model import LandSurface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sq(a: np.ndarray) -> np.ndarray:
    return np.asarray(a.values).ravel()


def build_static(site: str, dtype: torch.dtype = torch.float64) -> Static:
    """Build a Static dataclass from ecLand surfclim."""
    cli = xr.open_dataset(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfclim_{site}_2022-2022.nc")
    c = {v: float(_sq(cli[v])[0])
         for v in ["tvl", "tvh", "sotype", "cvl", "cvh", "cu", "landsea",
                    "geopot", "z0m", "lz0h", "lat", "lon"]}

    WET = 0.05
    land = c["landsea"]
    wet = WET * land
    urban = c["cu"] * land
    veg_total = max(c["cvl"] + c["cvh"], 1e-12)
    scale = (land - wet - urban) / veg_total if veg_total > 1e-12 else 0.0
    low = c["cvl"] * scale
    high = c["cvh"] * scale
    bare = land - wet - low - high

    tile_frac = np.zeros(N_BASE_TILES)
    tile_frac[2], tile_frac[3], tile_frac[5], tile_frac[7] = wet, low, high, bare

    return Static(
        tile_frac_base=torch.tensor(tile_frac, dtype=dtype).unsqueeze(0),
        land_frac=torch.tensor([land], dtype=dtype),
        orography=torch.tensor([c["geopot"]], dtype=dtype),
        soil_type=torch.tensor([c["sotype"]], dtype=torch.long),
        veg_type=torch.tensor([c["tvl"]], dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.tensor([0.99], dtype=dtype),
        z0m=torch.tensor([c["z0m"]], dtype=dtype),
        z0h=torch.tensor([math.exp(c["lz0h"])], dtype=dtype),
        z0q=torch.tensor([math.exp(c["lz0h"])], dtype=dtype),
        lat=torch.tensor([math.radians(c["lat"])], dtype=dtype),
        lon=torch.tensor([math.radians(c["lon"])], dtype=dtype),
    )


def build_init_state(site: str, dtype: torch.dtype = torch.float64) -> SurfState:
    """Build initial SurfState from ecLand surfinit."""
    ini = xr.open_dataset(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfinit_{site}_2022-2022.nc")
    soil_temp = np.asarray(ini["SoilTemp"].values).ravel()
    soil_moist = np.asarray(ini["SoilMoist"].values).ravel()
    swe = float(np.asarray(ini["SWE"].values).ravel()[0])
    canopy = float(np.asarray(ini["CanopInt"].values).ravel()[0])
    t0 = float(np.asarray(ini["AvgSurfT"].values).ravel()[0])
    return SurfState(
        skin_t=torch.full((1, N_BASE_TILES), t0, dtype=dtype),
        soil_t=torch.tensor(soil_temp, dtype=dtype).unsqueeze(0),
        soil_w_liq=torch.tensor(soil_moist, dtype=dtype).unsqueeze(0),
        snow_w=torch.tensor([swe], dtype=dtype),
        snow_frac=torch.zeros(1, dtype=dtype),
        skin_water=torch.tensor([canopy], dtype=dtype),
        sst=torch.tensor([t0], dtype=dtype),
    )


def load_forcing(site: str) -> Dict[str, np.ndarray]:
    """Read ERA5 hourly forcing, return dict of 1-D arrays."""
    met = xr.open_dataset(
        INPUT_ROOT / "forcing" / "LS-PRACTICALS" / f"met_era5HT_{site}_2022-2022.nc"
    )
    return {
        "Tair": _sq(met["Tair"]),
        "Qair": _sq(met["Qair"]),
        "Wind_E": _sq(met["Wind_E"]),
        "Wind_N": _sq(met["Wind_N"]),
        "PSurf": _sq(met["PSurf"]),
        "SWdown": _sq(met["SWdown"]),
        "LWdown": _sq(met["LWdown"]),
        "Rainf": _sq(met["Rainf"]),
        "Snowf": _sq(met["Snowf"]),
        "time": met["time"].values,
    }


def _solar_noon(lat_deg: float, lon_deg: float, year: int, doy: float) -> float:
    """Approximate solar noon (UTC hour) for a given lat/lon and day-of-year."""
    # simple approximation, offset by longitude
    noon_utc = 12.0 - lon_deg / 15.0
    # equation of time crude correction
    B = (doy - 81.0) * 360.0 / 365.0 * math.pi / 180.0
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    return noon_utc - eot / 60.0


def run_surf_pytorch(
    site: str,
    nday: int = 365,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    compile_mode: str | None = None,
    use_inference_mode: bool = True,
) -> Tuple[float, float, List[float]]:
    """Run surf_pytorch for *nday* days and return (wall_sec, sim_days_per_sec, [skin_t]*)."""
    static = build_static(site, dtype)
    init = build_init_state(site, dtype)
    forcing = load_forcing(site)

    cfg = SurfConfig(dtype=dtype, validate=False)
    # Move model components to target device
    if device.type == "cuda":
        static = static._replace(
            tile_frac_base=static.tile_frac_base.to(device),
            land_frac=static.land_frac.to(device),
            orography=static.orography.to(device),
            soil_type=static.soil_type.to(device),
            veg_type=static.veg_type.to(device),
            soil_thickness=static.soil_thickness.to(device),
            emis=static.emis.to(device),
            z0m=static.z0m.to(device),
            z0h=static.z0h.to(device),
            z0q=static.z0q.to(device),
            lat=static.lat.to(device),
            lon=static.lon.to(device),
        )
        init = SurfState(
            skin_t=init.skin_t.to(device),
            soil_t=init.soil_t.to(device),
            soil_w_liq=init.soil_w_liq.to(device),
            snow_w=init.snow_w.to(device),
            snow_frac=init.snow_frac.to(device),
            skin_water=init.skin_water.to(device),
            sst=init.sst.to(device),
        )
    model = LandSurface(cfg)
    if compile_mode is not None:
        try:
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
        except Exception as e:
            print(f"  [warn] torch.compile({compile_mode}) failed: {e}")
            print(f"  [warn] falling back to eager mode")
            compile_mode = None  # fallback

    cli = xr.open_dataset(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfclim_{site}_2022-2022.nc")
    lat_deg = float(cli["lat"].values.ravel()[0])
    lon_deg = float(cli["lon"].values.ravel()[0])
    g = float(PhysicalConstants().g)
    sigma = float(PhysicalConstants().rsigma)
    emis = 0.99
    dt = 3600.0
    zphista = float(cli["zphista"].values.ravel()[0]) if "zphista" in cli else 10.0

    # Duplicate the 48-hour ERA5 forcing to span nday days
    n_forcing_hours = len(forcing["Tair"])
    n_steps = nday * 24  # hourly steps

    state = init
    skin_traj = []

    # Warm-up: 2 steps
    _ctx = torch.inference_mode() if use_inference_mode else torch.no_grad()
    with _ctx:
        for _ in range(2):
            k = 0
            sw_down = float(forcing["SWdown"][k])
            lw_down = float(forcing["LWdown"][k])
            ts = float(state.skin_t.mean().item())
            lw_up = emis * sigma * ts ** 4
            net_lw = lw_down - lw_up
            tstamp = forcing["time"][k]
            doy = int(np.asarray(tstamp).astype("datetime64[D]").astype(int) % 365) + 1
            hour_utc = (np.asarray(tstamp).astype("datetime64[h]").astype(int) % 24) + 0.5
            mu0_val = solar_mu0(math.radians(lat_deg), math.radians(lon_deg), doy, float(hour_utc))
            # use noon mu0 for quick cycles
            noon = _solar_noon(lat_deg, lon_deg, 2022, doy)
            mu0_val = solar_mu0(math.radians(lat_deg), math.radians(lon_deg), doy, noon)

            f = Forcing(
                u=torch.tensor([float(forcing["Wind_E"][k])], dtype=dtype, device=device),
                v=torch.tensor([float(forcing["Wind_N"][k])], dtype=dtype, device=device),
                t=torch.tensor([float(forcing["Tair"][k])], dtype=dtype, device=device),
                q=torch.tensor([float(forcing["Qair"][k])], dtype=dtype, device=device),
                co2=torch.tensor([6e-4], dtype=dtype, device=device),
                p_surf=torch.tensor([float(forcing["PSurf"][k])], dtype=dtype, device=device),
                geom1=torch.tensor([g * zphista], dtype=dtype, device=device),
                fr_so=torch.tensor([sw_down * dt], dtype=dtype, device=device),
                fr_th=torch.tensor([net_lw * dt], dtype=dtype, device=device),
                mu0=torch.tensor([mu0_val], dtype=dtype, device=device),
                rain=torch.tensor([float(forcing["Rainf"][k]) * dt], dtype=dtype, device=device),
                snow=torch.tensor([float(forcing["Snowf"][k]) * dt], dtype=dtype, device=device),
                dt=torch.tensor([dt], dtype=dtype, device=device),
            )
            state, diag = model(state, f, static)

    # --- timed section ---
    t0 = time.perf_counter()
    with _ctx:
        for step in range(n_steps):
            k = step % n_forcing_hours
            sw_down = float(forcing["SWdown"][k])
            lw_down = float(forcing["LWdown"][k])
            ts = float(state.skin_t.mean().item())
            lw_up = emis * sigma * ts ** 4
            net_lw = lw_down - lw_up

            # Use solar noon for every step (quick-cycle benchmark)
            doy = (step % 365) + 1
            noon = _solar_noon(lat_deg, lon_deg, 2022, float(doy))
            mu0_val = solar_mu0(math.radians(lat_deg), math.radians(lon_deg), doy, noon)

            f = Forcing(
                u=torch.tensor([float(forcing["Wind_E"][k])], dtype=dtype, device=device),
                v=torch.tensor([float(forcing["Wind_N"][k])], dtype=dtype, device=device),
                t=torch.tensor([float(forcing["Tair"][k])], dtype=dtype, device=device),
                q=torch.tensor([float(forcing["Qair"][k])], dtype=dtype, device=device),
                co2=torch.tensor([6e-4], dtype=dtype, device=device),
                p_surf=torch.tensor([float(forcing["PSurf"][k])], dtype=dtype, device=device),
                geom1=torch.tensor([g * zphista], dtype=dtype, device=device),
                fr_so=torch.tensor([sw_down * dt], dtype=dtype, device=device),
                fr_th=torch.tensor([net_lw * dt], dtype=dtype, device=device),
                mu0=torch.tensor([mu0_val], dtype=dtype, device=device),
                rain=torch.tensor([float(forcing["Rainf"][k]) * dt], dtype=dtype, device=device),
                snow=torch.tensor([float(forcing["Snowf"][k]) * dt], dtype=dtype, device=device),
                dt=torch.tensor([dt], dtype=dtype, device=device),
            )
            state, diag = model(state, f, static)
            skin_traj.append(float(state.skin_t.mean().item()))
    wall = time.perf_counter() - t0

    sim_days = nday
    return wall, sim_days / wall, skin_traj


def run_fortran_ecoland(site: str, nday: int = 365) -> float:
    """Run ecLand Fortran for NSTOP steps and return sim-days/wall-sec."""
    run_dir = ECLAND / "tutorials" / "work" / "RUN" / f"{site}_2022-2022"
    namelist = run_dir / "input"
    backup = run_dir / "input.bak_bench"
    bin_path = ECLAND_BIN

    if not bin_path.exists():
        print("  [skip] Fortran binary not found")
        return float("nan")

    # Write a namelist with NSTOP = nday * 24 * 2  (half-hourly)
    # But ecLand Fortran uses NSTOP in units of TSTEP=1800s.
    # For 1-day benchmark, NSTOP=48.  For 365-day, NSTOP=17520.
    nstop = nday * 48  # 48 half-hours per day
    with open(namelist) as fh:
        lines = fh.readlines()
    # Backup original
    if not backup.exists():
        import shutil
        shutil.copy(namelist, backup)

    new_lines = []
    for line in lines:
        if line.strip().startswith("NSTOP"):
            new_lines.append(f" NSTOP={nstop},\n")
        elif line.strip().startswith("LWREFL"):
            new_lines.append(" LWREFL=.FALSE.,\n")
        else:
            new_lines.append(line)
    with open(namelist, "w") as fh:
        fh.writelines(new_lines)

    try:
        t0 = time.perf_counter()
        subprocess.run(
            [str(bin_path)],
            cwd=str(run_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=600,
        )
        wall = time.perf_counter() - t0
    finally:
        # Restore namelist
        if backup.exists():
            import shutil
            shutil.copy(backup, namelist)
    return nday / wall if wall > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="CA-001")
    parser.add_argument("--nday", type=int, default=365)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--skip-fortran", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  surf_pytorch Benchmark  |  site={args.site}  nday={args.nday}")
    print(f"{'='*70}\n")

    results: Dict[str, float] = {}

    # --- Fortran ecLand ---
    if not args.skip_fortran and ECLAND_BIN.exists():
        print("Fortran ecLand (gfortran -O3, float64, 1800s step) ...", flush=True)
        rates = []
        for _r in range(args.repeat):
            r = run_fortran_ecoland(args.site, args.nday)
            if not math.isnan(r):
                rates.append(r)
        if rates:
            avg = np.mean(rates)
            results["Fortran"] = avg
            print(f"  {avg:>10.1f}  sim-days/s  (avg {len(rates)} runs)\n")

    # --- PyTorch CPU eager ---
    dev_cpu = torch.device("cpu")
    print("PyTorch  CPU  eager  (float64) ...", flush=True)
    rates = []
    for _r in range(args.repeat):
        w, r, _ = run_surf_pytorch(
            args.site, args.nday, torch.float64, dev_cpu,
            compile_mode=None, use_inference_mode=True,
        )
        rates.append(r)
    avg = np.mean(rates)
    results["PyTorch-CPU-eager"] = avg
    print(f"  {avg:>10.1f}  sim-days/s\n")

    # --- PyTorch CPU torch.compile ---
    print("PyTorch  CPU  compile('reduce-overhead')  (float64) ...", flush=True)
    rates = []
    for _r in range(args.repeat):
        w, r, _ = run_surf_pytorch(
            args.site, args.nday, torch.float64, dev_cpu,
            compile_mode="reduce-overhead", use_inference_mode=True,
        )
        rates.append(r)
    avg = np.mean(rates)
    results["PyTorch-CPU-compile"] = avg
    print(f"  {avg:>10.1f}  sim-days/s\n")

    # --- PyTorch GPU eager ---
    if not args.skip_gpu and torch.cuda.is_available():
        dev_gpu = torch.device("cuda:2")  # GPU 2 is idle
        for dtype_name, dtype in [("float64", torch.float64), ("float32", torch.float32)]:
            print(f"PyTorch  GPU  eager  ({dtype_name}) ...", flush=True)
            rates = []
            for _r in range(args.repeat):
                w, r, _ = run_surf_pytorch(
                    args.site, args.nday, dtype, dev_gpu,
                    compile_mode=None, use_inference_mode=True,
                )
                if _r == 0:  # first GPU run includes warmup
                    pass
                rates.append(r)
            avg = np.mean(rates)
            key = f"PyTorch-GPU-eager-{dtype_name}"
            results[key] = avg
            print(f"  {avg:>10.1f}  sim-days/s\n")

        # --- PyTorch GPU torch.compile ---
        for dtype_name, dtype in [("float64", torch.float64), ("float32", torch.float32)]:
            print(f"PyTorch  GPU  compile('reduce-overhead')  ({dtype_name}) ...", flush=True)
            rates = []
            for _r in range(args.repeat):
                w, r, _ = run_surf_pytorch(
                    args.site, args.nday, dtype, dev_gpu,
                    compile_mode="reduce-overhead", use_inference_mode=True,
                )
                rates.append(r)
            avg = np.mean(rates)
            key = f"PyTorch-GPU-compile-{dtype_name}"
            results[key] = avg
            print(f"  {avg:>10.1f}  sim-days/s\n")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY  (site={args.site}, nday={args.nday})")
    print(f"{'='*70}")
    fortran_rate = results.get("Fortran", None)
    for label, rate in sorted(results.items(), key=lambda x: -x[1]):
        speedup = ""
        if fortran_rate is not None and fortran_rate > 0:
            speedup = f"  ({rate/fortran_rate:.1f}× vs Fortran)"
        print(f"  {label:<40s}  {rate:>10.1f} sim-days/s{speedup}")
    print()


if __name__ == "__main__":
    main()
