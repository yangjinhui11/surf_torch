#!/usr/bin/env python3
"""Single-column throughput benchmark: surf_pytorch vs Fortran ecLand.

Quick baseline without torch.compile.
"""

import math, os, subprocess, sys, time, argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import xarray as xr

# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
OPENIFS = REPO.parent
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
_future_warn_emitted = False
def _open_ds(path):
    """Suppress FutureWarning from xarray."""
    global _future_warn_emitted
    if not _future_warn_emitted:
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        _future_warn_emitted = True
    return xr.open_dataset(path)

def _sq(a):
    return np.asarray(a.values).ravel()


def build_static(site, dtype=torch.float64, device="cpu"):
    cli = _open_ds(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfclim_{site}_2022-2022.nc")
    c = {}
    for v in ["tvl","tvh","sotype","cvl","cvh","cu","landsea","geopot","z0m","lz0h","lat","lon"]:
        c[v] = float(_sq(cli[v])[0])
    WET = 0.05
    land = c["landsea"]
    wet = WET * land
    urban = c["cu"] * land
    veg_total = max(c["cvl"] + c["cvh"], 1e-12)
    scale = (land - wet - urban) / veg_total if veg_total > 1e-12 else 0.0
    low = c["cvl"] * scale; high = c["cvh"] * scale
    bare = land - wet - low - high
    tf = np.zeros(N_BASE_TILES)
    tf[2], tf[3], tf[5], tf[7] = wet, low, high, bare
    return Static(
        tile_frac_base=torch.tensor(tf, dtype=dtype, device=device).unsqueeze(0),
        land_frac=torch.tensor([land], dtype=dtype, device=device),
        orography=torch.tensor([c["geopot"]], dtype=dtype, device=device),
        soil_type=torch.tensor([c["sotype"]], dtype=torch.long),       # keep on CPU – index for LUT
        veg_type=torch.tensor([c["tvl"]], dtype=torch.long),            # keep on CPU – index for LUT
        soil_thickness=torch.tensor([0.07,0.21,0.72,1.89], dtype=dtype, device=device),
        emis=torch.tensor([0.99], dtype=dtype, device=device),
        z0m=torch.tensor([c["z0m"]], dtype=dtype, device=device),
        z0h=torch.tensor([math.exp(c["lz0h"])], dtype=dtype, device=device),
        z0q=torch.tensor([math.exp(c["lz0h"])], dtype=dtype, device=device),
        lat=torch.tensor([math.radians(c["lat"])], dtype=dtype, device=device),
        lon=torch.tensor([math.radians(c["lon"])], dtype=dtype, device=device),
    )


def build_init(site, dtype=torch.float64, device="cpu"):
    ini = _open_ds(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfinit_{site}_2022-2022.nc")
    st = _sq(ini["SoilTemp"]); sm = _sq(ini["SoilMoist"])
    swe = float(_sq(ini["SWE"])[0]); can = float(_sq(ini["CanopInt"])[0])
    t0 = float(_sq(ini["AvgSurfT"])[0])
    return SurfState(
        skin_t=torch.full((1, N_BASE_TILES), t0, dtype=dtype, device=device),
        soil_t=torch.tensor(st, dtype=dtype, device=device).unsqueeze(0),
        soil_w_liq=torch.tensor(sm, dtype=dtype, device=device).unsqueeze(0),
        snow_w=torch.tensor([swe], dtype=dtype, device=device),
        snow_frac=torch.zeros(1, dtype=dtype, device=device),
        skin_water=torch.tensor([can], dtype=dtype, device=device),
        sst=torch.tensor([t0], dtype=dtype, device=device),
    )


def load_forcing(site):
    met = _open_ds(INPUT_ROOT / "forcing" / "LS-PRACTICALS" / f"met_era5HT_{site}_2022-2022.nc")
    return {
        "Tair": _sq(met["Tair"]), "Qair": _sq(met["Qair"]),
        "Wind_E": _sq(met["Wind_E"]), "Wind_N": _sq(met["Wind_N"]),
        "PSurf": _sq(met["PSurf"]), "SWdown": _sq(met["SWdown"]),
        "LWdown": _sq(met["LWdown"]), "Rainf": _sq(met["Rainf"]),
        "Snowf": _sq(met["Snowf"]), "time": met["time"].values,
    }


def run_torch(site, nday, dtype=torch.float64, device="cpu"):
    """Run surf_pytorch and return (wall_sec, sim_days_per_sec)."""
    static = build_static(site, dtype, device)
    state = build_init(site, dtype, device)
    forcing = load_forcing(site)
    cfg = SurfConfig(dtype=dtype, validate=False)
    model = LandSurface(cfg)

    # Site geometry
    cli = _open_ds(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfclim_{site}_2022-2022.nc")
    lat_deg = float(cli["lat"].values.ravel()[0])
    lon_deg = float(cli["lon"].values.ravel()[0])
    g = float(PhysicalConstants().g)
    rsigma = float(PhysicalConstants().rsigma)
    emis = 0.99
    dt = 3600.0
    zphista = float(cli["zphista"].values.ravel()[0]) if "zphista" in cli else 10.0

    n_forcing_hours = len(forcing["Tair"])
    n_steps = nday * 24

    # Warm-up: 5 steps
    with torch.inference_mode():
        for _k in range(5):
            k = _k % n_forcing_hours
            sw_down = float(forcing["SWdown"][k])
            lw_down = float(forcing["LWdown"][k])
            ts = float(state.skin_t.mean().item())
            lw_up = emis * rsigma * ts ** 4
            net_lw = lw_down - lw_up
            doy = (_k % 365) + 1
            noon = 12.0 - lon_deg / 15.0
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
            state, _diag = model(state, f, static)

    # Timed section
    t0 = time.perf_counter()
    with torch.inference_mode():
        for step in range(n_steps):
            k = step % n_forcing_hours
            sw_down = float(forcing["SWdown"][k])
            lw_down = float(forcing["LWdown"][k])
            ts = float(state.skin_t.mean().item())
            lw_up = emis * rsigma * ts ** 4
            net_lw = lw_down - lw_up
            doy = (step % 365) + 1
            noon = 12.0 - lon_deg / 15.0
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
            state, _diag = model(state, f, static)

    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return wall, nday / wall


def run_fortran(site, nday):
    """Run ecLand Fortran NSTOP=nday*48 steps."""
    run_dir = ECLAND / "tutorials" / "work" / "RUN" / f"{site}_2022-2022"
    namelist = run_dir / "input"
    backup = run_dir / "input.bak_bench"
    if not ECLAND_BIN.exists():
        return float("nan")

    nstop = nday * 48
    with open(namelist) as fh:
        lines = fh.readlines()
    if not backup.exists():
        import shutil; shutil.copy(namelist, backup)
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
            [str(ECLAND_BIN)], cwd=str(run_dir),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600,
        )
        wall = time.perf_counter() - t0
    finally:
        if backup.exists():
            import shutil; shutil.copy(backup, namelist)
    return nday / wall if wall > 0 else float("nan")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="CA-001")
    p.add_argument("--nday", type=int, default=365)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--skip-fortran", action="store_true")
    p.add_argument("--skip-gpu", action="store_true")
    args = p.parse_args()

    print(f"\n{'='*70}")
    print(f"  surf_pytorch Single-Column Benchmark")
    print(f"  site={args.site}  nday={args.nday}  repeat={args.repeat}")
    print(f"{'='*70}\n")

    results = {}

    # -- Fortran --
    if not args.skip_fortran and ECLAND_BIN.exists():
        print("Fortran ecLand (gfortran -O3, float64, dt=1800s) ...", flush=True)
        rates = []
        for r in range(args.repeat):
            rate = run_fortran(args.site, args.nday)
            if not math.isnan(rate):
                rates.append(rate)
        if rates:
            avg = np.mean(rates)
            results["Fortran (float64)"] = avg
            print(f"  {avg:>10.1f}  sim-days/s\n")

    # -- PyTorch CPU float64 --
    print("PyTorch CPU  eager  float64  ...", flush=True)
    rates = []
    for r in range(args.repeat):
        w, rate = run_torch(args.site, args.nday, torch.float64, "cpu")
        rates.append(rate)
    avg = np.mean(rates)
    results["PyTorch CPU float64"] = avg
    print(f"  {avg:>10.1f}  sim-days/s\n")

    # -- PyTorch GPU float64 --
    if not args.skip_gpu and torch.cuda.is_available():
        dev = "cuda:2"  # idle GPU
        print(f"PyTorch GPU  eager  float64  ({dev}) ...", flush=True)
        rates = []
        for r in range(args.repeat):
            w, rate = run_torch(args.site, args.nday, torch.float64, dev)
            rates.append(rate)
        avg = np.mean(rates)
        results["PyTorch GPU float64"] = avg
        print(f"  {avg:>10.1f}  sim-days/s\n")

        # -- PyTorch GPU float32 --
        print(f"PyTorch GPU  eager  float32  ({dev}) ...", flush=True)
        rates = []
        for r in range(args.repeat):
            w, rate = run_torch(args.site, args.nday, torch.float32, dev)
            rates.append(rate)
        avg = np.mean(rates)
        results["PyTorch GPU float32"] = avg
        print(f"  {avg:>10.1f}  sim-days/s\n")

    # -- Summary --
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    fortran_rate = results.get("Fortran (float64)", None)
    for label in sorted(results, key=lambda x: -results[x]):
        rate = results[label]
        speedup = ""
        if fortran_rate is not None and fortran_rate > 0:
            speedup = f"  ({rate/fortran_rate:.1f}×)"
        print(f"  {label:<35s}  {rate:>10.1f} sim-days/s{speedup}")
    print()


if __name__ == "__main__":
    main()
