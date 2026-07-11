#!/usr/bin/env python3
"""Single-column benchmark with torch.compile after .item() removal."""

import math, os, sys, time, argparse
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OPENIFS = REPO.parent
ECLAND = OPENIFS / "thirdpaty" / "ecland-main"
INPUT_ROOT = ECLAND / "tutorials" / "ecland_input_tutorial"

sys.path.insert(0, str(REPO))
import warnings; warnings.filterwarnings("ignore")

from surf_pytorch.constants import ConstantBundle, PhysicalConstants
from surf_pytorch.config import SurfConfig, N_BASE_TILES
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.forcing import solar_mu0
from surf_pytorch.model import LandSurface
import xarray as xr

# ---------------------------------------------------------------------------
def _sq(a): return np.asarray(a.values).ravel()
def _ds(p): return xr.open_dataset(p)

def build_static(site, dtype=torch.float64):
    cli = _ds(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfclim_{site}_2022-2022.nc")
    c = {v: float(_sq(cli[v])[0]) for v in
         ["tvl","tvh","sotype","cvl","cvh","cu","landsea","geopot","z0m","lz0h","lat","lon"]}
    WET = 0.05; land = c["landsea"]; wet = WET * land; urban = c["cu"] * land
    vtotal = max(c["cvl"] + c["cvh"], 1e-12)
    scale = (land - wet - urban) / vtotal if vtotal > 1e-12 else 0.0
    low = c["cvl"] * scale; high = c["cvh"] * scale
    bare = land - wet - low - high
    tf = np.zeros(N_BASE_TILES); tf[2], tf[3], tf[5], tf[7] = wet, low, high, bare
    return Static(
        tile_frac_base=torch.tensor(tf, dtype=dtype).unsqueeze(0),
        land_frac=torch.tensor([land], dtype=dtype),
        orography=torch.tensor([c["geopot"]], dtype=dtype),
        soil_type=torch.tensor([c["sotype"]], dtype=torch.long),
        veg_type=torch.tensor([c["tvl"]], dtype=torch.long),
        soil_thickness=torch.tensor([0.07,0.21,0.72,1.89], dtype=dtype),
        emis=torch.tensor([0.99], dtype=dtype),
        z0m=torch.tensor([c["z0m"]], dtype=dtype),
        z0h=torch.tensor([math.exp(c["lz0h"])], dtype=dtype),
        z0q=torch.tensor([math.exp(c["lz0h"])], dtype=dtype),
        lat=torch.tensor([math.radians(c["lat"])], dtype=dtype),
        lon=torch.tensor([math.radians(c["lon"])], dtype=dtype),
    )


def build_init(site, dtype=torch.float64):
    ini = _ds(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfinit_{site}_2022-2022.nc")
    st = _sq(ini["SoilTemp"]); sm = _sq(ini["SoilMoist"])
    swe = float(_sq(ini["SWE"])[0]); can = float(_sq(ini["CanopInt"])[0])
    t0 = float(_sq(ini["AvgSurfT"])[0])
    return SurfState(
        skin_t=torch.full((1, N_BASE_TILES), t0, dtype=dtype),
        soil_t=torch.tensor(st, dtype=dtype).unsqueeze(0),
        soil_w_liq=torch.tensor(sm, dtype=dtype).unsqueeze(0),
        snow_w=torch.tensor([swe], dtype=dtype),
        snow_frac=torch.zeros(1, dtype=dtype),
        skin_water=torch.tensor([can], dtype=dtype),
        sst=torch.tensor([t0], dtype=dtype),
    )


def load_forcing(site):
    met = _ds(INPUT_ROOT / "forcing" / "LS-PRACTICALS" / f"met_era5HT_{site}_2022-2022.nc")
    return {"Tair": _sq(met["Tair"]), "Qair": _sq(met["Qair"]),
            "Wind_E": _sq(met["Wind_E"]), "Wind_N": _sq(met["Wind_N"]),
            "PSurf": _sq(met["PSurf"]), "SWdown": _sq(met["SWdown"]),
            "LWdown": _sq(met["LWdown"]), "Rainf": _sq(met["Rainf"]),
            "Snowf": _sq(met["Snowf"]), "time": met["time"].values}


def run_torch(site, nday, dtype=torch.float64, compile_mode=None):
    """Run surf_pytorch and return (wall_sec, sim_days_per_sec)."""
    static = build_static(site, dtype)
    state = build_init(site, dtype)
    forcing = load_forcing(site)
    cfg = SurfConfig(dtype=dtype, validate=False)
    model = LandSurface(cfg)

    cli = _ds(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfclim_{site}_2022-2022.nc")
    lat_deg = float(cli["lat"].values.ravel()[0])
    lon_deg = float(cli["lon"].values.ravel()[0])
    g = float(PhysicalConstants().g)
    rsigma = float(PhysicalConstants().rsigma)
    emis = 0.99; dt = 3600.0
    zphista = float(cli["zphista"].values.ravel()[0]) if "zphista" in cli else 10.0

    if compile_mode is not None:
        _dynamo_config = __import__("torch._dynamo.config", fromlist=["config"])
        _dynamo_config.capture_scalar_outputs = True
        _torch = __import__("torch")
        model = _torch.compile(model, mode=compile_mode, fullgraph=False)

    n_forcing_hours = len(forcing["Tair"])
    n_steps = nday * 24

    # Warm-up: 10 steps
    with torch.inference_mode():
        for _k in range(10):
            k = _k % n_forcing_hours
            sw_down = float(forcing["SWdown"][k])
            lw_down = float(forcing["LWdown"][k])
            ts = float(state.skin_t.mean().item())
            net_lw = lw_down - emis * rsigma * ts ** 4
            noon = 12.0 - lon_deg / 15.0
            mu0_val = solar_mu0(math.radians(lat_deg), math.radians(lon_deg), (_k%365)+1, noon)
            f = Forcing(
                u=torch.tensor([float(forcing["Wind_E"][k])], dtype=dtype),
                v=torch.tensor([float(forcing["Wind_N"][k])], dtype=dtype),
                t=torch.tensor([float(forcing["Tair"][k])], dtype=dtype),
                q=torch.tensor([float(forcing["Qair"][k])], dtype=dtype),
                co2=torch.tensor([6e-4], dtype=dtype),
                p_surf=torch.tensor([forcing["PSurf"][k]], dtype=dtype),
                geom1=torch.tensor([g * zphista], dtype=dtype),
                fr_so=torch.tensor([sw_down * dt], dtype=dtype),
                fr_th=torch.tensor([net_lw * dt], dtype=dtype),
                mu0=torch.tensor([mu0_val], dtype=dtype),
                rain=torch.tensor([0.0], dtype=dtype),
                snow=torch.tensor([0.0], dtype=dtype),
                dt=torch.tensor([dt], dtype=dtype),
            )
            state, _diag = model(state, f, static)

    # Timed
    t0 = time.perf_counter()
    with torch.inference_mode():
        for step in range(n_steps):
            k = step % n_forcing_hours
            sw_down = float(forcing["SWdown"][k])
            lw_down = float(forcing["LWdown"][k])
            ts = float(state.skin_t.mean().item())
            net_lw = lw_down - emis * rsigma * ts ** 4
            noon = 12.0 - lon_deg / 15.0
            mu0_val = solar_mu0(math.radians(lat_deg), math.radians(lon_deg), (step%365)+1, noon)
            f = Forcing(
                u=torch.tensor([float(forcing["Wind_E"][k])], dtype=dtype),
                v=torch.tensor([float(forcing["Wind_N"][k])], dtype=dtype),
                t=torch.tensor([float(forcing["Tair"][k])], dtype=dtype),
                q=torch.tensor([float(forcing["Qair"][k])], dtype=dtype),
                co2=torch.tensor([6e-4], dtype=dtype),
                p_surf=torch.tensor([forcing["PSurf"][k]], dtype=dtype),
                geom1=torch.tensor([g * zphista], dtype=dtype),
                fr_so=torch.tensor([sw_down * dt], dtype=dtype),
                fr_th=torch.tensor([net_lw * dt], dtype=dtype),
                mu0=torch.tensor([mu0_val], dtype=dtype),
                rain=torch.tensor([0.0], dtype=dtype),
                snow=torch.tensor([0.0], dtype=dtype),
                dt=torch.tensor([dt], dtype=dtype),
            )
            state, _diag = model(state, f, static)

    wall = time.perf_counter() - t0
    return wall, nday / wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="CA-001")
    p.add_argument("--nday", type=int, default=365)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--skip-compile", action="store_true")
    args = p.parse_args()

    print(f"\n{'='*70}")
    print(f"  surf_pytorch Optimized Benchmark  ({args.site}, {args.nday} days)")
    print(f"{'='*70}\n")

    results = {}

    # Eager baseline
    print("PyTorch CPU  eager  float64  ...", flush=True)
    rates = [run_torch(args.site, args.nday, torch.float64, None)[1] for _ in range(args.repeat)]
    results["CPU-eager"] = np.mean(rates)
    print(f"  {results['CPU-eager']:.1f} sim-days/s\n")

    if not args.skip_compile:
        # torch.compile with default mode
        print("PyTorch CPU  compile('default')  float64  ...", flush=True)
        try:
            rates = [run_torch(args.site, 30, torch.float64, "default")[1] for _ in range(min(args.repeat,2))]
            results["CPU-compile-default"] = np.mean(rates)
            print(f"  {results['CPU-compile-default']:.1f} sim-days/s (30-day test)\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")

        # torch.compile with reduce-overhead
        print("PyTorch CPU  compile('reduce-overhead')  float64  ...", flush=True)
        try:
            rates = [run_torch(args.site, 30, torch.float64, "reduce-overhead")[1] for _ in range(min(args.repeat,2))]
            results["CPU-compile-reduce-overhead"] = np.mean(rates)
            print(f"  {results['CPU-compile-reduce-overhead']:.1f} sim-days/s (30-day test)\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")

    print(f"{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    base = results.get("CPU-eager", 1.0)
    for label in sorted(results, key=lambda x: -results[x]):
        rate = results[label]
        print(f"  {label:<40s}  {rate:>8.1f} sim-days/s  ({rate/base:.1f}×)")
    print()


if __name__ == "__main__":
    main()
