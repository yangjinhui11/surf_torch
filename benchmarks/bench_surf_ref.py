#!/usr/bin/env python3
"""surf_pytorch vs OpenIFS Fortran SURF reference (libsurf_ref.so) benchmark.

Times the full physics chain (VEXCS→VEVAP→VDFDIFH→SOILT→SNOW→SOILW)
for single-column 365-day integration, comparing:
  1. PyTorch CPU  float64  (eager)
  2. Fortran ref   float64  (libsurf_ref.so via ctypes)
  3. PyTorch GPU  float64  (cuda eager)
"""
import math, os, sys, time, argparse, gc
from pathlib import Path
from typing import List
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "offline_driver"))
import warnings; warnings.filterwarnings("ignore")

from surf_pytorch.config import SurfConfig, N_BASE_TILES, TILE_LOW_VEG_DRY
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.forcing import solar_mu0
from surf_pytorch.model import LandSurface
from surf_pytorch.constants import ConstantBundle, PhysicalConstants
from offline_driver import SurfRef
import xarray as xr

# ---------------------------------------------------------------------------
INPUT_ROOT = REPO.parent / "thirdpaty" / "ecland-main" / "tutorials" / "ecland_input_tutorial"
ECLAND_BIN = REPO.parent / "thirdpaty" / "ecland-main" / "build" / "bin" / "ecland-master-dp"

def _sq(a): return np.asarray(a.values).ravel()
def _ds(p): return xr.open_dataset(p)


def build_single_column(site="CA-001", dtype=torch.float64, device="cpu"):
    """Build model, static, state, forcing dict and site metadata."""
    cli = _ds(str(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfclim_{site}_2022-2022.nc"))
    c = {v: float(_sq(cli[v])[0]) for v in
         ["tvl","tvh","sotype","cvl","cvh","cu","landsea","geopot","z0m","lz0h","lat","lon"]}
    WET = 0.05; land = c["landsea"]; wet = WET * land; urban = c["cu"] * land
    vtotal = max(c["cvl"] + c["cvh"], 1e-12)
    scale = (land - wet - urban) / vtotal if vtotal > 1e-12 else 0.0
    low = c["cvl"] * scale; high = c["cvh"] * scale
    bare = land - wet - low - high
    tf = np.zeros(N_BASE_TILES); tf[2], tf[3], tf[5], tf[7] = wet, low, high, bare

    def _mv(t):
        if isinstance(t, torch.Tensor): return t.to(device=device)
        return t

    static = Static(
        tile_frac_base=_mv(torch.tensor(tf, dtype=dtype).unsqueeze(0)),
        land_frac=_mv(torch.tensor([land], dtype=dtype)),
        orography=_mv(torch.tensor([c["geopot"]], dtype=dtype)),
        soil_type=_mv(torch.tensor([c["sotype"]], dtype=torch.long)),
        veg_type=_mv(torch.tensor([c["tvl"]], dtype=torch.long)),
        soil_thickness=_mv(torch.tensor([0.07,0.21,0.72,1.89], dtype=dtype)),
        emis=_mv(torch.tensor([0.99], dtype=dtype)),
        z0m=_mv(torch.tensor([c["z0m"]], dtype=dtype)),
        z0h=_mv(torch.tensor([math.exp(c["lz0h"])], dtype=dtype)),
        z0q=_mv(torch.tensor([math.exp(c["lz0h"])], dtype=dtype)),
        lat=_mv(torch.tensor([math.radians(c["lat"])], dtype=dtype)),
        lon=_mv(torch.tensor([math.radians(c["lon"])], dtype=dtype)),
    )

    ini = _ds(str(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfinit_{site}_2022-2022.nc"))
    st = _sq(ini["SoilTemp"]); sm = _sq(ini["SoilMoist"])
    swe = float(_sq(ini["SWE"])[0]); can = float(_sq(ini["CanopInt"])[0])
    t0 = float(_sq(ini["AvgSurfT"])[0])
    state = SurfState(
        skin_t=_mv(torch.full((1, N_BASE_TILES), t0, dtype=dtype)),
        soil_t=_mv(torch.tensor(st, dtype=dtype).unsqueeze(0)),
        soil_w_liq=_mv(torch.tensor(sm, dtype=dtype).unsqueeze(0)),
        snow_w=_mv(torch.tensor([swe], dtype=dtype)),
        snow_frac=_mv(torch.zeros(1, dtype=dtype)),
        skin_water=_mv(torch.tensor([can], dtype=dtype)),
        sst=_mv(torch.tensor([t0], dtype=dtype)),
    )

    met = _ds(str(INPUT_ROOT / "forcing" / "LS-PRACTICALS" / f"met_era5HT_{site}_2022-2022.nc"))
    forcing = {"Tair": _sq(met["Tair"]), "Qair": _sq(met["Qair"]),
               "Wind_E": _sq(met["Wind_E"]), "Wind_N": _sq(met["Wind_N"]),
               "PSurf": _sq(met["PSurf"]), "SWdown": _sq(met["SWdown"]),
               "LWdown": _sq(met["LWdown"]), "Rainf": _sq(met["Rainf"]),
               "Snowf": _sq(met["Snowf"]), "time": met["time"].values}

    cfg = SurfConfig(dtype=dtype, validate=False)
    model = LandSurface(cfg)
    if device != "cpu":
        model = model.to(device=device)
    return model, static, state, forcing, c, cli


def build_forcing(k, forcing, dtype, device, lat_deg, lon_deg, g, rsigma, emis, dt, zphista, doy, state):
    """Build forcing for step k (for single-column)."""
    sw_down = float(forcing["SWdown"][k])
    lw_down = float(forcing["LWdown"][k])
    ts = float(state.skin_t.mean().item())
    lw_up = emis * rsigma * ts ** 4
    net_lw = lw_down - lw_up
    noon = 12.0 - lon_deg / 15.0
    mu0_val = solar_mu0(math.radians(lat_deg), math.radians(lon_deg), doy, noon)
    return Forcing(
        u=torch.tensor([float(forcing["Wind_E"][k])], dtype=dtype, device=device),
        v=torch.tensor([float(forcing["Wind_N"][k])], dtype=dtype, device=device),
        t=torch.tensor([float(forcing["Tair"][k])], dtype=dtype, device=device),
        q=torch.tensor([float(forcing["Qair"][k])], dtype=dtype, device=device),
        co2=torch.tensor([6e-4], dtype=dtype, device=device),
        p_surf=torch.tensor([forcing["PSurf"][k]], dtype=dtype, device=device),
        geom1=torch.tensor([g * zphista], dtype=dtype, device=device),
        fr_so=torch.tensor([sw_down * dt], dtype=dtype, device=device),
        fr_th=torch.tensor([net_lw * dt], dtype=dtype, device=device),
        mu0=torch.tensor([mu0_val], dtype=dtype, device=device),
        rain=torch.tensor([float(forcing["Rainf"][k]) * dt], dtype=dtype, device=device),
        snow=torch.tensor([float(forcing["Snowf"][k]) * dt], dtype=dtype, device=device),
        dt=torch.tensor([dt], dtype=dtype, device=device),
    )


def bench_pytorch(site, nday, dtype=torch.float64, device="cpu"):
    """Benchmark full surf_pytorch forward pass for nday days."""
    model, static, state, fdict, c, cli = build_single_column(site, dtype, device)
    lat_deg = float(c["lat"]); lon_deg = float(c["lon"])
    g = float(PhysicalConstants().g); rsigma = float(PhysicalConstants().rsigma)
    emis = 0.99; dt = 3600.0
    zphista = float(cli["zphista"].values.ravel()[0]) if "zphista" in cli else 10.0
    n_forcing_hours = len(fdict["Tair"]); n_steps = nday * 24

    # Warmup 10 steps
    with torch.inference_mode():
        for _k in range(10):
            k = _k % n_forcing_hours; doy = (_k % 365) + 1
            f = build_forcing(k, fdict, dtype, device, lat_deg, lon_deg, g, rsigma, emis, dt, zphista, doy, state)
            state, _ = model(state, f, static)

    if device != "cpu": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for step in range(n_steps):
            k = step % n_forcing_hours; doy = (step % 365) + 1
            f = build_forcing(k, fdict, dtype, device, lat_deg, lon_deg, g, rsigma, emis, dt, zphista, doy, state)
            state, _ = model(state, f, static)
    if device != "cpu": torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return wall, nday / wall


def bench_fortran_ref(site, nday):
    """Benchmark Fortran SINGLE physics kernel through libsurf_ref.so.
    
    NOTE: This is NOT a full model run — it times a single representative
    kernel call (VEXCS) to estimate native Fortran speed.  The full
    Fortran SURF driver is not compiled standalone.
    """
    # Use a single VEXCS call as representative Fortran kernel benchmark
    ref = SurfRef()
    if not ref.is_available():
        return float("nan"), 0
    
    consts = ConstantBundle(PhysicalConstants(), dtype=torch.float64)
    ngrid = 1
    # Build dummy inputs
    pu = torch.ones(ngrid, dtype=torch.float64) * 3.0
    pv = torch.zeros(ngrid, dtype=torch.float64)
    pt = torch.ones(ngrid, dtype=torch.float64) * 280.0
    pq = torch.ones(ngrid, dtype=torch.float64) * 0.005
    paphms = torch.ones(ngrid, dtype=torch.float64) * 100000.0
    pgeomlev = torch.ones(ngrid, dtype=torch.float64) * 98.0
    pcptgzlev = torch.ones(ngrid, dtype=torch.float64) * consts.cpd.item() * 280.0
    pcpts = torch.ones(ngrid, dtype=torch.float64) * consts.cpd.item() * 278.0
    pqsam = torch.ones(ngrid, dtype=torch.float64) * 0.006
    pz0mm = torch.ones(ngrid, dtype=torch.float64) * 0.5
    pz0hm = torch.ones(ngrid, dtype=torch.float64) * 0.0025
    pz0qm = torch.ones(ngrid, dtype=torch.float64) * 0.0025
    pbuom = torch.zeros(ngrid, dtype=torch.float64)
    pucurr = torch.zeros(ngrid, dtype=torch.float64)
    pvcurr = torch.zeros(ngrid, dtype=torch.float64)
    pi10fgcv = torch.zeros(ngrid, dtype=torch.float64)
    
    # Warmup
    for _ in range(100):
        ref.vexcs(pu, pv, pt, pq, paphms, pgeomlev, pcptgzlev, pcpts, pqsam,
                   pz0mm, pz0hm, pz0qm, pbuom, pucurr, pvcurr, pi10fgcv, consts)
    
    n_calls = 5000
    t0 = time.perf_counter()
    for _ in range(n_calls):
        ref.vexcs(pu, pv, pt, pq, paphms, pgeomlev, pcptgzlev, pcpts, pqsam,
                   pz0mm, pz0hm, pz0qm, pbuom, pucurr, pvcurr, pi10fgcv, consts)
    wall = time.perf_counter() - t0
    usec_per_call = wall / n_calls * 1e6
    return usec_per_call, wall


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="CA-001")
    p.add_argument("--nday", type=int, default=365)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--skip-gpu", action="store_true")
    args = p.parse_args()

    print(f"\n{'='*70}")
    print(f"  surf_pytorch vs OpenIFS Fortran SURF Benchmark")
    print(f"  site={args.site}  nday={args.nday}  repeat={args.repeat}")
    print(f"{'='*70}\n")

    results = {}

    # --- PyTorch CPU ---
    print("surf_pytorch  CPU  float64  (full forward pass) ...", flush=True)
    rates = []
    for _ in range(args.repeat):
        w, r = bench_pytorch(args.site, args.nday, torch.float64, "cpu")
        rates.append(r)
    avg = np.mean(rates)
    results["PyTorch-CPU"] = avg
    print(f"  {avg:.1f} sim-days/s  ({args.nday/avg:.1f}s wall)\n")

    # --- PyTorch GPU ---
    if not args.skip_gpu and torch.cuda.is_available():
        dev = "cuda:2"
        for dtype_name, dtype in [("float64", torch.float64), ("float32", torch.float32)]:
            print(f"surf_pytorch  GPU  {dtype_name}  (full forward pass) ...", flush=True)
            try:
                model_sizes = [1]
                for bs in model_sizes:
                    w, r = bench_pytorch(args.site, args.nday, dtype, dev)
                    key = f"PyTorch-GPU-{dtype_name}"
                    results[key] = r
                    print(f"  {r:.1f} sim-days/s  ({args.nday/r:.1f}s wall)\n")
            except Exception as e:
                print(f"  FAILED: {e}\n")
        gc.collect(); torch.cuda.empty_cache()

    # --- Fortran reference kernel ---
    print("Fortran ref  VEXCS kernel  float64  (libsurf_ref.so) ...", flush=True)
    usec, _ = bench_fortran_ref(args.site, args.nday)
    if not math.isnan(usec):
        results["Fortran-VEXCS(us/call)"] = usec
        print(f"  {usec:.1f} µs/call\n")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for label in sorted(results, key=lambda x: -float(results[x]) if "VEXCS" not in x else 0):
        val = results[label]
        if "VEXCS" in label:
            print(f"  {label:<40s}  {val:>10.1f} µs/call")
        else:
            print(f"  {label:<40s}  {val:>10.1f} sim-days/s")
    print()

    # Compute how many VEXCS calls a full forward pass uses
    # ~1 call to _monin_obukhov_coeffs per timestep
    # Each has 2 VEXCS calls (lamin/visc + full)
    # Plus VDFDIFH calls VEXCS again
    # That's ~3 VEXCS calls per forward pass
    n_vexcs_per_step = 3
    if "Fortran-VEXCS(us/call)" in results and "PyTorch-CPU" in results:
        us_per_vexcs = results["Fortran-VEXCS(us/call)"]
        pytorch_rate = results["PyTorch-CPU"]
        pytorch_ms_per_step = 1000 / (pytorch_rate * 24)  # ms per hourly step
        fortran_us_per_step = us_per_vexcs * n_vexcs_per_step
        print(f"\n  For reference: Fortran VEXCS ×{n_vexcs_per_step} per step ≈ {fortran_us_per_step:.0f} µs")
        print(f"  PyTorch full step: {pytorch_ms_per_step:.1f} ms = {pytorch_ms_per_step*1000:.0f} µs")
        print(f"  Ratio (PyTorch step / Fortran VEXCS): {pytorch_ms_per_step*1000/fortran_us_per_step:.0f}×")


if __name__ == "__main__":
    main()
