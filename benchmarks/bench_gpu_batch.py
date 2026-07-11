#!/usr/bin/env python3
"""GPU batch benchmark for surf_pytorch — measures throughput vs column count."""

import math, os, sys, time, argparse, gc
from pathlib import Path
from typing import List
import numpy as np
import torch
import xarray as xr

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


def _sq(a): return np.asarray(a.values).ravel()
def _ds(p): return xr.open_dataset(p)


def build_model_and_data(site, dtype=torch.float64, device="cpu"):
    """Build everything for a single-column run. Returns (model, static, state, forcing_dict)."""
    cli = _ds(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfclim_{site}_2022-2022.nc")
    c = {v: float(_sq(cli[v])[0]) for v in
         ["tvl","tvh","sotype","cvl","cvh","cu","landsea","geopot","z0m","lz0h","lat","lon"]}
    WET = 0.05; land = c["landsea"]; wet = WET * land; urban = c["cu"] * land
    vtotal = max(c["cvl"] + c["cvh"], 1e-12)
    scale = (land - wet - urban) / vtotal if vtotal > 1e-12 else 0.0
    low = c["cvl"] * scale; high = c["cvh"] * scale
    bare = land - wet - low - high
    tf = np.zeros(N_BASE_TILES); tf[2], tf[3], tf[5], tf[7] = wet, low, high, bare

    # Keep index-like tensors on CPU, move others to device
    static = Static(
        tile_frac_base=torch.tensor(tf, dtype=dtype, device=device).unsqueeze(0),
        land_frac=torch.tensor([land], dtype=dtype, device=device),
        orography=torch.tensor([c["geopot"]], dtype=dtype, device=device),
        soil_type=torch.tensor([c["sotype"]], dtype=torch.long),         # CPU - index
        veg_type=torch.tensor([c["tvl"]], dtype=torch.long),             # CPU - index
        soil_thickness=torch.tensor([0.07,0.21,0.72,1.89], dtype=dtype, device=device),
        emis=torch.tensor([0.99], dtype=dtype, device=device),
        z0m=torch.tensor([c["z0m"]], dtype=dtype, device=device),
        z0h=torch.tensor([math.exp(c["lz0h"])], dtype=dtype, device=device),
        z0q=torch.tensor([math.exp(c["lz0h"])], dtype=dtype, device=device),
        lat=torch.tensor([math.radians(c["lat"])], dtype=dtype, device=device),
        lon=torch.tensor([math.radians(c["lon"])], dtype=dtype, device=device),
    )

    ini = _ds(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfinit_{site}_2022-2022.nc")
    st = _sq(ini["SoilTemp"]); sm = _sq(ini["SoilMoist"])
    swe = float(_sq(ini["SWE"])[0]); can = float(_sq(ini["CanopInt"])[0])
    t0 = float(_sq(ini["AvgSurfT"])[0])
    state = SurfState(
        skin_t=torch.full((1, N_BASE_TILES), t0, dtype=dtype, device=device),
        soil_t=torch.tensor(st, dtype=dtype, device=device).unsqueeze(0),
        soil_w_liq=torch.tensor(sm, dtype=dtype, device=device).unsqueeze(0),
        snow_w=torch.tensor([swe], dtype=dtype, device=device),
        snow_frac=torch.zeros(1, dtype=dtype, device=device),
        skin_water=torch.tensor([can], dtype=dtype, device=device),
        sst=torch.tensor([t0], dtype=dtype, device=device),
    )

    met = _ds(INPUT_ROOT / "forcing" / "LS-PRACTICALS" / f"met_era5HT_{site}_2022-2022.nc")
    forcing = {"Tair": _sq(met["Tair"]), "Qair": _sq(met["Qair"]),
               "Wind_E": _sq(met["Wind_E"]), "Wind_N": _sq(met["Wind_N"]),
               "PSurf": _sq(met["PSurf"]), "SWdown": _sq(met["SWdown"]),
               "LWdown": _sq(met["LWdown"]), "Rainf": _sq(met["Rainf"]),
               "Snowf": _sq(met["Snowf"]), "time": met["time"].values}
    
    cfg = SurfConfig(dtype=dtype, validate=False)
    model = LandSurface(cfg)
    return model, static, state, forcing, c, cli


def _move_to_device(state, static, device):
    """Move all tensors in state and static to the given device."""
    def _mv(t):
        if isinstance(t, torch.Tensor):
            return t.to(device)
        return t
    state_d = {k: _mv(getattr(state, k)) for k in state.__dataclass_fields__}
    static_d = {}
    for k in static.__dataclass_fields__:
        val = getattr(static, k)
        static_d[k] = _mv(val)
    return SurfState(**state_d), Static(**static_d)


def _build_forcing(k, forcing, dtype, device, lat_deg, lon_deg, g, rsigma, emis, dt, zphista, doy):
    """Build a Forcing object for step k."""
    sw_down = float(forcing["SWdown"][k])
    lw_down = float(forcing["LWdown"][k])
    # NOTE: need ts_prev for net_lw, but we don't have it here. Use a fixed value.
    # For benchmark purposes, use a constant net_lw.
    net_lw = lw_down - 100.0  # rough
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
        rain=torch.tensor([0.0], dtype=dtype, device=device),
        snow=torch.tensor([0.0], dtype=dtype, device=device),
        dt=torch.tensor([dt], dtype=dtype, device=device),
    )


def benchmark_batch(batch_size: int, nday: int, device: str, dtype: torch.dtype):
    """Benchmark surf_pytorch with batch_size columns on GPU."""
    site = "CA-001"
    model, static, state, forcing, c, cli = build_model_and_data(site, dtype, "cpu")
    
    lat_deg = float(c["lat"]); lon_deg = float(c["lon"])
    g = float(PhysicalConstants().g)
    rsigma = float(PhysicalConstants().rsigma)
    emis = 0.99; dt = 3600.0
    zphista = float(cli["zphista"].values.ravel()[0]) if "zphista" in cli else 10.0
    
    # Move model & state to device
    if device != "cpu":
        model = model.to(device)
        state, static = _move_to_device(state, static, device)
    
    # Expand to batch_size
    def _batch(t, replicas):
        if isinstance(t, torch.Tensor) and t.ndim >= 1:
            return t.repeat(replicas, *([1]*(t.ndim-1)))
        return t
    
    ncols = batch_size
    state_b = SurfState(
        skin_t=state.skin_t.repeat(ncols, 1),
        soil_t=state.soil_t.repeat(ncols, 1),
        soil_w_liq=state.soil_w_liq.repeat(ncols, 1),
        snow_w=state.snow_w.repeat(ncols),
        snow_frac=state.snow_frac.repeat(ncols),
        skin_water=state.skin_water.repeat(ncols),
        sst=state.sst.repeat(ncols) if state.sst is not None else None,
    )
    
    static_b = Static(
        **{k: (getattr(static, k).repeat(ncols, *([1]*(getattr(static, k).ndim-1)))
               if isinstance(getattr(static, k), torch.Tensor) and getattr(static, k).ndim > 0 and k not in ("soil_type","veg_type")
               else getattr(static, k))
           for k in static.__dataclass_fields__}
    )
    
    n_forcing_hours = len(forcing["Tair"])
    n_steps = nday * 24
    
    # Warmup
    with torch.inference_mode():
        for _k in range(5):
            k = _k % n_forcing_hours
            doy = (_k % 365) + 1
            f = _build_forcing(k, forcing, dtype, device, lat_deg, lon_deg, g, rsigma, emis, dt, zphista, doy)
            # Expand forcing to batch
            f_b = Forcing(
                u=f.u.repeat(ncols), v=f.v.repeat(ncols),
                t=f.t.repeat(ncols), q=f.q.repeat(ncols),
                co2=f.co2.repeat(ncols),
                p_surf=f.p_surf.repeat(ncols),
                geom1=f.geom1.repeat(ncols),
                fr_so=f.fr_so.repeat(ncols),
                fr_th=f.fr_th.repeat(ncols),
                mu0=f.mu0.repeat(ncols),
                rain=f.rain.repeat(ncols),
                snow=f.snow.repeat(ncols),
                dt=f.dt.repeat(ncols),
            )
            state_b, _ = model(state_b, f_b, static_b)
    
    if device != "cpu":
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        for step in range(n_steps):
            k = step % n_forcing_hours
            doy = (step % 365) + 1
            f = _build_forcing(k, forcing, dtype, device, lat_deg, lon_deg, g, rsigma, emis, dt, zphista, doy)
            f_b = Forcing(
                u=f.u.repeat(ncols), v=f.v.repeat(ncols),
                t=f.t.repeat(ncols), q=f.q.repeat(ncols),
                co2=f.co2.repeat(ncols),
                p_surf=f.p_surf.repeat(ncols),
                geom1=f.geom1.repeat(ncols),
                fr_so=f.fr_so.repeat(ncols),
                fr_th=f.fr_th.repeat(ncols),
                mu0=f.mu0.repeat(ncols),
                rain=f.rain.repeat(ncols),
                snow=f.snow.repeat(ncols),
                dt=f.dt.repeat(ncols),
            )
            state_b, _ = model(state_b, f_b, static_b)
    
    if device != "cpu":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    sim_days_per_sec = nday / wall
    return wall, sim_days_per_sec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nday", type=int, default=30)
    args = p.parse_args()

    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    batch_sizes = [1, 5, 10, 20, 50, 100]
    
    print(f"\n{'='*70}")
    print(f"  surf_pytorch GPU Batch Benchmark  ({args.nday} days, {dtype})")
    print(f"  device: {device}")
    print(f"{'='*70}\n")
    
    results_cpu = {}
    results_gpu = {}
    
    # CPU single-column baseline
    print("CPU single-column baseline ...", flush=True)
    w, r = benchmark_batch(1, args.nday, "cpu", dtype)
    results_cpu[1] = r
    print(f"  {r:.1f} sim-days/s\n")

    # GPU batch
    for bs in batch_sizes:
        if bs == 1 and device != "cpu":
            print(f"GPU batch={bs} ...", flush=True)
        elif bs > 1:
            print(f"GPU batch={bs} ...", flush=True)
        try:
            w, r = benchmark_batch(bs, args.nday, device, dtype)
            results_gpu[bs] = r
            throughput = r * bs  # column-days per second
            print(f"  {r:.1f} sim-days/s  (×{bs} columns = {throughput:.0f} col-days/s)")
        except Exception as e:
            print(f"  FAILED: {e}")
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    cpu_base = results_cpu.get(1, results_gpu.get(1, 1.0))
    print(f"  {'Variant':<25s} {'sim-days/s':>12s} {'col-days/s':>12s} {'vs CPU':>10s}")
    print(f"  {'-'*60}")
    print(f"  {'CPU 1-col':<25s} {cpu_base:>12.1f} {cpu_base:>12.0f} {'1.0×':>10s}")
    for bs in sorted(results_gpu):
        r = results_gpu[bs]
        tp = r * bs
        print(f"  {'GPU '+str(bs)+'-col':<25s} {r:>12.1f} {tp:>12.0f} {tp/cpu_base:>9.1f}×")
    print()


if __name__ == "__main__":
    main()
