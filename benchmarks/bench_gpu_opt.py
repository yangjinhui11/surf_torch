#!/usr/bin/env python3
"""surf_pytorch GPU batch benchmark — measures throughput vs column count.

Runs the full physics chain for nday days with batch_size columns,
all computation on GPU.  Compares:
  - CPU single-column baseline.
  - GPU 1 / 10 / 100 / 1k / 10k columns.
"""

import math, os, sys, time, argparse, gc
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import warnings; warnings.filterwarnings("ignore")

from surf_pytorch.config import SurfConfig, N_BASE_TILES
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.forcing import solar_mu0
from surf_pytorch.model import LandSurface
from surf_pytorch.constants import PhysicalConstants
import xarray as xr

INPUT_ROOT = REPO.parent / "thirdpaty" / "ecland-main" / "tutorials" / "ecland_input_tutorial"

def _sq(a): return np.asarray(a.values).ravel()
def _ds(p): return xr.open_dataset(p)


def build_1col(site="CA-001", dtype=torch.float64, device="cpu", *, diagnostics=False):
    """Build model + static + initial state for a single column."""
    cli = _ds(str(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfclim_{site}_2022-2022.nc"))
    c = {v: float(_sq(cli[v])[0]) for v in
         ["tvl","tvh","sotype","cvl","cvh","cu","landsea","geopot","z0m","lz0h","lat","lon"]}
    WET = 0.05; land = c["landsea"]; wet = WET * land; urban = c["cu"] * land
    vtotal = max(c["cvl"] + c["cvh"], 1e-12)
    scale = (land - wet - urban) / vtotal if vtotal > 1e-12 else 0.0
    low = c["cvl"] * scale; high = c["cvh"] * scale
    bare = land - wet - low - high
    tf = np.zeros(N_BASE_TILES); tf[2], tf[3], tf[5], tf[7] = wet, low, high, bare

    def _t(val, **kw):
        return torch.tensor(val, dtype=dtype, device=device, **kw)

    static = Static(
        tile_frac_base=_t(tf).unsqueeze(0),
        land_frac=_t([land]),
        orography=_t([c["geopot"]]),
        soil_type=torch.tensor([c["sotype"]], dtype=torch.long, device=device),
        veg_type=torch.tensor([c["tvl"]], dtype=torch.long, device=device),
        soil_thickness=_t([0.07,0.21,0.72,1.89]),
        emis=_t([0.99]),
        z0m=_t([c["z0m"]]), z0h=_t([math.exp(c["lz0h"])]),
        z0q=_t([math.exp(c["lz0h"])]),
        lat=_t([math.radians(c["lat"])]), lon=_t([math.radians(c["lon"])]),
    )

    ini = _ds(str(INPUT_ROOT / "clim" / "LS-PRACTICALS" / f"surfinit_{site}_2022-2022.nc"))
    st = _sq(ini["SoilTemp"]); sm = _sq(ini["SoilMoist"])
    swe = float(_sq(ini["SWE"])[0]); can = float(_sq(ini["CanopInt"])[0])
    t0 = float(_sq(ini["AvgSurfT"])[0])
    state = SurfState(
        skin_t=_t([[t0]*N_BASE_TILES]),
        soil_t=_t(st).unsqueeze(0),
        soil_w_liq=_t(sm).unsqueeze(0),
        snow_w=_t([swe]),
        snow_frac=_t([0.0]),
        skin_water=_t([can]),
        sst=_t([t0]),
    )

    met = _ds(str(INPUT_ROOT / "forcing" / "LS-PRACTICALS" / f"met_era5HT_{site}_2022-2022.nc"))
    forcing = {"Tair": _sq(met["Tair"]), "Qair": _sq(met["Qair"]),
               "Wind_E": _sq(met["Wind_E"]), "Wind_N": _sq(met["Wind_N"]),
               "PSurf": _sq(met["PSurf"]), "SWdown": _sq(met["SWdown"]),
               "LWdown": _sq(met["LWdown"]), "Rainf": _sq(met["Rainf"]),
               "Snowf": _sq(met["Snowf"])}

    cfg = SurfConfig(dtype=dtype, validate=False, compute_diagnostics=diagnostics)
    model = LandSurface(cfg)
    if device != "cpu":
        model = model.to(device=device)
    return model, static, state, forcing, c, cli


def _make_forcing(k, forcing, dtype, device, lat_deg, lon_deg, g, rsigma, emis, dt, zphista, doy, skin_t):
    """Build a Forcing object for step k. All tensors on device."""
    sw_down = float(forcing["SWdown"][k])
    lw_down = float(forcing["LWdown"][k])
    ts = float(skin_t.detach().mean().cpu().item())
    lw_up = emis * rsigma * ts ** 4
    net_lw = lw_down - lw_up
    noon = 12.0 - lon_deg / 15.0
    mu0_val = solar_mu0(math.radians(lat_deg), math.radians(lon_deg), doy, noon)

    def _t(**kw):
        return torch.tensor(**kw, dtype=dtype, device=device)

    return Forcing(
        u=_t(data=[float(forcing["Wind_E"][k])]),
        v=_t(data=[float(forcing["Wind_N"][k])]),
        t=_t(data=[float(forcing["Tair"][k])]),
        q=_t(data=[float(forcing["Qair"][k])]),
        co2=_t(data=[6e-4]),
        p_surf=_t(data=[forcing["PSurf"][k]]),
        geom1=_t(data=[g * zphista]),
        fr_so=_t(data=[sw_down * dt]),
        fr_th=_t(data=[net_lw * dt]),
        mu0=_t(data=[mu0_val]),
        rain=_t(data=[float(forcing["Rainf"][k]) * dt]),
        snow=_t(data=[float(forcing["Snowf"][k]) * dt]),
        dt=_t(data=[dt]),
    )


def _expand_1to(ncols, state, static):
    """Expand single-column state/static to ncols replicated columns."""
    r = ncols
    state_b = SurfState(
        skin_t=state.skin_t.repeat(r, 1),
        soil_t=state.soil_t.repeat(r, 1),
        soil_w_liq=state.soil_w_liq.repeat(r, 1),
        snow_w=state.snow_w.repeat(r),
        snow_frac=state.snow_frac.repeat(r),
        skin_water=state.skin_water.repeat(r),
        sst=state.sst.repeat(r) if state.sst is not None else None,
    )
    sd = {}
    for f in static.__dataclass_fields__:
        val = getattr(static, f)
        if isinstance(val, torch.Tensor) and val.ndim >= 1:
            sd[f] = val.repeat(r, *([1]*(val.ndim-1)))
        else:
            sd[f] = val
    static_b = Static(**sd)
    return state_b, static_b


def benchmark(batch_size: int, nday: int, dtype=torch.float64, device="cuda", *, diagnostics=False):
    """Run batch_size columns for nday days on device."""
    model, static, state, forcing, c, cli = build_1col(
        "CA-001", dtype, device, diagnostics=diagnostics,
    )
    lat_deg = float(c["lat"]); lon_deg = float(c["lon"])
    g = float(PhysicalConstants().g)
    rsigma = float(PhysicalConstants().rsigma)
    emis = 0.99; dt = 3600.0
    zphista = float(cli["zphista"].values.ravel()[0]) if "zphista" in cli else 10.0

    state_b, static_b = _expand_1to(batch_size, state, static)
    n_forcing_hours = len(forcing["Tair"])
    n_steps = nday * 24

    # Warmup: 5 steps
    with torch.inference_mode():
        for _k in range(5):
            k = _k % n_forcing_hours; doy = (_k % 365) + 1
            f = _make_forcing(k, forcing, dtype, device, lat_deg, lon_deg, g, rsigma, emis, dt, zphista, doy, state_b.skin_t)
            # Expand forcing from 1→N
            ncols = batch_size
            f_b = Forcing(
                u=f.u.repeat(ncols), v=f.v.repeat(ncols),
                t=f.t.repeat(ncols), q=f.q.repeat(ncols),
                co2=f.co2.repeat(ncols), p_surf=f.p_surf.repeat(ncols),
                geom1=f.geom1.repeat(ncols),
                fr_so=f.fr_so.repeat(ncols), fr_th=f.fr_th.repeat(ncols),
                mu0=f.mu0.repeat(ncols), rain=f.rain.repeat(ncols),
                snow=f.snow.repeat(ncols), dt=f.dt,  # dt stays 1-element
            )
            state_b, _ = model(state_b, f_b, static_b)

    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for step in range(n_steps):
            k = step % n_forcing_hours; doy = (step % 365) + 1
            f = _make_forcing(k, forcing, dtype, device, lat_deg, lon_deg, g, rsigma, emis, dt, zphista, doy, state_b.skin_t)
            ncols = batch_size
            f_b = Forcing(
                u=f.u.repeat(ncols), v=f.v.repeat(ncols),
                t=f.t.repeat(ncols), q=f.q.repeat(ncols),
                co2=f.co2.repeat(ncols), p_surf=f.p_surf.repeat(ncols),
                geom1=f.geom1.repeat(ncols),
                fr_so=f.fr_so.repeat(ncols), fr_th=f.fr_th.repeat(ncols),
                mu0=f.mu0.repeat(ncols), rain=f.rain.repeat(ncols),
                snow=f.snow.repeat(ncols), dt=f.dt,  # dt stays 1-element
            )
            state_b, _ = model(state_b, f_b, static_b)

    if device != "cpu":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    sim_days_per_sec = nday / wall
    col_days_per_sec = sim_days_per_sec * batch_size
    return wall, sim_days_per_sec, col_days_per_sec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nday", type=int, default=10)
    p.add_argument("--max-batch", type=int, default=1000)
    p.add_argument("--device", default=None, help="CUDA device visible to this process")
    p.add_argument("--diagnostics", action="store_true", help="Include per-step diagnostic packing")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        return
    device = args.device or "cuda"
    dtype = torch.float64

    print(f"\n{'='*70}")
    print(f"  surf_pytorch GPU Batch Benchmark")
    print(f"  device={device}  dtype={dtype}  nday={args.nday}")
    print(f"{'='*70}\n")

    results = {}

    # CPU baseline
    print("CPU  single-column  float64 ...", flush=True)
    try:
        w, r, tp = benchmark(1, args.nday, dtype, "cpu", diagnostics=args.diagnostics)
        results["CPU-1col"] = tp
        print(f"  {r:.1f} sim-days/s  ({tp:.0f} col-days/s)\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")

    # GPU batch sizes
    for bs in [1, 10, 50, 100, 500, 1000, 2000]:
        if bs > args.max_batch:
            break
        print(f"GPU  batch={bs:<5d}  float64 ...", flush=True)
        try:
            w, r, tp = benchmark(bs, args.nday, dtype, device, diagnostics=args.diagnostics)
            results[f"GPU-{bs}col"] = tp
            print(f"  {r:.1f} sim-days/s  ({tp:.0f} col-days/s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            break
        gc.collect(); torch.cuda.empty_cache()

    # Also try float32 for last few
    for bs in [1000]:
        print(f"GPU  batch={bs:<5d}  float32 ...", flush=True)
        try:
            w, r, tp = benchmark(bs, args.nday, torch.float32, device, diagnostics=args.diagnostics)
            results[f"GPU-{bs}col-f32"] = tp
            print(f"  {r:.1f} sim-days/s  ({tp:.0f} col-days/s)")
        except Exception as e:
            print(f"  FAILED: {e}")
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    cpu_base = results.get("CPU-1col", 1.0)
    print(f"  {'Config':<20s} {'col-days/s':>12s} {'vs CPU':>10s} {'GPU util':>10s}")
    print(f"  {'-'*55}")
    for label in sorted(results, key=lambda x: float(results[x]) if isinstance(results[x], (int,float)) else 0):
        tp = results[label]
        print(f"  {label:<20s} {tp:>12.0f} {tp/cpu_base:>9.1f}×")
    print()


if __name__ == "__main__":
    main()
