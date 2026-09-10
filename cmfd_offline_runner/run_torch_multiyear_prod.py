#!/usr/bin/env python3
"""Autonomous Torch offline SURF closure over continuous CMFD years, with diagnostics.

Chains the per-year CMFD datasets (year-boundary forcing linked to the next
year, as in run_multiyear_spinup.py) while saving the 10-day diagnostic
snapshots used by the autonomous Fortran comparison (as in
run_full_domain_offline.py).  Saved ``step`` values are cumulative across
years so that multi-year Fortran records can be matched directly.

Environment:
  SURF_CASE            output directory
  SURF_MULTIYEAR_DATA  per-year dataset root (default processed/multiyear_2009_2018)
  SURF_YEARS           START-END (default 2009:2018)
  SURF_INITIAL         shared restart (default initial_states/native_20d_terminal_complete.nc)
  SURF_OUTPUT_STRIDE   diagnostic stride in physics steps (default 480 = 10 days)
  SURF_PHYSICS_DT      default 1800 s
  SURF_DEVICE          default cuda
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

CODE = Path(os.environ["SURF_CODE"])
CASE = Path(os.environ["SURF_CASE"])
sys.path[:0] = [str(CODE), str(Path(__file__).parent)]

from run_full_domain_offline import solar_cosine_at_time  # noqa: E402
from replay_full_domain_controlled import DTYPE, load_initial  # noqa: E402
from diagnostic_scripts_compare_realcol import make_forcing  # noqa: E402
from surf_pytorch import LandSurface, SurfConfig  # noqa: E402

FORCING_DT = 10_800.0
DIAG_FIELDS = ("step", "skin_t", "soil_t", "soil_w", "snow_w", "snow_t", "snow_rho", "snow_alb")


def runtime_for_step(
    arrays: dict[str, np.ndarray], next_arrays: dict[str, np.ndarray], step: int,
    dt: float, lat: np.ndarray, lon: np.ndarray, device: str,
) -> SimpleNamespace:
    """OSM forcing contract with chained year-boundary records.

    Replicates ``run_full_domain_offline.interpolated_runtime`` (the contract
    validated against the Fortran driver): state meteorology is linearly
    interpolated at the physics-step boundaries, radiation and precipitation
    are held at the next 3-hourly record for the whole interval, the rain/snow
    split is fixed once per record from that record's float32 air temperature,
    and the solar geometry is evaluated at the held record's nominal time.
    Where that runner wraps to the first record of the same (cyclic) year,
    this version links to the first records of the following year's dataset,
    matching the chained per-year Fortran forcing files (``next:YYYY``).
    """
    start = step * dt
    nrecord = int(next(iter(arrays.values())).shape[0])

    def raw(source: dict[str, np.ndarray], name: str, index: int):
        # OSM reads f4 forcing records and interpolates in REAL8.  When the
        # forcing is preloaded to device (SURF_PRELOAD_FORCING=1) the slice is
        # a float64 torch tensor already on-device; otherwise read the f4 npy
        # record and promote to REAL8 -- both give the identical float64.
        value = source[name]
        if isinstance(value, torch.Tensor):
            return value[index].to(torch.float64)
        return np.asarray(value[index], dtype=np.float32).astype(np.float64)

    def sample(name: str, seconds: float) -> np.ndarray:
        record = min(int(seconds // FORCING_DT), nrecord - 1)
        alpha = (seconds - record * FORCING_DT) / FORCING_DT
        first = raw(arrays, name, record)
        if record + 1 >= nrecord:
            second = raw(next_arrays, name, record + 1 - nrecord)
        else:
            second = raw(arrays, name, record + 1)
        return first + alpha * (second - first)

    held_record = int(start // FORCING_DT) + 1
    from_next = held_record >= nrecord
    held_index = held_record - nrecord if from_next else held_record
    held_source = next_arrays if from_next else arrays

    held_fluxes = {"swdown_W_m2", "lwdown_W_m2", "precip_kg_m2_s"}
    runtime = SimpleNamespace(dt=dt)
    for name in arrays:
        if name in held_fluxes:
            value = raw(held_source, name, held_index)
            window = torch.stack((value, value)) if isinstance(value, torch.Tensor) else np.stack((value, value))
        else:
            a, b = sample(name, start), sample(name, start + dt)
            window = torch.stack((a, b)) if isinstance(a, torch.Tensor) else np.stack((a, b))
        setattr(runtime, name, torch.as_tensor(window, dtype=DTYPE, device=device))
    phase = raw(held_source, "tair_K", held_index) <= 273.15
    runtime.precip_snow_phase = torch.as_tensor(
        phase.cpu().numpy() if isinstance(phase, torch.Tensor) else phase,
        dtype=DTYPE, device=device)
    flux_time = (int(start // FORCING_DT) + 1) * FORCING_DT
    mu0 = solar_cosine_at_time(lat, lon, flux_time)
    runtime.mu0 = torch.as_tensor(np.stack((mu0, mu0)), dtype=DTYPE, device=device)
    runtime.zero = torch.zeros(lat.size, dtype=DTYPE, device=device)
    runtime.co2 = torch.full((lat.size,), 6.0e-4, dtype=DTYPE, device=device)
    runtime.dt_tensor = torch.tensor(dt, dtype=DTYPE, device=device)
    runtime.sigma = torch.tensor(5.6705085385054389e-08, dtype=DTYPE, device=device)
    return runtime


def main() -> None:
    root = Path(os.environ.get("SURF_MULTIYEAR_DATA", "/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018"))
    initial = Path(os.environ.get("SURF_INITIAL", "/data/yangjinhui/surf_pytorch/surf_paper/experiments/initial_states/native_20d_terminal_complete.nc"))
    years = [int(v) for v in os.environ.get("SURF_YEARS", "2009:2018").replace(":", "-").split("-")]
    years = list(range(years[0], years[1] + 1))
    stride = int(os.environ.get("SURF_OUTPUT_STRIDE", "480"))
    dt = float(os.environ.get("SURF_PHYSICS_DT", "1800"))
    device = os.environ.get("SURF_DEVICE", "cuda")
    if FORCING_DT % dt:
        raise ValueError("SURF_PHYSICS_DT must divide 10800 s")
    CASE.mkdir(parents=True, exist_ok=True)

    first = root / f"cmfd_valid_columns_{years[0]}_dataset"
    metadata = json.loads((first / "metadata.json").read_text())
    nvalid = int(metadata["ncols"])
    lat = np.asarray(np.load(first / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(first / "lon.npy", mmap_mode="r"), dtype=np.float64)

    state, static = load_initial(initial, nvalid, device, lat, lon)
    # SURFEXCDRIVER_CTL resets tiled PTSKTI from PTSKM1M at KSTEP=0 and
    # restores the water tile from PSST (offline driver restart contract).
    initial_skin_t = state.radiative_skin_t.unsqueeze(-1).expand_as(state.skin_t).clone()
    initial_skin_t[:, 0] = state.sst
    state = replace(state, skin_t=initial_skin_t)

    config = SurfConfig(dtype=DTYPE, device=device, validate=False, use_farquhar=True,
                        use_ags=False, levgen=True, lelwtl=True, lelaiv=False,
                        vup_boundary_mode="offline")
    model = LandSurface(config).to(device)
    args = SimpleNamespace(shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
                           shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
                           stress_u=None, stress_v=None, vup=None)

    outputs: dict[str, list] = {name: [] for name in DIAG_FIELDS}

    def save_output(global_step: int) -> None:
        outputs["step"].append(np.array(global_step, dtype=np.int64))
        outputs["skin_t"].append(state.skin_t.cpu().numpy())
        outputs["soil_t"].append(state.soil_t.cpu().numpy())
        outputs["soil_w"].append((state.soil_w_liq * static.soil_thickness * 1000.0).cpu().numpy())
        outputs["snow_w"].append(state.snow_w.cpu().numpy())
        outputs["snow_t"].append(state.snow_t.cpu().numpy())
        outputs["snow_rho"].append(state.snow_rho.cpu().numpy())
        outputs["snow_alb"].append(state.snow_alb.cpu().numpy())

    annual = []
    global_step = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for position, year in enumerate(years):
            data = root / f"cmfd_valid_columns_{year}_dataset"
            next_data = root / f"cmfd_valid_columns_{years[(position + 1) % len(years)]}_dataset"
            arrays = {n: np.load(data / f"{n}.npy", mmap_mode="r") for n in
                      ("tair_K", "qair_kg_kg", "psurf_Pa", "swdown_W_m2", "lwdown_W_m2", "wind_m_s", "precip_kg_m2_s")}
            next_arrays = {n: np.load(next_data / f"{n}.npy", mmap_mode="r") for n in arrays}
            if os.environ.get("SURF_PRELOAD_FORCING") == "1":
                # 一次性把全年 f4 记录载入 device；逐步插值直接在内存切片，
                # 消除每步 ~14 次 npy mmap 行读取（全域 ~97709 列时是主瓶颈）。
                # 数据仍是 float32，物理插值 promote 到 REAL8，与 mmap 路径位一致。
                def _pre(src):
                    return {n: torch.as_tensor(np.array(v, dtype=np.float32, copy=True),
                                               dtype=DTYPE, device=device)
                            for n, v in src.items()}
                arrays = _pre(arrays)
                next_arrays = _pre(next_arrays)
            nsteps = int(next(iter(arrays.values())).shape[0] * FORCING_DT / dt)
            year_started = time.perf_counter()
            for step in range(nsteps):
                fraction = model._tile_frac(static, config.n_tile, state=state)
                runtime = runtime_for_step(arrays, next_arrays, step, dt, lat, lon, device)
                emission = model._surface_emissivity(state, static, fraction, DTYPE, device)
                forcing = make_forcing(runtime, 0, state, replace(static, emis=emission), args, tile_frac=fraction)
                amount = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
                forcing = replace(forcing, rain=amount * (1.0 - runtime.precip_snow_phase),
                                  snow=amount * runtime.precip_snow_phase)
                state, _ = model(state, forcing, replace(static, emis=emission))
                global_step += 1
                if global_step % stride == 0:
                    save_output(global_step)
                if global_step % int(86_400 / dt) == 0:
                    finite = torch.isfinite(torch.cat([
                        getattr(state, name).reshape(-1)
                        for name in ("skin_t", "soil_t", "soil_w_liq", "snow_w", "snow_t", "snow_rho", "snow_alb")
                    ])).all()
                    if not bool(finite.cpu()):
                        raise FloatingPointError(f"non-finite state at {year} step {step + 1}")
            if global_step % stride != 0 or outputs["step"][-1] != global_step:
                save_output(global_step)
            torch.cuda.synchronize()
            diag = {
                "year": year, "steps": nsteps, "global_step": global_step,
                "seconds": time.perf_counter() - year_started,
                "mean_swe_kg_m2": float(state.snow_w.mean().cpu()),
                "max_swe_kg_m2": float(state.snow_w.max().cpu()),
                "mean_soil_water_kg_m2": float((state.soil_w_liq * static.soil_thickness * 1000.0).sum(dim=1).mean().cpu()),
                "records": len(outputs["step"]),
            }
            annual.append(diag)
            (CASE / "annual_diagnostics.json").write_text(json.dumps(annual, indent=2) + "\n")
            np.savez(CASE / "torch_year_terminal_states.npz",
                     **{f"year_{year}": v for f, v in (
                         ("skin_t", state.skin_t.cpu().numpy()), ("soil_t", state.soil_t.cpu().numpy()),
                         ("soil_w", (state.soil_w_liq * static.soil_thickness * 1000.0).cpu().numpy()),
                         ("snow_w", state.snow_w.cpu().numpy()), ("snow_t", state.snow_t.cpu().numpy()),
                         ("snow_rho", state.snow_rho.cpu().numpy()), ("snow_alb", state.snow_alb.cpu().numpy()))})
            print(json.dumps(diag), flush=True)
            np.savez(CASE / "torch_offline_daily.npz",
                     **{key: np.stack(value) for key, value in outputs.items()})

    summary = {
        "years": years, "nvalid": nvalid, "nsteps": global_step, "device": device,
        "physics_dt": dt, "output_stride": stride, "initial": str(initial),
        "saved_steps": [int(s) for s in outputs["step"]],
        "total_seconds": time.perf_counter() - started, "annual": annual,
    }
    np.savez(CASE / "torch_offline_daily.npz", **{key: np.stack(value) for key, value in outputs.items()})
    (CASE / "torch_offline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("years", "nsteps", "total_seconds")}), flush=True)


if __name__ == "__main__":
    main()
