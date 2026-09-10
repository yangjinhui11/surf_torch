#!/usr/bin/env python3
"""Diagnose where and why the anchored-day closure integration diverges.

Replicates the recurrence of modis_slab_asym_train.py exactly (build-up day
under no_grad from the archived state, forecast day, lambda_s fixed), but
records per physics step and per column: the closure anomaly dT, the grid
sensible heat flux H used by the recurrence, the shortwave blend weight w,
the implied beta, and the radiative skin temperature.  The analysis then
reports the first step at which max|dT| crosses 10/100/1e4 K, the worst
column (lat/lon), its local-solar-time series around the onset, and the
per-step growth factor of |dT| compared with beta (pure relaxation would
give |dT_{n+1}| = beta |dT_n|; a larger factor quantifies the flux
feedback gain).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from modis_parameter_identifiability import (  # noqa: E402
    DT_SECONDS,
    FORCING_NAMES,
    PARAMETER_NAMES,
    STEPS_PER_DAY,
    restore_state,
)

RHO_CP = 1.2 * 1005.7
SW_REF = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--state-archive", type=Path, required=True)
    parser.add_argument("--day", type=int, default=131)
    parser.add_argument("--lambda-value", type=float, default=0.05)
    parser.add_argument("--tau-day-hours", type=float, required=True)
    parser.add_argument("--tau-night-hours", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tau_day_s = torch.tensor(args.tau_day_hours * 3600.0, dtype=torch.float64)
    tau_night_s = torch.tensor(args.tau_night_hours * 3600.0, dtype=torch.float64)
    lam_value = args.lambda_value

    os.environ["SURF_CASE"] = str(args.case)
    os.environ["SURF_CODE"] = str(args.source)
    os.environ["SURF_DATA"] = str(args.data)
    sys.path[:0] = [str(args.source), str(args.runner)]
    from diagnostic_scripts_compare_realcol import make_forcing  # noqa: E402
    from replay_full_domain_controlled import load_initial  # noqa: E402
    from run_full_domain_offline import interpolated_runtime  # noqa: E402
    from surf_pytorch import LandSurface, SurfConfig  # noqa: E402

    metadata = json.loads((args.case / "metadata.json").read_text())
    nvalid = int(metadata["nvalid"])
    lat = np.asarray(np.load(args.data / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(args.data / "lon.npy", mmap_mode="r"), dtype=np.float64)
    forcing_full = {name: np.load(args.data / f"{name}.npy", mmap_mode="r") for name in FORCING_NAMES}

    archive = np.load(args.state_archive, mmap_mode="r")
    archive_steps = np.asarray(archive["step"], dtype=np.int64)
    archive_step = (args.day - 1) * STEPS_PER_DAY  # build-up day, then forecast day
    matching = np.flatnonzero(archive_steps == archive_step)
    if matching.size != 1:
        raise ValueError(f"no unique archive record for step {archive_step}")

    base_state, static = load_initial(args.case / "restartin.nc", nvalid, args.device, lat, lon)

    config = SurfConfig(
        dtype=torch.float64,
        device=args.device,
        differentiable=True,
        validate=False,
        use_farquhar=True,
        use_ags=False,
        levgen=True,
        lelwtl=True,
        lelaiv=False,
        vup_boundary_mode="offline",
        learnable_veg=PARAMETER_NAMES,
        learnable_soil=("lambdadry", "lamsat1", "rcsoil"),
    )
    model = LandSurface(config).to(args.device)

    forcing_options = SimpleNamespace(
        shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
        shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
        stress_u=None, stress_v=None, vup=None,
    )
    sw_ref_t = torch.tensor(SW_REF, dtype=torch.float64, device=args.device)
    dt_const = torch.as_tensor(DT_SECONDS, dtype=torch.float64, device=args.device)
    tau_day_t = tau_day_s.to(args.device)
    tau_night_t = tau_night_s.to(args.device)

    total_steps = 2 * STEPS_PER_DAY
    records = {
        name: np.zeros((total_steps, nvalid), dtype=np.float64)
        for name in ("dt_air", "heat", "skin", "weight", "beta")
    }
    cache = {"heat": None, "dt_air": None}

    def advance(current_state, physical_step: int):
        fraction = model._tile_frac(static, config.n_tile, state=current_state)
        runtime = interpolated_runtime(forcing_full, physical_step, DT_SECONDS, lat, lon, args.device)
        sw = runtime.swdown_W_m2[1].clamp(min=0.0)
        weight = sw / (sw + sw_ref_t)
        inv_tau = weight / tau_day_t + (1.0 - weight) / tau_night_t
        beta = torch.exp(-dt_const * inv_tau)
        emission = model._surface_emissivity(
            current_state, static, fraction, torch.float64, args.device)
        step_static = replace(static, emis=emission)
        forcing = make_forcing(runtime, 0, current_state, step_static, forcing_options,
                               tile_frac=fraction)
        precipitation = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
        forcing = replace(
            forcing,
            rain=precipitation * (1.0 - runtime.precip_snow_phase),
            snow=precipitation * runtime.precip_snow_phase,
        )
        if cache["dt_air"] is not None:
            forcing = replace(forcing, t=forcing.t + cache["dt_air"])
        new_state = model(current_state, forcing, step_static)[0]
        with torch.no_grad():
            cache["heat"] = (fraction * model._last_pahfsti).sum(dim=1).detach().clone()
        previous = cache["dt_air"] if cache["dt_air"] is not None \
            else torch.zeros_like(cache["heat"])
        cache["dt_air"] = beta * previous + (1.0 - beta) * (-lam_value * cache["heat"])
        return new_state

    state = restore_state(copy.deepcopy(base_state), static, model, archive,
                          int(matching[0]), slice(None), args.device)
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(total_steps):
            state = advance(state, archive_step + step)
            records["dt_air"][step] = cache["dt_air"].cpu().numpy()
            records["heat"][step] = cache["heat"].cpu().numpy()
            records["skin"][step] = state.radiative_skin_t.cpu().numpy()
    # weight/beta only depend on the forcing; fill them with a cheap second
    # pass over the interpolated shortwave (no model calls).
    for step in range(total_steps):
        runtime = interpolated_runtime(forcing_full, archive_step + step, DT_SECONDS,
                                       lat, lon, args.device)
        sw = np.asarray(runtime.swdown_W_m2[1].cpu())
        sw = np.clip(sw, 0.0, None)
        weight = sw / (sw + SW_REF)
        inv_tau = weight / (args.tau_day_hours * 3600.0) + (1.0 - weight) / (args.tau_night_hours * 3600.0)
        records["weight"][step] = weight
        records["beta"][step] = np.exp(-DT_SECONDS * inv_tau)
    elapsed = time.perf_counter() - started

    dt_air = records["dt_air"]
    per_step_max = np.max(np.abs(dt_air), axis=1)
    thresholds = (10.0, 100.0, 1e4, 1e8, float("inf"))
    first_cross = {}
    for threshold in thresholds:
        hits = np.flatnonzero(per_step_max > threshold)
        first_cross[str(threshold)] = int(hits[0]) if hits.size else None

    worst_column = int(np.argmax(np.max(np.abs(dt_air), axis=0)))
    series = {
        name: records[name][:, worst_column].tolist() for name in records
    }
    local_hour = ((np.arange(total_steps) * 0.5) + lon[worst_column] / 15.0) % 24.0
    growth = []
    for step in range(1, total_steps):
        prev = abs(dt_air[step - 1, worst_column])
        curr = abs(dt_air[step, worst_column])
        growth.append(curr / prev if prev > 1e-6 else None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        dt_air=dt_air, heat=records["heat"], skin=records["skin"],
        weight=records["weight"], beta=records["beta"],
        lat=lat, lon=lon, steps=np.arange(total_steps),
    )
    summary = {
        "config": {
            "day": args.day, "lambda_s": lam_value,
            "tau_day_h": args.tau_day_hours, "tau_night_h": args.tau_night_hours,
            "archive_step": int(archive_step), "total_steps": total_steps,
        },
        "first_cross": first_cross,
        "per_step_max_abs_dt_first40": per_step_max[:40].tolist(),
        "worst_column": {
            "index": worst_column, "lat": float(lat[worst_column]),
            "lon": float(lon[worst_column]),
            "max_abs_dt": float(np.max(np.abs(dt_air[:, worst_column]))),
            "local_hour_at_max": float(local_hour[int(np.argmax(np.abs(dt_air[:, worst_column])))]),
        },
        "worst_column_series": series,
        "worst_column_local_hours": local_hour.tolist(),
        "worst_column_growth_factors": growth,
        "elapsed_s": elapsed,
    }
    args.output.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"done": str(args.output), "first_cross": first_cross,
                      "worst": summary["worst_column"], "elapsed_s": round(elapsed, 1)}))


if __name__ == "__main__":
    main()
