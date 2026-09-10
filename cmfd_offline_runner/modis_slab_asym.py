#!/usr/bin/env python3
"""Day/night-asymmetric relaxation times for the surface-layer closure.

Same day-anchored protocol as modis_slab_maps.py, but the relaxation time
of the anomaly recurrence is no longer tied to lambda_s through a fixed
slab depth.  Two timescales are used,

    tau_t = ( w/tau_day + (1-w)/tau_night )^-1 ,
    w     = S / (S + S_ref) ,   S = surface downward SW at the step,

so the anomaly keeps its daytime amplitude (lambda_s unchanged) while its
night-time memory is a free parameter.  The trigger is the same surface
shortwave the physics uses (runtime.swdown_W_m2 at the flux-shifted
index), so the transition follows sunrise/sunset and clouds, not a clock.

Sweeps tau_day x tau_night on a grid plus the tied one-parameter baseline
(tau = lambda_s * rho*cp * 600 m) and the prior (lambda_s = 0).  For every
configuration it records, per window and pass, area-weighted bias/RMSE at
the 30-min sampling rule, per-day stats, and the forecast-day diurnal
composite of the anomaly and skin temperature binned by LOCAL SOLAR TIME
(the return schedule).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from netCDF4 import Dataset

from modis_parameter_identifiability import (  # noqa: E402
    DT_SECONDS,
    FORCING_NAMES,
    PARAMETER_NAMES,
    PERIODS,
    STEPS_PER_DAY,
    restore_state,
    target_steps,
)

RHO_CP = 1.2 * 1005.7  # J m-3 K-1
SW_REF = 50.0          # W m-2, half-weight point of the day/night blend
N_LOCAL_BINS = 48      # 0.5-h bins of local solar time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--state-archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, default=0.05)
    parser.add_argument("--calibration-days", default="271,281,291,301,311,321,331,341,351,361")
    parser.add_argument("--validation-days", default="91,101,111,121,131,141,151,161,171,181")
    parser.add_argument("--tau-day-hours", default="3,5,10")
    parser.add_argument("--tau-night-hours", default="1,2,3,5")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def load_observations(path: Path, days: list[int]) -> dict:
    with Dataset(path) as observations:
        data = {}
        for period in PERIODS:
            data[period] = {
                "lst": np.asarray(observations[f"lst_{period}"][days, :], dtype=np.float64),
                "time": np.asarray(observations[f"lst_{period}_view_time_utc"][days, :], dtype=np.float64),
                "strict": np.asarray(observations[f"lst_{period}_strict"][days, :]) == 1,
            }
    return data


def single_day(observed: dict, index: int) -> dict:
    return {
        period: {key: value[index: index + 1] for key, value in observed[period].items()}
        for period in PERIODS
    }


def main() -> None:
    args = parse_args()
    calibration_days = [int(v) for v in args.calibration_days.split(",")]
    validation_days = [int(v) for v in args.validation_days.split(",")]
    tau_day_hours = [float(v) for v in args.tau_day_hours.split(",")]
    tau_night_hours = [float(v) for v in args.tau_night_hours.split(",")]
    lam_value = args.lambda_value
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

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

    observed_all = {
        "calibration": load_observations(args.observations, calibration_days),
        "validation": load_observations(args.observations, validation_days),
    }

    archive = np.load(args.state_archive, mmap_mode="r")
    archive_steps = np.asarray(archive["step"], dtype=np.int64)

    base_state, static = load_initial(
        args.case / "restartin.nc", nvalid, args.device, lat, lon,
    )

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

    # tau_day_s = None -> prior (no closure); tau_day == tau_night -> scalar beta.
    configurations: dict[str, dict] = {"prior": {"tau_day_s": None, "tau_night_s": None}}
    tied_tau_s = lam_value * RHO_CP * 600.0
    configurations[f"tied_{tied_tau_s / 3600.0:.3f}h"] = {
        "tau_day_s": tied_tau_s, "tau_night_s": tied_tau_s,
    }
    for tau_day in tau_day_hours:
        for tau_night in tau_night_hours:
            configurations[f"td{tau_day:g}h_tn{tau_night:g}h"] = {
                "tau_day_s": tau_day * 3600.0, "tau_night_s": tau_night * 3600.0,
            }

    area_weight = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float64, device=args.device)
    weight_sum = float(area_weight.sum())
    local_bin = torch.empty((STEPS_PER_DAY, nvalid), dtype=torch.int64, device=args.device)
    for step in range(STEPS_PER_DAY):
        utc_mid_hours = (step + 0.5) * (DT_SECONDS / 3600.0)
        local_mid = np.mod(utc_mid_hours + lon / 15.0, 24.0)
        local_bin[step] = torch.as_tensor(
            np.minimum((local_mid / 24.0 * N_LOCAL_BINS).astype(np.int64), N_LOCAL_BINS - 1),
            device=args.device,
        )

    def make_advance(tau_day_s, tau_night_s):
        active = tau_day_s is not None
        if active:
            scalar_beta = (
                math.exp(-DT_SECONDS / tau_day_s) if tau_day_s == tau_night_s else None
            )
            tau_day_t = torch.tensor(tau_day_s, dtype=torch.float64, device=args.device)
            tau_night_t = torch.tensor(tau_night_s, dtype=torch.float64, device=args.device)
        else:
            scalar_beta = None
            tau_day_t = tau_night_t = None
        sw_ref_t = torch.tensor(SW_REF, dtype=torch.float64, device=args.device)
        cache = {"dt_air": None}

        def get_anomaly():
            if cache["dt_air"] is None:
                return torch.zeros(nvalid, dtype=torch.float64, device=args.device)
            return cache["dt_air"]

        def advance(current_state, physical_step: int):
            fraction = model._tile_frac(static, config.n_tile, state=current_state)
            runtime = interpolated_runtime(
                forcing_full, physical_step, DT_SECONDS, lat, lon, args.device,
            )
            if active and scalar_beta is None:
                sw = runtime.swdown_W_m2[1].clamp(min=0.0)
                weight = sw / (sw + sw_ref_t)
                inv_tau = weight / tau_day_t + (1.0 - weight) / tau_night_t
                beta_t = torch.exp(-DT_SECONDS * inv_tau)
            emission = model._surface_emissivity(
                current_state, static, fraction, torch.float64, args.device,
            )
            step_static = replace(static, emis=emission)
            forcing = make_forcing(runtime, 0, current_state, step_static, forcing_options, tile_frac=fraction)
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
                if active:
                    heat = (fraction * model._last_pahfsti).sum(dim=1).detach()
                    previous = cache["dt_air"] if cache["dt_air"] is not None \
                        else torch.zeros_like(heat)
                    beta = scalar_beta if scalar_beta is not None else beta_t
                    cache["dt_air"] = beta * previous + (1.0 - beta) * (-lam_value * heat)
            return new_state

        def reset():
            cache["dt_air"] = None

        return advance, reset, get_anomaly

    def forecast_day(day: int, advance, reset, get_anomaly, observed_day: dict):
        """One anchored day: 1-day build-up, then the sampled forecast day."""
        archive_step = (day - 1) * STEPS_PER_DAY
        matching = np.flatnonzero(archive_steps == archive_step)
        if matching.size != 1:
            raise ValueError(f"state archive has no unique record for step {archive_step}")
        state = restore_state(copy.deepcopy(base_state), static, model, archive, int(matching[0]),
                              slice(None), args.device)
        reset()
        bin_sum = torch.zeros(N_LOCAL_BINS, dtype=torch.float64, device=args.device)
        bin_weight = torch.zeros(N_LOCAL_BINS, dtype=torch.float64, device=args.device)
        dt_air_step = torch.zeros(STEPS_PER_DAY, dtype=torch.float64, device=args.device)
        skin_step = torch.zeros(STEPS_PER_DAY, dtype=torch.float64, device=args.device)
        with torch.no_grad():
            for step in range(STEPS_PER_DAY):
                state = advance(state, archive_step + step)
            predictions = {period: torch.zeros(nvalid, dtype=torch.float64, device=args.device)
                           for period in PERIODS}
            period_steps = {period: target_steps(observed_day[period]["time"][0]) for period in PERIODS}
            for step in range(STEPS_PER_DAY):
                state = advance(state, archive_step + STEPS_PER_DAY + step)
                completed = step + 1
                anomaly = get_anomaly()
                bin_sum.index_add_(0, local_bin[step], anomaly * area_weight)
                bin_weight.index_add_(0, local_bin[step], area_weight)
                dt_air_step[step] = (anomaly * area_weight).sum() / weight_sum
                skin_step[step] = (state.radiative_skin_t * area_weight).sum() / weight_sum
                for period in PERIODS:
                    mask = torch.as_tensor(period_steps[period] == completed, dtype=torch.bool,
                                           device=args.device)
                    predictions[period] = torch.where(mask, state.radiative_skin_t, predictions[period])
        composite = {
            "dt_air_bin_sum": bin_sum.cpu().numpy(),
            "bin_weight_sum": bin_weight.cpu().numpy(),
            "dt_air_step_mean": dt_air_step.cpu().numpy(),
            "skin_step_mean": skin_step.cpu().numpy(),
        }
        return predictions, period_steps, composite

    windows = {
        "calibration": (calibration_days, observed_all["calibration"]),
        "validation": (validation_days, observed_all["validation"]),
    }

    result = {
        "lambda_s": lam_value,
        "sw_ref_W_m2": SW_REF,
        "configurations": {
            key: {"tau_day_h": (None if cfg["tau_day_s"] is None else cfg["tau_day_s"] / 3600.0),
                  "tau_night_h": (None if cfg["tau_night_s"] is None else cfg["tau_night_s"] / 3600.0)}
            for key, cfg in configurations.items()
        },
        "windows": {name: days for name, (days, _) in windows.items()},
    }
    started = time.perf_counter()

    for config_key, cfg in configurations.items():
        advance, reset, get_anomaly = make_advance(cfg["tau_day_s"], cfg["tau_night_s"])
        result[config_key] = {}
        for window_name, (days, observed) in windows.items():
            composite_sum = {key: np.zeros_like(value, dtype=np.float64)
                             for key, value in
                             (("dt_air_bin_sum", np.empty(N_LOCAL_BINS)),
                              ("bin_weight_sum", np.empty(N_LOCAL_BINS)),
                              ("dt_air_step_mean", np.empty(STEPS_PER_DAY)),
                              ("skin_step_mean", np.empty(STEPS_PER_DAY)))}
            aggregate = {
                period: {"weighted_resid": 0.0, "sq": 0.0, "weight_sum": 0.0, "count": 0}
                for period in PERIODS
            }
            per_day = {}
            for group_day, day in enumerate(days):
                observed_day = single_day(observed, group_day)
                predictions, period_steps, composite = forecast_day(
                    day, advance, reset, get_anomaly, observed_day,
                )
                for period in PERIODS:
                    observation = observed_day[period]["lst"][0]
                    valid = observed_day[period]["strict"][0] & np.isfinite(observation) \
                        & (period_steps[period] >= 1)
                    if not valid.any():
                        per_day.setdefault(str(day), {})[period] = {
                            "n": 0, "bias_K": None, "rmse_K": None}
                        continue
                    valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=args.device)
                    weights = np.cos(np.deg2rad(lat))[valid]
                    residual = predictions[period][valid_tensor].cpu().numpy() - observation[valid]
                    aggregate[period]["weighted_resid"] += float((residual * weights).sum())
                    aggregate[period]["sq"] += float((residual ** 2 * weights).sum())
                    aggregate[period]["weight_sum"] += float(weights.sum())
                    aggregate[period]["count"] += int(residual.size)
                    per_day.setdefault(str(day), {})[period] = {
                        "n": int(valid.sum()),
                        "bias_K": float((residual * weights).sum() / weights.sum()),
                        "rmse_K": float(math.sqrt((residual ** 2 * weights).sum() / weights.sum())),
                    }
                for key in composite_sum:
                    composite_sum[key] += composite[key]
                print(json.dumps({"config": config_key, "window": window_name, "day": day,
                                  "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
            bins = composite_sum["bin_weight_sum"]
            with np.errstate(invalid="ignore", divide="ignore"):
                dt_air_local = np.where(bins > 0, composite_sum["dt_air_bin_sum"] / bins, np.nan)
            result[config_key][window_name] = {
                **{
                    period: {
                        "n": entry["count"],
                        "bias_K": entry["weighted_resid"] / entry["weight_sum"] if entry["count"] else None,
                        "rmse_K": math.sqrt(entry["sq"] / entry["weight_sum"]) if entry["count"] else None,
                    }
                    for period, entry in aggregate.items()
                },
                "per_day": per_day,
                "composite": {
                    "dt_air_local_bin_K": dt_air_local.tolist(),
                    "dt_air_utc_step_K": (composite_sum["dt_air_step_mean"] / len(days)).tolist(),
                    "skin_utc_step_K": (composite_sum["skin_step_mean"] / len(days)).tolist(),
                },
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"done": str(args.output),
                      "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
