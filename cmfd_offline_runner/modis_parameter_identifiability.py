#!/usr/bin/env python3
"""Diagnose vegetation-parameter identifiability from MODIS LST.

The experiment evaluates four seasonally matched one-day trajectories.  For
each day, SURF LST is sampled at the cell-specific Terra view time.  Reverse-
mode Rademacher projections estimate the Gauss--Newton information matrix of
the complete strict-QC observation vector without materializing its Jacobian.

The three controls are uniform logarithmic scales of the RVLAI, RVRSMIN, and
RVZ0M vegetation tables.  Gradients with respect to those scales are obtained
by projecting table gradients onto ``d p / d log(scale) = p`` at the reference
state.  No parameter update is performed by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from netCDF4 import Dataset


FORCING_NAMES = (
    "tair_K", "qair_kg_kg", "psurf_Pa", "swdown_W_m2",
    "lwdown_W_m2", "wind_m_s", "precip_kg_m2_s",
)
PARAMETER_NAMES = ("rvlai", "rvrsmin", "rvz0m")
PERIODS = ("day", "night")
STEPS_PER_DAY = 48
DT_SECONDS = 1800.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--state-archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-days", default="91,181,271,361")
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--lst-error-k", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--column-start", type=int, default=0)
    parser.add_argument("--column-count", type=int, default=None)
    return parser.parse_args()


def subset_dataclass(value: object, selected: slice, captured_nvalid: int) -> object:
    updates = {
        field.name: getattr(value, field.name)[selected]
        for field in fields(value)
        if isinstance(getattr(value, field.name), torch.Tensor)
        and getattr(value, field.name).ndim > 0
        and getattr(value, field.name).shape[0] == captured_nvalid
    }
    return replace(value, **updates)


def projected_scale_gradient(
    scalar: torch.Tensor,
    parameters: list[torch.Tensor],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        scalar, parameters, retain_graph=retain_graph, allow_unused=True,
    )
    values = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            values.append(parameter.new_zeros(()))
        else:
            values.append((gradient * parameter.detach()).sum())
    return torch.stack(values)


def restore_state(
    base_state: object,
    static: object,
    model: object,
    archive: np.lib.npyio.NpzFile,
    archive_index: int,
    selected: slice,
    device: str,
) -> object:
    dtype = torch.float64

    def tensor(name: str) -> torch.Tensor:
        values = np.asarray(archive[name][archive_index, selected], dtype=np.float64)
        return torch.as_tensor(values, dtype=dtype, device=device)

    skin_t = tensor("skin_t")
    soil_t = tensor("soil_t")
    soil_w = tensor("soil_w") / (static.soil_thickness.unsqueeze(0) * 1000.0)
    snow_w = tensor("snow_w")
    snow_rho = tensor("snow_rho")
    snow_t = tensor("snow_t")
    snow_alb = tensor("snow_alb")
    snow_frac = torch.clamp(10.0 * snow_w / torch.clamp(snow_rho, min=1.0e-12), 0.0, 1.0)
    provisional = replace(base_state, snow_w=snow_w, snow_rho=snow_rho, snow_frac=snow_frac)
    tile_fraction = model._tile_frac(static, skin_t.shape[1], state=provisional)
    radiative_skin_t = (skin_t * tile_fraction).sum(dim=1)
    return replace(
        base_state,
        skin_t=skin_t,
        soil_t=soil_t,
        soil_w_liq=soil_w,
        snow_w=snow_w,
        snow_rho=snow_rho,
        snow_t=snow_t,
        snow_alb=snow_alb,
        snow_frac=snow_frac,
        radiative_skin_t=radiative_skin_t,
        vupdz0_step=0,
        vupdz0_pustrti=None,
        vupdz0_pvstrti=None,
        vupdz0_pahfsti=None,
        vupdz0_pevapti=None,
        vupdz0_pzdl=None,
    )


def target_steps(hours_utc: np.ndarray) -> np.ndarray:
    result = np.full(hours_utc.shape, -1, dtype=np.int64)
    finite = np.isfinite(hours_utc)
    result[finite] = np.rint(hours_utc[finite] * 3600.0 / DT_SECONDS).astype(np.int64)
    result[finite] = np.clip(result[finite], 1, STEPS_PER_DAY)
    return result


def main() -> None:
    args = parse_args()
    if args.probes < 1:
        raise ValueError("--probes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    os.environ["SURF_CASE"] = str(args.case)
    os.environ["SURF_CODE"] = str(args.source)
    os.environ["SURF_DATA"] = str(args.data)
    sys.path[:0] = [str(args.source), str(args.runner)]
    from diagnostic_scripts_compare_realcol import make_forcing
    from replay_full_domain_controlled import load_initial
    from run_full_domain_offline import interpolated_runtime
    from surf_pytorch import LandSurface, SurfConfig

    metadata = json.loads((args.case / "metadata.json").read_text())
    captured_nvalid = int(metadata["nvalid"])
    column_count = captured_nvalid - args.column_start if args.column_count is None else args.column_count
    column_stop = args.column_start + column_count
    if args.column_start < 0 or column_count < 1 or column_stop > captured_nvalid:
        raise ValueError(
            f"column interval must lie in 0..{captured_nvalid}; got "
            f"[{args.column_start}, {column_stop})"
        )
    selected = slice(args.column_start, column_stop)
    full_lat = np.asarray(np.load(args.data / "lat.npy", mmap_mode="r"), dtype=np.float64)
    full_lon = np.asarray(np.load(args.data / "lon.npy", mmap_mode="r"), dtype=np.float64)
    lat = full_lat[selected]
    lon = full_lon[selected]
    forcing_arrays = {
        name: np.load(args.data / f"{name}.npy", mmap_mode="r")[:, selected]
        for name in FORCING_NAMES
    }
    sample_days = tuple(int(value) for value in args.sample_days.split(","))
    if any(day < 1 or day >= 365 for day in sample_days):
        raise ValueError("sample days must allow a preceding one-day adjustment")

    with Dataset(args.observations) as observations:
        obs_lat = np.asarray(observations["lat"][selected], dtype=np.float64)
        obs_lon = np.asarray(observations["lon"][selected], dtype=np.float64)
        if not np.allclose(obs_lat, lat, atol=1.0e-5) or not np.allclose(obs_lon, lon, atol=1.0e-5):
            raise ValueError("MODIS and CMFD columns are not aligned")
        observed = {
            period: {
                "lst": np.asarray(observations[f"lst_{period}"][sample_days, selected], dtype=np.float64),
                "time": np.asarray(observations[f"lst_{period}_view_time_utc"][sample_days, selected], dtype=np.float64),
                "strict": np.asarray(observations[f"lst_{period}_strict"][sample_days, selected]) == 1,
            }
            for period in PERIODS
        }

    archive = np.load(args.state_archive, mmap_mode="r")
    archive_steps = np.asarray(archive["step"], dtype=np.int64)
    area_weight = np.cos(np.deg2rad(lat))
    rng = np.random.default_rng(args.seed)
    results = []
    information_total = np.zeros((len(PARAMETER_NAMES), len(PARAMETER_NAMES)), dtype=np.float64)
    information_calibration = np.zeros_like(information_total)
    information_validation = np.zeros_like(information_total)

    forcing_options = SimpleNamespace(
        shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
        shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
        stress_u=None, stress_v=None, vup=None,
    )

    for sample_number, day in enumerate(sample_days):
        archive_step = (day - 1) * STEPS_PER_DAY
        matching = np.flatnonzero(archive_steps == archive_step)
        if matching.size != 1:
            raise ValueError(f"state archive has no unique record for step {archive_step}")

        base_state, static = load_initial(
            args.case / "restartin.nc", captured_nvalid, args.device,
            full_lat, full_lon,
        )
        base_state = subset_dataclass(base_state, selected, captured_nvalid)
        static = subset_dataclass(static, selected, captured_nvalid)
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
        )
        model = LandSurface(config).to(args.device)
        parameters = [getattr(model.veg_table, name) for name in PARAMETER_NAMES]
        state = restore_state(
            base_state, static, model, archive, int(matching[0]), selected,
            args.device,
        )

        def advance(current_state: object, physical_step: int) -> object:
            fraction = model._tile_frac(static, config.n_tile, state=current_state)
            runtime = interpolated_runtime(
                forcing_arrays, physical_step, DT_SECONDS, lat, lon, args.device,
            )
            emission = model._surface_emissivity(
                current_state, static, fraction, torch.float64, args.device,
            )
            step_static = replace(static, emis=emission)
            forcing = make_forcing(
                runtime, 0, current_state, step_static, forcing_options,
                tile_frac=fraction,
            )
            precipitation = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
            forcing = replace(
                forcing,
                rain=precipitation * (1.0 - runtime.precip_snow_phase),
                snow=precipitation * runtime.precip_snow_phase,
            )
            return model(current_state, forcing, step_static)[0]

        with torch.no_grad():
            for step in range(STEPS_PER_DAY):
                state = advance(state, archive_step + step)

        period_predictions = {
            period: torch.zeros(column_count, dtype=torch.float64, device=args.device)
            for period in PERIODS
        }
        period_steps = {
            period: target_steps(observed[period]["time"][sample_number])
            for period in PERIODS
        }
        for step in range(STEPS_PER_DAY):
            state = advance(state, archive_step + STEPS_PER_DAY + step)
            completed_step = step + 1
            for period in PERIODS:
                mask = torch.as_tensor(
                    period_steps[period] == completed_step,
                    dtype=torch.bool,
                    device=args.device,
                )
                period_predictions[period] = torch.where(
                    mask, state.radiative_skin_t, period_predictions[period],
                )

        day_information = np.zeros_like(information_total)
        period_results = {}
        for period_index, period in enumerate(PERIODS):
            observation = observed[period]["lst"][sample_number]
            valid = (
                observed[period]["strict"][sample_number]
                & np.isfinite(observation)
                & (period_steps[period] >= 1)
            )
            count = int(valid.sum())
            if count == 0:
                period_results[period] = {
                    "samples": 0,
                    "area_weight_sum": 0.0,
                    "bias_K": 0.0,
                    "rmse_K": 0.0,
                    "loss_gradient_log_scale": [0.0] * len(PARAMETER_NAMES),
                    "information": np.zeros_like(information_total).tolist(),
                    "information_raw": np.zeros_like(information_total).tolist(),
                }
                continue
            valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=args.device)
            weights = area_weight[valid]
            weight_sum = float(weights.sum())
            sqrt_normalized_weight = torch.as_tensor(
                np.sqrt(weights / weight_sum) / args.lst_error_k,
                dtype=torch.float64,
                device=args.device,
            )
            sqrt_raw_weight = torch.as_tensor(
                np.sqrt(weights) / args.lst_error_k,
                dtype=torch.float64,
                device=args.device,
            )
            prediction = period_predictions[period][valid_tensor]
            target = torch.as_tensor(
                observation[valid], dtype=torch.float64, device=args.device,
            )
            residual = (prediction - target) * sqrt_normalized_weight
            loss = 0.5 * (residual * residual).sum()
            loss_gradient = projected_scale_gradient(
                loss, parameters, retain_graph=True,
            ).detach().cpu().numpy()

            standardized_prediction = prediction * sqrt_raw_weight
            period_information_raw = np.zeros_like(information_total)
            for probe in range(args.probes):
                signs = rng.integers(0, 2, size=count, dtype=np.int8) * 2 - 1
                projection = (
                    standardized_prediction
                    * torch.as_tensor(signs, dtype=torch.float64, device=args.device)
                ).sum()
                keep = not (
                    sample_number == len(sample_days) - 1
                    and period_index == len(PERIODS) - 1
                    and probe == args.probes - 1
                )
                projected = projected_scale_gradient(
                    projection, parameters, retain_graph=keep,
                ).detach().cpu().numpy()
                period_information_raw += np.outer(projected, projected) / args.probes

            period_information = period_information_raw / weight_sum

            prediction_np = prediction.detach().cpu().numpy()
            difference = prediction_np - observation[valid]
            normalized_weights = weights / weight_sum
            period_results[period] = {
                "samples": count,
                "area_weight_sum": weight_sum,
                "bias_K": float(np.sum(normalized_weights * difference)),
                "rmse_K": float(np.sqrt(np.sum(normalized_weights * difference ** 2))),
                "loss_gradient_log_scale": loss_gradient.tolist(),
                "information": period_information.tolist(),
                "information_raw": period_information_raw.tolist(),
            }
            day_information += period_information

        results.append({
            "observation_day_index": day,
            "archive_step": archive_step,
            "periods": period_results,
            "information": day_information.tolist(),
        })
        information_total += day_information
        if sample_number < len(sample_days) // 2:
            information_calibration += day_information
        else:
            information_validation += day_information
        del state, model, parameters, period_predictions
        if args.device == "cuda":
            torch.cuda.empty_cache()

    def spectrum(matrix: np.ndarray) -> dict[str, object]:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        positive = eigenvalues[eigenvalues > eigenvalues[0] * 1.0e-12] if eigenvalues[0] > 0 else np.asarray([])
        condition = None if positive.size < len(PARAMETER_NAMES) else float(positive[0] / positive[-1])
        return {
            "eigenvalues": eigenvalues.tolist(),
            "eigenvectors_columns": eigenvectors.tolist(),
            "numerical_rank": int(positive.size),
            "condition_number": condition,
        }

    summary = {
        "parameter_controls": [f"log_scale_{name}" for name in PARAMETER_NAMES],
        "captured_ncolumns": captured_nvalid,
        "column_start": args.column_start,
        "column_stop": column_stop,
        "ncolumns": column_count,
        "sample_days_zero_based": list(sample_days),
        "calibration_days": list(sample_days[:len(sample_days) // 2]),
        "validation_days": list(sample_days[len(sample_days) // 2:]),
        "strict_lst_error_scale_K": args.lst_error_k,
        "rademacher_probes_per_period": args.probes,
        "results": results,
        "information_total": information_total.tolist(),
        "information_calibration": information_calibration.tolist(),
        "information_validation": information_validation.tolist(),
        "spectrum_total": spectrum(information_total),
        "spectrum_calibration": spectrum(information_calibration),
        "spectrum_validation": spectrum(information_validation),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "modis_parameter_identifiability.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        args.output / "modis_parameter_identifiability.npz",
        parameter_names=np.asarray(PARAMETER_NAMES),
        sample_days=np.asarray(sample_days),
        information_total=information_total,
        information_calibration=information_calibration,
        information_validation=information_validation,
    )
    print(json.dumps({
        "output": str(args.output),
        "spectrum_total": summary["spectrum_total"],
        "spectrum_calibration": summary["spectrum_calibration"],
        "spectrum_validation": summary["spectrum_validation"],
    }, indent=2))


if __name__ == "__main__":
    main()
