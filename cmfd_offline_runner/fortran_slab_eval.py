#!/usr/bin/env python3
"""Evaluate the two Fortran 2018 offline runs (prior and slab) against MODIS
under the same protocol as the torch evaluation: AvgSurfT operator (= torch
radiative_skin_t, tile-fraction-weighted mean skin temperature), view-time
nearest 3-hourly record, strict MODIS QC, area (cos lat) weighting.

o_gg.nc convention (verified by compare_avgskin_inst.py): AvgSurfT[j, 0, :]
is the instantaneous grid at global step 6j (j = 1..2920), so the record at
3(r+1) h into day-of-year D is j = (D-1)*8 + r + 1.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

PERIODS = ("day", "night")


def load_observations(path: Path, days: list[int]) -> dict:
    with Dataset(path) as observations:
        data = {}
        for period in PERIODS:
            data[period] = {
                "lst": np.asarray(observations[f"lst_{period}"][days, :], dtype=np.float64),
                "time": np.asarray(observations[f"lst_{period}_view_time_utc"][days, :], dtype=np.float64),
                "strict": np.asarray(observations[f"lst_{period}_strict"][days, :]) == 1,
            }
        data["lat"] = np.asarray(observations["lat"][:], dtype=np.float64)
    return data


def nearest_3h_record(view_hours: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(view_hours / 3.0).astype(int) - 1, 0, 7)


def evaluate_run(fortran_path: Path, windows: dict, lat: np.ndarray) -> dict:
    with Dataset(fortran_path) as run:
        avg = np.ma.filled(run["AvgSurfT"][:, 0, :], np.nan).astype(np.float64)
    nrecords, ncolumns = avg.shape
    if nrecords != 2921:
        raise ValueError(f"unexpected record count {nrecords} in {fortran_path}")
    area_weight = np.cos(np.deg2rad(lat))
    result = {}
    for window_name, (days, observed) in windows.items():
        aggregate = {period: {"weighted_resid": 0.0, "sq": 0.0, "weight_sum": 0.0, "count": 0}
                     for period in PERIODS}
        per_day = {}
        for group_day, day in enumerate(days):
            day_entry = {}
            for period in PERIODS:
                observation = observed[period]["lst"][group_day]
                valid = (observed[period]["strict"][group_day] & np.isfinite(observation)
                         & np.isfinite(observed[period]["time"][group_day]))
                if not valid.any():
                    day_entry[period] = {"n": 0, "bias_K": None, "rmse_K": None}
                    continue
                # `day` is the 0-based observation time index (doy = day+1);
                # the sampled model day is doy day+1, whose 3-hourly records
                # are j = day*8 + (r+1), r = 0..7 (record j is global step 6j).
                record_index = day * 8 + nearest_3h_record(observed[period]["time"][group_day]) + 1
                rows = record_index[valid]
                prediction = avg[rows, np.flatnonzero(valid)]
                residual = prediction - observation[valid]
                weights = area_weight[valid]
                aggregate[period]["weighted_resid"] += float((residual * weights).sum())
                aggregate[period]["sq"] += float((residual ** 2 * weights).sum())
                aggregate[period]["weight_sum"] += float(weights.sum())
                aggregate[period]["count"] += int(residual.size)
                day_entry[period] = {
                    "n": int(valid.sum()),
                    "bias_K": float((residual * weights).sum() / weights.sum()),
                    "rmse_K": float(math.sqrt((residual ** 2 * weights).sum() / weights.sum())),
                }
            per_day[str(day)] = day_entry
        result[window_name] = {
            **{f"{period}_3h": {
                "n": entry["count"],
                "bias_K": entry["weighted_resid"] / entry["weight_sum"] if entry["count"] else None,
                "rmse_K": math.sqrt(entry["sq"] / entry["weight_sum"]) if entry["count"] else None,
            } for period, entry in aggregate.items()},
            "per_day": per_day,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--slab-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--calibration-days", default="271,281,291,301,311,321,331,341,351,361")
    parser.add_argument("--validation-days", default="91,101,111,121,131,141,151,161,171,181")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration_days = [int(v) for v in args.calibration_days.split(",")]
    validation_days = [int(v) for v in args.validation_days.split(",")]
    observed_calibration = load_observations(args.observations, calibration_days)
    observed_validation = load_observations(args.observations, validation_days)
    windows = {"calibration": (calibration_days, observed_calibration),
               "validation": (validation_days, observed_validation)}

    result = {
        "fortran_prior": evaluate_run(args.prior_dir / "o_gg.nc", windows, observed_calibration["lat"]),
        "fortran_slab": evaluate_run(args.slab_dir / "o_gg.nc", windows, observed_calibration["lat"]),
        "calibration_days": calibration_days,
        "validation_days": validation_days,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    summary = {name: {window: {key: value for key, value in payload[window].items()
                               if key != "per_day"}
                      for window in payload}
               for name, payload in (("prior", result["fortran_prior"]), ("slab", result["fortran_slab"]))}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
