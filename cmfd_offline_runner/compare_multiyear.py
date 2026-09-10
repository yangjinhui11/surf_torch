#!/usr/bin/env python3
"""Compare the 10-year autonomous multi-year Torch run with chained Fortran years.

Matches the per-year Fortran ``o_gg.nc`` records (time coordinate, seconds
from each year start) against the Torch diagnostic records, whose ``step``
values are cumulative across years.  Reuses the field mapping, masks, and
metrics of ``compare_autonomous_annual``.

Environment:
  SURF_TORCH_MULTIYEAR   torch_offline_daily.npz of the multiyear run
  SURF_FORTRAN_YEARS     directory holding years/YYYY/o_gg.nc
  SURF_MULTIYEAR_DATA    per-year dataset root (lat/lon + ntime metadata)
  SURF_COMPARE_OUTPUT    output directory
  SURF_YEARS             START-END (default 2009:2018)
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from compare_autonomous_annual import (
    FIELDS,
    SOIL_T_FIELDS,
    SOIL_W_FIELDS,
    SNOW_FIELDS,
    field_rows,
    fortran_at,
    plot_maps,
    torch_at,
)


def main() -> None:
    torch_path = Path(os.environ["SURF_TORCH_MULTIYEAR"])
    years_root = Path(os.environ["SURF_FORTRAN_YEARS"])
    data_root = Path(os.environ["SURF_MULTIYEAR_DATA"])
    output_dir = Path(os.environ["SURF_COMPARE_OUTPUT"])
    years = [int(v) for v in os.environ.get("SURF_YEARS", "2009:2018").replace(":", "-").split("-")]
    years = list(range(years[0], years[1] + 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(np.load(data_root / f"cmfd_valid_columns_{years[0]}_dataset/lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(data_root / f"cmfd_valid_columns_{years[0]}_dataset/lon.npy", mmap_mode="r"), dtype=np.float64)

    rows: list[dict] = []
    swe_terminal: list[dict] = []
    offsets: dict[int, int] = {}
    offset = 0
    for year in years:
        metadata = json.loads((data_root / f"cmfd_valid_columns_{year}_dataset/metadata.json").read_text())
        nsteps = int(metadata["ntime"]) * 6
        offsets[year] = offset
        offset += nsteps

    with np.load(torch_path) as torch:
        saved = {int(step): index for index, step in enumerate(torch["step"])}
        for year in years:
            path = years_root / str(year) / "o_gg.nc"
            if not path.exists():
                print(f"missing {path}, skipped")
                continue
            try:
                dataset = Dataset(path)
            except OSError as error:
                # Year still integrating: o_gg.nc exists but is not readable.
                print(f"unreadable {path} ({error}), skipped")
                continue
            with dataset:
                dataset.set_auto_mask(False)
                times = np.asarray(dataset["time"][:], dtype=np.float64)
                for record, seconds in enumerate(times):
                    if seconds <= 0.0:
                        continue
                    local_step = int(round(seconds / 1800.0))
                    global_step = offsets[year] + local_step
                    torch_record = saved.get(global_step)
                    if torch_record is None:
                        continue
                    day = global_step * 1800.0 / 86_400.0
                    rows.extend(field_rows(torch, dataset, torch_record, record, global_step, day, lat, lon))
                    terminal = record == len(times) - 1
                    if terminal:
                        swe_t = torch_at(torch, SNOW_FIELDS[0], torch_record)
                        swe_f = fortran_at(dataset, SNOW_FIELDS[0], record)
                        swe_terminal.append({
                            "year": year, "day": day,
                            "fortran_mean_swe": float(np.mean(swe_f)), "torch_mean_swe": float(np.mean(swe_t)),
                            "fortran_max_swe": float(np.max(swe_f)), "torch_max_swe": float(np.max(swe_t)),
                            "rmse": float(np.sqrt(np.mean((swe_t - swe_f) ** 2))),
                            "max_abs_diff": float(np.max(np.abs(swe_t - swe_f))),
                        })
        last_year = years[-1]
        last_path = years_root / str(last_year) / "o_gg.nc"
        try:
            last_dataset = Dataset(last_path) if last_path.exists() else None
        except OSError:
            last_dataset = None
        if last_dataset is not None:
            with last_dataset as dataset:
                dataset.set_auto_mask(False)
                times = np.asarray(dataset["time"][:], dtype=np.float64)
                terminal_record = len(times) - 1
                terminal_step = offsets[last_year] + int(round(times[-1] / 1800.0))
                torch_record = saved.get(terminal_step)
                if torch_record is None:
                    # Per-year o_gg ends at day 360 while the Torch archive keeps
                    # 10-day marks plus the day-365 annual terminal; the exact
                    # terminal comparison lives in compare_multiyear_terminal.py.
                    print(f"no Torch record at step {terminal_step}; terminal maps skipped")
                else:
                    plot_maps(
                        torch, dataset, torch_record, terminal_record,
                        SOIL_T_FIELDS[:2] + SNOW_FIELDS[:2], lat, lon,
                        output_dir / "multiyear_terminal_maps.png",
                        f"Terminal year-{last_year} state after {years[-1] - years[0] + 1} continuous years",
                    )

    json.dump(rows, (output_dir / "multiyear_autonomous_metrics.json").open("w"), indent=1)
    with (output_dir / "multiyear_autonomous_metrics.csv").open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "multiyear_terminal_swe.json").write_text(json.dumps(swe_terminal, indent=2) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, constrained_layout=True)
    total_days = offset * 1800.0 / 86_400.0
    groups = ((SOIL_T_FIELDS + SOIL_W_FIELDS, "Soil states"), (SNOW_FIELDS, "Snow states"))
    colors = plt.get_cmap("tab10").colors
    for row_index, (fields, title) in enumerate(groups):
        for column_index, (metric_name, ylabel) in enumerate((
            ("rmse", "RMSE (native unit)"), ("relative_rmse_percent", "Relative RMSE (%)"),
        )):
            axis = axes[row_index, column_index]
            for color, field in zip(colors, fields):
                domain = "snow_fraction_gt_1e-3" if field.label in {"snow_t", "snow_rho", "snow_alb"} else "all_columns"
                points = [p for p in rows if p["field"] == field.label and p["domain"] == domain]
                axis.semilogy([float(p["day"]) for p in points], [float(p[metric_name]) for p in points],
                              color=color, linewidth=1.0, label=field.title)
            axis.set(title=f"{title}: {ylabel}", ylabel=ylabel, xlim=(0.0, total_days))
            axis.grid(alpha=0.25, which="both")
            axis.legend(fontsize=6.5, ncol=2, frameon=False)
    for axis in axes[1]:
        axis.set_xlabel("Continuous integration day")
        axis.set_xticks([365.25 * k for k in range(len(years) + 1)])
        axis.set_xticklabels([str(years[0] + k) for k in range(len(years) + 1)], fontsize=7)
    figure.suptitle(f"Autonomous offline SURF: Fortran-PyTorch error evolution, {years[0]}-{years[-1]}", fontsize=13)
    figure.savefig(output_dir / "multiyear_autonomous_error_growth.png", dpi=260)
    plt.close(figure)
    print(json.dumps({"rows": len(rows), "terminal_swe_records": len(swe_terminal)}))


if __name__ == "__main__":
    main()
