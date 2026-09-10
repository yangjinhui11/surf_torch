#!/usr/bin/env python3
"""Emit an OpenIFS offline-driver forcing NetCDF from the CMFD valid-column datasets.

Schema replicated from the validated template
``full_domain_30d_controlled_97709_unpadded/forcing``:

* dims: ``lon``(97709), ``lat``(1), ``time``(N+2)
* float32 data vars (time, lat, lon): Tair[K], Qair[kg/kg], PSurf[Pa],
  Wind[m/s], SWdown[W/m2], LWdown[W/m2], Rainf/Snowf[kg/m2s]
* ``_FillValue`` = 1e20 (f4), float64 ``lat``/``lon`` coordinate vars,
  ``time`` in "seconds since YYYY 01 01 00 00 00" starting at 0, step 10800.
* precipitation split: record air temperature (float32) <= 273.15 K.

The two terminal records extend the requested window.  Three modes:

* ``same``      : next two records of the same dataset (real continuation;
                  replicates the 30-day template when --ntime 240).
* ``wrap``      : first two records of the same dataset (cyclic year; the
                  standalone 1-year convention).
* ``next:YYYY`` : first two records of the following year's dataset (the
                  continuous multi-year chaining convention).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

FREEZE_K = np.float32(273.15)
VARIABLES = (
    ("Tair", "tair_K", "K"),
    ("Qair", "qair_kg_kg", "kg/kg"),
    ("PSurf", "psurf_Pa", "Pa"),
    ("Wind", "wind_m_s", "m/s"),
    ("SWdown", "swdown_W_m2", "W/m2"),
    ("LWdown", "lwdown_W_m2", "W/m2"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--ntime", type=int, default=None,
                        help="records to take from the year dataset (default: all)")
    parser.add_argument("--next-records", default="wrap",
                        help="'same', 'wrap', or 'next:YYYY' (validated in main)")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def dataset_dir(root: Path, year: int) -> Path:
    return root / f"cmfd_valid_columns_{year}_dataset"


def load_year(root: Path, year: int) -> dict[str, np.ndarray]:
    data = dataset_dir(root, year)
    arrays = {src: np.load(data / f"{src}.npy", mmap_mode="r") for _, src, _ in VARIABLES}
    arrays["precip_kg_m2_s"] = np.load(data / "precip_kg_m2_s.npy", mmap_mode="r")
    return arrays


def record(arrays: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return {name: np.asarray(values[index], dtype=np.float32) for name, values in arrays.items()}


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    arrays = load_year(root, args.year)
    ntime = int(args.ntime) if args.ntime is not None else int(next(iter(arrays.values())).shape[0])
    if not 0 < ntime <= next(iter(arrays.values())).shape[0]:
        raise ValueError(f"ntime {ntime} outside dataset")

    mode = args.next_records
    if mode not in ("same", "wrap") and not mode.startswith("next:"):
        raise SystemExit(
            f"--next-records: invalid choice: {mode!r} (choose from 'same', 'wrap', 'next:YYYY')")

    if mode == "same":
        tail = [record(arrays, ntime), record(arrays, ntime + 1)]
    elif mode == "wrap":
        tail = [record(arrays, 0), record(arrays, 1)]
    else:
        nxt = load_year(root, int(mode.split(":", 1)[1]))
        tail = [record(nxt, 0), record(nxt, 1)]

    lat = np.asarray(np.load(dataset_dir(root, args.year) / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(dataset_dir(root, args.year) / "lon.npy", mmap_mode="r"), dtype=np.float64)
    ncol = int(next(iter(arrays.values())).shape[1])
    if lat.size != ncol or lon.size != ncol:
        raise ValueError("coordinate/forcing column mismatch")

    times = np.arange(ntime + 2, dtype=np.float64) * 10_800.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(output, "w", format="NETCDF4") as ds:
        ds.createDimension("lon", ncol)
        ds.createDimension("lat", 1)
        ds.createDimension("time", None)
        v_lon = ds.createVariable("lon", "f8", ("lon",))
        v_lon.units = "degrees_east"
        v_lon[:] = lon
        v_lat = ds.createVariable("lat", "f8", ("lat",))
        v_lat.units = "degrees_north"
        v_lat[:] = np.asarray(lat, dtype=np.float32).mean(dtype=np.float32).reshape(1).astype(np.float64)
        v_time = ds.createVariable("time", "f8", ("time",))
        v_time.units = f"seconds since {args.year} 01 01 00 00 00"
        v_time[:] = times
        writers = {}
        for name, _, unit in VARIABLES:
            var = ds.createVariable(name, "f4", ("time", "lat", "lon"), fill_value=np.float32(1e20))
            var.units = unit
            writers[name] = var
        rain = ds.createVariable("Rainf", "f4", ("time", "lat", "lon"), fill_value=np.float32(1e20))
        rain.units = "kg/m2s"
        snow = ds.createVariable("Snowf", "f4", ("time", "lat", "lon"), fill_value=np.float32(1e20))
        snow.units = "kg/m2s"

        for index in range(ntime + 2):
            source = record(arrays, index) if index < ntime else tail[index - ntime]
            frozen = source["tair_K"] <= FREEZE_K
            column = slice(None)
            for name, src_name, _ in VARIABLES:
                writers[name][index, 0, column] = source[src_name]
            rain[index, 0, column] = source["precip_kg_m2_s"] * (~frozen)
            snow[index, 0, column] = source["precip_kg_m2_s"] * frozen
            if (index + 1) % 240 == 0:
                print(f"record {index + 1}/{ntime + 2}", flush=True)

    meta = {
        "year": args.year, "ntime": ntime, "next_records": args.next_records,
        "ncol": ncol, "output": str(output),
    }
    Path(str(output) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
