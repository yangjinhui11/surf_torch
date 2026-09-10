#!/usr/bin/env python3
"""Prepare daily MODIS CMG observations on the CMFD valid-column grid."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC


CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
PRODUCTS = {
    "lst": {"short_name": "MOD11C1", "version": "061", "doi": "10.5067/MODIS/MOD11C1.061"},
    "snow": {"short_name": "MOD10C1", "version": "61", "doi": "10.5067/MODIS/MOD10C1.061"},
}
DATE_PATTERN = re.compile(r"\.A(\d{4})(\d{3})\.")


def cmg_indices(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest containing 0.05-degree CMG cell indices."""
    if np.any((lat < -90.0) | (lat > 90.0)) or np.any((lon < -180.0) | (lon > 180.0)):
        raise ValueError("latitude or longitude lies outside the global CMG grid")
    row = np.floor((90.0 - lat) / 0.05).astype(np.int64)
    col = np.floor((lon + 180.0) / 0.05).astype(np.int64)
    return np.clip(row, 0, 3599), np.clip(col, 0, 7199)


def strict_lst_good(raw_lst: np.ndarray, qc: np.ndarray, view_time: np.ndarray) -> np.ndarray:
    """MOD11 strict-good mask: produced, good data, and <=1 K stated error."""
    mandatory_and_data_quality = qc & np.uint8(0x0F)
    lst_error = (qc >> np.uint8(6)) & np.uint8(0x03)
    return (raw_lst > 0) & (view_time <= 120) & (mandatory_and_data_quality == 0) & (lst_error == 0)


def usable_lst(raw_lst: np.ndarray, qc: np.ndarray, view_time: np.ndarray) -> np.ndarray:
    """Broader MOD11 mask: LST produced and stated error no larger than 2 K."""
    mandatory = qc & np.uint8(0x03)
    lst_error = (qc >> np.uint8(6)) & np.uint8(0x03)
    return (raw_lst > 0) & (view_time <= 120) & (mandatory <= 1) & (lst_error <= 1)


def _date_from_title(title: str) -> date:
    match = DATE_PATTERN.search(title)
    if match is None:
        raise ValueError(f"cannot extract MODIS acquisition date from {title!r}")
    return datetime.strptime("".join(match.groups()), "%Y%j").date()


def query_granules(session: requests.Session, product: dict[str, str], year: int) -> dict[date, str]:
    response = session.get(
        CMR_GRANULES,
        params={
            "short_name": product["short_name"],
            "version": product["version"],
            "temporal": f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z",
            "page_size": 2000,
        },
        timeout=120,
    )
    response.raise_for_status()
    result: dict[date, str] = {}
    for entry in response.json()["feed"]["entry"]:
        links = [
            item["href"] for item in entry.get("links", [])
            if item.get("href", "").startswith("https://")
            and item.get("href", "").endswith(".hdf")
            and not item.get("inherited", False)
        ]
        if not links:
            continue
        result[_date_from_title(entry["title"])] = links[0]
    return result


def download(session: requests.Session, url: str, destination: Path) -> None:
    """Download atomically; requests obtains Earthdata credentials from .netrc."""
    partial = destination.with_suffix(destination.suffix + ".part")
    with session.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                if block:
                    stream.write(block)
    partial.replace(destination)


def _take(sds, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Read only the rectangular source window enclosing all target columns."""
    row0, row1 = int(row.min()), int(row.max()) + 1
    col0, col1 = int(col.min()), int(col.max()) + 1
    window = np.asarray(sds[row0:row1, col0:col1])
    return window[row - row0, col - col0]


def read_lst(path: Path, row: np.ndarray, col: np.ndarray) -> dict[str, np.ndarray]:
    hdf = SD(str(path), SDC.READ)
    try:
        result: dict[str, np.ndarray] = {}
        for period in ("Day", "Night"):
            raw = _take(hdf.select(f"LST_{period}_CMG"), row, col).astype(np.uint16)
            qc = _take(hdf.select(f"QC_{period}"), row, col).astype(np.uint8)
            raw_time = _take(hdf.select(f"{period}_view_time"), row, col).astype(np.uint8)
            valid = (raw > 0) & (raw_time <= 120)
            strict = strict_lst_good(raw, qc, raw_time)
            good = usable_lst(raw, qc, raw_time)
            values = np.full(raw.shape, np.nan, dtype=np.float32)
            times = np.full(raw.shape, np.nan, dtype=np.float32)
            values[valid] = raw[valid].astype(np.float32) * np.float32(0.02)
            times[valid] = raw_time[valid].astype(np.float32) * np.float32(0.2)
            key = period.lower()
            result[f"lst_{key}"] = values
            result[f"lst_{key}_view_time_utc"] = times
            result[f"lst_{key}_qc"] = qc
            result[f"lst_{key}_good"] = good.astype(np.uint8)
            result[f"lst_{key}_strict"] = strict.astype(np.uint8)
        result["percent_land"] = _take(hdf.select("Percent_land_in_grid"), row, col).astype(np.uint8)
        return result
    finally:
        hdf.end()


def read_snow(path: Path, row: np.ndarray, col: np.ndarray) -> dict[str, np.ndarray]:
    hdf = SD(str(path), SDC.READ)
    try:
        raw = _take(hdf.select("Day_CMG_Snow_Cover"), row, col).astype(np.uint8)
        clear = _take(hdf.select("Day_CMG_Clear_Index"), row, col).astype(np.uint8)
        qa = _take(hdf.select("Snow_Spatial_QA"), row, col).astype(np.uint8)
        valid = (raw <= 100) & (clear <= 100) & (qa <= 1)
        snow = np.full(raw.shape, np.nan, dtype=np.float32)
        clear_fraction = np.full(raw.shape, np.nan, dtype=np.float32)
        snow[valid] = raw[valid].astype(np.float32) * np.float32(0.01)
        clear_fraction[valid] = clear[valid].astype(np.float32) * np.float32(0.01)
        return {
            "snow_fraction": snow,
            "snow_clear_fraction": clear_fraction,
            "snow_qa": qa,
            "snow_good": (valid & (clear >= 50)).astype(np.uint8),
        }
    finally:
        hdf.end()


def create_output(path: Path, year: int, days: list[date], lat: np.ndarray, lon: np.ndarray) -> Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = Dataset(path, "w", format="NETCDF4")
    output.createDimension("time", len(days))
    output.createDimension("column", lat.size)
    output.title = "MODIS Terra CMG observations sampled at CMFD valid land columns"
    output.history = f"created {datetime.now(timezone.utc).isoformat()} by prepare_modis_cmg.py"
    output.source = "MOD11C1.061 and MOD10C1.061"
    output.spatial_mapping = "containing 0.05-degree CMG cell for each CMFD column centre"
    output.lst_quality_filter = "strict: QC bits 0-3 and 6-7 equal 0; good: LST produced and QC bits 6-7 <=1 (stated error <=2 K)"
    output.snow_quality_filter = "Snow Spatial QA <=1; snow_good additionally requires clear index >=50%"
    epoch = date(year, 1, 1)
    time_var = output.createVariable("time", "i2", ("time",))
    time_var.units = f"days since {year}-01-01 00:00:00"
    time_var.calendar = "standard"
    time_var[:] = np.array([(day - epoch).days for day in days], dtype=np.int16)
    output.createVariable("lat", "f4", ("column",))[:] = lat.astype(np.float32)
    output.createVariable("lon", "f4", ("column",))[:] = lon.astype(np.float32)
    float_vars = (
        "lst_day", "lst_day_view_time_utc", "lst_night", "lst_night_view_time_utc",
        "snow_fraction", "snow_clear_fraction",
    )
    byte_vars = (
        "lst_day_qc", "lst_day_good", "lst_day_strict", "lst_night_qc", "lst_night_good", "lst_night_strict",
        "percent_land", "snow_qa", "snow_good",
    )
    chunks = (1, min(16384, lat.size))
    for name in float_vars:
        var = output.createVariable(name, "f4", ("time", "column"), zlib=True, complevel=4, chunksizes=chunks, fill_value=np.nan)
        if name.startswith("lst_") and "view_time" not in name:
            var.units = "K"
        elif "view_time" in name:
            var.units = "hours UTC"
        else:
            var.units = "1"
    for name in byte_vars:
        output.createVariable(name, "u1", ("time", "column"), zlib=True, complevel=4, chunksizes=chunks, fill_value=np.uint8(255))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--cmfd-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--max-days", type=int)
    args = parser.parse_args()

    lat = np.asarray(np.load(args.cmfd_data / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(args.cmfd_data / "lon.npy", mmap_mode="r"), dtype=np.float64)
    row, col = cmg_indices(lat, lon)
    all_days = [date(args.year, 1, 1) + timedelta(days=i) for i in range((date(args.year + 1, 1, 1) - date(args.year, 1, 1)).days)]
    days = all_days[: args.max_days] if args.max_days else all_days
    session = requests.Session()
    granules = {name: query_granules(session, product, args.year) for name, product in PRODUCTS.items()}
    missing = {name: [day.isoformat() for day in days if day not in records] for name, records in granules.items()}
    if any(missing.values()):
        raise RuntimeError(f"missing CMR granules: {missing}")

    cache_owner = None
    if args.cache is None:
        cache_owner = tempfile.TemporaryDirectory(prefix="surf_modis_")
        cache = Path(cache_owner.name)
    else:
        cache = args.cache
        cache.mkdir(parents=True, exist_ok=True)

    output = create_output(args.output, args.year, days, lat, lon)
    try:
        for index, day in enumerate(days):
            for product_name, reader in (("lst", read_lst), ("snow", read_snow)):
                url = granules[product_name][day]
                raw_path = cache / Path(urlparse(url).path).name
                if not raw_path.exists():
                    download(session, url, raw_path)
                values = reader(raw_path, row, col)
                for name, value in values.items():
                    output[name][index, :] = value
                if not args.keep_raw:
                    raw_path.unlink(missing_ok=True)
            output.sync()
            print(f"processed {day.isoformat()} ({index + 1}/{len(days)})", flush=True)
    finally:
        output.close()
        if cache_owner is not None:
            cache_owner.cleanup()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "year": args.year,
        "days": len(days),
        "columns": int(lat.size),
        "output": str(args.output),
        "products": PRODUCTS,
        "raw_files_retained": bool(args.keep_raw),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
