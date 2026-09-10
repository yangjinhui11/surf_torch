#!/usr/bin/env python3
"""Reproducibility manifest for the multi-year autonomous comparison experiment.

Records configuration, dataset metadata hashes, per-year namelists, runner
script hashes, and output summaries.  Large forcing/netCDF outputs are
recorded by size and (optionally) full sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path, quick: bool = False) -> str:
    digest = hashlib.sha256()
    if quick:
        with path.open("rb") as handle:
            digest.update(handle.read(4 * 1024 * 1024))
            handle.seek(max(0, path.stat().st_size - 4 * 1024 * 1024))
            digest.update(handle.read(4 * 1024 * 1024))
        return "quick:" + digest.hexdigest()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, full_hash: bool) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256(path, quick=not full_hash)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True,
                        help="torch multiyear experiment dir")
    parser.add_argument("--fortran-root", type=Path, required=True,
                        help="chained Fortran root (years/YYYY + forcing_cache)")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--years", default="2009:2018")
    parser.add_argument("--full-hashes", action="store_true",
                        help="hash complete large files (slow) instead of head+tail")
    args = parser.parse_args()

    years = [int(v) for v in args.years.replace(":", "-").split("-")]
    years = list(range(years[0], years[1] + 1))
    runner = Path(__file__).resolve().parent
    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": str(args.experiment.resolve()),
        "years": years,
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.source, capture_output=True, text=True).stdout.strip(),
        "source_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=args.source, capture_output=True, text=True).stdout.strip()),
        "runner_scripts": {p.name: record(p, True) for p in sorted(runner.glob("*.py")) + sorted(runner.glob("*.sh"))
                           if p.name in {
                               "make_fortran_forcing.py", "setup_fortran_rundir.sh", "run_fortran_multiyear.sh",
                               "run_torch_multiyear_diag.py", "compare_multiyear.py", "run_multiyear_spinup.py",
                               "run_full_domain_offline.py", "compare_autonomous_annual.py",
                           }},
        "dataset_metadata": {},
        "initial_state": None,
        "fortran_years": {},
        "torch_outputs": {},
        "hardware": {
            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip().splitlines()[:1],
            "cpu": subprocess.run(["lscpu", "-y"], capture_output=True, text=True).stdout.strip().split("\n")[0],
        },
        # Interpreter pinning is not optional: the 2026-08-22 decade run
        # launched under a foreign conda env (torch 2.6.0+cu124) reproduced
        # every earlier validation only to round-off seeds that the snow-soil
        # feedbacks amplified to ~1e-3 rms SWE within one year, so the run of
        # record and the bit-reproducibility probes must name the exact
        # interpreter and torch build.
        "interpreter": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "gpu_uuid": subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().splitlines()[:1],
    }

    for year in years:
        meta = args.data_root / f"cmfd_valid_columns_{year}_dataset/metadata.json"
        manifest["dataset_metadata"][str(year)] = record(meta, True) if meta.exists() else None
        year_dir = args.fortran_root / "years" / str(year)
        entry: dict[str, object] = {}
        for name in ("input", "o_gg.nc", "restartout.nc", "fortran.log"):
            path = year_dir / name
            entry[name] = record(path, True) if name == "input" and path.exists() else (
                record(path, args.full_hashes) if path.exists() else None)
        manifest["fortran_years"][str(year)] = entry

    # Hash the actual working-tree sources the run imports; git revision plus
    # a dirty flag cannot distinguish which uncommitted state executed.
    src_pkg = args.source / "surf_pytorch"
    manifest["source_files"] = {}
    for rel in ("model.py", "constants.py", "state.py", "forcing.py", "utils.py",
                "physics/snow.py", "physics/soil.py", "physics/canopy.py",
                "physics/exchange.py", "physics/snow_density.py", "physics/snow_albedo.py"):
        path = src_pkg / rel
        manifest["source_files"][rel] = record(path, True) if path.exists() else None

    summary = args.experiment / "torch_offline_summary.json"
    annual = args.experiment / "annual_diagnostics.json"
    npz = args.experiment / "torch_offline_daily.npz"
    terminals = args.experiment / "torch_year_terminal_states.npz"
    summary_json = json.loads(summary.read_text()) if summary.exists() else {}
    manifest["initial_state"] = summary_json.get("initial")
    manifest["torch_outputs"] = {
        "summary": json.loads(summary.read_text()) if summary.exists() else None,
        "annual_diagnostics": json.loads(annual.read_text()) if annual.exists() else None,
        "daily_npz": record(npz, args.full_hashes) if npz.exists() else None,
        "terminal_states_npz": record(terminals, args.full_hashes) if terminals.exists() else None,
    }

    out = args.experiment / "repro_manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"manifest written: {out}")


if __name__ == "__main__":
    main()
