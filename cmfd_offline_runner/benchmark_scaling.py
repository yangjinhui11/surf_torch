#!/usr/bin/env python3
"""Run reproducible CPU/GPU scaling benchmarks for the CMFD SURF driver."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_COLUMNS = (1_000, 10_000, 50_000, 97_709)


def parse_columns(value: str) -> tuple[int, ...]:
    columns = tuple(int(item) for item in value.split(",") if item)
    if not columns or any(item < 1 for item in columns):
        raise argparse.ArgumentTypeError("columns must be a non-empty list of positive integers")
    return columns


def parse_args() -> argparse.Namespace:
    runner_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="surf_pytorch source directory")
    parser.add_argument("--data", type=Path, required=True, help="processed CMFD forcing directory")
    parser.add_argument("--case-template", type=Path, required=True, help="case containing restartin.nc and metadata.json")
    parser.add_argument("--output", type=Path, required=True, help="benchmark artifact directory")
    parser.add_argument("--python", type=Path, required=True, help="Python executable used by the runner")
    parser.add_argument("--runner", type=Path, default=runner_dir / "run_cmfd_gpu.sh")
    parser.add_argument("--columns", type=parse_columns, default=DEFAULT_COLUMNS)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--modes", default="cpu,gpu", help="comma-separated subset of cpu,gpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_record(source: Path) -> dict[str, object]:
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=source)
    return {
        "path": str(source.resolve()),
        "git_head": command_output(["git", "rev-parse", "HEAD"], source),
        "git_status": (command_output(["git", "status", "--short"], source) or "").splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_size_bytes": len(diff),
    }


def link_case_inputs(case: Path, template: Path) -> None:
    case.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "restartin.nc"):
        target = (template / name).resolve()
        if not target.exists():
            raise FileNotFoundError(f"missing case input: {target}")
        link = case / name
        if link.is_symlink():
            if link.resolve() != target:
                link.unlink()
            else:
                continue
        elif link.exists():
            raise FileExistsError(f"refusing to overwrite non-symlink case input: {link}")
        link.symlink_to(target)


def run_case(
    args: argparse.Namespace,
    mode: str,
    columns: int,
    repeat: int,
) -> dict[str, object]:
    case = args.output / "cases" / f"{mode}_{columns}_rep{repeat:02d}"
    link_case_inputs(case, args.case_template)
    environment = os.environ.copy()
    environment.update({
        "PYTHON": str(args.python),
        "SURF_CODE": str(args.source),
        "SURF_DATA": str(args.data),
        "SURF_CASE": str(case),
        "SURF_DEVICE": "cpu" if mode == "cpu" else "cuda",
        "SURF_COLUMN_COUNT": str(columns),
        "SURF_PHYSICS_DT": "1800",
        "SURF_MAX_STEPS": str(args.steps),
        "SURF_OUTPUT_STRIDE": str(args.steps + 1),
        "SURF_WRITE_OUTPUT": "0",
        "SURF_FAIL_ON_NONFINITE": "0",
    })
    if mode == "cpu":
        threads = str(args.cpu_threads)
        environment.update({
            "SURF_CPU_THREADS": threads,
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
        })
        environment.pop("SURF_PRELOAD_FORCING", None)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["SURF_PRELOAD_FORCING"] = "1"
        if args.cuda_visible_devices is not None:
            environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    command = ["bash", str(args.runner)]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=case,
        env=environment,
        capture_output=True,
        text=True,
    )
    wall_seconds = time.perf_counter() - started
    (case / "run.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"benchmark failed for {mode}/{columns}/repeat {repeat}; see {case / 'run.log'}")
    summary_path = case / "torch_offline_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["nan"] or summary["output_enabled"] or summary["saved_steps"]:
        raise RuntimeError(f"invalid timing run summary: {summary_path}")
    return {
        "mode": mode,
        "columns": columns,
        "repeat": repeat,
        "case": str(case),
        "wall_seconds": wall_seconds,
        **summary,
    }


def median_record(records: list[dict[str, object]], mode: str, columns: int, steps: int) -> dict[str, object]:
    selected = [record for record in records if record["mode"] == mode and record["columns"] == columns]
    integration = [float(record["integration_seconds"]) for record in selected]
    setup = [float(record["setup_seconds"]) for record in selected]
    median_seconds = statistics.median(integration)
    return {
        "mode": mode,
        "columns": columns,
        "steps": steps,
        "repeats": len(selected),
        "integration_seconds_median": median_seconds,
        "integration_seconds_min": min(integration),
        "integration_seconds_max": max(integration),
        "setup_seconds_median": statistics.median(setup),
        "column_steps_per_second": columns * steps / median_seconds,
    }


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.repeats < 1 or args.cpu_threads < 1:
        raise ValueError("steps, repeats, and cpu threads must be positive")
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    if not modes or any(mode not in {"cpu", "gpu"} for mode in modes):
        raise ValueError("modes must contain only cpu and/or gpu")
    for path in (args.source, args.data, args.case_template, args.python, args.runner):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for columns in args.columns:
        for mode in modes:
            for repeat in range(1, args.repeats + 1):
                record = run_case(args, mode, columns, repeat)
                records.append(record)
                print(
                    f"{mode} columns={columns} repeat={repeat}: "
                    f"{float(record['integration_seconds']):.3f} s",
                    flush=True,
                )

    medians = [
        median_record(records, mode, columns, args.steps)
        for columns in args.columns
        for mode in modes
    ]
    by_key = {(record["mode"], record["columns"]): record for record in medians}
    scaling_rows = []
    if "cpu" in modes and "gpu" in modes:
        for columns in args.columns:
            cpu = by_key[("cpu", columns)]
            gpu = by_key[("gpu", columns)]
            scaling_rows.append({
                "columns": columns,
                "cpu_integration_seconds": cpu["integration_seconds_median"],
                "gpu_preload_setup_seconds": gpu["setup_seconds_median"],
                "gpu_preload_integration_seconds": gpu["integration_seconds_median"],
                "gpu_cpu_speedup": float(cpu["integration_seconds_median"]) / float(gpu["integration_seconds_median"]),
            })

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "columns": list(args.columns),
            "steps": args.steps,
            "physics_dt_s": 1800,
            "dtype": "float64",
            "forcing": "CMFD 2018",
            "cpu_threads": args.cpu_threads,
            "cuda_visible_devices": args.cuda_visible_devices,
            "write_output": False,
            "per_step_nonfinite_check": False,
        },
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "gpu_inventory": command_output([
                "nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]),
        },
        "source": source_record(args.source),
        "runner": {
            "path": str(args.runner.resolve()),
            "sha256": sha256(args.runner),
        },
        "runs": records,
        "median_results": medians,
        "scaling": scaling_rows,
    }
    (args.output / "performance_results.json").write_text(json.dumps(report, indent=2) + "\n")
    if scaling_rows:
        with (args.output / "performance_scaling.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scaling_rows[0]))
            writer.writeheader()
            writer.writerows(scaling_rows)
    print(json.dumps({"output": str(args.output), "scaling": scaling_rows}, indent=2))


if __name__ == "__main__":
    main()
