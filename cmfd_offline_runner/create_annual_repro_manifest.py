#!/usr/bin/env python3
"""Write a hash-based manifest for an autonomous annual SURF comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--fortran-10d", type=Path, required=True)
    parser.add_argument("--fortran-final", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def command(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def main() -> None:
    args = parse_args()
    experiment = args.experiment.resolve()
    source = args.source.resolve()
    comparison = experiment / "comparison"
    files = {
        "restart": experiment / "restartin.nc",
        "metadata": experiment / "metadata.json",
        "torch_output": experiment / "torch_offline_daily.npz",
        "torch_summary": experiment / "torch_offline_summary.json",
        "fortran_10d": args.fortran_10d,
        "fortran_final": args.fortran_final,
        "annual_metrics_csv": comparison / "annual_autonomous_metrics.csv",
        "annual_metrics_json": comparison / "annual_autonomous_metrics.json",
        "annual_error_growth_figure": comparison / "annual_autonomous_error_growth.png",
        "annual_soil_temperature_maps_figure": comparison / "annual_autonomous_terminal_soil_temperature_maps.png",
        "annual_soil_water_maps_figure": comparison / "annual_autonomous_terminal_soil_water_maps.png",
        "annual_snow_maps_figure": comparison / "annual_autonomous_terminal_snow_maps.png",
    }
    for pattern in ("*.py", "*.sh"):
        for script in sorted((experiment / "scripts").glob(pattern)):
            files[f"experiment_script:{script.name}"] = script
    for script in (
        Path(__file__),
        Path(__file__).with_name("compare_autonomous_annual.py"),
    ):
        files[f"analysis_script:{script.name}"] = script
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=source)
    python_environment = json.loads(command([
        str(args.python),
        "-c",
        "import json,platform,torch; print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}))",
    ]))
    summary = json.loads((experiment / "torch_offline_summary.json").read_text())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": str(experiment),
        "configuration": {
            "forcing": "CMFD 2018 processed valid-column dataset",
            "forcing_path": str(args.data.resolve()),
            "physics_dt_s": 1800,
            "forcing_interval_s": 10800,
            "offline_boundary_mode": "autonomous VUPDZ0/VDF",
            "precision": "float64 state and arithmetic; float32 retained forcing records",
            "preload_forcing": True,
            "cuda_visible_devices": args.cuda_visible_devices,
            "per_step_nonfinite_check": True,
            "output_stride_steps": summary["output_stride"],
            "saved_steps": summary["saved_steps"],
        },
        "source": {
            "path": str(source),
            "git_head": command(["git", "rev-parse", "HEAD"], cwd=source),
            "git_status_porcelain": command(["git", "status", "--short"], cwd=source).splitlines(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "tracked_diff_size_bytes": len(diff),
        },
        "environment": {
            **python_environment,
            "gpu_inventory": command([
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]).splitlines(),
        },
        "integration_summary": summary,
        "files": {name: file_record(path) for name, path in files.items()},
    }
    target = experiment / "reproducibility_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(target),
        "source_head": manifest["source"]["git_head"],
        "files_hashed": len(files),
    }, indent=2))


if __name__ == "__main__":
    main()
