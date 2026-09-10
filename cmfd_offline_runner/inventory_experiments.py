#!/usr/bin/env python3
"""Inventory every experiment directory under surf_paper/experiments.

Produces a compact JSON + Markdown table (directory, key files, sizes, and
any metadata/summary highlights) for the manuscript reproducibility section.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize(directory: Path) -> dict:
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    entry: dict[str, object] = {
        "total_bytes": total,
        "n_files": len(files),
        "top_files": [
            {"name": str(p.relative_to(directory)), "bytes": p.stat().st_size}
            for p in sorted(files, key=lambda p: -p.stat().st_size)[:6]
        ],
    }
    for name in ("metadata.json", "summary.json", "torch_offline_summary.json",
                 "annual_diagnostics.json", "chain_status.log"):
        path = directory / name
        if path.exists():
            try:
                entry[name] = json.loads(path.read_text())
            except (json.JSONDecodeError, ValueError):
                entry[name] = path.read_text(errors="replace")[:2000]
    return entry


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/data/yangjinhui/surf_pytorch/surf_paper/experiments")
    inventory = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        inventory[directory.name] = summarize(directory)
    out = root / "experiment_inventory.json"
    out.write_text(json.dumps(inventory, indent=1) + "\n")

    lines = ["| Experiment | Files | Size (GB) | Key output |", "|---|---|---|---|"]
    for name, entry in inventory.items():
        key = next((k for k in ("torch_offline_summary.json", "annual_diagnostics.json", "summary.json")
                    if k in entry), "-")
        hint = ""
        if key != "-":
            value = entry[key]
            hint = f"{key}" + (f" ({value.get('nsteps', value.get('years', ''))} steps)"
                               if isinstance(value, dict) else "")
        lines.append(f"| {name} | {entry['n_files']} | {entry['total_bytes'] / 1e9:.2f} | {hint or key} |")
    (root / "experiment_inventory.md").write_text("\n".join(lines) + "\n")
    print(f"inventory: {len(inventory)} experiments -> {out}")


if __name__ == "__main__":
    main()
