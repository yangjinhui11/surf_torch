#!/usr/bin/env python3
"""Compute regional gradients with respect to vegetation parameters."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


CASE = Path(os.environ["SURF_CASE"])
CODE = Path(os.environ["SURF_CODE"])
DATA = Path(os.environ["SURF_DATA"])
OUTDIR = Path(os.environ["SURF_PARAMETER_OUTPUT"])
NSTEPS = int(os.environ.get("SURF_PARAMETER_STEPS", "12"))
NCOLUMNS = int(os.environ.get("SURF_COLUMN_COUNT", "97709"))
START_STEP = int(os.environ.get("SURF_START_STEP", "0"))
WARMUP_STEPS = int(os.environ.get("SURF_WARMUP_STEPS", "48"))
SUBSET_INDEX = os.environ.get("SURF_SUBSET_INDEX")
DEVICE = os.environ.get("SURF_DEVICE", "cuda")
PARAM_NAMES = tuple(
    name.strip() for name in os.environ.get(
        "SURF_PARAMETER_NAMES", "rvlai,rvrsmin,rvz0m"
    ).split(",") if name.strip()
)
DTYPE = torch.float64

# Reuse the validated annual-state restoration helper without duplicating its
# state decoding logic. The imported module also defines an output directory,
# which is irrelevant for this parameter experiment.
os.environ.setdefault("SURF_SENSITIVITY_OUTPUT", str(OUTDIR))

sys.path[:0] = [str(CODE), str(Path(__file__).parent)]

from diagnostic_scripts_compare_realcol import make_forcing  # noqa: E402
from regional_tair_sensitivity import restore_sampled_state  # noqa: E402
from replay_full_domain_controlled import load_initial  # noqa: E402
from run_full_domain_offline import interpolated_runtime  # noqa: E402
from surf_pytorch import LandSurface, SurfConfig  # noqa: E402


def main() -> None:
    metadata = json.loads((CASE / "metadata.json").read_text())
    nvalid = int(metadata["nvalid"])
    if SUBSET_INDEX is not None:
        subset_index = int(SUBSET_INDEX)
        if NCOLUMNS != 1:
            raise ValueError("SURF_SUBSET_INDEX requires SURF_COLUMN_COUNT=1")
        if not 0 <= subset_index < nvalid:
            raise ValueError(f"SURF_SUBSET_INDEX must be in 0..{nvalid - 1}")
        selected = slice(subset_index, subset_index + 1)
    else:
        selected = slice(0, NCOLUMNS)
    if not 1 <= NCOLUMNS <= nvalid:
        raise ValueError(f"SURF_COLUMN_COUNT must be in 1..{nvalid}")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    full_lat = np.asarray(np.load(DATA / "lat.npy", mmap_mode="r"), dtype=np.float64)
    full_lon = np.asarray(np.load(DATA / "lon.npy", mmap_mode="r"), dtype=np.float64)
    lat, lon = full_lat[selected], full_lon[selected]
    arrays = {
        name: np.load(DATA / f"{name}.npy", mmap_mode="r")[:, selected]
        for name in (
            "tair_K", "qair_kg_kg", "psurf_Pa", "swdown_W_m2",
            "lwdown_W_m2", "wind_m_s", "precip_kg_m2_s",
        )
    }
    state, static = load_initial(CASE / "restartin.nc", nvalid, DEVICE, full_lat, full_lon)

    def subset_dataclass(value):
        updates = {
            field.name: getattr(value, field.name)[selected]
            for field in fields(value)
            if isinstance(getattr(value, field.name), torch.Tensor)
            and getattr(value, field.name).ndim > 0
            and getattr(value, field.name).shape[0] == nvalid
        }
        return replace(value, **updates)

    state, static = subset_dataclass(state), subset_dataclass(static)
    model = LandSurface(SurfConfig(
        dtype=DTYPE, device=DEVICE, differentiable=True, validate=False,
        use_farquhar=True, use_ags=False, levgen=True, lelwtl=True,
        lelaiv=False, vup_boundary_mode="offline",
        learnable_veg=PARAM_NAMES,
    )).to(DEVICE)
    state, archive_step = restore_sampled_state(state, static, selected, model)
    if archive_step is None:
        raise ValueError("A sampled annual state archive is required")
    if START_STEP != archive_step:
        raise ValueError(
            f"SURF_START_STEP={START_STEP} does not match archived step {archive_step}"
        )

    args = SimpleNamespace(
        shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
        shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
        stress_u=None, stress_v=None, vup=None,
    )

    def advance(current_state, physical_step):
        fraction = model._tile_frac(static, model.config.n_tile, state=current_state)
        runtime = interpolated_runtime(arrays, physical_step, 1800.0, lat, lon, DEVICE)
        emission = model._surface_emissivity(current_state, static, fraction, DTYPE, DEVICE)
        forcing = make_forcing(
            runtime, 0, current_state, replace(static, emis=emission), args,
            tile_frac=fraction,
        )
        precip_amount = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
        forcing = replace(
            forcing,
            rain=precip_amount * (1.0 - runtime.precip_snow_phase),
            snow=precip_amount * runtime.precip_snow_phase,
        )
        return model(current_state, forcing, replace(static, emis=emission))

    with torch.no_grad():
        for step in range(WARMUP_STEPS):
            state, _ = advance(state, START_STEP + step)

    evaporation = torch.zeros((), dtype=DTYPE, device=DEVICE)
    for step in range(NSTEPS):
        state, diag = advance(state, START_STEP + WARMUP_STEPS + step)
        evaporation = evaporation + diag.e_flux.mean() * 1800.0

    fraction = model._tile_frac(static, model.config.n_tile, state=state)
    terminal_skin = (state.skin_t * fraction).sum(dim=1).mean()
    terminal_swe = state.snow_w.mean()
    parameters = [getattr(model.veg_table, name) for name in PARAM_NAMES]

    losses = {
        "terminal_skin_t_K": terminal_skin,
        "cumulative_evaporation_kg_m2": evaporation,
        "terminal_swe_kg_m2": terminal_swe,
    }
    gradients = {}
    for loss_name, loss in losses.items():
        values = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True,
        )
        gradients[loss_name] = np.stack([
            np.zeros_like(parameter.detach().cpu().numpy()) if value is None
            else value.detach().cpu().numpy()
            for parameter, value in zip(parameters, values)
        ])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTDIR / "regional_parameter_sensitivity.npz",
        lat=lat,
        lon=lon,
        parameter_names=np.asarray(PARAM_NAMES),
        parameter_values=np.stack([p.detach().cpu().numpy() for p in parameters]),
        **gradients,
    )
    summary = {
        "ncolumns": NCOLUMNS,
        "nsteps": NSTEPS,
        "warmup_steps": WARMUP_STEPS,
        "start_step": START_STEP,
        "archive_state_step": archive_step,
        "parameter_names": list(PARAM_NAMES),
        "finite": {
            name: bool(np.isfinite(value).all())
            for name, value in gradients.items()
        },
        "terminal_skin_t_K": float(terminal_skin.detach().cpu()),
        "cumulative_evaporation_kg_m2": float(evaporation.detach().cpu()),
        "terminal_swe_kg_m2": float(terminal_swe.detach().cpu()),
    }
    (OUTDIR / "regional_parameter_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
