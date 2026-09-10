#!/usr/bin/env python3
"""Compute regional SURF skin-temperature sensitivity to local air temperature.

One reverse-mode calculation differentiates the terminal domain-mean skin
temperature with respect to an independent temperature offset at every land
column. SURF has no lateral exchange in this offline configuration, so the
domain-size-scaled gradient is the local column response map.
"""

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
OUTDIR = Path(os.environ["SURF_SENSITIVITY_OUTPUT"])
NSTEPS = int(os.environ.get("SURF_SENSITIVITY_STEPS", "12"))
START_STEP = int(os.environ.get("SURF_START_STEP", "0"))
WARMUP_STEPS = int(os.environ.get("SURF_WARMUP_STEPS", "0"))
STATE_ARCHIVE = os.environ.get("SURF_STATE_ARCHIVE")
STATE_INDEX = int(os.environ.get("SURF_STATE_INDEX", "-1"))
NCOLUMNS = int(os.environ.get("SURF_COLUMN_COUNT", "97709"))
SUBSET_INDEX = os.environ.get("SURF_SUBSET_INDEX")
DEVICE = os.environ.get("SURF_DEVICE", "cuda")
DTYPE = torch.float64

sys.path[:0] = [str(CODE), str(Path(__file__).parent)]

from diagnostic_scripts_compare_realcol import make_forcing  # noqa: E402
from replay_full_domain_controlled import load_initial  # noqa: E402
from run_full_domain_offline import interpolated_runtime  # noqa: E402
from surf_pytorch import LandSurface, SurfConfig  # noqa: E402


def restore_sampled_state(state, static, selected, model):
    """Restore slow prognostic fields from a sampled annual integration.

    The archive intentionally contains only output-relevant prognostic fields.
    Fast canopy and boundary-layer memory variables are therefore adjusted by
    the unperturbed warm-up before a derivative window begins.
    """
    if STATE_ARCHIVE is None:
        return state, None
    archive = np.load(STATE_ARCHIVE, mmap_mode="r")
    if not 0 <= STATE_INDEX < archive["step"].size:
        raise ValueError(
            f"SURF_STATE_INDEX must be in 0..{archive['step'].size - 1}"
        )

    def tensor(name, scale=1.0):
        values = np.asarray(archive[name][STATE_INDEX, selected], dtype=np.float64)
        return torch.as_tensor(values * scale, dtype=DTYPE, device=DEVICE)

    skin_t = tensor("skin_t")
    soil_t = tensor("soil_t")
    soil_w = tensor("soil_w") / (static.soil_thickness.unsqueeze(0) * 1000.0)
    snow_w = tensor("snow_w")
    snow_rho = tensor("snow_rho")
    snow_t = tensor("snow_t")
    snow_alb = tensor("snow_alb")
    snow_frac = torch.clamp(10.0 * snow_w / torch.clamp(snow_rho, min=1.0e-12), 0.0, 1.0)
    # ``radiative_skin_t`` is a diagnostic tile average in the online state.
    # Recompute it from the archived tile temperatures and matching snow cover.
    provisional = replace(state, snow_w=snow_w, snow_rho=snow_rho, snow_frac=snow_frac)
    tile_frac = model._tile_frac(static, skin_t.shape[1], state=provisional)
    radiative_skin_t = (skin_t * tile_frac).sum(dim=1)
    restored = replace(
        state, skin_t=skin_t, soil_t=soil_t, soil_w_liq=soil_w,
        snow_w=snow_w, snow_rho=snow_rho, snow_t=snow_t, snow_alb=snow_alb,
        snow_frac=snow_frac, radiative_skin_t=radiative_skin_t,
        vupdz0_step=0, vupdz0_pustrti=None, vupdz0_pvstrti=None,
        vupdz0_pahfsti=None, vupdz0_pevapti=None, vupdz0_pzdl=None,
    )
    return restored, int(archive["step"][STATE_INDEX])


def main() -> None:
    if os.environ.get("SURF_GRADIENT_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
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
        for name in ("tair_K", "qair_kg_kg", "psurf_Pa", "swdown_W_m2", "lwdown_W_m2", "wind_m_s", "precip_kg_m2_s")
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
    )).to(DEVICE)
    state, archive_step = restore_sampled_state(state, static, selected, model)
    if archive_step is None:
        initial_skin_t = state.radiative_skin_t.unsqueeze(-1).expand_as(state.skin_t).clone()
        initial_skin_t[:, 0] = state.sst
        state = replace(state, skin_t=initial_skin_t)
    elif START_STEP != archive_step:
        raise ValueError(
            f"SURF_START_STEP={START_STEP} does not match archived step {archive_step}"
        )
    args = SimpleNamespace(
        shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
        shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
        stress_u=None, stress_v=None, vup=None,
    )
    def advance(current_state, physical_step, tair_perturbation=None):
        fraction = model._tile_frac(static, model.config.n_tile, state=current_state)
        runtime = interpolated_runtime(arrays, physical_step, 1800.0, lat, lon, DEVICE)
        emission = model._surface_emissivity(current_state, static, fraction, DTYPE, DEVICE)
        forcing = make_forcing(runtime, 0, current_state, replace(static, emis=emission), args, tile_frac=fraction)
        precip_amount = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
        forcing = replace(
            forcing,
            t=forcing.t if tair_perturbation is None else forcing.t + tair_perturbation,
            rain=precip_amount * (1.0 - runtime.precip_snow_phase),
            snow=precip_amount * runtime.precip_snow_phase,
        )
        return model(current_state, forcing, replace(static, emis=emission))[0]

    # Re-equilibrate omitted fast stores and the autonomous VDF memory before
    # differentiating. This warm-up is deliberately outside the AD graph.
    with torch.no_grad():
        for step in range(WARMUP_STEPS):
            state = advance(state, START_STEP + step)

    tair_offset = torch.zeros(NCOLUMNS, dtype=DTYPE, device=DEVICE, requires_grad=True)
    for step in range(NSTEPS):
        state = advance(state, START_STEP + WARMUP_STEPS + step, tair_offset)

    fraction = model._tile_frac(static, model.config.n_tile, state=state)
    terminal_skin = (state.skin_t * fraction).sum(dim=1)
    domain_mean = terminal_skin.mean()
    gradient = torch.autograd.grad(domain_mean, tair_offset)[0]
    # Since each column is independent, undo the 1/N factor from the mean.
    local_sensitivity = gradient * NCOLUMNS
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTDIR / "regional_tair_sensitivity.npz",
        lat=lat,
        lon=lon,
        terminal_skin_t=terminal_skin.detach().cpu().numpy(),
        dskin_dtair=local_sensitivity.detach().cpu().numpy(),
    )
    summary = {
        "ncolumns": NCOLUMNS,
        "nsteps": NSTEPS,
        "start_step": START_STEP,
        "warmup_steps": WARMUP_STEPS,
        "archive_state_step": archive_step,
        "physics_dt_seconds": 1800,
        "domain_mean_skin_t_K": float(domain_mean.detach().cpu()),
        "sensitivity_min_K_per_K": float(local_sensitivity.min().detach().cpu()),
        "sensitivity_mean_K_per_K": float(local_sensitivity.mean().detach().cpu()),
        "sensitivity_max_K_per_K": float(local_sensitivity.max().detach().cpu()),
        "finite": bool(torch.isfinite(local_sensitivity).all()),
    }
    (OUTDIR / "regional_tair_sensitivity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
