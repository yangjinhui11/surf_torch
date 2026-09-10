#!/usr/bin/env python3
"""Determinism probe for the multiyear decade chain.

Replays the first N days of the chained 2009 integration from the shared
restart with the exact production loop (runtime_for_step + make_forcing +
LandSurface.forward), then dumps every tensor field of the final SurfState.

Variants are selected by
  --zsn 9000.0   the shipped permanent-snow threshold (mask active)
  --zsn 1e30     threshold neutralised (mask provably all-False; equivalent
                 to the pre-fix code for these states)
and by CUDA_VISIBLE_DEVICES in the launching shell (A6000 vs A100), so that
runs can be compared bitwise across code variant and GPU architecture.

Exact hex checksums of a few fields are printed every 5 days so divergence
onset can be located from the logs.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

CODE = Path(os.environ["SURF_CODE"])
sys.path[:0] = [str(CODE), str(Path(__file__).parent)]

import run_torch_multiyear_diag as diag  # noqa: E402  (sets sys.path itself)
from replay_full_domain_controlled import DTYPE, load_initial  # noqa: E402
from diagnostic_scripts_compare_realcol import make_forcing  # noqa: E402
from surf_pytorch import LandSurface, SurfConfig, SurfState  # noqa: E402

FORCING_DT = 10_800.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--zsn", type=float, default=9000.0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = "cuda"
    root = Path("/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018")
    initial = Path("/data/yangjinhui/surf_pytorch/surf_paper/experiments/initial_states/native_20d_terminal_complete.nc")

    first = root / "cmfd_valid_columns_2009_dataset"
    import json
    metadata = json.loads((first / "metadata.json").read_text())
    nvalid = int(metadata["ncols"])
    lat = np.asarray(np.load(first / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(first / "lon.npy", mmap_mode="r"), dtype=np.float64)

    state, static = load_initial(initial, nvalid, device, lat, lon)
    initial_skin_t = state.radiative_skin_t.unsqueeze(-1).expand_as(state.skin_t).clone()
    initial_skin_t[:, 0] = state.sst
    state = replace(state, skin_t=initial_skin_t)

    config = SurfConfig(dtype=DTYPE, device=device, validate=False, use_farquhar=True,
                        use_ags=False, levgen=True, lelwtl=True, lelaiv=False,
                        vup_boundary_mode="offline")
    model = LandSurface(config).to(device)
    # Variant knob: move the permanent-snow threshold (0-d tensor, same
    # dtype/device as the rest of the bundle).  Older code states without
    # the field simply skip the override (their reset branch does not
    # exist either).
    if hasattr(model.consts, "zsn_pert"):
        model.consts.zsn_pert = torch.tensor(args.zsn, dtype=DTYPE, device=device)

    sim_args = SimpleNamespace(shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
                               shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
                               stress_u=None, stress_v=None, vup=None)

    arrays = {n: np.load(root / f"cmfd_valid_columns_2009_dataset/{n}.npy", mmap_mode="r") for n in
              ("tair_K", "qair_kg_kg", "psurf_Pa", "swdown_W_m2", "lwdown_W_m2", "wind_m_s", "precip_kg_m2_s")}
    next_arrays = {n: np.load(root / f"cmfd_valid_columns_2010_dataset/{n}.npy", mmap_mode="r") for n in arrays}

    dt = 1800.0
    nsteps = args.days * int(86_400 / dt)
    print(f"probe: days={args.days} zsn={args.zsn:g} device={torch.cuda.get_device_name(0)}", flush=True)

    with torch.inference_mode():
        for step in range(nsteps):
            fraction = model._tile_frac(static, config.n_tile, state=state)
            runtime = diag.runtime_for_step(arrays, next_arrays, step, dt, lat, lon, device)
            emission = model._surface_emissivity(state, static, fraction, DTYPE, device)
            forcing = make_forcing(runtime, 0, state, replace(static, emis=emission), sim_args, tile_frac=fraction)
            amount = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
            forcing = replace(forcing, rain=amount * (1.0 - runtime.precip_snow_phase),
                              snow=amount * runtime.precip_snow_phase)
            state, _ = model(state, forcing, replace(static, emis=emission))
            if (step + 1) % int(5 * 86_400 / dt) == 0:
                swe = state.snow_w.sum().item()
                sw = (state.soil_w_liq * static.soil_thickness).sum().item()
                print(f"day {(step + 1) // 48:3d}: swe_sum={swe.hex()} soilw_sum={sw.hex()}", flush=True)

    payload = {}
    for f in fields(SurfState):
        v = getattr(state, f.name)
        if isinstance(v, torch.Tensor):
            payload[f.name] = v.detach().cpu().numpy()
    payload["__zsn"] = np.array(args.zsn)
    payload["__days"] = np.array(args.days)
    np.savez(args.out, **payload)
    print(f"saved {args.out}: snow_w max={state.snow_w.max().item():.13f} "
          f"mean={state.snow_w.mean().item():.15f}", flush=True)


if __name__ == "__main__":
    main()
