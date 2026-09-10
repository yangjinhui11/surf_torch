#!/usr/bin/env python3
"""Run the autonomous Torch offline SURF closure for a CMFD full-domain case."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from netCDF4 import Dataset

CASE = Path(os.environ["SURF_CASE"])
CODE = Path(os.environ["SURF_CODE"])
DATA = Path(os.environ.get(
    "SURF_DATA",
    "/data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset",
))
sys.path[:0] = [
    str(CODE),
    str(Path(__file__).parent),
]

from diagnostic_scripts_compare_realcol import make_forcing  # noqa: E402
from replay_full_domain_controlled import DTYPE, RuntimeWindow, load_initial  # noqa: E402
from surf_pytorch import LandSurface, SurfConfig  # noqa: E402
import surf_pytorch.model as model_module  # noqa: E402


FORCING_DT = 10_800.0


def solar_cosine_at_time(lat: np.ndarray, lon: np.ndarray, seconds: float) -> np.ndarray:
    """Analytic CMFD solar geometry at an arbitrary offline model time."""
    hours = seconds / 3600.0
    day = 1 + int(hours // 24.0)
    delta = math.radians(23.45) * math.sin(2.0 * math.pi * (284 + day) / 365.0)
    omega = np.deg2rad(15.0 * ((hours % 24.0) + lon / 15.0 - 12.0))
    return np.clip(
        np.sin(np.deg2rad(lat)) * math.sin(delta)
        + np.cos(np.deg2rad(lat)) * math.cos(delta) * np.cos(omega),
        0.0,
        1.0,
    )


def interpolated_runtime(
    arrays: dict[str, np.ndarray], physical_step: int, physics_dt: float,
    lat: np.ndarray, lon: np.ndarray, device: str,
) -> SimpleNamespace:
    """Build the two-state forcing window used by ``make_forcing``.

    OSM's ``ZDTFORC`` remains three hours while ``TSTEP`` may be shorter.
    The forcing reader linearly interpolates scalar meteorology between its
    records; this reproduces that contract before invoking the Torch kernel.
    """
    if FORCING_DT % physics_dt:
        raise ValueError("SURF_PHYSICS_DT must divide the 10800 s CMFD forcing interval")
    start = physical_step * physics_dt
    nrecord = next(iter(arrays.values())).shape[0]

    def sample(values: np.ndarray, seconds: float) -> np.ndarray:
        # The two terminal OSM forcing records repeat the first records of
        # the forcing year.  The Torch driver must use that same cycle at
        # an annual boundary rather than hold the final record.
        # Keep the index and interpolation phase in the same forcing cycle.
        cycle_seconds = seconds % (nrecord * FORCING_DT)
        record = int(cycle_seconds // FORCING_DT)
        next_record = (record + 1) % nrecord
        alpha = (cycle_seconds - record * FORCING_DT) / FORCING_DT
        # OSM reads f4 forcing values then performs the interpolation in REAL8.
        first = np.asarray(values[record], dtype=np.float32).astype(np.float64)
        second = np.asarray(values[next_record], dtype=np.float32).astype(np.float64)
        return first + alpha * (second - first)

    # OSM linearly interpolates state meteorology, but its offline alignment
    # holds radiation and precipitation at the next three-hour record for the
    # entire interval. This is also the legacy 10800 s ``shift_flux`` path.
    held_fluxes = {"swdown_W_m2", "lwdown_W_m2", "precip_kg_m2_s"}
    flux_record = (int(start // FORCING_DT) + 1) % nrecord
    runtime = SimpleNamespace()
    runtime.dt = physics_dt
    for name, values in arrays.items():
        if name in held_fluxes:
            flux = np.asarray(values[flux_record], dtype=np.float32).astype(np.float64)
            window = np.stack((flux, flux))
        else:
            window = np.stack((sample(values, start), sample(values, start + physics_dt)))
        setattr(runtime, name, torch.as_tensor(window, dtype=DTYPE, device=device))
    # CMFD supplies total precipitation. OSM's forcing preparation splits
    # it into Rainf/Snowf once per 3-hour record, using that record's air
    # temperature; it does not reclassify the held precipitation on each
    # 1800 s physics substep.
    runtime.precip_snow_phase = torch.as_tensor(
        np.asarray(arrays["tair_K"][flux_record], dtype=np.float32) <= 273.15,
        dtype=DTYPE,
        device=device,
    )
    flux_time = (int(start // FORCING_DT) + 1) * FORCING_DT
    mu0_flux = solar_cosine_at_time(lat, lon, flux_time)
    runtime.mu0 = torch.as_tensor(
        np.stack((mu0_flux, mu0_flux)), dtype=DTYPE, device=device,
    )
    runtime.zero = torch.zeros(lat.size, dtype=DTYPE, device=device)
    runtime.co2 = torch.full((lat.size,), 6.0e-4, dtype=DTYPE, device=device)
    runtime.dt_tensor = torch.tensor(physics_dt, dtype=DTYPE, device=device)
    runtime.sigma = torch.tensor(5.6705085385054389e-08, dtype=DTYPE, device=device)
    return runtime


def legacy_main() -> None:
    metadata = json.loads((CASE / "metadata.json").read_text())
    captured_nvalid = int(metadata["nvalid"])
    subset_index_env = os.environ.get("SURF_SUBSET_INDEX")
    subset_index = int(subset_index_env) if subset_index_env is not None else None
    if subset_index is not None and not 0 <= subset_index < captured_nvalid:
        raise ValueError(
            f"SURF_SUBSET_INDEX={subset_index} is outside 0..{captured_nvalid - 1}"
        )
    nvalid = 1 if subset_index is not None else captured_nvalid
    nsteps = int(os.environ.get("SURF_MAX_STEPS", "240"))
    physics_dt = float(os.environ.get("SURF_PHYSICS_DT", str(FORCING_DT)))
    stride = int(os.environ.get("SURF_OUTPUT_STRIDE", "8"))
    fail_on_nonfinite = os.environ.get("SURF_FAIL_ON_NONFINITE") == "1"
    trace_internal = os.environ.get("SURF_TRACE_INTERNAL") == "1"
    trace_step = int(os.environ.get("SURF_TRACE_STEP", "0"))
    trace_index_env = os.environ.get("SURF_TRACE_INDEX")
    trace_index = int(trace_index_env) if trace_index_env is not None else None
    trace_start = int(os.environ.get("SURF_TRACE_START", "0"))
    trace_stop = int(os.environ.get("SURF_TRACE_STOP", str(nsteps)))
    if trace_index is not None:
        if subset_index is not None and trace_index != subset_index:
            raise ValueError("SURF_TRACE_INDEX must equal SURF_SUBSET_INDEX")
        if subset_index is None and not 0 <= trace_index < captured_nvalid:
            raise ValueError(
                f"SURF_TRACE_INDEX={trace_index} is outside 0..{captured_nvalid - 1}"
            )
    capture_active = False
    requested_device = os.environ.get("SURF_DEVICE")
    if requested_device is not None and requested_device not in {"cpu", "cuda"}:
        raise ValueError("SURF_DEVICE must be 'cpu' or 'cuda'")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SURF_DEVICE=cuda was requested but CUDA is unavailable")
    device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
    lat = np.asarray(np.load(DATA / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(DATA / "lon.npy", mmap_mode="r"), dtype=np.float64)
    solar_lat = lat
    fortran_forcing = os.environ.get("SURF_FORTRAN_FORCING")
    if fortran_forcing:
        with Dataset(fortran_forcing) as dataset:
            encoded_lat = np.asarray(dataset["lat"][:], dtype=np.float64).reshape(-1)
        if encoded_lat.size == 1:
            solar_lat = np.full_like(lat, encoded_lat.item())
        elif encoded_lat.size == lat.size:
            solar_lat = encoded_lat
        else:
            raise ValueError(
                "Fortran forcing latitude has incompatible size "
                f"{encoded_lat.size}; expected 1 or {lat.size}"
            )
    arrays = {
        name: np.load(DATA / f"{name}.npy", mmap_mode="r")
        for name in ("tair_K", "qair_kg_kg", "psurf_Pa", "swdown_W_m2", "lwdown_W_m2", "wind_m_s", "precip_kg_m2_s")
    }
    state, static = load_initial(
        CASE / "restartin.nc", captured_nvalid, device, lat, lon,
    )
    if subset_index is not None:
        selected = slice(subset_index, subset_index + 1)

        def subset_dataclass(value):
            updates = {
                field.name: getattr(value, field.name)[selected]
                for field in fields(value)
                if isinstance(getattr(value, field.name), torch.Tensor)
                and getattr(value, field.name).ndim > 0
                and getattr(value, field.name).shape[0] == captured_nvalid
            }
            return replace(value, **updates)

        state, static = subset_dataclass(state), subset_dataclass(static)
        lat, lon, solar_lat = lat[selected], lon[selected], solar_lat[selected]
        arrays = {name: values[:, selected] for name, values in arrays.items()}
        trace_index = 0 if trace_index is not None else None
    # SURFEXCDRIVER_CTL resets tiled PTSKTI from PTSKM1M at KSTEP=0 and
    # then restores the water tile from PSST.  Restart SkinT is therefore
    # not the first-call land-tile temperature in OSM offline mode.
    initial_skin_t = state.radiative_skin_t.unsqueeze(-1).expand_as(state.skin_t).clone()
    initial_skin_t[:, 0] = state.sst
    state = replace(state, skin_t=initial_skin_t)
    config = SurfConfig(
        dtype=DTYPE, device=device, validate=False, use_farquhar=True,
        use_ags=False, levgen=True, lelwtl=True, lelaiv=False, vup_boundary_mode="offline",
    )
    model = LandSurface(config).to(device)
    captured_diffusion = {}
    if trace_internal:
        build_diffusion_inputs = model._build_diffusion_inputs

        def capture_diffusion_inputs(*args, **kwargs):
            value = build_diffusion_inputs(*args, **kwargs)
            if capture_active and "pcfhti" not in captured_diffusion:
                for name in (
                    "pcfhti", "pcfqti", "sskm1m", "qskm1m", "alphal", "alphas",
                    "dqsdT", "pfrti", "pssrflti", "pslrfl", "pemis", "pevapsnw", "ptskrad",
                ):
                    field = getattr(value, name)
                    if field is not None:
                        captured_diffusion[name] = field.detach().cpu().numpy()
            return value

        model._build_diffusion_inputs = capture_diffusion_inputs
        build_tile_vevap = model_module.compute_tile_vevap

        def capture_tile_vevap(*args, **kwargs):
            if capture_active and "jarvis_pwetl" not in captured_diffusion:
                for name in ("pwetl", "pweth", "pwetb", "pwetlu", "pweths"):
                    captured_diffusion[f"jarvis_{name}"] = kwargs[name].detach().cpu().numpy()
            return build_tile_vevap(*args, **kwargs)

        model_module.compute_tile_vevap = capture_tile_vevap
    args = SimpleNamespace(
        shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
        shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
        stress_u=None, stress_v=None, vup=None,
    )
    outputs = {name: [] for name in ("step", "skin_t", "soil_t", "soil_w", "snow_w", "snow_t", "snow_rho", "snow_alb")}
    internal = {}
    trace: dict[str, list[float]] = {"step": []}

    def traced(value: torch.Tensor, tile: int | None = None) -> float:
        if trace_index is None:
            raise RuntimeError("trace index is not configured")
        if value.numel() == 0:
            return float("nan")
        if value.ndim == 0:
            return float(value.detach().cpu())
        # Most fields are [column, tile], but SnowInputs retains a few
        # Fortran-shaped [tile, column] work arrays.
        if tile is None:
            chosen = value[trace_index] if value.shape[0] == nvalid else value[..., trace_index]
        else:
            chosen = value[trace_index, tile] if value.shape[0] == nvalid else value[tile, trace_index]
        return float(chosen.reshape(-1)[0].detach().cpu())

    def record_trace(step: int, fraction: torch.Tensor, forcing) -> None:
        """Save a compact single-column trajectory without changing the solve."""
        snow_in, snow_out = model._last_snow_inputs, model._last_snow_outputs
        if snow_in is None or snow_out is None:
            raise RuntimeError("snow trace requested before the snow solver ran")
        trace["step"].append(float(step + 1))
        for name in (
            "pssnm1m", "ptsnm1m", "prsnm1m", "pslrfl", "pssfc", "pssfl",
            "ptsfc", "ptsfl", "pevapsnw", "ptmst",
        ):
            trace.setdefault(f"snow_in_{name}", []).append(traced(getattr(snow_in, name)))
        for name in ("ptsn", "pgsn", "psstar", "plwc", "zfrsn", "pmsn", "pmsnint", "pemssn"):
            value = getattr(snow_out, name)
            trace.setdefault(f"snow_out_{name}", []).append(np.nan if value is None else traced(value))
        for name in ("snow_w", "snow_t", "snow_rho", "snow_alb", "skin_water"):
            trace.setdefault(f"state_{name}", []).append(traced(getattr(state, name)))
        for tile, label in ((4, "5"), (6, "7")):
            trace.setdefault(f"state_skin_t{label}", []).append(
                traced(state.skin_t, tile)
            )
        trace.setdefault("state_soil_t1", []).append(traced(state.soil_t, 0))
        trace.setdefault("state_soil_w1", []).append(traced(state.soil_w_liq, 0))
        for name in ("t", "q", "p_surf", "fr_so", "fr_th"):
            trace.setdefault(f"forcing_{name}", []).append(traced(getattr(forcing, name)))
        for tile, label in ((4, "5"), (6, "7")):
            trace.setdefault(f"fraction_{label}", []).append(traced(fraction, tile))
            for name in ("pfrti", "pahfsti", "pevapti", "pssrflti", "pslrflti"):
                trace.setdefault(f"snow_in_{name}{label}", []).append(traced(getattr(snow_in, name), tile))
        for name in (
            "_last_pahfsti", "_last_pevapti", "_last_pcfh_tile", "_last_pcfq_tile",
            "_last_vdf_ptsktip1_tile",
        ):
            value = getattr(model, name, None)
            if value is not None:
                for tile, label in ((4, "5"), (6, "7")):
                    trace.setdefault(name.removeprefix("_last_") + label, []).append(
                        traced(value, tile)
                    )
        # Diagnose the autonomous atmospheric closure before snow-process
        # inputs are consumed. These fields locate the first divergent step.
        for name in (
            "_last_pustrti", "_last_pvstrti", "_last_vup_pbuom0",
            "_last_vup_pzdl0", "_last_vup_praqti", "_last_vexcs_pzdl_tile",
        ):
            value = getattr(model, name, None)
            if value is not None:
                for tile, label in ((4, "5"), (6, "7")):
                    trace.setdefault(name.removeprefix("_last_") + label, []).append(
                        traced(value, tile)
                    )
    with torch.inference_mode():
        for step in range(nsteps):
            capture_active = trace_internal and step == trace_step
            # SURFRAD weights broadband emissivity with the current snow
            # partition, not the immutable snow-free base fractions.
            fraction = model._tile_frac(static, config.n_tile, state=state)
            runtime = interpolated_runtime(arrays, step, physics_dt, solar_lat, lon, device)
            emission = model._surface_emissivity(state, static, fraction, DTYPE, device)
            step_static = replace(static, emis=emission)
            forcing = make_forcing(runtime, 0, state, step_static, args, tile_frac=fraction)
            # ``make_forcing`` is shared with experiments whose precipitation
            # phase is diagnosed at the physics step. This OSM-compatible
            # offline driver consumes a pre-split 3-hour forcing instead.
            precip_amount = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
            forcing = replace(
                forcing,
                rain=precip_amount * (1.0 - runtime.precip_snow_phase),
                snow=precip_amount * runtime.precip_snow_phase,
            )
            if trace_internal and step == trace_step:
                ktvl_trace, ktvh_trace = model._dominant_veg_types(static)
                z0m_trace, z0h_trace, z0q_trace = model._per_tile_z0(
                    static, fraction, ktvl_trace, ktvh_trace, config.n_tile, DTYPE, device, state=state,
                )
                internal["pre_vup_z0m"] = z0m_trace.detach().cpu().numpy()
                internal["pre_vup_z0h"] = z0h_trace.detach().cpu().numpy()
                internal["pre_vup_z0q"] = z0q_trace.detach().cpu().numpy()
                for name in ("t", "q", "p_surf", "geom1", "u", "v"):
                    internal[f"trace_forcing_{name}"] = getattr(forcing, name).detach().cpu().numpy()
            if trace_internal and step == 0:
                ktvl, ktvh = model._dominant_veg_types(static)
                z0m, z0h, z0q = model._per_tile_z0(
                    static, fraction, ktvl, ktvh, config.n_tile, DTYPE, device, state=state,
                )
                model._vupdz0_cold_start_fluxes(forcing, z0m)
                internal["cold_pustrti"] = model._last_pustrti.detach().cpu().numpy()
                internal["cold_pvstrti"] = model._last_pvstrti.detach().cpu().numpy()
                internal["cold_z0m"] = z0m.detach().cpu().numpy()
                internal["cold_z0h"] = z0h.detach().cpu().numpy()
                internal["cold_z0q"] = z0q.detach().cpu().numpy()
                for name in ("t", "q", "p_surf", "geom1", "u", "v"):
                    internal[f"forcing_{name}"] = getattr(forcing, name).detach().cpu().numpy()
            state, _ = model(state, forcing, step_static)
            if fail_on_nonfinite:
                fields_to_check = ("skin_t", "soil_t", "soil_w_liq", "snow_w", "snow_t", "snow_rho", "snow_alb")
                failures = {}
                for name in fields_to_check:
                    bad = ~torch.isfinite(getattr(state, name))
                    if bad.any():
                        columns = torch.nonzero(bad.reshape(bad.shape[0], -1).any(dim=1)).flatten()
                        failures[name] = columns[:10].detach().cpu().tolist()
                if failures:
                    raise FloatingPointError(f"non-finite state at step {step + 1}: {failures}")
            if trace_index is not None and trace_start <= step < trace_stop:
                record_trace(step, fraction, forcing)
            if trace_internal and step == trace_step:
                for name, value in model._last_jarvis.items():
                    internal[f"jarvis_{name}"] = value.detach().cpu().numpy()
                for name in (
                    "_last_pustrti", "_last_pvstrti", "_last_pahfsti", "_last_pevapti",
                    "_last_vup_pzdl0", "_last_vup_pbuom0", "_last_vup_praqti",
                    "_last_vexcs_pcpts_tile", "_last_vexcs_pqsam_tile",
                    "_last_vexcs_z0m_tile", "_last_vexcs_z0h_tile", "_last_vexcs_z0q_tile",
                    "_last_pcfm_tile", "_last_pcfh_tile", "_last_pcfq_tile",
                    "_last_vexcs_pzdl_tile", "_last_vexcs_pbuom_input_tile",
                    "_last_vdf_pstr_tile", "_last_vdf_ptsk_tile", "_last_vdf_ptsktip1_tile",
                    "_last_vdf_pslrfl", "_last_vdf_ptskrad",
                ):
                    value = getattr(model, name, None)
                    if value is not None:
                        internal[name.removeprefix("_last_")] = value.detach().cpu().numpy()
            if (step + 1) % stride == 0:
                outputs["step"].append(np.array(step + 1, dtype=np.int64))
                outputs["skin_t"].append(state.skin_t.cpu().numpy())
                outputs["soil_t"].append(state.soil_t.cpu().numpy())
                outputs["soil_w"].append((state.soil_w_liq * static.soil_thickness * 1000.0).cpu().numpy())
                outputs["snow_w"].append(state.snow_w.cpu().numpy())
                outputs["snow_t"].append(state.snow_t.cpu().numpy())
                outputs["snow_rho"].append(state.snow_rho.cpu().numpy())
                outputs["snow_alb"].append(state.snow_alb.cpu().numpy())
    summary = {"nvalid": nvalid, "nsteps": nsteps, "device": device, "nan": bool(not torch.isfinite(state.skin_t).all())}
    np.savez(CASE / "torch_offline_daily.npz", **{key: np.stack(value) for key, value in outputs.items()})
    if trace_internal:
        internal.update({f"vdf_input_{name}": value for name, value in captured_diffusion.items()})
        np.savez(CASE / "torch_offline_internal_step1.npz", **internal)
    if trace_index is not None:
        np.savez(CASE / f"torch_offline_trace_{trace_index}.npz", **{
            name: np.asarray(values) for name, values in trace.items()
        })
    (CASE / "torch_offline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def _runtime_for_step(
    arrays: dict[str, np.ndarray], next_arrays: dict[str, np.ndarray], step: int,
    dt: float, lat: np.ndarray, lon: np.ndarray, device: str,
) -> SimpleNamespace:
    """OSM-compatible forcing window, with the final record linked to next year."""
    start = step * dt
    nrecord = next(iter(arrays.values())).shape[0]
    record = int(start // FORCING_DT)
    alpha = (start - record * FORCING_DT) / FORCING_DT
    next_record = record + 1
    held_record = record + 1
    held_next_year = held_record == nrecord
    if next_record == nrecord:
        next_record = 0

    def field(values: np.ndarray, name: str, index: int, use_next: bool = False) -> np.ndarray:
        source = next_arrays[name] if use_next else values
        return np.asarray(source[index], dtype=np.float32).astype(np.float64)

    held_fluxes = {"swdown_W_m2", "lwdown_W_m2", "precip_kg_m2_s"}
    runtime = SimpleNamespace(dt=dt)
    for name, values in arrays.items():
        first = field(values, name, record)
        second = field(values, name, next_record, next_record == 0 and record == nrecord - 1)
        if name in held_fluxes:
            value = field(values, name, 0 if held_next_year else held_record, held_next_year)
            window = np.stack((value, value))
        else:
            window = np.stack((first, first + alpha * (second - first)))
        setattr(runtime, name, torch.as_tensor(window, dtype=DTYPE, device=device))
    phase_t = field(arrays["tair_K"], "tair_K", 0 if held_next_year else held_record, held_next_year)
    runtime.precip_snow_phase = torch.as_tensor(phase_t <= 273.15, dtype=DTYPE, device=device)
    flux_seconds = start + FORCING_DT
    mu0 = solar_cosine_at_time(lat, lon, flux_seconds)
    runtime.mu0 = torch.as_tensor(np.stack((mu0, mu0)), dtype=DTYPE, device=device)
    runtime.zero = torch.zeros(lat.size, dtype=DTYPE, device=device)
    runtime.co2 = torch.full((lat.size,), 6.0e-4, dtype=DTYPE, device=device)
    runtime.dt_tensor = torch.tensor(dt, dtype=DTYPE, device=device)
    runtime.sigma = torch.tensor(5.6705085385054389e-08, dtype=DTYPE, device=device)
    return runtime


def _save_state(path: Path, state, year: int) -> None:
    values = {"year": np.asarray(year, dtype=np.int32)}
    for field in fields(state):
        value = getattr(state, field.name)
        if isinstance(value, torch.Tensor):
            values[field.name] = value.detach().cpu().numpy()
    np.savez_compressed(path, **values)


def main() -> None:
    """Generate a forcing-consistent state from continuous CMFD years."""
    root = Path(os.environ.get("SURF_MULTIYEAR_DATA", "/data/yangjinhui/surf_pytorch/processed/multiyear_2009_2018"))
    code = Path(os.environ["SURF_CODE"])
    output = Path(os.environ["SURF_CASE"])
    initial = Path(os.environ.get("SURF_INITIAL", "/data/yangjinhui/surf_pytorch/surf_paper/experiments/initial_states/native_20d_terminal_complete.nc"))
    years = [int(value) for value in os.environ.get("SURF_YEARS", "2009:2018").replace(":", "-").split("-")]
    if len(years) != 2 or years[0] > years[1]:
        raise ValueError("SURF_YEARS must be START-END")
    years = list(range(years[0], years[1] + 1))
    dt = float(os.environ.get("SURF_PHYSICS_DT", "1800"))
    device = os.environ.get("SURF_DEVICE", "cuda")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the multiyear spin-up requires SURF_DEVICE=cuda with an available GPU")
    if FORCING_DT % dt:
        raise ValueError("SURF_PHYSICS_DT must divide 10800 s")
    output.mkdir(parents=True, exist_ok=True)
    first_data = root / f"cmfd_valid_columns_{years[0]}_dataset"
    metadata = json.loads((first_data / "metadata.json").read_text())
    nvalid = int(metadata["ncols"])
    lat = np.asarray(np.load(first_data / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(first_data / "lon.npy", mmap_mode="r"), dtype=np.float64)
    state, static = load_initial(initial, nvalid, device, lat, lon)
    initial_skin_t = state.radiative_skin_t.unsqueeze(-1).expand_as(state.skin_t).clone()
    initial_skin_t[:, 0] = state.sst
    state = replace(state, skin_t=initial_skin_t)
    config = SurfConfig(dtype=DTYPE, device=device, validate=False, use_farquhar=True,
                        use_ags=False, levgen=True, lelwtl=True, lelaiv=False,
                        vup_boundary_mode="offline")
    model = LandSurface(config).to(device)
    args = SimpleNamespace(shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False,
                           shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net",
                           stress_u=None, stress_v=None, vup=None)
    names = ("tair_K", "qair_kg_kg", "psurf_Pa", "swdown_W_m2", "lwdown_W_m2", "wind_m_s", "precip_kg_m2_s")
    annual = []
    with torch.inference_mode():
        for position, year in enumerate(years):
            data = root / f"cmfd_valid_columns_{year}_dataset"
            next_data = root / f"cmfd_valid_columns_{years[(position + 1) % len(years)]}_dataset"
            arrays = {name: np.load(data / f"{name}.npy", mmap_mode="r") for name in names}
            next_arrays = {name: np.load(next_data / f"{name}.npy", mmap_mode="r") for name in names}
            nsteps = int(next(iter(arrays.values())).shape[0] * FORCING_DT / dt)
            started = __import__("time").monotonic()
            for step in range(nsteps):
                fraction = model._tile_frac(static, config.n_tile, state=state)
                runtime = _runtime_for_step(arrays, next_arrays, step, dt, lat, lon, device)
                emission = model._surface_emissivity(state, static, fraction, DTYPE, device)
                forcing = make_forcing(runtime, 0, state, replace(static, emis=emission), args, tile_frac=fraction)
                amount = runtime.precip_kg_m2_s[0] * runtime.dt_tensor
                forcing = replace(forcing, rain=amount * (1.0 - runtime.precip_snow_phase), snow=amount * runtime.precip_snow_phase)
                state, _ = model(state, forcing, replace(static, emis=emission))
                # Avoid host synchronization for every 1800-s step. A daily
                # collective check still fails before the next forcing day and
                # keeps the state evolution entirely on the GPU otherwise.
                if (step + 1) % int(86_400 / dt) == 0 or step + 1 == nsteps:
                    state_fields = ("skin_t", "soil_t", "soil_w_liq", "snow_w", "snow_t", "snow_rho", "snow_alb")
                    finite = torch.isfinite(torch.cat([
                        getattr(state, name).reshape(-1) for name in state_fields
                    ])).all()
                    if not bool(finite.cpu()):
                        invalid = [name for name in state_fields if not bool(torch.isfinite(getattr(state, name)).all().cpu())]
                        raise FloatingPointError(f"non-finite {invalid} in {year}, step {step + 1}")
            torch.cuda.synchronize()
            elapsed = __import__("time").monotonic() - started
            diag = {
                "year": year, "steps": nsteps, "seconds": elapsed,
                "mean_swe_kg_m2": float(state.snow_w.mean().cpu()), "max_swe_kg_m2": float(state.snow_w.max().cpu()),
                "mean_snow_fraction": float(state.snow_frac.mean().cpu()), "max_snow_fraction": float(state.snow_frac.max().cpu()),
                "mean_soil_water_kg_m2": float((state.soil_w_liq * static.soil_thickness * 1000.0).sum(dim=1).mean().cpu()),
            }
            annual.append(diag)
            _save_state(output / f"spinup_{year}_terminal_state.npz", state, year)
            (output / "annual_diagnostics.json").write_text(json.dumps(annual, indent=2) + "\n")
            print(json.dumps(diag), flush=True)
    _save_state(output / f"spinup_{years[0]}_{years[-1]}_final_state.npz", state, years[-1])
    (output / "summary.json").write_text(json.dumps({"years": years, "initial": str(initial), "device": device, "physics_dt_s": dt, "annual": annual}, indent=2) + "\n")


if __name__ == "__main__":
    main()
