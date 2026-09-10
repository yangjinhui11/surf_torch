#!/usr/bin/env python3
"""Run the autonomous Torch offline SURF closure for a CMFD full-domain case."""

from __future__ import annotations

import json
import math
import os
import sys
import time
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


def solar_cosine_at_time_torch(lat: torch.Tensor, lon: torch.Tensor, seconds: float) -> torch.Tensor:
    """Same solar geometry as solar_cosine_at_time but on torch tensors, so a
    CUDA-graph step performs no host->device copy.  lat/lon in degrees."""
    hours = seconds / 3600.0
    day = 1 + int(hours // 24.0)
    delta = math.radians(23.45) * math.sin(2.0 * math.pi * (284 + day) / 365.0)
    latr = torch.deg2rad(lat)
    omega = torch.deg2rad(15.0 * ((hours % 24.0) + lon / 15.0 - 12.0))
    return torch.clamp(
        torch.sin(latr) * math.sin(delta)
        + torch.cos(latr) * math.cos(delta) * torch.cos(omega),
        0.0,
        1.0,
    )


def interpolated_runtime(
    arrays: dict[str, np.ndarray | torch.Tensor], physical_step: int, physics_dt: float,
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

    def as_real8(values: np.ndarray | torch.Tensor, index: int) -> np.ndarray | torch.Tensor:
        """Promote one OSM float32 forcing record without changing its value."""
        if isinstance(values, torch.Tensor):
            return values[index].to(dtype=DTYPE)
        return np.asarray(values[index], dtype=np.float32).astype(np.float64)

    def sample(values: np.ndarray | torch.Tensor, seconds: float) -> np.ndarray | torch.Tensor:
        # The two terminal OSM forcing records repeat the first records of
        # the forcing year.  The Torch driver must use that same cycle at
        # an annual boundary rather than hold the final record.
        # Keep the index and interpolation phase in the same forcing cycle.
        cycle_seconds = seconds % (nrecord * FORCING_DT)
        record = int(cycle_seconds // FORCING_DT)
        next_record = (record + 1) % nrecord
        alpha = (cycle_seconds - record * FORCING_DT) / FORCING_DT
        # OSM reads f4 forcing values then performs the interpolation in REAL8.
        first = as_real8(values, record)
        second = as_real8(values, next_record)
        # DTFORC (dtforc.F90) interpolates as ZW*first + ZWP1*second with
        # ZW = 1-frac and ZWP1 = 1-ZW; the lerp form differs in the last ulp.
        zw = 1.0 - alpha
        zwp1 = 1.0 - zw
        return zw * first + zwp1 * second

    # OSM linearly interpolates state meteorology, but its offline alignment
    # holds radiation and precipitation at the next three-hour record for the
    # entire interval. This is also the legacy 10800 s ``shift_flux`` path.
    held_fluxes = {"swdown_W_m2", "lwdown_W_m2", "precip_kg_m2_s"}
    flux_record = (int(start // FORCING_DT) + 1) % nrecord
    runtime = SimpleNamespace()
    runtime.dt = physics_dt
    for name, values in arrays.items():
        if name in held_fluxes:
            flux = as_real8(values, flux_record)
            window = torch.stack((flux, flux)) if isinstance(flux, torch.Tensor) else np.stack((flux, flux))
        else:
            first, second = sample(values, start), sample(values, start + physics_dt)
            window = torch.stack((first, second)) if isinstance(first, torch.Tensor) else np.stack((first, second))
        setattr(runtime, name, window if isinstance(window, torch.Tensor) else torch.as_tensor(window, dtype=DTYPE, device=device))
    # CMFD supplies total precipitation. OSM's forcing preparation splits
    # it into Rainf/Snowf once per 3-hour record, using that record's air
    # temperature; it does not reclassify the held precipitation on each
    # 1800 s physics substep.
    phase = as_real8(arrays["tair_K"], flux_record) <= 273.15
    runtime.precip_snow_phase = phase.to(dtype=DTYPE) if isinstance(phase, torch.Tensor) else torch.as_tensor(
        phase, dtype=DTYPE, device=device,
    )
    flux_time = (int(start // FORCING_DT) + 1) * FORCING_DT
    if isinstance(lat, torch.Tensor):
        mu0_flux = solar_cosine_at_time_torch(lat, lon, flux_time)
        runtime.mu0 = torch.stack((mu0_flux, mu0_flux)).to(dtype=DTYPE, device=device)
    else:
        mu0_flux = solar_cosine_at_time(lat, lon, flux_time)
        runtime.mu0 = torch.as_tensor(
            np.stack((mu0_flux, mu0_flux)), dtype=DTYPE, device=device,
        )
    nlat = lat.numel() if isinstance(lat, torch.Tensor) else lat.size
    runtime.zero = torch.zeros(nlat, dtype=DTYPE, device=device)
    runtime.co2 = torch.full((nlat,), 6.0e-4, dtype=DTYPE, device=device)
    runtime.dt_tensor = torch.tensor(physics_dt, dtype=DTYPE, device=device)
    # SUCST RSIGMA: gfortran expands integer powers as multiplication
    # trees, which lands one ulp away from the libm pow value.  Reproduce
    # the gfortran grouping so the net longwave flux matches the F90 driver.
    rpi = 2.0 * math.asin(1.0)
    rpi5 = (rpi * rpi) * (rpi * rpi) * rpi
    # driver sucst (callpar1s YOMCST) RSIGMA: naive K**4/H**3 grouping.  The
    # net-LW forcing PFRTH = ZEMIS*(LWdown - RSIGMA*PTL**4) uses THIS value,
    # 5 ulp above the module RSIGMA that SURFSEB uses internally.
    rkbol4 = (1.380658e-23 * 1.380658e-23) * (1.380658e-23 * 1.380658e-23)
    den = 15.0 * (299792458.0 * 299792458.0) * (
        (6.6260755e-34 * 6.6260755e-34) * 6.6260755e-34
    )
    runtime.sigma = torch.tensor(2.0 * rpi5 * rkbol4 / den, dtype=DTYPE, device=device)
    return runtime


def main() -> None:
    started = time.perf_counter()
    metadata = json.loads((CASE / "metadata.json").read_text())
    captured_nvalid = int(metadata["nvalid"])
    subset_index_env = os.environ.get("SURF_SUBSET_INDEX")
    subset_index = int(subset_index_env) if subset_index_env is not None else None
    column_count_env = os.environ.get("SURF_COLUMN_COUNT")
    column_count = int(column_count_env) if column_count_env is not None else None
    if subset_index is not None and column_count is not None:
        raise ValueError("SURF_SUBSET_INDEX and SURF_COLUMN_COUNT cannot be combined")
    if subset_index is not None and not 0 <= subset_index < captured_nvalid:
        raise ValueError(
            f"SURF_SUBSET_INDEX={subset_index} is outside 0..{captured_nvalid - 1}"
        )
    if column_count is not None and not 1 <= column_count <= captured_nvalid:
        raise ValueError(
            f"SURF_COLUMN_COUNT={column_count} is outside 1..{captured_nvalid}"
        )
    selected = (
        slice(subset_index, subset_index + 1) if subset_index is not None
        else slice(0, column_count) if column_count is not None
        else None
    )
    nvalid = 1 if subset_index is not None else (column_count or captured_nvalid)
    nsteps = int(os.environ.get("SURF_MAX_STEPS", "240"))
    physics_dt = float(os.environ.get("SURF_PHYSICS_DT", str(FORCING_DT)))
    stride = int(os.environ.get("SURF_OUTPUT_STRIDE", "8"))
    write_output = os.environ.get("SURF_WRITE_OUTPUT", "1") != "0"
    fail_on_nonfinite = os.environ.get("SURF_FAIL_ON_NONFINITE") == "1"
    trace_internal = os.environ.get("SURF_TRACE_INTERNAL") == "1"
    trace_step = int(os.environ.get("SURF_TRACE_STEP", "0"))
    trace_index_env = os.environ.get("SURF_TRACE_INDEX")
    trace_index = int(trace_index_env) if trace_index_env is not None else None
    trace_start = int(os.environ.get("SURF_TRACE_START", "0"))
    trace_stop = int(os.environ.get("SURF_TRACE_STOP", str(nsteps)))
    modis_targets_path = os.environ.get("SURF_MODIS_TARGETS")
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
    cpu_threads_env = os.environ.get("SURF_CPU_THREADS")
    if device == "cpu" and cpu_threads_env is not None:
        cpu_threads = int(cpu_threads_env)
        if cpu_threads < 1:
            raise ValueError("SURF_CPU_THREADS must be positive")
        torch.set_num_threads(cpu_threads)
    else:
        cpu_threads = torch.get_num_threads() if device == "cpu" else None
    if write_output and stride < 1:
        raise ValueError("SURF_OUTPUT_STRIDE must be positive when output is enabled")
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
    if selected is not None:
        lat, lon, solar_lat = lat[selected], lon[selected], solar_lat[selected]
        arrays = {name: values[:, selected] for name, values in arrays.items()}
    if os.environ.get("SURF_CUDA_GRAPH") == "1":
        os.environ["SURF_PRELOAD_FORCING"] = "1"
        lat = torch.as_tensor(np.asarray(lat, dtype=np.float64), dtype=DTYPE, device=device)
        lon = torch.as_tensor(np.asarray(lon, dtype=np.float64), dtype=DTYPE, device=device)
        solar_lat = lat
    if os.environ.get("SURF_PRELOAD_FORCING") == "1":
        if device != "cuda":
            raise ValueError("SURF_PRELOAD_FORCING=1 requires SURF_DEVICE=cuda")
        # Retain the OSM forcing-file precision in device memory; records are
        # promoted to REAL8 only when the interpolation is evaluated.
        arrays = {
            name: torch.as_tensor(np.array(values, dtype=np.float32, copy=True), device=device)
            for name, values in arrays.items()
        }
    state, static = load_initial(
        CASE / "restartin.nc", captured_nvalid, device, lat, lon,
    )
    if selected is not None:
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
        trace_index = 0 if trace_index is not None else None
    # SURFEXCDRIVER_CTL resets tiled PTSKTI from PTSKM1M at KSTEP=0 and
    # then restores the water tile from PSST.  Restart SkinT is therefore
    # not the first-call land-tile temperature in OSM offline mode.
    if int(os.environ.get("SURF_RESTART_STEP", "0")) == 0:
        initial_skin_t = state.radiative_skin_t.unsqueeze(-1).expand_as(state.skin_t).clone()
        initial_skin_t[:, 0] = state.sst
        state = replace(state, skin_t=initial_skin_t)
    # else: mid-run restart (Fortran NSTART > 0) keeps the restart per-tile
    # SkinT; the KSTEP=0 broadcast only happens at a genuine cold start.
    config = SurfConfig(
        dtype=DTYPE, device=device, validate=False, use_farquhar=True,
        use_ags=False, levgen=True, lelwtl=True, lelaiv=False, vup_boundary_mode="offline",
    )
    model = LandSurface(config).to(device)
    modis_output = None
    modis_target_steps = None
    modis_time = None
    modis_time_units = None
    modis_calendar = None
    if modis_targets_path:
        steps_per_day = int(round(86_400.0 / physics_dt))
        if not math.isclose(steps_per_day * physics_dt, 86_400.0):
            raise ValueError("SURF_MODIS_TARGETS requires a physics timestep that divides one day")
        with Dataset(modis_targets_path) as observations:
            day_time = np.asarray(observations["lst_day_view_time_utc"][:], dtype=np.float64)
            night_time = np.asarray(observations["lst_night_view_time_utc"][:], dtype=np.float64)
            obs_lat = np.asarray(observations["lat"][:], dtype=np.float64)
            obs_lon = np.asarray(observations["lon"][:], dtype=np.float64)
            modis_time = np.asarray(observations["time"][:])
            modis_time_units = observations["time"].units
            modis_calendar = getattr(observations["time"], "calendar", "standard")
        if day_time.shape[1] != captured_nvalid or obs_lat.size != captured_nvalid:
            raise ValueError("MODIS target columns do not match the captured CMFD domain")
        if selected is not None:
            day_time, night_time = day_time[:, selected], night_time[:, selected]
            obs_lat, obs_lon = obs_lat[selected], obs_lon[selected]
        if not np.allclose(obs_lat, lat, atol=1.0e-5) or not np.allclose(obs_lon, lon, atol=1.0e-5):
            raise ValueError("MODIS and CMFD latitude/longitude columns are not aligned")
        n_obs_days = min(day_time.shape[0], math.ceil(nsteps / steps_per_day))
        day_time, night_time = day_time[:n_obs_days], night_time[:n_obs_days]
        modis_time = modis_time[:n_obs_days]
        day_offset = np.arange(n_obs_days, dtype=np.int64)[:, None] * steps_per_day

        def target_steps(hours_utc: np.ndarray) -> np.ndarray:
            finite = np.isfinite(hours_utc)
            within_day = np.ones(hours_utc.shape, dtype=np.int64)
            within_day[finite] = np.rint(hours_utc[finite] * 3600.0 / physics_dt).astype(np.int64)
            within_day = np.clip(within_day, 1, steps_per_day)
            result = day_offset + within_day
            result[~finite] = -1
            return result

        snow_time = np.mod(10.5 - lon[None, :] / 15.0, 24.0)
        snow_time = np.broadcast_to(snow_time, day_time.shape)
        modis_target_steps = {
            "lst_day": target_steps(day_time),
            "lst_night": target_steps(night_time),
            "snow_fraction": target_steps(snow_time),
        }
        modis_output = {
            name: np.full((n_obs_days, nvalid), np.nan, dtype=np.float32)
            for name in modis_target_steps
        }
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
    if device == "cuda":
        torch.cuda.synchronize()
    integration_started = time.perf_counter()
    outputs = {name: [] for name in ("step", "skin_t", "soil_t", "soil_w", "snow_w", "snow_t", "snow_rho", "snow_alb")}
    internal = {}
    trace: dict[str, list[float]] = {"step": []}
    avgskin_enabled = os.environ.get("SURF_AVGSKIN_3H") == "1"
    avgskin_interval = int(os.environ.get("SURF_AVGSKIN_INTERVAL", "6"))
    avgskin_acc = None
    avgskin_psk_acc = None
    avgskin_steps = []
    avgskin_values = []
    avgskin_psk_values = []
    avgskin_inst_values = []

    def save_output(completed_step: int) -> None:
        """Store one restart-state snapshot without changing the integration."""
        outputs["step"].append(np.array(completed_step, dtype=np.int64))
        outputs["skin_t"].append(state.skin_t.cpu().numpy())
        outputs["soil_t"].append(state.soil_t.cpu().numpy())
        outputs["soil_w"].append(
            (state.soil_w_liq * static.soil_thickness * 1000.0).cpu().numpy()
        )
        outputs["snow_w"].append(state.snow_w.cpu().numpy())
        outputs["snow_t"].append(state.snow_t.cpu().numpy())
        outputs["snow_rho"].append(state.snow_rho.cpu().numpy())
        outputs["snow_alb"].append(state.snow_alb.cpu().numpy())

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

    def record_trace(
        step: int,
        fraction_pre: torch.Tensor,
        fraction_post: torch.Tensor,
        forcing,
    ) -> None:
        """Save an explicitly time-aligned single-column trajectory.

        ``state`` and the diagnostic work arrays are the state after the
        completed call.  The process inputs were assembled using
        ``fraction_pre``; save both fractions so post-step state comparisons
        cannot accidentally be paired with the pre-step partition.
        """
        snow_in, snow_out = model._last_snow_inputs, model._last_snow_outputs
        if snow_in is None or snow_out is None:
            raise RuntimeError("snow trace requested before the snow solver ran")
        trace["step"].append(float(step + 1))
        trace.setdefault("time_start_seconds", []).append(float(step * physics_dt))
        trace.setdefault("time_end_seconds", []).append(float((step + 1) * physics_dt))
        for name in (
            "pssnm1m", "ptsnm1m", "prsnm1m", "pslrfl", "pssfc", "pssfl",
            "ptsfc", "ptsfl", "pevapsnw", "ptmst",
        ):
            trace.setdefault(f"snow_in_{name}", []).append(traced(getattr(snow_in, name)))
        for name in ("ptsn", "pgsn", "psstar", "plwc", "zfrsn", "pmsn", "pmsnint", "pemssn"):
            value = getattr(snow_out, name)
            trace.setdefault(f"snow_out_{name}", []).append(np.nan if value is None else traced(value))
        for name in (
            "snow_w", "snow_t", "snow_rho", "snow_alb", "snow_frac", "skin_water",
            "radiative_skin_t",
        ):
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
            trace.setdefault(f"fraction_pre_{label}", []).append(traced(fraction_pre, tile))
            trace.setdefault(f"fraction_post_{label}", []).append(traced(fraction_post, tile))
            for name in ("pfrti", "pahfsti", "pevapti", "pssrflti", "pslrflti"):
                trace.setdefault(f"snow_in_{name}{label}", []).append(traced(getattr(snow_in, name), tile))
        for tile in range(snow_in.pevapti.shape[1]):
            trace.setdefault(f"snow_in_pevapti_tile{tile + 1}", []).append(
                traced(snow_in.pevapti, tile)
            )
        for tile in range(fraction_pre.shape[1]):
            trace.setdefault(f"fraction_pre_tile{tile + 1}", []).append(
                traced(fraction_pre, tile)
            )
            trace.setdefault(f"fraction_post_tile{tile + 1}", []).append(
                traced(fraction_post, tile)
            )
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
        if model._last_vdf_ptskrad is not None:
            trace.setdefault("vdf_ptskrad", []).append(
                traced(model._last_vdf_ptskrad)
            )
        if model._last_vdf_ptsk_tile is not None:
            trace.setdefault("vdf_ptsk_gridmean", []).append(
                traced((model._last_vdf_ptsk_tile * fraction_pre).sum(dim=1))
            )
            trace.setdefault("vdf_ptsktip1_gridmean", []).append(
                traced((model._last_vdf_ptsktip1_tile * fraction_pre).sum(dim=1))
            )
            for tile in range(model._last_vdf_ptsk_tile.shape[1]):
                trace.setdefault(f"vdf_ptsk_tile{tile + 1}", []).append(
                    traced(model._last_vdf_ptsk_tile, tile)
                )
                trace.setdefault(f"vdf_ptsktip1_tile{tile + 1}", []).append(
                    traced(model._last_vdf_ptsktip1_tile, tile)
                )
        for name in ("_last_pcfh_tile", "_last_pcfq_tile"):
            value = getattr(model, name, None)
            if value is not None:
                for tile in range(value.shape[1]):
                    trace.setdefault(f"{name.removeprefix('_last_')}{tile + 1}", []).append(
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
    # Optional CUDA-graph fast path (opt-in via SURF_CUDA_GRAPH=1).  Captures
    # one whole step -- forcing build + model.forward -- into a single graph so
    # later steps launch once instead of issuing thousands of kernels.  Values
    # are bitwise-identical to eager; replay only removes launch overhead.
    use_graph = os.environ.get("SURF_CUDA_GRAPH") == "1" and device == "cuda"
    # VEXCS runs its high-iteration transient for KSTEP<=3 (KITT=3, LLINIT at
    # KSTEP=0) and KITT=1 afterwards, so capture only once the steady regime is
    # reached; earlier steps run eagerly.
    _graph_capture_step = 5
    _graph = None
    _graph_forcing = None
    _graph_state_in = None
    _graph_state_out = None
    _graph_static = None

    def _copy_tree(dst, src):
        for f in fields(dst):
            d = getattr(dst, f.name)
            sv = getattr(src, f.name)
            if isinstance(d, torch.Tensor) and isinstance(sv, torch.Tensor) \
                    and d.shape == sv.shape and d.dtype == sv.dtype:
                d.copy_(sv)

    # --- two-timescale surface-layer closure (SURF-Opt), env-gated ---
    slab_lambda = float(os.environ.get('SURF_SLAB_LAMBDA', '0') or 0)
    slab_tau_day = float(os.environ.get('SURF_SLAB_TAU_DAY_H', '6')) * 3600.0
    slab_tau_night = float(os.environ.get('SURF_SLAB_TAU_NIGHT_H', '3')) * 3600.0
    slab_sw_ref = 50.0
    slab_dt_air = None
    if slab_lambda > 1e-9:
        print(f'slab closure ON: lambda={slab_lambda} tau_day={slab_tau_day/3600.0}h '
              f'tau_night={slab_tau_night/3600.0}h', flush=True)
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
            if slab_dt_air is not None:
                forcing = replace(forcing, t=forcing.t + slab_dt_air)
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
            if use_graph:
                if _graph is None and step < _graph_capture_step:
                    state, _ = model(state, forcing, step_static)
                elif _graph is None and step == _graph_capture_step:
                    from dataclasses import replace as _dc_replace
                    _graph_state_in = _dc_replace(state, **{
                        f.name: (getattr(state, f.name).clone()
                                 if isinstance(getattr(state, f.name), torch.Tensor)
                                 else getattr(state, f.name))
                        for f in fields(state)})
                    gs = torch.cuda.Stream()
                    gs.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(gs):
                        _ = model(_graph_state_in, forcing, step_static)
                    torch.cuda.current_stream().wait_stream(gs)
                    _graph_forcing = forcing
                    _graph_static = step_static
                    _graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(_graph):
                        _graph_state_out, _ = model(
                            _graph_state_in, _graph_forcing, _graph_static)
                    state, _ = model(state, forcing, step_static)
                    _copy_tree(_graph_state_in, state)
                else:
                    _copy_tree(_graph_forcing, forcing)
                    # step_static.emis is recomputed from the current state
                    # every step (snow-dependent emissivity); refresh the
                    # captured buffer the same way as the forcing.
                    _copy_tree(_graph_static, step_static)
                    _graph.replay()
                    _copy_tree(state, _graph_state_out)
                    _copy_tree(_graph_state_in, _graph_state_out)
            else:
                state, _ = model(state, forcing, step_static)
            if slab_lambda > 1e-9:
                with torch.no_grad():
                    slab_heat = (fraction * model._last_pahfsti).sum(dim=1)
                    sw_now = runtime.swdown_W_m2[1].clamp(min=0.0)
                    weight = sw_now / (sw_now + slab_sw_ref)
                    beta = torch.exp(-physics_dt * (weight / slab_tau_day
                                                    + (1.0 - weight) / slab_tau_night))
                    previous = (slab_dt_air if slab_dt_air is not None
                                else torch.zeros_like(slab_heat))
                    slab_dt_air = beta * previous + (1.0 - beta) * (-slab_lambda * slab_heat)
            if avgskin_enabled:
                contribution = state.radiative_skin_t
                avgskin_acc = contribution if avgskin_acc is None else avgskin_acc + contribution
                if model._last_vdf_ptsk_tile is not None:
                    psk = (model._last_vdf_ptsk_tile * fraction).sum(dim=1)
                    avgskin_psk_acc = psk if avgskin_psk_acc is None else avgskin_psk_acc + psk
                if (step + 1) % avgskin_interval == 0:
                    avgskin_steps.append(step + 1)
                    avgskin_values.append((avgskin_acc / float(avgskin_interval)).detach().cpu().numpy())
                    avgskin_inst_values.append(contribution.detach().cpu().numpy())
                    avgskin_psk_values.append(
                        (avgskin_psk_acc / float(avgskin_interval)).detach().cpu().numpy()
                        if avgskin_psk_acc is not None else np.full(nvalid, np.nan))
                    avgskin_acc = None
                    avgskin_psk_acc = None
            if modis_output is not None and modis_target_steps is not None:
                completed_step = step + 1
                obs_day = min(step // steps_per_day, modis_output["lst_day"].shape[0] - 1)
                radiative_t = state.radiative_skin_t
                if radiative_t is None:
                    raise RuntimeError("MODIS observation operator requires radiative_skin_t")
                candidates = {
                    "lst_day": radiative_t,
                    "lst_night": radiative_t,
                    "snow_fraction": state.snow_frac,
                }
                for name, candidate in candidates.items():
                    mask = modis_target_steps[name][obs_day] == completed_step
                    if np.any(mask):
                        tensor_mask = torch.as_tensor(mask, device=device)
                        modis_output[name][obs_day, mask] = candidate[tensor_mask].detach().cpu().numpy().astype(np.float32)
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
                fraction_post = model._tile_frac(static, config.n_tile, state=state)
                record_trace(step, fraction, fraction_post, forcing)
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
            if write_output and (step + 1) % stride == 0:
                save_output(step + 1)
                elapsed = time.perf_counter() - integration_started
                print(
                    f"completed_step={step + 1}/{nsteps} elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )
    if device == "cuda":
        torch.cuda.synchronize()
    integration_seconds = time.perf_counter() - integration_started
    if write_output and (not outputs["step"] or int(outputs["step"][-1]) != nsteps):
        save_output(nsteps)
        print(
            f"completed_step={nsteps}/{nsteps} elapsed_seconds={integration_seconds:.1f} (terminal)",
            flush=True,
        )
    final_nonfinite_fields = []
    for name in ("skin_t", "soil_t", "soil_w_liq", "snow_w", "snow_t", "snow_rho", "snow_alb"):
        if not torch.isfinite(getattr(state, name)).all():
            final_nonfinite_fields.append(name)
    summary = {
        "nvalid": nvalid,
        "nsteps": nsteps,
        "device": device,
        "cpu_threads": cpu_threads,
        "nan": bool(final_nonfinite_fields),
        "nonfinite_fields": final_nonfinite_fields,
        "output_enabled": write_output,
        "saved_steps": [int(step) for step in outputs["step"]],
        "output_stride": stride,
        "setup_seconds": integration_started - started,
        "integration_seconds": integration_seconds,
        "modis_targets": modis_targets_path,
        "modis_observables": str(CASE / "surf_modis_observables.nc") if modis_output is not None else None,
    }
    if avgskin_enabled and avgskin_values:
        avgskin_path = CASE / "torch_avgskin_3h.npz"
        np.savez(avgskin_path, step=np.asarray(avgskin_steps, dtype=np.int64),
                 avg_skin_t=np.stack(avgskin_values).astype(np.float64),
                 avg_skin_t_ptsk=np.stack(avgskin_psk_values).astype(np.float64),
                 avg_skin_t_inst=np.stack(avgskin_inst_values).astype(np.float64))
        print(f"avgskin 3-hourly series saved: {avgskin_path} ({len(avgskin_steps)} records)", flush=True)
    if write_output:
        np.savez(CASE / "torch_offline_daily.npz", **{key: np.stack(value) for key, value in outputs.items()})
    if modis_output is not None:
        with Dataset(CASE / "surf_modis_observables.nc", "w", format="NETCDF4") as output:
            output.createDimension("time", next(iter(modis_output.values())).shape[0])
            output.createDimension("column", nvalid)
            output.source = "SURF observation operator sampled at MODIS target times"
            output.modis_targets = str(modis_targets_path)
            output.restart = str(CASE / "restartin.nc")
            output.physics_timestep_seconds = physics_dt
            output.time_matching = "nearest completed physics step to each MODIS UTC view time"
            output.lst_operator = "SURF native grid-mean radiative_skin_t (AvgSurfT)"
            output.snow_operator = "prognostic SURF snow fraction at 10:30 local solar time"
            time_variable = output.createVariable("time", modis_time.dtype, ("time",))
            time_variable.units = modis_time_units
            time_variable.calendar = modis_calendar
            time_variable[:] = modis_time
            output.createVariable("lat", "f4", ("column",))[:] = lat.astype(np.float32)
            output.createVariable("lon", "f4", ("column",))[:] = lon.astype(np.float32)
            chunks = (1, min(16384, nvalid))
            for name, values in modis_output.items():
                variable = output.createVariable(name, "f4", ("time", "column"), zlib=True, complevel=4, chunksizes=chunks, fill_value=np.nan)
                variable.units = "K" if name.startswith("lst_") else "1"
                variable[:] = values
    if trace_internal:
        internal.update({f"vdf_input_{name}": value for name, value in captured_diffusion.items()})
        np.savez(CASE / "torch_offline_internal_step1.npz", **internal)
    if trace_index is not None:
        np.savez(CASE / f"torch_offline_trace_{trace_index}.npz", **{
            name: np.asarray(values) for name, values in trace.items()
        })
    (CASE / "torch_offline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
