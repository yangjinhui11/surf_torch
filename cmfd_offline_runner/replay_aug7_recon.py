"""Stream the full-CMFD Fortran-boundary replay through PyTorch SURF."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import fields, replace
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from netCDF4 import Dataset


CASE = Path(os.environ.get(
    "SURF_CASE", "/data/yangjinhui/surf_pytorch/full_domain_30d_controlled_97709",
))
DATA = Path("/data/yangjinhui/surf_pytorch/processed/cmfd_valid_columns_2018_dataset")
CODE = Path(os.environ.get("SURF_CODE", "/home/qixiang/yangjinhui/openifs/surf_pytorch"))
sys.path[:0] = [str(CODE), "/data/yangjinhui/surf_pytorch"]

from diagnostic_scripts_compare_realcol import make_forcing  # noqa: E402
from surf_pytorch import LandSurface, SurfConfig  # noqa: E402
from surf_pytorch.state import Forcing, Static, SurfState  # noqa: E402
from surf_pytorch.tables import VegTable  # noqa: E402


DTYPE = torch.float64
BLOCK = 40
NTILE_FORTRAN = 9
NTILE_TORCH = 8
VUP_NAMES = (
    "vupdz0_pustrti", "vupdz0_pvstrti", "vupdz0_pahfsti",
    "vupdz0_pevapti", "vupdz0_z0mti", "vupdz0_z0hti",
    "vupdz0_z0qti", "vupdz0_pbuomti", "vupdz0_pzdlti",
    "vupdz0_praqti", "pfrti",
)
VDF_NAMES = ("pfrti", "pahfsti", "pevapti", "pslrflti", "ptsktip1")
RAD_NAMES = ("srfd", "slrfl", "ptl")


def scalar_field(ds: Dataset, name: str, nvalid: int) -> np.ndarray:
    return np.asarray(ds.variables[name][:], dtype=np.float64).reshape(-1)[:nvalid]


def load_initial(path: Path, nvalid: int, device: str, lat: np.ndarray, lon: np.ndarray) -> tuple[SurfState, Static]:
    with Dataset(path) as ds:
        soil_t = np.asarray(ds["SoilTemp"][:, 0, :nvalid], dtype=np.float64).T
        soil_w = np.asarray(ds["SoilMoist"][:, 0, :nvalid], dtype=np.float64).T
        skin_t = np.asarray(ds["SkinT"][:NTILE_TORCH, 0, :nvalid], dtype=np.float64).T
        snow_w = scalar_field(ds, "SWE", nvalid)
        snow_rho = scalar_field(ds, "snowdens", nvalid)
        snow_t = scalar_field(ds, "SnowT", nvalid)
        cvl, cvh = scalar_field(ds, "cvl", nvalid), scalar_field(ds, "cvh", nvalid)
        tvl = np.rint(scalar_field(ds, "tvl", nvalid)).astype(np.int64)
        tvh = np.rint(scalar_field(ds, "tvh", nvalid)).astype(np.int64)
        veg = VegTable(dtype=DTYPE, device=device)
        low = cvl * veg.gather("rvcov", torch.as_tensor(tvl, device=device)).cpu().numpy()
        high = cvh * veg.gather("rvcov", torch.as_tensor(tvh, device=device)).cpu().numpy()
        fraction = np.zeros((nvalid, NTILE_TORCH), dtype=np.float64)
        fraction[:, 3], fraction[:, 5] = low, high
        fraction[:, 7] = 1.0 - low - high
        sst = scalar_field(ds, "sst", nvalid)
        skin_t[:, 0] = sst
        thickness = np.asarray([0.07, 0.21, 0.72, 1.89], dtype=np.float64)
        state = SurfState(
            skin_t=torch.as_tensor(skin_t, dtype=DTYPE, device=device),
            soil_t=torch.as_tensor(soil_t, dtype=DTYPE, device=device),
            soil_w_liq=torch.as_tensor(soil_w / (1000.0 * thickness), dtype=DTYPE, device=device),
            snow_w=torch.as_tensor(snow_w, dtype=DTYPE, device=device),
            snow_frac=torch.as_tensor(np.clip(10.0 * snow_w / np.maximum(snow_rho, np.finfo(float).tiny), 0.0, 1.0), dtype=DTYPE, device=device),
            skin_water=torch.as_tensor(scalar_field(ds, "CanopInt", nvalid), dtype=DTYPE, device=device),
            sst=torch.as_tensor(sst, dtype=DTYPE, device=device),
            snow_rho=torch.as_tensor(snow_rho, dtype=DTYPE, device=device),
            snow_t=torch.as_tensor(snow_t, dtype=DTYPE, device=device),
            snow_alb=torch.as_tensor(scalar_field(ds, "SAlbedo", nvalid), dtype=DTYPE, device=device),
            radiative_skin_t=torch.as_tensor(scalar_field(ds, "AvgSurfT", nvalid), dtype=DTYPE, device=device),
        )
        static = Static(
            tile_frac_base=torch.as_tensor(fraction, dtype=DTYPE, device=device),
            land_frac=torch.as_tensor(scalar_field(ds, "landsea", nvalid), dtype=DTYPE, device=device),
            orography=torch.as_tensor(scalar_field(ds, "geopot", nvalid), dtype=DTYPE, device=device),
            soil_type=torch.as_tensor(np.rint(scalar_field(ds, "sotype", nvalid)).astype(np.int64), dtype=torch.long, device=device),
            veg_type=torch.as_tensor(tvl, dtype=torch.long, device=device),
            soil_thickness=torch.as_tensor(thickness, dtype=DTYPE, device=device),
            orography_stddev=torch.as_tensor(scalar_field(ds, "sdor", nvalid), dtype=DTYPE, device=device),
            emis=torch.full((nvalid,), 0.99, dtype=DTYPE, device=device),
            albd=torch.as_tensor(np.asarray(ds["Malbedo"][0, 0, :nvalid]), dtype=DTYPE, device=device),
            albw=torch.as_tensor(np.asarray(ds["Malbedo"][0, 0, :nvalid]), dtype=DTYPE, device=device),
            z0m=torch.as_tensor(scalar_field(ds, "z0m", nvalid), dtype=DTYPE, device=device),
            z0h=torch.as_tensor(scalar_field(ds, "lz0h", nvalid), dtype=DTYPE, device=device),
            z0q=torch.as_tensor(scalar_field(ds, "lz0h", nvalid), dtype=DTYPE, device=device),
            cvl_frac=torch.as_tensor(cvl, dtype=DTYPE, device=device),
            cvh_frac=torch.as_tensor(cvh, dtype=DTYPE, device=device),
            ice_land_frac=torch.as_tensor(scalar_field(ds, "glacierMask", nvalid), dtype=DTYPE, device=device),
            lai_low=torch.as_tensor(np.asarray(ds["Mlail"][0, 0, :nvalid]), dtype=DTYPE, device=device),
            lai_high=torch.as_tensor(np.asarray(ds["Mlaih"][0, 0, :nvalid]), dtype=DTYPE, device=device),
            veg_type_low=torch.as_tensor(tvl, dtype=torch.long, device=device),
            veg_type_high=torch.as_tensor(tvh, dtype=torch.long, device=device),
            lat=torch.deg2rad(torch.as_tensor(lat, dtype=DTYPE, device=device)),
            lon=torch.deg2rad(torch.as_tensor(lon, dtype=DTYPE, device=device)),
        )
    return state, static


class CaptureStream:
    def __init__(
        self, path: Path, names: tuple[str, ...], nvalid: int, tiles: int,
        record_columns: int | None = None,
    ):
        self.path, self.names, self.nvalid, self.tiles = path, names, nvalid, tiles
        self.record_columns = record_columns if record_columns is not None else nvalid
        if self.record_columns < nvalid:
            raise ValueError("record_columns must include all valid columns")
        self.widths = [BLOCK] * (self.record_columns // BLOCK) + ([self.record_columns % BLOCK] if self.record_columns % BLOCK else [])
        # The diagnostic writer emits one native integer block id followed
        # by big-endian REAL8 fields.  It does not write the legacy
        # (step, first, last) 12-byte header assumed by the first reader.
        self.step_bytes = sum(4 + len(names) * width * tiles * 8 for width in self.widths)
        size = path.stat().st_size
        if size % self.step_bytes:
            raise ValueError(f"{path} has {size} bytes, not an integer number of variable-width steps ({self.step_bytes})")
        self.steps = size // self.step_bytes

    def __iter__(self):
        with self.path.open("rb") as handle:
            for _ in range(self.steps):
                fields = {name: np.empty((self.record_columns, NTILE_TORCH), dtype=np.float64) if self.tiles > 1 else np.empty(self.record_columns, dtype=np.float64) for name in self.names}
                offset = 0
                for width in self.widths:
                    header = handle.read(4)
                    if len(header) != 4:
                        raise EOFError(f"truncated record header in {self.path}")
                    for name in self.names:
                        count = width * self.tiles
                        data = np.frombuffer(handle.read(count * 8), dtype=">f8", count=count)
                        if data.size != count:
                            raise EOFError(f"truncated {name} record in {self.path}")
                        if self.tiles > 1:
                            fields[name][offset:offset + width] = data.reshape(self.tiles, width).T[:, :NTILE_TORCH]
                        else:
                            fields[name][offset:offset + width] = data
                    offset += width
                yield {name: values[:self.nvalid] for name, values in fields.items()}


def solar_cosine(lat: np.ndarray, lon: np.ndarray, step: int) -> np.ndarray:
    hours = step * 3.0
    day = 1 + int(hours // 24.0)
    delta = math.radians(23.45) * math.sin(2.0 * math.pi * (284 + day) / 365.0)
    omega = np.deg2rad(15.0 * ((hours % 24.0) + lon / 15.0 - 12.0))
    return np.clip(np.sin(np.deg2rad(lat)) * math.sin(delta) + np.cos(np.deg2rad(lat)) * math.cos(delta) * np.cos(omega), 0.0, 1.0)


class RuntimeWindow:
    def __init__(self, arrays: dict[str, np.ndarray], step: int, lat: np.ndarray, lon: np.ndarray, device: str):
        self.dt = 10800.0
        for name, values in arrays.items():
            # The OSM forcing NetCDF fields are written as f4.  Quantise
            # the processed source arrays to that same boundary precision
            # before promoting them to the double-precision Torch solve.
            setattr(self, name, torch.as_tensor(
                np.asarray(values[step:step + 2], dtype=np.float32),
                dtype=DTYPE, device=device,
            ))
        self.mu0 = torch.as_tensor(np.stack((solar_cosine(lat, lon, step), solar_cosine(lat, lon, step + 1))), dtype=DTYPE, device=device)
        self.zero = torch.zeros(lat.size, dtype=DTYPE, device=device)
        self.co2 = torch.full((lat.size,), 6.0e-4, dtype=DTYPE, device=device)
        self.dt_tensor = torch.tensor(self.dt, dtype=DTYPE, device=device)
        self.sigma = torch.tensor(5.6705085385054389e-08, dtype=DTYPE, device=device)


def main() -> None:
    metadata = json.loads((CASE / "metadata.json").read_text())
    output_dir = Path(os.environ.get("SURF_OUTPUT_DIR", str(CASE)))
    output_dir.mkdir(parents=True, exist_ok=True)
    captured_nvalid = int(metadata["nvalid"])
    subset_index = os.environ.get("SURF_SUBSET_INDEX")
    subset_index = int(subset_index) if subset_index is not None else None
    if subset_index is not None and not 0 <= subset_index < captured_nvalid:
        raise ValueError(f"SURF_SUBSET_INDEX={subset_index} is outside 0..{captured_nvalid - 1}")
    nvalid = 1 if subset_index is not None else captured_nvalid
    max_steps = int(os.environ.get("SURF_MAX_STEPS", "240"))
    stride = int(os.environ.get("SURF_OUTPUT_STRIDE", "8"))
    trace_index = os.environ.get("SURF_TRACE_INDEX")
    trace_index = int(trace_index) if trace_index is not None else None
    if subset_index is not None and trace_index not in (None, subset_index):
        raise ValueError("SURF_TRACE_INDEX must equal SURF_SUBSET_INDEX for a subset replay")
    trace_label = trace_index
    if subset_index is not None:
        trace_index = 0 if trace_index is not None else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lat = np.asarray(np.load(DATA / "lat.npy", mmap_mode="r"), dtype=np.float64)
    lon = np.asarray(np.load(DATA / "lon.npy", mmap_mode="r"), dtype=np.float64)
    arrays = {name: np.load(DATA / f"{name}.npy", mmap_mode="r") for name in ("tair_K", "qair_kg_kg", "psurf_Pa", "swdown_W_m2", "lwdown_W_m2", "wind_m_s", "precip_kg_m2_s")}
    state, static = load_initial(CASE / "restartin.nc", captured_nvalid, device, lat, lon)
    if subset_index is not None:
        selected = slice(subset_index, subset_index + 1)

        def subset_fields(value):
            updates = {
                field.name: getattr(value, field.name)[selected]
                for field in fields(value)
                if isinstance(getattr(value, field.name), torch.Tensor)
                and getattr(value, field.name).ndim > 0
                and getattr(value, field.name).shape[0] == captured_nvalid
            }
            return replace(value, **updates)

        state, static = subset_fields(state), subset_fields(static)
        lat, lon = lat[selected], lon[selected]
        arrays = {name: values[:, selected] for name, values in arrays.items()}
    # OSM's offline setup fixes LELAIV=.FALSE.; therefore its vegetation
    # table LAI, rather than the optional climatological Mlai field, is
    # part of the reference configuration.
    config = SurfConfig(dtype=DTYPE, device=device, validate=False, use_farquhar=True, use_ags=False, levgen=True, lelwtl=True, lelaiv=False, vup_boundary_mode="external")
    model = LandSurface(config).to(device)
    # The capture stream contains physical land columns only.  The larger
    # NetCDF restart dimension is an I/O padding convention, not part of
    # the VUP/VDF records.
    vups = CaptureStream(CASE / "vup_11fields.bin", VUP_NAMES, captured_nvalid, NTILE_FORTRAN)
    vdfs = CaptureStream(CASE / "vdf_5fields.bin", VDF_NAMES, captured_nvalid, NTILE_FORTRAN)
    rads = CaptureStream(CASE / "rad_3fields.bin", RAD_NAMES, captured_nvalid, 1)
    available = min(vups.steps, vdfs.steps, rads.steps)
    if available < max_steps:
        raise RuntimeError(f"only {available} captured steps available; requested {max_steps}")
    snapshots = {name: [] for name in ("step", "avg_skin_t", "soil_t", "soil_w", "snow_w", "snow_t", "snow_rho", "snow_alb")}
    trace: dict[str, list[float]] = {"step": []}

    def trace_value(value: torch.Tensor, tile: int | None = None) -> float:
        if value.ndim == 0:
            selected = value
        else:
            selected = value[trace_index] if tile is None else value[trace_index, tile]
        return float(selected.detach().cpu())

    def record_snow_trace(step: int) -> None:
        """Persist the exact tensors handed to and returned by the snow core."""
        inp, out = model._last_snow_inputs, model._last_snow_outputs
        if inp is None or out is None:
            raise RuntimeError("snow trace requested before the snow solver ran")
        trace["step"].append(float(step))
        scalar_inputs = (
            "pssnm1m", "ptsnm1m", "prsnm1m", "ptsam1m_top", "pslrfl",
            "pssfc", "pssfl", "ptsfc", "ptsfl", "pevapsnw", "ptmst",
        )
        scalar_outputs = (
            "ptsn", "pgsn", "psstar", "plwc", "zfrsn", "pmsn",
            "pmsnint", "pemssn", "ptsfcin", "ptsflin",
        )
        tile_inputs = ("pfrti", "pahfsti", "pevapti", "pssrflti", "pslrflti")
        for name in scalar_inputs:
            trace.setdefault(name, []).append(trace_value(getattr(inp, name)))
        for name in scalar_outputs:
            value = getattr(out, name)
            trace.setdefault(name, []).append(np.nan if value is None else trace_value(value))
        trace.setdefault("llnosnow", []).append(float(trace_value(out.llnosnow.to(DTYPE))))
        for name in tile_inputs:
            value = getattr(inp, name)
            for tile, label in ((4, "5"), (6, "7")):
                trace.setdefault(f"{name}{label}", []).append(trace_value(value, tile))
        trace.setdefault("state_snow_w", []).append(trace_value(state.snow_w))
        trace.setdefault("state_snow_t", []).append(trace_value(state.snow_t))
        trace.setdefault("state_snow_rho", []).append(trace_value(state.snow_rho))
        trace.setdefault("vevap_pcsnw7", []).append(trace_value(model._last_vevap_pcsnw, 6))
        trace.setdefault("vevap_pcfq7", []).append(trace_value(model._last_vevap_pcfq, 6))
        trace.setdefault("jarvis_pweths", []).append(trace_value(model._last_jarvis["pweths"]))
        trace.setdefault("jarvis_laih", []).append(trace_value(model._last_jarvis["laih"]))
        trace.setdefault("jarvis_f1h", []).append(trace_value(model._last_jarvis["f1h"]))
        trace.setdefault("jarvis_f2h", []).append(trace_value(model._last_jarvis["f2h"]))
        trace.setdefault("jarvis_f3h", []).append(trace_value(model._last_jarvis["f3h"]))
        for name, value in (
            ("vexcs_pcfq7", model._last_pcfq_tile),
            ("vexcs_pqsam7", model._last_vexcs_pqsam_tile),
            ("vexcs_pcpts7", model._last_vexcs_pcpts_tile),
            ("vexcs_z0m7", model._last_vexcs_z0m_tile),
            ("vexcs_z0h7", model._last_vexcs_z0h_tile),
            ("vexcs_z0q7", model._last_vexcs_z0q_tile),
            ("vexcs_pzdl7", model._last_vexcs_pzdl_input_tile),
            ("vexcs_pbuom7", model._last_vexcs_pbuom_input_tile),
        ):
            trace.setdefault(name, []).append(trace_value(value, 6))
        trace.setdefault("forcing_q", []).append(trace_value(forcing.q))
        trace.setdefault("forcing_p", []).append(trace_value(forcing.p_surf))
        for level in range(state.soil_t.shape[1]):
            trace.setdefault(f"state_soil_t_l{level + 1}", []).append(
                trace_value(state.soil_t[:, level])
            )
            trace.setdefault(f"state_soil_w_l{level + 1}", []).append(
                trace_value(state.soil_w_liq[:, level])
            )
    args = SimpleNamespace(shift_flux=True, shift_state=False, shift_sw=False, shift_lw=False, shift_precip=False, use_tendencies=True, mu0_index="flux", lw_mode="net", stress_u=None, stress_v=None, vup=None)
    with torch.inference_mode():
        for step, (vup, vdf, radiation) in enumerate(islice(zip(vups, vdfs, rads), max_steps), start=1):
            if subset_index is not None:
                vup = {name: values[selected] for name, values in vup.items()}
                vdf = {name: values[selected] for name, values in vdf.items()}
                radiation = {name: values[selected] for name, values in radiation.items()}
            fraction = torch.as_tensor(vdf["pfrti"], dtype=DTYPE, device=device)
            runtime = RuntimeWindow(arrays, step - 1, lat, lon, device)
            args.vup = {name: value[None] for name, value in vup.items() if name in VUP_NAMES[:-1]}
            emission = model._surface_emissivity(state, static, fraction, DTYPE, device)
            forcing: Forcing = make_forcing(runtime, 0, state, replace(static, emis=emission), args, tile_frac=fraction)
            for name in VUP_NAMES[:-1]:
                # make_forcing has already selected the VUP time axis.
                # Do not squeeze a one-column grid axis.
                setattr(forcing, name, getattr(forcing, name))
            forcing.fr_so = torch.as_tensor(radiation["srfd"], dtype=DTYPE, device=device) * forcing.dt
            forcing.fr_th = torch.as_tensor(radiation["slrfl"], dtype=DTYPE, device=device) * forcing.dt
            forcing.radiative_skin_t = torch.as_tensor(radiation["ptl"], dtype=DTYPE, device=device)
            forcing.vdf_pfrti = fraction
            forcing.vdf_pahfsti = torch.as_tensor(vdf["pahfsti"], dtype=DTYPE, device=device)
            forcing.vdf_pevapti = torch.as_tensor(vdf["pevapti"], dtype=DTYPE, device=device)
            forcing.vdf_pslrflti = torch.as_tensor(vdf["pslrflti"], dtype=DTYPE, device=device)
            forcing.vdf_ptsktip1 = torch.as_tensor(vdf["ptsktip1"], dtype=DTYPE, device=device)
            state, _ = model(state, forcing, static)
            if trace_index is not None:
                record_snow_trace(step)
            if step % stride == 0:
                snapshots["step"].append(np.array(step, dtype=np.int64))
                snapshots["avg_skin_t"].append((state.skin_t * fraction).sum(dim=1).cpu().numpy())
                snapshots["soil_t"].append(state.soil_t.cpu().numpy())
                snapshots["soil_w"].append((state.soil_w_liq * static.soil_thickness * 1000.0).cpu().numpy())
                snapshots["snow_w"].append(state.snow_w.cpu().numpy())
                snapshots["snow_t"].append(state.snow_t.cpu().numpy())
                snapshots["snow_rho"].append(state.snow_rho.cpu().numpy())
                snapshots["snow_alb"].append(state.snow_alb.cpu().numpy())
    np.savez(output_dir / "torch_daily_controlled.npz", **{name: np.stack(value) for name, value in snapshots.items()})
    if trace_index is not None:
        np.savez(output_dir / f"torch_snow_trace_{trace_label}.npz", **{name: np.asarray(value) for name, value in trace.items()})
    summary = {"nvalid": nvalid, "captured_steps": int(available), "replayed_steps": max_steps, "snapshots": len(snapshots["step"]), "nan": bool(not torch.isfinite(state.skin_t).all())}
    (output_dir / "torch_daily_controlled_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
