#!/usr/bin/env python3
"""Stepwise single-column comparison between OSM Fortran and PyTorch SURF."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset


FORCING_KEYS = [
    "tair_K",
    "qair_kg_kg",
    "psurf_Pa",
    "swdown_W_m2",
    "lwdown_W_m2",
    "wind_m_s",
    "precip_kg_m2_s",
]


def add_code_path(path: Path) -> None:
    sys.path.insert(0, str(path))


def mu0_table(lat_deg: np.ndarray, lon_deg: np.ndarray, ntime: int, dt: float) -> np.ndarray:
    lat_rad = np.deg2rad(lat_deg)
    out = np.empty((ntime, lon_deg.size), dtype="float64")
    for step in range(ntime):
        hours = step * dt / 3600.0
        doy = 1 + int(hours // 24.0)
        hour_utc = hours % 24.0
        delta = math.radians(23.45) * math.sin(2.0 * math.pi * (284 + doy) / 365.0)
        hour_local = hour_utc + lon_deg / 15.0
        omega = np.deg2rad(15.0 * (hour_local - 12.0))
        mu0 = np.sin(lat_rad) * math.sin(delta) + np.cos(lat_rad) * math.cos(delta) * np.cos(omega)
        out[step] = np.clip(mu0, 0.0, 1.0)
    return out


class Runtime:
    def __init__(self, z: np.lib.npyio.NpzFile, idx: np.ndarray, device: str, dtype: torch.dtype):
        self.dt = 10800.0
        for key in FORCING_KEYS:
            arr = np.asarray(z[key][:, idx], dtype="float64")
            setattr(self, key, torch.as_tensor(arr, dtype=dtype, device=device))
        lon = np.asarray(z["lon"][idx], dtype="float64")
        lat = np.asarray(z["lat"][idx], dtype="float64")
        self.mu0 = torch.as_tensor(mu0_table(lat, lon, z["tair_K"].shape[0], self.dt), dtype=dtype, device=device)
        n = idx.size
        self.zero = torch.zeros((n,), dtype=dtype, device=device)
        self.co2 = torch.full((n,), 6.0e-4, dtype=dtype, device=device)
        self.dt_tensor = torch.tensor(self.dt, dtype=dtype, device=device)
        # IFS/OSM RSIGMA, reconstructed from its historical physical
        # constants in sucst.F90.  This must match the offline driver rather
        # than the newer CODATA Stefan-Boltzmann value.
        self.sigma = torch.tensor(5.6705085385054389e-08, dtype=dtype, device=device)


def init_from_fortran_restart(restart: Path, device: str, dtype: torch.dtype):
    """Load a single-column state and static fields from OSM ``restartin.nc``."""
    from surf_pytorch.constants import RTT
    from surf_pytorch.state import Static, SurfState
    from surf_pytorch.tables import VegTable

    if not restart.is_file():
        raise FileNotFoundError(f"Fortran restart file not found: {restart}")

    def scalar(ds: Dataset, name: str) -> float:
        return float(np.asarray(ds.variables[name][:]).reshape(-1)[0])

    def profile(ds: Dataset, name: str, nlev: int) -> np.ndarray:
        values = np.asarray(ds.variables[name][:], dtype="float64").reshape(-1)
        if values.size < nlev:
            raise ValueError(f"{name} has {values.size} values, expected {nlev}")
        return values[:nlev]

    with Dataset(restart) as ds:
        nlev = len(ds.dimensions["nlevs"])
        thickness = np.array([0.07, 0.21, 0.72, 1.89], dtype="float64")
        if nlev != thickness.size:
            raise ValueError(f"Unsupported nlevs={nlev}; expected {thickness.size}")
        skin_t = profile(ds, "SkinT", 8)
        avg_skin_t = scalar(ds, "AvgSurfT")
        soil_t = profile(ds, "SoilTemp", nlev)
        soil_water = profile(ds, "SoilMoist", nlev)
        snow_w, snow_rho = scalar(ds, "SWE"), scalar(ds, "snowdens")
        snow_t, snow_alb = scalar(ds, "SnowT"), scalar(ds, "SAlbedo")
        skin_water = scalar(ds, "CanopInt")
        cvl, cvh = scalar(ds, "cvl"), scalar(ds, "cvh")
        tvl, tvh = int(round(scalar(ds, "tvl"))), int(round(scalar(ds, "tvh")))
        soil_type = int(round(scalar(ds, "sotype")))
        land_frac, glacier = scalar(ds, "landsea"), scalar(ds, "glacierMask")
        z0m, z0h = scalar(ds, "z0m"), scalar(ds, "lz0h")
        orography, sst, sdor = scalar(ds, "geopot"), scalar(ds, "sst"), scalar(ds, "sdor")
        albedo = float(np.asarray(ds.variables["Malbedo"][0]).reshape(-1)[0])
        lai_low = float(np.asarray(ds.variables["Mlail"][0]).reshape(-1)[0])
        lai_high = float(np.asarray(ds.variables["Mlaih"][0]).reshape(-1)[0])
        lat, lon = scalar(ds, "lat"), scalar(ds, "lon")

    veg = VegTable(dtype=dtype, device=device)
    low_cover = cvl * float(veg.gather("rvcov", torch.tensor([tvl], device=device))[0])
    high_cover = cvh * float(veg.gather("rvcov", torch.tensor([tvh], device=device))[0])
    frac = np.zeros((1, 8), dtype="float64")
    frac[0, 3], frac[0, 5] = low_cover, high_cover
    frac[0, 7] = 1.0 - low_cover - high_cover
    snow_frac = np.clip(10.0 * snow_w / max(snow_rho, np.finfo("float64").tiny), 0.0, 1.0)
    # At the first offline VDF call only the water tile is initialized from
    # PSST; all other tiles retain their tiled SkinT restart state.  SnowT
    # history is retained even when SWE is zero.
    skin_t[0] = sst
    thickness_t = torch.as_tensor(thickness, dtype=dtype, device=device)
    # VDFDIFH1S receives the prognostic tiled SkinT restart state.  AvgSurfT
    # is a grid diagnostic and must not overwrite the tile temperatures.
    state = SurfState(
        skin_t=torch.as_tensor(skin_t[None, :], dtype=dtype, device=device),
        soil_t=torch.as_tensor(soil_t[None, :], dtype=dtype, device=device),
        soil_w_liq=torch.as_tensor((soil_water / (1000.0 * thickness))[None, :], dtype=dtype, device=device),
        snow_w=torch.full((1,), snow_w, dtype=dtype, device=device),
        snow_frac=torch.full((1,), snow_frac, dtype=dtype, device=device),
        skin_water=torch.full((1,), skin_water, dtype=dtype, device=device),
        sst=torch.full((1,), sst, dtype=dtype, device=device),
        snow_rho=torch.full((1,), snow_rho, dtype=dtype, device=device),
        snow_t=torch.full((1,), snow_t, dtype=dtype, device=device),
        snow_alb=torch.full((1,), snow_alb, dtype=dtype, device=device),
        radiative_skin_t=torch.full((1,), avg_skin_t, dtype=dtype, device=device),
    )
    static = Static(
        tile_frac_base=torch.as_tensor(frac, dtype=dtype, device=device),
        land_frac=torch.full((1,), land_frac, dtype=dtype, device=device),
        orography=torch.full((1,), orography, dtype=dtype, device=device),
        soil_type=torch.full((1,), soil_type, dtype=torch.long, device=device),
        veg_type=torch.full((1,), tvl, dtype=torch.long, device=device),
        soil_thickness=thickness_t,
        orography_stddev=torch.full((1,), sdor, dtype=dtype, device=device),
        emis=torch.full((1,), 0.99, dtype=dtype, device=device),
        albd=torch.full((1,), albedo, dtype=dtype, device=device),
        albw=torch.full((1,), albedo, dtype=dtype, device=device),
        z0m=torch.full((1,), z0m, dtype=dtype, device=device),
        z0h=torch.full((1,), z0h, dtype=dtype, device=device),
        z0q=torch.full((1,), z0h, dtype=dtype, device=device),
        cvl_frac=torch.full((1,), cvl, dtype=dtype, device=device),
        cvh_frac=torch.full((1,), cvh, dtype=dtype, device=device),
        ice_land_frac=torch.full((1,), glacier, dtype=dtype, device=device),
        lai_low=torch.full((1,), lai_low, dtype=dtype, device=device),
        lai_high=torch.full((1,), lai_high, dtype=dtype, device=device),
        veg_type_low=torch.full((1,), tvl, dtype=torch.long, device=device),
        veg_type_high=torch.full((1,), tvh, dtype=torch.long, device=device),
        lat=torch.full((1,), np.deg2rad(lat), dtype=dtype, device=device),
        lon=torch.full((1,), np.deg2rad(lon), dtype=dtype, device=device),
    )
    return state, static


def make_forcing(
    rt: Runtime, step: int, state, static, args, *, tile_frac=None,
    radiative_skin_t=None,
):
    from surf_pytorch.state import Forcing

    state_i = min(step + 1, rt.tair_K.shape[0] - 1) if args.shift_state else step
    sw_i = min(step + 1, rt.tair_K.shape[0] - 1) if (args.shift_flux or args.shift_sw) else step
    lw_i = min(step + 1, rt.tair_K.shape[0] - 1) if (args.shift_flux or args.shift_lw) else step
    pr_i = min(step + 1, rt.tair_K.shape[0] - 1) if (args.shift_flux or args.shift_precip) else step
    mu_i = sw_i if args.mu0_index == "flux" else state_i
    t = rt.tair_K[state_i]
    pr = rt.precip_kg_m2_s[pr_i]
    frac = static.tile_frac_base if tile_frac is None else tile_frac
    # PTL/PTSKRAD is retained from the preceding radiation/surface step.
    # It must not be reconstructed using fractions that changed after that
    # step's snow or skin-water prognostics.
    if radiative_skin_t is not None:
        skin_mean = radiative_skin_t
    elif state.radiative_skin_t is not None:
        skin_mean = state.radiative_skin_t
    else:
        skin_mean = (state.skin_t * frac).sum(dim=1)
    if args.lw_mode == "net":
        # Offline SURF uses the emissivity-scaled net longwave flux as the
        # surface thermal forcing: PEMIS * (LWdown - sigma*Tskin^4).
        skin4 = (skin_mean * skin_mean) * (skin_mean * skin_mean)
        # Keep the W/m^2 value authoritative: callpar1s.F90:800 computes
        # PFRTH directly in W/m^2 and the (x*dt)/dt round-trip double-rounds.
        net_th = static.emis * (rt.lwdown_W_m2[lw_i] - rt.sigma * skin4)
        fr_th = net_th * rt.dt
    else:
        net_th = None
        fr_th = rt.lwdown_W_m2[lw_i] * rt.dt
    # Precipitation may use the next forcing record in offline alignment.
    # Its rain/snow split must use that same record's temperature.
    snow_phase = (rt.tair_K[pr_i] <= 273.15).to(t.dtype)
    t_next = rt.tair_K[min(step + 1, rt.tair_K.shape[0] - 1)]
    q_next = rt.qair_kg_kg[min(step + 1, rt.tair_K.shape[0] - 1)]
    u_next = rt.wind_m_s[min(step + 1, rt.wind_m_s.shape[0] - 1)]
    sl_tendency = None
    q_tendency = None
    u_tendency = None
    v_tendency = None
    if args.use_tendencies:
        # Match the RCPD used by the offline Fortran VDF tendency.
        from surf_pytorch.constants import RCPD

        cp = torch.as_tensor(RCPD, dtype=t.dtype, device=t.device)
        # vdfmain1s: ZTE=(PT1-PT)/PTSPHY then ZSLGE=RCPD*ZTE (RVTMP2=0).
        sl_tendency = cp * ((t_next - t) / rt.dt)
        q_tendency = (q_next - rt.qair_kg_kg[state_i]) / rt.dt
        u_tendency = (u_next - rt.wind_m_s[state_i]) / rt.dt
        v_tendency = torch.zeros_like(u_tendency)
    u_stress = None
    v_stress = None
    if getattr(args, "stress_u", None) is not None:
        si = min(step, len(args.stress_u) - 1)
        u_stress = torch.as_tensor(args.stress_u[si:si+1], dtype=t.dtype, device=t.device)
        v_stress = torch.as_tensor(args.stress_v[si:si+1], dtype=t.dtype, device=t.device)
    vup = {}
    if getattr(args, "vup", None) is not None:
        first = next(iter(args.vup.values()))
        # A VUP stream is (time, Ngrid, Ntile). Forcing consumes a single
        # two-dimensional time slice; retaining a length-one time axis makes
        # the authoritative coupled boundary fail the model shape contract.
        vi = min(step, first.shape[0] - 1)
        for name, values in args.vup.items():
            current = values[vi] if values.ndim == 3 else values
            if np.isfinite(current).all():
                vup[name] = torch.as_tensor(
                    current, dtype=t.dtype, device=t.device,
                )
    return Forcing(
        u=rt.wind_m_s[state_i],
        v=rt.zero,
        t=t,
        q=rt.qair_kg_kg[state_i],
        co2=rt.co2,
        p_surf=rt.psurf_Pa[state_i],
        geom1=torch.full_like(t, 9.80665 * 10.0),
        fr_so=rt.swdown_W_m2[sw_i] * rt.dt,
        fr_th=fr_th,
        slrfl_grid_wm2=net_th,
        srfd_grid_wm2=rt.swdown_W_m2[sw_i],
        mu0=rt.mu0[mu_i],
        rain=pr * (1.0 - snow_phase) * rt.dt,
        snow=pr * snow_phase * rt.dt,
        dt=rt.dt_tensor,
        sl_tendency=sl_tendency,
        q_tendency=q_tendency,
        u_tendency=u_tendency,
        v_tendency=v_tendency,
        radiative_skin_t=radiative_skin_t,
        u_stress=u_stress,
        v_stress=v_stress,
        **vup,
    )


def read_fortran(fortran_nc: Path, nsteps: int) -> dict[str, np.ndarray]:
    with Dataset(fortran_nc) as ds:
        n = min(nsteps, len(ds.dimensions["time"]))
        soil_w = np.asarray(ds.variables["SoilMoist"][:n, 0, 0, 0], dtype="float64")
        return {
            "avg_skin_t": np.asarray(ds.variables["AvgSurfT"][:n, 0, 0], dtype="float64"),
            "snow_w": np.asarray(ds.variables["SWE"][:n, 0, 0], dtype="float64"),
            "snow_t": np.asarray(ds.variables["SnowT"][:n, 0, 0], dtype="float64"),
            "snow_rho": np.asarray(ds.variables["snowdens"][:n, 0, 0], dtype="float64"),
            "snow_alb": np.asarray(ds.variables["SAlbedo"][:n, 0, 0], dtype="float64"),
            "skin_water": np.asarray(ds.variables["CanopInt"][:n, 0, 0], dtype="float64"),
            "soil_t1": np.asarray(ds.variables["SoilTemp"][:n, 0, 0, 0], dtype="float64"),
            "soil_w1": soil_w,
        }


def torch_row(step: int, state, static, diag=None, frac=None) -> dict[str, float]:
    if frac is None:
        frac = static.tile_frac_base
    row = {
        "step": step,
        "torch_avg_skin_t": float(((state.skin_t * frac).sum(dim=1))[0].detach().cpu()),
        "torch_snow_w": float(state.snow_w[0].detach().cpu()),
        "torch_snow_t": float(state.snow_t[0].detach().cpu()) if state.snow_t is not None else float("nan"),
        "torch_snow_rho": float(state.snow_rho[0].detach().cpu()) if state.snow_rho is not None else float("nan"),
        "torch_snow_alb": float(state.snow_alb[0].detach().cpu()) if state.snow_alb is not None else float("nan"),
        "torch_snow_frac": float(state.snow_frac[0].detach().cpu()) if state.snow_frac is not None else float("nan"),
        "torch_skin_water": float(state.skin_water[0].detach().cpu()),
        "torch_soil_t1": float(state.soil_t[0, 0].detach().cpu()),
        "torch_soil_w1": float((state.soil_w_liq[0, 0] * static.soil_thickness[0] * 1000.0).detach().cpu()),
    }
    for name in ["h_flux_tile", "e_flux_tile", "str_tile", "g0_tile", "lh_flux_tile"]:
        arr = getattr(diag, name, None) if diag is not None else None
        for tile, label in [(4, "snow_low"), (6, "snow_high")]:
            key = f"torch_{label}_{name}"
            row[key] = float(arr[0, tile].detach().cpu()) if arr is not None else float("nan")
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("/data/yangjinhui/surf_pytorch/processed/cmfd_tibet_columns_2018_20000.npz"))
    p.add_argument("--code-dir", type=Path, default=Path("/home/qixiang/yangjinhui/openifs/surf_pytorch"))
    p.add_argument("--fortran-nc", type=Path, default=Path("/data/yangjinhui/surf_pytorch/fortran_single_col_1240_stepout/o_gg.nc"))
    p.add_argument("--restart", type=Path, default=None,
                   help="OSM restartin.nc; defaults to the directory of --fortran-nc.")
    p.add_argument("--column-index", type=int, default=12405,
                   help="Source-column index for the bundled SINGLE1240 case.")
    p.add_argument("--nsteps", type=int, default=100)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--shift-flux", action="store_true")
    p.add_argument("--shift-state", action="store_true")
    p.add_argument("--shift-sw", action="store_true")
    p.add_argument("--shift-lw", action="store_true")
    p.add_argument("--shift-precip", action="store_true")
    p.add_argument("--use-tendencies", action="store_true")
    p.add_argument("--mu0-index", choices=["state", "flux"], default="state")
    p.add_argument("--lw-mode", choices=["net", "down"], default="net")
    p.add_argument("--use-farquhar", action="store_true")
    p.add_argument("--use-fortran-ptskrad", action="store_true",
                   help="Use Fortran AvgSurfT as the carried PTSKRAD input for strict SURF-core comparison.")
    p.add_argument("--fortran-offline-alignment", action="store_true",
                   help="Apply the OSM three-hour forcing staggering: next-record radiation/precipitation, flux-time solar angle, and atmospheric tendencies.")
    p.add_argument("--stress-csv", type=Path, default=None,
                   help="Optional CSV with fortran_pustr/fortran_pvstr columns for VUPDZ0 warm-start.")
    p.add_argument("--vup-csv", type=Path, default=None,
                   help="Authoritative VUPDZ0 tile-boundary CSV. Supplying it selects coupled mode.")
    args = p.parse_args()
    if args.fortran_offline_alignment:
        args.shift_flux = True
        args.mu0_index = "flux"
        args.lw_mode = "net"
        args.use_tendencies = True
    args.stress_u = None
    args.stress_v = None
    if args.stress_csv is not None:
        stress = np.genfromtxt(args.stress_csv, delimiter=",", names=True)
        args.stress_u = np.atleast_1d(stress["fortran_pustr"]).astype("float64")
        args.stress_v = np.atleast_1d(stress["fortran_pvstr"]).astype("float64")
    args.vup = None
    if args.vup_csv is not None:
        raw = np.genfromtxt(args.vup_csv, delimiter=",", names=True)
        raw = np.atleast_1d(raw)
        args.vup = {}
        fields = [
            "vupdz0_pustrti", "vupdz0_pvstrti", "vupdz0_pahfsti",
            "vupdz0_pevapti", "vupdz0_z0mti", "vupdz0_z0hti",
            "vupdz0_z0qti", "vupdz0_pbuomti", "vupdz0_pzdlti",
            "vupdz0_praqti",
        ]
        native_names = {name: name.removeprefix("vupdz0_") for name in fields}
        if "tile" in raw.dtype.names:
            required = ["step", "tile", *native_names.values()]
            missing = [name for name in required if name not in raw.dtype.names]
            if missing:
                raise ValueError(f"{args.vup_csv} lacks native VUPDZ0 columns: {', '.join(missing)}")
            steps = np.unique(raw["step"])
            for name, native_name in native_names.items():
                values = np.full((steps.size, 8), np.nan, dtype="float64")
                for i, step_value in enumerate(steps):
                    rows = raw[raw["step"] == step_value]
                    for row in rows:
                        tile = int(row["tile"]) - 1
                        if 0 <= tile < 8:
                            values[i, tile] = float(row[native_name])
                if not np.isfinite(values).all():
                    raise ValueError(f"{args.vup_csv} has incomplete native VUPDZ0 {native_name} tiles")
                args.vup[name] = values
        else:
            for name in fields:
                columns = [f"{name}_{tile}" for tile in range(8)]
                missing = [column for column in columns if column not in raw.dtype.names]
                if missing:
                    raise ValueError(
                        f"{args.vup_csv} lacks authoritative {name} tiles: {', '.join(missing)}"
                    )
                values = np.column_stack([np.atleast_1d(raw[column]) for column in columns]).astype("float64")
                if not np.isfinite(values).all():
                    raise ValueError(f"{args.vup_csv} contains non-finite values for {name}")
                args.vup[name] = values

    add_code_path(args.code_dir)
    from surf_pytorch import LandSurface, SurfConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    z = np.load(args.src)
    idx = np.array([args.column_index], dtype=int)
    vup_boundary_mode = "coupled" if args.vup is not None else "offline"
    model = LandSurface(SurfConfig(
        dtype=dtype,
        device=device,
        validate=False,
        use_farquhar=args.use_farquhar,
        # OSM's offline VDF recomputes long-wave fluxes for each tile.
        lelwtl=True,
        vup_boundary_mode=vup_boundary_mode,
    )).to(device=device)
    restart = args.restart if args.restart is not None else args.fortran_nc.parent / "restartin.nc"
    state, static = init_from_fortran_restart(restart, device, dtype)
    rt = Runtime(z, idx, device, dtype)

    ref = read_fortran(args.fortran_nc, args.nsteps)
    rows = []
    initial_frac = model._tile_frac(static, model.config.n_tile, state=state)
    radiative_skin_t = None
    rows.append(torch_row(0, state, static, None, initial_frac))
    with torch.inference_mode():
        for step in range(args.nsteps - 1):
            frac_emis = model._tile_frac(static, model.config.n_tile, state=state)
            emis_step = model._surface_emissivity(state, static, frac_emis, dtype, state.skin_t.device)
            static_step = replace(static, emis=emis_step)
            strict_ptskrad = None
            if args.use_fortran_ptskrad:
                strict_ptskrad = torch.full_like(
                    state.snow_w, float(ref['avg_skin_t'][step])
                )
            forcing = make_forcing(
                rt, step, state, static_step, args, tile_frac=frac_emis,
                radiative_skin_t=strict_ptskrad,
            )
            state, _diag = model(state, forcing, static_step)
            radiative_skin_t = None
            snow_rho = state.snow_rho if state.snow_rho is not None else torch.full_like(state.snow_w, 100.0)
            snow_frac = torch.clamp(10.0 * state.snow_w / torch.clamp(snow_rho, min=1.0e-12), 0.0, 1.0)
            snow_frac = torch.where(state.snow_w > 1.0e-12, snow_frac, torch.zeros_like(snow_frac))
            state = state.like(snow_frac=snow_frac)
            # AvgSurfT in the Fortran offline output is the SURFPP/VDF
            # diagnostic for the just-completed step, weighted with the tile
            # fractions used during that step.  Do not re-average the same
            # skin temperatures with post-snow-update fractions; that creates
            # artificial spikes when snow cover changes within the step.
            rows.append(torch_row(step + 1, state, static, _diag, frac_emis))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "fortran_avg_skin_t", "torch_avg_skin_t", "diff_avg_skin_t",
        "fortran_swe", "torch_snow_w", "diff_swe",
        "fortran_snow_t", "torch_snow_t", "diff_snow_t",
        "fortran_snow_rho", "torch_snow_rho", "diff_snow_rho",
        "fortran_snow_alb", "torch_snow_alb", "diff_snow_alb",
        "torch_snow_frac",
        "torch_snow_low_h_flux_tile", "torch_snow_low_e_flux_tile",
        "torch_snow_low_str_tile", "torch_snow_low_g0_tile", "torch_snow_low_lh_flux_tile",
        "torch_snow_high_h_flux_tile", "torch_snow_high_e_flux_tile",
        "torch_snow_high_str_tile", "torch_snow_high_g0_tile", "torch_snow_high_lh_flux_tile",
        "fortran_canopint", "torch_skin_water", "diff_canopint",
        "fortran_soil_t1", "torch_soil_t1", "diff_soil_t1",
        "fortran_soil_w1", "torch_soil_w1", "diff_soil_w1",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows[: len(ref["avg_skin_t"])]:
            i = row["step"]
            out = {
                "step": i,
                "fortran_avg_skin_t": ref["avg_skin_t"][i],
                "torch_avg_skin_t": row["torch_avg_skin_t"],
                "diff_avg_skin_t": row["torch_avg_skin_t"] - ref["avg_skin_t"][i],
                "fortran_swe": ref["snow_w"][i],
                "torch_snow_w": row["torch_snow_w"],
                "diff_swe": row["torch_snow_w"] - ref["snow_w"][i],
                "fortran_snow_t": ref["snow_t"][i],
                "torch_snow_t": row["torch_snow_t"],
                "diff_snow_t": row["torch_snow_t"] - ref["snow_t"][i],
                "fortran_snow_rho": ref["snow_rho"][i],
                "torch_snow_rho": row["torch_snow_rho"],
                "diff_snow_rho": row["torch_snow_rho"] - ref["snow_rho"][i],
                "fortran_snow_alb": ref["snow_alb"][i],
                "torch_snow_alb": row["torch_snow_alb"],
                "diff_snow_alb": row["torch_snow_alb"] - ref["snow_alb"][i],
                "torch_snow_frac": row["torch_snow_frac"],
                "torch_snow_low_h_flux_tile": row["torch_snow_low_h_flux_tile"],
                "torch_snow_low_e_flux_tile": row["torch_snow_low_e_flux_tile"],
                "torch_snow_low_str_tile": row["torch_snow_low_str_tile"],
                "torch_snow_low_g0_tile": row["torch_snow_low_g0_tile"],
                "torch_snow_low_lh_flux_tile": row["torch_snow_low_lh_flux_tile"],
                "torch_snow_high_h_flux_tile": row["torch_snow_high_h_flux_tile"],
                "torch_snow_high_e_flux_tile": row["torch_snow_high_e_flux_tile"],
                "torch_snow_high_str_tile": row["torch_snow_high_str_tile"],
                "torch_snow_high_g0_tile": row["torch_snow_high_g0_tile"],
                "torch_snow_high_lh_flux_tile": row["torch_snow_high_lh_flux_tile"],
                "fortran_canopint": ref["skin_water"][i],
                "torch_skin_water": row["torch_skin_water"],
                "diff_canopint": row["torch_skin_water"] - ref["skin_water"][i],
                "fortran_soil_t1": ref["soil_t1"][i],
                "torch_soil_t1": row["torch_soil_t1"],
                "diff_soil_t1": row["torch_soil_t1"] - ref["soil_t1"][i],
                "fortran_soil_w1": ref["soil_w1"][i],
                "torch_soil_w1": row["torch_soil_w1"],
                "diff_soil_w1": row["torch_soil_w1"] - ref["soil_w1"][i],
            }
            writer.writerow(out)
    diffs = np.array([abs(r["torch_avg_skin_t"] - ref["avg_skin_t"][r["step"]]) for r in rows[: len(ref["avg_skin_t"])]])
    print({
        "device": device,
        "csv": str(args.out_csv),
        "mean_abs_avg_skin_t": float(np.nanmean(diffs)),
        "max_abs_avg_skin_t": float(np.nanmax(diffs)),
        "step1_diff_avg_skin_t": float(rows[1]["torch_avg_skin_t"] - ref["avg_skin_t"][1]) if len(rows) > 1 else None,
    })


if __name__ == "__main__":
    main()
