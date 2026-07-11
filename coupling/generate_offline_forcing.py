#!/usr/bin/env python3
"""Emit an OpenIFS offline-driver forcing NetCDF from the analytic generator.

The OpenIFS 1-D surface driver reads its atmospheric forcing from a single
NetCDF file (default name ``forcing``) whose schema is fixed by
``surf/offline/driver/sufcdf.F90`` (lines 115-131) and read back by
``rdfvar.F90``.  This script bridges that schema to
:func:`surf_pytorch.forcing.build_idealized_diurnal`, which already
produces a closed-form, self-consistent diurnal-cycle forcing time series
for the PyTorch port.  Using the same generator for both sides is what
makes the bit-level comparison meaningful: the Fortran driver and the
PyTorch model see *identical* inputs.

Units / sign conventions (the one place the two worlds differ)
--------------------------------------------------------------
``surf_pytorch.state.Forcing`` carries **cumulative** radiation and
precipitation over the coupling interval (``fr_so`` / ``fr_th`` in J/m^2,
``rain`` / ``snow`` in kg/m^2) -- already multiplied by ``dt`` -- because
that is the form ``VDFMAIN1S`` consumes and the form
``LandSurface.forward`` divides back by ``dt`` (see
``surf_pytorch/forcing.py`` header, "Units").

The OpenIFS forcing NetCDF, by contrast, holds **rates** (W/m^2 for
radiation, kg/m^2/s for precipitation): ``SWdown``/``LWdown`` are
downwelling flux densities and ``Rainf``/``Snowf`` are mass fluxes.  The
driver's ``dtforc.F90`` interpolates these rates in time and
``callpar1s.F90`` (lines 811-812) converts downwelling -> net (via albedo
and emissivity) before the surface call -- so the emitter's job is purely
to emit the *downwelling rates* and let the driver do its own
accumulation.

The sign mapping is therefore:

* ``SWdown`` = ``fr_so / dt``  (downwelling SW, >= 0)
* ``LWdown`` = ``(fr_th/dt) + emis_sfc*sigma*Tair^4``
  -- the generator's ``fr_th`` is the *net* LW (down-positive); to recover
  the downwelling component the driver expects, we add back the surface
  upwelling term ``emis*sigma*Tair^4`` that the generator subtracted in
  ``_net_longwave``.  With a matching ``Tair`` the round-trip is exact.
* ``Rainf`` = ``rain / dt`` (liquid precip rate)
* ``Snowf`` = ``snow / dt`` (solid precip rate)

Schema details (reverse-engineered from sufcdf.F90 / rdfvar.F90)
----------------------------------------------------------------
* dims: ``lon``, ``lat``, ``time`` (we emit NDIMCDF=2 single-cell:
  lon=lat=1 so the driver's 2-D reader is exercised with NLON=NLAT=1).
* coord vars: ``lat`` (degrees_north), ``lon`` (degrees_east),
  ``time`` (units ``hours since YYYY-MM-DD HH:MM:SS`` -- rdfvar supports
  seconds/minutes/hours/days since; we use hours for compactness).
* forcing vars: ``Tair``(K), ``Qair``(kg/kg), ``Wind_E``+``Wind_N``(m/s),
  ``PSurf``(Pa), ``SWdown``(W/m^2), ``LWdown``(W/m^2),
  ``Rainf``(kg/m^2/s), ``Snowf``(kg/m^2/s).  ``Ctpf`` (convective
  precip fraction) is omitted -> driver sets convective precip to 0.
  ``CO2air`` is omitted -> driver uses the global annual mean when
  ``LEAIRCO2COUP`` is off (the shim hardcodes 6e-4 kg/kg anyway).

All variables are written as float64 to match the driver's
``REAL(KIND=JPRB)`` (the Fortran side upcasts on read).

Usage
-----
    python generate_offline_forcing.py [--out forcing] [--steps 48] \
        [--dt 3600] [--lat 45.0] [--lon 0.0]

Run-directory setup (input namelist + surfclim/soilinit namelist fallback)
is handled by ``make_offline_run_dir.py``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
import sys
from typing import Tuple

import numpy as np
import netCDF4

# Make ``surf_pytorch`` importable when run as a standalone script from
# anywhere (the run-dir layout does not assume a particular CWD).
_REPO_ROOT = "/home/qixiang/yangjinhui/openifs"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from surf_pytorch.forcing import (  # noqa: E402  (sys.path tweak above)
    IdealizedDiurnalConfig,
    build_idealized_diurnal,
)
from surf_pytorch.constants import ConstantBundle, PhysicalConstants  # noqa: E402


# Canonical reference epoch for the forcing time axis.  Must be consistent
# with NAMRIP/NINDAT in the run-directory namelist: the driver checks
# ``RTIMTR`` (model midpoint) against ``RTSTFC`` (forcing start) and aborts
# if the run starts before the forcing.  We anchor everything at this date
# and emit the matching NAMRIP from make_offline_run_dir.py.
REF_DATE = _dt.datetime(1979, 1, 1, 0, 0, 0)


def _forcing_rates(forcing_list) -> Tuple[np.ndarray, ...]:
    """Convert cumulative Forcing tensors to per-step rate arrays.

    Returns 1-D numpy arrays of length ``len(forcing_list)``, each holding
    the *rate-level* value the OpenIFS forcing NetCDF expects (W/m^2 or
    kg/m^2/s).  The conversion from the generator's cumulative convention
    is documented in the module header.

    ``LWdown`` is reconstructed from the generator's *net* LW by adding
    back ``emis_sfc * sigma * Tair^4`` (the surface upwelling the
    generator subtracted to form the net).  We read the same sigma the
    generator used so the round-trip is exact to last ULP.
    """
    n = len(forcing_list)
    # Use the default constant bundle so sigma matches the generator's own
    # qsat / LW computations to last ULP (see forcing.build_idealized_diurnal).
    import torch
    bundle = ConstantBundle(PhysicalConstants(), dtype=torch.float64)
    sigma = float(bundle.rsigma.item())
    emis_sfc = IdealizedDiurnalConfig().emis_sfc

    tair = np.empty(n)
    qair = np.empty(n)
    wind_e = np.empty(n)
    wind_n = np.empty(n)
    psurf = np.empty(n)
    swdown = np.empty(n)
    lwdown = np.empty(n)
    rainf = np.empty(n)
    snowf = np.empty(n)
    dt = float(forcing_list[0].dt.item())

    for k, f in enumerate(forcing_list):
        t = float(f.t.item())
        # LWdown = net_LW/dt + emis*sigma*T^4 (undo _net_longwave's LWup).
        lw_net_rate = float(f.fr_th.item()) / dt
        lw_up = emis_sfc * sigma * t ** 4
        lwdown[k] = lw_net_rate + lw_up
        swdown[k] = float(f.fr_so.item()) / dt
        rainf[k] = float(f.rain.item()) / dt
        snowf[k] = float(f.snow.item()) / dt
        tair[k] = t
        qair[k] = float(f.q.item())
        wind_e[k] = float(f.u.item())
        wind_n[k] = float(f.v.item())
        psurf[k] = float(f.p_surf.item())

    return tair, qair, wind_e, wind_n, psurf, swdown, lwdown, rainf, snowf, dt


def write_forcing_nc(
    out_path: str,
    cfg: IdealizedDiurnalConfig,
    forcing_list,
    ref_date: _dt.datetime = REF_DATE,
) -> None:
    """Write a single-cell NDIMCDF=2 forcing NetCDF at ``out_path``.

    The file matches the schema ``sufcdf.F90`` expects: dims
    ``(lon=1, lat=1, time=N)``, coord vars ``lat``/``lon``/``time``, and
    the 9 forcing variables.  NetCDF4 classic with float64 data.
    """
    n_steps = len(forcing_list)
    dt_hours = cfg.dt_seconds / 3600.0

    (tair, qair, wind_e, wind_n, psurf,
     swdown, lwdown, rainf, snowf, _) = _forcing_rates(forcing_list)

    lat_deg = cfg.lat_deg
    lon_deg = cfg.lon_deg
    time_vals = np.arange(n_steps, dtype=np.float64) * dt_hours
    ref_str = ref_date.strftime("%Y-%m-%d %H:%M:%S")

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds:
        # --- dimensions -------------------------------------------------- #
        ds.createDimension("lon", 1)
        ds.createDimension("lat", 1)
        ds.createDimension("time", n_steps)

        # --- coordinate variables --------------------------------------- #
        v_lon = ds.createVariable("lon", "f8", ("lon",))
        v_lon.units = "degrees_east"
        v_lon.long_name = "longitude"
        v_lon[:] = [lon_deg]

        v_lat = ds.createVariable("lat", "f8", ("lat",))
        v_lat.units = "degrees_north"
        v_lat.long_name = "latitude"
        v_lat[:] = [lat_deg]

        v_time = ds.createVariable("time", "f8", ("time",))
        # rdfvar.F90 parses this exact prefix; "hours since" -> ZDELS=3600.
        v_time.units = f"hours since {ref_str}"
        v_time.long_name = "time"
        v_time[:] = time_vals

        # --- forcing variables (time, lat, lon) ------------------------- #
        # Variable shape (time, lat, lon): rdfvar reads VNDIMS=3 with the
        # lon dim first and lat second; netCDF4 stores in C order so we
        # declare dims (time, lat, lon) and assign the 1x1 column slice.
        def _mk(name, units, long_name):
            v = ds.createVariable(name, "f8", ("time", "lat", "lon"))
            v.units = units
            v.long_name = long_name
            return v

        _mk("Tair",   "K",        "Temperature")[:, 0, 0] = tair
        _mk("Qair",   "kg/kg",    "Specific humidity")[:, 0, 0] = qair
        _mk("Wind_E", "m/s",      "Wind speed u")[:, 0, 0] = wind_e
        _mk("Wind_N", "m/s",      "Wind speed v")[:, 0, 0] = wind_n
        _mk("PSurf",  "Pa",       "Pressure")[:, 0, 0] = psurf
        _mk("SWdown", "W/m2",     "Downward shortwave radiation")[:, 0, 0] = swdown
        _mk("LWdown", "W/m2",     "Downward longwave radiation")[:, 0, 0] = lwdown
        _mk("Rainf",  "kg/m2s",   "Rainfall")[:, 0, 0] = rainf
        _mk("Snowf",  "kg/m2s",   "Snowfall")[:, 0, 0] = snowf

        # --- global attributes ------------------------------------------ #
        ds.Conventions = "CF-1.0"
        ds.title = "OpenIFS offline forcing (synthetic diurnal cycle)"
        ds.source = "surf_pytorch.coupling.generate_offline_forcing"
        ds.history = (
            f"Generated {__import__('datetime').datetime.utcnow().isoformat()}Z "
            f"from build_idealized_diurnal (lat={lat_deg}, lon={lon_deg}, "
            f"dt={cfg.dt_seconds}s, n_steps={n_steps})"
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Emit an OpenIFS offline forcing NetCDF from the "
                    "analytic diurnal-cycle generator.",
    )
    p.add_argument("--out", default="forcing",
                   help="output NetCDF path (default: ./forcing)")
    p.add_argument("--steps", type=int, default=48,
                   help="number of forcing time steps (default 48 = 48h @1h)")
    p.add_argument("--dt", type=float, default=3600.0,
                   help="forcing timestep in seconds (default 3600)")
    p.add_argument("--lat", type=float, default=45.0,
                   help="site latitude in degrees (default 45)")
    p.add_argument("--lon", type=float, default=0.0,
                   help="site longitude in degrees (default 0)")
    p.add_argument("--day-of-year", type=int, default=172,
                   help="day of year 1..365 (default 172 ~ summer solstice)")
    p.add_argument("--tair-mean", type=float, default=290.0,
                   help="mean near-surface air temperature K (default 290)")
    p.add_argument("--tair-amp", type=float, default=8.0,
                   help="diurnal T amplitude K (default 8)")
    p.add_argument("--wind-speed", type=float, default=5.0,
                   help="near-surface wind speed m/s (default 5)")
    p.add_argument("--rh", type=float, default=0.7,
                   help="relative humidity 0..1 (default 0.7)")
    p.add_argument("--precip", action="store_true",
                   help="include a sample rain event (steps 10-14)")
    args = p.parse_args(argv)

    cfg = IdealizedDiurnalConfig(
        n_grid=1,
        dt_seconds=args.dt,
        n_steps=args.steps,
        lat_deg=args.lat,
        lon_deg=args.lon,
        day_of_year=args.day_of_year,
        tair_mean=args.tair_mean,
        tair_amp=args.tair_amp,
        wind_speed=args.wind_speed,
        rh=args.rh,
        precip_events=[(10, 14, 1e-4)] if args.precip else [],
    )

    print(f"[gen-forcing] generating {args.steps} steps @ {args.dt}s "
          f"(lat={args.lat}, lon={args.lon})")
    forcing_list, static, state = build_idealized_diurnal(cfg)
    write_forcing_nc(args.out, cfg, forcing_list)
    print(f"[gen-forcing] wrote {args.out}")

    # Echo the rate arrays so a human can sanity-check the magnitudes
    # without opening the file.
    (tair, qair, wind_e, wind_n, psurf, swdown, lwdown,
     rainf, snowf, _) = _forcing_rates(forcing_list)
    print(f"[gen-forcing] Tair   [{tair.min():.2f}, {tair.max():.2f}] K")
    print(f"[gen-forcing] Qair   [{qair.min():.3e}, {qair.max():.3e}] kg/kg")
    print(f"[gen-forcing] Wind_E [{wind_e.min():.2f}, {wind_e.max():.2f}] m/s")
    print(f"[gen-forcing] PSurf  [{psurf.min():.1f}, {psurf.max():.1f}] Pa")
    print(f"[gen-forcing] SWdown [{swdown.min():.1f}, {swdown.max():.1f}] W/m2")
    print(f"[gen-forcing] LWdown [{lwdown.min():.1f}, {lwdown.max():.1f}] W/m2")
    print(f"[gen-forcing] Rainf  [{rainf.min():.3e}, {rainf.max():.3e}] kg/m2/s")
    print(f"[gen-forcing] Snowf  [{snowf.min():.3e}, {snowf.max():.3e}] kg/m2/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
