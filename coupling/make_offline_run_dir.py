#!/usr/bin/env python3
"""Construct a complete OpenIFS offline-driver run directory.

Lays down the four artifacts the offline surface driver needs to reach the
PyTorch coupling hook (``cnt41s -> PYOIFS_INIT``):

1. ``input``  -- the master namelist (all 12 namelist groups).  Critically
   sets ``CFFORC='netcdf'`` (forcing is always NetCDF) but
   ``CFSURF/CFINIT`` to anything *other* than ``'netcdf'`` so that
   ``sugc1s``/``sugp1s``/``sugpd1s`` take their **namelist fallback path**
   (single-column cold-start from NAMGC1S/NAMGP1S/NAMGPD1S).  This avoids
   needing the licence-gated ``surfclim``/``soilinit`` NetCDF files.

2. ``forcing`` -- the atmospheric forcing NetCDF, written by
   :mod:`generate_offline_forcing` from the analytic diurnal-cycle
   generator.  This is the *only* NetCDF input the driver requires.

3. ``run_offline_coupled.sh`` link / wrapper invocation (documented in the
   generated README, not a file the driver itself reads).

4. ``README.md`` -- documents what the directory contains and how to run.

The namelist values are chosen to be self-consistent with the forcing:
``TSTEP == DTIMFC`` (so one model step == one forcing step, no time
interpolation), ``NINDAT`` anchored at the forcing reference epoch, and
``NSTOP`` matching the number of forcing steps.  ``LEPYOIFS=.TRUE.`` is
placed in NAMPHYOFF to redirect SURFTSTP to the PyTorch bridge; this also
forces ``LESURF=.FALSE.`` in NAMPHY because callpar1s dispatches the
surface call as ``IF(LESURF) ... ELSEIF(LEPYOIFS) ...`` -- both true would
run the Fortran SURFTSTP first and leave the PyTorch branch dead-coded.

Run-directory layout decisions
------------------------------
* NDIMCDF=2 with NLON=NLAT=1 (single cell).  This exercises the driver's
  2-D reader path (the most-tested one) with a degenerate 1x1 grid.  The
  alternative NDIMCDF=1 (1-D site list) requires the forcing vars to be
  2-D (lon,time) and trips the ``NILAT .NE. NLAT`` check differently.
* Soil/veg via NAMGPD1S: canonical pure-land column (ITVH=3, medium soil
  type 3) matching surf_pytorch's test defaults.
* Cold-start state via NAMGP1S: soil T 285 K, soil moisture 0.30 m3/m3,
  skin T 288 K, no snow -- matching ``IdealizedDiurnalConfig`` defaults.

Usage
-----
    python make_offline_run_dir.py [--dir run_pyoifs] [--steps 48] ...
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_SCRIPT_DIR = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from surf_pytorch.forcing import IdealizedDiurnalConfig  # noqa: E402

# Forcing epoch (must match generate_offline_forcing.REF_DATE and the
# NINDAT/NSSSSS we write below -- the driver checks that the run does not
# start before the forcing, see rdfvar.F90:180).
REF_DATE = _dt.datetime(1979, 1, 1, 0, 0, 0)


def _namelist_text(cfg: IdealizedDiurnalConfig, n_steps: int,
                   tstep: float, lepyoifs: bool) -> str:
    """Render the 12-group master namelist the driver reads as ``input``.

    Groups must appear in the order su0yom1s reads them (each subroutine
    does REWIND(NULNAM) before its own READ, so order is not strictly
    enforced, but keeping the canonical order aids debugging).
    """
    # NINDAT = YYYYMMDD, NSSSSS = seconds since 00UTC of the run start day.
    nindat = REF_DATE.year * 10000 + REF_DATE.month * 100 + REF_DATE.day
    nsssss = REF_DATE.hour * 3600 + REF_DATE.minute * 60 + REF_DATE.second
    # NSTOP: number of model steps.  With TSTEP==DTIMFC, one model step
    # consumes exactly one forcing record, so NSTOP = n_steps-1 keeps us
    # inside the forcing window (rdfvar reads NSTPFC records; the last
    # model step needs forcing[ NSTOP ] to exist).
    nstop = n_steps - 1
    cfsurf = "ascii"   # anything != 'netcdf' -> namelist fallback path

    lepyoifs_str = ".TRUE." if lepyoifs else ".FALSE."
    # LESURF must be the complement of LEPYOIFS: callpar1s.F90 has
    #   IF (LESURF) THEN CALL SURFTSTP(...)   ! Fortran surface
    #   ELSEIF (LEPYOIFS) THEN CALL PYOIFS_SURFTSTP(...)
    # If both are .TRUE. the Fortran SURFTSTP runs first (LECTESSEL=.FALSE.
    # -> uninitialized tile temps -> NaN) and the PyTorch branch is dead code.
    # So when LEPYOIFS=.TRUE. we force LESURF=.FALSE. to take the ELSEIF path;
    # when LEPYOIFS=.FALSE. (reference run) LESURF=.TRUE. runs the native surface.
    lesurf_str = ".FALSE." if lepyoifs else ".TRUE."

    # NAMGP1S cold-start: matches IdealizedDiurnalConfig defaults so the
    # PyTorch port's initial state and the Fortran cold-start agree to
    # last ULP on the first step (the bit-level comparison's baseline).
    soil_t = cfg.soil_t_init        # K, applied to all 4 soil layers
    soil_w = cfg.soil_w_init        # m3/m3, all 4 layers (liquid)
    skin_t = cfg.skin_t_init        # K, all 8 tiles
    sst = cfg.sst_init              # K (water/ice tile companion)

    return f"""&NAM1S
 CFFORC='netcdf', CFOUT='ascii', CFSURF='{cfsurf}', CFINIT='{cfsurf}',
 CSITE='pyoifs_offline', NDIMCDF=2, NCDFTYPE=4,
 LWRGG=.FALSE., LWREFL=.FALSE., LWRWAT=.FALSE., LWRSUS=.FALSE.,
 LWRSUB=.FALSE., LWREVA=.FALSE., LWRCLD=.FALSE., LWRCLM=.FALSE.,
 LWRLKE=.FALSE., LWRCO2=.FALSE., LWRVEG=.FALSE., LWRFRA=.FALSE.,
 LWRFOR=.FALSE., LWRBIO=.FALSE., LWRTIL=.FALSE., LWREXT=.FALSE.,
 LWRVTY=.FALSE., LWRD2M=.FALSE., LWRGGD=.FALSE.,
&END
&NAMRIP
 NINDAT={nindat}, NSSSSS={nsssss},
&END
&NAMCT01S
 NSTART=0, NSTOP={nstop}, NFRPOS=1, NFRHIS=1, CNMEXP='PYOIFS',
&END
&NAMDIM
 NLON=1, NLAT=1, NCSS=4, NCSNEC=1,
&END
&NAMDYN1S
 TSTEP={tstep}, TDT={tstep}, NACCTYPE=0,
 LPREINT=.FALSE., LSWINT=.FALSE., LFLXINT=.FALSE.,
&END
&NAMFORC
 ZDTFORC={cfg.dt_seconds}, ZPHISTA=10., ZUV=10.,
 CFORCU='forcing', CFORCV='forcing', CFORCT='forcing',
 CFORCQ='forcing', CFORCP='forcing', CFORCSW='forcing',
 CFORCLW='forcing', CFORCRAIN='forcing', CFORCSNOW='forcing',
&END
&NAMGC1S
 RLAT={cfg.lat_deg}, RLON={cfg.lon_deg}, RMASK=1.,
&END
&NAMGP1S
 TSLNU={soil_t},{soil_t},{soil_t},{soil_t},
 QLINU={soil_w},{soil_w},{soil_w},{soil_w},
 TILNU={soil_t},{soil_t},{soil_t},{soil_t},
 FSNNU=0., TSNNU=273.15, ASNNU=0., RSNNU=300.,
 TRENU={skin_t}, WRENU=0.,
 LAINU=0.,0., BSTRNU=0.,0., BSTR2NU=0.,0.,
&END
&NAMGPD1S
 VZ0F=1.5, VZ0H=1.5, VITM=1.,
 VCVL=0.3, VCVH=0.7, VCUR=0., VTVL=3., VTVH=3.,
 VSST={sst}, VCI=0., VEMISF=0.99, VSOTY=3., VSDOR=0.,
 VLDEPTH=0., VCLAKE=0., VCO2TYP=1.,
 VRSML=100., VRSMH=150., VR0L=200., VR0H=200.,
&END
&NAMOPTSURF
&END
&NAMPHY
 LESURF={lesurf_str}, LEVDIF=.TRUE., LEVGEN=.TRUE.,
 LECTESSEL=.FALSE., LEAGS=.FALSE., LEFLAKE=.FALSE.,
 LEOCML=.FALSE., LESNML=.FALSE., LEURBAN=.FALSE.,
 LEAIRCO2COUP=.FALSE.,
&END
&NAMPHYOFF
 LEPYOIFS={lepyoifs_str},
&END
"""


def _readme_text(run_dir: Path, cfg: IdealizedDiurnalConfig,
                 n_steps: int, tstep: float, lepyoifs: bool) -> str:
    pyoifs_flag = "ON (LEPYOIFS=.TRUE.)" if lepyoifs else "OFF (LEPYOIFS=.FALSE., Fortran SURFTSTP runs)"
    return f"""# OpenIFS offline-driver run directory (PyTorch coupling)

Generated by ``surf_pytorch/coupling/make_offline_run_dir.py``.

## Contents
- ``input``       -- master namelist (12 groups, CFSURF/CFINIT=ascii
  fallback so no surfclim/soilinit NetCDF needed; forcing is NetCDF).
- ``forcing``     -- atmospheric forcing NetCDF (analytic diurnal cycle,
  n_steps={n_steps} @ dt={cfg.dt_seconds}s, lat={cfg.lat_deg}/lon={cfg.lon_deg}).
- ``README.md``   -- this file.

## Run
From this directory:

    {run_offline_coupled_path(run_dir)}

The driver parses the namelist, reads ``forcing``, sets up the single-cell
grid (lat={cfg.lat_deg}, lon={cfg.lon_deg}, pure land), and enters the
timestep loop (NSTOP={n_steps - 1} steps x TSTEP={tstep}s).

## PyTorch coupling
LEPYOIFS = **{pyoifs_flag}**.

When ON, ``cnt41s`` calls ``PYOIFS_INIT`` before the loop and each
``callpar1s`` step redirects SURFTSTP to the PyTorch bridge
(libpyoifs.so -> surf_pytorch.LandSurface.forward) instead of the native
HTESSEL.

## Cold-start (must match surf_pytorch's IdealizedDiurnalConfig)
- soil T      : {cfg.soil_t_init} K (all 4 layers)
- soil moisture: {cfg.soil_w_init} m3/m3 (all 4 layers)
- skin T      : {cfg.skin_t_init} K (all tiles)
- SST         : {cfg.sst_init} K
- snow        : none
"""


def run_offline_coupled_path(run_dir: Path) -> str:
    """Return the command to launch the coupled exe from ``run_dir``."""
    script = _SCRIPT_DIR / "run_offline_coupled.sh"
    rel = os.path.relpath(script, run_dir)
    return f"RUN_DIR=$(pwd) bash {rel}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build an OpenIFS offline run directory for the "
                    "PyTorch coupling validation.",
    )
    p.add_argument("--dir", default="run_pyoifs",
                   help="run directory to create (default ./run_pyoifs)")
    p.add_argument("--steps", type=int, default=48,
                   help="forcing/model steps (default 48)")
    p.add_argument("--dt", type=float, default=3600.0,
                   help="timestep seconds, forcing == model (default 3600)")
    p.add_argument("--lat", type=float, default=45.0)
    p.add_argument("--lon", type=float, default=0.0)
    p.add_argument("--day-of-year", type=int, default=172)
    p.add_argument("--tair-mean", type=float, default=290.0)
    p.add_argument("--tair-amp", type=float, default=8.0)
    p.add_argument("--wind-speed", type=float, default=5.0)
    p.add_argument("--rh", type=float, default=0.7)
    p.add_argument("--precip", action="store_true",
                   help="include a sample rain event")
    p.add_argument("--no-pyoifs", action="store_true",
                   help="emit LEPYOIFS=.FALSE. (run native Fortran as reference)")
    args = p.parse_args(argv)

    run_dir = Path(args.dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

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
        soil_t_init=285.0,
        soil_w_init=0.30,
        skin_t_init=288.0,
        sst_init=290.0,
    )

    # 1. forcing NetCDF
    gen = _SCRIPT_DIR / "generate_offline_forcing.py"
    forcing_path = run_dir / "forcing"
    cmd = [
        sys.executable, str(gen),
        "--out", str(forcing_path),
        "--steps", str(args.steps),
        "--dt", str(args.dt),
        "--lat", str(args.lat),
        "--lon", str(args.lon),
        "--day-of-year", str(args.day_of_year),
        "--tair-mean", str(args.tair_mean),
        "--tair-amp", str(args.tair_amp),
        "--wind-speed", str(args.wind_speed),
        "--rh", str(args.rh),
    ]
    if args.precip:
        cmd.append("--precip")
    print(f"[make-run-dir] generating forcing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 2. input namelist
    nl = _namelist_text(cfg, args.steps, args.dt, not args.no_pyoifs)
    (run_dir / "input").write_text(nl)
    print(f"[make-run-dir] wrote {run_dir / 'input'}")

    # 3. README
    (run_dir / "README.md").write_text(
        _readme_text(run_dir, cfg, args.steps, args.dt, not args.no_pyoifs)
    )
    print(f"[make-run-dir] wrote {run_dir / 'README.md'}")

    print(f"\n[make-run-dir] run directory ready: {run_dir}")
    print(f"[make-run-dir] to launch: cd {run_dir} && "
          f"RUN_DIR=$(pwd) bash {os.path.relpath(_SCRIPT_DIR / 'run_offline_coupled.sh', run_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
