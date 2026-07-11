"""Idealized diurnal-cycle forcing generator for the land-surface model.

This module produces a self-consistent atmospheric forcing time series
plus a matching static descriptor and cold-start state, so a multi-step
column run can be exercised without any external data file.  The
forcing is **analytic** (closed-form diurnal cycle): no NetCDF reader,
no interpolation -- every field is computed from a small
:class:`IdealizedDiurnalConfig`.

Two design points carry over from the IFS convention (see
:mod:`surf_pytorch.state`):

* **Units.**  The :class:`Forcing` container carries *cumulative*
  radiation and precipitation (``fr_so`` / ``fr_th`` in J/m^2 and
  ``rain`` / ``snow`` in kg/m^2 over the coupling interval), because
  that is the form the upstream ``VDFMAIN1S`` consumes and the form
  ``LandSurface.forward`` divides back by ``dt`` (model.py §5).  The
  generator therefore computes everything at the *rate* level first
  (W/m^2, kg/m^2/s) and multiplies by ``dt`` before packing, exactly
  as the per-step test builder ``tests/test_forward_m5._make_inputs``
  does (``fr_so = 200.0 * 3600.0``).
* **Sign convention.**  ``fr_so`` is downwelling short-wave (>= 0);
  ``fr_th`` is the *net* long-wave, positive downward.  The default
  ``-300*3600`` used across the test-suite is a net radiative *loss*
  (upward), so the generator's net LW must come out negative on a
  typical night-to-day cycle -- which it does (see
  :func:`_net_longwave`).

The static descriptor uses the canonical **pure-land** tile fractions
(``tests/test_forward_m5._make_inputs``): wet-skin 0.05 / low-veg-dry
0.30 / high-veg-dry 0.30 / bare-soil 0.35, no water / ice / snow tiles.
That keeps every column a genuine land column so the HTESSEL soil
solves actually fire and the water / energy budgets are non-trivial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .constants import ConstantBundle, PhysicalConstants
from .state import Forcing, Static, SurfState
from .utils import qsat


# --------------------------------------------------------------------------- #
# Solar ephemeris (simplified, analytic, testable)
# --------------------------------------------------------------------------- #
def solar_mu0(lat_rad: float, lon_rad: float, day_of_year: int,
              hour_utc: float) -> float:
    """Solar zenith-angle cosine ``mu0`` (>= 0, 0 at night).

    A deliberately simple closed form (Cooper 1969 declination + a
    single hour-angle term) -- good enough to drive a realistic diurnal
    SW cycle for the harness, and trivially unit-testable.  The full
    OpenIFS ephemeris (``cpg1s.F90`` 297-300) carries the ~0.5 K/year
    orbital corrections that are irrelevant at the precision a column
    validation harness needs.

    Parameters
    ----------
    lat_rad, lon_rad:
        Site latitude / longitude in radians.
    day_of_year:
        1..365 (1 January).  172 ~= northern summer solstice.
    hour_utc:
        Fractional hours since 00 UTC of the day.

    Returns
    -------
    float
        ``mu0`` clamped to ``[0, 1]``; zero when the sun is below the
        horizon.
    """
    # Declination (Cooper 1969): delta = 23.45 deg * sin(2pi/365 (284+N)).
    delta = math.radians(23.45) * math.sin(2.0 * math.pi * (284 + day_of_year)
                                           / 365.0)
    # Solar time (hours): UTC + longitude/15deg, with the -12h offset that
    # turns a clock hour into the hour-angle convention (noon = 0).
    hour_local_solar = hour_utc + math.degrees(lon_rad) / 15.0
    omega = math.radians(15.0) * (hour_local_solar - 12.0)
    mu0 = (math.sin(lat_rad) * math.sin(delta)
           + math.cos(lat_rad) * math.cos(delta) * math.cos(omega))
    return max(0.0, min(1.0, mu0))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class IdealizedDiurnalConfig:
    """Knobs for the idealized diurnal-cycle forcing.

    Defaults describe a mid-latitude (45 N) summer day with a modest
    8 K diurnal temperature swing, a light 5 m/s wind, 70 % RH and no
    precipitation -- a clean drying scenario.  Flip
    :attr:`precip_events` to inject rain / snow pulses.
    """

    # --- Grid / time ---------------------------------------------------- #
    n_grid: int = 4
    dt_seconds: float = 3600.0
    n_steps: int = 48                 # 48 h at 1 h dt

    # --- Site ----------------------------------------------------------- #
    lat_deg: float = 45.0
    lon_deg: float = 0.0
    day_of_year: int = 172            # ~ summer solstice

    # --- Near-surface state -------------------------------------------- #
    tair_mean: float = 290.0          # K
    tair_amp: float = 8.0             # K, peak-to-mean
    tair_peak_local_hour: float = 14.0  # hour of the diurnal T maximum
    rh: float = 0.7                   # relative humidity, 0..1
    wind_speed: float = 5.0           # m/s (u component; v = 0)

    # --- Radiation knobs ------------------------------------------------ #
    sw_peak: float = 800.0            # W/m^2 surface SW at mu0=1, before tau
    tau_atm: float = 0.7              # broadband atmospheric transmittance
    atm_emis: float = 0.75            # effective sky emissivity for LWdown
    emis_sfc: float = 0.99            # surface emissivity (Static.emis too)

    # --- Pressure / composition ---------------------------------------- #
    p_surf: float = 1.0e5             # Pa
    co2: float = 6e-4                 # kg/kg
    geom1: float = 50.0               # m^2/s^2 lowest-level geopotential

    # --- Precipitation -------------------------------------------------- #
    # Each tuple is (start_step, end_step, rate_kg_m2_s); the pulse applies
    # to steps k with start_step <= k < end_step.  Rain/snow phase is split
    # by Tair vs RTT at the step.
    precip_events: List[Tuple[int, int, float]] = field(default_factory=list)

    # --- Cold-start state ---------------------------------------------- #
    soil_type: int = 3                # medium (canonical test default)
    veg_type: int = 3
    soil_t_init: float = 285.0
    soil_w_init: float = 0.30         # volumetric, m3/m3
    skin_t_init: float = 288.0
    sst_init: float = 290.0

    # --- Numeric -------------------------------------------------------- #
    dtype: torch.dtype = torch.float64
    device: Optional[str] = None


# --------------------------------------------------------------------------- #
# Internal rate-level builders (W/m^2, kg/m^2/s)
# --------------------------------------------------------------------------- #
def _air_temperature(cfg: IdealizedDiurnalConfig, hour_local: float) -> float:
    """Tair (K) at a given local solar hour, peaking at tair_peak_local_hour."""
    phase = 2.0 * math.pi * (hour_local - cfg.tair_peak_local_hour) / 24.0
    return cfg.tair_mean + cfg.tair_amp * math.cos(phase)


def _shortwave_down(cfg: IdealizedDiurnalConfig, mu0: float) -> float:
    """Downwelling surface SW (W/m^2, >= 0)."""
    return cfg.sw_peak * mu0 * cfg.tau_atm


def _net_longwave(cfg: IdealizedDiurnalConfig, tair: float,
                  sigma: float) -> float:
    """Net long-wave flux (W/m^2), positive downward.

    ``LWdown = atm_emis * sigma * Tair^4`` (effective sky emissivity form)
    and ``LWup = emis_sfc * sigma * Tair^4`` (skin proxy = Tair -- the
    prognostic skin-T is unknown at forcing-construction time).  The net
    is therefore typically *negative* (a radiative loss) for Tair ~ 290 K
    with the defaults, matching the sign convention of the test-suite
    default ``fr_th = -300*3600``.
    """
    lw_down = cfg.atm_emis * sigma * tair ** 4
    lw_up = cfg.emis_sfc * sigma * tair ** 4
    return lw_down - lw_up


# --------------------------------------------------------------------------- #
# Public builder
# --------------------------------------------------------------------------- #
def build_idealized_diurnal(
    cfg: Optional[IdealizedDiurnalConfig] = None,
    consts: Optional[ConstantBundle] = None,
) -> Tuple[List[Forcing], Static, SurfState]:
    """Build a forcing time series + matching static + cold-start state.

    Parameters
    ----------
    cfg:
        :class:`IdealizedDiurnalConfig` (defaults to a mid-latitude
        summer day with no precip).
    consts:
        Optional :class:`ConstantBundle` override.  Defaults to a
        fresh :class:`PhysicalConstants` bundle at ``cfg.dtype``; this
        matches what :class:`LandSurface` builds internally, so the
        saturation functions used here (RH -> q) agree with the model's
        own qsat to last ULP.

    Returns
    -------
    (forcing_list, static, initial_state)
        ``forcing_list`` has ``cfg.n_steps`` entries, one per coupling
        interval.  ``static`` carries the pure-land tile fractions and
        the site lat/lon.  ``initial_state`` is a cold-start
        :class:`SurfState`.
    """
    cfg = cfg or IdealizedDiurnalConfig()
    dtype = cfg.dtype
    device = cfg.device
    bundle = ConstantBundle(
        consts or PhysicalConstants(), dtype=dtype, device=device,
    )
    sigma = float(bundle.rsigma.item())
    rtt = float(bundle.rtt.item())
    dt = float(cfg.dt_seconds)

    n = cfg.n_grid
    lat_rad = math.radians(cfg.lat_deg)
    lon_rad = math.radians(cfg.lon_deg)

    # Constant-per-column vectors that every step shares.
    u = torch.full((n,), cfg.wind_speed, dtype=dtype, device=device)
    v = torch.zeros((n,), dtype=dtype, device=device)
    co2 = torch.full((n,), cfg.co2, dtype=dtype, device=device)
    p_surf = torch.full((n,), cfg.p_surf, dtype=dtype, device=device)
    geom1 = torch.full((n,), cfg.geom1, dtype=dtype, device=device)
    dt_t = torch.tensor(dt, dtype=dtype, device=device)

    forcing_list: List[Forcing] = []
    for k in range(cfg.n_steps):
        # Mid-interval local solar hour (UTC + lon/15h).
        hour_utc = (k + 0.5) * dt / 3600.0
        hour_local = hour_utc + cfg.lon_deg / 15.0
        tair = _air_temperature(cfg, hour_local)
        mu0 = solar_mu0(lat_rad, lon_rad, cfg.day_of_year, hour_utc)

        sw_down = _shortwave_down(cfg, mu0)          # W/m^2
        net_lw = _net_longwave(cfg, tair, sigma)     # W/m^2 (down-positive)

        # Specific humidity from fixed RH: q = RH * qsat(Tair, p_surf).
        tair_t = torch.full((n,), tair, dtype=dtype, device=device)
        q_sat = qsat(tair_t, p_surf, bundle)
        q = cfg.rh * q_sat

        # Precipitation throughfall: sum the rates of all events covering
        # this step, split rain / snow by Tair vs RTT.
        rain_rate = 0.0
        snow_rate = 0.0
        for (k0, k1, rate) in cfg.precip_events:
            if k0 <= k < k1:
                if tair >= rtt:
                    rain_rate += rate
                else:
                    snow_rate += rate

        forcing_list.append(Forcing(
            u=u.clone(), v=v.clone(),
            t=tair_t, q=q, co2=co2.clone(),
            p_surf=p_surf.clone(), geom1=geom1.clone(),
            fr_so=torch.full((n,), sw_down * dt, dtype=dtype, device=device),
            fr_th=torch.full((n,), net_lw * dt, dtype=dtype, device=device),
            mu0=torch.full((n,), mu0, dtype=dtype, device=device),
            rain=torch.full((n,), rain_rate * dt, dtype=dtype, device=device),
            snow=torch.full((n,), snow_rate * dt, dtype=dtype, device=device),
            dt=dt_t.clone(),
        ))

    static, initial_state = _build_static_and_state(cfg, dtype, device)
    return forcing_list, static, initial_state


# --------------------------------------------------------------------------- #
# Static + cold-start state
# --------------------------------------------------------------------------- #
def _build_static_and_state(
    cfg: IdealizedDiurnalConfig, dtype: torch.dtype, device,
) -> Tuple[Static, SurfState]:
    """Canonical pure-land static descriptor + cold-start state.

    Tile fractions mirror ``tests/test_forward_m5._make_inputs``:
    wet-skin 0.05 / low-veg-dry 0.30 / high-veg-dry 0.30 / bare-soil
    0.35 (pure land, no water / ice / snow tiles).  Soil-layer
    thicknesses are the SurfConfig default ``(0.07, 0.21, 0.72, 1.89)``
    so the budget tests can divide by ``rdaw`` cleanly.
    """
    from .config import N_BASE_TILES

    n = cfg.n_grid
    ntile = N_BASE_TILES
    nsoil = 4
    soil_thickness = torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype,
                                  device=device)

    frac = torch.zeros((n, ntile), dtype=dtype, device=device)
    frac[:, 2] = 0.05   # wet skin
    frac[:, 3] = 0.30   # low-veg dry
    frac[:, 5] = 0.30   # high-veg dry
    frac[:, 7] = 0.35   # bare soil

    static = Static(
        tile_frac_base=frac,
        land_frac=torch.ones((n,), dtype=dtype, device=device),
        orography=torch.full((n,), 50.0, dtype=dtype, device=device),
        soil_type=torch.full((n,), cfg.soil_type, dtype=torch.long,
                             device=device),
        veg_type=torch.full((n,), cfg.veg_type, dtype=torch.long,
                            device=device),
        soil_thickness=soil_thickness,
        emis=torch.full((n,), cfg.emis_sfc, dtype=dtype, device=device),
        lat=torch.full((n,), math.radians(cfg.lat_deg), dtype=dtype,
                       device=device),
        lon=torch.full((n,), math.radians(cfg.lon_deg), dtype=dtype,
                       device=device),
    )

    initial_state = SurfState(
        skin_t=torch.full((n, ntile), cfg.skin_t_init, dtype=dtype,
                          device=device),
        soil_t=torch.full((n, nsoil), cfg.soil_t_init, dtype=dtype,
                          device=device),
        soil_w_liq=torch.full((n, nsoil), cfg.soil_w_init, dtype=dtype,
                              device=device),
        snow_w=torch.zeros((n,), dtype=dtype, device=device),
        snow_frac=torch.zeros((n,), dtype=dtype, device=device),
        skin_water=torch.zeros((n,), dtype=dtype, device=device),
        sst=torch.full((n,), cfg.sst_init, dtype=dtype, device=device),
    )
    return static, initial_state


__all__ = [
    "IdealizedDiurnalConfig",
    "solar_mu0",
    "build_idealized_diurnal",
]
