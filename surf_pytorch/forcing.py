"""陆面模式的理想化日循环强迫生成器。

本模块产生一套自洽的大气强迫时间序列，外加匹配的静态描述符与冷启动
状态，使多步单列运行无需任何外部数据文件即可演练。强迫是**解析式**的
（闭式日循环）：不读 NetCDF、不做插值——每个场都由一个小巧的
:class:`IdealizedDiurnalConfig` 算出。

沿袭 IFS 约定的两个设计点（见 :mod:`surf_pytorch.state`）：

* **单位。** :class:`Forcing` 容器携带*累积*辐射与降水
  （耦合间隔内的 ``fr_so`` / ``fr_th``，单位 J/m^2，以及 ``rain`` /
  ``snow``，单位 kg/m^2），因为这正是上游 ``VDFMAIN1S`` 所消费、也是
  ``LandSurface.forward`` 要除回 ``dt`` 的形式（model.py §5）。因此
  生成器先在*速率*层面（W/m^2、kg/m^2/s）计算所有量，再乘以 ``dt``
  打包，与单步测试构造器 ``tests/test_forward_m5._make_inputs`` 的做法
  完全一致（``fr_so = 200.0 * 3600.0``）。
* **符号约定。** ``fr_so`` 为向下短波（>= 0）；``fr_th`` 为*净*长波，
  向下为正。测试套件默认的 ``-300*3600`` 是净辐射*损失*（向上），因此
  生成器的净长波在典型夜-日循环中应得出负值——事实如此（见
  :func:`_net_longwave`）。

静态描述符使用规范的**纯陆地**瓦片份额
（``tests/test_forward_m5._make_inputs``）：湿皮肤 0.05 / 低植被-干
0.30 / 高植被-干 0.30 / 裸土 0.35，不含水/冰/雪瓦片。这使每一列都是
真正的陆地列，HTESSEL 土壤求解才会真正触发，水量/能量收支才有意义。
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
# Solar ephemeris (IFS fctast/cpg1s algorithm)
# --------------------------------------------------------------------------- #
_IFS_RDAY = 86400.0
_IFS_REA = 1.49597870e11
_IFS_REPSM = 0.409093
_IFS_JD_EPOCH = 2446432.0
_IFS_REF_YEAR = 2022


def _ifs_jd_from_calendar(year: int, month: int, day: int) -> float:
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return (day + (153 * m + 2) // 5 + 365 * y + y // 4
            - y // 100 + y // 400 - 32045)


def _ifs_doy_to_jd(year: int, day_of_year: int, hour_utc: float) -> float:
    import calendar
    month = 1
    remaining = day_of_year
    while remaining > calendar.monthrange(year, month)[1]:
        remaining -= calendar.monthrange(year, month)[1]
        month += 1
    return _ifs_jd_from_calendar(year, month, remaining) + hour_utc / 24.0


def _ifs_astronomy(rtime_seconds: float):
    pteta = rtime_seconds / (_IFS_RDAY * 365.25)
    rel = 1.7535 + 6.283076 * pteta
    rem = 6.240075 + 6.283020 * pteta
    rlls = 4.8951 + 6.283076 * pteta
    rllls = (4.8952 + 6.283320 * pteta - 0.0075 * math.sin(rel)
             - 0.0326 * math.cos(rel) - 0.0003 * math.sin(2.0 * rel)
             + 0.0002 * math.cos(2.0 * rel))
    decl = math.asin(math.sin(_IFS_REPSM) * math.sin(rllls))
    eqt = (591.8 * math.sin(2.0 * rlls) - 459.4 * math.sin(rem)
           + 39.5 * math.sin(rem) * math.cos(2.0 * rlls)
           - 12.7 * math.sin(4.0 * rlls) - 4.8 * math.sin(2.0 * rem))
    rrs = _IFS_REA * (1.0001 - 0.0163 * math.sin(rel)
                      + 0.0037 * math.cos(rel))
    rip0 = 1370.0 * _IFS_REA * _IFS_REA / (rrs * rrs)
    angle = (eqt + rtime_seconds % _IFS_RDAY) * 2.0 * math.pi / _IFS_RDAY
    return math.sin(decl), math.cos(decl), math.cos(angle), math.sin(angle), rip0


def _ifs_mu0_from_rtime(lat_rad: float, lon_rad: float,
                        rtime_seconds: float) -> float:
    sin_dec, cos_dec, cos_ha, sin_ha, _ = _ifs_astronomy(rtime_seconds)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.sqrt(1.0 - sin_lat * sin_lat)
    mu0 = (sin_dec * sin_lat - cos_dec * cos_ha * cos_lat * math.cos(lon_rad)
           + cos_dec * sin_ha * cos_lat * math.sin(lon_rad))
    return max(0.0, min(1.0, mu0))


def ifs_mu0_from_datetime(lat_rad: float, lon_rad: float,
                          year: int, month: int, day: int,
                          hour_utc: float) -> float:
    jd = _ifs_jd_from_calendar(year, month, day) + hour_utc / 24.0
    return _ifs_mu0_from_rtime(lat_rad, lon_rad,
                               (jd - _IFS_JD_EPOCH) * _IFS_RDAY)


def solar_mu0(lat_rad: float, lon_rad: float, day_of_year: int,
              hour_utc: float) -> float:
    """Solar zenith-angle cosine ``mu0`` (>= 0, 0 at night).

    Legacy day-of-year wrapper around the IFS absolute-time ephemeris.

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
    jd = _ifs_doy_to_jd(_IFS_REF_YEAR, day_of_year, hour_utc)
    return _ifs_mu0_from_rtime(lat_rad, lon_rad,
                               (jd - _IFS_JD_EPOCH) * _IFS_RDAY)


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
    sigma = float(bundle.rsigma_drv.item())
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
