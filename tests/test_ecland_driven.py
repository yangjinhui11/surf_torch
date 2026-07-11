"""Real-data health-regression driven by ecLand ERA5 site forcing.

The ``surf_pytorch`` test suite is built around *idealized* diurnal-
cycle forcing (see :mod:`surf_pytorch.forcing`) and *synthetic*
single-column references assembled from the per-kernel Fortran ports.
This module adds the complementary validation axis: drive the model
with **real** ERA5 near-surface forcing from the ecLand tutorial
dataset and assert physical plausibility over a multi-step run.

Data source
-----------
``thirdpaty/ecland-main/tutorials/ecland_input_tutorial/{forcing,clim}/
LS-PRACTICALS/*_<SITE>_2022-2022.nc`` -- the same cabauw (NL) station
sliced into two periods:

* ``CA-001`` : 25 hourly steps, mid-January winter (cold, low SW).
* ``VG-000`` : 43 hourly steps, early spring.

Both files are ERA5-Hourly-Tiled forcing (Tair / Qair / wind / SWdown /
LWdown / Rainf / Snowf / PSurf) plus the matching ``surfclim`` static
descriptor and ``surfinit`` initial state.

Why not a golden-file comparison
--------------------------------
The ecLand Fortran model is **not built** in this repository, so we
cannot reproduce its per-timestep output to do a ULP-level comparison.
Instead we run a *physical-health* regression: the surf_pytorch port
must advance the real forcing without NaNs, keep state within physical
bounds, close the surface energy / water budget, and produce a
distinguishable-but-comparable Jarvis vs Farquhar skin-T response.

Unit / sign conventions (verified against the port source)
----------------------------------------------------------
* ``Forcing`` radiation / precip are **cumulative** over the coupling
  interval (J/m^2 and kg/m^2) -- the ecLand ``SWdown`` / ``LWdown`` /
  ``Rainf`` / ``Snowf`` are *rates* (W/m^2, kg/m^2/s), so they are
  multiplied by ``dt`` here.
* ``Forcing.fr_th`` is the **net** long-wave flux (positive = downward,
  typically negative = surface cooling), NOT gross downwelling.  The
  SEB kernel consumes ``slrfl = fr_th/dt`` directly as the linearised
  net LW (``surf_pytorch/physics/seb.py:367``).  ecLand's ``LWdown`` is
  gross downwelling, so the net is reconstructed as
  ``LWdown - emis*sigma*AvgSurfT**4`` -- exactly the form the idealized
  generator uses (``forcing._net_longwave``) and the form upstream uses
  for its first radiation guess.
* ``Static.soil_type`` / ``veg_type`` carry the **raw 1-based ecLand
  type index** (sotype=3, tvl=1).  The port's ``VegTable.gather`` /
  ``SoilTable`` index the (21,) / (8,) LUTs directly with this value
  (index 0 is a water sentinel that mirrors type 8) -- no ``-1`` offset.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# netCDF stack is available in this environment; skip the whole module
# gracefully if it ever disappears (e.g. minimal CI image).
xarray = pytest.importorskip("xarray")
import numpy as np   # noqa: E402  (import after the skip guard)
import torch         # noqa: E402

from surf_pytorch.constants import ConstantBundle, PhysicalConstants  # noqa: E402
from surf_pytorch.config import SurfConfig, N_BASE_TILES  # noqa: E402
from surf_pytorch.state import Forcing, Static, SurfState  # noqa: E402
from surf_pytorch.forcing import solar_mu0  # noqa: E402
from surf_pytorch.model import LandSurface  # noqa: E402


# --------------------------------------------------------------------------- #
# Data-root resolution
# --------------------------------------------------------------------------- #
# Walk up from this test file to find the ecland tutorial directory.
# Resolves to <repo>/thirdpaty/ecland-main/tutorials/ecland_input_tutorial.
def _ecland_root() -> Path:
    here = Path(__file__).resolve()
    # test file: <repo>/surf_pytorch/tests/test_ecland_driven.py
    repo = here.parents[2]
    return repo / "thirdpaty" / "ecland-main" / "tutorials" / "ecland_input_tutorial"


_ECLAND_ROOT = _ecland_root()

_SITES = ("CA-001", "VG-000")


def _files_for(site: str) -> Dict[str, Path]:
    """Return the met / surfclim / surfinit netCDF paths for ``site``."""
    fdir = _ECLAND_ROOT / "forcing" / "LS-PRACTICALS"
    cdir = _ECLAND_ROOT / "clim" / "LS-PRACTICALS"
    return {
        "met": fdir / f"met_era5HT_{site}_2022-2022.nc",
        "clim": cdir / f"surfclim_{site}_2022-2022.nc",
        "init": cdir / f"surfinit_{site}_2022-2022.nc",
    }


# Skip the whole module if the ecland tutorial data is absent.
_missing = [s for s in _SITES if not all(p.exists() for p in _files_for(s).values())]
if _missing:
    pytest.skip(
        f"ecLand tutorial data not found for sites {_missing} under "
        f"{_ECLAND_ROOT}; skipping real-data regression.",
        allow_module_level=True,
    )


# --------------------------------------------------------------------------- #
# Site loader: netCDF -> raw arrays
# --------------------------------------------------------------------------- #
def _load_site(site: str) -> Dict[str, object]:
    """Open the three ecLand netCDFs for ``site`` and return raw arrays.

    Returns a dict with keys:
        * ``forcing`` : dict of 1-D (nstep,) numpy arrays (rates).
        * ``clim``    : dict of scalar / monthly numpy arrays (static).
        * ``init``    : dict of initial-state arrays.
        * ``time``    : (nstep,) datetime64 array (mid-timestep stamps).
        * ``lat`` / ``lon`` : site coords (degrees).
    """
    paths = _files_for(site)
    met = xarray.open_dataset(paths["met"])
    cli = xarray.open_dataset(paths["clim"])
    ini = xarray.open_dataset(paths["init"])

    def _squeeze(arr):
        return np.asarray(arr.values).ravel()

    forcing = {v: _squeeze(met[v]) for v in
               ["Tair", "Qair", "Wind_E", "Wind_N", "PSurf",
                "SWdown", "LWdown", "Rainf", "Snowf"]}

    time = met["time"].values if "time" in met else None

    # Static scalars (single value per site).
    lat = float(np.asarray(cli["lat"].values).ravel()[0])
    lon = float(np.asarray(cli["lon"].values).ravel()[0])
    clim = {
        "lat": lat, "lon": lon,
        "tvl": int(np.asarray(cli["tvl"].values).ravel()[0]),
        "tvh": int(np.asarray(cli["tvh"].values).ravel()[0]),
        "sotype": int(np.asarray(cli["sotype"].values).ravel()[0]),
        "cvl": float(np.asarray(cli["cvl"].values).ravel()[0]),
        "cvh": float(np.asarray(cli["cvh"].values).ravel()[0]),
        "cu": float(np.asarray(cli["cu"].values).ravel()[0]),
        "landsea": float(np.asarray(cli["landsea"].values).ravel()[0]),
        "geopot": float(np.asarray(cli["geopot"].values).ravel()[0]),
        "z0m": float(np.asarray(cli["z0m"].values).ravel()[0]),
        "lz0h": float(np.asarray(cli["lz0h"].values).ravel()[0]),
        "zuv": float(np.asarray(cli["zuv"].values).ravel()[0]),
        "zphista": float(np.asarray(cli["zphista"].values).ravel()[0]),
    }

    init = {
        "SoilTemp": np.asarray(ini["SoilTemp"].values).ravel(),
        "SoilMoist": np.asarray(ini["SoilMoist"].values).ravel(),
        "AvgSurfT": float(np.asarray(ini["AvgSurfT"].values).ravel()[0]),
        "CanopInt": float(np.asarray(ini["CanopInt"].values).ravel()[0]),
        "SWE": float(np.asarray(ini["SWE"].values).ravel()[0]),
    }
    return {"forcing": forcing, "clim": clim, "init": init,
            "time": time, "lat": lat, "lon": lon}


# --------------------------------------------------------------------------- #
# Mapping: ecLand raw -> surf_pytorch Forcing / Static / SurfState
# --------------------------------------------------------------------------- #
# Coupling interval: ecLand tutorial forcing is hourly.  The port's
# Forcing carries cumulative fluxes, so radiation (W/m^2) and precip
# (kg/m^2/s) are multiplied by this dt.
_DT_SECONDS = 3600.0

# CO2 is absent from the ecLand tutorial met files; use the port's
# standard kg/kg mixing ratio (forcing.py IdealizedDiurnalConfig default).
_CO2 = 6.0e-4

# Standard interception-reservoir (wet-skin) tile fraction, matching the
# idealized generator and the canonical test default.
_WET_SKIN_FRAC = 0.05


def _tile_fractions(cvl: float, cvh: float, cu: float,
                    landsea: float, ntile: int = N_BASE_TILES) -> np.ndarray:
    """Allocate the 8 base tile fractions from ecLand cover fields.

    ecLand provides aggregate cover fractions: ``cvl`` (low veg),
    ``cvh`` (high veg), ``cu`` (urban) and ``landsea`` (land fraction).
    The port does not yet split these into the IFS 8-tile scheme
    internally (it is a documented M5 milestone placeholder), so we
    build a pure-land allocation here:

    * tile 2 (wet_skin)  : fixed interception fraction.
    * tile 3 (low_veg)   : ``cvl`` scaled onto the residual land.
    * tile 5 (high_veg)  : ``cvh`` scaled onto the residual land.
    * tile 7 (bare_soil) : the remaining land (incl. ``cu`` for this
      cold-start approximation, since the urban tile is not active).

    Fractions sum to ``landsea``.  This mirrors the idealized generator's
    pure-land convention (``forcing._build_static_and_state``).
    """
    frac = np.zeros(ntile, dtype=np.float64)
    land = float(landsea)
    wet = _WET_SKIN_FRAC * land
    # Low / high veg cover scaled onto the non-wet-skin, non-urban land.
    veg_total = max(cvl + cvh, 1e-12)
    urban = cu * land                            # urban fraction folded into
    scale = (land - wet - urban) / veg_total if veg_total > 1e-12 else 0.0
    low = cvl * scale
    high = cvh * scale
    # Bare-soil tile absorbs the residual land incl. urban cover (the
    # port's urban tile is not active in this 8-tile cold-start).
    bare = land - wet - low - high
    frac[2] = wet
    frac[3] = low
    frac[5] = high
    frac[7] = bare
    return frac


def _to_inputs(raw: Dict, consts: ConstantBundle,
               dtype: torch.dtype) -> Tuple[List[Forcing], Static, SurfState]:
    """Map ecLand raw arrays to port (forcing_list, static, state0)."""
    f = raw["forcing"]
    c = raw["clim"]
    i = raw["init"]
    n = f["Tair"].size

    g = float(consts.g.item())
    sigma = float(consts.rsigma.item())
    emis = 0.99                          # surface emissivity (idealized default)

    dt_t = torch.tensor(_DT_SECONDS, dtype=dtype)
    lat_rad = math.radians(c["lat"])
    lon_rad = math.radians(c["lon"])

    # Static (constant) per-column tensors -- single column (n_grid=1).
    tile_frac = _tile_fractions(c["cvl"], c["cvh"], c["cu"], c["landsea"])
    static = Static(
        tile_frac_base=torch.tensor(tile_frac, dtype=dtype).unsqueeze(0),
        land_frac=torch.tensor([c["landsea"]], dtype=dtype),
        orography=torch.tensor([c["geopot"]], dtype=dtype),
        soil_type=torch.tensor([c["sotype"]], dtype=torch.long),
        veg_type=torch.tensor([c["tvl"]], dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.tensor([emis], dtype=dtype),
        z0m=torch.tensor([c["z0m"]], dtype=dtype),
        z0h=torch.tensor([math.exp(c["lz0h"])], dtype=dtype),
        z0q=torch.tensor([math.exp(c["lz0h"])], dtype=dtype),
        lat=torch.tensor([lat_rad], dtype=dtype),
        lon=torch.tensor([lon_rad], dtype=dtype),
    )

    # Initial state.
    ntile = N_BASE_TILES
    nsoil = 4
    state0 = SurfState(
        skin_t=torch.full((1, ntile), i["AvgSurfT"], dtype=dtype),
        soil_t=torch.tensor(i["SoilTemp"], dtype=dtype).unsqueeze(0),
        soil_w_liq=torch.tensor(i["SoilMoist"], dtype=dtype).unsqueeze(0),
        snow_w=torch.tensor([i["SWE"]], dtype=dtype),
        snow_frac=torch.zeros(1, dtype=dtype),
        skin_water=torch.tensor([i["CanopInt"]], dtype=dtype),
        sst=torch.tensor([i["AvgSurfT"]], dtype=dtype),
    )

    forcing_list: List[Forcing] = []
    for k in range(n):
        # Radiation: rates -> cumulative, with net-LW reconstruction.
        sw_down = float(f["SWdown"][k])                       # W/m^2 downwelling
        lw_down = float(f["LWdown"][k])                       # W/m^2 downwelling
        lw_up = emis * sigma * i["AvgSurfT"] ** 4             # surface emission
        net_lw = lw_down - lw_up                              # net (down-positive)

        # mu0 from the calendar stamp (mid-timestep).
        tstamp = raw["time"][k]
        day_of_year = int(np.asarray(tstamp).astype("datetime64[D]")
                          .astype(int) % 365) + 1
        hour_utc = (np.asarray(tstamp).astype("datetime64[h]")
                    .astype(int) % 24) + 0.5
        mu0 = solar_mu0(lat_rad, lon_rad, day_of_year, float(hour_utc))

        forcing_list.append(Forcing(
            u=torch.tensor([float(f["Wind_E"][k])], dtype=dtype),
            v=torch.tensor([float(f["Wind_N"][k])], dtype=dtype),
            t=torch.tensor([float(f["Tair"][k])], dtype=dtype),
            q=torch.tensor([float(f["Qair"][k])], dtype=dtype),
            co2=torch.tensor([_CO2], dtype=dtype),
            p_surf=torch.tensor([float(f["PSurf"][k])], dtype=dtype),
            geom1=torch.tensor([g * c["zphista"]], dtype=dtype),
            fr_so=torch.tensor([sw_down * _DT_SECONDS], dtype=dtype),
            fr_th=torch.tensor([net_lw * _DT_SECONDS], dtype=dtype),
            mu0=torch.tensor([mu0], dtype=dtype),
            rain=torch.tensor([float(f["Rainf"][k]) * _DT_SECONDS], dtype=dtype),
            snow=torch.tensor([float(f["Snowf"][k]) * _DT_SECONDS], dtype=dtype),
            dt=dt_t.clone(),
        ))

    return forcing_list, static, state0


# --------------------------------------------------------------------------- #
# Shared fixture: loaded + mapped inputs for a site
# --------------------------------------------------------------------------- #
def _build_bundle(site: str):
    raw = _load_site(site)
    consts = ConstantBundle(PhysicalConstants(), dtype=torch.float64)
    forcing_list, static, state0 = _to_inputs(raw, consts, torch.float64)
    return raw, consts, forcing_list, static, state0


def _grid_mean(tile_field: torch.Tensor, frac: torch.Tensor) -> float:
    """Fraction-weighted grid mean of a (Ngrid, Ntile) field.

    Inactive zero-fraction tiles (water / ice / snow tiles, which carry
    no mass in this cold-start and may drift numerically) are correctly
    dropped by the weighting -- this mirrors the upstream tile-averaging
    in ``vdfmain1s`` (the per-tile fluxes / state are summed with
    ``PFRTI`` to form the grid mean).
    """
    return float((tile_field * frac).sum().item())


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("site", _SITES)
def test_ecland_loads_and_maps(site):
    """NetCDF load + field mapping produces well-formed, physical inputs."""
    raw, consts, forcing_list, static, state0 = _build_bundle(site)

    # Forcing list length matches the file.
    assert len(forcing_list) == raw["forcing"]["Tair"].size

    # Static fields are physical.
    assert static.tile_frac_base.shape == (1, N_BASE_TILES)
    fr = static.tile_frac_base[0]
    assert torch.all(fr >= 0), f"negative tile fraction in {site}: {fr}"
    assert torch.allclose(fr.sum(), static.land_frac[0], atol=1e-9)
    # Land-only site: water / ice / snow tiles are zero.
    assert fr[0].item() == 0.0 and fr[1].item() == 0.0

    # Initial state ranges.
    assert 200.0 < state0.skin_t.mean().item() < 330.0
    assert torch.all(state0.soil_t > 200.0) and torch.all(state0.soil_t < 330.0)
    assert torch.all(state0.soil_w_liq >= 0) and torch.all(state0.soil_w_liq < 0.5)

    # Forcing: no NaN / Inf, radiation sign correct (SW >= 0; net LW
    # typically negative = cooling at these winter/spring sites).
    for fk in forcing_list:
        for fld in ["u", "v", "t", "q", "p_surf", "fr_so", "fr_th",
                    "rain", "snow", "mu0"]:
            t = getattr(fk, fld)
            assert torch.isfinite(t).all(), f"non-finite {fld} in {site}"
        assert fk.fr_so.item() >= 0, f"negative SW in {site}"
    assert any(fk.mu0.item() > 0 for fk in forcing_list), \
        f"no daylight in {site} forcing (mu0 all zero)"


@pytest.mark.parametrize("site", _SITES)
def test_ecland_multistep_runs_jarvis(site):
    """Full multi-step advance with Jarvis closure stays physical."""
    raw, consts, forcing_list, static, state0 = _build_bundle(site)

    cfg = SurfConfig(dtype=torch.float64, use_farquhar=False, validate=True)
    model = LandSurface(cfg)
    frac = static.tile_frac_base                          # (1, ntile)

    state = state0
    skin_t_history = [_grid_mean(state.skin_t, frac)]
    for k, forcing in enumerate(forcing_list):
        new_state, diag = model(state, forcing, static)
        assert torch.isfinite(new_state.skin_t).all(), \
            f"NaN skin_t at step {k} in {site}"
        assert torch.isfinite(new_state.soil_t).all(), \
            f"NaN soil_t at step {k} in {site}"
        assert torch.isfinite(new_state.soil_w_liq).all(), \
            f"NaN soil_w at step {k} in {site}"
        # Soil moisture stays in [0, saturation].
        assert torch.all(new_state.soil_w_liq >= 0) and \
            torch.all(new_state.soil_w_liq < 0.5), \
            f"soil_w out of [0,0.5] at step {k} in {site}"
        # Skin-T: fraction-weighted grid mean must stay in a generous
        # physical envelope.  (Inactive zero-fraction tiles -- snow /
        # water -- may drift since they carry no mass and are dropped
        # by the grid mean, mirroring the upstream tile-averaging.)
        gm = _grid_mean(new_state.skin_t, frac)
        assert 200.0 < gm < 330.0, \
            f"grid-mean skin_t out of [200,330]K at step {k} in {site}: {gm:.2f}"
        state = new_state
        skin_t_history.append(_grid_mean(state.skin_t, frac))

    # State actually evolves (not frozen); the run sees a real diurnal range.
    skin_t_history = np.asarray(skin_t_history)
    assert np.ptp(skin_t_history) > 0.1, \
        f"skin_t did not evolve in {site} (range={np.ptp(skin_t_history):.3f}K)"


@pytest.mark.parametrize("site", _SITES)
def test_ecland_multistep_runs_farquhar(site):
    """Farquhar closure advances the real forcing without NaNs."""
    raw, consts, forcing_list, static, state0 = _build_bundle(site)

    cfg = SurfConfig(dtype=torch.float64, use_farquhar=True, validate=True)
    model = LandSurface(cfg)
    frac = static.tile_frac_base

    state = state0
    for k, forcing in enumerate(forcing_list):
        new_state, diag = model(state, forcing, static)
        assert torch.isfinite(new_state.skin_t).all(), \
            f"NaN skin_t at step {k} in {site} (Farquhar)"
        assert torch.isfinite(new_state.soil_w_liq).all(), \
            f"NaN soil_w at step {k} in {site} (Farquhar)"
        assert torch.all(new_state.soil_w_liq >= 0) and \
            torch.all(new_state.soil_w_liq < 0.5)
        gm = _grid_mean(new_state.skin_t, frac)
        assert 200.0 < gm < 330.0, \
            f"grid-mean skin_t out of [200,330]K at step {k} in {site} (Farquhar): {gm:.2f}"
        state = new_state


@pytest.mark.parametrize("site", _SITES)
def test_ecland_jarvis_vs_farquhar_distinguishable(site):
    """Jarvis and Farquhar produce comparable-but-distinct skin-T."""
    raw, consts, forcing_list, static, state0 = _build_bundle(site)

    def _run(use_farquhar: bool) -> torch.Tensor:
        cfg = SurfConfig(dtype=torch.float64,
                         use_farquhar=use_farquhar, validate=True)
        model = LandSurface(cfg)
        frac = static.tile_frac_base
        state = state0
        history = [_grid_mean(state.skin_t, frac)]
        for forcing in forcing_list:
            state, _ = model(state, forcing, static)
            history.append(_grid_mean(state.skin_t, frac))
        return torch.tensor(history)

    h_jar = _run(False)
    h_far = _run(True)

    delta = (h_far - h_jar).abs()
    # Distinguishable: at least one step differs beyond numerical noise.
    assert delta.max().item() > 1.0e-3, \
        f"Jarvis vs Farquhar identical in {site} (max dT={delta.max():.2e}K)"
    # Comparable: differences stay within a single order-of-magnitude band
    # (both close the same SEB; stomatal-resistance swap should not flip
    # the skin-T regime).
    assert delta.max().item() < 10.0, \
        f"Jarvis vs Farquhar diverge too far in {site} " \
        f"(max dT={delta.max():.2f}K)"


@pytest.mark.parametrize("site", _SITES)
def test_ecland_energy_water_budget(site):
    """Single-step surface energy and water budgets close.

    Surface energy balance (per ``surf_pytorch/physics/seb.py`` docstring,
    all fluxes positive = downward into the surface)::

        PSSRFL + PSTR + PSSH + PSLH - PG0 = 0
        SW_down + net_LW + H     + LE    - G    = 0

    where ``SW_down`` is the *gross* downwelling SW (the kernel applies
    the surface albedo internally via the SW-transmission term), and the
    per-tile fluxes are summed with the tile fractions to the grid mean.
    The qsat-linearisation closure leaves a small (~few W/m^2) residual,
    which we bound relative to the forcing scale.
    """
    raw, consts, forcing_list, static, state0 = _build_bundle(site)

    cfg = SurfConfig(dtype=torch.float64, use_farquhar=False, validate=True)
    model = LandSurface(cfg)
    frac = static.tile_frac_base                              # (1, ntile)

    # Use a daytime step (mu0 > 0) for a non-trivial energy budget.
    k_day = next((k for k, fk in enumerate(forcing_list)
                  if fk.mu0.item() > 0.01), 0)
    forcing = forcing_list[k_day]
    state = state0

    new_state, diag = model(state, forcing, static)
    dt = float(forcing.dt.item())

    # ---- Energy budget (W/m^2) ----
    sw_down = forcing.fr_so.item() / dt                       # gross downwelling
    net_lw = _grid_mean(diag.str_tile, frac)                  # net LW (PSTR)
    h_flux = float(diag.h_flux.mean().item())                 # sensible (PSSH)
    le = _grid_mean(diag.slh_tile, frac)                      # latent (PSLH)
    g0 = _grid_mean(diag.g0_tile, frac)                       # ground (PG0)

    for name, val in [("sw_down", sw_down), ("net_lw", net_lw),
                      ("h_flux", h_flux), ("le", le), ("g0", g0)]:
        assert math.isfinite(val), f"non-finite {name} in {site}"

    # SEB residual: SW_down + net_LW + H + LE - G  ~=  0.
    seb_residual = sw_down + net_lw + h_flux + le - g0
    scale = max(abs(sw_down), abs(net_lw), 50.0)
    assert abs(seb_residual) < 0.1 * scale, (
        f"energy budget residual too large in {site} at step {k_day}: "
        f"SW={sw_down:.2f} netLW={net_lw:.2f} H={h_flux:.2f} "
        f"LE={le:.2f} G={g0:.2f} residual={seb_residual:.2f}")

    # ---- Water budget (kg/m^2 over the step) ----
    # precip = evap + runoff + d(soil_water)  [+ interception clamp term]
    e_flux = float(diag.e_flux.mean().item())                 # kg/m^2/s
    precip = forcing.rain.item() + forcing.snow.item()
    evap = e_flux * dt
    runoff = (float(diag.sub_flux.mean().item()) * dt
              if diag.sub_flux is not None else 0.0)
    # d(soil water) in kg/m^2 = d(volume fraction) * soil depth * rho_water.
    rho_w = float(consts.rho_h2o.item())
    dvol = float((new_state.soil_w_liq - state.soil_w_liq).mean().item())
    dsoilw = dvol * float(static.soil_thickness.sum().item()) * rho_w
    water_residual = precip - evap - runoff - dsoilw
    assert math.isfinite(water_residual), f"non-finite water budget in {site}"
    # These sites are dry (no precip), so the column slowly dries via
    # evaporation; the budget should close to within ~1 mm equivalent
    # (interception-clamp / snow-reservoir mass exchange is zero here).
    assert abs(water_residual) < 1.0, (
        f"water budget residual too large in {site}: "
        f"precip={precip:.4f} evap={evap:.4f} runoff={runoff:.4f} "
        f"dsoilw={dsoilw:.4f} residual={water_residual:.4f}")


@pytest.mark.parametrize("site", _SITES)
def test_ecland_differentiable(site):
    """The model backprops cleanly through one real-forcing step."""
    raw, consts, forcing_list, static, state0 = _build_bundle(site)

    cfg = SurfConfig(dtype=torch.float64, use_farquhar=False,
                     differentiable=True, validate=False)
    model = LandSurface(cfg)

    # Make the initial skin-T a leaf that requires grad.
    state = SurfState(**{**state0.__dict__,
                         "skin_t": state0.skin_t.clone().requires_grad_(True)})
    forcing = forcing_list[0]

    new_state, _ = model(state, forcing, static)
    loss = new_state.skin_t.sum()
    loss.backward()

    assert state.skin_t.grad is not None, \
        f"no gradient propagated to skin_t in {site}"
    assert torch.isfinite(state.skin_t.grad).all(), \
        f"non-finite skin_t gradient in {site}"
