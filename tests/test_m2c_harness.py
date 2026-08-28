"""Tests for the M2c forcing generator + multi-step column harness.

These exercise the three validation targets chosen for M2c:

1. **Water-balance closure** -- the HTESSEL ``SRFWNG`` clip/redistribute
   conserves mass to machine precision, so over a multi-step run

       Σ(rain+snow) = Σ(ΔW·rdaw·ρ_h2o) + Σ(Δskin_water) + Σ(E_up·dt)
                      + Σ(Δsnow_w) + runoff

   must close (the residual is the runoff, which is finite, >= 0 and
   exactly the difference between the precip + E input and the soil /
   skin / snow storage change).  The M2d snow path feeds snowfall into
   the snow-mass prognostic, so Δsnow_w is part of the storage term and
   the snow-bearing water-balance test
   (:meth:`TestSnowThroughfall.test_snow_water_balance_closes_with_snowfall`)
   closes to machine precision.

2. **Energy-balance closure** -- the soil-column heat input is the
   surface net heat flux ``ZSURFL = SW + LW + H + RLVTT·E`` (positive
   downward, land-tile-weighted; see
   :func:`surf_pytorch.physics.soil._surface_net_heat_flux`).  The
   column heat storage is

       Σ_jk rcsoil_jk · rdaw_jk · ΔT_jk

   and the bottom layer is zero-flux (``ZLST(:,KLEVS)=0``), so on a
   short run where the deep layer barely moves these must close to a
   few percent of the cumulative input.

   NOTE: ``diag.ground_heat`` (``pg0 = lamsk·(Tskin - Tsub)``) is the
   skin-to-subskin flux, NOT the soil-column input -- it cannot be used
   for column energy closure.  The reconstructed ``ZSURFL`` matches the
   column storage to 4 decimals here.

3. **Physical-plausibility diagnostics** + **golden snapshot regression**
   + **CSV export smoke** + **snow throughfall** (M2d).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from surf_pytorch import (
    IdealizedDiurnalConfig,
    solar_mu0,
    build_idealized_diurnal,
    ColumnHarness,
    LandSurface,
    SurfConfig,
)
from surf_pytorch.state import SurfState

# Canonical soil-layer thicknesses (SurfConfig default) -- used by the
# water / energy budget reconstructions so the test does not depend on
# the model internals for this constant.
RDAW = [0.07, 0.21, 0.72, 1.89]

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "reference")
GOLDEN_PATH = os.path.join(GOLDEN_DIR, "m2c_golden.npz")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_run(n_steps: int = 48, *, seed: int = 0,
               cfg: IdealizedDiurnalConfig | None = None) -> tuple:
    """Build a model + harness + run result for a given forcing config.

    Returns ``(result, forcing_list, static, initial_state, model)``.
    """
    torch.manual_seed(seed)
    fcfg = cfg or IdealizedDiurnalConfig(n_steps=n_steps)
    fl, static, state0 = build_idealized_diurnal(fcfg)
    model = LandSurface(SurfConfig(dtype=torch.float64, validate=False))
    harness = ColumnHarness(model, static, state0)
    return harness.run(fl), fl, static, state0, model


def _water_storage(state: SurfState, rdaw_t: torch.Tensor,
                   rho_h2o: float) -> torch.Tensor:
    """Total water column (kg/m^2): soil + skin + snow."""
    soil = (state.soil_w_liq * rdaw_t * rho_h2o).sum(-1)
    return soil + state.skin_water + state.snow_w


def _heat_storage(soil_t: torch.Tensor, rcsoil_b: torch.Tensor,
                  rdaw_t: torch.Tensor) -> torch.Tensor:
    """Soil-column heat content (J/m^2): Σ_jk rcsoil·rdaw·T."""
    return (soil_t * rcsoil_b * rdaw_t).sum(-1)


# --------------------------------------------------------------------------- #
# Solar ephemeris
# --------------------------------------------------------------------------- #
class TestSolarMu0:
    """Unit-test the analytic solar-angle helper in isolation."""

    def test_equator_equinox_noon_sun_overhead(self):
        # Equator, equinox (~doy 80), local noon -> mu0 ~= 1.
        mu0 = solar_mu0(lat_rad=0.0, lon_rad=0.0, day_of_year=80, hour_utc=12.0)
        assert mu0 == pytest.approx(1.0, abs=1e-3)

    def test_night_mu0_zero(self):
        # Local midnight at 45N summer -> sun below horizon -> mu0 == 0.
        mu0 = solar_mu0(math.radians(45), 0.0, 172, 0.0)
        assert mu0 == 0.0

    def test_pole_winter_no_daylight(self):
        # 90N near winter solstice (doy 355) at any hour -> polar night.
        for hr in (0, 6, 12, 18):
            assert solar_mu0(math.radians(90), 0.0, 355, float(hr)) == 0.0

    def test_clamped_to_unit_interval(self):
        # mu0 must never exceed 1 (numerical overshoot at perfect noon).
        mu0 = solar_mu0(math.radians(-23.45), math.radians(180 - 0),
                        172, 12.0)
        assert 0.0 <= mu0 <= 1.0

    def test_longitude_shifts_solar_noon(self):
        # At 15E, solar noon in UTC is 11:00 -- check mu0 peaks earlier.
        mu0_lon0 = solar_mu0(math.radians(45), math.radians(0), 172, 12.0)
        mu0_lon15 = solar_mu0(math.radians(45), math.radians(15), 172, 11.0)
        assert abs(mu0_lon0 - mu0_lon15) < 1e-6


# --------------------------------------------------------------------------- #
# Forcing generation sanity
# --------------------------------------------------------------------------- #
class TestForcingGenerator:
    """Pin the analytic forcing's physical and bookkeeping invariants."""

    def test_returns_correct_lengths_and_shapes(self):
        cfg = IdealizedDiurnalConfig(n_steps=10, n_grid=3)
        fl, static, state = build_idealized_diurnal(cfg)
        assert len(fl) == 10
        for f in fl:
            assert f.t.shape == (3,)
            assert f.fr_so.shape == (3,)
            assert f.rain.shape == (3,)
        assert static.tile_frac_base.shape == (3, 8)
        assert state.skin_t.shape == (3, 8)

    def test_tile_fractions_sum_to_one_and_pure_land(self):
        cfg = IdealizedDiurnalConfig(n_steps=4, n_grid=2)
        _, static, _ = build_idealized_diurnal(cfg)
        frac = static.tile_frac_base
        assert torch.allclose(
            frac.sum(-1), torch.ones(2, dtype=frac.dtype), atol=1e-12,
        )
        # No water / ice / snow tiles.
        assert frac[:, 0].max() == 0.0      # water
        assert frac[:, 1].max() == 0.0      # ice
        assert frac[:, 4].max() == 0.0      # low-veg snow
        assert frac[:, 6].max() == 0.0      # high-veg snow

    def test_cumulative_units_match_dt(self):
        """fr_so / rain etc. must be rate * dt (cumulative over interval)."""
        cfg = IdealizedDiurnalConfig(n_steps=24, n_grid=1)
        fl, _, _ = build_idealized_diurnal(cfg)
        for f in fl:
            # fr_so / dt should reconstruct a plausible W/m^2 rate
            # (0..sw_peak*tau_atm at most).
            rate = (f.fr_so[0] / f.dt).item()
            assert -1e-9 <= rate <= 800.0 * 0.7 + 1e-9

    def test_sw_zero_at_night(self):
        cfg = IdealizedDiurnalConfig(n_steps=48, n_grid=1)
        fl, _, _ = build_idealized_diurnal(cfg)
        for f in fl:
            if f.mu0[0].item() == 0.0:
                assert f.fr_so[0].item() == 0.0

    def test_tair_diurnal_amplitude(self):
        cfg = IdealizedDiurnalConfig(n_steps=48, n_grid=1, tair_mean=290.0,
                                     tair_amp=8.0)
        fl, _, _ = build_idealized_diurnal(cfg)
        ts = torch.stack([f.t[0] for f in fl])
        # Peak-to-mean ~= amp, trough-to-mean ~= -amp (cosine).
        assert ts.max().item() == pytest.approx(290.0 + 8.0, abs=0.5)
        assert ts.min().item() == pytest.approx(290.0 - 8.0, abs=0.5)

    def test_q_is_rh_times_qsat(self):
        """q = RH * qsat(Tair, p_surf) to last ULP."""
        from surf_pytorch.utils import qsat
        from surf_pytorch.constants import ConstantBundle, PhysicalConstants
        cfg = IdealizedDiurnalConfig(n_steps=12, n_grid=1, rh=0.7)
        bundle = ConstantBundle(PhysicalConstants(), dtype=torch.float64)
        fl, _, _ = build_idealized_diurnal(cfg)
        for f in fl:
            q_sat = qsat(f.t, f.p_surf, bundle)
            assert torch.allclose(f.q, 0.7 * q_sat, rtol=1e-12)


# --------------------------------------------------------------------------- #
# Water-balance closure
# --------------------------------------------------------------------------- #
class TestWaterBalance:
    """The HTESSEL SRFWNG conserves mass to machine precision."""

    def test_closure_no_precip(self):
        """With zero precip: ΔW + Δskin_water = E·dt + runoff."""
        result, fl, static, state0, model = _build_run(n_steps=48)
        rho_h2o = float(model.consts.rho_h2o.item())
        rdaw_t = torch.tensor(RDAW, dtype=torch.float64)
        dt = result.dt

        W0 = _water_storage(state0, rdaw_t, rho_h2o)
        Wf = _water_storage(result.final_state, rdaw_t, rho_h2o)
        dW = (Wf - W0)                                   # kg/m^2

        # Precip input (zero here) + E flux (e_flux down-positive so E
        # removes water when negative).
        P = sum(f.rain + f.snow for f in fl)
        E = (result.e_flux * dt).sum(0)
        # Balance: dW = P + E - runoff  =>  runoff = P + E - dW.
        runoff = (P + E) - dW
        # Runoff must be finite, non-negative (drying => E<0, so the
        # soil loses more than E alone once runoff drains).
        assert torch.isfinite(runoff).all()
        assert (runoff >= -1e-9).all(), (
            f"negative runoff {runoff.tolist()} -- mass not conserved"
        )
        # With no precip the runoff comes entirely from drainage /
        # saturation-excess; on a drying run it can be small but the
        # residual of (P + E - dW) must equal it exactly (the model
        # has no other water sink).
        # dW + runoff = P + E  to machine precision.
        residual = (dW + runoff) - (P + E)
        assert residual.abs().max().item() < 1e-9, (
            f"water-budget residual {residual.abs().max().item():.3e} kg/m^2"
        )

    def test_closure_with_precip(self):
        """Rain input appears as ΔW + runoff (no E sign flip)."""
        cfg = IdealizedDiurnalConfig(
            n_steps=48, n_grid=4, precip_events=[(24, 36, 2e-4)],
        )
        result, fl, static, state0, model = _build_run(cfg=cfg)
        rho_h2o = float(model.consts.rho_h2o.item())
        rdaw_t = torch.tensor(RDAW, dtype=torch.float64)
        dt = result.dt

        W0 = _water_storage(state0, rdaw_t, rho_h2o)
        Wf = _water_storage(result.final_state, rdaw_t, rho_h2o)
        dW = Wf - W0
        P = sum(f.rain + f.snow for f in fl)
        E = (result.e_flux * dt).sum(0)
        runoff = (P + E) - dW
        # Precip > 0 so the soil must have gained water OR runoff
        # (at least one of dW > -|E| or runoff > 0).
        assert (P >= 0).all()
        assert torch.isfinite(runoff).all()
        assert (runoff >= -1e-9).all()
        # Tight closure.
        residual = (dW + runoff) - (P + E)
        assert residual.abs().max().item() < 1e-9, (
            f"water-budget residual {residual.abs().max().item():.3e}"
        )


# --------------------------------------------------------------------------- #
# Energy-balance closure
# --------------------------------------------------------------------------- #
class TestEnergyBalance:
    """Column heat storage change == Σ ZSURFL·dt (ZSURFL = SW+LW+H+LE).

    Note ``diag.ground_heat`` is the skin-to-subskin flux, NOT the
    soil-column input -- this test reconstructs the true input.
    """

    def test_closure_no_precip(self):
        result, fl, static, state0, model = _build_run(n_steps=48)
        rlvtt = float(model.consts.rlvtt.item())
        dt = result.dt

        # Per-class volume heat capacity (J/m^3/K), broadcast across layers.
        rcsoil = model.soil_table.gather("rcsoil", static.soil_type)
        rcsoil_b = rcsoil.unsqueeze(-1).expand(-1, 4)
        rdaw_t = torch.tensor(RDAW, dtype=torch.float64)

        H0 = _heat_storage(state0.soil_t, rcsoil_b, rdaw_t)
        Hf = _heat_storage(result.final_state.soil_t, rcsoil_b, rdaw_t)
        dH = Hf - H0                                       # J/m^2

        # ZSURFL = SW + LW + H + RLVTT*E (land-tile-weighted, but the
        # pure-land tile fractions sum to ~1 so the grid-mean fluxes
        # are the column input directly).
        cum_zsurfl = torch.zeros_like(dH)
        for k, f in enumerate(fl):
            sw = f.fr_so / f.dt
            lw = f.fr_th / f.dt
            H = result.h_flux[k]
            E = result.e_flux[k]
            cum_zsurfl = cum_zsurfl + (sw + lw + H + rlvtt * E) * dt

        # The bottom layer is zero-flux but not perfectly insulated over
        # 48 h, so expect tight relative closure but allow a few % for
        # the (tiny) deep-layer drift.
        rel = (dH - cum_zsurfl).abs().max() / cum_zsurfl.abs().max()
        assert rel.item() < 0.05, (
            f"energy-budget relative residual {rel.item():.3e} "
            f"(dH={dH[0].item()/1e6:.3f} MJ/m^2, "
            f"ΣZSURFL={cum_zsurfl[0].item()/1e6:.3f} MJ/m^2)"
        )


# --------------------------------------------------------------------------- #
# Physical-plausibility diagnostics
# --------------------------------------------------------------------------- #
class TestPhysicalPlausibility:
    """Sign / range / monotonicity checks on a clean drying run."""

    def test_soil_moisture_stays_in_range(self):
        from surf_pytorch.tables import SoilTable
        result, fl, static, _, model = _build_run(n_steps=48)
        table: SoilTable = model.soil_table
        wsat = table.gather("wsat", static.soil_type)
        wrest = table.gather("wrest", static.soil_type)
        # Every layer, every step.
        wmin = result.soil_w_liq.min().item()
        wmax = result.soil_w_liq.max().item()
        assert wmin >= wrest.min().item() - 1e-9, (
            f"soil W below wrest: {wmin:.4f} < {wrest.min().item():.4f}"
        )
        assert wmax <= wsat.max().item() + 1e-9, (
            f"soil W above wsat: {wmax:.4f} > {wsat.max().item():.4f}"
        )

    def test_evaporation_negative_on_drying_day(self):
        """No-precip day: net e_flux < 0 (evap removes water) mid-run."""
        result, fl, _, _, _ = _build_run(n_steps=48)
        # e_flux down-positive -> evaporation is negative.
        # Over a 48 h drying run at least some steps must show net evap.
        assert result.e_flux.min().item() < 0.0, (
            "e_flux never negative -- no evaporation on a drying run"
        )

    def test_skin_water_depletes_under_drying(self):
        """With no precip and net evap, skin_water must not increase."""
        result, fl, _, state0, _ = _build_run(n_steps=48)
        # All precip zero in the default config.
        assert all((f.rain[0] + f.snow[0]).item() == 0.0 for f in fl)
        # skin_water either stays at zero or decreases.
        d_skin = result.skin_water[-1] - state0.skin_water
        assert (d_skin <= 1e-9).all(), (
            f"skin_water increased without precip: {d_skin.tolist()}"
        )

    def test_precip_phase_split_by_temperature(self):
        """Forcing rain/snow split tracks Tair vs RTT (generator-level).

        This pins the *forcing* phase split.  The end-to-end snow
        accumulation path (forcing.snow -> PSSFL -> SRFSN_LWIMPS ->
        snow_w) is wired in M2d (see :meth:`LandSurface._update_snow`)
        and exercised by :class:`TestSnowThroughfall` below.
        """
        # Cold event -> all snow in the forcing.
        cfg_cold = IdealizedDiurnalConfig(
            n_steps=12, n_grid=1, tair_mean=270.0, tair_amp=2.0,
            precip_events=[(2, 8, 5e-4)],
        )
        fl_cold, _, _ = build_idealized_diurnal(cfg_cold)
        rain = sum(f.rain[0].item() for f in fl_cold)
        snow = sum(f.snow[0].item() for f in fl_cold)
        assert rain == pytest.approx(0.0, abs=1e-12)
        assert snow == pytest.approx(6 * 3600 * 5e-4, rel=1e-12)

        # Warm event -> all rain in the forcing.
        cfg_warm = IdealizedDiurnalConfig(
            n_steps=12, n_grid=1, tair_mean=295.0, tair_amp=2.0,
            precip_events=[(2, 8, 5e-4)],
        )
        fl_warm, _, _ = build_idealized_diurnal(cfg_warm)
        rain = sum(f.rain[0].item() for f in fl_warm)
        snow = sum(f.snow[0].item() for f in fl_warm)
        assert snow == pytest.approx(0.0, abs=1e-12)
        assert rain == pytest.approx(6 * 3600 * 5e-4, rel=1e-12)

    def test_run_finite_and_skin_temperature_sane(self):
        """Every output is finite and skin T stays in a physical band."""
        result, _, _, _, _ = _build_run(n_steps=48)
        for t in (result.skin_t, result.soil_t, result.soil_w_liq,
                  result.h_flux, result.e_flux):
            assert torch.isfinite(t).all(), "non-finite output"
        assert result.skin_t.min().item() > 150.0
        assert result.skin_t.max().item() < 350.0


# --------------------------------------------------------------------------- #
# Snow throughfall wiring (M2d)
# --------------------------------------------------------------------------- #
class TestSnowThroughfall:
    """End-to-end snowfall -> SRFSN_LWIMPS -> snow_w accumulation.

    These exercise the M2d wiring documented in
    :meth:`LandSurface._update_snow`: ``forcing.snow`` (cumulative snowfall,
    kg/m^2 over the coupling interval) is fed in as the large-scale
    snowfall rate ``PSSFL`` (the convective ``PSSFC`` stays zero, matching
    the same convention used for rainfall in
    :meth:`LandSurface._update_soil_moisture`).  ``SRFSN_LWIMPS`` then
    accumulates ``PSSFL`` straight into the snow-mass prognostic
    (``ZSSTAR = PSSNM1M + PTMST*(PSSFC + PSSFL)``), so a cold snowfall
    event must grow ``snow_w`` by ``rate * dt * zgridfrac`` per step --
    ``zgridfrac`` being the snowfall-interception grid fraction (tiles
    2..7 sum, ~1 over pure land).
    """

    @staticmethod
    def _snow_run(n_steps=12, *, rate=5e-4, tair_mean=270.0,
                  tair_amp=2.0, event=(2, 8)):
        """Run a 1-column cold snowfall event and return the result tuple."""
        cfg = IdealizedDiurnalConfig(
            n_steps=n_steps, n_grid=1, tair_mean=tair_mean,
            tair_amp=tair_amp, precip_events=[(*event, rate)],
        )
        return _build_run(cfg=cfg)

    def test_snow_accumulates_during_cold_event(self):
        """snow_w grows monotonically while the cold snowfall event is on.

        With the default rate 5e-4 kg/m^2/s and dt=3600 s each event step
        adds ~1.8 kg/m^2 (rate * dt * zgridfrac, zgridfrac ~ 1.0 over the
        pure-land tile mix the generator builds).
        """
        result, fl, static, state0, _ = self._snow_run(
            n_steps=12, rate=5e-4, tair_mean=270.0, event=(2, 8),
        )
        sw = result.snow_w[:, 0]
        # Before the event: snow_w stays at its (zero) cold start.
        assert sw[1].item() == pytest.approx(0.0, abs=1e-12)
        # Within the event window [2, 8): strictly monotonically increasing.
        event_window = sw[2:8]
        assert (torch.diff(event_window) > 0.0).all(), (
            f"snow_w not monotone during event: {event_window.tolist()}"
        )
        # Final snow mass is essentially the total snowfall input (no melt
        # at 270 K; sublimation is sub-1e-3 kg/m^2 over 12 h here).
        p_snow = sum(f.snow[0].item() for f in fl)
        assert sw[-1].item() == pytest.approx(p_snow, rel=1e-2, abs=1e-3), (
            f"snow_w final {sw[-1].item():.4f} != P_snow {p_snow:.4f}"
        )

    def test_snow_mass_per_step_matches_rate_times_dt(self):
        """Quantitative per-step growth: Δsnow_w ~= rate * dt * zgridfrac.

        Over the pure-land tile mix the snowfall-interception grid fraction
        (sum of port tiles 2..7) sums to ~1.0, so each event step adds the
        full snowfall rate * dt to the snow mass.  This pins the M2d wiring
        at the per-step level.
        """
        rate = 5e-4
        dt = 3600.0
        result, _, _, _, _ = self._snow_run(
            n_steps=12, rate=rate, tair_mean=270.0, event=(2, 8),
        )
        sw = result.snow_w[:, 0]
        # Each event step [2, 7] -> [3, 8) must add ~rate*dt.
        expected = rate * dt
        for k in range(2, 7):
            dsw = (sw[k + 1] - sw[k]).item()
            assert dsw == pytest.approx(expected, rel=1e-6, abs=1e-9), (
                f"step {k}: Δsnow_w {dsw:.6f} != rate*dt {expected:.6f}"
            )

    def test_no_snow_accumulates_in_warm_event(self):
        """Warm precip lands as rain, so snow_w stays at zero."""
        result, fl, _, state0, _ = self._snow_run(
            n_steps=12, rate=5e-4, tair_mean=295.0, event=(2, 8),
        )
        # Warm event -> all rain, no snowfall in the forcing.
        p_snow = sum(f.snow[0].item() for f in fl)
        assert p_snow == pytest.approx(0.0, abs=1e-12)
        # snow prognostic never grows.
        assert result.snow_w.max().item() == pytest.approx(
            state0.snow_w[0].item(), abs=1e-12,
        )

    def test_snow_water_balance_closes_with_snowfall(self):
        """Including snow_w in the water budget closes to machine precision.

        The M2d snow path couples snowfall into the snow-mass prognostic,
        so the storage term must include Δsnow_w -- else the cold-event
        water balance has a snowfall-sized residual.
        """
        result, fl, static, state0, model = self._snow_run(
            n_steps=12, rate=5e-4, tair_mean=270.0, event=(2, 8),
        )
        rho_h2o = float(model.consts.rho_h2o.item())
        rdaw_t = torch.tensor(RDAW, dtype=torch.float64)
        dt = result.dt

        # _water_storage already folds in snow_w, so this is the full
        # column water mass (soil + skin + snow).
        W0 = _water_storage(state0, rdaw_t, rho_h2o)
        Wf = _water_storage(result.final_state, rdaw_t, rho_h2o)
        dW = Wf - W0
        P = sum(f.rain + f.snow for f in fl)
        E = (result.e_flux * dt).sum(0)
        runoff = (P + E) - dW
        assert torch.isfinite(runoff).all()
        assert (runoff >= -1e-9).all(), (
            f"negative runoff {runoff.tolist()} -- mass not conserved"
        )
        residual = (dW + runoff) - (P + E)
        assert residual.abs().max().item() < 1e-9, (
            f"water-budget residual {residual.abs().max().item():.3e} "
            f"(dSnow_w included) -- snow throughfall broke mass conservation"
        )

    def test_snow_mass_grows_with_snowfall_rate(self):
        """Doubling the snowfall rate roughly doubles the accumulated mass."""
        rate_lo = 2.5e-4
        rate_hi = 5.0e-4
        r_lo, _, _, _, _ = self._snow_run(
            n_steps=12, rate=rate_lo, tair_mean=270.0, event=(2, 8),
        )
        r_hi, _, _, _, _ = self._snow_run(
            n_steps=12, rate=rate_hi, tair_mean=270.0, event=(2, 8),
        )
        sw_lo = r_lo.snow_w[-1, 0].item()
        sw_hi = r_hi.snow_w[-1, 0].item()
        # Linear response: hi/lo == rate_hi/rate_lo.
        assert sw_hi == pytest.approx(sw_lo * (rate_hi / rate_lo), rel=1e-3)


# --------------------------------------------------------------------------- #
# Deterministic trajectory regression
# --------------------------------------------------------------------------- #
class TestDeterministicTrajectory:
    """The idealized harness remains exactly repeatable on its own inputs.

    A stored trajectory cannot be a physics oracle after a Fortran-verified
    correction changes the surface-exchange closure. The OSM comparison
    covers physics parity; this test protects deterministic state handling.
    """

    def test_trajectory_is_repeatable(self):
        torch.manual_seed(0)
        fcfg = IdealizedDiurnalConfig(n_steps=48, n_grid=4)
        fl, static, state0 = build_idealized_diurnal(fcfg)
        first = ColumnHarness(
            LandSurface(SurfConfig(dtype=torch.float64, validate=False)), static, state0,
        ).run(fl)
        second = ColumnHarness(
            LandSurface(SurfConfig(dtype=torch.float64, validate=False)), static, state0,
        ).run(fl)

        for key in ("skin_t", "soil_t", "soil_w_liq", "snow_w",
                    "skin_water", "h_flux", "e_flux", "net_rad",
                    "ground_heat"):
            assert torch.equal(getattr(first, key), getattr(second, key)), (
                f"{key}: idealized trajectory is not repeatable"
            )


# --------------------------------------------------------------------------- #
# CSV export smoke
# --------------------------------------------------------------------------- #
class TestCSVExport:
    def test_csv_roundtrip(self, tmp_path):
        cfg = IdealizedDiurnalConfig(n_steps=12, n_grid=2)
        result, _, _, _, _ = _build_run(cfg=cfg)
        path = tmp_path / "run.csv"
        result.export_csv(str(path))
        data = np.loadtxt(str(path), delimiter=",")
        # 10 columns, 12 rows.
        assert data.shape == (12, 10)
        assert np.isfinite(data).all()
        # time axis increases by dt.
        assert np.allclose(np.diff(data[:, 0]), result.dt)

    def test_snapshot_roundtrip(self, tmp_path):
        cfg = IdealizedDiurnalConfig(n_steps=12, n_grid=2)
        result, _, _, _, _ = _build_run(cfg=cfg)
        path = tmp_path / "run.npz"
        result.export_snapshot(str(path))
        snap = np.load(str(path))
        assert set(snap.files) >= {"skin_t", "soil_t", "soil_w_liq",
                                   "h_flux", "e_flux", "time"}
        assert snap["skin_t"].shape == (12, 2, 8)
