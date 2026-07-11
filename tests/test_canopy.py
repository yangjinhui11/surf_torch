"""Unit tests for the skin-reservoir / canopy interception update
(``surf_pytorch.physics.canopy``).

These tests exercise the Python port of ``SRFWL`` -- the skin-reservoir
(canopy interception) prognostic update used by ``SURFTSTP_CTL``
(``surftstp_ctl_mod.F90:612``).  They cover the *internal* mathematical
correctness of the port: the five algorithm stages (clamp, evaporate,
collect dew, budget, intercept precip) and their sign / mass-balance
invariants.

Test groups:

* ``TestCanopyBranches`` -- the three regimes (evaporation / dew /
  precipitation interception) trigger correctly and produce the
  expected outputs for trivial inputs.
* ``TestCanopyMassBudget`` -- the skin-reservoir mass budget closes:
  PWL_out = PWL_in + (throughfall_in - rainfall_in) + dew - evap.
* ``TestCanopySignConventions`` -- PEINTTI sign (positive = reservoir
  absorbed flux, negative = released), throughfall <= rainfall.
* ``TestCanopyForwardIntegration`` -- forward() threads the new
  prognostic: new_state.skin_water advances; soil-W sees PEINTTI.
* ``TestCanopyVectorisation`` -- per-column independence and shape
  preservation across a batch.
* ``TestCanopyDifferentiable`` -- autograd flows through every branch.
"""

from __future__ import annotations

import pytest
import torch

from surf_pytorch import SurfConfig, LandSurface
from surf_pytorch.config import (
    TILE_WATER, TILE_ICE, TILE_WET_SKIN,
    TILE_LOW_VEG_DRY, TILE_LOW_VEG_SNOW,
    TILE_HIGH_VEG_DRY, TILE_HIGH_VEG_SNOW, TILE_BARE_SOIL,
)
from surf_pytorch.physics.canopy import (
    CanopyInputs, CanopyOutputs, update_canopy,
)
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.tables.veg import VegTable


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def veg_table() -> VegTable:
    return VegTable(dtype=torch.float64)


@pytest.fixture
def config() -> SurfConfig:
    return SurfConfig(dtype=torch.float64)


# --------------------------------------------------------------------------- #
# Helper: build self-consistent CanopyInputs.
# --------------------------------------------------------------------------- #
def _make_inputs(
    *,
    N: int = 2,
    pwlm1m: float = 0.0,
    prsfl_kg_m2s: float = 0.0,
    prsfc_kg_m2s: float = 0.0,
    pevap_wet_kg_m2s: float = 0.0,
    pevap_low_kg_m2s: float = 0.0,
    pevap_high_kg_m2s: float = 0.0,
    pwlmx: float = 0.2,
    cvl: float = 0.5,
    cvh: float = 0.35,
    land: bool = True,
    dtype: torch.dtype = torch.float64,
) -> CanopyInputs:
    """Build a CanopyInputs with a canonical 8-tile slab.

    The tile fractions are chosen so that cvl = low-veg-dry + low-veg-snow
    and cvh = high-veg-dry + high-veg-snow (the cover fractions the
    update reconstructs internally).  All rain is large-scale by default.
    """
    ntile = 8
    pfrti = torch.zeros(N, ntile, dtype=dtype)
    # water + ice + wet-skin + low-veg-dry + low-veg-snow +
    # high-veg-dry + high-veg-snow + bare-soil.
    wet = 0.05
    low_dry = cvl * 0.9
    low_snow = cvl * 0.1
    high_dry = cvh * 0.9
    high_snow = cvh * 0.1
    bare = 1.0 - wet - cvl - cvh
    pfrti[:, TILE_WET_SKIN] = wet
    pfrti[:, TILE_LOW_VEG_DRY] = low_dry
    pfrti[:, TILE_LOW_VEG_SNOW] = low_snow
    pfrti[:, TILE_HIGH_VEG_DRY] = high_dry
    pfrti[:, TILE_HIGH_VEG_SNOW] = high_snow
    pfrti[:, TILE_BARE_SOIL] = bare

    pevapti = torch.zeros(N, ntile, dtype=dtype)
    pevapti[:, TILE_WET_SKIN] = pevap_wet_kg_m2s
    pevapti[:, TILE_LOW_VEG_DRY] = pevap_low_kg_m2s
    pevapti[:, TILE_HIGH_VEG_DRY] = pevap_high_kg_m2s

    land_t = torch.tensor([land] * N, dtype=torch.bool)
    return CanopyInputs(
        ptmst=torch.tensor(3600.0, dtype=dtype),
        pwlm1m=torch.full((N,), pwlm1m, dtype=dtype),
        pcvl=torch.full((N,), cvl, dtype=dtype),
        pcvh=torch.full((N,), cvh, dtype=dtype),
        pwlmx=torch.full((N,), pwlmx, dtype=dtype),
        pfrti=pfrti,
        pevapti=pevapti,
        prsfc=torch.full((N,), prsfc_kg_m2s, dtype=dtype),
        prsfl=torch.full((N,), prsfl_kg_m2s, dtype=dtype),
        pevapsnw=torch.zeros(N, dtype=dtype),
        ldland=land_t,
    )


# --------------------------------------------------------------------------- #
# 1. Branch selection
# --------------------------------------------------------------------------- #
class TestCanopyBranches:
    """The three regimes (evaporation / dew / interception) trigger."""

    def test_no_flux_no_rain_keeps_reservoir(self, veg_table):
        # No rain, no evap, no dew -> PWL unchanged, zero throughfall.
        inp = _make_inputs(pwlm1m=0.1)
        out = update_canopy(inp, veg_table)
        torch.testing.assert_close(out.pwl, inp.pwlm1m, rtol=0, atol=1e-12)
        torch.testing.assert_close(out.ptsfl, torch.zeros_like(out.ptsfl),
                                   rtol=0, atol=1e-12)

    def test_evaporation_drains_reservoir(self, veg_table):
        # Negative PEVAPTI on the wet-skin tile drains PWL, floored at 0.
        inp = _make_inputs(pwlm1m=0.05, pevap_wet_kg_m2s=-1e-4)
        out = update_canopy(inp, veg_table)
        # Evaporated mass = dt * frac_wet * pevap = 3600*0.05*(-1e-4)
        # = -0.018, so PWL = max(0, 0.05 - 0.018) = 0.032.
        expected = max(0.0, 0.05 + 3600.0 * 0.05 * (-1e-4))
        torch.testing.assert_close(out.pwl[0].item(), expected,
                                   rtol=1e-9, atol=1e-12)

    def test_evaporation_floored_at_zero(self, veg_table):
        # Strong evaporation floors PWL at zero (no negative reservoir).
        inp = _make_inputs(pwlm1m=0.001, pevap_wet_kg_m2s=-1e-2)
        out = update_canopy(inp, veg_table)
        assert out.pwl.min().item() >= 0.0
        torch.testing.assert_close(out.pwl, torch.zeros_like(out.pwl),
                                   rtol=0, atol=1e-12)

    def test_dew_fills_reservoir_up_to_capacity(self, veg_table):
        # Positive PEVAPTI on a vegetated tile deposits dew up to PWLMX.
        inp = _make_inputs(pwlm1m=0.0, pevap_low_kg_m2s=1e-3, pwlmx=0.05)
        out = update_canopy(inp, veg_table)
        # Dew = min(PWLMX - PWL, dt*frac*pevap) = min(0.05, 3600*0.45*1e-3)
        #     = min(0.05, 1.62) = 0.05 -> saturates.
        torch.testing.assert_close(out.pwl, inp.pwlmx, rtol=1e-9, atol=1e-12)

    def test_precip_interception_saturates_reservoir(self, veg_table):
        # Heavy rain intercepts until the reservoir saturates; the rest
        # is throughfall.
        inp = _make_inputs(pwlm1m=0.0, prsfl_kg_m2s=1e-2, pwlmx=0.2,
                           cvl=0.5, cvh=0.35)
        out = update_canopy(inp, veg_table)
        # Reservoir saturates to PWLMX.
        torch.testing.assert_close(out.pwl, inp.pwlmx, rtol=1e-9, atol=1e-12)
        # Throughfall < raw rainfall (interception removed some).
        assert (out.ptsfl < inp.prsfl).all()

    def test_sea_point_zero_reservoir_passthrough(self, veg_table):
        # Sea points: PWL forced to 0, throughfall = raw rainfall.
        inp = _make_inputs(pwlm1m=0.3, prsfl_kg_m2s=1e-3, land=False)
        out = update_canopy(inp, veg_table)
        torch.testing.assert_close(out.pwl, torch.zeros_like(out.pwl),
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(out.ptsfl, inp.prsfl, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. Mass budget closure
# --------------------------------------------------------------------------- #
class TestCanopyMassBudget:
    """The skin-reservoir mass budget closes."""

    def test_mass_budget_rain_only(self, veg_table):
        # PWL_out - PWL_in = (PRSFL - PTSFL) * dt   [no evap, no dew]
        inp = _make_inputs(pwlm1m=0.05, prsfl_kg_m2s=5e-3, pwlmx=0.5)
        out = update_canopy(inp, veg_table)
        lhs = out.pwl - inp.pwlm1m
        rhs = (inp.prsfl - out.ptsfl) * 3600.0
        torch.testing.assert_close(lhs, rhs, rtol=1e-9, atol=1e-12)

    def test_mass_budget_evap_only(self, veg_table):
        # PWL_out - PWL_in = dt * frac_wet * pevap_wet   [no rain]
        inp = _make_inputs(pwlm1m=0.1, pevap_wet_kg_m2s=-2e-4)
        out = update_canopy(inp, veg_table)
        lhs = out.pwl - inp.pwlm1m
        rhs = 3600.0 * inp.pfrti[:, TILE_WET_SKIN] * (-2e-4)
        # Only where the floor did not bind (PWL stays > 0).
        torch.testing.assert_close(lhs, rhs, rtol=1e-9, atol=1e-12)

    def test_mass_budget_combined(self, veg_table):
        # Rain + evap: PWL_out - PWL_in = (PRSFL-PTSFL)*dt + dt*frac*pevap.
        inp = _make_inputs(pwlm1m=0.05, prsfl_kg_m2s=3e-3,
                           pevap_wet_kg_m2s=-1e-4, pwlmx=0.5)
        out = update_canopy(inp, veg_table)
        lhs = out.pwl - inp.pwlm1m
        rhs = ((inp.prsfl - out.ptsfl) * 3600.0
               + 3600.0 * inp.pfrti[:, TILE_WET_SKIN] * (-1e-4))
        torch.testing.assert_close(lhs, rhs, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# 3. Sign conventions
# --------------------------------------------------------------------------- #
class TestCanopySignConventions:
    """PEINTTI and throughfall carry the correct sign."""

    def test_peintti_negative_on_evaporation(self, veg_table):
        # Evaporation of the reservoir: PEINTTI(:, wet-skin) < 0
        # (reservoir released water the soil must see as extra input).
        inp = _make_inputs(pwlm1m=0.1, pevap_wet_kg_m2s=-1e-4)
        out = update_canopy(inp, veg_table)
        pe_wet = out.peintti[:, TILE_WET_SKIN]
        assert (pe_wet <= 0).all()

    def test_peintti_positive_on_dew(self, veg_table):
        # Dew collection: PEINTTI(:, tile) > 0 (reservoir absorbed flux
        # the soil therefore does NOT see).
        inp = _make_inputs(pwlm1m=0.0, pevap_low_kg_m2s=1e-3, pwlmx=0.5)
        out = update_canopy(inp, veg_table)
        pe_low = out.peintti[:, TILE_LOW_VEG_DRY]
        assert (pe_low >= 0).all()

    def test_throughfall_le_rainfall(self, veg_table):
        # Interception can only remove water, never add it.
        inp = _make_inputs(pwlm1m=0.0, prsfl_kg_m2s=5e-3, pwlmx=0.3)
        out = update_canopy(inp, veg_table)
        assert (out.ptsfl <= inp.prsfl + 1e-15).all()

    def test_peintti_zero_when_no_flux(self, veg_table):
        inp = _make_inputs(pwlm1m=0.0)
        out = update_canopy(inp, veg_table)
        torch.testing.assert_close(out.peintti,
                                   torch.zeros_like(out.peintti),
                                   rtol=0, atol=1e-12)

    def test_pwl_bounded_by_capacity(self, veg_table):
        # PWL can never exceed PWLMX.
        inp = _make_inputs(pwlm1m=0.0, prsfl_kg_m2s=1e-2, pwlmx=0.15)
        out = update_canopy(inp, veg_table)
        assert (out.pwl <= inp.pwlmx + 1e-9).all()
        assert (out.pwl >= 0.0).all()


# --------------------------------------------------------------------------- #
# 4. forward() integration
# --------------------------------------------------------------------------- #
def _make_forward_slab(
    *, N: int = 2, rain_kg_m2: float = 0.0, skin_water: float = 0.0,
    dtype: torch.dtype = torch.float64,
):
    """Build a (state, forcing, static, cfg) slab for forward() tests."""
    ntile, nsoil = 8, 4
    cfg = SurfConfig(dtype=dtype, validate=True)
    frac = torch.zeros(N, 8, dtype=dtype)
    wet, cvl, cvh = 0.05, 0.5, 0.35
    frac[:, TILE_WET_SKIN] = wet
    frac[:, TILE_LOW_VEG_DRY] = cvl * 0.9
    frac[:, TILE_LOW_VEG_SNOW] = cvl * 0.1
    frac[:, TILE_HIGH_VEG_DRY] = cvh * 0.9
    frac[:, TILE_HIGH_VEG_SNOW] = cvh * 0.1
    frac[:, TILE_BARE_SOIL] = 1.0 - wet - cvl - cvh
    static = Static(
        tile_frac_base=frac, land_frac=torch.ones(N, dtype=dtype),
        orography=torch.zeros(N, dtype=dtype),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=torch.full((N,), 1, dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        albd=None, albw=None, emis=None,
        z0m=None, z0h=None, z0q=None,
        lat=torch.full((N,), 0.5, dtype=dtype),
        lon=torch.full((N,), 0.1, dtype=dtype),
    )
    forcing = Forcing(
        u=torch.full((N,), 3.0, dtype=dtype), v=torch.full((N,), 1.0, dtype=dtype),
        t=torch.full((N,), 285.0, dtype=dtype), q=torch.full((N,), 0.005, dtype=dtype),
        co2=torch.full((N,), 6e-4, dtype=dtype), p_surf=torch.full((N,), 1.0e5, dtype=dtype),
        geom1=torch.full((N,), 20000.0, dtype=dtype),
        fr_so=torch.full((N,), 5e5, dtype=dtype), fr_th=torch.full((N,), -1e5, dtype=dtype),
        mu0=torch.full((N,), 0.5, dtype=dtype),
        rain=torch.full((N,), rain_kg_m2, dtype=dtype),
        snow=torch.zeros(N, dtype=dtype),
        dt=torch.tensor(3600.0, dtype=dtype),
    )
    state = SurfState(
        skin_t=torch.full((N, ntile), 285.0, dtype=dtype),
        soil_t=torch.full((N, nsoil), 280.0, dtype=dtype),
        soil_w_liq=torch.full((N, nsoil), 0.3, dtype=dtype),
        snow_w=torch.zeros(N, dtype=dtype), snow_frac=torch.zeros(N, dtype=dtype),
        skin_water=torch.full((N,), skin_water, dtype=dtype),
    )
    return state, forcing, static, cfg


class TestCanopyForwardIntegration:
    """forward() threads the skin-reservoir prognostic end-to-end."""

    def test_skin_water_advances_with_rain(self):
        state, forcing, static, cfg = _make_forward_slab(rain_kg_m2=5.0)
        model = LandSurface(cfg)
        ns, _ = model(state, forcing, static)
        # Heavy rain -> reservoir fills (interception > 0).
        assert (ns.skin_water > state.skin_water).all()
        assert (ns.skin_water >= 0).all()

    def test_skin_water_carry_when_zero_rain(self):
        # No rain, no evap-driving divergence: skin_water stays ~constant.
        state, forcing, static, cfg = _make_forward_slab(
            rain_kg_m2=0.0, skin_water=0.05)
        model = LandSurface(cfg)
        ns, _ = model(state, forcing, static)
        # The reservoir may evaporate slightly if the wet-skin tile dries,
        # but it cannot go negative.
        assert (ns.skin_water >= 0).all()

    def test_soil_w_sees_interception(self):
        # With canopy interception active, the soil-W budget closes
        # against the throughfall (rainfall after interception), not the
        # raw rainfall.  Verify soil-W is finite and bounded.
        state, forcing, static, cfg = _make_forward_slab(rain_kg_m2=3.0)
        model = LandSurface(cfg)
        ns, _ = model(state, forcing, static)
        assert torch.isfinite(ns.soil_w_liq).all()
        # Soil moisture stays within physical bounds [0, WSAT ~0.47].
        assert (ns.soil_w_liq >= 0).all()
        assert (ns.soil_w_liq <= 0.5).all()

    def test_forward_finite_after_canopy(self):
        # Sanity: all prognostics finite after the canopy step.
        state, forcing, static, cfg = _make_forward_slab(rain_kg_m2=2.0)
        model = LandSurface(cfg)
        ns, diag = model(state, forcing, static)
        assert torch.isfinite(ns.skin_t).all()
        assert torch.isfinite(ns.skin_water).all()
        assert torch.isfinite(ns.soil_w_liq).all()


# --------------------------------------------------------------------------- #
# 5. Vectorisation
# --------------------------------------------------------------------------- #
class TestCanopyVectorisation:
    """Per-column independence and shape preservation across a batch."""

    def test_batch_shape_preserved(self, veg_table):
        N = 5
        inp = _make_inputs(N=N, pwlm1m=0.1, prsfl_kg_m2s=2e-3)
        out = update_canopy(inp, veg_table)
        assert out.pwl.shape == (N,)
        assert out.ptsfl.shape == (N,)
        assert out.ptsfc.shape == (N,)
        assert out.peintti.shape == (N, 8)

    def test_column_independence(self, veg_table):
        # Two columns with identical inputs give identical outputs.
        inp = _make_inputs(N=2, pwlm1m=0.1, prsfl_kg_m2s=2e-3)
        out = update_canopy(inp, veg_table)
        torch.testing.assert_close(out.pwl[0], out.pwl[1], rtol=0, atol=0)
        torch.testing.assert_close(out.ptsfl[0], out.ptsfl[1], rtol=0, atol=0)

    def test_mixed_columns_independent(self, veg_table):
        # Column 0 rains, column 1 does not -- outputs differ as expected.
        inp = _make_inputs(N=2, pwlm1m=0.0, prsfl_kg_m2s=0.0)
        # Override: column 0 gets rain, column 1 stays dry.
        inp.prsfl[0] = 5e-3
        inp.prsfl[1] = 0.0
        out = update_canopy(inp, veg_table)
        # Column 0 fills (interception), column 1 stays at PWLM1M = 0.
        assert out.pwl[0] > out.pwl[1]
        assert out.ptsfl[0] > 0
        torch.testing.assert_close(out.ptsfl[1], torch.zeros((), dtype=torch.float64),
                                   rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# 6. Differentiability
# --------------------------------------------------------------------------- #
class TestCanopyDifferentiable:
    """Autograd flows through every branch of the canopy update."""

    def test_grad_flows_through_evaporation(self, veg_table):
        inp = _make_inputs(pwlm1m=0.1, pevap_wet_kg_m2s=-1e-4)
        inp.pwlm1m.requires_grad_(True)
        out = update_canopy(inp, veg_table)
        out.pwl.sum().backward()
        assert inp.pwlm1m.grad is not None
        assert torch.isfinite(inp.pwlm1m.grad).all()

    def test_grad_flows_through_interception(self, veg_table):
        inp = _make_inputs(pwlm1m=0.0, prsfl_kg_m2s=3e-3, pwlmx=0.5)
        inp.prsfl.requires_grad_(True)
        out = update_canopy(inp, veg_table)
        out.pwl.sum().backward()
        assert inp.prsfl.grad is not None
        assert torch.isfinite(inp.prsfl.grad).all()

    def test_grad_flows_through_peintti(self, veg_table):
        inp = _make_inputs(pwlm1m=0.1, pevap_wet_kg_m2s=-1e-4)
        inp.pevapti.requires_grad_(True)
        out = update_canopy(inp, veg_table)
        out.peintti.sum().backward()
        assert inp.pevapti.grad is not None
        assert torch.isfinite(inp.pevapti.grad).all()

    def test_forward_differentiable_through_canopy(self):
        # Full forward() with canopy: gradients flow to the initial
        # skin-reservoir state.
        state, forcing, static, cfg = _make_forward_slab(
            rain_kg_m2=2.0, skin_water=0.05)
        cfg = SurfConfig(dtype=torch.float64, validate=False, differentiable=True)
        model = LandSurface(cfg)
        state.skin_water.requires_grad_(True)
        ns, _ = model(state, forcing, static)
        ns.skin_water.sum().backward()
        assert state.skin_water.grad is not None
        assert torch.isfinite(state.skin_water.grad).all()
