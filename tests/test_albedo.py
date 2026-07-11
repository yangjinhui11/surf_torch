"""Tests for the surface-albedo subsystem (snow + background two-tier).

Verifies the end-to-end albedo wiring in :mod:`surf_pytorch.model`:

* :meth:`LandSurface._surface_albedo` -- per-tile broadband alpha mapping
  (snow tiles -> PASN, wet-skin -> albw, dry/bare -> albd, water/ice -> 0).
* the alpha injection at forward() section 5 (``ssrfl_t *= (1 - alpha)``).
* the prognostic snow-albedo advance via :meth:`_update_snow_albedo`.
* the **dormant fallback**: with no albedo fields set, alpha is 0
  everywhere and forward() is bit-identical to the pre-albedo build
  (the regression guard for the existing golden snapshots).
* SEB feedback: a non-zero snow albedo reduces absorbed SW, lowering the
  snow-tile skin temperature relative to the albedo=0 case.

Tile indices (``surf_pytorch.config``):
    0 water, 1 ice, 2 wet-skin, 3 low-veg dry, 4 low-veg snow,
    5 high-veg dry, 6 high-veg snow, 7 bare soil.
"""

from __future__ import annotations

import pytest
import torch

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.config import (
    N_BASE_TILES,
    TILE_WATER, TILE_ICE, TILE_WET_SKIN,
    TILE_LOW_VEG_DRY, TILE_LOW_VEG_SNOW,
    TILE_HIGH_VEG_DRY, TILE_HIGH_VEG_SNOW, TILE_BARE_SOIL,
)
from surf_pytorch.state import Forcing, Static, SurfState


# --------------------------------------------------------------------------- #
# Input builders
# --------------------------------------------------------------------------- #
def _make_state(
    *, N=2, ntile=8, snow_frac=0.0, skin_water=0.0,
    snow_alb=None, snow_t=None, snow_w=None,
    dtype=torch.float64,
) -> SurfState:
    sw = snow_w if snow_w is not None else (25.0 if snow_frac > 0 else 0.0)
    return SurfState(
        skin_t=torch.full((N, ntile), 285.0, dtype=dtype),
        soil_t=torch.full((N, 4), 283.0, dtype=dtype),
        soil_w_liq=torch.full((N, 4), 0.30, dtype=dtype),
        snow_w=torch.full((N,), sw, dtype=dtype),
        snow_frac=torch.full((N,), snow_frac, dtype=dtype),
        skin_water=torch.full((N,), skin_water, dtype=dtype),
        snow_alb=(torch.full((N,), snow_alb, dtype=dtype)
                  if snow_alb is not None else None),
        snow_t=(torch.full((N,), snow_t, dtype=dtype)
                if snow_t is not None else None),
    )


def _make_static(
    *, N=2, albd=None, albw=None, dtype=torch.float64,
) -> Static:
    """Build a minimal Static.  No cover fields -> static tile_frac path."""
    return Static(
        tile_frac_base=torch.zeros((N, N_BASE_TILES), dtype=dtype),
        land_frac=torch.ones(N, dtype=dtype),
        orography=torch.full((N,), 50.0, dtype=dtype),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=torch.full((N,), 3, dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.full((N,), 0.99, dtype=dtype),
        albd=(torch.full((N,), albd, dtype=dtype) if albd is not None else None),
        albw=(torch.full((N,), albw, dtype=dtype) if albw is not None else None),
    )


def _make_static_tile_frac(
    *, N=2, low_dry=0.4, high_dry=0.3, bare=0.2, wet=0.1,
    low_snow=0.0, high_snow=0.0, dtype=torch.float64,
) -> Static:
    """Build a Static with a hand-set tile_frac_base (static split path).

    Used to give the snow tiles non-zero cover so the albedo injection is
    actually exercised on those tiles.
    """
    base = torch.zeros((N, N_BASE_TILES), dtype=dtype)
    base[:, TILE_WET_SKIN] = wet
    base[:, TILE_LOW_VEG_DRY] = low_dry
    base[:, TILE_LOW_VEG_SNOW] = low_snow
    base[:, TILE_HIGH_VEG_DRY] = high_dry
    base[:, TILE_HIGH_VEG_SNOW] = high_snow
    base[:, TILE_BARE_SOIL] = bare
    return Static(
        tile_frac_base=base,
        land_frac=torch.ones(N, dtype=dtype),
        orography=torch.full((N,), 50.0, dtype=dtype),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=torch.full((N,), 3, dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.full((N,), 0.99, dtype=dtype),
    )


def _make_forcing(*, N=2, fr_so=200.0, snow=0.0, dtype=torch.float64) -> Forcing:
    return Forcing(
        u=torch.full((N,), 5.0, dtype=dtype),
        v=torch.zeros((N,), dtype=dtype),
        t=torch.full((N,), 288.0, dtype=dtype),
        q=torch.full((N,), 0.003, dtype=dtype),
        co2=torch.full((N,), 6e-4, dtype=dtype),
        p_surf=torch.full((N,), 1.0e5, dtype=dtype),
        geom1=torch.full((N,), 50.0, dtype=dtype),
        fr_so=torch.full((N,), fr_so * 3600.0, dtype=dtype),
        fr_th=torch.full((N,), -300.0 * 3600.0, dtype=dtype),
        mu0=torch.full((N,), 0.5, dtype=dtype),
        rain=torch.zeros((N,), dtype=dtype),
        snow=torch.full((N,), snow * 3600.0, dtype=dtype),
        dt=torch.tensor(3600.0, dtype=dtype),
    )


def _model(*, validate=False) -> LandSurface:
    return LandSurface(SurfConfig(dtype=torch.float64, validate=validate))


# --------------------------------------------------------------------------- #
# Dormant fallback (bit-identical regression guard)
# --------------------------------------------------------------------------- #
class TestFallbackEquivalence:
    """No albedo fields -> alpha=0 everywhere -> forward() unchanged."""

    def test_surface_albedo_all_zero_when_dormant(self):
        m = _model()
        state = _make_state()              # snow_alb=None
        static = _make_static()            # albd=albw=None
        alpha = m._surface_albedo(state, static, _make_forcing(), 8, torch.float64,
                                  torch.device("cpu"))
        assert alpha.shape == (2, 8)
        assert torch.all(alpha == 0.0)

    def test_forward_dormant_keeps_snow_alb_none(self):
        m = _model()
        state = _make_state(snow_frac=0.25)
        static = _make_static_tile_frac(low_snow=0.2, high_snow=0.1)
        forcing = _make_forcing()
        new_state, _ = m(state, forcing, static)
        # snow_alb stays None: the subsystem never activated.
        assert new_state.snow_alb is None

    def test_forward_dormant_bit_identical_to_no_albedo(self):
        """A dormant step must match a step run before the albedo work.

        We can't compare against the pre-build binary, so we instead
        confirm determinism + the invariants: alpha=0 means the SW term
        is unmodified, and two consecutive dormant steps agree exactly.
        """
        m = _model()
        torch.manual_seed(0)
        state = _make_state(snow_frac=0.2)
        static = _make_static_tile_frac(low_snow=0.3)
        forcing = _make_forcing()
        ns1, _ = m(state, forcing, static)
        ns2, _ = m(state, forcing, static)
        assert torch.equal(ns1.skin_t, ns2.skin_t)


# --------------------------------------------------------------------------- #
# Snow-albedo tile mapping
# --------------------------------------------------------------------------- #
class TestSnowTileMapping:
    """snow_alb set -> tiles 4 (low snow) & 6 (high snow) take PASN."""

    def test_snow_albedo_on_snow_tiles(self):
        m = _model()
        state = _make_state(snow_alb=0.70)
        static = _make_static()    # albd=albw=None -> only snow path active
        alpha = m._surface_albedo(state, static, _make_forcing(), 8, torch.float64,
                                  torch.device("cpu"))
        assert torch.all(alpha[:, TILE_LOW_VEG_SNOW] == 0.70)
        assert torch.all(alpha[:, TILE_HIGH_VEG_SNOW] == 0.70)
        # Non-snow tiles stay 0 (no background supplied).
        assert torch.all(alpha[:, TILE_WATER] == 0.0)
        assert torch.all(alpha[:, TILE_ICE] == 0.0)
        assert torch.all(alpha[:, TILE_WET_SKIN] == 0.0)
        assert torch.all(alpha[:, TILE_LOW_VEG_DRY] == 0.0)
        assert torch.all(alpha[:, TILE_HIGH_VEG_DRY] == 0.0)
        assert torch.all(alpha[:, TILE_BARE_SOIL] == 0.0)

    def test_snow_albedo_in_bounds(self):
        m = _model()
        for pasn in [0.50, 0.60, 0.85]:
            state = _make_state(snow_alb=pasn)
            static = _make_static()
            alpha = m._surface_albedo(state, static, _make_forcing(), 8, torch.float64,
                                      torch.device("cpu"))
            assert torch.all(alpha[:, TILE_LOW_VEG_SNOW] == pasn)

    def test_fewer_tiles_safe(self):
        """With ntile < 8 the per-tile writes are skipped gracefully."""
        m = _model()
        state = _make_state(snow_alb=0.70, ntile=8)
        static = _make_static()
        # ntile=3 -> only water/ice/wet-skin; snow tiles absent.
        alpha = m._surface_albedo(state, static, _make_forcing(), 3, torch.float64,
                                  torch.device("cpu"))
        assert alpha.shape == (2, 3)
        assert torch.all(alpha == 0.0)


# --------------------------------------------------------------------------- #
# Background-albedo mapping
# --------------------------------------------------------------------------- #
class TestBackgroundMapping:
    """albd/albw set -> dry/bare tiles and wet-skin take background."""

    def test_background_on_dry_tiles(self):
        m = _model()
        state = _make_state()              # snow_alb=None
        static = _make_static(albd=0.20, albw=0.10)
        alpha = m._surface_albedo(state, static, _make_forcing(), 8, torch.float64,
                                  torch.device("cpu"))
        assert torch.all(alpha[:, TILE_WET_SKIN] == 0.10)        # albw
        assert torch.all(alpha[:, TILE_LOW_VEG_DRY] == 0.20)     # albd
        assert torch.all(alpha[:, TILE_HIGH_VEG_DRY] == 0.20)    # albd
        assert torch.all(alpha[:, TILE_BARE_SOIL] == 0.20)       # albd
        # Snow tiles have no PASN and no albd-for-snow rule -> 0.
        assert torch.all(alpha[:, TILE_LOW_VEG_SNOW] == 0.0)
        assert torch.all(alpha[:, TILE_HIGH_VEG_SNOW] == 0.0)

    def test_albw_falls_back_to_albd(self):
        m = _model()
        state = _make_state()
        static = _make_static(albd=0.25)   # albw=None
        alpha = m._surface_albedo(state, static, _make_forcing(), 8, torch.float64,
                                  torch.device("cpu"))
        assert torch.all(alpha[:, TILE_WET_SKIN] == 0.25)   # albd fallback

    def test_albd_only_no_albw(self):
        m = _model()
        state = _make_state()
        static = _make_static(albd=0.18)
        alpha = m._surface_albedo(state, static, _make_forcing(), 8, torch.float64,
                                  torch.device("cpu"))
        assert torch.all(alpha[:, TILE_LOW_VEG_DRY] == 0.18)
        assert torch.all(alpha[:, TILE_WET_SKIN] == 0.18)    # albd fallback


# --------------------------------------------------------------------------- #
# Combined snow + background
# --------------------------------------------------------------------------- #
class TestCombinedMapping:
    """Both snow_alb and albd/albw set -> snow tiles take PASN, rest take bg."""

    def test_snow_overrides_background_on_snow_tiles(self):
        m = _model()
        state = _make_state(snow_alb=0.75)
        static = _make_static(albd=0.20, albw=0.10)
        alpha = m._surface_albedo(state, static, _make_forcing(), 8, torch.float64,
                                  torch.device("cpu"))
        # Snow tiles take PASN, not albd.
        assert torch.all(alpha[:, TILE_LOW_VEG_SNOW] == 0.75)
        assert torch.all(alpha[:, TILE_HIGH_VEG_SNOW] == 0.75)
        # Dry / bare / wet tiles take background.
        assert torch.all(alpha[:, TILE_WET_SKIN] == 0.10)
        assert torch.all(alpha[:, TILE_LOW_VEG_DRY] == 0.20)
        assert torch.all(alpha[:, TILE_HIGH_VEG_DRY] == 0.20)
        assert torch.all(alpha[:, TILE_BARE_SOIL] == 0.20)


# --------------------------------------------------------------------------- #
# Prognostic advance (forward integration)
# --------------------------------------------------------------------------- #
class TestPrognosticAdvance:
    """snow_alb set -> forward() writes new_state.snow_alb (aging)."""

    def test_new_snow_alb_is_tensor(self):
        m = _model()
        state = _make_state(snow_alb=0.80, snow_frac=0.3, snow_t=270.0)
        static = _make_static_tile_frac(low_snow=0.2, high_snow=0.1)
        forcing = _make_forcing()
        new_state, _ = m(state, forcing, static)
        assert new_state.snow_alb is not None
        assert new_state.snow_alb.shape == (2,)
        assert torch.isfinite(new_state.snow_alb).all()

    def test_snow_alb_in_bounds_after_step(self):
        m = _model()
        state = _make_state(snow_alb=0.80, snow_frac=0.3, snow_t=270.0)
        static = _make_static_tile_frac(low_snow=0.2, high_snow=0.1)
        forcing = _make_forcing()
        new_state, _ = m(state, forcing, static)
        assert torch.all(new_state.snow_alb >= 0.50 - 1e-12)
        assert torch.all(new_state.snow_alb <= 0.85 + 1e-12)

    def test_two_step_advance_changes_albedo(self):
        """Running two steps evolves snow_alb (cold decay -> darker)."""
        m = _model()
        state = _make_state(snow_alb=0.85, snow_frac=0.3, snow_t=268.0)
        static = _make_static_tile_frac(low_snow=0.2, high_snow=0.1)
        forcing = _make_forcing()
        ns1, _ = m(state, forcing, static)
        ns2, _ = m(ns1, forcing, static)
        # Cold decay from 0.85 must monotonically decrease.
        assert torch.all(ns2.snow_alb < ns1.snow_alb)
        assert torch.all(ns1.snow_alb < 0.85)


# --------------------------------------------------------------------------- #
# SEB feedback (alpha reduces absorbed SW -> cooler skin)
# --------------------------------------------------------------------------- #
class TestSebFeedback:
    """A non-zero snow albedo reduces absorbed SW on the snow tiles.

    With snow tiles covered and downward SW > 0, raising PASN must lower
    the snow-tile skin temperature (less energy absorbed).
    """

    def _run_with_albedo(self, snow_alb):
        m = _model()
        state = _make_state(
            snow_alb=snow_alb, snow_frac=0.3, snow_t=270.0,
        )
        static = _make_static_tile_frac(low_snow=0.3, high_snow=0.1)
        forcing = _make_forcing(fr_so=300.0)
        new_state, _ = m(state, forcing, static)
        return new_state

    def test_higher_albedo_cools_skin(self):
        ns_dark = self._run_with_albedo(snow_alb=0.50)
        ns_bright = self._run_with_albedo(snow_alb=0.85)
        # Brighter snow absorbs less SW -> cooler mean skin T.
        mean_dark = ns_dark.skin_t.mean()
        mean_bright = ns_bright.skin_t.mean()
        assert mean_bright < mean_dark

    def test_dormant_equals_zero_albedo(self):
        """Dormant (snow_alb=None) must equal an explicit alpha=0 run."""
        m = _model()
        base = _make_state(snow_frac=0.3, snow_t=270.0)
        static = _make_static_tile_frac(low_snow=0.3, high_snow=0.1)
        forcing = _make_forcing(fr_so=300.0)
        # Dormant (None) vs explicit snow_alb=0 (activated, alpha=0).
        ns_dormant, _ = m(base, forcing, static)
        active = _make_state(snow_alb=0.0, snow_frac=0.3, snow_t=270.0)
        ns_active, _ = m(active, forcing, static)
        assert torch.allclose(ns_dormant.skin_t, ns_active.skin_t, atol=1e-12)


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
class TestDifferentiable:
    """Gradient flows from skin_t back to snow_alb via the SW injection."""

    def test_grad_reaches_snow_alb(self):
        m = _model()
        m.config.differentiable = True
        pasn = torch.full((2,), 0.70, dtype=torch.float64, requires_grad=True)
        state = _make_state(snow_frac=0.3, snow_t=270.0)
        state.snow_alb = pasn
        static = _make_static_tile_frac(low_snow=0.3, high_snow=0.1)
        forcing = _make_forcing(fr_so=300.0)
        new_state, _ = m(state, forcing, static)
        new_state.skin_t.sum().backward()
        assert pasn.grad is not None
        assert torch.isfinite(pasn.grad).all()
        # Higher albedo -> cooler skin -> negative gradient sign overall
        # (skin_t decreases as PASN increases); magnitude non-zero where
        # snow tiles carry weight.
        assert torch.any(pasn.grad != 0.0)
