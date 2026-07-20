"""Tests for the cvl/cvh -> 8-tile split (SURFBC land-branch port).

These exercise :meth:`LandSurface._compute_tile_frac` -- the faithful
``surfbc_ctl_mod.F90`` land-branch PFRTI(2..8) split driven by the
vegetation cover fractions (PCVL / PCVH / PCIL) plus the prognostic
snow-cover fraction (ZCVS) and the wet-skin ratio (ZCVW).  The ported
equations (surfbc_ctl_mod.F90:368-377) are::

    ZFRTIT   = clamp(PCIL, 0, 1)
    ZPCVL    = PCVL * (1 - ZFRTIT)
    ZPCVH    = PCVH * (1 - ZFRTIT)
    ZPCVB    = 1 - ZPCVL - ZPCVH - ZFRTIT
    PWLMX    = RWLMAX*(1-PCVL-PCVH+PCVL*PLAIL+PCVH*PLAIH)
    ZCVW     = clamp(PWLM1M / PWLMX, 0, 1)
    PFRTI(2) = ZFRTIT * (1 - ZCVS)              # ice, snow-free
    PFRTI(3) = ZCVW * (1 - ZCVS) * (ZPCVL+ZPCVH+ZPCVB)   # wet-skin
    PFRTI(4) = ZPCVL * (1 - ZCVS) * (1 - ZCVW)  # low-veg dry
    PFRTI(5) = ZCVS * (ZPCVB + ZPCVL + ZFRTIT)  # snow on low/bare/ice
    PFRTI(6) = ZPCVH * (1 - ZCVS) * (1 - ZCVW)  # high-veg dry
    PFRTI(7) = ZCVS * ZPCVH                      # snow under high veg
    PFRTI(8) = ZPCVB * (1 - ZCVS) * (1 - ZCVW)  # bare soil

The test groups:

* TestFallbackEquivalence -- the no-cover path stays bit-identical to
  the legacy ``tile_frac_base`` static split (regression guard).
* TestSumToOne -- the split sums to 1 across the land tiles for a
  spread of cover / snow / wet-skin regimes.
* TestSnowScaling -- increasing snow_frac moves mass from the dry tiles
  into the snow tiles monotonically.
* TestWetSkinScaling -- increasing skin_water (ZCVW) moves mass from
  the dry veg / bare tiles into the wet-skin tile monotonically.
* TestBounds -- all fractions stay in [0, 1].
* TestAnalyticValues -- hand-computed fractions for a known regime.
* TestForwardIntegration -- a full forward step runs and stays finite
  when the cover inputs are active (end-to-end wiring).
"""

from __future__ import annotations

import pytest
import torch

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.config import (
    N_BASE_TILES,
    TILE_ICE, TILE_WET_SKIN,
    TILE_LOW_VEG_DRY, TILE_LOW_VEG_SNOW,
    TILE_HIGH_VEG_DRY, TILE_HIGH_VEG_SNOW, TILE_BARE_SOIL,
)
from surf_pytorch.state import Forcing, Static, SurfState


# --------------------------------------------------------------------------- #
# Input builders
# --------------------------------------------------------------------------- #
def _make_state(
    *, N=2, ntile=8, snow_frac=0.0, skin_water=0.0,
    dtype=torch.float64,
) -> SurfState:
    return SurfState(
        skin_t=torch.full((N, ntile), 285.0, dtype=dtype),
        soil_t=torch.full((N, 4), 283.0, dtype=dtype),
        soil_w_liq=torch.full((N, 4), 0.30, dtype=dtype),
        snow_w=torch.full((N,), 25.0 if snow_frac > 0 else 0.0, dtype=dtype),
        snow_frac=torch.full((N,), snow_frac, dtype=dtype),
        skin_water=torch.full((N,), skin_water, dtype=dtype),
    )


def _make_static(
    *, N=2, cvl=0.4, cvh=0.4, ice_land=0.0,
    lai_low=None, lai_high=None,
    veg_type_low=None, veg_type_high=None,
    dtype=torch.float64,
) -> Static:
    """Build a Static that *activates* the dynamic tile-split path.

    The cover inputs (cvl_frac / cvh_frac) are the sentinel fields;
    tile_frac_base is left zero (unused on the dynamic path).
    """
    return Static(
        tile_frac_base=torch.zeros((N, N_BASE_TILES), dtype=dtype),
        land_frac=torch.ones(N, dtype=dtype),
        orography=torch.full((N,), 50.0, dtype=dtype),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=torch.full((N,), 3, dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.full((N,), 0.99, dtype=dtype),
        cvl_frac=torch.full((N,), cvl, dtype=dtype),
        cvh_frac=torch.full((N,), cvh, dtype=dtype),
        ice_land_frac=torch.full((N,), ice_land, dtype=dtype),
        lai_low=(torch.full((N,), lai_low, dtype=dtype)
                 if lai_low is not None else None),
        lai_high=(torch.full((N,), lai_high, dtype=dtype)
                  if lai_high is not None else None),
        veg_type_low=(torch.full((N,), veg_type_low, dtype=torch.long)
                      if veg_type_low is not None else None),
        veg_type_high=(torch.full((N,), veg_type_high, dtype=torch.long)
                       if veg_type_high is not None else None),
    )


def _make_forcing(*, N=2, dtype=torch.float64) -> Forcing:
    return Forcing(
        u=torch.full((N,), 5.0, dtype=dtype),
        v=torch.zeros((N,), dtype=dtype),
        t=torch.full((N,), 288.0, dtype=dtype),
        q=torch.full((N,), 0.003, dtype=dtype),
        co2=torch.full((N,), 6e-4, dtype=dtype),
        p_surf=torch.full((N,), 1.0e5, dtype=dtype),
        geom1=torch.full((N,), 50.0, dtype=dtype),
        fr_so=torch.full((N,), 200.0 * 3600.0, dtype=dtype),
        fr_th=torch.full((N,), -300.0 * 3600.0, dtype=dtype),
        mu0=torch.full((N,), 0.5, dtype=dtype),
        rain=torch.zeros((N,), dtype=dtype),
        snow=torch.zeros((N,), dtype=dtype),
        dt=torch.tensor(3600.0, dtype=dtype),
    )


def _model(*, validate=False) -> LandSurface:
    return LandSurface(SurfConfig(dtype=torch.float64, validate=validate))


# --------------------------------------------------------------------------- #
# Fallback equivalence (no cover inputs -> legacy static split)
# --------------------------------------------------------------------------- #
class TestFallbackEquivalence:
    """Without cover inputs the split must equal the legacy tile_frac_base."""

    def test_state_ignored_without_covers(self):
        m = _model()
        base = torch.zeros((2, 8), dtype=torch.float64)
        base[:, TILE_WET_SKIN] = 0.05
        base[:, TILE_LOW_VEG_DRY] = 0.30
        base[:, TILE_HIGH_VEG_DRY] = 0.30
        base[:, TILE_BARE_SOIL] = 0.35
        static = Static(
            tile_frac_base=base,
            land_frac=torch.ones(2, dtype=torch.float64),
            orography=torch.full((2,), 50.0, dtype=torch.float64),
            soil_type=torch.full((2,), 3, dtype=torch.long),
            veg_type=torch.full((2,), 3, dtype=torch.long),
            soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89],
                                        dtype=torch.float64),
        )
        state = _make_state(snow_frac=0.5, skin_water=1e-3)
        with_state = m._tile_frac(static, 8, state=state)
        without = m._tile_frac(static, 8)
        assert torch.equal(with_state, without)
        assert torch.equal(with_state, base)

    def test_cover_fractions_returns_none_without_covers(self):
        m = _model()
        static = Static(
            tile_frac_base=torch.zeros((2, 8), dtype=torch.float64),
            land_frac=torch.ones(2, dtype=torch.float64),
            orography=torch.full((2,), 50.0, dtype=torch.float64),
            soil_type=torch.full((2,), 3, dtype=torch.long),
            veg_type=torch.full((2,), 3, dtype=torch.long),
            soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89],
                                        dtype=torch.float64),
        )
        assert m._cover_fractions(static) is None


# --------------------------------------------------------------------------- #
# Sum-to-one invariants
# --------------------------------------------------------------------------- #
class TestSumToOne:
    @pytest.mark.parametrize("cvl,cvh,ice", [
        (0.4, 0.4, 0.0),
        (0.3, 0.5, 0.0),
        (0.0, 0.0, 0.0),   # all bare soil
        (0.2, 0.2, 0.3),   # with land ice
    ])
    @pytest.mark.parametrize("snow_frac", [0.0, 0.25, 0.5, 1.0])
    def test_sums_to_one(self, cvl, cvh, ice, snow_frac):
        m = _model()
        static = _make_static(cvl=cvl, cvh=cvh, ice_land=ice)
        state = _make_state(snow_frac=snow_frac, skin_water=0.0)
        frac = m._tile_frac(static, 8, state=state)
        s = frac.sum(dim=-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-12)

    def test_sums_to_one_with_wet_skin(self):
        m = _model()
        static = _make_static(cvl=0.4, cvh=0.4, ice_land=0.0)
        # PWLMX determines ZCVW; use a skin_water that yields a mid range.
        state = _make_state(snow_frac=0.0, skin_water=0.1)
        frac = m._tile_frac(static, 8, state=state)
        s = frac.sum(dim=-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-12)


# --------------------------------------------------------------------------- #
# Snow-cover scaling
# --------------------------------------------------------------------------- #
class TestSnowScaling:
    def test_more_snow_increases_snow_tiles(self):
        """Increasing snow_frac must not decrease the combined snow tiles."""
        m = _model()
        static = _make_static(cvl=0.4, cvh=0.4, ice_land=0.0)
        snow_combined_prev = None
        for sf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            state = _make_state(snow_frac=sf, skin_water=0.0)
            frac = m._tile_frac(static, 8, state=state)
            snow_combined = (frac[:, TILE_LOW_VEG_SNOW]
                             + frac[:, TILE_HIGH_VEG_SNOW]).mean()
            if snow_combined_prev is not None:
                assert snow_combined >= snow_combined_prev - 1e-12
            snow_combined_prev = snow_combined

    def test_full_snow_collapses_to_snow_tiles_only(self):
        """ZCVS=1 -> all mass on the two snow tiles (dry tiles vanish)."""
        m = _model()
        static = _make_static(cvl=0.4, cvh=0.4, ice_land=0.0)
        state = _make_state(snow_frac=1.0, skin_water=0.0)
        frac = m._tile_frac(static, 8, state=state)
        # Dry veg / bare / wet-skin must be zero under full snow.
        for idx in [TILE_WET_SKIN, TILE_LOW_VEG_DRY,
                    TILE_HIGH_VEG_DRY, TILE_BARE_SOIL]:
            assert frac[:, idx].abs().max() < 1e-12
        # SURFBC corrects PCVLC/PCVHC by RVCOV(type 3)=0.9, then the
        # no-FLake branch uses PFRTI5=ZCVS*(1-PCVH), PFRTI7=ZCVS*PCVH.
        assert torch.allclose(frac[:, TILE_LOW_VEG_SNOW],
                              torch.full((2,), 0.64, dtype=torch.float64),
                              atol=1e-12)
        assert torch.allclose(frac[:, TILE_HIGH_VEG_SNOW],
                              torch.full((2,), 0.36, dtype=torch.float64),
                              atol=1e-12)

    def test_zero_snow_vanishes_snow_tiles(self):
        """ZCVS=0 -> snow tiles are zero."""
        m = _model()
        static = _make_static(cvl=0.4, cvh=0.4, ice_land=0.0)
        state = _make_state(snow_frac=0.0, skin_water=0.0)
        frac = m._tile_frac(static, 8, state=state)
        for idx in [TILE_LOW_VEG_SNOW, TILE_HIGH_VEG_SNOW]:
            assert frac[:, idx].abs().max() < 1e-12


# --------------------------------------------------------------------------- #
# Wet-skin scaling
# --------------------------------------------------------------------------- #
class TestWetSkinScaling:
    def test_more_skin_water_increases_wet_skin_tile(self):
        m = _model()
        static = _make_static(cvl=0.4, cvh=0.4, ice_land=0.0)
        prev = None
        for sw in [0.0, 0.05, 0.1, 0.2, 0.5]:
            state = _make_state(snow_frac=0.0, skin_water=sw)
            frac = m._tile_frac(static, 8, state=state)
            wet = frac[:, TILE_WET_SKIN].mean()
            if prev is not None:
                assert wet >= prev - 1e-12
            prev = wet

    def test_saturated_skin_makes_dry_tiles_vanish(self):
        """ZCVW=1 (skin_water >= PWLMX) -> dry veg / bare tiles zero."""
        m = _model()
        static = _make_static(cvl=0.4, cvh=0.4, ice_land=0.0)
        # PWLMX = RWLMAX*(1-0.4-0.4+0.4*LAI+0.4*LAI).  Use a large value
        # to guarantee ZCVW clamps to 1.
        state = _make_state(snow_frac=0.0, skin_water=100.0)
        frac = m._tile_frac(static, 8, state=state)
        for idx in [TILE_LOW_VEG_DRY, TILE_HIGH_VEG_DRY, TILE_BARE_SOIL]:
            assert frac[:, idx].abs().max() < 1e-12
        # Wet-skin tile holds the full cover.
        assert torch.allclose(frac[:, TILE_WET_SKIN],
                              torch.full((2,), 1.0, dtype=torch.float64),
                              atol=1e-9)


# --------------------------------------------------------------------------- #
# Bounds
# --------------------------------------------------------------------------- #
class TestBounds:
    def test_all_fractions_in_unit_interval(self):
        m = _model()
        # Sweep an adversarial grid of covers / snow / skin water.
        for cvl, cvh, ice in [(0.4, 0.4, 0.0), (0.2, 0.2, 0.3),
                              (0.0, 0.0, 0.0), (0.49, 0.49, 0.0)]:
            for sf in [0.0, 0.5, 1.0]:
                for sw in [0.0, 0.3, 5.0]:
                    static = _make_static(cvl=cvl, cvh=cvh, ice_land=ice)
                    state = _make_state(snow_frac=sf, skin_water=sw)
                    frac = m._tile_frac(static, 8, state=state)
                    assert (frac >= -1e-12).all(), (cvl, cvh, ice, sf, sw)
                    assert (frac <= 1.0 + 1e-12).all(), (cvl, cvh, ice, sf, sw)


# --------------------------------------------------------------------------- #
# Analytic values
# --------------------------------------------------------------------------- #
class TestAnalyticValues:
    def test_no_snow_dry_skin_hand_values(self):
        """cvl=0.4, cvh=0.4, ice=0, ZCVS=0, ZCVW=0.

        Expected after RVCOV(type 3)=0.9 correction: low-dry 0.36,
        high-dry 0.36, bare 0.28; everything else 0.
        """
        m = _model()
        static = _make_static(cvl=0.4, cvh=0.4, ice_land=0.0)
        state = _make_state(snow_frac=0.0, skin_water=0.0)
        frac = m._tile_frac(static, 8, state=state)
        f = frac[0]
        assert f[TILE_WET_SKIN].item() == pytest.approx(0.0, abs=1e-12)
        assert f[TILE_LOW_VEG_DRY].item() == pytest.approx(0.36, abs=1e-12)
        assert f[TILE_HIGH_VEG_DRY].item() == pytest.approx(0.36, abs=1e-12)
        assert f[TILE_BARE_SOIL].item() == pytest.approx(0.28, abs=1e-12)
        assert f[TILE_LOW_VEG_SNOW].item() == pytest.approx(0.0, abs=1e-12)
        assert f[TILE_HIGH_VEG_SNOW].item() == pytest.approx(0.0, abs=1e-12)
        assert f[TILE_ICE].item() == pytest.approx(0.0, abs=1e-12)

    def test_land_ice_feeds_ice_tile(self):
        """ice_land=0.3, no snow, dry -> ice tile = 0.3, covers rescaled."""
        m = _model()
        static = _make_static(cvl=0.4, cvh=0.4, ice_land=0.3)
        state = _make_state(snow_frac=0.0, skin_water=0.0)
        frac = m._tile_frac(static, 8, state=state)
        f = frac[0]
        assert f[TILE_ICE].item() == pytest.approx(0.3, abs=1e-12)
        assert f[TILE_LOW_VEG_DRY].item() == pytest.approx(0.252, abs=1e-12)
        assert f[TILE_HIGH_VEG_DRY].item() == pytest.approx(0.252, abs=1e-12)
        assert f[TILE_BARE_SOIL].item() == pytest.approx(0.196, abs=1e-12)


# --------------------------------------------------------------------------- #
# Dominant veg-type split
# --------------------------------------------------------------------------- #
class TestVegTypeSplit:
    def test_split_used_when_provided(self):
        m = _model()
        static = _make_static(veg_type_low=5, veg_type_high=6)
        vl, vh = m._dominant_veg_types(static)
        assert torch.equal(vl, torch.full((2,), 5, dtype=torch.long))
        assert torch.equal(vh, torch.full((2,), 6, dtype=torch.long))

    def test_fallback_uses_veg_type_for_both(self):
        m = _model()
        static = _make_static()   # veg_type default 3
        vl, vh = m._dominant_veg_types(static)
        assert torch.equal(vl, static.veg_type)
        assert torch.equal(vh, static.veg_type)


# --------------------------------------------------------------------------- #
# Interception capacity consistency
# --------------------------------------------------------------------------- #
class TestInterceptionCapacity:
    def test_canonical_formula_used_with_covers(self):
        """PWLMX = RWLMAX*(1-cvl-cvh+cvl*LAIL+cvh*LAIH) on the cover path."""
        m = _model()
        static = _make_static(cvl=0.4, cvh=0.4, lai_low=3.0, lai_high=5.0)
        frac = m._tile_frac(static, 8, state=_make_state())
        ktvl, ktvh = m._dominant_veg_types(static)
        wlmx = m._interception_capacity(static, frac, ktvl, ktvh)
        rwlmax = m.soil_table.consts.rwlmax
        pcvl = 0.4 * 0.9
        pcvh = 0.4 * 0.9
        expected = rwlmax * (1.0 - pcvl - pcvh + pcvl * 3.0 + pcvh * 5.0)
        assert torch.allclose(wlmx, torch.full((2,), float(expected),
                                               dtype=torch.float64),
                              atol=1e-12)

    def test_pwlmx_matches_compute_tile_frac_internal(self):
        """The PWLMX here must equal the one _compute_tile_frac uses for
        ZCVW (same canonical inputs)."""
        m = _model()
        static = _make_static(cvl=0.4, cvh=0.4, lai_low=3.0, lai_high=5.0)
        frac = m._tile_frac(static, 8, state=_make_state(skin_water=0.0))
        ktvl, ktvh = m._dominant_veg_types(static)
        wlmx = m._interception_capacity(static, frac, ktvl, ktvh)
        # At skin_water = PWLMX/2 the wet-skin tile should be 0.5*cover.
        sw = wlmx / 2.0
        frac2 = m._tile_frac(static, 8, state=_make_state(skin_water=float(sw[0])))
        # cover total = 0.4+0.4+0.2 = 1.0 here; wet-skin should be 0.5.
        assert frac2[:, TILE_WET_SKIN].mean().item() == pytest.approx(
            0.5, abs=1e-9)


# --------------------------------------------------------------------------- #
# End-to-end forward integration
# --------------------------------------------------------------------------- #
class TestForwardIntegration:
    def test_forward_runs_with_dynamic_split(self):
        m = _model()
        static = _make_static(cvl=0.4, cvh=0.4, ice_land=0.0,
                              veg_type_low=3, veg_type_high=3)
        state = _make_state(snow_frac=0.25, skin_water=0.0)
        forcing = _make_forcing()
        new_state, diag = m(state, forcing, static)
        assert torch.isfinite(new_state.skin_t).all()
        assert torch.isfinite(new_state.soil_t).all()
        assert torch.isfinite(new_state.snow_w).all()

    def test_forward_runs_dry_no_snow(self):
        m = _model()
        static = _make_static(cvl=0.3, cvh=0.3, ice_land=0.0)
        state = _make_state(snow_frac=0.0, skin_water=0.0)
        forcing = _make_forcing()
        new_state, diag = m(state, forcing, static)
        assert torch.isfinite(new_state.skin_t).all()

    def test_differentiable_through_split(self):
        """The split must be autograd-traceable (uses differentiable=True).

        The cover inputs are the natural optimisable leaves; mark them
        requires_grad and confirm a backward pass reaches them.
        """
        m = _model()
        m.config.differentiable = True
        cvl = torch.full((2,), 0.4, dtype=torch.float64, requires_grad=True)
        cvh = torch.full((2,), 0.4, dtype=torch.float64, requires_grad=True)
        static = _make_static(ice_land=0.0)
        static.cvl_frac = cvl
        static.cvh_frac = cvh
        state = _make_state(snow_frac=0.25, skin_water=0.1)
        forcing = _make_forcing()
        new_state, _ = m(state, forcing, static)
        loss = new_state.skin_t.sum()
        loss.backward()
        assert cvl.grad is not None
        assert torch.isfinite(cvl.grad).all()
        assert cvh.grad is not None
        assert torch.isfinite(cvh.grad).all()


# --------------------------------------------------------------------------- #
# Vectorisation
# --------------------------------------------------------------------------- #
class TestVectorisation:
    def test_per_column_independent(self):
        """Different columns get different splits driven by their covers."""
        m = _model()
        N = 4
        static = Static(
            tile_frac_base=torch.zeros((N, 8), dtype=torch.float64),
            land_frac=torch.ones(N, dtype=torch.float64),
            orography=torch.full((N,), 50.0, dtype=torch.float64),
            soil_type=torch.full((N,), 3, dtype=torch.long),
            veg_type=torch.full((N,), 3, dtype=torch.long),
            soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89],
                                        dtype=torch.float64),
            cvl_frac=torch.tensor([0.4, 0.0, 0.3, 0.5], dtype=torch.float64),
            cvh_frac=torch.tensor([0.4, 0.0, 0.5, 0.3], dtype=torch.float64),
            ice_land_frac=torch.zeros(N, dtype=torch.float64),
        )
        state = SurfState(
            skin_t=torch.full((N, 8), 285.0, dtype=torch.float64),
            soil_t=torch.full((N, 4), 283.0, dtype=torch.float64),
            soil_w_liq=torch.full((N, 4), 0.30, dtype=torch.float64),
            snow_w=torch.zeros(N, dtype=torch.float64),
            snow_frac=torch.zeros(N, dtype=torch.float64),
            skin_water=torch.zeros(N, dtype=torch.float64),
        )
        frac = m._tile_frac(static, 8, state=state)
        # Column 1 (all bare): bare-soil tile = 1.
        assert frac[1, TILE_BARE_SOIL].item() == pytest.approx(1.0, abs=1e-12)
        # Column 0 (cvl=cvh=0.4): corrected low-dry = 0.36.
        assert frac[0, TILE_LOW_VEG_DRY].item() == pytest.approx(0.36, abs=1e-12)
        # Column 3 (cvl=0.5, cvh=0.3): corrected bare = 1 - 0.45 - 0.27 = 0.28.
        assert frac[3, TILE_BARE_SOIL].item() == pytest.approx(0.28, abs=1e-12)
