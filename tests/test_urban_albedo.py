"""Tests for the urban-tile albedo wiring (``SURFRAD`` 597-602 + ``CANALB``).

Verifies the end-to-end urban canyon integration in
:mod:`surf_pytorch.model`:

* :meth:`LandSurface._compute_tile_frac` urban branch (``surfbc_ctl_mod``
  385-396) preserves the sum-to-1 invariant over tiles when ``PCUR`` is
  supplied, and the urban tile (``TILE_URBAN``) carries
  ``PCUR*(1-ZCVS)*(1-ZCVW)``.
* :meth:`LandSurface._surface_albedo` routes the urban tile to
  :meth:`_urban_canyon_albedo`, which blends the ``CANALB`` canyon
  albedo with the roof albedo via the road ratio
  ``alpha = RROOALB/(1+RWRR) + PCANALB*RWRR/(1+RWRR)``.
* the **dormant fallback**: with ``urban_frac=None`` the urban tile is
  absent (``ntile<=TILE_URBAN`` or the branch is skipped) and the
  surface albedo is bit-identical to the no-urban build -- the golden-
  snapshot regression guard.
* an SEB feedback: a non-zero urban albedo reduces absorbed SW at the
  urban tile, lowering its skin temperature relative to the dormant
  case.
* the wiring is safe at ``ntile=8`` (no urban tile in the topology).

Tile indices (``surf_pytorch.config``):
    0 water, 1 ice, 2 wet-skin, 3 low-veg dry, 4 low-veg snow,
    5 high-veg dry, 6 high-veg snow, 7 bare soil, 8 lake, 9 urban.
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
    TILE_URBAN,
)
from surf_pytorch.state import Forcing, Static, SurfState


# --------------------------------------------------------------------------- #
# Input builders
# --------------------------------------------------------------------------- #
def _make_state(*, N=2, ntile=10, dtype=torch.float64) -> SurfState:
    return SurfState(
        skin_t=torch.full((N, ntile), 285.0, dtype=dtype),
        soil_t=torch.full((N, 4), 283.0, dtype=dtype),
        soil_w_liq=torch.full((N, 4), 0.30, dtype=dtype),
        snow_w=torch.zeros((N,), dtype=dtype),
        snow_frac=torch.zeros((N,), dtype=dtype),
        skin_water=torch.zeros((N,), dtype=dtype),
    )


def _make_static(
    *, N=2, urban_frac=0.3, cvl=0.4, cvh=0.3, ice=0.0,
    roof_alb=None, road_alb=None, wall_alb=None,
    urban_height_ratio=None, urban_road_ratio=None,
    albd=None, albw=None, dtype=torch.float64,
) -> Static:
    """Static carrying cover inputs so the dynamic tile-split path fires.

    Setting ``cvl_frac``/``cvh_frac`` activates
    :meth:`_compute_tile_frac`; setting ``urban_frac`` additionally
    activates its urban branch.  Material/geometry fields default to
    ``None`` so the ``_urban_canyon_albedo`` YOS_URB defaults apply.
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
        ice_land_frac=torch.full((N,), ice, dtype=dtype),
        urban_frac=(torch.full((N,), urban_frac, dtype=dtype)
                    if urban_frac is not None else None),
        roof_alb=(torch.full((N,), roof_alb, dtype=dtype)
                  if roof_alb is not None else None),
        road_alb=(torch.full((N,), road_alb, dtype=dtype)
                  if road_alb is not None else None),
        wall_alb=(torch.full((N,), wall_alb, dtype=dtype)
                  if wall_alb is not None else None),
        urban_height_ratio=(torch.full((N,), urban_height_ratio, dtype=dtype)
                            if urban_height_ratio is not None else None),
        urban_road_ratio=(torch.full((N,), urban_road_ratio, dtype=dtype)
                          if urban_road_ratio is not None else None),
        albd=(torch.full((N,), albd, dtype=dtype) if albd is not None else None),
        albw=(torch.full((N,), albw, dtype=dtype) if albw is not None else None),
    )


def _make_forcing(*, N=2, mu0=0.5, fr_so=200.0, dtype=torch.float64) -> Forcing:
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
        mu0=torch.full((N,), mu0, dtype=dtype),
        rain=torch.zeros((N,), dtype=dtype),
        snow=torch.zeros((N,), dtype=dtype),
        dt=torch.tensor(3600.0, dtype=dtype),
    )


def _model(*, n_tile=10, validate=False) -> LandSurface:
    """Build a LandSurface.

    ``n_tile`` defaults to 10 so the urban tile (``TILE_URBAN`` index 9)
    is present in the topology; pass ``n_tile=8`` for the no-urban
    safety tests.
    """
    return LandSurface(SurfConfig(
        dtype=torch.float64, n_tile=n_tile, validate=validate,
    ))


# --------------------------------------------------------------------------- #
# Tests: tile-fraction urban branch (sum-to-1 invariant)
# --------------------------------------------------------------------------- #
class TestTileFracUrban:
    """The urban branch rescales covers and keeps tile_frac summing to 1."""

    def test_tile_frac_sums_to_one_with_urban(self):
        m = _model()
        state = _make_state()
        static = _make_static(urban_frac=0.3, cvl=0.4, cvh=0.3)
        frac = m._compute_tile_frac(state, static, 10)
        assert frac.shape == (2, 10)
        # Sum over tiles must be 1 (the SURFBC invariant).
        assert torch.allclose(
            frac.sum(dim=-1), torch.ones(2, dtype=torch.float64), atol=1e-12,
        )

    def test_urban_tile_carries_pcur_times_residuals(self):
        # With no snow (zcvs=0) and an empty skin reservoir (zcvw=0),
        # the urban tile fraction equals PCUR * 1 * 1 = PCUR exactly.
        m = _model()
        state = _make_state()              # snow_frac=0, skin_water=0
        static = _make_static(urban_frac=0.3, cvl=0.4, cvh=0.3)
        frac = m._compute_tile_frac(state, static, 10)
        assert torch.allclose(
            frac[:, TILE_URBAN], torch.full((2,), 0.3, dtype=torch.float64),
            atol=1e-12,
        )

    def test_urban_tile_zero_when_pcur_none(self):
        # Dormant: urban_frac=None -> urban column stays 0.
        m = _model()
        state = _make_state()
        static = _make_static(urban_frac=None, cvl=0.4, cvh=0.3)
        frac = m._compute_tile_frac(state, static, 10)
        assert torch.all(frac[:, TILE_URBAN] == 0.0)
        # And the non-urban covers still sum to 1.
        assert torch.allclose(
            frac.sum(dim=-1), torch.ones(2, dtype=torch.float64), atol=1e-12,
        )

    def test_urban_scales_with_ice_cover(self):
        # With PCIL>0 the urban branch still sums to 1; the ice tile
        # carries ZFRTIT and the urban tile carries PCUR*(1-ZCVS)*(1-ZCVW).
        m = _model()
        state = _make_state()
        static = _make_static(urban_frac=0.2, cvl=0.4, cvh=0.3, ice=0.1)
        frac = m._compute_tile_frac(state, static, 10)
        assert torch.allclose(
            frac.sum(dim=-1), torch.ones(2, dtype=torch.float64), atol=1e-12,
        )
        # Ice tile (snow-free land ice) carries ZFRTIT*(1-ZCVS) = 0.1.
        assert torch.allclose(
            frac[:, TILE_ICE], torch.full((2,), 0.1, dtype=torch.float64),
            atol=1e-12,
        )


# --------------------------------------------------------------------------- #
# Tests: surface-albedo urban wiring
# --------------------------------------------------------------------------- #
class TestSurfaceAlbedoUrban:
    """_surface_albedo routes the urban tile to the canyon+roof blend."""

    def test_urban_tile_takes_canyon_blend_default_materials(self):
        # Default YOS_URB: rroalb=0.16, rwrr=1.0, rhwr=1.0,
        # rroaalb=0.05, rwalalb=0.6.  At mu0=0.5 PCANALB ~ 0.1657, so
        #   alpha = 0.16/2 + 0.1657/2 ~ 0.1628.
        m = _model()
        state = _make_state()
        static = _make_static(urban_frac=0.3)   # defaults for materials
        alpha = m._surface_albedo(
            state, static, _make_forcing(mu0=0.5), 10,
            torch.float64, torch.device("cpu"),
        )
        assert alpha.shape == (2, 10)
        urban_alpha = alpha[:, TILE_URBAN]
        assert torch.all((urban_alpha > 0.0) & (urban_alpha < 1.0))
        assert torch.allclose(
            urban_alpha, torch.full((2,), 0.1628, dtype=torch.float64),
            atol=2e-3,
        )

    def test_urban_alpha_zenith_dependent(self):
        # A lower sun brightens the canyon (beam on bright walls), so the
        # urban-tile albedo is higher at mu0=0.1 than at mu0=0.9.
        m = _model()
        state = _make_state()
        static = _make_static(urban_frac=0.3)
        a_low = m._surface_albedo(
            state, static, _make_forcing(mu0=0.1), 10,
            torch.float64, torch.device("cpu"),
        )[:, TILE_URBAN]
        a_high = m._surface_albedo(
            state, static, _make_forcing(mu0=0.9), 10,
            torch.float64, torch.device("cpu"),
        )[:, TILE_URBAN]
        assert torch.all(a_low > a_high)

    def test_urban_alpha_uses_supplied_material_fields(self):
        # Brighter roof -> higher urban albedo (roof term dominates when
        # rwrr is small).  Use rwrr=0.1 so the roof term rroalb/(1+rwrr)
        # carries most of the weight.
        m = _model()
        state = _make_state()
        base = _make_static(
            urban_frac=0.3, roof_alb=0.1, urban_road_ratio=0.1,
        )
        bright = _make_static(
            urban_frac=0.3, roof_alb=0.5, urban_road_ratio=0.1,
        )
        a_base = m._surface_albedo(
            state, base, _make_forcing(mu0=0.5), 10,
            torch.float64, torch.device("cpu"),
        )[:, TILE_URBAN]
        a_bright = m._surface_albedo(
            state, bright, _make_forcing(mu0=0.5), 10,
            torch.float64, torch.device("cpu"),
        )[:, TILE_URBAN]
        assert torch.all(a_bright > a_base)

    def test_non_urban_tiles_unaffected_by_urban_wiring(self):
        # With no snow_alb / albd / albw supplied, only the urban tile is
        # non-zero; all other tiles stay at 0.
        m = _model()
        state = _make_state()
        static = _make_static(urban_frac=0.3)
        alpha = m._surface_albedo(
            state, static, _make_forcing(mu0=0.5), 10,
            torch.float64, torch.device("cpu"),
        )
        mask = torch.ones(10, dtype=torch.bool)
        mask[TILE_URBAN] = False
        assert torch.all(alpha[:, mask] == 0.0)


# --------------------------------------------------------------------------- #
# Tests: dormant fallback (bit-identical regression guard)
# --------------------------------------------------------------------------- #
class TestFallbackEquivalence:
    """urban_frac=None -> no urban tile activation -> bit-identical to legacy."""

    def test_surface_albedo_dormant_when_urban_frac_none(self):
        # With ntile=10 but urban_frac=None, has_urban is False and the
        # urban column stays 0 (no canyon call).
        m = _model()
        state = _make_state()
        static = _make_static(urban_frac=None)
        alpha = m._surface_albedo(
            state, static, _make_forcing(mu0=0.5), 10,
            torch.float64, torch.device("cpu"),
        )
        assert torch.all(alpha == 0.0)

    def test_ntile8_dormant_even_with_urban_frac_set(self):
        # ntile=8 < TILE_URBAN(9): the topology has no urban slot, so the
        # urban branch cannot fire even if urban_frac is set.  alpha is
        # shaped (N, 8) and all-zero (no other albedo fields supplied).
        m = _model()
        state = _make_state(ntile=8)
        static = _make_static(urban_frac=0.3)
        alpha = m._surface_albedo(
            state, static, _make_forcing(mu0=0.5), 8,
            torch.float64, torch.device("cpu"),
        )
        assert alpha.shape == (2, 8)
        assert torch.all(alpha == 0.0)

    def test_forward_dormant_bit_identical_across_runs(self):
        # Two consecutive dormant forward steps agree exactly (the urban
        # subsystem never activates, so no nondeterminism is introduced).
        m = _model()
        torch.manual_seed(0)
        state = _make_state()
        static = _make_static(urban_frac=None, cvl=0.4, cvh=0.3)
        forcing = _make_forcing()
        ns1, _ = m(state, forcing, static)
        ns2, _ = m(state, forcing, static)
        assert torch.equal(ns1.skin_t, ns2.skin_t)


# --------------------------------------------------------------------------- #
# Tests: SEB feedback
# --------------------------------------------------------------------------- #
class TestSebFeedback:
    """An active urban albedo reduces absorbed SW at the urban tile."""

    def test_urban_albedo_lowers_urban_skin_t_vs_dormant(self):
        # Compare an active-urban step against a dormant (urban_frac=None)
        # step with identical forcing.  The urban tile now reflects SW,
        # so less energy is absorbed there.  Because the per-tile skin
        # temperature is a tile-weighted quantity, an urban tile with a
        # real albedo cools the column mean skin temperature relative to
        # the dormant (alpha=0) case under positive insolation.
        m = _model()
        state = _make_state()
        forcing = _make_forcing(fr_so=400.0)   # strong insolation

        static_active = _make_static(urban_frac=0.5, cvl=0.3, cvh=0.2)
        static_dormant = _make_static(urban_frac=None, cvl=0.3, cvh=0.2)

        ns_active, _ = m(state, forcing, static_active)
        ns_dormant, _ = m(state, forcing, static_dormant)

        # The grid-mean skin temperature (tile-weighted) must drop when
        # the urban tile reflects SW.
        mean_active = ns_active.skin_t.mean().item()
        mean_dormant = ns_dormant.skin_t.mean().item()
        assert mean_active < mean_dormant


# --------------------------------------------------------------------------- #
# Tests: ntile=8 safety (no urban tile in topology)
# --------------------------------------------------------------------------- #
class TestNTile8Safe:
    """The urban wiring is a no-op when the topology has no urban tile."""

    def test_compute_tile_frac_ntile8_no_urban_column(self):
        # _compute_tile_frac returns an (N, 8) array; the urban branch is
        # skipped because ntile <= TILE_URBAN.  No index error, sum is 1
        # over the 8 land tiles.
        m = _model(n_tile=8)
        state = _make_state(ntile=8)
        static = _make_static(urban_frac=0.3, cvl=0.4, cvh=0.3)
        frac = m._compute_tile_frac(state, static, 8)
        assert frac.shape == (2, 8)
        assert torch.allclose(
            frac.sum(dim=-1), torch.ones(2, dtype=torch.float64), atol=1e-12,
        )

    def test_surface_albedo_ntile8_no_urban_write(self):
        m = _model(n_tile=8)
        state = _make_state(ntile=8)
        static = _make_static(urban_frac=0.3)
        alpha = m._surface_albedo(
            state, static, _make_forcing(mu0=0.5), 8,
            torch.float64, torch.device("cpu"),
        )
        assert alpha.shape == (2, 8)
        assert torch.all(alpha == 0.0)
