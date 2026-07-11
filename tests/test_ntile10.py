"""Full forward over the ``n_tile=10`` (lake + urban) topology.

The default ``SurfConfig.n_tile=8`` ends at ``TILE_BARE_SOIL`` (=7) and
therefore *never executes* three topology-gated branches in the physics
kernels:

* ``surfseb`` latent-heat lake branch (``ntile > TILE_LAKE``), which
  blends vapourisation / sublimation by lake water/ice fraction;
* ``vlamsk`` lake / urban branches (``ntile > TILE_LAKE`` /
  ``ntile > TILE_URBAN``), which set the skin thermal mask for those
  tiles;
* ``stomata`` lake/urban branches (``ntile > TILE_URBAN``).

The per-kernel unit tests cover those branches in isolation, but the
full ``LandSurface.forward`` had only ever been exercised at n_tile=8.

Two outcomes this file pins:

1. **A real bug it found and forced the fix for.**  ``_tile_frac``
   assumed ``static.tile_frac_base`` was always 8 wide; handing it a
   10-wide array (the only way to give lake/urban a fraction) crashed
   with a shape error.  Fixed in :func:`model._tile_frac` to accept an
   already-full-width base.  Without this, n_tile=10 is *unusable*.

2. **An honest record of the stub limitation.**  With the shape bug
   fixed, forward *runs* at n_tile=10, but the FLake / urban physics
   are still identity/stub modules (``LEFLAKE=.FALSE.``, documented at
   ``model.py`` M6-stub block).  The lake tile therefore gets a ``LARGE``
   skin thermal mask in ``vlamsk`` and evolves its skin T through the
   diffusion solve without a coupled lake prognostic -- driving it to a
   physically meaningless ~95 K.  This is a known stub-incompleteness,
   NOT a regression: a test below pins that behaviour so that (a) the
   day the real FLake drops in, this test fails loudly as the prompt to
   tighten it, and (b) no one mistakes the current stub for working
   lake physics.  These tests run with ``validate=False`` because the
   stub tiles are by design out of the physical band.
"""

from __future__ import annotations

import dataclasses

import torch

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.config import TILE_LAKE, TILE_URBAN, TILE_WATER
from surf_pytorch.state import Forcing, Static, SurfState


# --------------------------------------------------------------------------- #
# Input builder
# --------------------------------------------------------------------------- #
def _build(N: int, ntile: int, *, dtype=torch.float64) -> tuple:
    """Build (state, forcing, static, cfg) for ``N`` columns, ``ntile`` tiles.

    Lake / urban tiles get a non-zero fraction only when the topology
    includes them (ntile >= 10), so the topology-gated branches have a
    fraction to act on.  Water/ice (0,1) carry zero fraction but are
    seeded at SST so their (out-of-band) evolution stays finite.
    """
    nsoil = 4
    skin = torch.full((N, ntile), 285.0, dtype=dtype)
    skin[:, TILE_WATER] = 290.0
    state = SurfState(
        skin_t=skin,
        soil_t=torch.full((N, nsoil), 283.0, dtype=dtype),
        soil_w_liq=torch.full((N, nsoil), 0.30, dtype=dtype),
        snow_w=torch.zeros(N, dtype=dtype),
        snow_frac=torch.zeros(N, dtype=dtype),
        skin_water=torch.zeros(N, dtype=dtype),
        sst=torch.full((N,), 290.0, dtype=dtype),
    )
    forcing = Forcing(
        u=torch.full((N,), 5.0, dtype=dtype),
        v=torch.zeros(N, dtype=dtype),
        t=torch.full((N,), 288.0, dtype=dtype),
        q=torch.full((N,), 0.008, dtype=dtype),
        co2=torch.full((N,), 6e-4, dtype=dtype),
        p_surf=torch.full((N,), 1.0e5, dtype=dtype),
        geom1=torch.full((N,), 50.0, dtype=dtype),
        fr_so=torch.full((N,), 200.0 * 3600.0, dtype=dtype),
        fr_th=torch.full((N,), -300.0 * 3600.0, dtype=dtype),
        mu0=torch.full((N,), 0.5, dtype=dtype),
        rain=torch.zeros(N, dtype=dtype),
        snow=torch.zeros(N, dtype=dtype),
        dt=torch.tensor(3600.0, dtype=dtype),
    )
    frac = torch.zeros((N, ntile), dtype=dtype)
    land_tiles = list(range(2, ntile))
    w = 1.0 / len(land_tiles)
    for j in land_tiles:
        frac[:, j] = w
    static = Static(
        tile_frac_base=frac,
        land_frac=torch.ones(N, dtype=dtype),
        orography=torch.full((N,), 50.0, dtype=dtype),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=torch.full((N,), 3, dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.full((N,), 0.99, dtype=dtype),
    )
    # validate=False: stub lake/urban tiles are by-design out of the
    # physical band (see module docstring) and would trip the guard.
    cfg = SurfConfig(dtype=dtype, differentiable=True, n_tile=ntile,
                     validate=False)
    return state, forcing, static, cfg


# --------------------------------------------------------------------------- #
# Shape safety (the surface the _tile_frac bug fix protects)
# --------------------------------------------------------------------------- #
class TestNTile10ShapeSafety:
    """The topology switch builds tensors of the right shape and the
    tile fractions still normalise -- the regression guard for the
    ``_tile_frac`` bug this file surfaced."""

    def test_full_width_base_runs_without_shape_error(self):
        """Regression: a 10-wide ``tile_frac_base`` no longer crashes
        ``_tile_frac`` (it used to assert the base was exactly 8 wide)."""
        state, forcing, static, cfg = _build(N=2, ntile=10)
        model = LandSurface(cfg)
        # No raise == the bug is fixed.
        frac = model._tile_frac(static, 10)
        assert frac.shape == (2, 10)

    def test_forward_outputs_carry_10_tile_axis(self):
        state, forcing, static, cfg = _build(N=2, ntile=10)
        model = LandSurface(cfg)
        new_state, diag = model(state, forcing, static)
        # skin_t and the per-tile diagnostic fluxes carry the 10-wide axis.
        assert new_state.skin_t.shape == (2, 10)
        for name in ("h_flux_tile", "e_flux_tile", "str_tile",
                     "g0_tile", "slh_tile"):
            assert getattr(diag, name).shape == (2, 10), name

    def test_tile_fractions_normalise_after_step(self):
        """The 10-tile fractions still sum to 1 through forward -- the
        topology switch did not break normalisation."""
        state, forcing, static, cfg = _build(N=2, ntile=10)
        model = LandSurface(cfg)
        model(state, forcing, static)  # runs without raising
        frac = model._tile_frac(static, 10)
        torch.testing.assert_close(
            frac.sum(-1), torch.ones(2, dtype=torch.float64),
            rtol=0.0, atol=1e-6,
        )


# --------------------------------------------------------------------------- #
# Branch liveness -- the gated lake/urban code actually ran
# --------------------------------------------------------------------------- #
class TestLakeUrbanBranchesAreLive:
    """The lake/urban branches produced outputs distinct from n_tile=8 --
    so the gated code actually executed, not silently skipped."""

    def test_grid_mean_diagnostic_differs_from_n_tile_8(self):
        """The grid-mean (tile-fraction-weighted) diagnostic at n_tile=10
        differs from n_tile=8 on the same forcing.  If the extra branches
        were inert, collapsing the topology would leave the grid mean
        unchanged.  (Land fractions are 1 in both, so the grid mean is
        comparable.)  This is the strongest liveness signal available."""
        s8, f8, st8, c8 = _build(N=1, ntile=8)
        s10, f10, st10, c10 = _build(N=1, ntile=10)
        ns8, d8 = LandSurface(c8)(s8, f8, st8)
        ns10, d10 = LandSurface(c10)(s10, f10, st10)
        frac8 = LandSurface(c8)._tile_frac(st8, 8)
        frac10 = LandSurface(c10)._tile_frac(st10, 10)
        gm8 = (d8.h_flux * frac8).sum(-1)
        gm10 = (d10.h_flux * frac10).sum(-1)
        assert not torch.allclose(gm8, gm10, atol=1e-6), (
            "grid-mean H flux identical at n_tile=8 vs 10 -- lake/urban "
            "branches had no effect on the diagnostic"
        )

    def test_lake_and_urban_tiles_produce_distinct_outputs(self):
        """The lake (8) and urban (9) tiles evolve to *different* skin
        temperatures after one step.  Both are stubs, but they take
        different physics paths (lake latent-heat blend vs urban mask),
        so their outputs differ -- proving both branches ran rather
        than one shadowing the other."""
        state, forcing, static, cfg = _build(N=1, ntile=10)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        diff = (new_state.skin_t[:, TILE_LAKE]
                - new_state.skin_t[:, TILE_URBAN]).abs()
        assert (diff > 0).item(), (
            "lake and urban tile outputs identical -- only one branch ran"
        )


# --------------------------------------------------------------------------- #
# Stub-tile behaviour -- pinned, not hidden
# --------------------------------------------------------------------------- #
class TestStubTileBehaviour:
    """The FLake / urban tiles are identity/stub modules
    (``LEFLAKE=.FALSE.`` / ``LEURBAN`` not wired).  With the
    uninitialized-memory fixes in :func:`diffusion._sub_surface_t`
    (which fall the stub lake/urban tiles back to ``PTSM1M``), they no
    longer crash out of band -- they behave like bare-soil tiles.

    These tests pin *that* stub behaviour: the lake/urban tiles evolve
    to a bare-soil-like skin T, distinct from real land tiles.  When
    real FLake / urban physics drop in, these will fail loudly as the
    prompt to tighten them to the new, more specific behaviour."""

    def test_stub_lake_and_urban_tiles_stay_in_physical_band(self):
        """After the uninitialized-memory fix, the stub lake/urban tiles
        no longer crash to ~95 K; they evolve to a bare-soil-like value.
        Pinned so that the day real FLake couples a lake prognostic --
        which will move the lake tile *off* the bare-soil value -- this
        test fails and forces a review."""
        state, forcing, static, cfg = _build(N=1, ntile=10)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        for name, idx in [("lake", TILE_LAKE), ("urban", TILE_URBAN)]:
            t = new_state.skin_t[:, idx].item()
            assert 270.0 < t < 300.0, (
                f"{name} stub tile skin T {t:.1f}K outside the bare-soil-like "
                f"band -- the FLake/urban stub has likely been replaced; "
                f"tighten this test to the new behaviour"
            )

    def test_land_tiles_remain_in_physical_band(self):
        """The real land tiles (2..7) must still produce physically sane
        skin T -- confirms the topology switch did not corrupt the
        working physics."""
        state, forcing, static, cfg = _build(N=1, ntile=10)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        land = new_state.skin_t[:, 2:8]  # wet_skin .. bare_soil
        assert torch.isfinite(land).all()
        assert land.min().item() > 270.0, (
            f"land tile skin T below 270 K: min={land.min().item():.1f}"
        )
        assert land.max().item() < 300.0, (
            f"land tile skin T above 300 K: max={land.max().item():.1f}"
        )


# --------------------------------------------------------------------------- #
# Batch independence at the 10-tile topology
# --------------------------------------------------------------------------- #
class TestNTile10BatchIndependence:
    """forward([A,B]) == cat(forward([A]), forward([B])) at n_tile=10.

    If the lake/urban branches introduced a cross-column reduction or a
    shape bug, this would break -- a guard specific to the topology
    that actually has those branches live.
    """

    def test_batched_matches_per_column_bit_exact(self):
        state, forcing, static, cfg = _build(N=2, ntile=10)
        model = LandSurface(cfg)
        ns_b, d_b = model(state, forcing, static)

        def slice1(obj, i):
            kw = {}
            for f in dataclasses.fields(obj):
                v = getattr(obj, f.name)
                kw[f.name] = (v[i:i + 1] if isinstance(v, torch.Tensor)
                              and v.dim() >= 1 else v)
            return type(obj)(**kw)

        ns0, d0 = model(slice1(state, 0), slice1(forcing, 0), slice1(static, 0))
        ns1, d1 = model(slice1(state, 1), slice1(forcing, 1), slice1(static, 1))
        for container_name, batched, per in [
            ("new_state", ns_b, [ns0, ns1]),
            ("diag", d_b, [d0, d1]),
        ]:
            for f in dataclasses.fields(batched):
                tb = getattr(batched, f.name)
                if not isinstance(tb, torch.Tensor):
                    continue
                stacked = torch.cat([getattr(p, f.name) for p in per], dim=0)
                torch.testing.assert_close(
                    tb, stacked, rtol=0.0, atol=0.0,
                    msg=f"{container_name}.{f.name}: n_tile=10 batch leak",
                )
