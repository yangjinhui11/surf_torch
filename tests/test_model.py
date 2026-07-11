"""End-to-end smoke tests for :class:`surf_pytorch.LandSurface`.

These exercise the public ``forward`` API with self-consistent inputs
and check the high-level invariants:

* output shapes match the input batch / tile geometries;
* the ocean tile skin-T pins to SST in the ``lambda_skin = LARGE`` limit
  (the SEB closure forces ``Tsk = Tsrf`` there);
* the SEB energy balance closure holds per tile;
* the model is chainable across multiple timesteps;
* autograd runs end-to-end when ``differentiable=True``;
* the ``validate=True`` default safety net (``_validate_state``) passes
  on a sane cold start and raises on each of its failure modes.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from surf_pytorch import SurfConfig, ConstantBundle, LandSurface
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.physics.seb import _tile_solar_transmission


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_inputs(
    *, N: int = 4, dtype: torch.dtype = torch.float64,
    differentiable: bool = False,
) -> tuple[SurfState, Forcing, Static, SurfConfig]:
    """Build a minimal self-consistent (state, forcing, static) triple."""
    cfg = SurfConfig(dtype=dtype, differentiable=differentiable)
    ntile, nsoil = cfg.n_tile, cfg.n_soil

    state = SurfState(
        skin_t=torch.full((N, ntile), 288.0, dtype=dtype),
        soil_t=torch.full((N, nsoil), 285.0, dtype=dtype),
        soil_w_liq=torch.full((N, nsoil), 0.3, dtype=dtype),
        snow_w=torch.zeros((N,), dtype=dtype),
        snow_frac=torch.zeros((N,), dtype=dtype),
        skin_water=torch.zeros((N,), dtype=dtype),
        sst=torch.full((N,), 290.0, dtype=dtype),
    )
    forcing = Forcing(
        u=torch.full((N,), 5.0, dtype=dtype),
        v=torch.zeros((N,), dtype=dtype),
        t=torch.full((N,), 290.0, dtype=dtype),
        q=torch.full((N,), 0.008, dtype=dtype),
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
    # Tile fractions: water=0.5, low-veg=0.2, high-veg=0.2, bare-soil=0.1.
    frac = torch.zeros((N, ntile), dtype=dtype)
    frac[:, 0] = 0.5
    frac[:, 3] = 0.2
    frac[:, 5] = 0.2
    frac[:, 7] = 0.1
    static = Static(
        tile_frac_base=frac,
        land_frac=torch.full((N,), 0.5, dtype=dtype),
        orography=torch.full((N,), 50.0, dtype=dtype),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=torch.full((N,), 3, dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.full((N,), 0.99, dtype=dtype),
    )
    return state, forcing, static, cfg


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestLandSurfaceForward:
    """End-to-end checks of the public ``LandSurface.forward`` API."""

    def test_output_shapes(self):
        state, forcing, static, cfg = _make_inputs(N=4)
        model = LandSurface(cfg)
        new_state, diag = model(state, forcing, static)
        assert new_state.skin_t.shape == state.skin_t.shape
        assert diag.h_flux_tile.shape == state.skin_t.shape
        assert diag.h_flux.shape == (4,)
        assert diag.e_flux.shape == (4,)
        assert diag.net_rad.shape == (4,)
        assert diag.ground_heat.shape == (4,)

    def test_ocean_tile_pins_to_sst(self, atol_loose):
        # lambda_skin=LARGE for the water tile forces the *instantaneous*
        # SEB skin T (PTSK) to SST.  forward() carries the t+1 extrapolation
        # PTSKTIP1 = (1/RVDIFTS)*PTSK + (1-1/RVDIFTS)*PTSKTI as the
        # prognostic state (matches the upstream vdfdifh1s.F90 contract:
        # PTSKTIP1 is the INTENT(OUT) field the driver writes back to
        # PTSKTI).  The offline single-column driver pins RVDIFTS=1.0
        # (surf/offline/driver/suvdf.F90:57), so the over-implicit weight
        # collapses: ZTPFAC2=1.0, ZTPFAC3=0.0, and PTSKTIP1 = PTSK exactly
        # -- the water tile reaches SST in a single step.  (With the full-
        # model RVDIFTS=1.5 a single step would mix to (2/3)*SST+(1/3)*288
        # and only converge to SST over several steps.)
        state, forcing, static, cfg = _make_inputs(N=2)
        model = LandSurface(cfg)

        # (1) Single step: with RVDIFTS=1.0 PTSKTIP1 = PTSK = SST exactly.
        new_state, _ = model(state, forcing, static)
        sst = state.sst
        water_t = new_state.skin_t[:, 0]
        expected = sst.clone()
        assert torch.allclose(water_t, expected, atol=atol_loose), (
            f"water PTSKTIP1 = {water_t.tolist()}, "
            f"expected SST (RVDIFTS=1.0 -> PTSKTIP1=PTSK=SST) = {expected.tolist()}"
        )

        # (2) Multi-step convergence: after several steps of steady
        #     forcing the over-implicit relaxation drives PTSKTI toward
        #     SST.  Use the new_state as the next cold-start so the
        #     cold-start term converges to SST geometrically.
        s = state
        for _ in range(20):
            s, _ = model(s, forcing, static)
        assert torch.allclose(s.skin_t[:, 0], sst, atol=1e-6), (
            f"water skin T did not converge to SST after 20 steps: "
            f"{s.skin_t[:, 0].tolist()} vs SST {sst.tolist()}"
        )

    def test_state_is_not_mutated(self):
        state, forcing, static, cfg = _make_inputs(N=2)
        original = state.skin_t.clone()
        model = LandSurface(cfg)
        _ = model(state, forcing, static)
        assert torch.equal(state.skin_t, original), (
            "forward must not mutate the input SurfState"
        )

    def test_seb_closure_per_tile(self):
        # The forward-run SEB must satisfy the per-tile closure
        # SSR_eff + STR + SSH + SLH - G0 = 0 to rounding level,
        # where SSR_eff = SSR * FRSR (per-tile SW transmission factor).
        state, forcing, static, cfg = _make_inputs(N=3)
        model = LandSurface(cfg)
        new_state, diag = model(state, forcing, static)

        dt = forcing.dt
        ssr_t = (forcing.fr_so / dt).unsqueeze(-1).expand_as(diag.h_flux_tile)
        ngrid, ntile = diag.h_flux_tile.shape
        frsr = _tile_solar_transmission(ngrid, ntile, ssr_t.dtype, ssr_t.device)
        resid = ssr_t * frsr + diag.str_tile + diag.h_flux_tile \
            + diag.slh_tile - diag.g0_tile
        assert resid.abs().max().item() < 1e-3, (
            f"per-tile SEB residual max = {resid.abs().max().item():.3e}"
        )

    def test_chainable_across_steps(self):
        # The model can be called repeatedly with the previous output.
        state, forcing, static, cfg = _make_inputs(N=2)
        model = LandSurface(cfg)
        for _ in range(3):
            state, diag = model(state, forcing, static)
            assert torch.isfinite(state.skin_t).all()
        # After 3 steps with steady forcing, skin T should be in a
        # physically-sane band.
        assert state.skin_t.min().item() > 250.0
        assert state.skin_t.max().item() < 320.0

    def test_grad_runs_end_to_end(self):
        # differentiable=True -> autograd graph is built and a scalar
        # loss on skin T backprops to forcing inputs that require grad.
        state, forcing, static, cfg = _make_inputs(
            N=2, differentiable=True,
        )
        # Mark t-1 skin T as a leaf we want gradients for.
        state.skin_t.requires_grad_(True)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        loss = new_state.skin_t.sum()
        loss.backward()
        assert state.skin_t.grad is not None
        assert torch.isfinite(state.skin_t.grad).all()
        assert state.skin_t.grad.abs().sum() > 0

    def test_no_grad_default(self):
        # differentiable=False (default) -> no graph built.
        state, forcing, static, cfg = _make_inputs(N=2)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        assert not new_state.skin_t.requires_grad


# --------------------------------------------------------------------------- #
# SurfConfig.validate (the default safety net)
# --------------------------------------------------------------------------- #
class TestValidateGuard:
    """Cover the ``validate=True`` default path and its failure modes.

    The default ``SurfConfig(validate=True)`` runs
    :meth:`LandSurface._validate_state` after every forward step -- the
    only runtime invariant guard on the default config.  Every other
    test class disables it (``validate=False``), so the guard itself is
    otherwise never exercised: neither that it passes on a sane cold
    start, nor that it raises ``AssertionError`` when each of its three
    invariants (tile-fraction sum, finiteness, skin-T band) is broken.
    """

    def test_default_config_has_validate_on(self):
        # Guard against silently flipping the default off.
        assert SurfConfig().validate is True

    def test_validate_passes_on_sane_cold_start(self):
        """validate=True must run cleanly on the canonical cold-start slab.

        The ``_make_inputs`` slab has water + land tiles summing to 1.0
        and finite forcing, so the default-config forward must complete
        and return a finite, in-band skin T without raising.
        """
        state, forcing, static, cfg = _make_inputs(N=3)
        assert cfg.validate is True
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        assert torch.isfinite(new_state.skin_t).all()
        # validate already asserted the 150/350 band internally; the
        # fact that forward returned means all three checks passed.

    def test_validate_off_disables_guard(self):
        """validate=False must not raise even on a broken tile-fraction sum.

        The guard is gated purely on ``config.validate``; with it off a
        slab whose tiles do not sum to 1 must still run (the physics
        does not itself enforce the sum).
        """
        state, forcing, static, _ = _make_inputs(N=2)
        static_bad = dataclasses.replace(
            static, tile_frac_base=static.tile_frac_base.clone(),
        )
        static_bad.tile_frac_base[:, 0] = 0.3   # sum -> 0.8
        model = LandSurface(SurfConfig(dtype=torch.float64, validate=False))
        # Must not raise.
        new_state, _ = model(state, forcing, static_bad)
        assert torch.isfinite(new_state.skin_t).all()

    def test_raises_on_tile_fractions_not_summing_to_one(self):
        """Tiles summing to != 1 trips the tile-fraction invariant."""
        state, forcing, static, _ = _make_inputs(N=2)
        static_bad = dataclasses.replace(
            static, tile_frac_base=static.tile_frac_base.clone(),
        )
        static_bad.tile_frac_base[:, 0] = 0.3   # was 0.5 -> sum 0.8
        model = LandSurface(SurfConfig(dtype=torch.float64, validate=True))
        with pytest.raises(AssertionError, match="tile fractions"):
            model(state, forcing, static_bad)

    def test_raises_on_non_finite_skin_temperature(self):
        """A NaN in the air-temperature forcing propagates to a non-finite
        skin T, which the finiteness invariant must catch."""
        state, forcing, static, _ = _make_inputs(N=2)
        forcing_bad = dataclasses.replace(
            forcing, t=torch.full((2,), float("nan"), dtype=torch.float64),
        )
        model = LandSurface(SurfConfig(dtype=torch.float64, validate=True))
        with pytest.raises(AssertionError, match="non-finite skin temperature"):
            model(state, forcing_bad, static)

    def test_raises_on_skin_temperature_above_band(self):
        """Extreme hot forcing (warm air + huge downwelling SW) drives the
        skin T above the 350 K guard rail, tripping the upper-band check."""
        state, forcing, static, _ = _make_inputs(N=2)
        forcing_hot = dataclasses.replace(
            forcing,
            t=torch.full((2,), 360.0, dtype=torch.float64),
            fr_so=torch.full((2,), 5000.0 * 3600.0, dtype=torch.float64),
            fr_th=torch.full((2,), 200.0 * 3600.0, dtype=torch.float64),
        )
        model = LandSurface(SurfConfig(dtype=torch.float64, validate=True))
        with pytest.raises(AssertionError, match="skin T above 350 K"):
            model(state, forcing_hot, static)
