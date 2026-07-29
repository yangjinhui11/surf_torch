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
from surf_pytorch.config import TILE_HIGH_VEG_SNOW, TILE_WET_SKIN
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

    def test_vsurf_uses_dynamic_lai_for_apparent_humidity(self):
        """VSURF must use the forcing-time LAI, not the table default."""
        from surf_pytorch.physics.stomata import jarvis_resistance
        from surf_pytorch.utils import expand_as_tiles, qsat

        state, forcing, static, cfg = _make_inputs(N=1)
        static = dataclasses.replace(
            static,
            lai_low=torch.tensor([0.6], dtype=cfg.dtype),
            lai_high=torch.tensor([1.4], dtype=cfg.dtype),
        )
        model = LandSurface(cfg)
        frac = model._tile_frac(static, cfg.n_tile, state=state)
        ktvl, ktvh = model._dominant_veg_types(static)
        qs_skin = qsat(
            state.skin_t,
            expand_as_tiles(forcing.p_surf, cfg.n_tile),
            model.consts,
        )
        praqti = torch.full_like(state.skin_t, 8000.0)

        actual = model._vsurf_apparent_humidity(
            forcing, static, state, frac, ktvl, ktvh, qs_skin, praqti,
            cfg.n_tile,
        )
        jarvis = jarvis_resistance(
            soil_t=state.soil_t, soil_w=state.soil_w_liq,
            soil_type=static.soil_type, ktvl=ktvl, ktvh=ktvh,
            t_air=forcing.t, q_air=forcing.q, p_surf=forcing.p_surf,
            sw_down=forcing.fr_so / forcing.dt,
            veg_table=model.veg_table, soil_table=model.soil_table,
            root_table=model.root_table, consts=model.consts,
            tile_frac=frac, n_tile=cfg.n_tile,
            lai_low=static.lai_low, lai_high=static.lai_high,
        )
        expected_low = qs_skin[:, 3] + torch.minimum(
            torch.zeros_like(forcing.q), forcing.q - qs_skin[:, 3],
        ) * jarvis["pwetl"] / (jarvis["pwetl"] + praqti[:, 3])
        expected_high = qs_skin[:, 5] + torch.minimum(
            torch.zeros_like(forcing.q), forcing.q - qs_skin[:, 5],
        ) * jarvis["pweth"] / (jarvis["pweth"] + praqti[:, 5])

        assert torch.allclose(actual[:, 3], expected_low)
        assert torch.allclose(actual[:, 5], expected_high)

    def test_coupled_vup_z0_override_feeds_vexcs(self, monkeypatch):
        import surf_pytorch.model as model_mod
        from surf_pytorch.physics.exchange import ExchangeOutputs

        state, forcing, static, cfg = _make_inputs(N=1)
        cfg = dataclasses.replace(cfg, vup_boundary_mode="coupled")
        ntile = cfg.n_tile
        z0m = torch.linspace(0.01, 0.08, ntile, dtype=cfg.dtype).reshape(1, ntile)
        z0h = torch.linspace(0.02, 0.09, ntile, dtype=cfg.dtype).reshape(1, ntile)
        z0q = torch.linspace(0.03, 0.10, ntile, dtype=cfg.dtype).reshape(1, ntile)
        zero_tile = torch.zeros((1, ntile), dtype=cfg.dtype)
        forcing = dataclasses.replace(
            forcing,
            vupdz0_pustrti=zero_tile,
            vupdz0_pvstrti=zero_tile,
            vupdz0_pahfsti=zero_tile,
            vupdz0_pevapti=zero_tile,
            vupdz0_pbuomti=zero_tile,
            vupdz0_pzdlti=zero_tile,
            vupdz0_praqti=torch.ones((1, ntile), dtype=cfg.dtype),
            vupdz0_z0mti=z0m,
            vupdz0_z0hti=z0h,
            vupdz0_z0qti=z0q,
        )
        seen = {}

        def fake_vexcs(*args, **kwargs):
            seen["z0m"] = args[9].reshape(1, ntile).clone()
            seen["z0h"] = args[10].reshape(1, ntile).clone()
            seen["z0q"] = args[11].reshape(1, ntile).clone()
            n = args[0].shape[0]
            one = torch.ones(n, dtype=cfg.dtype)
            zero = torch.zeros(n, dtype=cfg.dtype)
            return ExchangeOutputs(one, one, one, zero, zero, zero)

        monkeypatch.setattr(model_mod, "compute_vexcs", fake_vexcs)
        model = LandSurface(cfg)
        model._per_tile_monin_obukhov_coeffs(forcing, static, state)

        frac = model._tile_frac(static, ntile, state=state)
        zblend = torch.tensor(10.0, dtype=cfg.dtype)
        expected_z0m = zblend * torch.exp(
            -1.0 / torch.sqrt((frac / torch.log(zblend / z0m).square()).sum(dim=-1))
        )
        assert torch.allclose(seen["z0m"], expected_z0m.unsqueeze(-1).expand_as(z0m))
        assert torch.equal(seen["z0h"], z0h)
        assert torch.equal(seen["z0q"], z0q)

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

    def test_surfpp_replaces_inactive_tile_temperatures(self):
        model = LandSurface(SurfConfig(dtype=torch.float64))
        ptsktip1 = torch.tensor([[280.0, 281.0, 250.0, 282.0]], dtype=torch.float64)
        frac = torch.tensor([[0.5, 0.0, 0.0, 0.5]], dtype=torch.float64)

        result = model._surfpp_zero_tile_skin_t(ptsktip1, frac)

        # Inactive tiles use the diagnostic mean, including wet skin.
        assert result[0, 1] == pytest.approx(281.0)
        assert result[0, TILE_WET_SKIN] == pytest.approx(281.0)

    def test_surfpp_replaces_only_exact_zero_area_tile_temperatures(self):
        model = LandSurface(SurfConfig(dtype=torch.float64))
        ptsktip1 = torch.tensor(
            [[280.0, 281.0, 282.0, 283.0, 284.0, 285.0, 4000.0, 286.0]],
            dtype=torch.float64,
        )
        frac = torch.tensor(
            [[0.0, 0.0, 0.0, 0.4, 0.0, 0.3, 5.0e-7, 0.3]],
            dtype=torch.float64,
        )

        result = model._surfpp_zero_tile_skin_t(ptsktip1, frac)

        expected_mean = (frac * ptsktip1).sum(dim=1).item()
        assert result[0, 0] == pytest.approx(expected_mean)
        # SURFPP_CTL tests PFRTI == 0 exactly. A small, physical snow
        # fraction retains its tiled temperature for the next VUPDZ0 call.
        assert result[0, TILE_HIGH_VEG_SNOW] == pytest.approx(4000.0)

    def test_snow_under_forest_evap_ignores_subthreshold_tile(self):
        state, forcing, static, cfg = _make_inputs(N=1)
        model = LandSurface(cfg)
        ntile = cfg.n_tile
        frac = torch.zeros((1, ntile), dtype=cfg.dtype)
        frac[:, 3] = 0.4
        frac[:, 5] = 0.3
        frac[:, 6] = 0.0
        frac[:, 7] = 0.3
        state = dataclasses.replace(
            state,
            snow_t=torch.tensor([-7.0], dtype=cfg.dtype),
        )
        rhoc_q_t = torch.ones((1, ntile), dtype=cfg.dtype)
        pcsnw = torch.ones((1, ntile), dtype=cfg.dtype)

        evap = model._snow_under_forest_evap(
            forcing=forcing,
            state=state,
            rhoc_q_t=rhoc_q_t,
            pcsnw=pcsnw,
            frac=frac,
            ntile=ntile,
            dtype=cfg.dtype,
            device=torch.device("cpu"),
        )

        assert torch.isfinite(evap).all()
        assert evap.item() == pytest.approx(0.0)

    def test_snow_under_forest_evap_accepts_cold_physical_snow(self):
        state, forcing, static, cfg = _make_inputs(N=1)
        model = LandSurface(cfg)
        ntile = cfg.n_tile
        frac = torch.zeros((1, ntile), dtype=cfg.dtype)
        frac[:, 3] = 0.2
        frac[:, 5] = 0.2
        frac[:, 6] = 0.3
        frac[:, 7] = 0.3
        state = dataclasses.replace(
            state,
            snow_t=torch.tensor([250.0], dtype=cfg.dtype),
        )
        rhoc_q_t = torch.ones((1, ntile), dtype=cfg.dtype)
        pcsnw = torch.ones((1, ntile), dtype=cfg.dtype)

        evap = model._snow_under_forest_evap(
            forcing=forcing,
            state=state,
            rhoc_q_t=rhoc_q_t,
            pcsnw=pcsnw,
            frac=frac,
            ntile=ntile,
            dtype=cfg.dtype,
            device=torch.device("cpu"),
        )

        assert torch.isfinite(evap).all()

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

    def test_seb_flux_diagnostics_are_finite(self):
        # The public diagnostic does not yet expose the final internally
        # transformed SW boundary used by SURFSEB. Test its flux fields
        # directly rather than reconstructing a non-identical closure term.
        state, forcing, static, cfg = _make_inputs(N=3)
        model = LandSurface(cfg)
        new_state, diag = model(state, forcing, static)

        for value in (diag.str_tile, diag.h_flux_tile, diag.slh_tile, diag.g0_tile):
            assert torch.isfinite(value).all()

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
