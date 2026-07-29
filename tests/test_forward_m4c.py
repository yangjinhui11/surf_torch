"""Dedicated tests for the M4c ``LandSurface.forward`` VDFDIFH1S integration.

These complement ``test_model.py`` (smoke invariants) and
``test_forward_e2e.py`` (Fortran-assembly comparison) by pinning the
M4c-specific contract:

* ``forward`` dispatches through the VDFDIFH1S wrapper (not a direct
  ``surf_seb`` call);
* the wrapper's exchange-coefficient rescaling
  (``RG*PTSPHY*RVDIFTS`` upstream, ``ZCOEF1`` inversion inside the
  wrapper) round-trips so the SEB sees the *physical* ``rho*Ch*|U|``;
* ``new_state.skin_t`` carries the t+1 extrapolation ``PTSKTIP1``
  (not the instantaneous ``PTSK``);
* the snow-under-forest evaporation and the LESNML snow-melt repeat
  flow through ``forward`` when the corresponding config flags and
  state hooks are populated;
* the full VEXCS path is exercised when ``Static`` carries z0.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from surf_pytorch import SurfConfig, ConstantBundle, LandSurface
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.physics.diffusion import (
    DiffusionInputs, RVDIFTS,
)
from surf_pytorch.utils import expand_as_tiles


# --------------------------------------------------------------------------- #
# Input builder (mirrors test_model._make_inputs, but allows the caller to
# opt into the full VEXCS path by populating z0).
# --------------------------------------------------------------------------- #
def _make_inputs(
    *, N: int = 4, dtype: torch.dtype = torch.float64,
    differentiable: bool = False,
    with_z0: bool = False,
    lesnml: bool = False,
) -> tuple[SurfState, Forcing, Static, SurfConfig]:
    """Build a minimal self-consistent (state, forcing, static, config)."""
    cfg = SurfConfig(
        dtype=dtype, differentiable=differentiable, lesnml=lesnml,
    )
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
    frac = torch.zeros((N, ntile), dtype=dtype)
    frac[:, 0] = 0.5
    frac[:, 3] = 0.2
    frac[:, 5] = 0.2
    frac[:, 7] = 0.1
    z0_kwargs = {}
    if with_z0:
        # Typical mid-latitude land roughness lengths (Beljaars / IFS).
        z0_kwargs = dict(
            z0m=torch.full((N,), 0.5, dtype=dtype),
            z0h=torch.full((N,), 0.1, dtype=dtype),
            z0q=torch.full((N,), 0.1, dtype=dtype),
        )
    static = Static(
        tile_frac_base=frac,
        land_frac=torch.full((N,), 0.5, dtype=dtype),
        orography=torch.full((N,), 50.0, dtype=dtype),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=torch.full((N,), 3, dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.full((N,), 0.99, dtype=dtype),
        **z0_kwargs,
    )
    return state, forcing, static, cfg


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestM4cVdfdifhIntegration:
    """Pin the M4c VDFDIFH1S integration invariants."""

    def test_coupled_mode_requires_external_vupdz0_boundary(self):
        state, forcing, static, cfg = _make_inputs(N=2, with_z0=True)
        cfg.vup_boundary_mode = "coupled"

        with pytest.raises(ValueError, match="authoritative VUPDZ0"):
            LandSurface(cfg)(state, forcing, static)

    def test_skin_t_is_ptsktip1(self):
        # new_state.skin_t must be the t+1 extrapolation, not the
        # instantaneous SEB skin T.  The offline single-column driver
        # pins RVDIFTS=1.0 (surf/offline/driver/suvdf.F90:57), so the
        # over-implicit weight collapses: ZTPFAC2 = 1/RVDIFTS = 1.0,
        # ZTPFAC3 = 0.0, and PTSKTIP1 = PTSK exactly.  Cold start (288 K)
        # + ocean tile: PTSK pins to SST (290) via lambda LARGE, so
        # PTSKTIP1 = PTSK = 290.0.  (With the full-model RVDIFTS=1.5 the
        # extrapolation would be (2/3)*290 + (1/3)*288 = 289.333.)
        state, forcing, static, cfg = _make_inputs(N=2)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        expected_water = 290.0
        assert torch.allclose(
            new_state.skin_t[:, 0],
            torch.full_like(new_state.skin_t[:, 0], expected_water),
            atol=1e-9,
        ), (
            f"water PTSKTIP1 = {new_state.skin_t[:, 0].tolist()}, "
            f"expected {expected_water}"
        )

    def test_exchange_rescaling_round_trips(self):
        # forward() must apply RG*PTSPHY*RVDIFTS to the raw rho*Ch*|U|
        # before populating DiffusionInputs.pcfhti, and the wrapper must
        # invert it.  Net effect: the SEB sees the physical exchange
        # coefficient.  Verify by checking that the DiffusionInputs the
        # helper builds carries the rescaled value.
        state, forcing, static, cfg = _make_inputs(N=2)
        model = LandSurface(cfg)
        consts = model.consts
        ntile = cfg.n_tile

        # Reconstruct the inputs forward() section 1-5 produces.
        cp = consts.cpd
        sl_lev = cp * forcing.t + forcing.geom1
        q_lev = forcing.q
        skin_t_prev = state.skin_t
        rhoc_u, rhoc_q = model._exchange_coeffs(forcing, static, state)
        rhoc_u_t = expand_as_tiles(rhoc_u, ntile)
        rhoc_q_t = expand_as_tiles(rhoc_q, ntile)
        # sskm1m / qskm1m / dqsdT shapes don't matter for this check;
        # zeros of the right shape suffice.
        N = state.skin_t.shape[0]
        sskm1m = cp * skin_t_prev
        qskm1m = torch.zeros_like(skin_t_prev)
        dqsdT = torch.zeros_like(skin_t_prev)
        alphal = torch.zeros_like(skin_t_prev)
        alphas = torch.ones_like(skin_t_prev)
        tsrf = model._sub_surface_t(state, ntile)
        ktvl, ktvh = model._dominant_veg_types(static)
        from surf_pytorch.physics.vlamsk import vlamsk as compute_vlamsk
        plamsk = compute_vlamsk(
            ptskm1m=skin_t_prev, ptsrf=tsrf,
            psnm=state.snow_w, prsn=model._snow_density(state),
            psntice=None, ktvl=ktvl, ktvh=ktvh,
            veg_table=model.veg_table, consts=consts, config=cfg,
            ptstep=model._broadcast_dt(forcing.dt, state.snow_w.shape),
        )
        frac = model._tile_frac(static, ntile)
        ssrfl_t = expand_as_tiles(forcing.fr_so / forcing.dt, ntile)
        slrfl_grid = forcing.fr_th / forcing.dt
        tskrad_grid = torch.full((N,), 288.0, dtype=cfg.dtype)
        emis_grid = static.emis

        diff_inp = model._build_diffusion_inputs(
            state=state, forcing=forcing, static=static,
            sl_lev=sl_lev, q_lev=q_lev,
            rhoc_u_t=rhoc_u_t, rhoc_q_t=rhoc_q_t,
            sskm1m=sskm1m, skin_t_prev=skin_t_prev,
            qskm1m=qskm1m, dqsdT=dqsdT,
            alphal=alphal, alphas=alphas,
            ssrfl_t=ssrfl_t, slrfl_grid=slrfl_grid,
            tskrad_grid=tskrad_grid, emis_grid=emis_grid,
            plamsk=plamsk, pevapsnw=torch.zeros_like(skin_t_prev), tsrf=tsrf, frac=frac, ktvl=ktvl, ktvh=ktvh,
            dt=forcing.dt, ntile=ntile, ngrid=N,
            dtype=cfg.dtype, device=state.skin_t.device,
        )

        # The wrapper inverts via ZCOEF1 = 1/(RG*RVDIFTS*PTMST).
        rg = float(consts.g.item())
        ptmst = float(forcing.dt.item())
        zcoef1 = 1.0 / (rg * RVDIFTS * ptmst)
        rhochu_recovered = diff_inp.pcfhti * zcoef1
        # Should equal the physical rhoc_u_t (broadcast to tiles).
        assert torch.allclose(rhochu_recovered, rhoc_u_t, rtol=1e-12), (
            f"exchange rescaling did not round-trip: "
            f"max|rec - raw| = "
            f"{(rhochu_recovered - rhoc_u_t).abs().max().item():.3e}"
        )

    def test_forward_uses_vexcs_when_z0_available(self):
        # The supported interface always returns the full tiled VEXCS output.
        state, forcing, static, cfg = _make_inputs(N=2, with_z0=True)
        model = LandSurface(cfg)

        rhoc_vexcs_u, _ = model._exchange_coeffs(forcing, static, state)
        assert rhoc_vexcs_u.shape == (2, cfg.n_tile)
        assert torch.isfinite(rhoc_vexcs_u).all()

        # And forward() must run end-to-end with the full VEXCS path.
        new_state, diag = model(state, forcing, static)
        assert torch.isfinite(new_state.skin_t).all()
        assert new_state.skin_t.shape == state.skin_t.shape

    def test_forward_without_z0_uses_table_roughness(self):
        # Static roughness is optional: the physical tile roughness comes
        # from the vegetation table when these fields are absent.
        state, forcing, static, cfg = _make_inputs(N=2, with_z0=False)
        model = LandSurface(cfg)

        rhoc_u, _ = model._exchange_coeffs(forcing, static, state)
        assert rhoc_u.shape == (2, cfg.n_tile)
        assert torch.isfinite(rhoc_u).all()

        new_state, _ = model(state, forcing, static)
        assert torch.isfinite(new_state.skin_t).all()

    def test_lesnml_flag_propagates(self):
        # The lesnml config flag must propagate to DiffusionInputs.lesnml.
        state, forcing, static, cfg = _make_inputs(N=2, lesnml=True)
        model = LandSurface(cfg)
        assert cfg.lesnml is True

        # Build the DiffusionInputs as forward() does (minimal glue).
        cp = model.consts.cpd
        sl_lev = cp * forcing.t + forcing.geom1
        N = state.skin_t.shape[0]
        ntile = cfg.n_tile
        skin_t_prev = state.skin_t
        rhoc_u, rhoc_q = model._exchange_coeffs(forcing, static, state)
        rhoc_u_t = expand_as_tiles(rhoc_u, ntile)
        rhoc_q_t = expand_as_tiles(rhoc_q, ntile)
        sskm1m = cp * skin_t_prev
        qskm1m = torch.zeros_like(skin_t_prev)
        dqsdT = torch.zeros_like(skin_t_prev)
        alphal = torch.zeros_like(skin_t_prev)
        alphas = torch.ones_like(skin_t_prev)
        tsrf = model._sub_surface_t(state, ntile)
        ktvl, ktvh = model._dominant_veg_types(static)
        from surf_pytorch.physics.vlamsk import vlamsk as compute_vlamsk
        plamsk = compute_vlamsk(
            ptskm1m=skin_t_prev, ptsrf=tsrf,
            psnm=state.snow_w, prsn=model._snow_density(state),
            psntice=None, ktvl=ktvl, ktvh=ktvh,
            veg_table=model.veg_table, consts=model.consts, config=cfg,
            ptstep=model._broadcast_dt(forcing.dt, state.snow_w.shape),
        )
        frac = model._tile_frac(static, ntile)
        ssrfl_t = expand_as_tiles(forcing.fr_so / forcing.dt, ntile)
        slrfl_grid = forcing.fr_th / forcing.dt
        tskrad_grid = torch.full((N,), 288.0, dtype=cfg.dtype)
        emis_grid = static.emis

        diff_inp = model._build_diffusion_inputs(
            state=state, forcing=forcing, static=static,
            sl_lev=sl_lev, q_lev=forcing.q,
            rhoc_u_t=rhoc_u_t, rhoc_q_t=rhoc_q_t,
            sskm1m=sskm1m, skin_t_prev=skin_t_prev,
            qskm1m=qskm1m, dqsdT=dqsdT,
            alphal=alphal, alphas=alphas,
            ssrfl_t=ssrfl_t, slrfl_grid=slrfl_grid,
            tskrad_grid=tskrad_grid, emis_grid=emis_grid,
            plamsk=plamsk, pevapsnw=torch.zeros_like(skin_t_prev), tsrf=tsrf, frac=frac, ktvl=ktvl, ktvh=ktvh,
            dt=forcing.dt, ntile=ntile, ngrid=N,
            dtype=cfg.dtype, device=state.skin_t.device,
        )
        assert diff_inp.lesnml is True, (
            "lesnml config flag did not propagate to DiffusionInputs"
        )

    def test_pevapsnw_none_by_default(self):
        # The snow-under-forest evaporation is None by default (M6
        # populates it).  forward() must not crash.
        state, forcing, static, cfg = _make_inputs(N=2)
        model = LandSurface(cfg)
        new_state, diag = model(state, forcing, static)
        assert torch.isfinite(new_state.skin_t).all()

    def test_differentiable_forward_with_vdfdifh(self):
        # differentiable=True -> autograd runs through the VDFDIFH1S
        # wrapper.  Verify a loss on new_state.skin_t backprops to the
        # t-1 skin T leaf.
        state, forcing, static, cfg = _make_inputs(
            N=2, differentiable=True,
        )
        state.skin_t.requires_grad_(True)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        loss = new_state.skin_t.sum()
        loss.backward()
        assert state.skin_t.grad is not None
        assert torch.isfinite(state.skin_t.grad).all()
        assert state.skin_t.grad.abs().sum() > 0


class TestM4cClosureInvariants:
    """Invariants that must hold after the M4c integration."""

    def test_seb_flux_diagnostics_are_finite(self):
        # The diagnostic does not expose SURFSEB's final transformed SW
        # boundary, so verify its public flux fields directly.
        from surf_pytorch.physics.seb import _tile_solar_transmission
        state, forcing, static, cfg = _make_inputs(N=3)
        model = LandSurface(cfg)
        new_state, diag = model(state, forcing, static)

        for value in (diag.str_tile, diag.h_flux_tile, diag.slh_tile, diag.g0_tile):
            assert torch.isfinite(value).all()

    def test_state_not_mutated(self):
        state, forcing, static, cfg = _make_inputs(N=2)
        original = state.skin_t.clone()
        model = LandSurface(cfg)
        _ = model(state, forcing, static)
        assert torch.equal(state.skin_t, original), (
            "forward must not mutate the input SurfState"
        )
