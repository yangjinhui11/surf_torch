"""Dedicated tests for the M5d ``LandSurface.forward`` integration.

These pin the contract that ``forward`` advances both soil prognostics
by one coupling interval via the HTESSEL solves:

* M5d-soilt (SRFRCGS + SRFTS + SRFWDIFS) advances soil temperature;
* M5d-soilw (SRFWEXC_VG + SRFWDIF + SRFWINC + SRFWNG) advances soil
  moisture.

Invariants pinned:

* ``new_state.soil_t`` / ``new_state.soil_w_liq`` differ from the
  t-1 state on land columns;
* the soil-T result bit-matches a direct call to
  :func:`surf_pytorch.physics.soil.compute_soil_temperature` fed the
  same fluxes VDFDIFH1S produced (no hidden arithmetic in the
  forward-path glue);
* the soil-W result bit-matches a direct call to
  :func:`surf_pytorch.physics.soil.compute_soil_moisture` fed the same
  fluxes and precipitation throughfall;
* the water budget closes on land columns:
  ``precip_in - evap_out = column_delta + (surface + sub-surface runoff)``
  with a tight tolerance;
* sea columns (``land_frac == 0``) keep their t-1 soil state;
* the full forward path is autograd-able (differentiable=True).
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from surf_pytorch import SurfConfig, LandSurface
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.physics.soil import (
    SoilTempInputs, SoilMoistInputs,
    compute_soil_temperature as port_soilt,
    compute_soil_moisture as port_soilw,
)
from surf_pytorch.utils import expand_as_tiles


# --------------------------------------------------------------------------- #
# Input builder (mirrors test_forward_m4c._make_inputs, but with full-land
# tile fractions so the soil solve actually fires).
# --------------------------------------------------------------------------- #
def _make_inputs(
    *,
    N: int = 4,
    dtype: torch.dtype = torch.float64,
    differentiable: bool = False,
    land_frac: float = 1.0,
    tsoil: float = 285.0,
    wsoil: float = 0.30,
    rain_kg_m2: float = 0.0,
    tair: float = 290.0,
) -> tuple[SurfState, Forcing, Static, SurfConfig]:
    """Build a minimal self-consistent (state, forcing, static, config)."""
    cfg = SurfConfig(dtype=dtype, differentiable=differentiable)
    ntile, nsoil = cfg.n_tile, cfg.n_soil

    state = SurfState(
        skin_t=torch.full((N, ntile), 288.0, dtype=dtype),
        soil_t=torch.full((N, nsoil), tsoil, dtype=dtype),
        soil_w_liq=torch.full((N, nsoil), wsoil, dtype=dtype),
        snow_w=torch.zeros((N,), dtype=dtype),
        snow_frac=torch.zeros((N,), dtype=dtype),
        skin_water=torch.zeros((N,), dtype=dtype),
        sst=torch.full((N,), 290.0, dtype=dtype),
    )
    forcing = Forcing(
        u=torch.full((N,), 5.0, dtype=dtype),
        v=torch.zeros(N, dtype=dtype),
        t=torch.full((N,), tair, dtype=dtype),
        q=torch.full((N,), 0.008, dtype=dtype),
        co2=torch.full((N,), 6e-4, dtype=dtype),
        p_surf=torch.full((N,), 1.0e5, dtype=dtype),
        geom1=torch.full((N,), 50.0, dtype=dtype),
        fr_so=torch.full((N,), 200.0 * 3600.0, dtype=dtype),
        fr_th=torch.full((N,), -300.0 * 3600.0, dtype=dtype),
        mu0=torch.full((N,), 0.5, dtype=dtype),
        rain=torch.full((N,), rain_kg_m2, dtype=dtype),
        snow=torch.zeros((N,), dtype=dtype),
        dt=torch.tensor(3600.0, dtype=dtype),
    )
    # Pure-land tile fractions: wet-skin / low-veg dry / high-veg dry /
    # bare soil (no water / ice / snow tiles).
    frac = torch.zeros((N, ntile), dtype=dtype)
    frac[:, 2] = 0.05
    frac[:, 3] = 0.30
    frac[:, 5] = 0.30
    frac[:, 7] = 0.35
    static = Static(
        tile_frac_base=frac,
        land_frac=torch.full((N,), land_frac, dtype=dtype),
        orography=torch.full((N,), 50.0, dtype=dtype),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=torch.full((N,), 3, dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.full((N,), 0.99, dtype=dtype),
    )
    return state, forcing, static, cfg


# --------------------------------------------------------------------------- #
# Helper: reconstruct the SoilTempInputs the forward() path builds, so we
# can call compute_soil_temperature directly and compare.
# --------------------------------------------------------------------------- #
def _rebuild_soiltemp_inputs(
    model: LandSurface,
    state: SurfState,
    forcing: Forcing,
    static: Static,
    cfg: SurfConfig,
) -> tuple[SoilTempInputs, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run forward()'s pre-soil glue and return the (inp, ssrfl_t, slrfl,
    pfrti) tuple needed to call compute_soil_temperature directly.

    Mirrors model.forward sections 1-6b.  Returns the SoilTempInputs the
    forward path feeds to compute_soil_temperature, plus the auxiliary
    tensors so a test can re-pack it for a direct comparison.
    """
    ngrid = state.skin_t.shape[0]
    ntile = cfg.n_tile
    dtype = cfg.dtype
    device = state.skin_t.device

    model._surf_step_count = state.vupdz0_step
    model._last_pustrti = state.vupdz0_pustrti
    model._last_pvstrti = state.vupdz0_pvstrti
    model._last_pahfsti = state.vupdz0_pahfsti
    model._last_pevapti = state.vupdz0_pevapti
    model._last_pzdl = state.vupdz0_pzdl

    cp = model.consts.cpd
    sl_lev = cp * forcing.t + forcing.geom1
    q_lev = forcing.q
    skin_t_prev = state.skin_t
    rhoc_u, rhoc_q = model._exchange_coeffs(forcing, static, state)
    # Mirror forward()'s per-tile handling: in per_tile_exchange mode the
    # helper already returns (Ngrid, ntile); otherwise broadcast the grid-
    # mean (Ngrid,) tensors across tiles.
    if cfg.per_tile_exchange:
        rhoc_u_t, rhoc_q_t = rhoc_u, rhoc_q
    else:
        rhoc_u_t = expand_as_tiles(rhoc_u, ntile)
        rhoc_q_t = expand_as_tiles(rhoc_q, ntile)
    sskm1m = cp * skin_t_prev
    # Saturation fields -- mirror forward() but don't need to be exact
    # for this comparison (they only affect pssh / pjq, which are taken
    # from the live diff_out below).
    from surf_pytorch.utils import qsat, dqsat_dt_ifs
    p_surf_t = expand_as_tiles(forcing.p_surf, ntile)
    qskm1m = qsat(skin_t_prev, p_surf_t, model.consts)
    dqsdT = dqsat_dt_ifs(skin_t_prev, p_surf_t, model.consts)
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
    frac = model._tile_frac(static, ntile, state=state)
    ssrfl_t = expand_as_tiles(forcing.fr_so / forcing.dt, ntile)
    slrfl_grid = forcing.fr_th / forcing.dt
    tskrad_grid = torch.full((ngrid,), 288.0, dtype=dtype)
    emis_grid = static.emis

    # Use the same alpha computation forward() uses.  For the bit-match
    # test below it is essential that we run the *actual* forward path and
    # read diff_out back; alphas only enter through diff_out.pssh / pjq.
    # Both closures (Jarvis and Farquhar) share the downstream VEVAP
    # chain; only the canopy-resistance source for tiles 4/6/7 differs
    # (see LandSurface.forward §6).
    from surf_pytorch.physics.stomata import (
        jarvis_resistance as compute_jarvis_resistance,
        build_tile_vevap as compute_tile_vevap,
        farquhar_resistance as compute_farquhar_resistance,
    )
    jarvis = compute_jarvis_resistance(
        soil_t=state.soil_t, soil_w=state.soil_w_liq,
        soil_type=static.soil_type, ktvl=ktvl, ktvh=ktvh,
        t_air=forcing.t, q_air=forcing.q, p_surf=forcing.p_surf,
        sw_down=forcing.fr_so / forcing.dt,
        veg_table=model.veg_table, soil_table=model.soil_table,
        root_table=model.root_table, consts=model.consts,
        tile_frac=frac, n_tile=ntile,
    )
    if cfg.use_farquhar:
        resist = compute_farquhar_resistance(
            forcing=forcing, skin_t_prev=skin_t_prev, qskm1m=qskm1m,
            ssrfl_grid=forcing.fr_so / forcing.dt, ktvl=ktvl, ktvh=ktvh,
            jarvis=jarvis, veg_table=model.veg_table, consts=model.consts,
            land_mask=static.land_frac > 0,
        )
    else:
        resist = jarvis
    wlmx_grid = model._interception_capacity(static, frac, ktvl, ktvh)
    t_sub_grid = tsrf[:, 0]
    vegc = model.veg_table.consts
    vevap_out = compute_tile_vevap(
        pwetl=resist["pwetl"], pweth=resist["pweth"],
        pwetb=resist["pwetb"],
        rhocqu_grid=rhoc_q, p_surf=forcing.p_surf,
        t_air=forcing.t, q_air=forcing.q,
        skin_t=skin_t_prev, t_sub_grid=t_sub_grid,
        qs_skin=qskm1m, wlmx_grid=wlmx_grid,
        consts=model.consts, config=cfg,
        pwetlu=resist["pwetl"], pweths=resist["pweths"],
        ptmst=forcing.dt,
        rlhaero=vegc.rlhaero, rlhaeros=vegc.rlhaeros,
    )
    alphas = vevap_out["pcsat"]
    alphal = vevap_out["pcair"]

    diff_inp = model._build_diffusion_inputs(
        state=state, forcing=forcing, static=static,
        sl_lev=sl_lev, q_lev=q_lev,
        rhoc_u_t=rhoc_u_t, rhoc_q_t=rhoc_q_t,
        sskm1m=sskm1m, skin_t_prev=skin_t_prev,
        qskm1m=qskm1m, dqsdT=dqsdT,
        alphal=alphal, alphas=alphas,
        ssrfl_t=ssrfl_t, slrfl_grid=slrfl_grid,
        tskrad_grid=tskrad_grid, emis_grid=emis_grid,
        plamsk=plamsk, tsrf=tsrf, frac=frac, ktvl=ktvl, ktvh=ktvh,
        pevapsnw=torch.zeros((ngrid,), dtype=dtype, device=device),
        dt=forcing.dt, ntile=ntile, ngrid=ngrid,
        dtype=dtype, device=device,
    )
    from surf_pytorch.physics.diffusion import vdfdifh1s as compute_vdfdifh1s
    diff_out = compute_vdfdifh1s(diff_inp, model.consts, cfg)
    surfpp_pssh, surfpp_pjq = model._surfpp_zero_tile_fluxes(diff_out.pssh, diff_out.pjq, frac)
    diff_out = replace(diff_out, pssh=surfpp_pssh, pjq=surfpp_pjq)

    # _make_inputs sets snow_w=0 and no snow tiles, so the SRFSN_LWIMPS
    # no-snow branch fires in forward() and yields PGSN exactly zero;
    # pass the same zero here to keep the bit-match comparison valid.
    pgsn_zero = torch.zeros((ngrid,), dtype=dtype, device=device)
    inp = model._update_soil_temperature(
        state=state, static=static, diff_out=diff_out,
        pssrflti=ssrfl_t, pslrfl=slrfl_grid, pfrti=frac,
        pgsn=pgsn_zero,
        dt=forcing.dt, ngrid=ngrid, dtype=dtype, device=device,
    )
    # _update_soil_temperature returns the new soil T tensor; for the
    # bit-match test we need the SoilTempInputs that produced it, so
    # rebuild it here from the same pieces.
    from surf_pytorch.config import (
        TILE_LOW_VEG_DRY, TILE_LOW_VEG_SNOW,
        TILE_HIGH_VEG_DRY, TILE_HIGH_VEG_SNOW,
    )
    pcvl = frac.new_zeros(frac.shape[:-1])
    pcvh = frac.new_zeros(frac.shape[:-1])
    if ntile > TILE_LOW_VEG_DRY:
        pcvl = pcvl + frac[..., TILE_LOW_VEG_DRY]
    if ntile > TILE_LOW_VEG_SNOW:
        pcvl = pcvl + frac[..., TILE_LOW_VEG_SNOW]
    if ntile > TILE_HIGH_VEG_DRY:
        pcvh = pcvh + frac[..., TILE_HIGH_VEG_DRY]
    if ntile > TILE_HIGH_VEG_SNOW:
        pcvh = pcvh + frac[..., TILE_HIGH_VEG_SNOW]
    pgsn = torch.zeros((ngrid,), dtype=dtype, device=device)
    ldland = static.land_frac > 0.5
    soil_inp = SoilTempInputs(
        ptsam1m=state.soil_t, pwsam1m=state.soil_w_liq,
        pfrti=frac, pahfsti=diff_out.pssh, pevapti=diff_out.pjq,
        pssrflti=ssrfl_t, pslrfl=slrfl_grid, pslrflti=diff_out.pstr,
        pgsn=pgsn,
        ksoty=static.soil_type, pcvl=pcvl, pcvh=pcvh,
        ldland=ldland, ptmst=forcing.dt,
    )
    return soil_inp, ssrfl_t, slrfl_grid, frac


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestM5dSoiltIntegration:
    """Pin the M5d-soilt forward() integration invariants."""

    def test_soil_t_advances_on_land(self):
        # On a full-land column with a net-upward energy flux the soil
        # temperature must drop after one coupling interval.  The flux is
        # net-upward only when *both* radiation and the sensible heat flux
        # point out of the ground: with the per-tile exchange (the offline
        # driver's physics) the vegetation tiles carry a large Ch, so a
        # stable column (T_air > T_skin) drives a strong *downward*
        # sensible flux that can outweigh the radiative cooling.  We
        # therefore force an *unstable* column (T_air=280 < T_soil=285)
        # so the sensible flux is unambiguously upward and the soil must
        # cool -- the test then pins the cooling sign robustly.
        state, forcing, static, cfg = _make_inputs(
            N=2, land_frac=1.0, tsoil=285.0, tair=280.0,
        )
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        # Top layer must cool measurably (>= 0.1 K).
        d_top = (new_state.soil_t[:, 0] - state.soil_t[:, 0]).abs()
        assert d_top.min().item() > 0.1, (
            f"top-layer soil T did not advance: dT = {d_top.tolist()}"
        )
        # And it must cool (negative dT) since net flux is upward.
        dT = new_state.soil_t[:, 0] - state.soil_t[:, 0]
        assert (dT < 0).all(), (
            f"top-layer soil T warmed under net-upward flux: dT = {dT.tolist()}"
        )

    def test_soil_t_unchanged_on_sea(self):
        # Sea columns (land_frac = 0) keep their t-1 soil T.
        state, forcing, static, cfg = _make_inputs(N=2, land_frac=0.0)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        assert torch.equal(new_state.soil_t, state.soil_t), (
            "sea-column soil T mutated by the soil-temperature update"
        )

    def test_soil_t_advances_per_column(self):
        # Mixed land/sea batch: land columns advance, sea column does not.
        state, forcing, static, cfg = _make_inputs(N=2, land_frac=1.0)
        # Flip the second column to sea.
        static.land_frac = torch.tensor([1.0, 0.0], dtype=cfg.dtype)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        d0 = (new_state.soil_t[0] - state.soil_t[0]).abs().max().item()
        d1 = (new_state.soil_t[1] - state.soil_t[1]).abs().max().item()
        assert d0 > 0.1, f"land column did not advance: dT_max = {d0}"
        assert d1 == 0.0, f"sea column mutated: dT_max = {d1}"

    def test_forward_matches_direct_soil_temperature_call(self):
        # The soil T produced by forward() must bit-match the result of
        # calling compute_soil_temperature directly with the same fluxes.
        # This catches any hidden arithmetic in the forward-path glue
        # (rescaling, transposing, dtype-casting).
        state, forcing, static, cfg = _make_inputs(N=3, land_frac=1.0)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)

        soil_inp, _, _, _ = _rebuild_soiltemp_inputs(
            model, state, forcing, static, cfg,
        )
        direct_out = port_soilt(
            soil_inp, model.soil_table, model.consts, cfg,
        )
        torch.testing.assert_close(
            new_state.soil_t, direct_out.ptsa, rtol=0, atol=0,
        ), (
            "forward() soil T != direct compute_soil_temperature() output"
        )

    def test_soil_t_shape_preserved(self):
        state, forcing, static, cfg = _make_inputs(N=5)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        assert new_state.soil_t.shape == state.soil_t.shape
        assert new_state.soil_t.dtype == state.soil_t.dtype

    def test_soil_t_finite(self):
        state, forcing, static, cfg = _make_inputs(N=3, tsoil=272.5)
        # 272.5 K exercises the freeze-window branch of SRFRCGS; assert
        # the result is finite (catches NaN/inf from a bad port).
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        assert torch.isfinite(new_state.soil_t).all(), (
            "non-finite soil temperature after surface step"
        )

    def test_state_not_mutated(self):
        state, forcing, static, cfg = _make_inputs(N=2)
        original = state.soil_t.clone()
        model = LandSurface(cfg)
        _ = model(state, forcing, static)
        assert torch.equal(state.soil_t, original), (
            "forward must not mutate the input SurfState.soil_t"
        )

    def test_differentiable_forward_backprops_to_soil_t(self):
        # differentiable=True -> autograd runs through the soil-T solve.
        # A loss on new_state.soil_t must backprop to the t-1 soil T leaf.
        state, forcing, static, cfg = _make_inputs(
            N=2, differentiable=True,
        )
        state.soil_t.requires_grad_(True)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        loss = new_state.soil_t.sum()
        loss.backward()
        assert state.soil_t.grad is not None
        assert torch.isfinite(state.soil_t.grad).all()
        # The Thomas solve couples layers, but the Jacobian is dense, so
        # every layer's T at t-1 should receive non-zero gradient.
        assert state.soil_t.grad.abs().sum() > 0


class TestM5dSoiltClosure:
    """Invariants that must hold after the M5d-soilt integration."""

    def test_soil_t_cools_under_outgoing_radiation(self):
        # Sanity-check the sign of the soil-T tendency against the net
        # upward energy flux.  The net flux is upward only when radiation
        # (SW=200, LW=-300 -> net -100 W/m^2) AND the sensible flux both
        # point out of the ground.  With the per-tile exchange the
        # vegetation tiles carry a large Ch, so a stable column
        # (T_air > T_skin) drives a downward sensible flux that can
        # outweigh the radiative cooling; we therefore force an unstable
        # column (T_air=280 < T_soil=285) so the soil must cool.
        state, forcing, static, cfg = _make_inputs(
            N=2, land_frac=1.0, tsoil=285.0, tair=280.0,
        )
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        # All layers cool (or stay put for the deepest layer that
        # barely couples on a 1-hour step).
        dT = new_state.soil_t - state.soil_t
        # Top layer strictly cools.
        assert (dT[:, 0] < 0).all(), (
            f"top layer did not cool: dT = {dT[:, 0].tolist()}"
        )

    def test_deep_layer_smaller_response(self):
        # The implicit Thomas solve smooths the surface flux over the
        # thermal diffusion timescale, so deeper layers must move less
        # than the top layer.
        state, forcing, static, cfg = _make_inputs(N=2, land_frac=1.0)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        dT = (new_state.soil_t - state.soil_t).abs()
        # |dT| must be monotonically non-increasing with depth.
        for j in range(dT.shape[-1] - 1):
            assert (dT[:, j] >= dT[:, j + 1] - 1e-12).all(), (
                f"|dT| not monotonic with depth at layer {j}: "
                f"{dT.tolist()}"
            )


# --------------------------------------------------------------------------- #
# Helper: reconstruct the SoilMoistInputs the forward() path builds, so we
# can call compute_soil_moisture directly and compare.
# --------------------------------------------------------------------------- #
def _rebuild_soilmoist_inputs(
    model: LandSurface,
    state: SurfState,
    forcing: Forcing,
    static: Static,
    cfg: SurfConfig,
) -> SoilMoistInputs:
    """Run forward()'s pre-soilw glue and return the SoilMoistInputs
    that forward() feeds to compute_soil_moisture.

    Mirrors model.forward sections 1-6c.  Returns the SoilMoistInputs
    rebuilt from the same pieces the forward path uses, so a test can
    call compute_soil_moisture directly and verify bit-exact agreement.
    """
    ngrid = state.skin_t.shape[0]
    ntile = cfg.n_tile
    dtype = cfg.dtype
    device = state.skin_t.device

    model._surf_step_count = state.vupdz0_step
    model._last_pustrti = state.vupdz0_pustrti
    model._last_pvstrti = state.vupdz0_pvstrti
    model._last_pahfsti = state.vupdz0_pahfsti
    model._last_pevapti = state.vupdz0_pevapti
    model._last_pzdl = state.vupdz0_pzdl

    # Re-run the SEB to get diff_out.pjq.
    cp = model.consts.cpd
    sl_lev = cp * forcing.t + forcing.geom1
    skin_t_prev = state.skin_t
    rhoc_u, rhoc_q = model._exchange_coeffs(forcing, static, state)
    # Mirror forward()'s per-tile handling (see _rebuild_soiltemp_inputs).
    if cfg.per_tile_exchange:
        rhoc_u_t, rhoc_q_t = rhoc_u, rhoc_q
    else:
        rhoc_u_t = expand_as_tiles(rhoc_u, ntile)
        rhoc_q_t = expand_as_tiles(rhoc_q, ntile)
    sskm1m = cp * skin_t_prev
    from surf_pytorch.utils import qsat, dqsat_dt_ifs
    p_surf_t = expand_as_tiles(forcing.p_surf, ntile)
    qskm1m = qsat(skin_t_prev, p_surf_t, model.consts)
    dqsdT = dqsat_dt_ifs(skin_t_prev, p_surf_t, model.consts)
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
    frac = model._tile_frac(static, ntile, state=state)
    ssrfl_t = expand_as_tiles(forcing.fr_so / forcing.dt, ntile)
    slrfl_grid = forcing.fr_th / forcing.dt
    tskrad_grid = torch.full((ngrid,), 288.0, dtype=dtype)
    emis_grid = static.emis

    # Same alpha closure as LandSurface.forward §6 (Jarvis or Farquhar,
    # both feeding the downstream VEVAP chain).  See _rebuild_soiltemp_inputs.
    from surf_pytorch.physics.stomata import (
        jarvis_resistance as compute_jarvis_resistance,
        build_tile_vevap as compute_tile_vevap,
        farquhar_resistance as compute_farquhar_resistance,
    )
    jarvis = compute_jarvis_resistance(
        soil_t=state.soil_t, soil_w=state.soil_w_liq,
        soil_type=static.soil_type, ktvl=ktvl, ktvh=ktvh,
        t_air=forcing.t, q_air=forcing.q, p_surf=forcing.p_surf,
        sw_down=forcing.fr_so / forcing.dt,
        veg_table=model.veg_table, soil_table=model.soil_table,
        root_table=model.root_table, consts=model.consts,
        tile_frac=frac, n_tile=ntile,
    )
    if cfg.use_farquhar:
        resist = compute_farquhar_resistance(
            forcing=forcing, skin_t_prev=skin_t_prev, qskm1m=qskm1m,
            ssrfl_grid=forcing.fr_so / forcing.dt, ktvl=ktvl, ktvh=ktvh,
            jarvis=jarvis, veg_table=model.veg_table, consts=model.consts,
            land_mask=static.land_frac > 0,
        )
    else:
        resist = jarvis
    wlmx_grid = model._interception_capacity(static, frac, ktvl, ktvh)
    t_sub_grid = tsrf[:, 0]
    vegc = model.veg_table.consts
    vevap_out = compute_tile_vevap(
        pwetl=resist["pwetl"], pweth=resist["pweth"],
        pwetb=resist["pwetb"],
        rhocqu_grid=rhoc_q, p_surf=forcing.p_surf,
        t_air=forcing.t, q_air=forcing.q,
        skin_t=skin_t_prev, t_sub_grid=t_sub_grid,
        qs_skin=qskm1m, wlmx_grid=wlmx_grid,
        consts=model.consts, config=cfg,
        pwetlu=resist["pwetl"], pweths=resist["pweths"],
        ptmst=forcing.dt,
        rlhaero=vegc.rlhaero, rlhaeros=vegc.rlhaeros,
    )
    alphas = vevap_out["pcsat"]
    alphal = vevap_out["pcair"]

    diff_inp = model._build_diffusion_inputs(
        state=state, forcing=forcing, static=static,
        sl_lev=sl_lev, q_lev=forcing.q,
        rhoc_u_t=rhoc_u_t, rhoc_q_t=rhoc_q_t,
        sskm1m=sskm1m, skin_t_prev=skin_t_prev,
        qskm1m=qskm1m, dqsdT=dqsdT,
        alphal=alphal, alphas=alphas,
        ssrfl_t=ssrfl_t, slrfl_grid=slrfl_grid,
        tskrad_grid=tskrad_grid, emis_grid=emis_grid,
        plamsk=plamsk, tsrf=tsrf, frac=frac, ktvl=ktvl, ktvh=ktvh,
        pevapsnw=torch.zeros((ngrid,), dtype=dtype, device=device),
        dt=forcing.dt, ntile=ntile, ngrid=ngrid,
        dtype=dtype, device=device,
    )
    from surf_pytorch.physics.diffusion import vdfdifh1s as compute_vdfdifh1s
    diff_out = compute_vdfdifh1s(diff_inp, model.consts, cfg)
    surfpp_pssh, surfpp_pjq = model._surfpp_zero_tile_fluxes(diff_out.pssh, diff_out.pjq, frac)
    diff_out = replace(diff_out, pssh=surfpp_pssh, pjq=surfpp_pjq)

    # Skin-reservoir / canopy interception update (M6, SRFWL) -- mirrors
    # LandSurface.forward section 6.  Produces the rainfall throughfall
    # PTSFL and the per-tile interception-layer evaporation mismatch
    # PEINTTI that the soil-moisture solve consumes.
    canopy_out = model._update_canopy(
        state=state, static=static, diff_out=diff_out, frac=frac,
        wlmx_grid=wlmx_grid, ktvl=ktvl, ktvh=ktvh,
        forcing=forcing, dt=forcing.dt,
        pevapsnw=torch.zeros((ngrid,), dtype=dtype, device=device),
        ngrid=ngrid, dtype=dtype, device=device,
    )

    ptsfl = canopy_out.ptsfl
    ptsfc = torch.zeros((ngrid,), dtype=dtype, device=device)
    pmsn = torch.zeros((ngrid,), dtype=dtype, device=device)
    pemssn = torch.zeros((ngrid,), dtype=dtype, device=device)
    pevapsnw = torch.zeros((ngrid,), dtype=dtype, device=device)
    peintti = canopy_out.peintti
    psdor = torch.zeros((ngrid,), dtype=dtype, device=device)
    ldland = static.land_frac > 0.5

    return SoilMoistInputs(
        pwsam1m=state.soil_w_liq, ptsam1m=state.soil_t,
        pfrti=frac, pevapti=diff_out.pjq,
        ptsfc=ptsfc, ptsfl=ptsfl,
        pmsn=pmsn, pemssn=pemssn, pevapsnw=pevapsnw, peintti=peintti,
        ksoty=static.soil_type, ktvl=static.veg_type, ktvh=static.veg_type,
        psdor=psdor, ldland=ldland, ptmst=forcing.dt,
    )


class TestM5dSoilwIntegration:
    """Pin the M5d-soilw forward() integration invariants."""

    def test_soil_w_advances_with_precip(self):
        # With 1 mm/h rain (1 kg/m^2 over the 1-hour step), the top
        # soil layer must wet up measurably.
        state, forcing, static, cfg = _make_inputs(
            N=2, land_frac=1.0, rain_kg_m2=1.0,
        )
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        d_top = (new_state.soil_w_liq[:, 0] - state.soil_w_liq[:, 0])
        assert (d_top > 0).all(), (
            f"top-layer soil moisture did not wet up under precip: "
            f"dW = {d_top.tolist()}"
        )

    def test_soil_w_unchanged_on_sea(self):
        # Sea columns keep their t-1 soil moisture (LDLAND=.FALSE. branch).
        state, forcing, static, cfg = _make_inputs(
            N=2, land_frac=0.0, rain_kg_m2=1.0,
        )
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        assert torch.equal(new_state.soil_w_liq, state.soil_w_liq), (
            "sea-column soil moisture mutated by the soil-moisture update"
        )

    def test_soil_w_advances_per_column(self):
        # Mixed land/sea batch: land column advances, sea column does not.
        state, forcing, static, cfg = _make_inputs(
            N=2, land_frac=1.0, rain_kg_m2=1.0,
        )
        # Flip the second column to sea.
        static.land_frac = torch.tensor([1.0, 0.0], dtype=cfg.dtype)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        d0 = (new_state.soil_w_liq[0] - state.soil_w_liq[0]).abs().max().item()
        d1 = (new_state.soil_w_liq[1] - state.soil_w_liq[1]).abs().max().item()
        assert d0 > 1e-4, f"land column did not advance: dW_max = {d0}"
        assert d1 == 0.0, f"sea column mutated: dW_max = {d1}"

    def test_soil_w_clamped_to_physical_range(self):
        # Heavy precipitation must not push soil moisture beyond WSAT,
        # and drought must not drive it below WRES.  Run with saturated
        # initial moisture + heavy rain; result must stay <= WSAT.
        from surf_pytorch.tables import SoilTable
        st = SoilTable(dtype=torch.float64)
        ksoty = 3   # matches _make_inputs default
        wsat = st.gather("wsat", torch.tensor([ksoty], dtype=torch.long))

        # Start saturated, add large precip.
        state, forcing, static, cfg = _make_inputs(
            N=2, land_frac=1.0,
            wsoil=float(wsat.item()), rain_kg_m2=50.0,
        )
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        assert (new_state.soil_w_liq <= wsat.item() + 1e-12).all(), (
            f"soil moisture exceeded WSAT={wsat.item():.4f}: "
            f"max = {new_state.soil_w_liq.max().item():.4f}"
        )
        # And start very dry, force evaporation -- but we cannot easily
        # set negative evap via _make_inputs (evap comes from the SEB
        # closure).  At least check the floor is respected on the wet
        # case for now; the M5b unit tests pin the dry floor.

    def test_forward_matches_direct_soil_moisture_call(self):
        # The soil W produced by forward() must bit-match the result of
        # calling compute_soil_moisture directly with the same fluxes
        # and precipitation throughfall.  This catches any hidden
        # arithmetic in the forward-path glue.
        state, forcing, static, cfg = _make_inputs(
            N=3, land_frac=1.0, rain_kg_m2=1.0,
        )
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)

        moist_inp = _rebuild_soilmoist_inputs(model, state, forcing, static, cfg)
        direct_out = port_soilw(
            moist_inp, model.soil_table, model.veg_table,
            model.consts, cfg,
        )
        torch.testing.assert_close(
            new_state.soil_w_liq, direct_out.pwsa, rtol=0, atol=0,
        ), (
            "forward() soil W != direct compute_soil_moisture() output"
        )

    def test_soil_w_shape_preserved(self):
        state, forcing, static, cfg = _make_inputs(N=5, rain_kg_m2=1.0)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        assert new_state.soil_w_liq.shape == state.soil_w_liq.shape
        assert new_state.soil_w_liq.dtype == state.soil_w_liq.dtype

    def test_soil_w_finite(self):
        # Exercise the freeze window + precip simultaneously.
        state, forcing, static, cfg = _make_inputs(
            N=3, tsoil=272.5, rain_kg_m2=1.0,
        )
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        assert torch.isfinite(new_state.soil_w_liq).all(), (
            "non-finite soil moisture after surface step"
        )

    def test_state_not_mutated_soil_w(self):
        state, forcing, static, cfg = _make_inputs(N=2, rain_kg_m2=1.0)
        original = state.soil_w_liq.clone()
        model = LandSurface(cfg)
        _ = model(state, forcing, static)
        assert torch.equal(state.soil_w_liq, original), (
            "forward must not mutate the input SurfState.soil_w_liq"
        )

    def test_differentiable_forward_backprops_to_soil_w(self):
        # differentiable=True -> autograd runs through the soil-W solve.
        state, forcing, static, cfg = _make_inputs(
            N=2, differentiable=True, rain_kg_m2=1.0,
        )
        state.soil_w_liq.requires_grad_(True)
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)
        loss = new_state.soil_w_liq.sum()
        loss.backward()
        assert state.soil_w_liq.grad is not None
        assert torch.isfinite(state.soil_w_liq.grad).all()
        assert state.soil_w_liq.grad.abs().sum() > 0


class TestM5dSoilwClosure:
    """Water-budget closure invariants for the M5d-soilw integration."""

    def test_water_budget_closes_with_precip(self):
        # Water budget on land columns:
        #   throughfall_in - evap_out = column_delta + (PROS + PROD)
        # where throughfall_in = rainfall reaching the soil after the
        # canopy interception (_update_canopy) has skimmed its share into
        # the skin reservoir PWL, and column_delta = sum_j (W_new - W_old)
        # * RDAW_j * rho_h2o.  The intercepted water stays in PWL
        # (new_state.skin_water), which is a separate reservoir and is not
        # part of the soil-column budget.
        # Use a 4-layer default config.
        state, forcing, static, cfg = _make_inputs(
            N=2, land_frac=1.0, rain_kg_m2=2.0,
        )
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)

        moist_inp = _rebuild_soilmoist_inputs(model, state, forcing, static, cfg)
        direct_out = port_soilw(
            moist_inp, model.soil_table, model.veg_table,
            model.consts, cfg,
        )

        rdat = torch.tensor(cfg.soil_thickness[:cfg.n_soil], dtype=cfg.dtype)
        rho_h2o = float(model.soil_table.consts.rho_h2o)

        # Column water delta (kg/m^2): positive = water gained.
        col_delta = (
            (new_state.soil_w_liq - state.soil_w_liq) * rdat * rho_h2o
        ).sum(dim=-1)                                            # (Ngrid,)

        # Throughfall into the soil (kg/m^2): the large-scale throughfall
        # PTSFL (rainfall after interception) produced by _update_canopy.
        # moist_inp.ptsfl is the rate (kg/m^2/s); multiply by dt.
        ptmst = float(forcing.dt.item())
        throughfall_in = moist_inp.ptsfl * ptmst                  # kg/m^2

        # Evap out (kg/m^2): from the per-tile moisture flux, only the
        # soil-viewing tiles (wet-skin 3, low-veg dry 4, high-veg dry 6,
        # bare-soil 8 -- 0-indexed 2, 3, 5, 7).  PEVAPTI > 0 = dew (in);
        # PEVAPTI < 0 = evap (out).  For the budget "out" is positive,
        # so evap_out = -PEVAPTI * dt * sum(soil_tile_frac).
        from surf_pytorch.config import (
            TILE_WET_SKIN, TILE_LOW_VEG_DRY,
            TILE_HIGH_VEG_DRY, TILE_BARE_SOIL,
        )
        soil_tiles = (TILE_WET_SKIN, TILE_LOW_VEG_DRY,
                      TILE_HIGH_VEG_DRY, TILE_BARE_SOIL)
        # diff_out.pjq is the per-tile moisture flux (kg/m^2/s).  We
        # reconstruct it from moist_inp.pevapti.
        pevapti = moist_inp.pevapti                              # (Ngrid, Ntile)
        pfrti = moist_inp.pfrti
        soil_frac_sum = (
            pfrti[:, TILE_WET_SKIN] + pfrti[:, TILE_LOW_VEG_DRY]
            + pfrti[:, TILE_HIGH_VEG_DRY] + pfrti[:, TILE_BARE_SOIL]
        )
        evap_tile_mean = sum(pfrti[:, t] * pevapti[:, t] for t in soil_tiles)
        # If any tile fraction is off (soil_frac_sum < 1), we still need
        # to count the per-tile flux that actually leaves the column.
        # In our test slab soil_frac_sum == 1 (no water / ice tiles).
        evap_out = -evap_tile_mean * ptmst                       # kg/m^2

        # Runoff out (kg/m^2): PROS (surface) + PROD (sub-surface).
        runoff_out = direct_out.pros + direct_out.prod           # (Ngrid,)

        # Budget: throughfall_in - evap_out = col_delta + runoff_out
        lhs = throughfall_in - evap_out
        rhs = col_delta + runoff_out
        resid = (lhs - rhs).abs()
        # Tolerance: budget must close to ~1e-9 kg/m^2 (the only source
        # of mismatch is FP rounding in the Thomas solve and the
        # clip/redistribute step).  Use 1e-6 as a generous guard.
        assert resid.max().item() < 1e-6, (
            f"water budget residual too large: max|lhs - rhs| = "
            f"{resid.max().item():.3e} kg/m^2 "
            f"(throughfall={throughfall_in.tolist()}, evap={evap_out.tolist()}, "
            f"col_delta={col_delta.tolist()}, runoff={runoff_out.tolist()})"
        )

    def test_water_budget_closes_zero_precip(self):
        # Edge case: zero precipitation, zero evap from SEB -> column
        # water content must be conserved (only redistribution between
        # layers; no runoff unless VIC fires on saturated moisture).
        # With wsoil=0.30 (mid-range, well below WSAT), VIC does not
        # fire, so column_delta + runoff should be ~0.
        state, forcing, static, cfg = _make_inputs(
            N=2, land_frac=1.0, rain_kg_m2=0.0, wsoil=0.30,
        )
        model = LandSurface(cfg)
        new_state, _ = model(state, forcing, static)

        moist_inp = _rebuild_soilmoist_inputs(model, state, forcing, static, cfg)
        direct_out = port_soilw(
            moist_inp, model.soil_table, model.veg_table,
            model.consts, cfg,
        )

        rdat = torch.tensor(cfg.soil_thickness[:cfg.n_soil], dtype=cfg.dtype)
        rho_h2o = float(model.soil_table.consts.rho_h2o)
        col_delta = (
            (new_state.soil_w_liq - state.soil_w_liq) * rdat * rho_h2o
        ).sum(dim=-1)
        from surf_pytorch.config import (
            TILE_WET_SKIN, TILE_LOW_VEG_DRY,
            TILE_HIGH_VEG_DRY, TILE_BARE_SOIL,
        )
        soil_tiles = (TILE_WET_SKIN, TILE_LOW_VEG_DRY,
                      TILE_HIGH_VEG_DRY, TILE_BARE_SOIL)
        pevapti = moist_inp.pevapti
        pfrti = moist_inp.pfrti
        evap_tile_mean = sum(pfrti[:, t] * pevapti[:, t] for t in soil_tiles)
        ptmst = float(forcing.dt.item())
        evap_out = -evap_tile_mean * ptmst
        precip_in = forcing.rain
        runoff_out = direct_out.pros + direct_out.prod

        lhs = precip_in - evap_out
        rhs = col_delta + runoff_out
        resid = (lhs - rhs).abs()
        assert resid.max().item() < 1e-6, (
            f"water budget residual too large (zero-precip case): "
            f"max|lhs - rhs| = {resid.max().item():.3e} kg/m^2 "
            f"(precip={precip_in.tolist()}, evap={evap_out.tolist()}, "
            f"col_delta={col_delta.tolist()}, runoff={runoff_out.tolist()})"
        )
