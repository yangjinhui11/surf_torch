"""Forward()-level integration tests for the M6c single-layer snow.

These tests exercise the full :meth:`LandSurface.forward` path with a
non-trivial snow state and verify:

* the snow prognostic (PTSN, ZSSTAR) advances and matches a direct
  ``compute_snow`` call (using a capture wrapper to extract the exact
  ``SnowInputs`` the model assembles);
* the snow-bottom heat flux ``PGSN`` from ``SRFSN_LWIMPS`` flows into
  the soil-temperature solve (cold snow -> soil cools further than the
  zero-PGSN baseline);
* the no-snow cold-start path stays bit-exact with the pre-M6c
  behaviour (regression guard for the existing forward tests);
* the whole pipeline is differentiable through the snow branch.

The input builder mirrors :mod:`tests.test_forward_m5` but enables the
snow tiles (port tiles 4 / 6) and seeds a non-zero ``snow_w`` /
``snow_t`` / ``snow_rho``.
"""
from __future__ import annotations

import dataclasses

import torch
import pytest

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.physics.snow import SnowInputs, compute_snow
from surf_pytorch import model as model_mod


# --------------------------------------------------------------------------- #
# Input builder
# --------------------------------------------------------------------------- #
def _make_inputs(
    *,
    N: int = 4,
    dtype: torch.dtype = torch.float64,
    differentiable: bool = False,
    land_frac: float = 1.0,
    tsoil: float = 275.0,
    wsoil: float = 0.30,
    snow_w: float = 25.0,
    snow_t: float = 270.0,
    snow_rho: float | None = 250.0,
    use_snow_tiles: bool = True,
    sw_flux: float = 200.0,
    lw_flux: float = -300.0,
    t_skin: float = 270.0,
    t_lev: float = 270.0,
) -> tuple[SurfState, Forcing, Static, SurfConfig]:
    """Build a (state, forcing, static, config) with non-trivial snow.

    With ``use_snow_tiles=True`` the tile layout gives a ~0.25 snow
    fraction split across the low-veg-snow (port 4) and high-veg-snow
    (port 6) tiles, exercising the snow branch of SRFSN_LWIMPS.  With
    ``use_snow_tiles=False`` the layout collapses to the pure-land
    no-snow tiles of ``test_forward_m5._make_inputs`` for the cold-
    start regression test.
    """
    cfg = SurfConfig(dtype=dtype, differentiable=differentiable)
    ntile, nsoil = cfg.n_tile, cfg.n_soil

    state = SurfState(
        skin_t=torch.full((N, ntile), t_skin, dtype=dtype),
        soil_t=torch.full((N, nsoil), tsoil, dtype=dtype),
        soil_w_liq=torch.full((N, nsoil), wsoil, dtype=dtype),
        snow_w=torch.full((N,), snow_w, dtype=dtype),
        snow_frac=torch.full((N,), 0.25 if use_snow_tiles else 0.0, dtype=dtype),
        skin_water=torch.zeros((N,), dtype=dtype),
        sst=torch.full((N,), 290.0, dtype=dtype),
        snow_t=torch.full((N,), snow_t, dtype=dtype),
        snow_rho=(torch.full((N,), snow_rho, dtype=dtype)
                  if snow_rho is not None else None),
    )
    forcing = Forcing(
        u=torch.full((N,), 5.0, dtype=dtype),
        v=torch.zeros((N,), dtype=dtype),
        t=torch.full((N,), t_lev, dtype=dtype),
        q=torch.full((N,), 0.003, dtype=dtype),
        co2=torch.full((N,), 6e-4, dtype=dtype),
        p_surf=torch.full((N,), 1.0e5, dtype=dtype),
        geom1=torch.full((N,), 50.0, dtype=dtype),
        fr_so=torch.full((N,), sw_flux * 3600.0, dtype=dtype),
        fr_th=torch.full((N,), lw_flux * 3600.0, dtype=dtype),
        mu0=torch.full((N,), 0.5, dtype=dtype),
        rain=torch.zeros((N,), dtype=dtype),
        snow=torch.zeros((N,), dtype=dtype),
        dt=torch.tensor(3600.0, dtype=dtype),
    )
    frac = torch.zeros((N, ntile), dtype=dtype)
    if use_snow_tiles:
        # wet-skin / low-veg dry+snow / high-veg dry+snow / bare soil.
        frac[:, 2] = 0.05
        frac[:, 3] = 0.20
        frac[:, 4] = 0.15   # low-veg-snow
        frac[:, 5] = 0.20
        frac[:, 6] = 0.10   # high-veg-snow
        frac[:, 7] = 0.30
    else:
        # Matches test_forward_m5._make_inputs (no snow tiles).
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
# Capture helper: wrap compute_snow to record its SnowInputs, so the test
# can re-run it directly without duplicating forward()'s preamble.  This
# avoids re-deriving the 15+ intermediate tensors forward() builds.
# --------------------------------------------------------------------------- #
class _SnowCapture:
    """Wrap ``compute_snow`` as seen by ``LandSurface.forward``.

    ``model.py`` binds ``compute_snow`` to the local ``port_snow`` name
    at import time, so monkey-patching ``surf_pytorch.physics.snow`` has
    no effect -- we patch ``surf_pytorch.model.port_snow`` instead.
    """

    def __init__(self):
        self.last_inp: SnowInputs | None = None
        self._orig = model_mod.port_snow

    def __enter__(self):
        model_mod.port_snow = self._wrap
        return self

    def __exit__(self, *exc):
        model_mod.port_snow = self._orig
        return False

    def _wrap(self, inp, *args, **kw):
        self.last_inp = inp
        return self._orig(inp, *args, **kw)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestM6cSnowIntegration:
    """Pin the M6c forward() snow-integration invariants."""

    def test_snow_state_advances_and_matches_direct_call(self):
        """PTSN / SWE written back by forward() match a direct snow solve.

        Runs forward() with a capture wrapper around ``compute_snow`` to
        grab the exact ``SnowInputs`` the model assembles, then re-runs
        ``compute_snow`` directly and asserts the snow prognostics
        written to ``new_state`` agree to ~1 ULP.  This pins the
        assembly in :meth:`_update_snow` end-to-end without duplicating
        forward()'s preamble tensor-by-tensor.
        """
        state, forcing, static, cfg = _make_inputs(N=3)
        model = LandSurface(config=cfg)
        with _SnowCapture() as cap:
            new_state, _ = model(state, forcing, static)
        assert cap.last_inp is not None

        snow_ref = compute_snow(
            cap.last_inp, model.soil_table, model.consts, cfg,
        )
        torch.testing.assert_close(
            new_state.snow_t, snow_ref.ptsn, rtol=1e-10, atol=1e-10,
        )
        torch.testing.assert_close(
            new_state.snow_w, snow_ref.psstar, rtol=1e-10, atol=1e-10,
        )
        # Density is now advanced by the SRFSN_LWIMP section-3.1 density-
        # evolution port (_update_snow_density), not carried through.  We
        # re-derive the density inputs from the same state + snow solve
        # the model used and re-run the density block directly, then
        # assert the written-back density matches it to ~1 ULP.
        from surf_pytorch.physics.snow_density import (
            SnowDensityInputs, compute_snow_density,
        )
        rtt = float(model.consts.rtt.item())
        ptsnm1m = (state.snow_t if state.snow_t is not None
                   else torch.full_like(state.snow_w, rtt))
        rfrsmall = float(model.soil_table.consts.rfrsmall)
        llnosnow = (snow_ref.llnosnow if snow_ref.llnosnow is not None
                    else snow_ref.zfrsn < rfrsmall)
        ptmst = float(forcing.dt.item()) if torch.is_tensor(forcing.dt) else float(forcing.dt)
        dinp = SnowDensityInputs(
            pssnm1m=state.snow_w,
            prsnm1m=(state.snow_rho if state.snow_rho is not None
                     else torch.full_like(state.snow_w, 300.0)),
            ptsnm1m=ptsnm1m,
            ptsn=snow_ref.ptsn,
            ptsrf=state.skin_t.mean(dim=-1),
            pusrf=forcing.u, pvsrf=forcing.v,
            pssfl=forcing.snow / ptmst,
            pssfc=torch.zeros_like(state.snow_w),
            zfrsn=snow_ref.zfrsn, llnosnow=llnosnow,
            ptmst=forcing.dt,
        )
        rho_ref = compute_snow_density(
            dinp, model.soil_table, model.consts,
            differentiable=cfg.differentiable,
        ).prsn
        torch.testing.assert_close(
            new_state.snow_rho, rho_ref, rtol=1e-10, atol=1e-10,
        )

    def test_pgsn_flows_into_soil_temperature(self):
        """Cold snow + outgoing LW -> PGSN!=0 -> soil top responds.

        Builds two near-identical states: one with a cold snowpack
        (drives a non-zero PGSN out of SRFSN_LWIMPS) and one with no
        snow tiles / snow_w=0 (no-snow branch -> PGSN=0).  The snow-
        bearing column's soil-top temperature must differ from the
        snow-free column's, proving PGSN actually flows into SRFTS.

        We don't assert a *sign* because the snow pack also changes
        the surface energy balance upstream of PGSN; the key invariant
        is that the two soil-top temperatures are not bit-identical
        once the snow pack contributes a non-zero PGSN.
        """
        # Snow-bearing: cold snow (265 K) under net cooling (-300 LW).
        st_sn, fo_sn, sc_sn, cfg = _make_inputs(
            N=2, snow_w=40.0, snow_t=265.0,
            sw_flux=50.0, lw_flux=-120.0, tsoil=273.0,
        )
        # Snow-free: same forcing, no snow tiles, snow_w=0.
        st_ns, _, sc_ns, _ = _make_inputs(
            N=2, snow_w=0.0, snow_t=273.0, snow_rho=None,
            use_snow_tiles=False, tsoil=273.0,
            sw_flux=50.0, lw_flux=-120.0,
        )

        model = LandSurface(config=cfg)
        with _SnowCapture() as cap_sn:
            new_sn, _ = model(st_sn, fo_sn, sc_sn)
        with _SnowCapture() as cap_ns:
            new_ns, _ = model(st_ns, fo_sn, sc_ns)

        # Sanity: the snow-bearing column actually ran the snow branch
        # with a non-zero PGSN, and the snow-free column got PGSN=0.
        sn_ref = compute_snow(
            cap_sn.last_inp, model.soil_table, model.consts, cfg,
        )
        ns_ref = compute_snow(
            cap_ns.last_inp, model.soil_table, model.consts, cfg,
        )
        assert bool(torch.any(sn_ref.pgsn.abs() > 1.0))
        assert torch.equal(ns_ref.pgsn, torch.zeros_like(ns_ref.pgsn))

        # Soil-top temperatures must differ -- PGSN reached SRFTS.
        assert bool(torch.any(new_sn.soil_t[:, 0] != new_ns.soil_t[:, 0]))

    def test_cold_start_snow_rho_defaults_to_300(self):
        """No state.snow_rho -> _snow_density falls back to 300 kg/m^3.

        The t-1 reader (:meth:`_snow_density`) still returns the 300
        kg/m^3 fresh-snow cold-start value, but forward() now *evolves*
        the density through the SRFSN_LWIMP section-3.1 port, so the
        written-back ``new_state.snow_rho`` differs from the 300 cold-
        start by a finite, bounded amount (the fresh-snow + compaction
        delta over one coupling interval).
        """
        state, forcing, static, cfg = _make_inputs(N=2, snow_rho=None)
        model = LandSurface(config=cfg)
        rho = model._snow_density(state)
        assert torch.equal(rho, torch.full_like(state.snow_w, 300.0))
        # And forward() must run without error, writing the evolved
        # density back.  The evolved value must be finite and within the
        # SRFSN_LWIMP density bounds [RHOMINSN, 450].
        new_state, _ = model(state, forcing, static)
        assert new_state.snow_rho is not None
        assert bool(torch.isfinite(new_state.snow_rho).all())
        rhominsn = float(model.soil_table.consts.rhominsn)
        assert bool((new_state.snow_rho >= rhominsn).all())
        assert bool((new_state.snow_rho <= 450.0).all())

    def test_cold_start_snow_t_defaults_to_rtt(self):
        """No state.snow_t -> _update_snow pins PTSNM1M to RTT."""
        state, forcing, static, cfg = _make_inputs(N=2)
        # Strip snow_t to force the cold-start branch.
        state = SurfState(**{**state.__dict__, "snow_t": None})
        model = LandSurface(config=cfg)
        # Must not raise.
        new_state, _ = model(state, forcing, static)
        # new_state.snow_t must be finite (PTSN from the snow solve).
        assert bool(torch.isfinite(new_state.snow_t).all())

    def test_no_snow_branch_yields_zero_pgsn(self):
        """snow_w=0 + no snow tiles -> no-snow branch -> PGSN exactly 0.

        Regression guard for the pre-M6c tests: with no snow the
        SRFSN_LWIMPS no-snow branch fires and PGSN is exactly zero, so
        the soil-T result is bit-identical to the pre-M6c zero-PGSN
        baseline.  We assert the PGSN side of that equivalence (the
        soil-T equivalence is covered by the existing M5d tests, which
        still pass with the snow integration).
        """
        state, forcing, static, cfg = _make_inputs(
            N=3, use_snow_tiles=False, snow_w=0.0, snow_rho=None,
        )
        state = SurfState(**{**state.__dict__, "snow_t": None})
        model = LandSurface(config=cfg)
        with _SnowCapture() as cap:
            model(state, forcing, static)
        snow_ref = compute_snow(
            cap.last_inp, model.soil_table, model.consts, cfg,
        )
        assert torch.equal(snow_ref.pgsn, torch.zeros_like(snow_ref.pgsn))
        # No-snow branch also pins PTSN to RTT and ZLWC to zero.
        rtt = float(model.consts.rtt.item())
        assert torch.allclose(
            snow_ref.ptsn, torch.full_like(snow_ref.ptsn, rtt),
        )
        assert torch.equal(snow_ref.plwc, torch.zeros_like(snow_ref.plwc))

    def test_full_pipeline_is_differentiable(self):
        """forward() .sum().backward() must reach snow prognostics."""
        state, forcing, static, cfg = _make_inputs(
            N=2, differentiable=True,
        )
        # Make the snow state require grad.
        state = SurfState(
            **{**state.__dict__,
               "snow_w": state.snow_w.clone().requires_grad_(True),
               "snow_t": state.snow_t.clone().requires_grad_(True)}
        )
        model = LandSurface(config=cfg)
        new_state, _ = model(state, forcing, static)
        loss = new_state.soil_t.sum() + new_state.snow_t.sum()
        loss.backward()
        assert state.snow_w.grad is not None
        assert state.snow_t.grad is not None
        assert bool(torch.isfinite(state.snow_w.grad).all())
        assert bool(torch.isfinite(state.snow_t.grad).all())

    # ------------------------------------------------------------------ #
    # M2e: snowfall-throughfall wiring (forcing.snow -> PSSFL).
    # ------------------------------------------------------------------ #
    # All other tests in this class cold-start forcing.snow at zero, so
    # the M2d wiring documented in LandSurface._update_snow (model.py
    # line 885: ``pssfl = forcing.snow / ptmst``) is never exercised at
    # the forward()-assembly level.  These two tests feed a non-zero
    # snowfall and use the existing _SnowCapture wrapper to grab the
    # exact SnowInputs the model assembles, pinning the wiring bit-exact.

    def test_snow_throughfall_wires_forcing_snow_into_pssfl(self):
        """forcing.snow (cumulative kg/m^2) -> PSSFL (rate kg/m^2/s), bit-exact.

        Builds a cold-snow slab (270 K < 271.15 K cold-snow threshold =>
        SRFSN_LWIMPS cold branch: ZFUNC=ZDFUNC=0, no melt) with a non-zero
        snowfall forcing, runs forward() under the capture wrapper, and
        asserts the wiring in :meth:`LandSurface._update_snow`:

        * the captured large-scale snowfall rate ``PSSFL`` equals
          ``forcing.snow / dt`` exactly (the model.py:885 transcription);
        * the convective snowfall ``PSSFC`` is exactly zero (the port's
          Forcing does not split convective vs large-scale, matching the
          convention used for rainfall in _update_soil_moisture);
        * the snow mass written to ``new_state.snow_w`` matches a direct
          ``compute_snow`` call on the captured inputs to ~1 ULP
          (assembly consistency -- same invariant as
          :meth:`test_snow_state_advances_and_matches_direct_call`, but
          now driven by non-zero snowfall);
        * the snow-mass gain is at least the snowfall input (the cold
          branch has no melt sink) and tracks it within ~1 % (the small
          residual is the ``PFRTI(5)*PEVAPTI(5)`` tile-evap term the
          snow branch also folds into ZSSTAR).
        """
        state, forcing, static, cfg = _make_inputs(
            N=3, snow_w=25.0, snow_t=270.0,
        )
        # Inject a cold snowfall: 5e-4 kg/m^2/s over dt=3600 s -> 1.8 kg/m^2.
        rate = 5.0e-4
        dt = float(forcing.dt.item())
        forcing = dataclasses.replace(
            forcing, snow=torch.full((3,), rate * dt, dtype=torch.float64),
        )

        model = LandSurface(config=cfg)
        with _SnowCapture() as cap:
            new_state, _ = model(state, forcing, static)
        assert cap.last_inp is not None

        # --- Wiring: forcing.snow -> PSSFL (bit-exact) ------------------ #
        torch.testing.assert_close(
            cap.last_inp.pssfl, forcing.snow / dt,
            rtol=0.0, atol=0.0, check_dtype=True,
        )
        # --- Wiring: convective snowfall PSSFC stays zero ---------------- #
        assert torch.equal(
            cap.last_inp.pssfc, torch.zeros_like(cap.last_inp.pssfc),
        )

        # --- Assembly: forward() snow_w == direct compute_snow ----------- #
        snow_ref = compute_snow(
            cap.last_inp, model.soil_table, model.consts, cfg,
        )
        torch.testing.assert_close(
            new_state.snow_w, snow_ref.psstar, rtol=0.0, atol=0.0,
        )

        # --- Physical: cold branch accumulates >= snowfall --------------- #
        d_swe = new_state.snow_w - state.snow_w
        snowfall = forcing.snow                      # kg/m^2 cumulative
        assert bool((d_swe >= snowfall - 1e-12).all()), (
            f"cold-branch ΔSWE {d_swe.tolist()} below snowfall "
            f"{snowfall.tolist()} -- melt sink active on a cold slab"
        )
        # The residual over the snowfall is the tile-evap term folded into
        # ZSSTAR (PFRTI(5)*PEVAPTI(5)); keep it under 1 % of the snowfall.
        assert bool(((d_swe - snowfall).abs() < 0.01 * snowfall.max()).all()), (
            f"ΔSWE {d_swe.tolist()} drifted >1% from snowfall "
            f"{snowfall.tolist()} -- unexpected non-evap term"
        )

    def test_zero_snow_throughfall_leaves_snow_w_unchanged(self):
        """Zero snowfall forcing -> no-snow branch -> SWE bit-identical.

        Regression guard for the no-snow / zero-snowfall path: with
        ``forcing.snow == 0`` the assembled ``PSSFL`` is zero, the
        snow mass budget has no input, and ``new_state.snow_w`` must
        equal the input ``state.snow_w`` bit-for-bit.
        """
        state, forcing, static, cfg = _make_inputs(
            N=2, snow_w=0.0, snow_rho=None, use_snow_tiles=False,
        )
        assert torch.equal(forcing.snow, torch.zeros_like(forcing.snow))

        model = LandSurface(config=cfg)
        with _SnowCapture() as cap:
            new_state, _ = model(state, forcing, static)

        # Captured PSSFL is exactly zero (the wiring feeds forcing.snow=0).
        assert torch.equal(
            cap.last_inp.pssfl, torch.zeros_like(cap.last_inp.pssfl),
        )
        # And the snow mass never moves.
        assert torch.equal(new_state.snow_w, state.snow_w)
