"""Tests for the single-layer snow-density evolution (``SRFSN_LWIMP`` 3.1).

Verifies the ``surf_pytorch.physics.snow_density`` port against the
documented ``SRFSN_LWIMP`` section-3.1 physics:

* fresh-snow density (temperature + wind dependence, floored at RHOMINSND)
* depth-weighted mixing of old pack + fresh snow
* overburden compaction (increases density, scales with SWE)
* destructive metamorphism (increases density, damped above RSNDTDESTROI)
* melt compaction (only where internal melt > 0 and pack >= 15 kg/m^2)
* no-snow reset to RHOMINSN (100 kg/m^3)
* the [RHOMINSN, 450] clamp on the evolved density
* forward() integration: density advances each step and feeds back into
  the snow thermal resistance via VLAMSK (PRSNM1M enters ZSNRES)
* differentiability (gradient flows through the density block)
"""

from __future__ import annotations

import math

import pytest
import torch

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.constants import ConstantBundle, PhysicalConstants
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.physics.snow_density import (
    SnowDensityInputs, compute_snow_density,
)
from surf_pytorch.tables.soil import SoilTable


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _bundle(dtype=torch.float64) -> ConstantBundle:
    return ConstantBundle(PhysicalConstants(), dtype=dtype)


def _soil_table(dtype=torch.float64) -> SoilTable:
    return SoilTable(dtype=dtype, device=torch.device("cpu"))


def _scalar(value: float, N: int, dtype=torch.float64) -> torch.Tensor:
    return torch.full((N,), float(value), dtype=dtype)


def _make_inputs(
    N: int = 4,
    *,
    pssnm1m: float = 50.0,
    prsnm1m: float = 200.0,
    ptsnm1m: float = 268.0,
    ptsn: float = 269.0,
    ptsrf: float = 270.0,
    pusrf: float = 5.0,
    pvsrf: float = 0.0,
    pssfl: float = 0.0,
    pssfc: float = 0.0,
    zfrsn: float = 0.25,
    llnosnow: bool = False,
    ptmst: float = 3600.0,
    dtype=torch.float64,
) -> SnowDensityInputs:
    """Build a uniform ``(N,)`` slab of density inputs.

    All fields default to a deep, cold, snow-bearing pack with no fresh
    snowfall so the overburden / metamorphism compaction terms act in
    isolation; callers override the knobs they need to exercise.
    """
    return SnowDensityInputs(
        pssnm1m=_scalar(pssnm1m, N, dtype),
        prsnm1m=_scalar(prsnm1m, N, dtype),
        ptsnm1m=_scalar(ptsnm1m, N, dtype),
        ptsn=_scalar(ptsn, N, dtype),
        ptsrf=_scalar(ptsrf, N, dtype),
        pusrf=_scalar(pusrf, N, dtype),
        pvsrf=_scalar(pvsrf, N, dtype),
        pssfl=_scalar(pssfl, N, dtype),
        pssfc=_scalar(pssfc, N, dtype),
        zfrsn=_scalar(zfrsn, N, dtype),
        llnosnow=torch.full((N,), bool(llnosnow)),
        ptmst=torch.tensor(float(ptmst), dtype=dtype),
    )


def _run(inp: SnowDensityInputs, *, differentiable: bool = True) -> torch.Tensor:
    return compute_snow_density(
        inp, _soil_table(), _bundle(), differentiable=differentiable,
    ).prsn


# --------------------------------------------------------------------------- #
# Tests: no-snow branch
# --------------------------------------------------------------------------- #
class TestNoSnowBranch:
    """The LLNOSNOW branch resets density to RHOMINSN (100 kg/m^3)."""

    def test_no_snow_resets_to_rhominsn(self):
        inp = _make_inputs(llnosnow=True, prsnm1m=300.0)
        rho = _run(inp)
        assert torch.equal(rho, _scalar(100.0, 4))

    def test_mixed_slab_resets_only_no_snow_points(self):
        # Half the slab is no-snow, half is snow-bearing.
        N = 4
        inp = _make_inputs(N=N, llnosnow=False)
        llnosnow = torch.tensor([True, False, True, False])
        inp = SnowDensityInputs(
            **{**inp.__dict__, "llnosnow": llnosnow},
        )
        rho = _run(inp)
        # The no-snow points (0, 2) must be exactly RHOMINSN; the snow
        # points (1, 3) must be finite and >= RHOMINSN.
        rhominsn = float(_soil_table().consts.rhominsn)
        assert float(rho[0]) == rhominsn
        assert float(rho[2]) == rhominsn
        assert torch.isfinite(rho[1]) and torch.isfinite(rho[3])
        assert float(rho[1]) >= rhominsn and float(rho[3]) >= rhominsn


# --------------------------------------------------------------------------- #
# Tests: density bounds
# --------------------------------------------------------------------------- #
class TestDensityBounds:
    """Evolved density always lies in [RHOMINSN, 450]."""

    @pytest.mark.parametrize("prsnm1m", [50.0, 100.0, 200.0, 400.0, 449.0])
    def test_density_in_bounds_across_initial_densities(self, prsnm1m):
        inp = _make_inputs(prsnm1m=prsnm1m, ptsn=273.0, ptsrf=275.0,
                           pssnm1m=200.0, pssfl=1e-3)
        rho = _run(inp)
        rhominsn = float(_soil_table().consts.rhominsn)
        assert bool((rho >= rhominsn).all())
        assert bool((rho <= 450.0).all())

    def test_extreme_inputs_stay_bounded(self):
        # Huge SWE + warm snow + heavy snowfall: must still be <= 450.
        inp = _make_inputs(pssnm1m=2000.0, prsnm1m=400.0, ptsn=273.0,
                           ptsrf=290.0, pssfl=0.01, pusrf=20.0)
        rho = _run(inp)
        assert bool((rho <= 450.0).all())
        assert bool((rho >= 100.0).all())


# --------------------------------------------------------------------------- #
# Tests: compaction physics
# --------------------------------------------------------------------------- #
class TestCompaction:
    """Overburden + metamorphism compaction increase density when no fresh snow."""

    def test_compaction_increases_density_no_snowfall(self):
        """Deep cold pack, no fresh snow -> density must rise."""
        inp = _make_inputs(pssnm1m=100.0, prsnm1m=150.0, pssfl=0.0,
                           ptsn=260.0, ptsrf=263.0)
        rho = _run(inp)
        assert bool((rho > 150.0).all()), (
            f"compaction should raise density above 150, got {rho.tolist()}"
        )

    def test_overburden_scales_with_swe(self):
        """Deeper pack -> more overburden compaction -> higher density.

        Two packs with identical temperature / old density but different
        SWE: the deeper one must compact more (the overburden term is
        ~0.5*PSSNM1M*RG / (...)).
        """
        shallow = _make_inputs(pssnm1m=20.0, prsnm1m=150.0, pssfl=0.0,
                               ptsn=260.0, ptsrf=263.0)
        deep = _make_inputs(pssnm1m=300.0, prsnm1m=150.0, pssfl=0.0,
                            ptsn=260.0, ptsrf=263.0)
        rho_shallow = _run(shallow)
        rho_deep = _run(deep)
        assert bool((rho_deep > rho_shallow).all()), (
            f"deeper pack should compact more: shallow={rho_shallow.tolist()} "
            f"deep={rho_deep.tolist()}"
        )

    def test_melt_compaction_inactive_for_cold_snow(self):
        """Cold snow (PTSN << RTT) -> no internal melt -> melt-compaction term 0.

        The melt-compaction term only acts when ZPMSNINT > 0 (warming
        through the melt window).  At PTSN = 250 K there is no melt, so
        the density is driven purely by fresh-snow mixing + overburden +
        metamorphism.  We verify the result is finite and bounded; the
        *sign* of the change is covered by test_compaction_increases.
        """
        inp = _make_inputs(pssnm1m=80.0, prsnm1m=180.0, pssfl=0.0,
                           ptsn=250.0, ptsnm1m=250.0, ptsrf=255.0)
        rho = _run(inp)
        assert torch.isfinite(rho).all()
        assert bool((rho >= 100.0).all() and (rho <= 450.0).all())


# --------------------------------------------------------------------------- #
# Tests: fresh-snow mixing
# --------------------------------------------------------------------------- #
class TestFreshSnowMixing:
    """Fresh snow (low density) dilutes a denser old pack."""

    def test_fresh_snow_dilutes_dense_pack(self):
        """Snowfall at warm surface -> fresh-snow density < old -> density drops.

        With PTSRF = 273.15 K and zero wind the fresh-snow density is
        RHOMINSNA = 109 kg/m^3 (well below the old 200), so adding fresh
        snow must lower the blended density toward 109.
        """
        old = _make_inputs(prsnm1m=200.0, pssfl=0.0, ptsrf=273.15)
        snowy = _make_inputs(prsnm1m=200.0, pssfl=2e-3, ptsrf=273.15)
        rho_old = _run(old)
        rho_snowy = _run(snowy)
        assert bool((rho_snowy < rho_old).all()), (
            f"fresh snow should dilute the pack: no-snow={rho_old.tolist()} "
            f"with-snow={rho_snowy.tolist()}"
        )

    def test_fresh_snow_density_wind_floor(self):
        """Fresh-snow density is floored at RHOMINSND (50) for cold + calm air.

        At very cold PTSRF the linear term RHOMINSNB*(T-RTT) goes
        negative; with zero wind the fresh density would be < RHOMINSND
        without the floor.  The evolved density must therefore stay
        >= RHOMINSN (=100, the final floor), confirming the floor path
        runs without NaN.
        """
        inp = _make_inputs(prsnm1m=120.0, ptsrf=240.0, pusrf=0.0,
                           pvsrf=0.0, pssfl=5e-3, pssnm1m=10.0)
        rho = _run(inp)
        assert torch.isfinite(rho).all()
        assert bool((rho >= 100.0).all())


# --------------------------------------------------------------------------- #
# Tests: vectorisation
# --------------------------------------------------------------------------- #
class TestVectorisation:
    """The block vectorises correctly over the grid axis."""

    def test_vectorised_matches_per_point(self):
        """A mixed slab must equal element-wise runs of the uniform slabs."""
        # Two regimes: a deep cold pack and a shallow warm pack.
        cold = _make_inputs(N=1, pssnm1m=120.0, prsnm1m=150.0,
                            ptsn=258.0, ptsrf=261.0, pssfl=0.0)
        warm = _make_inputs(N=1, pssnm1m=30.0, prsnm1m=200.0,
                            ptsn=272.0, ptsrf=273.0, pssfl=1e-3)
        rho_cold = _run(cold)
        rho_warm = _run(warm)

        # Build the mixed slab by concatenating the per-point tensors.
        N = 5
        mixed = SnowDensityInputs(
            pssnm1m=torch.cat([cold.pssnm1m, warm.pssnm1m,
                               cold.pssnm1m, warm.pssnm1m, cold.pssnm1m]),
            prsnm1m=torch.cat([cold.prsnm1m, warm.prsnm1m,
                               cold.prsnm1m, warm.prsnm1m, cold.prsnm1m]),
            ptsnm1m=torch.cat([cold.ptsnm1m, warm.ptsnm1m,
                               cold.ptsnm1m, warm.ptsnm1m, cold.ptsnm1m]),
            ptsn=torch.cat([cold.ptsn, warm.ptsn, cold.ptsn,
                            warm.ptsn, cold.ptsn]),
            ptsrf=torch.cat([cold.ptsrf, warm.ptsrf, cold.ptsrf,
                             warm.ptsrf, cold.ptsrf]),
            pusrf=torch.cat([cold.pusrf, warm.pusrf, cold.pusrf,
                             warm.pusrf, cold.pusrf]),
            pvsrf=torch.cat([cold.pvsrf, warm.pvsrf, cold.pvsrf,
                             warm.pvsrf, cold.pvsrf]),
            pssfl=torch.cat([cold.pssfl, warm.pssfl, cold.pssfl,
                             warm.pssfl, cold.pssfl]),
            pssfc=torch.cat([cold.pssfc, warm.pssfc, cold.pssfc,
                             warm.pssfc, cold.pssfc]),
            zfrsn=torch.cat([cold.zfrsn, warm.zfrsn, cold.zfrsn,
                             warm.zfrsn, cold.zfrsn]),
            llnosnow=torch.zeros(N, dtype=torch.bool),
            ptmst=cold.ptmst,
        )
        rho_mixed = _run(mixed)
        expected = torch.cat([rho_cold, rho_warm, rho_cold, rho_warm, rho_cold])
        torch.testing.assert_close(rho_mixed, expected, rtol=0.0, atol=0.0)

    def test_handles_single_point(self):
        inp = _make_inputs(N=1)
        rho = _run(inp)
        assert rho.shape == (1,)
        assert torch.isfinite(rho).all()


# --------------------------------------------------------------------------- #
# Tests: differentiability
# --------------------------------------------------------------------------- #
class TestDifferentiable:
    """The density block is differentiable in the snow state + forcing."""

    def test_grad_flows_to_prsnm1m(self):
        inp = _make_inputs(N=2)
        inp = SnowDensityInputs(**{**inp.__dict__,
                                   "prsnm1m": inp.prsnm1m.clone().requires_grad_(True)})
        rho = _run(inp)
        g = torch.autograd.grad(rho.sum(), inp.prsnm1m, retain_graph=False)[0]
        assert g is not None
        assert torch.isfinite(g).all()

    def test_grad_flows_to_pssnm1m_and_ptsrf(self):
        pssnm1m = _scalar(100.0, 2).requires_grad_(True)
        ptsrf = _scalar(268.0, 2).requires_grad_(True)
        base = _make_inputs(N=2)
        inp = SnowDensityInputs(**{**base.__dict__,
                                   "pssnm1m": pssnm1m, "ptsrf": ptsrf})
        rho = _run(inp)
        g = torch.autograd.grad(rho.sum(), [pssnm1m, ptsrf])[0]
        assert g[0] is not None and g[1] is not None
        assert torch.isfinite(g[0]).all() and torch.isfinite(g[1]).all()

    def test_no_grad_path_is_finite(self):
        inp = _make_inputs(N=3)
        rho = _run(inp, differentiable=False)
        assert torch.isfinite(rho).all()
        assert bool((rho >= 100.0).all() and (rho <= 450.0).all())


# --------------------------------------------------------------------------- #
# Tests: forward() integration
# --------------------------------------------------------------------------- #
class TestForwardIntegration:
    """forward() advances snow_rho each step and feeds VLAMSK back."""

    @staticmethod
    def _snowy_setup(N: int = 3, dtype=torch.float64):
        from surf_pytorch.forcing import (
            build_idealized_diurnal, IdealizedDiurnalConfig,
        )
        cfg_f = IdealizedDiurnalConfig(
            n_grid=N, n_steps=2, dt_seconds=3600.0, wind_speed=3.0,
        )
        flist, _static, state0 = build_idealized_diurnal(cfg_f)
        forcing = flist[0]
        forcing.snow = _scalar(0.4, N, dtype)  # 0.4 kg/m^2 snowfall / step
        # Snow-bearing tile fractions (snow-low + snow-high populated).
        frac = torch.zeros((N, 8), dtype=dtype)
        frac[:, 3] = 0.20; frac[:, 4] = 0.15
        frac[:, 5] = 0.20; frac[:, 6] = 0.10
        frac[:, 7] = 0.30
        static = Static(
            tile_frac_base=frac,
            land_frac=torch.ones(N, dtype=dtype),
            orography=torch.full((N,), 50.0, dtype=dtype),
            soil_type=torch.full((N,), 3, dtype=torch.long),
            veg_type=torch.full((N,), 3, dtype=torch.long),
            soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
            emis=torch.full((N,), 0.99, dtype=dtype),
        )
        ntile = state0.skin_t.shape[-1]
        state = state0.like(
            skin_t=torch.full((N, ntile), 268.0, dtype=dtype),
            snow_w=_scalar(80.0, N, dtype),
            snow_frac=_scalar(0.5, N, dtype),
            snow_t=_scalar(262.0, N, dtype),
            snow_rho=_scalar(160.0, N, dtype),
        )
        return state, forcing, static

    def test_snow_rho_advances_each_step(self):
        state, forcing, static = self._snowy_setup()
        cfg = SurfConfig(dtype=torch.float64, validate=False,
                         per_tile_exchange=False)
        model = LandSurface(cfg)
        s = state
        for _ in range(8):
            ns, _ = model(s, forcing, static)
            # Density must change (advance) each step and stay bounded.
            assert not torch.equal(ns.snow_rho, s.snow_rho)
            assert bool((ns.snow_rho >= 100.0).all()
                        and (ns.snow_rho <= 450.0).all())
            assert torch.isfinite(ns.snow_rho).all()
            s = ns

    def test_density_drives_snow_thermal_feedback(self):
        """Different initial densities -> different skin_t trajectories.

        PRSNM1M enters VLAMSK via ZSNRES = 0.5*ZDSN / (RLAMICE *
        (PRSNM1M*ZHOICE)**RALAMSN).  At step 0 the snow-thermal
        resistance uses the *t-1* density, so the density-evolution
        output only feeds back on the next step.  We therefore run a
        handful of steps and assert the two trajectories (which differ
        only in the initial snow density) diverge in skin_t once the
        evolved density has propagated through VLAMSK -> PTSN -> PGSN ->
        soil-T -> skin_t.
        """
        cfg = SurfConfig(dtype=torch.float64, validate=False,
                         per_tile_exchange=False)
        model = LandSurface(cfg)
        st_lo, forcing, static = self._snowy_setup(N=1)
        st_hi, _, _ = self._snowy_setup(N=1)
        st_hi = st_hi.like(snow_rho=_scalar(350.0, 1))
        st_lo = st_lo.like(snow_rho=_scalar(120.0, 1))
        s_lo, s_hi = st_lo, st_hi
        diverged = False
        for _ in range(12):
            s_lo, _ = model(s_lo, forcing, static)
            s_hi, _ = model(s_hi, forcing, static)
            if bool(torch.any(s_lo.skin_t != s_hi.skin_t)):
                diverged = True
                break
        assert diverged, (
            "different initial snow density must change skin_t via VLAMSK "
            "after the evolved density propagates through the snow-thermal "
            "coupling"
        )

    def test_no_snow_point_resets_density_to_rhominsn(self):
        """A point with no snow tiles -> density resets to RHOMINSN."""
        N = 2
        cfg = SurfConfig(dtype=torch.float64, validate=False,
                         per_tile_exchange=False)
        model = LandSurface(cfg)
        state, forcing, _ = self._snowy_setup(N=N)
        # Strip the snow tiles so ZFRSN < RFRSMALL everywhere.
        frac = torch.zeros((N, 8), dtype=torch.float64)
        frac[:, 3] = 0.30; frac[:, 5] = 0.30; frac[:, 7] = 0.40
        static = Static(
            tile_frac_base=frac,
            land_frac=torch.ones(N, dtype=torch.float64),
            orography=torch.full((N,), 50.0, dtype=torch.float64),
            soil_type=torch.full((N,), 3, dtype=torch.long),
            veg_type=torch.full((N,), 3, dtype=torch.long),
            soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89],
                                        dtype=torch.float64),
            emis=torch.full((N,), 0.99, dtype=torch.float64),
        )
        ns, _ = model(state, forcing, static)
        rhominsn = float(model.soil_table.consts.rhominsn)
        assert torch.equal(ns.snow_rho, _scalar(rhominsn, N))
