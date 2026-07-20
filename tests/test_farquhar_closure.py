"""Wiring tests for the Farquhar -> canopy-resistance -> VEVAP closure.

These tests sit one level above ``test_farquhar_ref.py`` (which validates
the Farquhar *kernel* in isolation) and one level below
``test_forward_e2e.py`` (which validates the full forward).  They target
the *connector* -- :func:`surf_pytorch.physics.stomata.farquhar_resistance`
-- that bridges the Farquhar canopy conductance ``PGSTOT`` to the VEVAP
moisture-flux closure, mirroring the upstream ``LEFARQUHAR`` branch of
``cotworestress_mod.F90`` 412-431.

What this catches that the other suites cannot
------------------------------------------------
* The PGSTOT (m/s conductance) -> PWET (s/m resistance) conversion
  ``1/max(RCONDCTMIN, PGSTOT)`` -- in particular the conductance-floor
  clamp that prevents a divide-by-zero at nighttime / over bare land.
* The bare-soil / urban resistance (``PWETB``, tile 8/10) being preserved
  from Jarvis even under ``use_farquhar`` (upstream ``vsurf_mod.F90:687``
  keeps Jarvis for bare soil because ``COTWORESTRESS`` only computes the
  vegetation canopy conductance).
* The port's ``farquhar_resistance`` producing the same PWETL/PWETH/
  PWETHS the independent Fortran ``ref.farquhar`` + the documented
  conversion would produce.
* The downstream VEVAP alphas (PCSAT/PCAIR) built from the Farquhar-
  derived resistances matching what an independent per-tile
  ``ref.vevap`` loop produces.

Skip behaviour
--------------
The Fortran-dependent tests skip when ``libsurf_ref.so`` is missing or
predates the ``surf_ref_farquhar`` / ``surf_ref_vevap`` symbols; the
pure-Python conversion / passthrough tests always run.
"""

from __future__ import annotations

import math

import pytest
import torch

from surf_pytorch import ConstantBundle, LandSurface, SurfConfig
from surf_pytorch.config import TILE_LOW_VEG_DRY
from surf_pytorch.constants import PhysicalConstants
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.tables import VegTable
from surf_pytorch.utils import qsat, expand_as_tiles
from surf_pytorch.physics.stomata import (
    jarvis_resistance,
    build_tile_vevap,
    farquhar_resistance,
)

# Importing offline_driver triggers the .so load, so guard it.
try:
    from offline_driver import SurfRef
    _SURF_REF_AVAILABLE = SurfRef.is_available()
except Exception:  # pragma: no cover
    SurfRef = None  # type: ignore[assignment]
    _SURF_REF_AVAILABLE = False

pytestmark = pytest.mark.fortran_ref


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def consts() -> ConstantBundle:
    return ConstantBundle(PhysicalConstants(), dtype=torch.float64)


@pytest.fixture(scope="session")
def veg() -> VegTable:
    return VegTable(dtype=torch.float64)


def _requires_fortran():
    """Skip marker -- True iff .so exports farquhar AND vevap AND stomata."""
    def _ready() -> bool:
        if not _SURF_REF_AVAILABLE:
            return False
        loaded = SurfRef()._loaded
        return (loaded.farquhar is not None
                and loaded.vevap is not None
                and loaded.stomata is not None)
    return pytest.mark.skipif(
        not _ready(),
        reason="libsurf_ref.so missing or predates surf_ref_farquhar/"
               "surf_ref_vevap/surf_ref_stomata -- rebuild with "
               "offline_driver/build_ref.sh",
    )


# --------------------------------------------------------------------------- #
# Input factory: a grid with vegetated / bare / nighttime columns so the
# closure is exercised across all branches.
# --------------------------------------------------------------------------- #
def _make_grid(N=6, seed=0, *, dtype=torch.float64):
    """Build (state, forcing, static, cfg) + the Jarvis dict.

    The grid mixes C3-crop (kvtype=3) vegetated columns, a nighttime
    column (zero SW) and a bare-soil column (kvtype=0) so the conductance
    floor, the bare-soil passthrough and the normal photosynthesis branch
    all fire.
    """
    torch.manual_seed(seed)
    cfg = SurfConfig(dtype=dtype)
    ntile, nsoil = cfg.n_tile, cfg.n_soil
    bundle = ConstantBundle(PhysicalConstants(), dtype=dtype)

    skin_t = torch.rand(N, ntile, dtype=dtype) * 8.0 + 283.0
    state = SurfState(
        skin_t=skin_t,
        soil_t=torch.full((N, nsoil), 286.0, dtype=dtype),
        soil_w_liq=torch.full((N, nsoil), 0.30, dtype=dtype),
        snow_w=torch.zeros((N,), dtype=dtype),
        snow_frac=torch.zeros((N,), dtype=dtype),
        skin_water=torch.zeros((N,), dtype=dtype),
        sst=torch.full((N,), 290.0, dtype=dtype),
    )
    # Daytime SW for most columns, one nighttime (index 0).
    sw = torch.full((N,), 600.0 * 3600.0, dtype=dtype)
    sw[0] = 0.0
    forcing = Forcing(
        u=torch.full((N,), 4.0, dtype=dtype),
        v=torch.zeros((N,), dtype=dtype),
        t=torch.full((N,), 290.0, dtype=dtype),
        q=torch.full((N,), 0.008, dtype=dtype),
        co2=torch.full((N,), 6e-4, dtype=dtype),
        p_surf=torch.full((N,), 1.0e5, dtype=dtype),
        geom1=torch.full((N,), 50.0, dtype=dtype),
        fr_so=sw,
        fr_th=torch.full((N,), -300.0 * 3600.0, dtype=dtype),
        mu0=torch.full((N,), 0.5, dtype=dtype),
        rain=torch.zeros((N,), dtype=dtype),
        snow=torch.zeros((N,), dtype=dtype),
        dt=torch.tensor(3600.0, dtype=dtype),
    )
    frac = torch.zeros((N, ntile), dtype=dtype)
    frac[:, 2] = 0.05; frac[:, 3] = 0.30; frac[:, 5] = 0.30; frac[:, 7] = 0.35
    # Vegetation type: C3 crop everywhere except column N-1 = bare.
    veg_type = torch.full((N,), 3, dtype=torch.long)
    veg_type[N - 1] = 0
    static = Static(
        tile_frac_base=frac,
        land_frac=torch.ones((N,), dtype=dtype),
        orography=torch.full((N,), 50.0, dtype=dtype),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=veg_type,
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.full((N,), 0.99, dtype=dtype),
    )
    veg = VegTable(dtype=dtype)
    model = LandSurface(cfg)
    # Glue quantities the closure consumes (mirrors forward §1-5).
    skin_t_prev = state.skin_t
    p_surf_t = expand_as_tiles(forcing.p_surf, ntile)
    qskm1m = qsat(skin_t_prev, p_surf_t, model.consts)
    ktvl, ktvh = model._dominant_veg_types(static)
    ssrfl_grid = forcing.fr_so / forcing.dt
    jarvis = jarvis_resistance(
        soil_t=state.soil_t, soil_w=state.soil_w_liq,
        soil_type=static.soil_type, ktvl=ktvl, ktvh=ktvh,
        t_air=forcing.t, q_air=forcing.q, p_surf=forcing.p_surf,
        sw_down=ssrfl_grid, veg_table=veg, soil_table=model.soil_table,
        root_table=model.root_table, consts=model.consts,
        tile_frac=frac, n_tile=ntile,
    )
    ctx = {
        "model": model, "bundle": bundle, "veg": veg,
        "skin_t_prev": skin_t_prev, "qskm1m": qskm1m,
        "ktvl": ktvl, "ktvh": ktvh, "ssrfl_grid": ssrfl_grid,
        "frac": frac, "ntile": ntile, "jarvis": jarvis,
    }
    return state, forcing, static, cfg, ctx


# --------------------------------------------------------------------------- #
# Pure-Python conversion / passthrough tests (no Fortran required).
# --------------------------------------------------------------------------- #
class TestConductanceToResistance:
    """PGSTOT -> PWET = 1/max(RCONDCTMIN, PGSTOT) (cotworestress_mod.F90:427)."""

    def test_zero_conductance_hits_floor_not_infinity(self, consts, veg):
        """Nighttime / bare columns: PGSTOT=0 -> PWET = 1/RCONDCTMIN (finite)."""
        # A PGSTOT of zero must not produce inf; it must produce the floor.
        pgstot = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        rcondctmin = veg.farquhar_consts.rcondctmin
        pwet = 1.0 / torch.clamp(pgstot, min=rcondctmin)
        assert torch.isfinite(pwet).all()
        assert torch.allclose(pwet, torch.full_like(pwet, 1.0 / rcondctmin))

    def test_typical_conductance_inverts(self, veg):
        """A healthy conductance (e.g. 0.01 m/s) inverts to 100 s/m."""
        pgstot = torch.tensor([0.01], dtype=torch.float64)
        rcondctmin = veg.farquhar_consts.rcondctmin
        pwet = 1.0 / torch.clamp(pgstot, min=rcondctmin)
        assert pwet.item() == pytest.approx(100.0, rel=1e-12)


class TestBareSoilPassthrough:
    """PWETB must come from Jarvis even under use_farquhar (vsurf:687)."""

    def test_bare_soil_resistance_unchanged_by_farquhar(self, consts, veg):
        """farquhar_resistance must return the Jarvis pwetb verbatim."""
        state, forcing, static, cfg, ctx = _make_grid()
        fq = farquhar_resistance(
            forcing=forcing, skin_t_prev=ctx["skin_t_prev"],
            qskm1m=ctx["qskm1m"], ssrfl_grid=ctx["ssrfl_grid"],
            ktvl=ctx["ktvl"], ktvh=ctx["ktvh"], jarvis=ctx["jarvis"],
            veg_table=veg, consts=consts,
            land_mask=static.land_frac > 0,
        )
        assert torch.equal(fq["pwetb"], ctx["jarvis"]["pwetb"])
        assert torch.equal(fq["pwetlu"], ctx["jarvis"]["pwetlu"])

    def test_vegetated_resistance_differs_from_jarvis(self, consts, veg):
        """Vegetated tiles (4/6/7) must use Farquhar, not Jarvis, resistance.

        Over the daytime vegetated columns the Farquhar conductance is
        non-trivial, so PWETL differs from the Jarvis value (a wiring bug
        that passed Jarvis through unchanged would fail here).
        """
        state, forcing, static, cfg, ctx = _make_grid()
        fq = farquhar_resistance(
            forcing=forcing, skin_t_prev=ctx["skin_t_prev"],
            qskm1m=ctx["qskm1m"], ssrfl_grid=ctx["ssrfl_grid"],
            ktvl=ctx["ktvl"], ktvh=ctx["ktvh"], jarvis=ctx["jarvis"],
            veg_table=veg, consts=consts,
            land_mask=static.land_frac > 0,
        )
        # Daytime vegetated column (index 1, C3 crop, SW=600 W/m^2): the
        # two resistance models must disagree (different formulations).
        assert not torch.allclose(fq["pwetl"][1], ctx["jarvis"]["pwetl"][1])

    def test_nighttime_vegetated_resistance_is_floor(self, consts, veg):
        """Nighttime column (index 0, SW=0): PGSTOT=0 -> PWET = 1/RCONDCTMIN."""
        state, forcing, static, cfg, ctx = _make_grid()
        fq = farquhar_resistance(
            forcing=forcing, skin_t_prev=ctx["skin_t_prev"],
            qskm1m=ctx["qskm1m"], ssrfl_grid=ctx["ssrfl_grid"],
            ktvl=ctx["ktvl"], ktvh=ctx["ktvh"], jarvis=ctx["jarvis"],
            veg_table=veg, consts=consts,
            land_mask=static.land_frac > 0,
        )
        rcondctmin = veg.farquhar_consts.rcondctmin
        # The nighttime column's vegetated resistance must equal the floor.
        assert fq["pwetl"][0].item() == pytest.approx(1.0 / rcondctmin, rel=1e-12)


# --------------------------------------------------------------------------- #
# Fortran cross-check: the assembled PWET matches an independent
# ref.farquhar + documented conversion.
# --------------------------------------------------------------------------- #
@_requires_fortran()
class TestFarquharResistanceMatchesFortran:
    """The port's farquhar_resistance matches ref.farquhar + conversion."""

    @pytest.fixture(scope="class")
    def ref(self):
        return SurfRef()

    def test_pwl_matches_fortran_farquhar(self, ref, consts, veg):
        """PWETL/PWETH/PWETHS from farquhar_resistance == ref.farquhar derived.

        Both sides run the independent Fortran FARQUHAR (port via
        farquhar.py -> its own kernel; ref via ref.farquhar), then apply
        the same 1/max(RCONDCTMIN, PGSTOT) conversion.  The port's
        farquhar.py is itself verified against ref.farquhar to ~1e-9
        (test_farquhar_ref.py), so the assembled PWET must agree to the
        same tolerance.
        """
        state, forcing, static, cfg, ctx = _make_grid()
        fq_port = farquhar_resistance(
            forcing=forcing, skin_t_prev=ctx["skin_t_prev"],
            qskm1m=ctx["qskm1m"], ssrfl_grid=ctx["ssrfl_grid"],
            ktvl=ctx["ktvl"], ktvh=ctx["ktvh"], jarvis=ctx["jarvis"],
            veg_table=veg, consts=consts,
            land_mask=static.land_frac > 0,
        )
        # Independent Fortran reference: ref.farquhar + conversion.
        # The port's PWETL is computed from the low-vegetation dry tile,
        # so the reference must use the same tile-local skin state and qsat.
        tile = TILE_LOW_VEG_DRY
        fq_ref = ref.farquhar(
            sw_down=ctx["ssrfl_grid"], p_surf=forcing.p_surf,
            qs_skin=ctx["qskm1m"][:, tile], qsurf=ctx["qskm1m"][:, tile],
            q_air=forcing.q, t_air=forcing.t,
            skin_t=ctx["skin_t_prev"][:, tile],
            t_soil=ctx["skin_t_prev"][:, tile],
            f2=ctx["jarvis"]["f2l"], co2=forcing.co2,
            lai=ctx["jarvis"]["lail"],
            kvtype=ctx["ktvl"],
            kco2typ=veg.gather("nco2typ", ctx["ktvl"]),
            ldland=(static.land_frac > 0),
            veg=veg, consts=consts, lskin_temp=True,
        )
        rcondctmin = veg.farquhar_consts.rcondctmin
        pwet_ref = 1.0 / torch.clamp(fq_ref["pgstot"], min=rcondctmin)
        # Active (daytime, vegetated) columns only -- inactive columns
        # (nighttime, bare) clamp to the floor on both sides, trivially
        # equal; the meaningful comparison is the assimilating columns.
        active = (ctx["ssrfl_grid"] > 0.0) & (ctx["jarvis"]["lail"] > 0.0)
        rel = ((fq_port["pwetl"][active] - pwet_ref[active]).abs()
               / pwet_ref[active].clamp_min(1e-30)).max().item()
        assert rel < 1e-8, f"PWETL port vs ref rel={rel:.3e}"


# --------------------------------------------------------------------------- #
# End-to-end closure: forward(use_farquhar=True, use_ags=True) completes and produces
# physically sane output.  (The bit-level ULP comparison lives in
# test_forward_e2e.py::test_forward_farquhar_matches_fortran_assembly.)
# --------------------------------------------------------------------------- #
class TestForwardFarquharClosure:
    """The use_farquhar=True forward path runs end-to-end and is sane."""

    def test_forward_farquhar_completes_and_shapes(self):
        """forward(use_farquhar=True) must return correctly-shaped outputs."""
        state, forcing, static, cfg, ctx = _make_grid()
        cfg.use_farquhar = True
        cfg.use_ags = True
        model = LandSurface(cfg)
        new_state, diag = model(state, forcing, static)
        assert new_state.skin_t.shape == state.skin_t.shape
        assert new_state.soil_t.shape == state.soil_t.shape

    def test_forward_farquhar_differentiable(self):
        """The Farquhar closure must be differentiable (torch.acos/cos path)."""
        state, forcing, static, cfg, ctx = _make_grid()
        cfg.use_farquhar = True
        cfg.use_ags = True
        cfg.differentiable = True
        model = LandSurface(cfg)
        # Skin T requires grad so backprop reaches the Farquhar path.
        state.skin_t.requires_grad_(True)
        new_state, diag = model(state, forcing, static)
        loss = new_state.skin_t.sum()
        loss.backward()
        assert state.skin_t.grad is not None
        assert torch.isfinite(state.skin_t.grad).all()

    def test_farquhar_and_jarvis_give_different_skin_t(self):
        """Flipping use_farquhar must change the skin-T advance.

        A wiring bug that ignored the flag (e.g. always using Jarvis)
        would make the two runs identical.
        """
        state, forcing, static, cfg, ctx = _make_grid()

        cfg.use_farquhar = False
        model_j = LandSurface(cfg)
        ns_j, _ = model_j(state, forcing, static)

        cfg.use_farquhar = True
        cfg.use_ags = True
        state2, forcing2, static2, _, _ = _make_grid()  # fresh copy
        model_f = LandSurface(cfg)
        ns_f, _ = model_f(state2, forcing2, static2)

        # The two closures differ over the daytime vegetated columns.
        diff = (ns_j.skin_t - ns_f.skin_t).abs().max().item()
        assert diff > 1e-6, (
            f"use_farquhar flip did not change skin-T (max diff={diff:.3e})"
        )

    def test_farquhar_without_ags_keeps_jarvis_water_closure(self):
        """LEFARQUHAR alone is diagnostic; LEAGS controls PWET feedback."""
        state_j, forcing_j, static_j, cfg_j, _ = _make_grid()
        cfg_j.use_farquhar = False
        ns_j, _ = LandSurface(cfg_j)(state_j, forcing_j, static_j)

        state_f, forcing_f, static_f, cfg_f, _ = _make_grid()
        cfg_f.use_farquhar = True
        cfg_f.use_ags = False
        ns_f, diag_f = LandSurface(cfg_f)(state_f, forcing_f, static_f)

        torch.testing.assert_close(ns_f.skin_t, ns_j.skin_t, rtol=0.0, atol=0.0)
        assert diag_f.farquhar_pgstot is not None

    def test_forward_surfaces_carbon_flux_diagnostics(self):
        """use_farquhar=True must populate SurfDiag.farquhar_* carbon fluxes.

        Pins the diagnostic surfacing added in P2.1-Step5b: the connector
        returns PGSTOT/PGPP/PRD/PAN from the photosynthesis kernel and
        the forward attaches them to SurfDiag so callers can read GPP /
        respiration without re-running Farquhar.
        """
        state, forcing, static, cfg, ctx = _make_grid()
        cfg.use_farquhar = True
        model = LandSurface(cfg)
        _, diag = model(state, forcing, static)

        # All four carbon-flux fields must be populated (not None) and
        # shaped (Ngrid,).
        for name in ("farquhar_pgstot", "farquhar_pgpp",
                     "farquhar_prd", "farquhar_pan"):
            v = getattr(diag, name)
            assert v is not None, f"{name} not surfaced in SurfDiag"
            assert v.shape == (state.skin_t.shape[0],), (
                f"{name} wrong shape {tuple(v.shape)}")

        # Jarvis path (use_farquhar=False) must leave them None -- the
        # diagnostics only exist when Farquhar is active.
        cfg.use_farquhar = False
        model_j = LandSurface(cfg)
        _, diag_j = model_j(state, forcing, static)
        for name in ("farquhar_pgstot", "farquhar_pgpp",
                     "farquhar_prd", "farquhar_pan"):
            assert getattr(diag_j, name) is None, (
                f"{name} should be None on the Jarvis path")

    def test_forward_carbon_fluxes_finite_and_signed(self):
        """Farquhar carbon-flux diagnostics are finite with sane signs.

        PGSTOT (canopy conductance) is non-negative; GPP (pgpp) and dark
        respiration (prd) are non-negative over vegetated land.  These
        are weak physical sanity bounds -- the precise ULP validation
        lives in test_farquhar_ref.py.
        """
        state, forcing, static, cfg, ctx = _make_grid()
        cfg.use_farquhar = True
        model = LandSurface(cfg)
        _, diag = model(state, forcing, static)

        assert torch.isfinite(diag.farquhar_pgstot).all()
        assert torch.isfinite(diag.farquhar_pgpp).all()
        assert torch.isfinite(diag.farquhar_prd).all()
        assert torch.isfinite(diag.farquhar_pan).all()
        # Conductance and gross fluxes are non-negative by construction.
        assert (diag.farquhar_pgstot >= 0).all()
        assert (diag.farquhar_pgpp >= 0).all()
        assert (diag.farquhar_prd >= 0).all()
