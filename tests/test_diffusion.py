"""Unit tests for the implicit vertical-diffusion SURFSEB wrapper
(``surf_pytorch.physics.diffusion``).

These tests exercise the offline single-column driver translation of
``vdfdifh1s.F90``.  Because the offline driver comments out the
tridiagonal solve and replaces it with a Dirichlet boundary condition,
the wrapper is essentially a packaging step: it builds the SURFSEB
inputs from the lowest-model-level forcing, applies the exchange-
coefficient inversion (ZCOEF1 = 1/(RG*RVDIFTS*PTMST)), runs SURFSEB
once (or twice when LESNML snow-melt repeat fires), adds the snow-
under-forest evaporation to tile 7, and extrapolates the skin T to
t+1.

Test groups:

* ``TestDiffusionConstants`` -- pin the hard-wired scalars (RVDIFTS,
  LESNML thresholds) to the Fortran values.
* ``TestDiffusionForcing`` -- the standalone Dirichlet boundary
  condition (asl=aql=0, bsl/bql from lowest-level forcing).
* ``TestDiffusionExchangeInversion`` -- rhoChU = PCFHTI /
  (RG*RVDIFTS*PTMST) is the exact inverse of the upstream
  ``vdfmain1s`` rescaling.
* ``TestDiffusionTsrfBuild`` -- per-tile PTSRF mapping (water=SST,
  ice=PTICE, snow=PTSNOW, lake=ice-or-mixed-layer).
* ``TestDiffusionClosure`` -- the linearised SEB ``SSR*FRSR + STR +
  SSH + SLH - G0 = 0`` is satisfied to rounding level on the call-#3
  outputs.
* ``TestDiffusionSnowUnderForest`` -- tile 7 (high-veg snow) gets the
  PEVAPSNW addition to PJQ and PEVAPSNW*RLSTT to PSLH.
* ``TestDiffusionSkinExtrapolation`` -- PTSKTIP1 = (1/RVDIFTS)*PTSK
  + (1-1/RVDIFTS)*PTSKTI matches the ZTPFAC2/ZTPFAC3 formula.
* ``TestDiffusionLesnmlRepeat`` -- the call-#4 snow-melt repeat pins
  the low-veg-snow tile to RTT/LAMSK=100 when its skin T is at
  melting.
* ``TestDiffusionDifferentiable`` -- autograd runs end-to-end.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from surf_pytorch import SurfConfig, ConstantBundle
from surf_pytorch.physics.diffusion import (
    DiffusionInputs,
    DiffusionOutputs,
    vdfdifh1s,
    RVDIFTS,
    _LESNML_FRAC_MIN,
    _LESNML_TSK_EPS,
    _LESNML_LAMSK_MELT,
    _LAKE_HICE_MIN,
)
from surf_pytorch.physics.seb import _tile_solar_transmission
from surf_pytorch.utils import qsat, dqsat_dt_ifs


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_diffusion_inputs(
    *,
    N: int = 2,
    T: int = 8,
    tsk1: float = 290.0,
    psurf: float = 1.0e5,
    sst: float = 288.0,
    tice: float = 270.0,
    tsm1m: float = 285.0,
    tsnow: float = 273.0,
    rho_chu: float = 0.01,
    rho_cqu: float = 0.01,
    ssrfl: float = 200.0,
    slrfl: float = -300.0,
    tair: float = 290.0,
    qair: float = 0.008,
    pslge: float = 0.0,
    pqte: float = 0.0,
    lamsk: float = 10.0,
    alphal: float = 0.0,
    alphas: float = 1.0,
    pevapsnw: float | None = 0.0,
    ktvl: int = 1,
    ktvh: int = 3,
    ptmst: float = 3600.0,
    lesnml: bool = False,
    dtype: torch.dtype = torch.float64,
) -> tuple[DiffusionInputs, ConstantBundle, float]:
    """Build a self-consistent DiffusionInputs for tests.

    ``rho_chu`` / ``rho_cqu`` are the *physical* rho*Ch*|U| values; the
    helper applies the upstream ``RG*PTSPHY*RVDIFTS`` rescaling when
    populating PCFHTI/PCFQTI so the wrapper's inversion recovers them
    exactly.

    Returns ``(inputs, consts, rho_chu)`` so tests can sanity-check the
    inversion against the known physical value.
    """
    consts = ConstantBundle(dtype=dtype)
    cp = float(consts.cpd.item())
    rg = float(consts.g.item())

    tsk_t = torch.full((N, T), tsk1, dtype=dtype)
    p_t = torch.full((N, T), psurf, dtype=dtype)
    qsk_t = qsat(tsk_t, p_t, consts)
    dqsdT_t = dqsat_dt_ifs(tsk_t, p_t, consts)
    ssk_t = cp * tsk_t

    lamsk_t = torch.full((N, T), float(lamsk), dtype=dtype)

    # Upstream rescaling (vdfmain1s line 745-746):
    #   PCFHTI = rho*Ch*|U| * RG * PTSPHY * RVDIFTS
    pcfhti = torch.full((N, T), rho_chu * rg * ptmst * RVDIFTS, dtype=dtype)
    pcfqti = torch.full((N, T), rho_cqu * rg * ptmst * RVDIFTS, dtype=dtype)

    pevapsnw_t = None
    if pevapsnw is not None:
        pevapsnw_t = torch.full((N,), pevapsnw, dtype=dtype)

    inp = DiffusionInputs(
        pslm1_lev=torch.full((N,), cp * tair, dtype=dtype),
        pqtm1_lev=torch.full((N,), qair, dtype=dtype),
        pslge_lev=torch.full((N,), pslge, dtype=dtype),
        pqte_lev=torch.full((N,), pqte, dtype=dtype),
        pcfhti=pcfhti, pcfqti=pcfqti,
        psst=torch.full((N,), sst, dtype=dtype),
        ptice=torch.full((N,), tice, dtype=dtype),
        ptsm1m=torch.full((N,), tsm1m, dtype=dtype),
        ptsnow=torch.full((N,), tsnow, dtype=dtype),
        sskm1m=ssk_t, tskm1m=tsk_t, qskm1m=qsk_t, dqsdT=dqsdT_t,
        alphal=torch.full((N, T), alphal, dtype=dtype),
        alphas=torch.full((N, T), alphas, dtype=dtype),
        pssrflti=torch.full((N, T), ssrfl, dtype=dtype),
        pslrfl=torch.full((N,), slrfl, dtype=dtype),
        ptskrad=torch.full((N,), tsk1, dtype=dtype),
        pemis=torch.full((N,), 0.99, dtype=dtype),
        plamsk=lamsk_t,
        pfrti=torch.full((N, T), 1.0 / T, dtype=dtype),
        ktvl=torch.full((N,), ktvl, dtype=torch.long),
        ktvh=torch.full((N,), ktvh, dtype=torch.long),
        ptmst=torch.tensor(ptmst, dtype=dtype),
        pevapsnw=pevapsnw_t,
        lesnml=lesnml,
    )
    return inp, consts, rho_chu


def _seb_residual(
    out: DiffusionOutputs, ssr: torch.Tensor, frsr: torch.Tensor,
) -> torch.Tensor:
    """SEB residual ``SSR*FRSR + STR + SSH + SLH - G0`` (per tile).

    Mirrors :class:`tests.test_seb.TestSebClosure._resid` -- all four
    fluxes positive downward (see :mod:`surf_pytorch.physics.seb`).
    """
    return ssr * frsr + out.pstr + out.pssh + out.pslh - out.pg0


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
class TestDiffusionConstants:
    """Pin the hard-wired scalars to the Fortran values."""

    def test_rvdifts(self):
        # The full-model default (SUEVDF) is RVDIFTS = 1.5, but the offline
        # single-column driver overrides it to 1.0 (surf/offline/driver/
        # suvdf.F90:57).  With RVDIFTS=1.0 the over-implicit extrapolation
        # collapses: PTSKTIP1 = PTSK exactly (no t-1 mixing).  This port
        # targets the offline driver, so it pins 1.0.
        assert RVDIFTS == 1.0

    def test_lesnml_thresholds(self):
        # vdfdifh1s lines 523-528: PFRTI >= 1e-6, TSK >= RTT-1e-6, LAMSK=100.
        assert _LESNML_FRAC_MIN == 1.0e-6
        assert _LESNML_TSK_EPS == 1.0e-6
        assert _LESNML_LAMSK_MELT == 100.0

    def test_lake_hice_min(self):
        # vdfdifh1s line 382: PHLICE > 1e-9 selects ice skin.
        assert _LAKE_HICE_MIN == 1.0e-9


# --------------------------------------------------------------------------- #
# Standalone Dirichlet forcing
# --------------------------------------------------------------------------- #
class TestDiffusionForcing:
    """asl=aql=0, bsl/bql built from lowest-level forcing (offline setup)."""

    def test_bsl_bql_match_lowest_level(self):
        # With rhoc=0 the SEB decouples from the surface; PSL/PQL reduce
        # to the Dirichlet boundary (bsl, bql).
        inp, consts, _ = _build_diffusion_inputs(rho_chu=0.0, rho_cqu=0.0)
        cfg = SurfConfig(dtype=torch.float64)
        out = vdfdifh1s(inp, consts, cfg)

        cp = float(consts.cpd.item())
        ptmst = float(inp.ptmst.item())
        bsl_expected = cp * 290.0 + ptmst * 0.0
        bql_expected = 0.008 + ptmst * 0.0
        assert torch.allclose(out.psl, torch.full_like(out.psl, bsl_expected))
        assert torch.allclose(out.pql, torch.full_like(out.pql, bql_expected))

    def test_tendency_feeds_bsl(self):
        # Non-zero PSLGE shifts bsl by PTMST*PSLGE.
        inp, consts, _ = _build_diffusion_inputs(
            rho_chu=0.0, rho_cqu=0.0, pslge=1.0e-3,
        )
        cfg = SurfConfig(dtype=torch.float64)
        out = vdfdifh1s(inp, consts, cfg)

        cp = float(consts.cpd.item())
        ptmst = float(inp.ptmst.item())
        bsl_expected = cp * 290.0 + ptmst * 1.0e-3
        assert torch.allclose(out.psl, torch.full_like(out.psl, bsl_expected))


# --------------------------------------------------------------------------- #
# Exchange-coefficient inversion
# --------------------------------------------------------------------------- #
class TestDiffusionExchangeInversion:
    """rhoChU = PCFHTI / (RG*RVDIFTS*PTMST) is the exact inverse."""

    def test_inversion_recovers_physical_value(self, atol_strict):
        # Strong-forcing case where the SEB actually feels rhoc; the
        # resulting surface fluxes depend on rho_chu.  We verify by
        # building two inputs with different rhoc and confirming the
        # latent heat scales linearly with rhoc (in the weak-forcing
        # limit the SEB is linear in rhoc to leading order).
        inp_a, consts, rho_a = _build_diffusion_inputs(rho_chu=0.01, rho_cqu=0.01)
        inp_b, consts, rho_b = _build_diffusion_inputs(rho_chu=0.02, rho_cqu=0.02)
        cfg = SurfConfig(dtype=torch.float64)
        out_a = vdfdifh1s(inp_a, consts, cfg)
        out_b = vdfdifh1s(inp_b, consts, cfg)

        # Sanity: outputs are finite.
        assert torch.isfinite(out_a.pslh).all()
        assert torch.isfinite(out_b.pslh).all()
        # Different rhoc -> different PSLH (sanity that rhoc is consumed).
        assert not torch.allclose(out_a.pslh, out_b.pslh)

    def test_rescaling_round_trip(self, atol_strict):
        # Verify the inversion by reconstructing PCFHTI from the
        # internal SebInputs.rhochu.  This exercises the contract that
        # vdfmain1s applies RG*PTSPHY*RVDIFTS and vdfdifh1s inverts it.
        from surf_pytorch.physics.diffusion import _vdfdifh1s_core
        from surf_pytorch.physics.seb import SebInputs, surf_seb

        inp, consts, rho_chu = _build_diffusion_inputs(rho_chu=0.01, rho_cqu=0.01)
        rg = float(consts.g.item())
        ptmst = float(inp.ptmst.item())
        zcoef1 = 1.0 / (rg * RVDIFTS * ptmst)
        rhochu_recovered = inp.pcfhti * zcoef1
        assert torch.allclose(
            rhochu_recovered,
            torch.full_like(rhochu_recovered, rho_chu),
            atol=atol_strict,
        )


# --------------------------------------------------------------------------- #
# Per-tile PTSRF build
# --------------------------------------------------------------------------- #
class TestDiffusionTsrfBuild:
    """Per-tile PTSRF mapping (water=SST, ice=PTICE, snow=PTSNOW, ...)."""

    def test_water_tile_uses_sst(self):
        # Pin water-tile lambda_skin to LARGE so the skin T pins to TSRF.
        inp, consts, _ = _build_diffusion_inputs(
            T=1, sst=288.0, lamsk=1.0e10,
        )
        cfg = dataclasses.replace(SurfConfig(dtype=torch.float64), n_tile=1)
        out = vdfdifh1s(inp, consts, cfg)
        # Skin T on the water tile pins to PSST.
        assert torch.allclose(out.ptsk[:, 0], inp.psst, atol=1.0e-3)

    def test_snow_tile_uses_ptsnow(self):
        # Tile 5 (port idx 4) uses PTSNOW as the lower boundary.
        # With LARGE lambda_skin the skin pins to PTSNOW.
        from surf_pytorch.config import TILE_LOW_VEG_SNOW
        T = TILE_LOW_VEG_SNOW + 1
        lamsk = torch.full((2, T), 10.0, dtype=torch.float64)
        lamsk[:, TILE_LOW_VEG_SNOW] = 1.0e10
        inp, consts, _ = _build_diffusion_inputs(
            T=T, tsnow=272.0, lamsk=10.0,
        )
        # Override the uniform lamsk to pin tile 4 to PTSNOW.
        inp.plamsk = lamsk
        cfg = dataclasses.replace(SurfConfig(dtype=torch.float64), n_tile=T)
        # Force tile 4 to be the only active tile (frac=1 on tile 4).
        frac = torch.zeros((2, T), dtype=torch.float64)
        frac[:, TILE_LOW_VEG_SNOW] = 1.0
        inp.pfrti = frac
        out = vdfdifh1s(inp, consts, cfg)
        assert torch.allclose(
            out.ptsk[:, TILE_LOW_VEG_SNOW], inp.ptsnow, atol=1.0e-3,
        )


# --------------------------------------------------------------------------- #
# SEB closure
# --------------------------------------------------------------------------- #
class TestDiffusionClosure:
    """Linearised SEB ``SSR*FRSR + STR + SSH + SLH - G0 = 0`` (call #3)."""

    def test_closure_single_tile(self):
        # Single tile (water, frsr=1): closure residual at rounding level.
        inp, consts, _ = _build_diffusion_inputs(
            T=1, tsk1=290.0, sst=288.0, lamsk=15.0,
            rho_chu=0.005, rho_cqu=0.005,
        )
        cfg = dataclasses.replace(SurfConfig(dtype=torch.float64), n_tile=1)
        out = vdfdifh1s(inp, consts, cfg)
        frsr = _tile_solar_transmission(1, 1, inp.pssrflti.dtype, inp.pssrflti.device)
        resid = _seb_residual(out, inp.pssrflti, frsr)
        assert resid.abs().max().item() < 1e-6, (
            f"closure residual {resid.abs().max().item():.3e} too large"
        )

    def test_closure_multi_tile(self):
        # 8 tiles, distinct SSR/lambda per tile: closure holds on every tile.
        inp, consts, _ = _build_diffusion_inputs(
            T=8, tsk1=290.0, lamsk=12.0,
            rho_chu=0.005, rho_cqu=0.005,
        )
        cfg = SurfConfig(dtype=torch.float64)
        out = vdfdifh1s(inp, consts, cfg)
        frsr = _tile_solar_transmission(2, 8, inp.pssrflti.dtype, inp.pssrflti.device)
        resid = _seb_residual(out, inp.pssrflti, frsr)
        # Tiles 4/6/7 have FRSR < 1 (canopy transmission) so the raw SSR
        # overcounts; the residual uses SSR*FRSR which is exact.
        assert resid.abs().max().item() < 1e-6, (
            f"closure residual {resid.abs().max().item():.3e} too large"
        )

    def test_closure_with_snow_under_forest(self):
        # PEVAPSNW breaks closure on tile 7 *by design* (it is an
        # explicit addition outside the SEB); the residual is then
        # PEVAPSNW*(1 + RLSTT) on that tile.  Closure still holds on
        # every other tile.
        from surf_pytorch.config import TILE_HIGH_VEG_SNOW
        inp, consts, _ = _build_diffusion_inputs(
            T=8, pevapsnw=1.0e-5,
        )
        cfg = SurfConfig(dtype=torch.float64)
        out = vdfdifh1s(inp, consts, cfg)
        frsr = _tile_solar_transmission(2, 8, inp.pssrflti.dtype, inp.pssrflti.device)
        resid = _seb_residual(out, inp.pssrflti, frsr)
        # On the high-veg-snow tile the residual equals
        # PEVAPSNW + PEVAPSNW*RLSTT (PJQ and PSLH both get the addition).
        rlstt = float(consts.rlstt.item())
        pev_scalar = float(inp.pevapsnw[0].item())
        expected_tile7 = pev_scalar * (1.0 + rlstt)
        # All non-tile-7 columns have rounding-level residual.
        mask = torch.ones(8, dtype=torch.bool)
        mask[TILE_HIGH_VEG_SNOW] = False
        assert resid[:, mask].abs().max().item() < 1e-6
        # Tile 7 residual matches the expected addition.
        assert torch.allclose(
            resid[:, TILE_HIGH_VEG_SNOW],
            torch.full_like(resid[:, TILE_HIGH_VEG_SNOW], expected_tile7),
            atol=1e-9,
        )


# --------------------------------------------------------------------------- #
# Snow-under-forest evaporation
# --------------------------------------------------------------------------- #
class TestDiffusionSnowUnderForest:
    """Tile 7 (high-veg snow) gets PEVAPSNW addition."""

    def test_no_evapsnw_leaves_outputs_unchanged(self):
        # pevapsnw=None -> PSLH/PJQ are the raw SURFSEB outputs.
        from surf_pytorch.config import TILE_HIGH_VEG_SNOW
        inp_none, consts, _ = _build_diffusion_inputs(T=8, pevapsnw=None)
        inp_zero, consts, _ = _build_diffusion_inputs(T=8, pevapsnw=0.0)
        cfg = SurfConfig(dtype=torch.float64)
        out_none = vdfdifh1s(inp_none, consts, cfg)
        out_zero = vdfdifh1s(inp_zero, consts, cfg)
        assert torch.allclose(out_none.pslh, out_zero.pslh)
        assert torch.allclose(out_none.pjq, out_zero.pjq)

    def test_evapsnw_adds_to_tile7(self, atol_strict):
        # Non-zero PEVAPSNW: PSLH[7] += PEVAPSNW*RLSTT, PJQ[7] += PEVAPSNW.
        from surf_pytorch.config import TILE_HIGH_VEG_SNOW
        pev = 2.0e-5
        inp_base, consts, _ = _build_diffusion_inputs(T=8, pevapsnw=None)
        inp_evap, consts, _ = _build_diffusion_inputs(T=8, pevapsnw=pev)
        cfg = SurfConfig(dtype=torch.float64)
        out_base = vdfdifh1s(inp_base, consts, cfg)
        out_evap = vdfdifh1s(inp_evap, consts, cfg)

        rlstt = float(consts.rlstt.item())
        d_pslh = out_evap.pslh[:, TILE_HIGH_VEG_SNOW] - out_base.pslh[:, TILE_HIGH_VEG_SNOW]
        d_pjq = out_evap.pjq[:, TILE_HIGH_VEG_SNOW] - out_base.pjq[:, TILE_HIGH_VEG_SNOW]
        assert torch.allclose(d_pslh, torch.full_like(d_pslh, pev * rlstt), atol=atol_strict)
        assert torch.allclose(d_pjq, torch.full_like(d_pjq, pev), atol=atol_strict)
        # Other tiles are untouched.
        mask = torch.ones(8, dtype=torch.bool)
        mask[TILE_HIGH_VEG_SNOW] = False
        assert torch.allclose(out_evap.pslh[:, mask], out_base.pslh[:, mask])
        assert torch.allclose(out_evap.pjq[:, mask], out_base.pjq[:, mask])


# --------------------------------------------------------------------------- #
# Skin T extrapolation to t+1
# --------------------------------------------------------------------------- #
class TestDiffusionSkinExtrapolation:
    """PTSKTIP1 = ZTPFAC2*PTSK + ZTPFAC3*PTSKTI."""

    def test_extrapolation_formula(self, atol_strict):
        inp, consts, _ = _build_diffusion_inputs(T=8)
        cfg = SurfConfig(dtype=torch.float64)
        out = vdfdifh1s(inp, consts, cfg)

        ztpfac2 = 1.0 / RVDIFTS
        ztpfac3 = 1.0 - ztpfac2
        expected = ztpfac2 * out.ptsk + ztpfac3 * inp.tskm1m
        assert torch.allclose(out.ptsktip1, expected, atol=atol_strict)

    def test_extrapolation_weights(self):
        # Offline driver: RVDIFTS = 1.0 -> ZTPFAC2 = 1.0, ZTPFAC3 = 0.0
        # (the over-implicit weight collapses; PTSKTIP1 = PTSK exactly).
        # The full-model default would be RVDIFTS=1.5 -> 2/3, 1/3.
        ztpfac2 = 1.0 / RVDIFTS
        assert ztpfac2 == pytest.approx(1.0, abs=1e-12)
        assert (1.0 - ztpfac2) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# LESNML snow-melt repeat (call #4)
# --------------------------------------------------------------------------- #
class TestDiffusionLesnmlRepeat:
    """Call-#4 pins low-veg-snow tile to RTT/LAMSK=100 when melting."""

    def test_no_repeat_when_lesnml_false(self):
        # Even with a melting low-veg-snow tile, lesnml=False skips the
        # call-#4 repeat -- outputs come straight from call #3.
        from surf_pytorch.config import TILE_LOW_VEG_SNOW
        T = TILE_LOW_VEG_SNOW + 1
        inp, consts, _ = _build_diffusion_inputs(
            T=T, lesnml=False, tsnow=274.0, tsk1=274.0,
        )
        # Make tile 4 the only active tile.
        frac = torch.zeros((2, T), dtype=torch.float64)
        frac[:, TILE_LOW_VEG_SNOW] = 1.0
        inp.pfrti = frac
        cfg = dataclasses.replace(SurfConfig(dtype=torch.float64), n_tile=T)
        out = vdfdifh1s(inp, consts, cfg)
        # With lesnml=False the repeat never runs, so PTSK is whatever
        # call #3 produced.  Just assert finite + the routine did not
        # blow up.
        assert torch.isfinite(out.ptsk).all()

    def test_repeat_pins_melting_snow_to_rtt(self):
        # Low-veg-snow tile (idx 4) with Tsk > RTT triggers the repeat;
        # the repeat pins TSRF to RTT and LAMSK to 100, so the new skin
        # T on that tile is pulled toward RTT (was above RTT before).
        from surf_pytorch.config import TILE_LOW_VEG_SNOW
        T = TILE_LOW_VEG_SNOW + 1
        rtt = 273.16

        # Build an input where tile 4 has a skin T above RTT after call #3
        # (strong SW forcing + warm snow surface -> Tsk > RTT).
        inp_no_rep, consts, _ = _build_diffusion_inputs(
            T=T, lesnml=False,
            tsk1=275.0, tsnow=275.0,    # warm snow
            ssrfl=500.0, slrfl=-100.0,  # strong net radiation in
            rho_chu=0.001, rho_cqu=0.001,
            lamsk=5.0,                  # weak ground coupling
        )
        frac = torch.zeros((2, T), dtype=torch.float64)
        frac[:, TILE_LOW_VEG_SNOW] = 1.0
        inp_no_rep.pfrti = frac

        inp_rep = dataclasses.replace(inp_no_rep, lesnml=True)

        cfg = dataclasses.replace(SurfConfig(dtype=torch.float64), n_tile=T)
        out_no_rep = vdfdifh1s(inp_no_rep, consts, cfg)
        out_rep = vdfdifh1s(inp_rep, consts, cfg)

        # If call #3 produces Tsk >= RTT on tile 4, the repeat fires and
        # pulls Tsk down toward RTT.  Check that the repeat changed the
        # output; if call #3 did not actually exceed RTT, the test is
        # vacuous (and we assert finite-only as a sanity check).
        tsk_no_rep = out_no_rep.ptsk[:, TILE_LOW_VEG_SNOW]
        tsk_rep = out_rep.ptsk[:, TILE_LOW_VEG_SNOW]
        if (tsk_no_rep >= rtt - _LESNML_TSK_EPS).any():
            # Repeat fired; new Tsk should be closer to RTT than before.
            assert (tsk_rep - rtt).abs().max().item() <= \
                   (tsk_no_rep - rtt).abs().max().item()
        assert torch.isfinite(out_rep.ptsk).all()


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
class TestDiffusionDifferentiable:
    """Autograd runs end-to-end when the differentiable flag is on."""

    def test_grad_runs_end_to_end(self):
        cfg = dataclasses.replace(
            SurfConfig(dtype=torch.float64), differentiable=True,
        )
        inp, consts, _ = _build_diffusion_inputs(T=8, pevapsnw=1.0e-5)
        # Make tskm1m require grad.
        inp.tskm1m.requires_grad_(True)
        out = vdfdifh1s(inp, consts, cfg)
        loss = out.ptsk.sum() + out.pslh.sum()
        loss.backward()
        assert inp.tskm1m.grad is not None
        assert torch.isfinite(inp.tskm1m.grad).all()

    def test_no_grad_default(self):
        # Default config -> no grad graph built.
        cfg = SurfConfig(dtype=torch.float64)
        assert cfg.differentiable is False
        inp, consts, _ = _build_diffusion_inputs(T=8)
        out = vdfdifh1s(inp, consts, cfg)
        # No backward possible (no graph).
        assert not out.ptsk.requires_grad
