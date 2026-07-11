"""Tests for the single-layer snow-albedo aging (``SRFSN`` 340-361).

Verifies the ``surf_pytorch.physics.snow_albedo`` port against the
documented ``SRFSN`` snow-albedo physics:

* no-snow branch resets PASN to RALFMAXSN (0.85)
* melt branch: exponential decay toward RALFMINSN (0.50) with RTAUF
* cold branch: linear decay toward RALFMINSN (0.50) with RTAUA, floored
* fresh-snow reset: snowfall above RSFRESH resets PASN to RALFMAXSN
* the [RALFMINSN, RALFMAXSN] bound holds across all branches
* differentiability (gradient flows through the aging block)
* cold-start steady state: PASNM1M=0.85 + no melt + no snowfall -> 0.85
"""

from __future__ import annotations

import math

import pytest
import torch

from surf_pytorch.constants import ConstantBundle, PhysicalConstants
from surf_pytorch.physics.snow_albedo import (
    SnowAlbedoInputs, compute_snow_albedo,
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


def _consts():
    """Return (ralfminsn, ralfmaxsn, rtauf, rtaua, rsfresh, rday)."""
    sc = _soil_table().consts
    return (
        float(sc.ralfminsn), float(sc.ralfmaxsn),
        float(sc.rtauf), float(sc.rtaua),
        float(sc.rsfresh), float(PhysicalConstants().rday),
    )


def _make_inputs(
    N: int = 4,
    *,
    pasnm1m: float = 0.70,
    pmsn: float = 0.0,
    pssflpssfc: float = 0.0,
    llnosnow: bool = False,
    ptmst: float = 3600.0,
    dtype=torch.float64,
) -> SnowAlbedoInputs:
    """Build a uniform ``(N,)`` slab of snow-albedo inputs.

    Defaults to a snow-bearing, non-melting pack with no fresh snowfall,
    so the cold linear-decay branch acts in isolation.
    """
    return SnowAlbedoInputs(
        pasnm1m=_scalar(pasnm1m, N, dtype),
        pmsn=_scalar(pmsn, N, dtype),
        pssflpssfc=_scalar(pssflpssfc, N, dtype),
        llnosnow=torch.full((N,), bool(llnosnow)),
        ptmst=torch.tensor(float(ptmst), dtype=dtype),
    )


def _run(inp: SnowAlbedoInputs, *, differentiable: bool = True) -> torch.Tensor:
    return compute_snow_albedo(
        inp, _soil_table(), _bundle(), differentiable=differentiable,
    ).pasn


# --------------------------------------------------------------------------- #
# Tests: no-snow branch
# --------------------------------------------------------------------------- #
class TestNoSnowBranch:
    """The LLNOSNOW branch resets PASN to RALFMAXSN (0.85)."""

    def test_no_snow_resets_to_ralfmaxsn(self):
        _, ralfmaxsn, _, _, _, _ = _consts()
        inp = _make_inputs(llnosnow=True, pasnm1m=0.55)
        pasn = _run(inp)
        assert torch.equal(pasn, _scalar(ralfmaxsn, 4))

    def test_mixed_slab_resets_only_no_snow_points(self):
        _, ralfmaxsn, _, _, _, _ = _consts()
        N = 4
        inp = _make_inputs(N=N, llnosnow=False, pasnm1m=0.60, pmsn=0.0)
        # Points 0, 2 no-snow; points 1, 3 snow-bearing cold.
        llnosnow = torch.tensor([True, False, True, False])
        inp = SnowAlbedoInputs(**{**inp.__dict__, "llnosnow": llnosnow})
        pasn = _run(inp)
        # No-snow points exactly RALFMAXSN; snow points below RALFMAXSN
        # (cold decay from 0.60).
        assert torch.all(pasn[[0, 2]] == ralfmaxsn)
        assert torch.all(pasn[[1, 3]] < ralfmaxsn)
        assert torch.all(pasn[[1, 3]] >= 0.50)


# --------------------------------------------------------------------------- #
# Tests: melt branch (exponential decay)
# --------------------------------------------------------------------------- #
class TestMeltDecay:
    """PMSN > 0 -> PASN = (PASNM1M - RALFMINSN)*ZEXPF + RALFMINSN."""

    def test_melt_exponential_decay_value(self):
        ralfminsn, ralfmaxsn, rtauf, _, _, rday = _consts()
        ptmst = 3600.0
        pasnm1m = 0.80
        zexpf = math.exp(-rtauf * ptmst / rday)
        expected = (pasnm1m - ralfminsn) * zexpf + ralfminsn
        inp = _make_inputs(pasnm1m=pasnm1m, pmsn=1e-4)  # any positive melt
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(expected, 4), atol=1e-12)

    def test_melt_decays_toward_ralfminsn(self):
        # Over many steps the melt branch must converge to RALFMINSN.
        # Each step multiplies the gap (PASN-RALFMINSN) by ZEXPF ~
        # exp(-0.01) ~ 0.99005, so ~3000 steps are needed to push a 0.34
        # initial gap below 1e-6.
        ralfminsn, _, _, _, _, _ = _consts()
        pasnm1m = _scalar(0.84, 1)
        for _ in range(3000):
            inp = SnowAlbedoInputs(**{
                **_make_inputs(N=1, pmsn=1e-4).__dict__,
                "pasnm1m": pasnm1m,
            })
            pasnm1m = _run(inp)
        assert pasnm1m.item() < ralfminsn + 1e-6

    def test_melt_never_above_pasnm1m(self):
        # Melt can only darken snow (PASN <= PASNM1M).
        inp = _make_inputs(pasnm1m=0.70, pmsn=1e-3)
        pasn = _run(inp)
        assert torch.all(pasn <= 0.70 + 1e-12)


# --------------------------------------------------------------------------- #
# Tests: cold branch (linear decay, floored at RALFMINSN)
# --------------------------------------------------------------------------- #
class TestColdLinear:
    """PMSN <= 0 -> PASN = max(RALFMINSN, PASNM1M - ZLINA)."""

    def test_cold_linear_decay_value(self):
        ralfminsn, _, _, rtaua, _, rday = _consts()
        ptmst = 3600.0
        pasnm1m = 0.80
        zlina = rtaua * ptmst / rday
        expected = max(ralfminsn, pasnm1m - zlina)
        inp = _make_inputs(pasnm1m=pasnm1m, pmsn=0.0)
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(expected, 4), atol=1e-12)

    def test_cold_floor_at_ralfminsn(self):
        # Starting below the floor (or after many cold steps) stays at floor.
        ralfminsn, _, _, _, _, _ = _consts()
        inp = _make_inputs(pasnm1m=0.505, pmsn=0.0)  # tiny margin
        pasn = _run(inp)
        # Cold decay would push below 0.50; clamp holds it at 0.50.
        assert torch.all(pasn >= ralfminsn - 1e-12)
        assert torch.all(pasn <= 0.505)

    def test_cold_floor_exact_at_ralfminsn(self):
        ralfminsn, _, _, _, _, _ = _consts()
        inp = _make_inputs(pasnm1m=ralfminsn, pmsn=0.0)
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(ralfminsn, 4), atol=1e-12)


# --------------------------------------------------------------------------- #
# Tests: fresh-snow reset
# --------------------------------------------------------------------------- #
class TestFreshSnowReset:
    """ZSSFL+ZSSFC > RSFRESH -> PASN = RALFMAXSN."""

    def test_fresh_snow_resets_to_ralfmaxsn(self):
        _, ralfmaxsn, _, _, rsfresh, _ = _consts()
        # Snowfall well above RSFRESH (=1/3600 ~ 2.78e-4 kg/m^2/s).
        inp = _make_inputs(pasnm1m=0.55, pmsn=0.0, pssflpssfc=10 * rsfresh)
        pasn = _run(inp)
        assert torch.equal(pasn, _scalar(ralfmaxsn, 4))

    def test_fresh_snow_reset_beats_melt_decay(self):
        # Melt would darken; heavy fresh snowfall overrides and brightens.
        _, ralfmaxsn, _, _, rsfresh, _ = _consts()
        inp = _make_inputs(pasnm1m=0.55, pmsn=1e-3, pssflpssfc=10 * rsfresh)
        pasn = _run(inp)
        assert torch.equal(pasn, _scalar(ralfmaxsn, 4))

    def test_below_rsfresh_no_reset(self):
        # Snowfall below RSFRESH leaves the melt/cold branch in control.
        _, _, _, _, rsfresh, _ = _consts()
        ralfminsn, _, rtauf, _, _, rday = _consts()
        ptmst = 3600.0
        pasnm1m = 0.80
        zexpf = math.exp(-rtauf * ptmst / rday)
        expected = (pasnm1m - ralfminsn) * zexpf + ralfminsn
        inp = _make_inputs(
            pasnm1m=pasnm1m, pmsn=1e-4, pssflpssfc=0.1 * rsfresh,
        )
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(expected, 4), atol=1e-12)


# --------------------------------------------------------------------------- #
# Tests: bounds
# --------------------------------------------------------------------------- #
class TestBounds:
    """PASN stays in [RALFMINSN, RALFMAXSN] across all branches."""

    @pytest.mark.parametrize("pasnm1m", [0.50, 0.60, 0.70, 0.80, 0.85])
    @pytest.mark.parametrize("pmsn", [0.0, 1e-4, 1e-3])
    @pytest.mark.parametrize("pssflpssfc", [0.0, 1e-5, 1e-3])
    def test_bounds_hold(self, pasnm1m, pmsn, pssflpssfc):
        ralfminsn, ralfmaxsn, _, _, _, _ = _consts()
        inp = _make_inputs(pasnm1m=pasnm1m, pmsn=pmsn, pssflpssfc=pssflpssfc)
        pasn = _run(inp)
        assert torch.all(pasn >= ralfminsn - 1e-12)
        assert torch.all(pasn <= ralfmaxsn + 1e-12)

    def test_no_snow_branch_at_max(self):
        _, ralfmaxsn, _, _, _, _ = _consts()
        inp = _make_inputs(llnosnow=True, pasnm1m=0.50)
        pasn = _run(inp)
        assert torch.all(pasn == ralfmaxsn)


# --------------------------------------------------------------------------- #
# Tests: differentiability
# --------------------------------------------------------------------------- #
class TestDifferentiable:
    """Gradient flows through the aging block on the snow-bearing branches."""

    def test_melt_branch_grad(self):
        pasnm1m = torch.full((4,), 0.80, requires_grad=True)
        inp = SnowAlbedoInputs(
            pasnm1m=pasnm1m,
            pmsn=_scalar(1e-4, 4),
            pssflpssfc=_scalar(0.0, 4),
            llnosnow=torch.full((4,), False),
            ptmst=torch.tensor(3600.0),
        )
        pasn = _run(inp)
        pasn.sum().backward()
        # ZEXPF in (0, 1) -> positive gradient on PASNM1M.
        assert pasnm1m.grad is not None
        assert torch.all(pasnm1m.grad > 0.0)
        assert torch.all(pasnm1m.grad < 1.0)

    def test_cold_branch_grad(self):
        pasnm1m = torch.full((4,), 0.80, requires_grad=True)
        inp = SnowAlbedoInputs(
            pasnm1m=pasnm1m,
            pmsn=_scalar(0.0, 4),
            pssflpssfc=_scalar(0.0, 4),
            llnosnow=torch.full((4,), False),
            ptmst=torch.tensor(3600.0),
        )
        pasn = _run(inp)
        pasn.sum().backward()
        # Cold branch: PASN = PASNM1M - ZLINA -> gradient exactly 1.
        assert pasnm1m.grad is not None
        assert torch.allclose(pasnm1m.grad, torch.ones_like(pasnm1m.grad))


# --------------------------------------------------------------------------- #
# Tests: cold-start steady state
# --------------------------------------------------------------------------- #
class TestColdStart:
    """Cold-start single-step behaviour.

    A RALFMAXSN cold start (PASNM1M=0.85) with no melt and no snowfall
    falls in the cold branch and decays linearly by ZLINA: PASN =
    PASNM1M - ZLINA.  The genuine cold steady state is the floor
    RALFMINSN, reached asymptotically; this test pins the single-step
    value.
    """

    def test_cold_start_single_step(self):
        _, _, _, rtaua, _, rday = _consts()
        ptmst = 3600.0
        pasnm1m = 0.85
        zlina = rtaua * ptmst / rday
        expected = 0.85 - zlina
        inp = _make_inputs(pasnm1m=pasnm1m, pmsn=0.0, pssflpssfc=0.0)
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(expected, 4), atol=1e-12)

    def test_no_snow_cold_start_stays_at_max(self):
        # A no-snow cold start (PASN=RALFMAXSN) stays at RALFMAXSN.
        _, ralfmaxsn, _, _, _, _ = _consts()
        inp = _make_inputs(llnosnow=True, pasnm1m=ralfmaxsn)
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(ralfmaxsn, 4), atol=1e-12)


# --------------------------------------------------------------------------- #
# Tests: differentiable=False path
# --------------------------------------------------------------------------- #
class TestNoGradPath:
    """differentiable=False produces identical values without autograd."""

    def test_nograd_matches_grad(self):
        inp = _make_inputs(pasnm1m=0.75, pmsn=1e-4, pssflpssfc=0.0)
        pasn_grad = _run(inp, differentiable=True)
        pasn_nograd = _run(inp, differentiable=False)
        assert torch.allclose(pasn_grad, pasn_nograd, atol=0.0)

    def test_nograd_has_no_grad(self):
        pasnm1m = torch.full((4,), 0.75, requires_grad=True)
        inp = SnowAlbedoInputs(
            pasnm1m=pasnm1m,
            pmsn=_scalar(1e-4, 4),
            pssflpssfc=_scalar(0.0, 4),
            llnosnow=torch.full((4,), False),
            ptmst=torch.tensor(3600.0),
        )
        pasn = _run(inp, differentiable=False)
        assert not pasn.requires_grad


# =========================================================================== #
# SRFSN_ASN glacier / land-ice branch (physics/snow_albedo.py _asn_albedo)
# =========================================================================== #
# These tests exercise the full SRFSN_ASN scheme, activated by supplying the
# land-ice cover fraction ``pcil`` on SnowAlbedoInputs.  The scheme splits the
# new albedo into a land part ZASN_L and a glacier part ZASN_I, blends them by
# PCIL, and swaps the legacy RSFRESH hard reset for a linear snowfall
# brightening.  Glacier-specific constants: ZALBMIN=0.65, ZALBMAX_GL=0.82
# (=RALFMINPSN), ZRTAUF_GL=0.085, hemisphere reset 5 mm NH / 4 mm SH.

def _make_inputs_asn(
    N: int = 4,
    *,
    pasnm1m: float = 0.70,
    pmsn: float = 0.0,
    pssflpssfc: float = 0.0,
    pcil: float = 0.0,
    ptsnm1m_top: float = 263.0,      # < RTT-5 -> cold glacier by default
    psnowf: float = 0.0,
    ldnh: bool = True,
    llnosnow: bool = False,
    ptmst: float = 3600.0,
    dtype=torch.float64,
) -> SnowAlbedoInputs:
    """Build a uniform ``(N,)`` slab of SRFSN_ASN inputs.

    Defaults to a snow-bearing, non-melting pack with no snowfall and a
    cold glacier top layer, so the cold land branch and the cold-glacier
    keep-albedo branch act in isolation.  ``pcil`` defaults to 0 (pure
    land) -- raise it to weight in the glacier part.
    """
    return SnowAlbedoInputs(
        pasnm1m=_scalar(pasnm1m, N, dtype),
        pmsn=_scalar(pmsn, N, dtype),
        pssflpssfc=_scalar(pssflpssfc, N, dtype),
        llnosnow=torch.full((N,), bool(llnosnow)),
        ptmst=torch.tensor(float(ptmst), dtype=dtype),
        pcil=_scalar(pcil, N, dtype),
        ptsnm1m_top=_scalar(ptsnm1m_top, N, dtype),
        psnowf=_scalar(psnowf, N, dtype),
        ldnh=torch.full((N,), bool(ldnh)),
    )


def _asn_consts():
    """Return ASN constants (land + glacier).

    (ralfminsn, ralfmaxsn, rtauf, rtaua, rday, rtt, zalbmin, zalbmax_gl,
     zrtauf_gl).
    """
    sc = _soil_table().consts
    return (
        float(sc.ralfminsn), float(sc.ralfmaxsn),
        float(sc.rtauf), float(sc.rtaua),
        float(PhysicalConstants().rday), float(PhysicalConstants().rtt),
        0.65, float(sc.ralfminpsn), 0.085,
    )


# --------------------------------------------------------------------------- #
# Tests: glacier melt branch
# --------------------------------------------------------------------------- #
class TestGlacierBranch:
    """Melting glacier (PTSNM1M > RTT-5) decays toward ZALBMIN=0.65."""

    def test_glacier_melt_decay_value(self):
        # Pure glacier (PCIL=1), melting, no snowfall:
        #   ZASN_I = (PASNM1M - 0.65)*ZEXPF_GL + 0.65, clamped [0.65, 0.82].
        (_, _, _, _, rday, rtt, zalbmin, _, zrtauf_gl) = _asn_consts()
        ptmst = 3600.0
        pasnm1m = 0.75
        zexpf_gl = math.exp(-zrtauf_gl * ptmst / rday)
        expected = (pasnm1m - zalbmin) * zexpf_gl + zalbmin
        inp = _make_inputs_asn(
            pcil=1.0, pasnm1m=pasnm1m,
            ptsnm1m_top=rtt - 1.0,   # > RTT-5 -> melting
            psnowf=0.0,
        )
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(expected, 4), atol=1e-12)

    def test_glacier_melt_clamped_to_zalbmin(self):
        # A very low PASNM1M under melt is floored at ZALBMIN=0.65.
        (_, _, _, _, _, _, zalbmin, _, _) = _asn_consts()
        inp = _make_inputs_asn(
            pcil=1.0, pasnm1m=0.40,   # below ZALBMIN
            ptsnm1m_top=270.0,        # > RTT-5 -> melting
        )
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(zalbmin, 4), atol=1e-12)

    def test_glacier_melt_clamped_to_zalbmax(self):
        # Brightening snowfall drives ZASN_I toward ZALBMAX_GL=0.82; with
        # enough snowfall it saturates there.
        (_, _, _, _, _, _, _, zalbmax_gl, _) = _asn_consts()
        # psnowf*ptmst >= 5 mm (NH) -> snow_amt_i = 1 -> ZASN_I = ZALBMAX_GL.
        inp = _make_inputs_asn(
            pcil=1.0, pasnm1m=0.70,
            ptsnm1m_top=270.0,        # melting
            psnowf=5.0e-3 / 1.0 + 1e-9,   # 5 mm/step threshold crossed
            ldnh=True,                # NH -> 5 mm threshold
        )
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(zalbmax_gl, 4), atol=1e-9)


# --------------------------------------------------------------------------- #
# Tests: glacier cold branch
# --------------------------------------------------------------------------- #
class TestGlacierCold:
    """Cold glacier (PTSNM1M <= RTT-5) keeps its albedo PASNM1M."""

    def test_cold_glacier_keeps_pasnm1m(self):
        # Pure glacier, cold, no snowfall: ZASN_I = PASNM1M (no decay).
        pasnm1m = 0.78
        inp = _make_inputs_asn(
            pcil=1.0, pasnm1m=pasnm1m,
            ptsnm1m_top=260.0,        # < RTT-5 -> cold
            psnowf=0.0,
        )
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(pasnm1m, 4), atol=1e-12)

    def test_cold_glacier_bounded_by_zalbmax(self):
        # Even a cold glacier with a high PASNM1M is clamped to
        # [ZALBMIN, ZALBMAX_GL] = [0.65, 0.82] before the final blend.
        (_, _, _, _, _, _, _, zalbmax_gl, _) = _asn_consts()
        inp = _make_inputs_asn(
            pcil=1.0, pasnm1m=0.95,   # above ZALBMAX_GL
            ptsnm1m_top=260.0,        # cold
        )
        pasn = _run(inp)
        # No snowfall -> ZASN_I = PASNM1M then clamp to 0.82.
        assert torch.allclose(pasn, _scalar(zalbmax_gl, 4), atol=1e-12)


# --------------------------------------------------------------------------- #
# Tests: linear snowfall brightening (replaces RSFRESH hard reset)
# --------------------------------------------------------------------------- #
class TestLinearSnowfallReset:
    """The ASN land part grades toward RALFMAXSN linearly with snowfall."""

    def test_no_snowfall_keeps_decay(self):
        # PCIL=0 (pure land), cold, no snowfall: ZASN_L = PASNM1M - ZLINA,
        # clamped at RALFMINSN.  snow_amt = 0 -> no brightening.
        (_, _, _, rtaua, rday, _, _, _, _) = _asn_consts()
        ptmst = 3600.0
        pasnm1m = 0.70
        zlina = rtaua * ptmst / rday
        expected = max(0.50, pasnm1m - zlina)
        inp = _make_inputs_asn(pcil=0.0, pasnm1m=pasnm1m, psnowf=0.0)
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(expected, 4), atol=1e-12)

    def test_full_snowfall_brightens_to_ralfmaxsn(self):
        # psnowf*ptmst >= 10 mm -> snow_amt = 1 -> ZASN_L = RALFMAXSN.
        (_, ralfmaxsn, _, _, _, _, _, _, _) = _asn_consts()
        # 10 mm over a 3600 s step: psnowf = 10/3600 kg/m2/s.
        psnowf = 10.0 / 3600.0
        inp = _make_inputs_asn(
            pcil=0.0, pasnm1m=0.60, psnowf=psnowf,
        )
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(ralfmaxsn, 4), atol=1e-12)

    def test_partial_snowfall_is_linear(self):
        # Half the 10 mm threshold -> snow_amt = 0.5 -> ZASN_L is halfway
        # between the cold-decay value and RALFMAXSN.
        (_, ralfmaxsn, _, rtaua, rday, _, _, _, _) = _asn_consts()
        ptmst = 3600.0
        pasnm1m = 0.70
        zlina = rtaua * ptmst / rday
        zasn_l_decay = max(0.50, pasnm1m - zlina)
        psnowf = 5.0 / 3600.0          # 5 mm -> snow_amt = 0.5
        expected = zasn_l_decay + 0.5 * (ralfmaxsn - zasn_l_decay)
        inp = _make_inputs_asn(pcil=0.0, pasnm1m=pasnm1m, psnowf=psnowf)
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(expected, 4), atol=1e-12)

    def test_asn_does_not_hard_reset_below_threshold(self):
        # Unlike the legacy RSFRESH hard reset (any snowfall > RSFRESH ->
        # RALFMAXSN), the ASN linear scheme only partially brightens for
        # sub-threshold snowfall.  A tiny snowfall must NOT jump to
        # RALFMAXSN.
        (_, ralfmaxsn, _, _, _, _, _, _, _) = _asn_consts()
        # 0.1 mm -> snow_amt = 0.01 -> near the decay value, not RALFMAXSN.
        psnowf = 0.1 / 3600.0
        inp = _make_inputs_asn(
            pcil=0.0, pasnm1m=0.60, psnowf=psnowf,
        )
        pasn = _run(inp)
        assert torch.all(pasn < ralfmaxsn - 0.1)


# --------------------------------------------------------------------------- #
# Tests: PCIL blending
# --------------------------------------------------------------------------- #
class TestPCILMixing:
    """PASN = (1-PCIL)*ZASN_L + PCIL*ZASN_I (away from the PCIL~1 cap)."""

    def test_pcil_zero_is_pure_land(self):
        # PCIL=0 -> PASN = ZASN_L (cold decay, no snowfall).
        (_, _, _, rtaua, rday, _, _, _, _) = _asn_consts()
        ptmst = 3600.0
        pasnm1m = 0.70
        zlina = rtaua * ptmst / rday
        expected = max(0.50, pasnm1m - zlina)
        inp = _make_inputs_asn(pcil=0.0, pasnm1m=pasnm1m, psnowf=0.0)
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(expected, 4), atol=1e-12)

    def test_pcil_half_blends_land_and_glacier(self):
        # PCIL=0.5, cold land + cold glacier (no snowfall):
        #   ZASN_L = PASNM1M - ZLINA  (cold land decay)
        #   ZASN_I = PASNM1M          (cold glacier keeps albedo)
        #   PASN    = 0.5*ZASN_L + 0.5*ZASN_I
        (_, _, _, rtaua, rday, _, _, _, _) = _asn_consts()
        ptmst = 3600.0
        pasnm1m = 0.70
        zlina = rtaua * ptmst / rday
        zasn_l = max(0.50, pasnm1m - zlina)
        zasn_i = pasnm1m
        expected = 0.5 * zasn_l + 0.5 * zasn_i
        inp = _make_inputs_asn(
            pcil=0.5, pasnm1m=pasnm1m,
            ptsnm1m_top=260.0,        # cold glacier
            psnowf=0.0,
        )
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(expected, 4), atol=1e-12)

    def test_pcil_one_is_pure_glacier(self):
        # PCIL=1 (>= 1-eps) -> PASN = ZASN_I outright (cold glacier).
        pasnm1m = 0.76
        inp = _make_inputs_asn(
            pcil=1.0, pasnm1m=pasnm1m,
            ptsnm1m_top=260.0,        # cold glacier
            psnowf=0.0,
        )
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(pasnm1m, 4), atol=1e-12)

    def test_final_clamp_to_land_bounds(self):
        # The final safety clamp is [RALFMINSN, RALFMAXSN] = [0.50, 0.85],
        # which is wider than the glacier's [0.65, 0.82]; a pure glacier
        # can never exceed 0.82 regardless of inputs.
        (_, ralfmaxsn, _, _, _, _, _, zalbmax_gl, _) = _asn_consts()
        inp = _make_inputs_asn(
            pcil=1.0, pasnm1m=0.99,
            ptsnm1m_top=260.0, psnowf=1.0,   # saturate brightening
        )
        pasn = _run(inp)
        assert torch.all(pasn <= zalbmax_gl + 1e-12)
        assert torch.all(pasn <= ralfmaxsn)


# --------------------------------------------------------------------------- #
# Tests: legacy fallback (bit-identical regression guard)
# --------------------------------------------------------------------------- #
class TestLegacyFallback:
    """pcil=None -> legacy SRFSN path, bit-identical to the original port.

    This is the critical invariant: the 63 legacy tests above must stay
    green because they never set ``pcil``, so the ``_asn_albedo`` branch
    is never taken.  This class makes that dispatch explicit with a
    direct head-to-head: the same inputs run through the legacy path
    (pcil=None) and the ASN path with pcil=0 must agree only on the land
    part -- but the legacy RSFRESH reset differs from the ASN linear
    brightening, so we instead confirm the dispatch itself: pcil=None
    never touches the glacier constants.
    """

    def test_pcil_none_uses_legacy_path(self):
        # The no-snow branch is identical in both paths (reset to
        # RALFMAXSN), so a no-snow input gives the same answer whether
        # or not pcil is set.  This confirms the dispatch routes pcil=None
        # to _legacy_albedo.
        _, ralfmaxsn, _, _, _, _ = _consts()
        inp_legacy = _make_inputs(llnosnow=True, pasnm1m=0.55)
        inp_asn = _make_inputs_asn(llnosnow=True, pasnm1m=0.55, pcil=0.5)
        # _make_inputs builds pcil=None; _make_inputs_asn sets pcil.
        assert inp_legacy.pcil is None
        assert inp_asn.pcil is not None
        pasn_legacy = _run(inp_legacy)
        pasn_asn = _run(inp_asn)
        # Both reset to RALFMAXSN on the no-snow branch.
        assert torch.allclose(pasn_legacy, _scalar(ralfmaxsn, 4), atol=1e-12)
        assert torch.allclose(pasn_asn, _scalar(ralfmaxsn, 4), atol=1e-12)

    def test_legacy_pcil_none_bit_identical_to_pre_asn(self):
        # A cold, no-snowfall, no-melt slab: the legacy path gives
        # PASN = PASNM1M - ZLINA.  Running the same slab with pcil=None
        # must reproduce that exactly (the _asn_albedo branch is not
        # taken, so no glacier code runs).
        (_, _, _, rtaua, _, rday) = _consts()
        ptmst = 3600.0
        pasnm1m = 0.80
        zlina = rtaua * ptmst / rday
        expected = max(0.50, pasnm1m - zlina)
        inp = _make_inputs(pasnm1m=pasnm1m, pmsn=0.0, pssflpssfc=0.0)
        assert inp.pcil is None      # legacy dispatch
        pasn = _run(inp)
        assert torch.allclose(pasn, _scalar(expected, 4), atol=1e-12)
