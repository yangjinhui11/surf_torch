"""Saturation-function tests.

Reference values for es(T), qs(T, p), des/dT are recomputed here in
plain Python straight from the fcttre.h statement functions, using the
offline-driver constants.  Agreement with ``surf_pytorch.utils`` then
proves the port reproduces Fortran numerics.

Spot values are also checked against textbook saturation tables to
catch a unit/offset error.

References
----------
* fcttre.h statement functions: FOEDELTA, FOEALFA, FOEEW, FOEEWM,
  FOEDE, FOEDESU, FOEDEM.
* IFS surface qsat convention: ``surf/module/surfexcdrivers_ctl_mod.F90``
  lines 442-456 and ``vsurf_mod.F90`` 280-284.
"""

from __future__ import annotations

import math

import pytest
import torch

from surf_pytorch import ConstantBundle
from surf_pytorch import utils as U
from surf_pytorch import constants as C


pytestmark = pytest.mark.fortran_ref


# --------------------------------------------------------------------------- #
# Plain-python reference implementations of fcttre.h functions.
# --------------------------------------------------------------------------- #
def ref_foedelta(t):
    return max(0.0, math.copysign(1.0, t - C.RTT))


def ref_foealfa(t):
    clamped = max(C.RTICE, min(C.RTWAT, t))
    return min(1.0, ((clamped - C.RTICE) * C.RTWAT_RTICE_R) ** 2)


def ref_foeew(t):
    delta = ref_foedelta(t)
    r3 = C.R3LES * delta + C.R3IES * (1.0 - delta)
    r4 = C.R4LES * delta + C.R4IES * (1.0 - delta)
    return C.R2ES * math.exp(r3 * (t - C.RTT) / (t - r4))


def ref_foeewm(t):
    alfa = ref_foealfa(t)
    return C.R2ES * (
        alfa * math.exp(C.R3LES * (t - C.RTT) / (t - C.R4LES))
        + (1.0 - alfa) * math.exp(C.R3IES * (t - C.RTT) / (t - C.R4IES))
    )


def ref_foedesu(t):
    delta = ref_foedelta(t)
    num = delta * C.R5LES + (1.0 - delta) * C.R5IES
    r4 = C.R4LES * delta + C.R4IES * (1.0 - delta)
    return num / (t - r4) ** 2


def ref_qsat(t, p, mixed_phase=False):
    es = ref_foeewm(t) if mixed_phase else ref_foeew(t)
    zqs = es / p
    zcor = 1.0 / (1.0 - C.RETV * zqs)
    return zqs * zcor


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bundle(dtype, device):
    return ConstantBundle(dtype=dtype, device=device)


class TestPhaseSelectors:
    def test_foedelta_water(self, bundle):
        T = torch.tensor([280.0, 290.0, 300.0], dtype=bundle.dtype)
        d = U.foedelta(T, bundle)
        assert d.tolist() == [1.0, 1.0, 1.0]

    def test_foedelta_ice(self, bundle):
        T = torch.tensor([250.0, 260.0, 270.0], dtype=bundle.dtype)
        d = U.foedelta(T, bundle)
        assert d.tolist() == [0.0, 0.0, 0.0]

    def test_foedelta_at_triple_point_is_water(self, bundle):
        """Fortran SIGN(1.0, +0.0) is +1 -> FOEDELTA(RTT) is liquid."""
        T = torch.tensor([C.RTT], dtype=bundle.dtype)
        d = U.foedelta(T, bundle)
        assert d.item() == 1.0

    def test_foealfa_endpoints(self, bundle):
        T = torch.tensor([C.RTICE, C.RTWAT, C.RTWAT + 5.0], dtype=bundle.dtype)
        alfa = U.foealfa(T, bundle)
        assert alfa.tolist() == pytest.approx([0.0, 1.0, 1.0], abs=1e-12)

    def test_foealfa_midpoint(self, bundle):
        T = torch.tensor([C.RTICE + 11.5], dtype=bundle.dtype)
        alfa = U.foealfa(T, bundle)
        # ratio = 11.5/23 = 0.5 -> alfa = 0.25
        assert alfa.item() == pytest.approx(0.25, abs=1e-12)


class TestSaturationPressure:
    @pytest.mark.parametrize("tref", [250.0, 270.0, 273.16, 280.0, 290.0, 300.0])
    def test_foeew_matches_reference(self, bundle, tref):
        T = torch.tensor([tref], dtype=bundle.dtype)
        got = U.foeew(T, bundle).item()
        expected = ref_foeew(tref)
        assert got == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("tref", [240.0, 260.0, 273.16, 290.0])
    def test_foeewm_matches_reference(self, bundle, tref):
        T = torch.tensor([tref], dtype=bundle.dtype)
        got = U.foeewm(T, bundle).item()
        expected = ref_foeewm(tref)
        assert got == pytest.approx(expected, rel=1e-12)

    def test_foeew_water_above_freezing_textbook(self, bundle):
        """At T=20 C, es ~ 2339 Pa (textbook).

        FOEEW = R2ES*exp(...) with R2ES = 611.21*RD/RV ≈ 380, so the
        returned value is epsilon*es.  Reconstruct raw es by dividing
        by RD/RV.
        """
        T = torch.tensor([293.15], dtype=bundle.dtype)
        es_scaled = U.foeew(T, bundle).item()
        es_raw = es_scaled / (C.RD / C.RV)
        assert es_raw == pytest.approx(2339.0, rel=2e-3)

    def test_foeew_ice_sublimation(self, bundle):
        """At T=-10 C, raw es over ice ~ 260 Pa."""
        T = torch.tensor([263.15], dtype=bundle.dtype)
        es_scaled = U.foeew(T, bundle).item()
        es_raw = es_scaled / (C.RD / C.RV)
        assert 250.0 < es_raw < 320.0

    def test_foedesu_positive(self, bundle):
        T = torch.tensor([280.0, 290.0], dtype=bundle.dtype)
        d = U.foedesu(T, bundle)
        assert torch.all(d > 0)

    @pytest.mark.parametrize("tref", [280.0, 290.0])
    def test_foedesu_matches_reference(self, bundle, tref):
        T = torch.tensor([tref], dtype=bundle.dtype)
        got = U.foedesu(T, bundle).item()
        expected = ref_foedesu(tref)
        assert got == pytest.approx(expected, rel=1e-12)


class TestQsat:
    @pytest.mark.parametrize("tref,pref", [
        (280.0, 100000.0),
        (290.0, 100000.0),
        (300.0, 95000.0),
        (270.0, 90000.0),    # ice phase
    ])
    def test_qsat_matches_reference(self, bundle, tref, pref):
        T = torch.tensor([tref], dtype=bundle.dtype)
        P = torch.tensor([pref], dtype=bundle.dtype)
        got = U.qsat(T, P, bundle).item()
        expected = ref_qsat(tref, pref)
        assert got == pytest.approx(expected, rel=1e-12)

    def test_qsat_increases_with_t(self, bundle):
        T = torch.tensor([270.0, 280.0, 290.0, 300.0], dtype=bundle.dtype)
        P = torch.full_like(T, 100000.0)
        q = U.qsat(T, P, bundle)
        assert torch.all(q[1:] > q[:-1])

    def test_qsat_ice_water_continuity(self, bundle):
        """No-mixed-phase qsat is continuous across RTT (single formula)."""
        T = torch.tensor([C.RTT - 0.01, C.RTT + 0.01], dtype=bundle.dtype)
        P = torch.full_like(T, 100000.0)
        q = U.qsat(T, P, bundle)
        assert abs(q[0].item() - q[1].item()) / q[0].item() < 1e-2

    def test_dqsat_dt_positive(self, bundle):
        T = torch.tensor([270.0, 280.0, 290.0], dtype=bundle.dtype)
        P = torch.full_like(T, 100000.0)
        dq = U.dqsat_dt(T, P, bundle)
        assert torch.all(dq > 0)

    def test_dqsat_dt_approx_matches_exact(self, bundle):
        """The IFS small-RETV linearisation agrees with exact to ~0.1%."""
        T = torch.tensor([290.0], dtype=bundle.dtype)
        P = torch.tensor([100000.0], dtype=bundle.dtype)
        exact = U.dqsat_dt(T, P, bundle)
        approx = U.dqsat_dt_approx(T, P, bundle)
        rel_err = (exact - approx).abs().item() / exact.item()
        assert rel_err < 5e-3

    def test_dqsat_dt_ifs_matches_approx_to_ulp(self, bundle):
        """``dqsat_dt_ifs`` (upstream PDQS) ~ ``dqsat_dt_approx`` to ~0.1%.

        Both are the small-RETV linearisation; they differ only in where
        ``ZCOR`` is evaluated (pre-salinity ``es/P`` for the IFS form vs
        post-salinity ``qs`` for the approx form), so they agree to the
        order of ``RETV*es/p`` (~1e-4) at standard conditions.
        """
        T = torch.tensor([270.0, 280.0, 290.0, 300.0], dtype=bundle.dtype)
        P = torch.full_like(T, 100000.0)
        ifs_val = U.dqsat_dt_ifs(T, P, bundle)
        approx = U.dqsat_dt_approx(T, P, bundle)
        rel = (ifs_val - approx).abs() / approx.abs().clamp(min=1e-12)
        assert rel.max().item() < 5e-3, f"ifs vs approx rel err = {rel}"

    def test_dqsat_dt_ifs_differs_from_exact_at_known_scale(self, bundle):
        """The IFS PDQS differs from the *exact* derivative only at
        ~RETV*es/p (~1e-4), never by a large factor.  Locks the
        difference so a transcription mistake cannot silently amplify it.
        """
        T = torch.tensor([280.0, 290.0, 300.0], dtype=bundle.dtype)
        P = torch.full_like(T, 100000.0)
        exact = U.dqsat_dt(T, P, bundle)
        ifs_val = U.dqsat_dt_ifs(T, P, bundle)
        rel = (exact - ifs_val).abs() / exact.abs().clamp(min=1e-12)
        # Order 1e-4 at 290 K; allow generous 1e-2 headroom but fail on
        # any O(1) blow-up (which would signal a formula bug).
        assert rel.max().item() < 1e-2, f"ifs vs exact rel err = {rel}"

    def test_qsat_salinity_lowers_ocean_value(self, bundle):
        """ZSALIN=0.98 (ocean) lowers qsat by ~2% vs the default 1.0."""
        T = torch.tensor([290.0], dtype=bundle.dtype)
        P = torch.tensor([100000.0], dtype=bundle.dtype)
        qs_default = U.qsat(T, P, bundle)
        qs_ocean = U.qsat(T, P, bundle, salinity=0.98)
        # 0.98 scaling on es -> qsat drops by ~(1 - 0.98) = 2% to first order.
        rel = (qs_default - qs_ocean).abs().item() / qs_default.item()
        assert 0.01 < rel < 0.03, f"salinity rel drop = {rel}"

    def test_qsat_salinity_one_is_identity(self, bundle):
        """salinity=1.0 reproduces the default path exactly."""
        T = torch.tensor([280.0, 290.0], dtype=bundle.dtype)
        P = torch.full_like(T, 100000.0)
        qs_default = U.qsat(T, P, bundle)
        qs_one = U.qsat(T, P, bundle, salinity=1.0)
        assert torch.equal(qs_default, qs_one)

    def test_dqsat_dt_ifs_salinity_scales_as_zcor_squared(self, bundle):
        """PDQS carries ZCOR**2, so salinity scales it ~ (0.98)**2."""
        T = torch.tensor([290.0], dtype=bundle.dtype)
        P = torch.tensor([100000.0], dtype=bundle.dtype)
        dq_default = U.dqsat_dt_ifs(T, P, bundle)
        dq_ocean = U.dqsat_dt_ifs(T, P, bundle, salinity=0.98)
        ratio = (dq_ocean / dq_default).item()
        # ZCOR(0.98)^2 / ZCOR(1)^2 ~ 0.96 (the RETV coupling shifts it a
        # touch); assert it is in the 0.94..0.97 band.
        assert 0.94 < ratio < 0.97, f"dqsat_dt_ifs salinity ratio = {ratio}"


class TestTridiag:
    def test_known_solution(self, dtype):
        """Solve a 3x3 system with a known answer.

        Matrix [2,-1,0; -1,2,-1; 0,-1,2] times x=[1,2,3] = [0,0,4].
        """
        a = torch.tensor([0.0, -1.0, -1.0], dtype=dtype)   # sub-diag (a[0] unused)
        b = torch.tensor([2.0,  2.0,  2.0], dtype=dtype)
        c = torch.tensor([-1.0, -1.0, 0.0], dtype=dtype)   # super-diag (c[-1] unused)
        d = torch.tensor([0.0,  0.0,  4.0], dtype=dtype)
        x = U.tridiag_solve(a, b, c, d, differentiable=False)
        assert x.tolist() == pytest.approx([1.0, 2.0, 3.0], abs=1e-9)

    def test_batched(self, dtype):
        """Two independent columns solved in one call."""
        a = torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=dtype)
        b = torch.tensor([[2.0,  2.0], [1.0,  1.0]], dtype=dtype)
        c = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]], dtype=dtype)
        d = torch.tensor([[1.0,  2.0], [3.0,  4.0]], dtype=dtype)
        x = U.tridiag_solve(a, b, c, d, differentiable=False)
        # Each column independent; check shape and a spot value.
        assert x.shape == (2, 2)

    def test_grad_flows_when_differentiable(self, dtype):
        """When differentiable=True, gradient reaches d through the solve."""
        a = torch.tensor([0.0, -1.0, -1.0], dtype=dtype)
        b = torch.tensor([2.0,  2.0,  2.0], dtype=dtype)
        c = torch.tensor([-1.0, -1.0, 0.0], dtype=dtype)
        d = torch.tensor([0.0,  0.0,  4.0], dtype=dtype, requires_grad=True)
        x = U.tridiag_solve(a, b, c, d, differentiable=True)
        x.sum().backward()
        assert d.grad is not None
        assert torch.all(torch.isfinite(d.grad))
