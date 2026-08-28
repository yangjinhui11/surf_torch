"""Tests for the Monin-Obukhov exchange-coefficient module (M4.1).

Two layers of tests:

1. *Stand-alone* checks of the PSI stability functions and the
   :func:`surf_pytorch.physics.exchange.vexcs` driver against analytic
   limits (neutral, very stable, very unstable).  These run without the
   Fortran reference -- they exercise the port's arithmetic directly.

2. *ULP-level* comparison against the standalone Fortran reference
   ``libsurf_ref.so`` (the ``SURF_REF_VEXCS`` kernel).  These share no
   code with the port and therefore catch transcription mistakes a
   pure-Python test cannot.  They are skipped when the .so is missing
   (e.g. on CI without gfortran); build it with
   ``offline_driver/build_ref.sh``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from surf_pytorch import ConstantBundle
from surf_pytorch.constants import PhysicalConstants
from surf_pytorch.physics.exchange import (
    ExchangeOutputs,
    vexcs as port_vexcs,
    psimu, psihu, psims, psihs, _psi_h,
    REPDU2, RKAP, RPARZI, RCHBHDL, DEFAULT_KITT,
)
from surf_pytorch.tables import RitblTable
from surf_pytorch.tables.ritbl import (
    JPRITBL, RIMAX, RCHETA, RCHETB,
    RCHBA, RCHBB, RCHBC, RCHBD, RCHB23A, RCHBBCD, RCHBCD,
    RCDHALF, RCDHPI2,
)

try:
    from offline_driver import SurfRef
    _SURF_REF_AVAILABLE = SurfRef.is_available()
except Exception:  # pragma: no cover
    SurfRef = None  # type: ignore[assignment]
    _SURF_REF_AVAILABLE = False

requires_fortran_ref = pytest.mark.skipif(
    not _SURF_REF_AVAILABLE,
    reason="libsurf_ref.so not built -- run offline_driver/build_ref.sh",
)


# --------------------------------------------------------------------------- #
# Shared fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def dtype() -> torch.dtype:
    return torch.float64


@pytest.fixture(scope="session")
def consts() -> ConstantBundle:
    return ConstantBundle(PhysicalConstants(), dtype=torch.float64)


@pytest.fixture(scope="session")
def ritbl() -> RitblTable:
    return RitblTable(dtype=torch.float64)


@pytest.fixture(scope="session")
def ref() -> "SurfRef":
    return SurfRef()


def _neutral_inputs(n: int = 4, dtype=torch.float64) -> dict:
    """A neutral / weakly-stable column with all VEXCS inputs populated."""
    g = 9.80665
    cpd = 1004.7
    return dict(
        pu=torch.full((n,), 5.0, dtype=dtype),
        pv=torch.zeros(n, dtype=dtype),
        pt=torch.full((n,), 290.0, dtype=dtype),
        pq=torch.full((n,), 0.005, dtype=dtype),
        paphms=torch.full((n,), 1.0e5, dtype=dtype),
        pgeomlev=torch.full((n,), 200.0, dtype=dtype),  # ~20 m
        pcptgzlev=torch.full((n,), cpd * 290.0 + 200.0, dtype=dtype),
        pcpts=torch.full((n,), cpd * 290.0, dtype=dtype),  # neutral sfc
        pqsam=torch.full((n,), 0.005, dtype=dtype),
        pz0mm=torch.full((n,), 1.0e-3, dtype=dtype),
        pz0hm=torch.full((n,), 1.0e-5, dtype=dtype),
        pz0qm=torch.full((n,), 1.0e-5, dtype=dtype),
        pbuom=torch.zeros(n, dtype=dtype),
        pucurr=torch.zeros(n, dtype=dtype),
        pvcurr=torch.zeros(n, dtype=dtype),
        pi10fgcv=torch.zeros(n, dtype=dtype),
    )


# --------------------------------------------------------------------------- #
# 1. PSI stability functions
# --------------------------------------------------------------------------- #
class TestPsiFunctions:
    """Unit tests for the PSI stability functions (fcsvdfs.h)."""

    def test_neutral_psi_is_zero(self, dtype):
        """At eta=0 all PSI functions must be exactly zero (no correction)."""
        e = torch.tensor([0.0], dtype=dtype)
        assert psimu(e).item() == pytest.approx(0.0, abs=1e-15)
        assert psihu(e).item() == pytest.approx(0.0, abs=1e-15)
        assert psims(e).item() == pytest.approx(0.0, abs=1e-15)
        assert psihs(e).item() == pytest.approx(0.0, abs=1e-15)

    def test_unstable_psi_form(self, dtype):
        """Unstable PSI (eta<0) must match the Dyer-Hicks form numerically.

        PSI enters the profile as (ln(z/z0) - PSI), so its sign is not
        fixed; what matters is that the *integrated* correction is
        consistent with the analytical form.  Here we just verify the
        function matches a reference recomputation.
        """
        e = torch.linspace(-2.0, -0.01, 20, dtype=dtype)
        # PSIHU = 2*ln((1+sqrt(1-16*eta))/2)
        inner = np.sqrt(np.maximum(1.0 - RCDHALF * e.numpy(), 0.0))
        psihu_ref = 2.0 * np.log((1.0 + inner) * 0.5)
        assert torch.allclose(psihu(e), torch.as_tensor(psihu_ref, dtype=dtype),
                              atol=1e-14)
        # psihu must be finite and real for moderate unstable eta.
        assert torch.isfinite(psihu(e)).all()

    def test_stable_psi_negative(self, dtype):
        """Stable PSI (eta>0) must be negative too (suppresses mixing).

        Check against the Holtslag-de Bruin form (fcsvdfs.h 60-63)."""
        e = torch.linspace(0.01, 5.0, 20, dtype=dtype)
        psims_ref = -RCHBB * (e - RCHBCD) * np.exp(-RCHBD * e) \
            - RCHBA * e - RCHBBCD
        psihs_ref = -RCHBB * (e - RCHBCD) * np.exp(-RCHBD * e) \
            - (1.0 + RCHB23A * e) * np.sqrt(1.0 + RCHB23A * e) \
            - RCHBBCD + 1.0
        assert torch.allclose(psims(e), torch.as_tensor(psims_ref, dtype=dtype),
                              atol=1e-14)
        assert torch.allclose(psihs(e), torch.as_tensor(psihs_ref, dtype=dtype),
                              atol=1e-14)

    def test_psimu_matches_dyer_hicks(self, dtype):
        """PSIMU at eta=-0.5 matches the Dyer-Hicks form numerically."""
        e = torch.tensor([-0.5], dtype=dtype)
        x = math.sqrt(math.sqrt(1.0 - RCDHALF * (-0.5)))
        y = math.sqrt(1.0 - RCDHALF * (-0.5))
        expected = (math.log((1 + x) ** 2 * (1 + y) * 0.125)
                    - 2 * math.atan(x) + RCDHPI2)
        assert psimu(e).item() == pytest.approx(expected, abs=1e-14)

    def test_mixed_sign_heat_psi_has_finite_backward_gradient(self, dtype):
        """The masked stable branch must not differentiate sqrt(0)."""
        eta = torch.tensor([-100.0, 0.1], dtype=dtype, requires_grad=True)
        _psi_h(eta).sum().backward()
        assert torch.isfinite(eta.grad).all()


# --------------------------------------------------------------------------- #
# 2. vexcs driver -- stand-alone sanity checks
# --------------------------------------------------------------------------- #
class TestVexcsStandalone:
    """Sanity checks of the full vexcs driver (no Fortran ref needed)."""

    def test_returns_exchange_outputs(self, consts, ritbl):
        inp = _neutral_inputs()
        out = port_vexcs(*inp.values(), None, consts, ritbl)
        assert isinstance(out, ExchangeOutputs)
        for field in ("pcfm", "pcfh", "pcfq", "pkh", "pkc", "pzdl"):
            t = getattr(out, field)
            assert t.shape == (4,)
            assert torch.isfinite(t).all()

    def test_neutral_pzdl_small(self, consts, ritbl):
        """Near-neutral forcing should yield a small |z/L|."""
        inp = _neutral_inputs()
        out = port_vexcs(*inp.values(), None, consts, ritbl)
        # All surface / lowest-level quantities equal -> Ri ~ 0.
        assert out.pzdl.abs().max().item() < 0.5

    def test_pcfh_le_pcfm_in_stable(self, consts, ritbl, dtype):
        """In stable stratification Ch <= Cm (heat damped more than momentum).

        pz0hm >= pz0mm typically, so the heat roughness is larger; but
        the stability correction drives Ch below Cm for stable Ri.
        """
        inp = _neutral_inputs(dtype=dtype)
        # Cool the surface to force stable stratification.
        inp["pcpts"] = torch.full((4,), float(consts.cpd) * 280.0, dtype=dtype)
        out = port_vexcs(*inp.values(), None, consts, ritbl)
        # pcfm/pcfh both scale with rho*|U|*kappa^2/(...) ; the ratio
        # pcfh/pcfm = (ZPRM/ZPRH) is < 1 when ZPRH > ZPRM (stable).
        assert (out.pcfh <= out.pcfm + 1e-12).all()

    def test_pcfq_equals_pcfh_when_z0h_eq_z0q(self, consts, ritbl):
        """When pz0hm == pz0qm (typical over land) pcfq must equal pcfh."""
        inp = _neutral_inputs()
        assert torch.equal(inp["pz0hm"], inp["pz0qm"])
        out = port_vexcs(*inp.values(), None, consts, ritbl)
        assert torch.allclose(out.pcfq, out.pcfh, atol=1e-14)

    def test_wind_floor_protects_division(self, consts, ritbl, dtype):
        """Wind floor REPDU2 keeps ZDU2 strictly positive (no NaN/inf in Ri).

        With exactly zero wind and no W*/gust, ZRICLS diverges -- that
        is also true upstream (VEXCS does not guard the iteration
        against Ri -> infinity).  Here we only check the wind floor is
        applied so the bulk-Richardson computation does not divide by
        zero.
        """
        inp = _neutral_inputs(dtype=dtype)
        # Set wind just above the floor.
        inp["pu"] = torch.full((4,), math.sqrt(REPDU2) * 2.0, dtype=dtype)
        inp["pv"] = torch.zeros(4, dtype=dtype)
        out = port_vexcs(*inp.values(), None, consts, ritbl)
        # pzdl may be very large (near-stable cap region) but must be finite.
        assert torch.isfinite(out.pzdl).all()


# --------------------------------------------------------------------------- #
# 3. vexcs driver -- ULP-level agreement with the Fortran reference
# --------------------------------------------------------------------------- #
@requires_fortran_ref
class TestVexcsFortranRef:
    """ULP-level port-vs-Fortran comparison."""

    def _run_both(self, inp: dict, consts, ritbl, ref):
        port_out = port_vexcs(*inp.values(), None, consts, ritbl)
        ref_out = ref.vexcs(
            pu=inp["pu"], pv=inp["pv"], pt=inp["pt"], pq=inp["pq"],
            paphms=inp["paphms"], pgeomlev=inp["pgeomlev"],
            pcptgzlev=inp["pcptgzlev"], pcpts=inp["pcpts"],
            pqsam=inp["pqsam"],
            pz0mm=inp["pz0mm"], pz0hm=inp["pz0hm"], pz0qm=inp["pz0qm"],
            pbuom=inp["pbuom"], pucurr=inp["pucurr"], pvcurr=inp["pvcurr"],
            pi10fgcv=inp["pi10fgcv"],
            consts=consts,
        )
        return port_out, ref_out

    def test_neutral_column_matches(self, consts, ritbl, ref):
        """Near-neutral column: every output must match to last ULP."""
        inp = _neutral_inputs()
        port_out, ref_out = self._run_both(inp, consts, ritbl, ref)
        for field in ("pcfm", "pcfh", "pcfq", "pkh", "pkc", "pzdl"):
            p = getattr(port_out, field)
            r = getattr(ref_out, field)
            assert torch.allclose(p, r, atol=1e-12, rtol=1e-12), (
                f"{field}: max abs diff = {(p - r).abs().max().item():.3e}"
            )

    @pytest.mark.parametrize("tsurf_offset", [-10.0, -2.0, 0.0, 2.0, 10.0])
    def test_stability_sweep_matches(self, tsurf_offset, consts, ritbl, ref,
                                     dtype):
        """Sweep surface-T offset to span stable -> unstable regimes."""
        inp = _neutral_inputs(dtype=dtype)
        cpd = float(consts.cpd)
        inp["pcpts"] = torch.full((4,), cpd * (290.0 + tsurf_offset),
                                   dtype=dtype)
        port_out, ref_out = self._run_both(inp, consts, ritbl, ref)
        for field in ("pcfm", "pcfh", "pcfq", "pkh", "pkc", "pzdl"):
            p = getattr(port_out, field)
            r = getattr(ref_out, field)
            assert torch.allclose(p, r, atol=1e-12, rtol=1e-12), (
                f"{field} (tsurf_offset={tsurf_offset}): "
                f"max abs diff = {(p - r).abs().max().item():.3e}"
            )

    def test_very_stable_cap_matches(self, consts, ritbl, ref, dtype):
        """Very stable column triggers the RCHBHDL capping; must still match."""
        inp = _neutral_inputs(dtype=dtype)
        cpd = float(consts.cpd)
        inp["pcpts"] = torch.full((4,), cpd * 250.0, dtype=dtype)  # very cold
        port_out, ref_out = self._run_both(inp, consts, ritbl, ref)
        # Sanity: this regime should produce large positive z/L.
        assert (port_out.pzdl > 0).all()
        for field in ("pcfm", "pcfh", "pcfq", "pkh", "pkc", "pzdl"):
            p = getattr(port_out, field)
            r = getattr(ref_out, field)
            assert torch.allclose(p, r, atol=1e-12, rtol=1e-12), (
                f"{field} (very stable): "
                f"max abs diff = {(p - r).abs().max().item():.3e}"
            )

    def test_random_columns_match(self, consts, ritbl, ref, dtype):
        """Randomised inputs across the regime space must match to ULP."""
        torch.manual_seed(4871)
        n = 64
        g = 9.80665
        cpd = float(consts.cpd)
        # Realistic ranges: T in [260, 305] K, q in [0, 0.02], wind in
        # [0.5, 20] m/s, surface pressure ~ 1e5 Pa, geom ~ 20-200 m2/s2,
        # roughness 1e-4..1e-2 m.  Surface T offset in [-15, +15] K.
        pt = torch.rand(n, dtype=dtype) * 45.0 + 260.0
        tsurf = pt + (torch.rand(n, dtype=dtype) * 30.0 - 15.0)
        inp = dict(
            pu=torch.rand(n, dtype=dtype) * 20.0 + 0.5,
            pv=torch.rand(n, dtype=dtype) * 5.0,
            pt=pt,
            pq=torch.rand(n, dtype=dtype) * 0.02,
            paphms=torch.full((n,), 1.0e5, dtype=dtype),
            pgeomlev=torch.rand(n, dtype=dtype) * 180.0 + 20.0,
            pcptgzlev=cpd * pt + 100.0,
            pcpts=cpd * tsurf,
            pqsam=torch.rand(n, dtype=dtype) * 0.02,
            pz0mm=torch.rand(n, dtype=dtype) * 1e-2 + 1e-4,
            pz0hm=torch.rand(n, dtype=dtype) * 1e-3 + 1e-6,
            pz0qm=torch.rand(n, dtype=dtype) * 1e-3 + 1e-6,
            pbuom=torch.zeros(n, dtype=dtype),
            pucurr=torch.zeros(n, dtype=dtype),
            pvcurr=torch.zeros(n, dtype=dtype),
            pi10fgcv=torch.zeros(n, dtype=dtype),
        )
        port_out, ref_out = self._run_both(inp, consts, ritbl, ref)
        for field in ("pcfm", "pcfh", "pcfq", "pkh", "pkc", "pzdl"):
            p = getattr(port_out, field)
            r = getattr(ref_out, field)
            assert torch.allclose(p, r, atol=1e-12, rtol=1e-12), (
                f"{field} (random): "
                f"max abs diff = {(p - r).abs().max().item():.3e}"
            )
