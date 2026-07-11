"""Constants tests.

Every value here is the exact literal that appears in the offline driver
(``surf/offline/driver/sucst.F90`` for yomcst, ``suphec.F90`` for
yoethf).  Pinning these in tests catches transcription errors early
before the physics core consumes them.
"""

from __future__ import annotations

import math

import pytest
import torch

from surf_pytorch import PhysicalConstants, ConstantBundle
from surf_pytorch import constants as C


pytestmark = pytest.mark.fortran_ref


# --------------------------------------------------------------------------- #
# yomcst (sucst.F90)
# --------------------------------------------------------------------------- #
def test_fundamental_constants():
    assert C.RPI == 2.0 * math.asin(1.0)
    assert C.RCLUM == 299792458.0
    assert C.RHPLA == 6.6260755e-34
    assert C.RKBOL == 1.380658e-23
    assert C.RNAVO == 6.0221367e23


def test_geoide_and_rotation():
    assert C.RG == 9.80665
    assert C.RA == 6371229.0
    # sidereal day = RDAY/(1 + RDAY/RSIYEA), ROMEGA = 2*pi/RSIDAY
    assert abs(C.ROMEGA - 7.29212e-5) < 1e-9


def test_gas_phase():
    # RD = 1000 * R / RMD, where R = RNAVO*RKBOL
    R = C.RNAVO * C.RKBOL
    assert C.RD == 1000.0 * R / C.RMD
    assert C.RV == 1000.0 * R / C.RMV
    assert C.RCPD == 3.5 * C.RD
    assert abs(C.RETV - (C.RV / C.RD - 1.0)) < 1e-15


def test_phase_transition():
    assert C.RTT == 273.16
    assert C.RLVTT == 2.5008e6
    assert C.RLSTT == 2.8345e6
    assert C.RLMLT == C.RLSTT - C.RLVTT
    assert abs(C.RLMLT - 3.337e5) < 1e-2  # canonical latent heat of fusion


def test_stefan_boltzmann():
    # IFS computes RSIGMA from RCLUM/RHPLA/RKBOL (slightly outdated
    # values), so the result is 5.6705085e-8 -- NOT the modern CODATA
    # 5.670374e-8.  Pin to the Fortran-computed value, with the
    # expression matching sucst.F90 line-for-line.
    expected = 2.0 * math.pi**5 * C.RKBOL**4 / (15.0 * C.RCLUM**2 * C.RHPLA**3)
    assert abs(C.RSIGMA - expected) < 1e-30
    assert abs(C.RSIGMA - 5.6705085385e-8) < 1e-17


# --------------------------------------------------------------------------- #
# yoethf (suphec.F90) -- offline driver overrides
# --------------------------------------------------------------------------- #
def test_yoethf_overrides():
    """The offline driver overrides canonical values."""
    # RVTMP2 must be 0 (NOT RCW/RCPV - 1).
    assert C.RVTMP2 == 0.0
    # R2ES = 611.21 * RD / RV (NOT RESTT*RD/RV).
    assert abs(C.R2ES - 611.21 * C.RD / C.RV) < 1e-12
    # saturation curve slopes unchanged.
    assert C.R3LES == 17.502
    assert C.R3IES == 22.587
    assert C.R4LES == 32.19
    assert C.R4IES == -0.7
    assert C.R5LES == C.R3LES * (C.RTT - C.R4LES)
    assert C.R5IES == C.R3IES * (C.RTT - C.R4IES)


def test_phase_transition_thresholds():
    assert C.RTWAT == C.RTT
    assert C.RTICE == C.RTT - 23.0
    assert C.RTICECU == C.RTT - 23.0
    assert abs(C.RTWAT_RTICE_R - 1.0 / 23.0) < 1e-12


def test_saturation_curve_coefficients():
    """FOEEW/FOEEQS curve constants from sucst section 9."""
    assert C.RESTT == 611.14
    RGAMW_expected = (C.RCW - C.RCPV) / C.RV
    RBETW_expected = C.RLVTT / C.RV + RGAMW_expected * C.RTT
    RALPW_expected = math.log(C.RESTT) + RBETW_expected / C.RTT + RGAMW_expected * math.log(C.RTT)
    assert abs(C.RGAMW - RGAMW_expected) < 1e-9
    assert abs(C.RBETW - RBETW_expected) < 1e-3
    assert abs(C.RALPW - RALPW_expected) < 1e-6


# --------------------------------------------------------------------------- #
# ConstantBundle (tensor view)
# --------------------------------------------------------------------------- #
def test_bundle_dtype_and_device(dtype, device):
    b = ConstantBundle(dtype=dtype, device=device)
    assert b.dtype == dtype
    assert b.rtt.dtype == dtype
    assert b.rtt.device == device
    assert b.rtt.item() == C.RTT
    # all physical-constant attributes are tensors of correct dtype
    from surf_pytorch.constants import PhysicalConstants
    for name in PhysicalConstants.__dataclass_fields__:
        t = getattr(b, name)
        assert isinstance(t, torch.Tensor), f"{name} not a tensor"
        assert t.dtype == dtype, f"{name} wrong dtype"
        assert t.requires_grad is False, f"{name} should not require grad"


def test_bundle_no_requires_grad(dtype):
    """Constants are leaves with requires_grad=False -- no graph leakage."""
    b = ConstantBundle(dtype=dtype)
    assert not b.rtt.requires_grad
    assert not b.r2es.requires_grad
    assert not b.retv.requires_grad
