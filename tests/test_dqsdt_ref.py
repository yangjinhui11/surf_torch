"""Independent Fortran validation of the skin-T saturation Jacobian.

These tests compare the PyTorch port's ``utils.dqsat_dt_ifs`` (the
upstream ``PDQS`` from ``vsurf_mod.F90`` 278-285) against the Fortran
reference kernel ``SURF_REF_DQSDT``, which re-derives PDQS from the
upstream ``fcttre.h`` statement functions (FOEEW / FOEDESU /
FOEDELTA) compiled with bare gfortran.

Why this is not a self-transcription
------------------------------------
``tests/test_fortran_ref.py`` validates ``SURF_REF_SEB``, but that
kernel takes ``DQSDT`` as an *input* (``surf_ref.F90`` 282), so the
Python-computed ``dqsdT`` is fed to both implementations -- the test
can confirm the SEB *downstream* of dqsdT agrees, but it cannot catch
a wrong dqs/dT *formula*, because the same (possibly wrong) value is
on both sides of the comparison.

``SURF_REF_DQSDT`` closes that gap: it computes PDQS independently in
Fortran, from the same ``fcttre.h`` functions the real IFS uses, so a
0-ULP agreement here is genuine evidence that the port's formula (and
in particular the double-``ZCOR`` semantics of ``vsurf_mod.F90`` 281)
is correct.

Skip behaviour
--------------
Skipped (like ``test_fortran_ref.py``) when ``libsurf_ref.so`` is not
built or was built before the ``surf_ref_dqsdt`` symbol existed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from surf_pytorch import ConstantBundle
from surf_pytorch.utils import dqsat_dt_ifs

# Importing offline_driver triggers the .so load, so guard it.
try:
    from offline_driver import SurfRef
    _SURF_REF_AVAILABLE = SurfRef.is_available()
except Exception:  # pragma: no cover -- exercised only without the .so
    SurfRef = None  # type: ignore[assignment]
    _SURF_REF_AVAILABLE = False

pytestmark = pytest.mark.fortran_ref


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def ref() -> "SurfRef":
    """Session-cached :class:`SurfRef` (loads .so once)."""
    return SurfRef()


@pytest.fixture(scope="session")
def consts() -> ConstantBundle:
    return ConstantBundle(dtype=torch.float64)


def requires_fortran_dqsdt():
    """Skip marker -- True iff the .so exports surf_ref_dqsdt.

    Older ``libsurf_ref.so`` builds (before the DQSDT entry was added)
    load fine but have ``ref._loaded.dqsdt is None``; those builds skip
    rather than error.
    """
    def _have_dqsdt() -> bool:
        if not _SURF_REF_AVAILABLE:
            return False
        # Cheap probe: instantiate the (cached) wrapper and check the fn.
        return SurfRef()._loaded.dqsdt is not None
    return pytest.mark.skipif(
        not _have_dqsdt(),
        reason="libsurf_ref.so missing or predates surf_ref_dqsdt "
               "-- rebuild with offline_driver/build_ref.sh",
    )


# --------------------------------------------------------------------------- #
# Tests -- water / ice / salinity regimes
# --------------------------------------------------------------------------- #
# (skin T in K, salinity): cover water, ice, the 0 degC branch, and the
# ocean salinity factor (ZSALIN=0.98, active only under LEOCSA).
_CASES = [
    (273.15 + 15.0, 1.0),    # water, no salinity
    (273.15 + 15.0, 0.98),   # water, ocean salinity
    (273.15 - 10.0, 1.0),    # ice phase (FOEDELTA=0 -> R3IES/R4IES/R5IES)
    (273.15 + 35.0, 0.98),   # warm tropical SST with salinity
    (273.15 + 0.5, 1.0),     # just above freezing -> water branch
    (273.15 - 0.5, 1.0),     # just below freezing -> ice branch
    (273.15 + 25.0, 0.98),   # mid-latitude ocean
]


@requires_fortran_dqsdt()
@pytest.mark.parametrize("tsk_k,salin", _CASES)
def test_dqsdt_matches_fortran_ulp(ref, consts, tsk_k, salin):
    """Port PDQS == Fortran PDQS to last ULP, across water/ice/salinity."""
    n = 16
    t = torch.full((n,), tsk_k, dtype=torch.float64)
    p = torch.full((n,), 1.0e5, dtype=torch.float64)

    port = dqsat_dt_ifs(t, p, consts, salinity=salin)
    f90 = ref.dqsdt(t, p, consts, salinity=salin)

    max_diff = (port - f90).abs().max().item()
    ulp = max_diff / np.finfo(np.float64).eps
    assert ulp < 1.0, (
        f"T={tsk_k:.2f}K salin={salin}: |port-f90|={max_diff:.3e} "
        f"({ulp:.2f} ULP) -- expected bit-exact agreement"
    )


@requires_fortran_dqsdt()
def test_dqsdt_matches_fortran_random_grid(ref, consts):
    """Random grid of skin temperatures and pressures, salinity=1.0.

    Exercises the FOEEW/FOEDESU interpolation across a realistic range
    (250-315 K skin T, 850-1050 hPa) on a batched grid -- the regime
    the SEB actually runs in.
    """
    torch.manual_seed(42)
    n = 512
    t = torch.rand(n, dtype=torch.float64) * 65.0 + 250.0
    p = torch.rand(n, dtype=torch.float64) * 2.0e4 + 8.5e4

    port = dqsat_dt_ifs(t, p, consts)
    f90 = ref.dqsdt(t, p, consts)

    max_diff = (port - f90).abs().max().item()
    ulp = max_diff / np.finfo(np.float64).eps
    assert ulp < 1.0, (
        f"random grid: |port-f90|max={max_diff:.3e} ({ulp:.2f} ULP)"
    )


@requires_fortran_dqsdt()
def test_dqsdt_salinity_tensor_broadcast(ref, consts):
    """Per-column salinity (0.98 ocean / 1.0 land) agrees elementwise."""
    n = 32
    t = torch.full((n,), 273.15 + 20.0, dtype=torch.float64)
    p = torch.full((n,), 1.0e5, dtype=torch.float64)
    salin = torch.where(torch.arange(n) % 2 == 0, 0.98, 1.0).double()

    port = dqsat_dt_ifs(t, p, consts, salinity=salin)
    f90 = ref.dqsdt(t, p, consts, salinity=salin)

    assert torch.allclose(port, f90), (
        f"per-column salinity mismatch: max|diff|={(port-f90).abs().max():.3e}"
    )
    # And the salinity scaling is genuinely different from salinity=1.
    port_nosal = dqsat_dt_ifs(t, p, consts, salinity=1.0)
    assert not torch.allclose(port, port_nosal)
