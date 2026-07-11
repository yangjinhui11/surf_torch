"""Shared pytest fixtures for surf-pytorch tests."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

# Make both the ``surf_pytorch`` package and the sibling
# ``offline_driver`` package importable regardless of the CWD pytest is
# invoked from.  Without this, ``import offline_driver`` (used by the
# Fortran-reference comparison tests) only works when pytest is run
# from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture(scope="session")
def dtype() -> torch.dtype:
    """Default test dtype: float64, matches the Fortran JPRB."""
    return torch.float64


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def atol_strict() -> float:
    """Tolerance for values that must match Fortran to last ULP (1e-12)."""
    return 1e-12


@pytest.fixture(scope="session")
def atol_loose() -> float:
    """Looser tolerance for compound-derived quantities (1e-9)."""
    return 1e-9


def assert_allclose(actual, expected, *, atol, rtol=1e-9, msg=""):
    """torch-aware allclose that prints a useful diff on failure."""
    if not isinstance(actual, torch.Tensor):
        actual = torch.as_tensor(actual, dtype=torch.float64)
    if not isinstance(expected, torch.Tensor):
        expected = torch.as_tensor(expected, dtype=torch.float64)
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(
            f"{msg or 'tensors differ'}: max abs diff = {diff:.3e} "
            f"(atol={atol:.0e}); actual={actual.tolist()}, "
            f"expected={expected.tolist()}"
        )
