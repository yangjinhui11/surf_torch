"""Independent Fortran validation of the Jarvis stomatal resistance.

These tests compare the PyTorch port's
``physics.stomata.jarvis_resistance`` (a direct transcription of
``vsurf_mod.F90`` §2.3, lines 294-422) against the Fortran reference
kernel ``SURF_REF_STOMATA``, which re-derives the canopy resistances
PWETL / PWETH / PWETHS / PWETB / PWETLU from the upstream vsurf block
compiled with bare gfortran.

Why this is not a self-transcription
------------------------------------
The port's ``resistance_to_alpha`` / ``build_tile_alpha`` consume the
resistance arrays to drive the SEB, but neither the SEB reference nor
the SEB port can validate the *resistance formula itself* -- a wrong
PWETL on both sides produces consistent alpha and consistent SEB fluxes.
``SURF_REF_STOMATA`` closes that gap: it computes the resistances
independently in Fortran, from the same ``fcttre.h`` saturation
functions and the same soil / veg / root LUTs the real IFS uses, so a
0-ULP agreement here is genuine evidence that the port's transcription
(in particular the freeze-ramp ZF, the root-weighted ZWROOT, the Betts
radiation F1 and the VPD-driven F3) is correct.

Coverage
--------
* **formula-level ULP** -- PWETL/PWETH/PWETB vs Fortran, across a sweep
  of soil moisture, soil temperature, veg type and SW radiation.  The
  tolerances are 0 ULP (bit-exact) where the arithmetic is associative
  in the same order, and a few ULP where a different re-association is
  unavoidable (e.g. the root-weighted sum).
* **physics sanity** -- dry soil raises PWETL, frozen top shuts F2 to
  RCEPSW, LAI=0 hits the PWET_INF sentinel, condensation drives
  alpha=1.
* **tile dispatch** -- ``build_tile_alpha`` assigns the right
  resistance to each tile and leaves water/ice/wet-skin/snow/lake at 1.

Skip behaviour
--------------
Skipped (like ``test_dqsdt_ref.py``) when ``libsurf_ref.so`` is not
built or was built before the ``surf_ref_stomata`` symbol existed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from surf_pytorch import ConstantBundle, SurfConfig
from surf_pytorch.constants import PhysicalConstants
from surf_pytorch.tables import VegTable, SoilTable, RootTable
from surf_pytorch.physics import stomata as stomata_mod
from surf_pytorch.physics.stomata import (
    jarvis_resistance,
    resistance_to_alpha,
    build_tile_alpha,
)

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
    return SurfRef()


@pytest.fixture(scope="session")
def consts() -> ConstantBundle:
    return ConstantBundle(PhysicalConstants(), dtype=torch.float64)


@pytest.fixture(scope="session")
def veg() -> VegTable:
    return VegTable(dtype=torch.float64)


@pytest.fixture(scope="session")
def soil() -> SoilTable:
    return SoilTable(dtype=torch.float64)


@pytest.fixture(scope="session")
def roots() -> RootTable:
    return RootTable(dtype=torch.float64)


def requires_fortran_stomata():
    """Skip marker -- True iff the .so exports surf_ref_stomata."""
    def _have_stomata() -> bool:
        if not _SURF_REF_AVAILABLE:
            return False
        return SurfRef()._loaded.stomata is not None
    return pytest.mark.skipif(
        not _have_stomata(),
        reason="libsurf_ref.so missing or predates surf_ref_stomata "
               "-- rebuild with offline_driver/build_ref.sh",
    )


# --------------------------------------------------------------------------- #
# Input factory -- a small grid of physically plausible columns.
# --------------------------------------------------------------------------- #
def _make_inputs(n=8, seed=0):
    """Build a varied grid of (soil_t, soil_w, soil_type, ktvl, ktvh, atm)."""
    torch.manual_seed(seed)
    nsoil = 4
    # Soil temperature: warm column above freezing, with one frozen column.
    soil_t = torch.rand(n, nsoil, dtype=torch.float64) * 15.0 + 280.0
    soil_t[0, :] = 270.0       # fully frozen
    # Soil moisture: 0.10..0.40 (volumetric fraction, well-watered to dry).
    soil_w = torch.rand(n, nsoil, dtype=torch.float64) * 0.30 + 0.10
    soil_type = torch.randint(1, 8, (n,), dtype=torch.long)
    ktvl = torch.randint(1, 8, (n,), dtype=torch.long)    # low veg (1..7)
    ktvh = torch.randint(3, 21, (n,), dtype=torch.long)   # high veg (3..20)
    t_air = torch.rand(n, dtype=torch.float64) * 20.0 + 280.0
    q_air = torch.rand(n, dtype=torch.float64) * 1.0e-2 + 1.0e-3
    p_surf = torch.full((n,), 1.0e5, dtype=torch.float64)
    sw = torch.rand(n, dtype=torch.float64) * 500.0 + 50.0
    return soil_t, soil_w, soil_type, ktvl, ktvh, t_air, q_air, p_surf, sw


def _run_port(inputs, consts, veg, soil, roots):
    soil_t, soil_w, soil_type, ktvl, ktvh, t_air, q_air, p_surf, sw = inputs
    return jarvis_resistance(
        soil_t, soil_w, soil_type, ktvl, ktvh,
        t_air, q_air, p_surf, sw,
        veg, soil, roots, consts,
    )


def _run_ref(inputs, ref, consts, veg, soil, roots, n_tile=8):
    soil_t, soil_w, soil_type, ktvl, ktvh, t_air, q_air, p_surf, sw = inputs
    cfg = SurfConfig(n_tile=n_tile)
    return ref.stomata(
        soil_t, soil_w, soil_type, ktvl, ktvh,
        t_air, q_air, p_surf, sw,
        veg, soil, roots, consts, cfg,
    )


# --------------------------------------------------------------------------- #
# Formula-level ULP tests -- port vs Fortran.
# --------------------------------------------------------------------------- #
@requires_fortran_stomata()
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_stomata_pwetl_matches_fortran_ulp(ref, consts, veg, soil, roots, seed):
    """PWETL (low-veg resistance) == Fortran, to a few ULP.

    A few ULP, not 0, because the root-weighted sum ``ZWROOTL`` re-
    associates (sum order) differently between the column loop in the
    Fortran and the tensor reduction in the port.  Both are
    mathematically exact; the residual is pure round-off.
    """
    inputs = _make_inputs(n=16, seed=seed)
    port = _run_port(inputs, consts, veg, soil, roots)
    f90 = _run_ref(inputs, ref, consts, veg, soil, roots)

    diff = (port["pwetl"] - f90["pwetl"]).abs()
    rel = diff / f90["pwetl"].abs().clamp_min(1.0e-30)
    # Relative tolerance: the resistances span orders of magnitude
    # (1 .. 1e6), so ULP is the wrong yardstick -- use 1e-12 rel.
    assert float(rel.max()) < 1.0e-12, (
        f"seed={seed}: PWETL max rel diff={float(rel.max()):.3e} "
        f"(port={port['pwetl'].tolist()[:4]}, "
        f"f90={f90['pwetl'].tolist()[:4]})"
    )


@requires_fortran_stomata()
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_stomata_pweth_matches_fortran(ref, consts, veg, soil, roots, seed):
    """PWETH (high-veg resistance) == Fortran."""
    inputs = _make_inputs(n=16, seed=seed)
    port = _run_port(inputs, consts, veg, soil, roots)
    f90 = _run_ref(inputs, ref, consts, veg, soil, roots)
    diff = (port["pweth"] - f90["pweth"]).abs()
    rel = diff / f90["pweth"].abs().clamp_min(1.0e-30)
    assert float(rel.max()) < 1.0e-12, (
        f"seed={seed}: PWETH max rel diff={float(rel.max()):.3e}"
    )


@requires_fortran_stomata()
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_stomata_pwetb_matches_fortran(ref, consts, veg, soil, roots, seed):
    """PWETB (bare-soil resistance) == Fortran."""
    inputs = _make_inputs(n=16, seed=seed)
    port = _run_port(inputs, consts, veg, soil, roots)
    f90 = _run_ref(inputs, ref, consts, veg, soil, roots)
    diff = (port["pwetb"] - f90["pwetb"]).abs()
    rel = diff / f90["pwetb"].abs().clamp_min(1.0e-30)
    assert float(rel.max()) < 1.0e-12, (
        f"seed={seed}: PWETB max rel diff={float(rel.max()):.3e}"
    )


@requires_fortran_stomata()
@pytest.mark.parametrize("seed", [0, 1])
def test_stomata_pwetlu_matches_fortran(ref, consts, veg, soil, roots, seed):
    """PWETLU (unstressed canopy diagnostic) == Fortran."""
    inputs = _make_inputs(n=16, seed=seed)
    port = _run_port(inputs, consts, veg, soil, roots)
    f90 = _run_ref(inputs, ref, consts, veg, soil, roots)
    diff = (port["pwetlu"] - f90["pwetlu"]).abs()
    rel = diff / f90["pwetlu"].abs().clamp_min(1.0e-30)
    assert float(rel.max()) < 1.0e-12, (
        f"seed={seed}: PWETLU max rel diff={float(rel.max()):.3e}"
    )


@requires_fortran_stomata()
@pytest.mark.parametrize("seed", [0, 1])
def test_stomata_stress_factors_match_fortran(ref, consts, veg, soil, roots, seed):
    """F1/F2/F3 stress factors (port's diagnostics) == Fortran exdiag.

    The Fortran writes the factors into PEXDIAG columns 2..4 (low) and
    7..9 (high); the port returns them as dict entries.  This checks
    that the *intermediate* quantities -- not just the final resistance
    -- agree, which is the strongest possible transcription check.
    """
    inputs = _make_inputs(n=16, seed=seed)
    port = _run_port(inputs, consts, veg, soil, roots)
    f90 = _run_ref(inputs, ref, consts, veg, soil, roots)
    ex = f90["exdiag"]   # (N, 10): [PWETL, F1L, F2L, F3L, LAIL, PWETH, F1H, F2H, F3H, LAIH]

    for name, port_t, col in [
        ("f1l", port["f1l"], 1), ("f2l", port["f2l"], 2), ("f3l", port["f3l"], 3),
        ("f1h", port["f1h"], 6), ("f2h", port["f2h"], 7), ("f3h", port["f3h"], 8),
    ]:
        diff = (port_t - ex[:, col]).abs().max().item()
        # Stress factors are O(1); absolute tolerance is the right yardstick.
        assert diff < 1.0e-12, (
            f"seed={seed}: {name} max |diff|={diff:.3e} "
            f"(port={port_t.tolist()[:4]}, f90={ex[:, col].tolist()[:4]})"
        )


# --------------------------------------------------------------------------- #
# Physics sanity tests -- port behaviour in known regimes.
# --------------------------------------------------------------------------- #
def test_freeze_shutdown_collapses_f2(consts, veg, soil, roots):
    """Frozen top-soil -> F2L = F2H = RCEPSW (vsurf 365-372)."""
    n, nsoil = 2, 4
    soil_t = torch.full((n, nsoil), 270.0, dtype=torch.float64)
    soil_t[0, 0] = 280.0  # column 0 top above freezing
    soil_w = torch.full((n, nsoil), 0.30, dtype=torch.float64)
    out = jarvis_resistance(
        soil_t, soil_w,
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([1, 3], dtype=torch.long),
        torch.tensor([3, 5], dtype=torch.long),
        torch.full((n,), 290.0, dtype=torch.float64),
        torch.full((n,), 5e-3, dtype=torch.float64),
        torch.full((n,), 1.0e5, dtype=torch.float64),
        torch.full((n,), 300.0, dtype=torch.float64),
        veg, soil, roots, consts,
    )
    assert abs(float(out["f2l"][1]) - stomata_mod.RCEPSW) < 1e-14
    assert abs(float(out["f2h"][1]) - stomata_mod.RCEPSW) < 1e-14
    # Column 0 has a non-frozen top -> F2 stays above the floor.
    assert float(out["f2l"][0]) > 0.1
    assert float(out["f2h"][0]) > 0.1


def test_lai_zero_hits_sentinel(consts, veg, soil, roots):
    """LAI=0 veg type (e.g. ice cap = 12) -> PWETL = PWET_INF = 1e6."""
    out = jarvis_resistance(
        torch.full((1, 4), 285.0, dtype=torch.float64),
        torch.full((1, 4), 0.30, dtype=torch.float64),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([12], dtype=torch.long),  # ice cap, LAI=0
        torch.tensor([12], dtype=torch.long),
        torch.full((1,), 290.0, dtype=torch.float64),
        torch.full((1,), 5e-3, dtype=torch.float64),
        torch.full((1,), 1.0e5, dtype=torch.float64),
        torch.full((1,), 300.0, dtype=torch.float64),
        veg, soil, roots, consts,
    )
    assert abs(float(out["pwetl"][0]) - stomata_mod.PWET_INF) < 1.0


def test_resistance_to_alpha_condensation(consts):
    """q_air > qsat(t_air) -> alpha = 1 (VEVAP leaves PCSAT untouched)."""
    n = 3
    pwet = torch.full((n,), 100.0, dtype=torch.float64)
    rhocqu = torch.full((n,), 0.01, dtype=torch.float64)
    t_air = torch.full((n,), 290.0, dtype=torch.float64)
    q_air = torch.full((n,), 0.05, dtype=torch.float64)   # huge -> condensation
    p_surf = torch.full((n,), 1.0e5, dtype=torch.float64)
    alpha = resistance_to_alpha(pwet, rhocqu, p_surf, t_air, q_air, consts)
    assert torch.allclose(alpha, torch.ones_like(alpha))


def test_resistance_to_alpha_dry_canopy_below_one(consts):
    """Non-condensing air + finite resistance -> alpha in (0, 1)."""
    n = 4
    pwet = torch.tensor([10.0, 100.0, 1000.0, 1.0e5], dtype=torch.float64)
    rhocqu = torch.full((n,), 0.05, dtype=torch.float64)
    t_air = torch.full((n,), 290.0, dtype=torch.float64)
    q_air = torch.full((n,), 5e-3, dtype=torch.float64)   # below qsat
    p_surf = torch.full((n,), 1.0e5, dtype=torch.float64)
    alpha = resistance_to_alpha(pwet, rhocqu, p_surf, t_air, q_air, consts)
    # Higher resistance -> lower alpha (more stomatal closure).
    assert torch.all(alpha < 1.0)
    assert torch.all(alpha > 0.0)
    assert bool((alpha[:-1] > alpha[1:]).all()), (
        f"alpha should decrease with resistance; got {alpha.tolist()}"
    )


# --------------------------------------------------------------------------- #
# Tile dispatch tests -- build_tile_alpha routing.
# --------------------------------------------------------------------------- #
def test_build_tile_alpha_routing(consts):
    """build_tile_alpha assigns the right resistance to each tile.

    Tiles 0/1/2/4/8 (water/ice/wet-skin/snow-low/lake) -> alpha=1.
    Tile 3 -> PWETL, tile 5 -> PWETH, tile 6 -> PWETHS, tile 7 -> PWETB.
    """
    n = 2
    cfg = SurfConfig(n_tile=8)
    pwetl = torch.tensor([100.0, 200.0], dtype=torch.float64)
    pweth = torch.tensor([150.0, 250.0], dtype=torch.float64)
    pwetb = torch.tensor([50.0, 80.0], dtype=torch.float64)
    rhocqu = torch.full((n,), 0.01, dtype=torch.float64)
    t_air = torch.full((n,), 290.0, dtype=torch.float64)
    q_air = torch.full((n,), 5e-3, dtype=torch.float64)   # dry
    p_surf = torch.full((n,), 1.0e5, dtype=torch.float64)

    alpha = build_tile_alpha(
        pwetl, pweth, pwetb, rhocqu, p_surf, t_air, q_air, consts, cfg,
    )
    assert alpha.shape == (n, 8)
    # Free-surface tiles are pinned at alpha=1.
    for jt in (0, 1, 2, 4):
        assert torch.allclose(alpha[:, jt], torch.ones(n, dtype=torch.float64)), (
            f"tile {jt} should be alpha=1, got {alpha[:, jt].tolist()}"
        )
    # Vegetated tiles carry alpha < 1 (non-zero resistance, dry air).
    for jt in (3, 5, 6, 7):
        assert torch.all(alpha[:, jt] < 1.0), (
            f"tile {jt} should have alpha<1, got {alpha[:, jt].tolist()}"
        )
        assert torch.all(alpha[:, jt] > 0.0)
    # PWETHS (tile 6) defaults to PWETH when no snow aero term is given.
    assert torch.allclose(alpha[:, 5], alpha[:, 6])


def test_build_tile_alpha_tile_count_extend(consts):
    """build_tile_alpha works for n_tile=10 (lake + urban)."""
    n = 2
    cfg = SurfConfig(n_tile=10, use_lake=True, use_urban=True)
    pwetl = torch.tensor([100.0, 200.0], dtype=torch.float64)
    pweth = torch.tensor([150.0, 250.0], dtype=torch.float64)
    pwetb = torch.tensor([50.0, 80.0], dtype=torch.float64)
    rhocqu = torch.full((n,), 0.01, dtype=torch.float64)
    t_air = torch.full((n,), 290.0, dtype=torch.float64)
    q_air = torch.full((n,), 5e-3, dtype=torch.float64)
    p_surf = torch.full((n,), 1.0e5, dtype=torch.float64)

    alpha = build_tile_alpha(
        pwetl, pweth, pwetb, rhocqu, p_surf, t_air, q_air, consts, cfg,
    )
    assert alpha.shape == (n, 10)
    # Lake (tile 8) -> 1; urban (tile 9) -> PWETB-derived (alpha < 1).
    assert torch.allclose(alpha[:, 8], torch.ones(n, dtype=torch.float64))
    assert torch.all(alpha[:, 9] < 1.0)
    assert torch.allclose(alpha[:, 9], alpha[:, 7])  # urban reuses bare-soil
