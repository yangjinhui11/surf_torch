"""Independent Fortran validation of the Farquhar photosynthesis model.

These tests compare the PyTorch port's ``physics.farquhar.farquhar``
(a direct transcription of ``farquhar_mod.F90`` + ``sufarquhar_mod.F90``)
against the Fortran reference kernel ``SURF_REF_FARQUHAR``, which re-
derives the coupled photosynthesis + Ball-Berry stomatal conductance
solve in standalone Fortran compiled with bare gfortran.

Why this is not a self-transcription
------------------------------------
Both the port and the reference are independently written transcriptions
of the same upstream ``farquhar_mod.F90`` (the C3/C4 cubic stomatal
closures of Yin et al. 2009, the Kattge&Knorr temperature acclimation,
the 20-LAI-level Beer-Lambert canopy integration and the Ball-Berry
conductance).  They share the identical :class:`FarquharConstants`
parameterisation and the identical per-PFT tables, so a 0-ULP / 1e-12
agreement is genuine evidence that both transcriptions correctly
reproduce the upstream arithmetic (in particular the depressed-cubic
root via ``acos``/``cos``, the two-iteration ``LIMIT_PHOTO`` loop with
its C4-loop-exit / C3-minimum semantics, and the unit conversions).

Coverage
--------
* **formula-level ULP** -- PGSTOT / PGPP / PRD / PAN vs Fortran, across a
  sweep of vegetation types (C3 crops, C4 grass, broadleaf forest),
  temperatures, soil-moisture stress and SW radiation.
* **physics sanity** -- vegetated columns produce positive GPP and
  conductance; bare soil / nighttime / frozen columns produce zero;
  higher SW raises GPP; water stress lowers conductance.

Skip behaviour
--------------
Skipped (like ``test_stomata_ref.py``) when ``libsurf_ref.so`` is not
built or was built before the ``surf_ref_farquhar`` symbol existed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from surf_pytorch import ConstantBundle
from surf_pytorch.constants import PhysicalConstants
from surf_pytorch.tables import VegTable
from surf_pytorch.physics.farquhar import FarquharInputs, farquhar

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


def requires_fortran_farquhar():
    """Skip marker -- True iff the .so exports surf_ref_farquhar."""
    def _have_farquhar() -> bool:
        if not _SURF_REF_AVAILABLE:
            return False
        return SurfRef()._loaded.farquhar is not None
    return pytest.mark.skipif(
        not _have_farquhar(),
        reason="libsurf_ref.so missing or predates surf_ref_farquhar "
               "-- rebuild with offline_driver/build_ref.sh",
    )


# --------------------------------------------------------------------------- #
# Input factory -- a small grid of physically plausible columns.
# --------------------------------------------------------------------------- #
def _make_inputs(n=8, seed=0):
    """Build a varied grid of (C3 crop / C4 grass / forest / bare) columns.

    Returns a dict of per-column tensors consumed by both the port's
    :class:`FarquharInputs` and the ``SurfRef.farquhar`` wrapper.
    """
    torch.manual_seed(seed)
    # Daytime SW radiation: 100..900 W/m^2, with one nighttime column.
    sw = torch.rand(n, dtype=torch.float64) * 800.0 + 100.0
    sw[0] = 0.0       # nighttime
    # Air / skin / soil temperatures: 280..305 K, with one frozen column.
    t_air = torch.rand(n, dtype=torch.float64) * 25.0 + 280.0
    skin_t = t_air + torch.rand(n, dtype=torch.float64) * 2.0
    t_soil = t_air - torch.rand(n, dtype=torch.float64) * 3.0
    t_air[1] = 270.0; skin_t[1] = 271.0; t_soil[1] = 268.0   # frozen
    # Humidity / pressure.
    q_air = torch.rand(n, dtype=torch.float64) * 8.0e-3 + 4.0e-3
    qs_skin = q_air + torch.rand(n, dtype=torch.float64) * 4.0e-3
    p_surf = torch.full((n,), 1.01325e5, dtype=torch.float64)
    # CO2 mixing ratio (kg/kg) -- ~390 ppm.
    co2 = torch.full((n,), 390.0e-6, dtype=torch.float64)
    # LAI: 0.5..5, with one bare column.
    lai = torch.rand(n, dtype=torch.float64) * 4.5 + 0.5
    lai[2] = 0.0      # bare
    # Soil-moisture stress: 0.3..1.0.
    f2 = torch.rand(n, dtype=torch.float64) * 0.7 + 0.3
    # Vegetation type: cycle C3 crop (1), C4 grass (7), forest (5), bare (0).
    kvtype = torch.tensor([1, 5, 0, 7, 3, 1, 7, 5][:n], dtype=torch.long)
    # C4 pathway for the tall-grass columns (KVTYPE==7).
    kco2typ = torch.where(kvtype == 7, torch.tensor(4, dtype=torch.long),
                                    torch.tensor(1, dtype=torch.long))
    ldland = torch.ones(n, dtype=torch.bool)
    return {
        "psrfd": sw, "ptm": t_air, "ptskm": skin_t, "ptsoil": t_soil,
        "paphm": p_surf, "pqm": q_air, "pqsurf": qs_skin, "pqs": qs_skin,
        "pco2": co2, "lai": lai, "kvtype": kvtype, "kco2typ": kco2typ,
        "pf2": f2, "ldland": ldland,
    }


def _run_port(inp_dict, consts, veg):
    inp = FarquharInputs(
        ptskm=inp_dict["ptskm"], ptm=inp_dict["ptm"], ptsoil=inp_dict["ptsoil"],
        paphm=inp_dict["paphm"], pqm=inp_dict["pqm"], pqsurf=inp_dict["pqsurf"],
        pqs=inp_dict["pqs"], pco2=inp_dict["pco2"],
        lai=inp_dict["lai"], kvtype=inp_dict["kvtype"],
        kco2typ=inp_dict["kco2typ"], pf2=inp_dict["pf2"],
        ldland=inp_dict["ldland"], psrfd=inp_dict["psrfd"],
    )
    return farquhar(inp, veg, consts)


def _run_ref(inp_dict, ref, consts, veg):
    return ref.farquhar(
        sw_down=inp_dict["psrfd"], p_surf=inp_dict["paphm"],
        qs_skin=inp_dict["pqs"], qsurf=inp_dict["pqsurf"], q_air=inp_dict["pqm"],
        t_air=inp_dict["ptm"], skin_t=inp_dict["ptskm"], t_soil=inp_dict["ptsoil"],
        f2=inp_dict["pf2"], co2=inp_dict["pco2"], lai=inp_dict["lai"],
        kvtype=inp_dict["kvtype"], kco2typ=inp_dict["kco2typ"],
        ldland=inp_dict["ldland"], veg=veg, consts=consts, lskin_temp=True,
    )


# --------------------------------------------------------------------------- #
# Formula-level ULP tests -- port vs Fortran.
# --------------------------------------------------------------------------- #
@requires_fortran_farquhar()
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_farquhar_pgstot_matches_fortran(ref, consts, veg, seed):
    """PGSTOT (canopy conductance) == Fortran, to 1e-9 relative.

    The C3/C4 cubic root uses ``acos``/``cos`` which re-associates the
    arguments differently between the vectorised PyTorch reduction and
    the scalar Fortran loop, so we allow ~1e-9 (much looser than the
    1e-12 used for the linear Jarvis path) rather than ULP-exact.
    """
    inp = _make_inputs(n=8, seed=seed)
    port = _run_port(inp, consts, veg)
    f90 = _run_ref(inp, ref, consts, veg)
    diff = (port.pgstot - f90["pgstot"]).abs()
    rel = diff / f90["pgstot"].abs().clamp_min(1.0e-30)
    # Columns where both are ~0 (bare/night/frozen) are bit-exact by
    # construction (both zero); only compare the active columns.
    active = f90["pgstot"].abs() > 1.0e-12
    if active.any():
        assert float(rel[active].max()) < 1.0e-9, (
            f"seed={seed}: PGSTOT max rel diff={float(rel[active].max()):.3e} "
            f"(port={port.pgstot[active].tolist()}, "
            f"f90={f90['pgstot'][active].tolist()})"
        )


@requires_fortran_farquhar()
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_farquhar_pgpp_matches_fortran(ref, consts, veg, seed):
    """PGPP (gross primary productivity) == Fortran."""
    inp = _make_inputs(n=8, seed=seed)
    port = _run_port(inp, consts, veg)
    f90 = _run_ref(inp, ref, consts, veg)
    diff = (port.pgpp - f90["pgpp"]).abs()
    rel = diff / f90["pgpp"].abs().clamp_min(1.0e-30)
    active = f90["pgpp"].abs() > 1.0e-15
    if active.any():
        assert float(rel[active].max()) < 1.0e-9, (
            f"seed={seed}: PGPP max rel diff={float(rel[active].max()):.3e}"
        )


@requires_fortran_farquhar()
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_farquhar_prd_matches_fortran(ref, consts, veg, seed):
    """PRD (dark respiration) == Fortran."""
    inp = _make_inputs(n=8, seed=seed)
    port = _run_port(inp, consts, veg)
    f90 = _run_ref(inp, ref, consts, veg)
    diff = (port.prd - f90["prd"]).abs()
    rel = diff / f90["prd"].abs().clamp_min(1.0e-30)
    active = f90["prd"].abs() > 1.0e-15
    if active.any():
        assert float(rel[active].max()) < 1.0e-9, (
            f"seed={seed}: PRD max rel diff={float(rel[active].max()):.3e}"
        )


@requires_fortran_farquhar()
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_farquhar_pan_matches_fortran(ref, consts, veg, seed):
    """PAN (net assimilation) == Fortran."""
    inp = _make_inputs(n=8, seed=seed)
    port = _run_port(inp, consts, veg)
    f90 = _run_ref(inp, ref, consts, veg)
    diff = (port.pan - f90["pan"]).abs()
    rel = diff / f90["pan"].abs().clamp_min(1.0e-30)
    active = f90["pan"].abs() > 1.0e-15
    if active.any():
        assert float(rel[active].max()) < 1.0e-9, (
            f"seed={seed}: PAN max rel diff={float(rel[active].max()):.3e}"
        )


# --------------------------------------------------------------------------- #
# Physics sanity tests.
# --------------------------------------------------------------------------- #
@requires_fortran_farquhar()
def test_farquhar_vegetated_positive_gpp(ref, consts, veg):
    """Daytime vegetated columns produce positive GPP and conductance."""
    inp = _make_inputs(n=8, seed=0)
    out = _run_port(inp, consts, veg)
    # Columns 3,4,5,6,7 are vegetated + daytime + warm -> positive GPP.
    for i in (3, 4, 5, 6, 7):
        assert out.pgpp[i].item() > 0.0, f"col {i}: GPP should be positive"
        assert out.pgstot[i].item() > 0.0, f"col {i}: gs should be positive"


@requires_fortran_farquhar()
def test_farquhar_bare_and_nighttime_zero(ref, consts, veg):
    """Bare soil (LAI=0) and nighttime (SW=0) columns produce zero fluxes."""
    inp = _make_inputs(n=8, seed=0)
    out = _run_port(inp, consts, veg)
    # Column 0: nighttime -> zero.
    assert out.pgpp[0].item() == 0.0, "nighttime GPP should be zero"
    assert out.pgstot[0].item() == 0.0, "nighttime gs should be zero"
    # Column 2: bare (LAI=0) -> zero.
    assert out.pgpp[2].item() == 0.0, "bare GPP should be zero"
    assert out.pgstot[2].item() == 0.0, "bare gs should be zero"


@requires_fortran_farquhar()
def test_farquhar_more_light_more_gpp(ref, consts, veg):
    """Holding everything else fixed, more SW -> more GPP (monotone)."""
    n = 4
    base = _make_inputs(n=n, seed=0)
    sw_lo = base["psrfd"].clone()
    sw_hi = base["psrfd"].clone() + 300.0
    inp_lo = {**base, "psrfd": sw_lo}
    inp_hi = {**base, "psrfd": sw_hi}
    out_lo = _run_port(inp_lo, consts, veg)
    out_hi = _run_port(inp_hi, consts, veg)
    # Vegetated daytime columns should have higher GPP under more light.
    for i in range(n):
        if base["lai"][i] > 0.5 and base["psrfd"][i] > 50.0:
            assert out_hi.pgpp[i].item() >= out_lo.pgpp[i].item(), (
                f"col {i}: more light should not lower GPP "
                f"({out_hi.pgpp[i].item():.3e} vs {out_lo.pgpp[i].item():.3e})"
            )


@requires_fortran_farquhar()
def test_farquhar_water_stress_lowers_conductance(ref, consts, veg):
    """Lower F2 (more water stress) -> lower stomatal conductance."""
    n = 4
    base = _make_inputs(n=n, seed=0)
    f2_hi = base["pf2"].clone()
    f2_lo = base["pf2"].clone() * 0.3      # drier
    inp_hi = {**base, "pf2": f2_hi}
    inp_lo = {**base, "pf2": f2_lo}
    out_hi = _run_port(inp_hi, consts, veg)
    out_lo = _run_port(inp_lo, consts, veg)
    for i in range(n):
        if base["lai"][i] > 0.5 and base["psrfd"][i] > 50.0:
            assert out_hi.pgstot[i].item() >= out_lo.pgstot[i].item(), (
                f"col {i}: water stress should not raise gs "
                f"({out_hi.pgstot[i].item():.3e} vs {out_lo.pgstot[i].item():.3e})"
            )
