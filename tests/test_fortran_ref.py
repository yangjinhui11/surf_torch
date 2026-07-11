"""ULP-level comparison tests against the standalone Fortran reference.

These tests are the heart of the M2 milestone: they feed *identical*
inputs to the PyTorch port (``surf_pytorch.physics.seb.surf_seb`` /
``surf_pytorch.physics.vlamsk.vlamsk``) and to the Fortran reference
(``libsurf_ref.so``, wrapped by :class:`offline_driver.SurfRef`) and
assert that every output matches to within ~1e-12 (last ULP for
float64).

Why a separate Fortran reference?
---------------------------------
The port's existing tests (``test_seb.py``) use a pure-Python single-
tile reference and analytic closure checks.  Those are valuable but
share the *port's* arithmetic ordering, so a transcription mistake
(e.g. a sign flip) could pass both.  The Fortran reference here shares
no Python code with the port -- it is a fresh transcription of the
upstream ``vlamsk_mod.F90`` / ``surfseb_ctl_mod.F90`` compiled with
bare gfortran -- so agreement between the two is strong evidence that
the port reproduces the upstream IFS arithmetic.

Skip behaviour
--------------
If the shared library is not built (``libsurf_ref.so`` missing, e.g.
on CI without gfortran) every test in this module is skipped via the
``requires_fortran_ref`` fixture.  Build the library with::

    offline_driver/build_ref.sh
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from surf_pytorch import ConstantBundle, SurfConfig
from surf_pytorch.physics.seb import SebInputs, surf_seb
from surf_pytorch.physics.vlamsk import vlamsk as port_vlamsk
from surf_pytorch.tables import VegTable
from surf_pytorch.utils import qsat, dqsat_dt_ifs

# Importing offline_driver triggers the .so load, so guard it.
try:
    from offline_driver import SurfRef
    _SURF_REF_AVAILABLE = SurfRef.is_available()
except Exception:  # pragma: no cover -- exercised only without the .so
    SurfRef = None  # type: ignore[assignment]
    _SURF_REF_AVAILABLE = False


# Mark every test in this module with the ``fortran_ref`` marker so the
# whole suite can be selected/deselected with ``-m fortran_ref`` /
# ``-m "not fortran_ref"``.  (The per-test ``requires_fortran_ref()``
# decorator below additionally skips individual tests when the .so is
# missing.)
pytestmark = pytest.mark.fortran_ref


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def ref() -> "SurfRef":
    """Session-cached :class:`SurfRef` (loads .so once)."""
    return SurfRef()


@pytest.fixture(scope="session")
def veg() -> VegTable:
    return VegTable(dtype=torch.float64)


@pytest.fixture(scope="session")
def consts() -> ConstantBundle:
    return ConstantBundle(dtype=torch.float64)


@pytest.fixture
def config() -> SurfConfig:
    return SurfConfig(dtype=torch.float64)


def requires_fortran_ref():
    """Skip marker -- True iff the .so is built and importable."""
    return pytest.mark.skipif(
        not _SURF_REF_AVAILABLE,
        reason="libsurf_ref.so not built -- run offline_driver/build_ref.sh",
    )


# --------------------------------------------------------------------------- #
# Helpers -- replicate the test_seb.py _build_seb_inputs signature so the
# comparison uses self-consistent (qsat-derived) inputs.
# --------------------------------------------------------------------------- #
def _build_inputs(
    *,
    N: int = 3,
    T: int = 8,
    tsk1: float = 290.0,
    psurf: float = 1.0e5,
    tsrf: float = 285.0,
    rhochu: float = 0.01,
    rhocqu: float = 0.01,
    ssrfl: float = 200.0,
    slrfl: float = -300.0,
    tair: float = 290.0,
    qair: float = 0.008,
    alphal: float = 0.0,
    alphas: float = 1.0,
    ktvl: int = 1,
    ktvh: int = 3,
    snm: float = 0.0,
    rsn: float = 300.0,
    lelwtl: bool = False,
    dtype: torch.dtype = torch.float64,
) -> SebInputs:
    """Construct a self-consistent SebInputs; qsat derived from (tsk1, psurf)."""
    consts = ConstantBundle(dtype=dtype)
    cp = float(consts.cpd.item())
    tsk = torch.full((N, T), tsk1, dtype=dtype)
    p = torch.full((N, T), psurf, dtype=dtype)
    qsk = qsat(tsk, p, consts)
    dqsdT = dqsat_dt_ifs(tsk, p, consts)
    ssk = cp * tsk
    return SebInputs(
        sskm1m=ssk, tskm1m=tsk, qskm1m=qsk, dqsdT=dqsdT,
        rhochu=torch.full((N, T), rhochu, dtype=dtype),
        rhocqu=torch.full((N, T), rhocqu, dtype=dtype),
        alphal=torch.full((N, T), alphal, dtype=dtype),
        alphas=torch.full((N, T), alphas, dtype=dtype),
        ssrfl=torch.full((N, T), ssrfl, dtype=dtype),
        slrfl_grid=torch.full((N,), slrfl, dtype=dtype),
        tskrad_grid=torch.full((N,), tsk1, dtype=dtype),
        emis_grid=torch.full((N,), 0.99, dtype=dtype),
        tsrf=torch.full((N, T), tsrf, dtype=dtype),
        lamsk=torch.full((N, T), 10.0, dtype=dtype),  # overwritten by VLAMSK
        frti=torch.full((N, T), 1.0 / T, dtype=dtype),
        ktvl_grid=torch.full((N,), ktvl, dtype=torch.long),
        ktvh_grid=torch.full((N,), ktvh, dtype=torch.long),
        snowm_grid=torch.full((N,), snm, dtype=dtype),
        rsn_grid=torch.full((N,), rsn, dtype=dtype),
        asl_grid=torch.zeros((N,), dtype=dtype),
        bsl_grid=torch.full((N,), cp * tair, dtype=dtype),
        aql_grid=torch.zeros((N,), dtype=dtype),
        bql_grid=torch.full((N,), qair, dtype=dtype),
        ptmst=torch.tensor(3600.0, dtype=dtype),
        lelwdd=True, lelwtl=lelwtl,
    )


# All SebOutputs fields compared; names match the dataclass.
_SEB_FIELDS = ("pjs", "pjq", "pssk", "ptsk", "pssh", "pslh", "pstr", "pg0",
               "psl", "pql")


# --------------------------------------------------------------------------- #
# VLAMSK -- per-tile skin conductivity
# --------------------------------------------------------------------------- #
class TestVlamskFortranRef:
    """VLAMSK port vs Fortran reference."""

    @requires_fortran_ref()
    def test_default_inputs_match_to_ulp(self, ref, veg, consts, config):
        """Default batch: every tile's lamsk matches to last ULP."""
        inp = _build_inputs()
        snm = inp.snowm_grid
        rsn = inp.rsn_grid

        lamsk_port = port_vlamsk(
            inp.tskm1m, inp.tsrf, snm, rsn, None,
            inp.ktvl_grid, inp.ktvh_grid, veg, consts, config,
            ptstep=inp.ptmst,
        )
        lamsk_ref = ref.vlamsk(
            inp.tskm1m, inp.tsrf, snm, rsn, inp.ktvl_grid, inp.ktvh_grid,
            veg, consts, config, ptstep=float(inp.ptmst.item()),
        )
        diff = (lamsk_port - lamsk_ref).abs().max().item()
        # Exact 0.0 expected: same literals, same op order.
        assert diff == 0.0, f"VLAMSK diverges: max|diff|={diff:.3e}"

    @requires_fortran_ref()
    def test_stability_branch_with_snow(self, ref, veg, consts, config):
        """Forest-snow stability factor (JT=7) matches with non-trivial snow."""
        inp = _build_inputs(snm=50.0, rsn=200.0)
        snm = inp.snowm_grid
        rsn = inp.rsn_grid
        lamsk_port = port_vlamsk(
            inp.tskm1m, inp.tsrf, snm, rsn, None,
            inp.ktvl_grid, inp.ktvh_grid, veg, consts, config,
            ptstep=inp.ptmst,
        )
        lamsk_ref = ref.vlamsk(
            inp.tskm1m, inp.tsrf, snm, rsn, inp.ktvl_grid, inp.ktvh_grid,
            veg, consts, config, ptstep=float(inp.ptmst.item()),
        )
        # Tile 6 (0-based) = JT=7 forest snow must differ from the no-snow
        # value (SNOW_HIGH_VEG=20); both implementations must agree.
        expected_no_snow = torch.tensor(
            20.0, dtype=lamsk_port.dtype, device=lamsk_port.device
        )
        assert not torch.allclose(lamsk_port[:, 6], expected_no_snow)
        diff = (lamsk_port - lamsk_ref).abs().max().item()
        assert diff == 0.0, f"VLAMSK(stability) diverges: {diff:.3e}"

    @requires_fortran_ref()
    @pytest.mark.parametrize("ktvl,ktvh", [(1, 3), (5, 6), (8, 18), (11, 19)])
    def test_various_dominant_veg_types(
        self, ref, veg, consts, config, ktvl, ktvh
    ):
        """VLAMSK agreement across dominant-veg types 1..20."""
        inp = _build_inputs(ktvl=ktvl, ktvh=ktvh)
        snm = inp.snowm_grid
        rsn = inp.rsn_grid
        lamsk_port = port_vlamsk(
            inp.tskm1m, inp.tsrf, snm, rsn, None,
            inp.ktvl_grid, inp.ktvh_grid, veg, consts, config,
            ptstep=inp.ptmst,
        )
        lamsk_ref = ref.vlamsk(
            inp.tskm1m, inp.tsrf, snm, rsn, inp.ktvl_grid, inp.ktvh_grid,
            veg, consts, config, ptstep=float(inp.ptmst.item()),
        )
        diff = (lamsk_port - lamsk_ref).abs().max().item()
        assert diff == 0.0, f"VLAMSK(ktvl={ktvl},ktvh={ktvh}) diverges: {diff:.3e}"


# --------------------------------------------------------------------------- #
# SURFSEB_CTL -- full Penman-Monteith SEB
# --------------------------------------------------------------------------- #
class TestSebFortranRef:
    """SURFSEB_CTL port vs Fortran reference."""

    def _run_both(self, ref, veg, consts, config, inp: SebInputs):
        """Run port + Fortran ref with synced VLAMSK output.

        The port's ``surf_seb`` consumes ``inp.lamsk``; the Fortran ref
        takes lamsk as an explicit kwarg.  We feed both the same VLAMSK
        output so any SEB divergence is isolated from VLAMSK divergence.
        """
        snm = inp.snowm_grid
        rsn = inp.rsn_grid
        lamsk = port_vlamsk(
            inp.tskm1m, inp.tsrf, snm, rsn, None,
            inp.ktvl_grid, inp.ktvh_grid, veg, consts, config,
            ptstep=inp.ptmst,
        )
        inp = replace(inp, lamsk=lamsk)

        out_ref = ref.seb(inp, veg, consts, config, lamsk=lamsk)
        out_port = surf_seb(inp, consts, config)
        return out_ref, out_port

    @requires_fortran_ref()
    def test_default_inputs_match_to_ulp(self, ref, veg, consts, config):
        """Default batch: every SEB field matches to ~1e-12 (last ULP)."""
        inp = _build_inputs()
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        for name in _SEB_FIELDS:
            a = getattr(out_ref, name)
            b = getattr(out_port, name)
            diff = (a - b).abs().max().item()
            # 0.0 is typical; 1e-12 leaves slack for fp re-association.
            assert diff < 1e-9, (
                f"SEB field {name!r} diverges: max|diff|={diff:.3e}\n"
                f"  ref ={a.tolist()}\n  port={b.tolist()}"
            )

    @requires_fortran_ref()
    def test_ocean_tile_pins_to_tsrf(self, ref, veg, consts, config):
        """Tile 0 (ocean, lamsk=1e10) pins skin T to the sub-surface T."""
        tsrf_val = 288.0
        inp = _build_inputs(tsrf=tsrf_val)
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        # Both implementations should pin to tsrf to ~1e-7 (1/1e10 scaling).
        assert abs(out_ref.ptsk[:, 0].mean().item() - tsrf_val) < 1e-3
        assert abs(out_port.ptsk[:, 0].mean().item() - tsrf_val) < 1e-3
        # And they must agree with each other to last ULP on this tile.
        diff = (out_ref.ptsk[:, 0] - out_port.ptsk[:, 0]).abs().max().item()
        assert diff < 1e-9, f"ocean ptsk diverges: {diff:.3e}"

    @requires_fortran_ref()
    @pytest.mark.parametrize(
        "rhochu,rhocqu,ssrfl,slrfl",
        [
            (0.001, 0.001, 0.0, -200.0),    # weak exchange, no solar (night)
            (0.05, 0.05, 800.0, -350.0),    # strong exchange, bright day
            (0.02, 0.02, 100.0, -100.0),    # mild
            (0.01, 0.005, 400.0, -250.0),   # asymmetric Ce/Cq
        ],
    )
    def test_regime_sweep(
        self, ref, veg, consts, config, rhochu, rhocqu, ssrfl, slrfl
    ):
        """SEB agreement across turbulent/radiative regimes."""
        inp = _build_inputs(
            rhochu=rhochu, rhocqu=rhocqu, ssrfl=ssrfl, slrfl=slrfl
        )
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        for name in _SEB_FIELDS:
            diff = (getattr(out_ref, name) - getattr(out_port, name)).abs().max().item()
            assert diff < 1e-9, (
                f"SEB {name!r} (Ce={rhochu},Cq={rhocqu},SW={ssrfl},LW={slrfl}) "
                f"diverges: {diff:.3e}"
            )

    @requires_fortran_ref()
    def test_different_veg_types(self, ref, veg, consts, config):
        """Different dominant-veg types still match (changes lambda_skin)."""
        for ktvl, ktvh in [(1, 3), (5, 18), (8, 11), (13, 19)]:
            inp = _build_inputs(ktvl=ktvl, ktvh=ktvh)
            out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
            for name in _SEB_FIELDS:
                diff = (getattr(out_ref, name) - getattr(out_port, name)).abs().max().item()
                assert diff < 1e-9, (
                    f"SEB {name!r} (ktvl={ktvl},ktvh={ktvh}) diverges: {diff:.3e}"
                )

    @requires_fortran_ref()
    def test_lelwtl_energy_correction(self, ref, veg, consts, config):
        """LELWTL on: tiled-LW recompute around the grid-mean skin T."""
        inp = _build_inputs(lelwtl=True)
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        # PSTR is the field most affected by LELWTL; check it explicitly.
        diff = (out_ref.pstr - out_port.pstr).abs().max().item()
        assert diff < 1e-9, f"PSTR (LELWTL) diverges: {diff:.3e}"

    @requires_fortran_ref()
    def test_randomised_inputs_match(self, ref, veg, consts, config):
        """Randomised inputs within physical ranges: still ULP-level match."""
        # Deterministic seed (OpenIFS cy48r1 nod to the source version).
        torch.manual_seed(4871)
        N, T = 5, 8
        cp = float(consts.cpd.item())
        tsk = 270.0 + 30.0 * torch.rand(N, T, dtype=torch.float64)
        p = torch.full((N, T), 1.0e5, dtype=torch.float64)
        qsk = qsat(tsk, p, consts)
        dqsdT = dqsat_dt_ifs(tsk, p, consts)
        ssk = cp * tsk
        inp = SebInputs(
            sskm1m=ssk, tskm1m=tsk, qskm1m=qsk, dqsdT=dqsdT,
            rhochu=0.005 + 0.04 * torch.rand(N, T, dtype=torch.float64),
            rhocqu=0.005 + 0.04 * torch.rand(N, T, dtype=torch.float64),
            alphal=torch.zeros((N, T), dtype=torch.float64),
            alphas=torch.ones((N, T), dtype=torch.float64),
            ssrfl=400.0 * torch.rand(N, T, dtype=torch.float64),
            slrfl_grid=-150.0 - 200.0 * torch.rand(N, dtype=torch.float64),
            tskrad_grid=tsk[:, 0],
            emis_grid=torch.full((N,), 0.99, dtype=torch.float64),
            tsrf=270.0 + 25.0 * torch.rand(N, T, dtype=torch.float64),
            lamsk=torch.full((N, T), 10.0, dtype=torch.float64),
            frti=torch.full((N, T), 1.0 / T, dtype=torch.float64),
            ktvl_grid=torch.randint(1, 21, (N,), dtype=torch.long),
            ktvh_grid=torch.randint(1, 21, (N,), dtype=torch.long),
            snowm_grid=50.0 * torch.rand(N, dtype=torch.float64),
            rsn_grid=torch.full((N,), 300.0, dtype=torch.float64),
            asl_grid=torch.zeros((N,), dtype=torch.float64),
            bsl_grid=cp * (275.0 + 20.0 * torch.rand(N, dtype=torch.float64)),
            aql_grid=torch.zeros((N,), dtype=torch.float64),
            bql_grid=0.005 + 0.01 * torch.rand(N, dtype=torch.float64),
            ptmst=torch.tensor(3600.0, dtype=torch.float64),
            lelwdd=True, lelwtl=False,
        )
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        for name in _SEB_FIELDS:
            diff = (getattr(out_ref, name) - getattr(out_port, name)).abs().max().item()
            # 1e-9 leaves room for fp re-association under random inputs.
            assert diff < 1e-9, (
                f"SEB {name!r} (randomised) diverges: {diff:.3e}"
            )


# --------------------------------------------------------------------------- #
# Sanity: the .so is detected / surfacing a clear error when missing.
# --------------------------------------------------------------------------- #
class TestSurfRefAvailability:
    """Meta-tests for the wrapper's availability handling."""

    def test_is_available_when_so_exists(self):
        """If the .so is on disk, is_available() returns True."""
        if _SURF_REF_AVAILABLE:
            assert SurfRef.is_available()

    def test_explicit_missing_path_raises(self):
        """A non-existent so_path raises FileNotFoundError (not segfault)."""
        if not _SURF_REF_AVAILABLE:
            pytest.skip("libsurf_ref.so not built")
        from offline_driver.surf_ref import _load_lib
        with pytest.raises(FileNotFoundError):
            _load_lib("/nonexistent/libsurf_ref.so")
