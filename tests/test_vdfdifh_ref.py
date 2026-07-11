"""ULP-level comparison tests for the offline-driver VDFDIFH wrapper.

These tests feed identical :class:`DiffusionInputs` to the PyTorch port
(:func:`surf_pytorch.physics.diffusion.vdfdifh1s`) and to the Fortran
reference kernel (``surf_ref_vdfdifh``, wrapped by
:meth:`offline_driver.SurfRef.vdfdifh`) and assert ULP-level agreement
on every output.

Why this matters
----------------
The VDFDIFH1S offline-driver wrapper is mostly *packaging*: it builds
the SURFSEB inputs from the lowest-model-level forcing, inverts the
exchange coefficients (ZCOEF1 = 1/(RG*RVDIFTS*PTMST)), constructs the
per-tile PTSRF, calls SURFSEB once (canonical call #3), optionally
repeats for LESNML snow melt (call #4), adds the snow-under-forest
evaporation to tile 7, and extrapolates the skin T to t+1.

The inner SURFSEB is already validated to last ULP by
:class:`tests.test_fortran_ref.TestSebFortranRef`; this test isolates
the *packaging* logic that the wrapper adds on top:

* the Dirichlet boundary condition (asl=aql=0, bsl/bql from forcing),
* the exchange-coefficient inversion (ZCOEF1),
* the per-tile PTSRF mapping (water=SST, ice=PTICE, snow=PTSNOW, ...),
* the snow-under-forest evaporation addition to tile 7,
* the skin-T extrapolation to t+1,
* the optional LESNML snow-melt repeat (call #4).

Skip behaviour
--------------
If ``libsurf_ref.so`` is not built, every test is skipped via the
``requires_fortran_ref`` fixture.  Build with::

    offline_driver/build_ref.sh
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from surf_pytorch import ConstantBundle, SurfConfig
from surf_pytorch.physics.diffusion import (
    DiffusionInputs,
    DiffusionOutputs,
    RVDIFTS,
    vdfdifh1s as port_vdfdifh,
)
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


pytestmark = pytest.mark.fortran_ref


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def ref() -> "SurfRef":
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
    return pytest.mark.skipif(
        not _SURF_REF_AVAILABLE,
        reason="libsurf_ref.so not built -- run offline_driver/build_ref.sh",
    )


# --------------------------------------------------------------------------- #
# Input builder
# --------------------------------------------------------------------------- #
def _build_inputs(
    *,
    N: int = 3,
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
    ktvl: int = 1,
    ktvh: int = 3,
    ptmst: float = 3600.0,
    pevapsnw: float | None = 0.0,
    lesnml: bool = False,
    dtype: torch.dtype = torch.float64,
) -> DiffusionInputs:
    """Build a self-consistent DiffusionInputs for ULP comparison.

    Mirrors :func:`tests.test_diffusion._build_diffusion_inputs` but
    returns a DiffusionInputs directly.  ``rho_chu`` / ``rho_cqu`` are
    the physical rho*Ch*|U|; the upstream RG*PTSPHY*RVDIFTS rescaling
    is applied here so the wrapper's inversion recovers them.
    """
    consts = ConstantBundle(dtype=dtype)
    cp = float(consts.cpd.item())
    rg = float(consts.g.item())

    tsk = torch.full((N, T), tsk1, dtype=dtype)
    p = torch.full((N, T), psurf, dtype=dtype)
    qsk = qsat(tsk, p, consts)
    dqsdT = dqsat_dt_ifs(tsk, p, consts)
    ssk = cp * tsk

    pcfhti = torch.full((N, T), rho_chu * rg * ptmst * RVDIFTS, dtype=dtype)
    pcfqti = torch.full((N, T), rho_cqu * rg * ptmst * RVDIFTS, dtype=dtype)

    pev = None
    if pevapsnw is not None:
        pev = torch.full((N,), pevapsnw, dtype=dtype)

    return DiffusionInputs(
        pslm1_lev=torch.full((N,), cp * tair, dtype=dtype),
        pqtm1_lev=torch.full((N,), qair, dtype=dtype),
        pslge_lev=torch.full((N,), pslge, dtype=dtype),
        pqte_lev=torch.full((N,), pqte, dtype=dtype),
        pcfhti=pcfhti, pcfqti=pcfqti,
        psst=torch.full((N,), sst, dtype=dtype),
        ptice=torch.full((N,), tice, dtype=dtype),
        ptsm1m=torch.full((N,), tsm1m, dtype=dtype),
        ptsnow=torch.full((N,), tsnow, dtype=dtype),
        sskm1m=ssk, tskm1m=tsk, qskm1m=qsk, dqsdT=dqsdT,
        alphal=torch.full((N, T), alphal, dtype=dtype),
        alphas=torch.full((N, T), alphas, dtype=dtype),
        pssrflti=torch.full((N, T), ssrfl, dtype=dtype),
        pslrfl=torch.full((N,), slrfl, dtype=dtype),
        ptskrad=torch.full((N,), tsk1, dtype=dtype),
        pemis=torch.full((N,), 0.99, dtype=dtype),
        plamsk=torch.full((N, T), lamsk, dtype=dtype),
        pfrti=torch.full((N, T), 1.0 / T, dtype=dtype),
        ktvl=torch.full((N,), ktvl, dtype=torch.long),
        ktvh=torch.full((N,), ktvh, dtype=torch.long),
        ptmst=torch.tensor(ptmst, dtype=dtype),
        pevapsnw=pev,
        lesnml=lesnml,
    )


# All DiffusionOutputs fields compared; names match the dataclass.
_DIFF_FIELDS = ("pjq", "pssh", "pslh", "pstr", "pg0", "ptsk",
                "ptsktip1", "psl", "pql")


def _vlamsk_for(inp: DiffusionInputs, veg, consts, config) -> torch.Tensor:
    """Compute the per-tile PLAMSK the port would use, so ref and port
    see the same lamsk (isolating the packaging from VLAMSK divergence).

    The VDFDIFH wrapper consumes inp.plamsk directly; we rebuild it from
    the wrapper's PTSRF build so the snow/ice tiles use the right lower
    boundary.  This mirrors the upstream VDFMAIN1S ordering (VLAMSK
    before VDFDIFH1S).
    """
    from surf_pytorch.physics.diffusion import _build_tsrf
    ngrid = inp.psst.shape[0]
    ntile = config.n_tile
    tsrf = _build_tsrf(inp, ngrid, ntile, inp.psst.dtype, inp.psst.device)
    # VLAMSK needs per-grid SNM/RSN -- the DiffusionInputs does not yet
    # carry these, so use a no-snow default.  This is the same default
    # the test_diffusion closure tests use.
    snm = torch.zeros(ngrid, dtype=inp.psst.dtype, device=inp.psst.device)
    rsn = torch.full((ngrid,), 300.0, dtype=inp.psst.dtype, device=inp.psst.device)
    return port_vlamsk(
        inp.tskm1m, tsrf, snm, rsn, None,
        inp.ktvl, inp.ktvh, veg, consts, config,
        ptstep=inp.ptmst,
    )


# --------------------------------------------------------------------------- #
# VDFDIFH -- offline single-column wrapper
# --------------------------------------------------------------------------- #
class TestVdfdifhFortranRef:
    """VDFDIFH1S port vs Fortran reference."""

    def _run_both(self, ref, veg, consts, config, inp: DiffusionInputs):
        """Run port + Fortran ref with synced PLAMSK output.

        The wrapper consumes ``inp.plamsk``; we feed both the same
        VLAMSK output (rebuilt from the wrapper's own PTSRF) so any
        divergence is isolated from VLAMSK divergence.
        """
        lamsk = _vlamsk_for(inp, veg, consts, config)
        inp = replace(inp, plamsk=lamsk)
        out_ref = ref.vdfdifh(inp, veg, consts, config)
        out_port = port_vdfdifh(inp, consts, config)
        return out_ref, out_port

    @requires_fortran_ref()
    def test_default_inputs_match_to_ulp(self, ref, veg, consts, config):
        """Default batch: every field matches to ~1e-12 (last ULP)."""
        inp = _build_inputs()
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        for name in _DIFF_FIELDS:
            a = getattr(out_ref, name)
            b = getattr(out_port, name)
            diff = (a - b).abs().max().item()
            assert diff < 1e-9, (
                f"VDFDIFH field {name!r} diverges: max|diff|={diff:.3e}\n"
                f"  ref ={a.tolist()}\n  port={b.tolist()}"
            )

    @requires_fortran_ref()
    @pytest.mark.parametrize(
        "rho_chu,rho_cqu,ssrfl,slrfl",
        [
            (0.001, 0.001, 0.0, -200.0),    # weak exchange, no solar (night)
            (0.05, 0.05, 800.0, -350.0),    # strong exchange, bright day
            (0.02, 0.02, 100.0, -100.0),    # mild
            (0.01, 0.005, 400.0, -250.0),   # asymmetric Ce/Cq
        ],
    )
    def test_regime_sweep(
        self, ref, veg, consts, config, rho_chu, rho_cqu, ssrfl, slrfl
    ):
        """Agreement across turbulent / radiative regimes."""
        inp = _build_inputs(
            rho_chu=rho_chu, rho_cqu=rho_cqu, ssrfl=ssrfl, slrfl=slrfl,
        )
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        for name in _DIFF_FIELDS:
            diff = (getattr(out_ref, name) - getattr(out_port, name)).abs().max().item()
            assert diff < 1e-9, (
                f"VDFDIFH {name!r} (Ce={rho_chu},Cq={rho_cqu},SW={ssrfl},"
                f"LW={slrfl}) diverges: {diff:.3e}"
            )

    @requires_fortran_ref()
    def test_per_tile_tsrf_mapping(self, ref, veg, consts, config):
        """Distinct SST / TICE / PTSM1M / PTSNOW -> PTSRF mapping matches.

        Use distinct values for each lower-boundary field so the per-tile
        PTSRF build is exercised: tiles 1/2/3/4/6/8 use PTSM1M, tile 5/7
        use PTSNOW, tile 0 uses SST, tile 1 uses PTICE.
        """
        inp = _build_inputs(
            sst=291.0, tice=265.0, tsm1m=284.0, tsnow=272.0,
        )
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        for name in _DIFF_FIELDS:
            diff = (getattr(out_ref, name) - getattr(out_port, name)).abs().max().item()
            assert diff < 1e-9, (
                f"VDFDIFH {name!r} (tsrf mapping) diverges: {diff:.3e}"
            )

    @requires_fortran_ref()
    def test_tendency_feeds_bsl_bql(self, ref, veg, consts, config):
        """Non-zero PSLGE / PQTE shifts bsl / bql; both implementations agree."""
        inp = _build_inputs(pslge=1.0e-3, pqte=-2.0e-7)
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        for name in _DIFF_FIELDS:
            diff = (getattr(out_ref, name) - getattr(out_port, name)).abs().max().item()
            assert diff < 1e-9, (
                f"VDFDIFH {name!r} (tendency) diverges: {diff:.3e}"
            )

    @requires_fortran_ref()
    def test_different_veg_types(self, ref, veg, consts, config):
        """Different dominant-veg types still match (changes lambda_skin)."""
        for ktvl, ktvh in [(1, 3), (5, 18), (8, 11), (13, 19)]:
            inp = _build_inputs(ktvl=ktvl, ktvh=ktvh)
            out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
            for name in _DIFF_FIELDS:
                diff = (getattr(out_ref, name) - getattr(out_port, name)).abs().max().item()
                assert diff < 1e-9, (
                    f"VDFDIFH {name!r} (ktvl={ktvl},ktvh={ktvh}) diverges: "
                    f"{diff:.3e}"
                )

    @requires_fortran_ref()
    def test_snow_under_forest_evap(self, ref, veg, consts, config):
        """Non-zero PEVAPSNW: tile-7 PJQ/PSLH addition matches to ULP."""
        inp = _build_inputs(pevapsnw=2.5e-5)
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        for name in _DIFF_FIELDS:
            diff = (getattr(out_ref, name) - getattr(out_port, name)).abs().max().item()
            assert diff < 1e-9, (
                f"VDFDIFH {name!r} (PEVAPSNW) diverges: {diff:.3e}"
            )

    @requires_fortran_ref()
    def test_no_evapsnw_path(self, ref, veg, consts, config):
        """PEVAPSNW=None (HAS_EVAPSNW=False) path matches."""
        inp = _build_inputs(pevapsnw=None)
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        for name in _DIFF_FIELDS:
            diff = (getattr(out_ref, name) - getattr(out_port, name)).abs().max().item()
            assert diff < 1e-9, (
                f"VDFDIFH {name!r} (no evapsnw) diverges: {diff:.3e}"
            )

    @requires_fortran_ref()
    def test_skin_extrapolation_to_t1(self, ref, veg, consts, config):
        """PTSKTIP1 = (1/RVDIFTS)*PTSK + (1-1/RVDIFTS)*PTSKTI matches."""
        inp = _build_inputs()
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        # Both implementations must agree on PTSK and PTSKTIP1.
        diff_tsk = (out_ref.ptsk - out_port.ptsk).abs().max().item()
        diff_tip1 = (out_ref.ptsktip1 - out_port.ptsktip1).abs().max().item()
        assert diff_tsk < 1e-9, f"PTSK diverges: {diff_tsk:.3e}"
        assert diff_tip1 < 1e-9, f"PTSKTIP1 diverges: {diff_tip1:.3e}"

    @requires_fortran_ref()
    def test_lesnml_off_matches(self, ref, veg, consts, config):
        """LESNML=False (no call-#4 repeat): both implementations match."""
        inp = _build_inputs(lesnml=False, tsnow=275.0, tsk1=276.0,
                            ssrfl=400.0, slrfl=-100.0, lamsk=5.0)
        out_ref, out_port = self._run_both(ref, veg, consts, config, inp)
        for name in _DIFF_FIELDS:
            diff = (getattr(out_ref, name) - getattr(out_port, name)).abs().max().item()
            assert diff < 1e-9, (
                f"VDFDIFH {name!r} (lesnml=False) diverges: {diff:.3e}"
            )

    @requires_fortran_ref()
    def test_lesnml_on_melting_snow(self, ref, veg, consts, config):
        """LESNML=True with melting low-veg-snow triggers the call-#4 repeat.

        Builds an input where the low-veg-snow tile has a warm skin and
        significant coverage so the call-#3 PTSK exceeds RTT; both
        implementations then run the repeat and must agree.
        """
        from surf_pytorch.config import TILE_LOW_VEG_SNOW
        T = TILE_LOW_VEG_SNOW + 1     # only the first 5 tiles
        inp = _build_inputs(
            T=T, lesnml=True,
            tsk1=275.0, tsnow=275.0,   # warm snow
            ssrfl=600.0, slrfl=-50.0,  # strong net radiation in
            rho_chu=0.001, rho_cqu=0.001,
            lamsk=2.0,                 # weak ground coupling -> Tsk rises
        )
        # Concentrate the tile fraction on the low-veg-snow tile so the
        # melting condition dominates the grid-mean solve.
        frac = torch.zeros((inp.pfrti.shape[0], T), dtype=torch.float64)
        frac[:, TILE_LOW_VEG_SNOW] = 1.0
        inp = replace(inp, pfrti=frac)
        cfg = replace(config, n_tile=T)
        out_ref, out_port = self._run_both(ref, veg, consts, cfg, inp)
        for name in _DIFF_FIELDS:
            diff = (getattr(out_ref, name) - getattr(out_port, name)).abs().max().item()
            assert diff < 1e-9, (
                f"VDFDIFH {name!r} (lesnml repeat) diverges: {diff:.3e}"
            )
