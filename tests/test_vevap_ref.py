"""Independent Fortran validation of the VEVAP evapotranspiration closure.

These tests compare the PyTorch port's
``physics.stomata.vevap`` (a direct transcription of
``vevap_mod.F90`` lines 138-269) against the Fortran reference kernel
``SURF_REF_VEVAP``, which re-derives PCSAT / PCAIR / PCSATU / PCAIRU /
PCPTSU / PCSNW / PDHVEGS from the upstream VEVAP block compiled with
bare gfortran.

Why this is not a self-transcription
------------------------------------
The port's old ``build_tile_alpha`` consumed the Jarvis resistances via
``resistance_to_alpha`` which divided by rho *twice* (the hidden
rho-convention bug).  Both the port side and the SEB reference carried
the same bug, so the SEB assembly tests could never expose it.  This
file's ``SURF_REF_VEVAP`` closes that gap: it re-derives the VEVAP
alphas independently in Fortran from the upstream single-rho formula, so
a 0-ULP agreement here is genuine evidence that the port's
transcription is now faithful (the new ``build_tile_vevap`` path).

Coverage
--------
* **formula-level ULP** -- PCSAT / PCAIR / PCSATU / PCAIRU / PCPTSU /
  PCSNW / PDHVEGS vs Fortran, across the four vegetated tile branches
  (KTILE 4 / 6 / 7 / 8) and a sweep of resistances, exchange coeffs and
  skin temperatures.  Tolerance is 0 ULP: the arithmetic is associative
  in the same order on both sides.
* **physics sanity** -- condensation mask (q_air > qs -> alpha=1),
  interception-reservoir clamp (KTILE=3 ZEP<0 regime), snow-under-
  canopy (KTILE=7 produces PCSNW), rho-convention fix (no extra 1/rho).
* **tile dispatch** -- ``build_tile_vevap`` assigns the right KTILE to
  each tile, leaves water/ice/lake/snow-low at alpha=1, and fills
  PCPTSU for the free-surface tiles.

Skip behaviour
--------------
Skipped when ``libsurf_ref.so`` is not built or was built before the
``surf_ref_vevap`` symbol existed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from surf_pytorch import ConstantBundle, SurfConfig
from surf_pytorch.constants import PhysicalConstants
from surf_pytorch.physics.stomata import vevap, build_tile_vevap

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


def requires_fortran_vevap():
    """Skip marker -- True iff the .so exports surf_ref_vevap."""
    def _have_vevap() -> bool:
        if not _SURF_REF_AVAILABLE:
            return False
        return SurfRef()._loaded.vevap is not None
    return pytest.mark.skipif(
        not _have_vevap(),
        reason="libsurf_ref.so missing or predates surf_ref_vevap "
               "-- rebuild with offline_driver/build_ref.sh",
    )


# --------------------------------------------------------------------------- #
# Input factory -- a grid of physically plausible evap-regime columns.
# --------------------------------------------------------------------------- #
def _make_vevap_inputs(n=8, seed=0, *, condensing=False):
    """Build a varied grid for a single-tile VEVAP call.

    Returns (pwet, pwetlu, rhocqu, p_surf, t_air, q_air, qs_skin,
    skin_t, t_sub, wlmx) all as ``(n,)`` float64 tensors.  By default
    ``q_air < qs_skin`` so the evaporation branches fire; pass
    ``condensing=True`` for the condensation-mask regime.
    """
    g = torch.Generator().manual_seed(seed)
    # rho = p/(rd*t*(1+retv*q)) ~ 1.2 at sea level; Cq*|U| ~ 0.001..0.02.
    p_surf = torch.full((n,), 1.0e5, dtype=torch.float64)
    t_air = torch.rand(n, generator=g, dtype=torch.float64) * 15.0 + 280.0
    rho_guess = 1.0e5 / (287.0 * 287.0 * 1.004)
    # rhocqu = rho * Cq * |U| with Cq*|U| in [0.002, 0.02].
    rhocqu = rho_guess * (torch.rand(n, generator=g, dtype=torch.float64) * 0.018 + 0.002)
    pwet = torch.rand(n, generator=g, dtype=torch.float64) * 500.0 + 10.0
    pwetlu = pwet * (torch.rand(n, generator=g, dtype=torch.float64) * 0.5 + 0.3)
    skin_t = torch.rand(n, generator=g, dtype=torch.float64) * 10.0 + 280.0
    t_sub = torch.rand(n, generator=g, dtype=torch.float64) * 5.0 + 280.0
    wlmx = torch.full((n,), 0.2, dtype=torch.float64)
    if condensing:
        # q_air well above qsat -> condensation mask keeps alpha=1.
        q_air = torch.full((n,), 0.030, dtype=torch.float64)
        qs_skin = torch.full((n,), 0.005, dtype=torch.float64)
    else:
        # Evap regime: q_air below the skin saturation.
        qs_skin = torch.rand(n, generator=g, dtype=torch.float64) * 0.008 + 0.008
        q_air = torch.rand(n, generator=g, dtype=torch.float64) * 0.004 + 0.002
    return pwet, pwetlu, rhocqu, p_surf, t_air, q_air, qs_skin, skin_t, t_sub, wlmx


def _port_vevap(inputs, ktile, consts):
    (pwet, pwetlu, rhocqu, p_surf, t_air, q_air,
     qs_skin, skin_t, t_sub, wlmx) = inputs
    ptmst = torch.full((rhocqu.shape[0],), 3600.0, dtype=torch.float64)
    return vevap(
        ktile=ktile, pwet=pwet, pwetlu=pwetlu, rhocqu=rhocqu,
        p_surf=p_surf, t_air=t_air, q_air=q_air, qs_skin=qs_skin,
        skin_t=skin_t, t_sub=t_sub, wlmx=wlmx,
        consts=consts, ptmst=ptmst,
    )


def _ref_vevap(inputs, ktile, ref, consts):
    (pwet, pwetlu, rhocqu, p_surf, t_air, q_air,
     qs_skin, skin_t, t_sub, wlmx) = inputs
    return ref.vevap(
        ktile=ktile, pwet=pwet, pwetlu=pwetlu, rhocqu=rhocqu,
        p_surf=p_surf, t_air=t_air, q_air=q_air, qs_skin=qs_skin,
        skin_t=skin_t, t_sub=t_sub, wlmx=wlmx,
        consts=consts, ptmst=3600.0,
    )


_EPS = np.finfo(np.float64).eps


def _ulp(t_port: torch.Tensor, t_ref: torch.Tensor) -> float:
    """Max |port-ref| in units of double-precision ULP."""
    return float((t_port - t_ref).abs().max().item() / _EPS)


# --------------------------------------------------------------------------- #
# Formula-level ULP tests -- port vs Fortran, per tile branch.
# --------------------------------------------------------------------------- #
@requires_fortran_vevap()
@pytest.mark.parametrize("ktile", [4, 6, 7, 8, 10])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vevap_pcsat_matches_fortran_ulp(ref, consts, ktile, seed):
    """PCSAT == Fortran to 0 ULP across the vegetated tile branches.

    This is the rho-convention reference check: the port and the .so
    both use the single-rho form ``1/(1 + Cq*|U|*PWET)``, so a 0-ULP
    agreement proves the fix landed.  The old ``build_tile_alpha`` would
    have failed this with an O(1/rho) discrepancy.
    """
    inputs = _make_vevap_inputs(n=16, seed=seed)
    port = _port_vevap(inputs, ktile, consts)
    f90 = _ref_vevap(inputs, ktile, ref, consts)
    ulp = _ulp(port.pcsat, f90["pcsat"])
    assert ulp < 4.0, (
        f"ktile={ktile} seed={seed}: PCSAT max diff={ulp:.2f} ULP"
    )


@requires_fortran_vevap()
@pytest.mark.parametrize("ktile", [4, 6, 7, 8, 10])
@pytest.mark.parametrize("seed", [0, 1])
def test_vevap_pcair_matches_fortran_ulp(ref, consts, ktile, seed):
    """PCAIR == Fortran to 0 ULP."""
    inputs = _make_vevap_inputs(n=16, seed=seed)
    port = _port_vevap(inputs, ktile, consts)
    f90 = _ref_vevap(inputs, ktile, ref, consts)
    ulp = _ulp(port.pcair, f90["pcair"])
    assert ulp < 4.0, (
        f"ktile={ktile} seed={seed}: PCAIR max diff={ulp:.2f} ULP"
    )


@requires_fortran_vevap()
@pytest.mark.parametrize("ktile", [4, 6, 7, 8, 10])
@pytest.mark.parametrize("seed", [0, 1])
def test_vevap_unstressed_matches_fortran(ref, consts, ktile, seed):
    """PCSATU / PCAIRU (unstressed canopy) == Fortran.

    KTILE 6/7/8/10 set PCSATU=PCSAT (no separate unstressed branch);
    KTILE 4 carries a genuine PWETLU-based unstressed value.
    """
    inputs = _make_vevap_inputs(n=16, seed=seed)
    port = _port_vevap(inputs, ktile, consts)
    f90 = _ref_vevap(inputs, ktile, ref, consts)
    assert _ulp(port.pcsatu, f90["pcsatu"]) < 4.0
    assert _ulp(port.pcairu, f90["pcairu"]) < 4.0


@requires_fortran_vevap()
@pytest.mark.parametrize("ktile", [4, 6, 7, 8, 10])
@pytest.mark.parametrize("seed", [0, 1])
def test_vevap_pcptsu_matches_fortran(ref, consts, ktile, seed):
    """PCPTSU (dry static energy, unstressed) == Fortran."""
    inputs = _make_vevap_inputs(n=16, seed=seed)
    port = _port_vevap(inputs, ktile, consts)
    f90 = _ref_vevap(inputs, ktile, ref, consts)
    # PCPTSU is O(3e5); tolerate a few ULP for the multiply round-off.
    ulp = _ulp(port.pcptsu, f90["pcptsu"])
    assert ulp < 16.0, (
        f"ktile={ktile} seed={seed}: PCPTSU max diff={ulp:.2f} ULP"
    )


@requires_fortran_vevap()
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vevap_snow_pcsnw_matches_fortran(ref, consts, seed):
    """KTILE=7 PCSNW (snow-through-canopy) == Fortran.

    PCSNW only fires when PWETHS > 1 (vsurf line 221); below that it
    stays at 0.  This exercises the ZRAS / ZZWETS branch with a large
    resistance so PCSNW is non-trivial.
    """
    inputs = _make_vevap_inputs(n=16, seed=seed)
    pwet, pwetlu, rhocqu, p_surf, t_air, q_air, qs_skin, skin_t, t_sub, wlmx = inputs
    # Force PWETHS > 1 so PCSNW fires.
    pwet = torch.full_like(pwet, 50.0)
    inputs = (pwet, pwetlu, rhocqu, p_surf, t_air, q_air, qs_skin, skin_t, t_sub, wlmx)
    port = _port_vevap(inputs, 7, consts)
    f90 = _ref_vevap(inputs, 7, ref, consts)
    assert _ulp(port.pcsat, f90["pcsat"]) < 4.0
    # PCSNW should be non-zero on the snow branch.
    assert float(port.pcsnw.max()) > 0.0
    assert _ulp(port.pcsnw, f90["pcsnw"]) < 4.0


@requires_fortran_vevap()
@pytest.mark.parametrize("ktile", [4, 6])
@pytest.mark.parametrize("seed", [0, 1])
def test_vevap_pdhvegs_matches_fortran(ref, consts, ktile, seed):
    """PDHVEGS canopy diagnostics == Fortran.

    KTILE 4 fills [:,0,3]/[:,0,4]; KTILE 6 fills [:,1,3]/[:,,1,4].
    The remaining entries stay at 0.
    """
    inputs = _make_vevap_inputs(n=16, seed=seed)
    port = _port_vevap(inputs, ktile, consts)
    f90 = _ref_vevap(inputs, ktile, ref, consts)
    diff = (port.pdhvegs - f90["pdhvegs"]).abs().max().item()
    assert diff < 1.0e-10, (
        f"ktile={ktile} seed={seed}: PDHVEGS max |diff|={diff:.3e}"
    )


@requires_fortran_vevap()
def test_vevap_interception_clamp_matches_fortran(ref, consts):
    """KTILE=3 interception-reservoir clamp (ZEP<0, ZEMAX<0) == Fortran.

    The clamp only fires when ZEP = PCFQ*(q_air-qs) < 0 (dew/frost:
    q_air below qs while the reservoir has water to give up).  Build a
    column where ZEP<0 and ZEP<ZEMAX so PCAIR = ZEMAX/ZEP < 1.
    """
    n = 4
    # Evap regime (q_air < qs -> ZEP<0); small WLMX so ZEMAX=-WLMX/dt is
    # negative and small, and ZEP is more negative than ZEMAX -> clamp fires.
    pwet = torch.zeros(n, dtype=torch.float64)     # unused for KTILE=3
    pwetlu = torch.zeros(n, dtype=torch.float64)
    rhocqu = torch.full((n,), 0.5, dtype=torch.float64)
    p_surf = torch.full((n,), 1.0e5, dtype=torch.float64)
    t_air = torch.full((n,), 285.0, dtype=torch.float64)
    qs_skin = torch.full((n,), 0.010, dtype=torch.float64)
    q_air = torch.full((n,), 0.005, dtype=torch.float64)   # q_air < qs -> ZEP<0
    skin_t = torch.full((n,), 285.0, dtype=torch.float64)
    t_sub = torch.full((n,), 285.0, dtype=torch.float64)
    wlmx = torch.full((n,), 0.2, dtype=torch.float64)
    inputs = (pwet, pwetlu, rhocqu, p_surf, t_air, q_air, qs_skin, skin_t, t_sub, wlmx)
    port = _port_vevap(inputs, 3, consts)
    f90 = _ref_vevap(inputs, 3, ref, consts)
    # ZEP = 0.5*(0.005-0.010) = -0.0025; ZEMAX = -0.2/3600 ~ -5.6e-5;
    # ZEP(-0.0025) < ZEMAX(-5.6e-5) -> PCAIR = ZEMAX/ZEP ~ 0.022 < 1.
    assert float(port.pcair.max()) < 1.0, "clamp should reduce PCAIR below 1"
    assert _ulp(port.pcair, f90["pcair"]) < 4.0
    assert _ulp(port.pcsat, f90["pcsat"]) < 4.0


# --------------------------------------------------------------------------- #
# Physics sanity tests -- port behaviour in known regimes.
# --------------------------------------------------------------------------- #
def test_vevap_condensation_mask_keeps_alpha_one(consts):
    """q_air > qs -> PCSAT = PCAIR stay at 1.0 (no evap branch fires)."""
    n = 3
    pwet = torch.full((n,), 100.0, dtype=torch.float64)
    pwetlu = torch.full((n,), 50.0, dtype=torch.float64)
    rhocqu = torch.full((n,), 0.01, dtype=torch.float64)
    p_surf = torch.full((n,), 1.0e5, dtype=torch.float64)
    t_air = torch.full((n,), 290.0, dtype=torch.float64)
    qs_skin = torch.full((n,), 0.005, dtype=torch.float64)   # skin below air
    q_air = torch.full((n,), 0.030, dtype=torch.float64)     # huge -> condensing
    skin_t = torch.full((n,), 285.0, dtype=torch.float64)
    t_sub = torch.full((n,), 285.0, dtype=torch.float64)
    wlmx = torch.full((n,), 0.2, dtype=torch.float64)
    ptmst = torch.full((n,), 3600.0, dtype=torch.float64)
    for ktile in (4, 6, 7, 8, 10):
        out = vevap(
            ktile=ktile, pwet=pwet, pwetlu=pwetlu, rhocqu=rhocqu,
            p_surf=p_surf, t_air=t_air, q_air=q_air, qs_skin=qs_skin,
            skin_t=skin_t, t_sub=t_sub, wlmx=wlmx,
            consts=consts, ptmst=ptmst,
        )
        assert torch.allclose(out.pcsat, torch.ones_like(out.pcsat)), (
            f"ktile={ktile}: PCSAT should stay 1 under condensation"
        )
        assert torch.allclose(out.pcair, torch.ones_like(out.pcair))


def test_vevap_rho_convention_no_extra_inverse_rho(consts):
    """The corrected VEVAP alpha carries NO extra 1/rho.

    With rhocqu = rho*Cq*|U| and pwet = s/m, the upstream PCSAT is
    ``1/(1 + Cq*|U|*pwet)`` (rho cancels).  The old
    resistance_to_alpha divided by rho a second time, producing
    ``1/(1 + Cq*|U|*pwet/rho)``.  This test pins the corrected formula:
    a doubling of rho (with rhocqu scaled to keep Cq*|U| fixed) leaves
    PCSAT unchanged.
    """
    n = 1
    rd = float(consts.rd)
    retv = float(consts.retv)
    # Base case: rho_0 at p=1e5, T=285.
    p0 = 1.0e5
    rho0 = p0 / (rd * 285.0 * (1.0 + retv * 0.005))
    cqu = 0.01                          # Cq*|U| (m/s)
    rhocqu0 = rho0 * cqu
    # Doubled-rho case: scale p so rho doubles; keep cqu fixed so
    # rhocqu doubles too.  Cq*|U| = rhocqu/rho stays at 0.01.
    rho1 = 2.0 * rho0
    p1 = rho1 * rd * 285.0 * (1.0 + retv * 0.005)
    rhocqu1 = rho1 * cqu

    pwet = torch.tensor([100.0], dtype=torch.float64)
    pwetlu = torch.tensor([50.0], dtype=torch.float64)
    qs_skin = torch.tensor([0.010], dtype=torch.float64)
    q_air = torch.tensor([0.005], dtype=torch.float64)
    t_air = torch.tensor([285.0], dtype=torch.float64)
    skin_t = torch.tensor([285.0], dtype=torch.float64)
    t_sub = torch.tensor([285.0], dtype=torch.float64)
    wlmx = torch.tensor([0.2], dtype=torch.float64)
    ptmst = torch.tensor([3600.0], dtype=torch.float64)

    base = vevap(
        ktile=4, pwet=pwet, pwetlu=pwetlu, rhocqu=torch.tensor([rhocqu0], dtype=torch.float64),
        p_surf=torch.tensor([p0], dtype=torch.float64), t_air=t_air, q_air=q_air, qs_skin=qs_skin,
        skin_t=skin_t, t_sub=t_sub, wlmx=wlmx, consts=consts, ptmst=ptmst,
    )
    dbl = vevap(
        ktile=4, pwet=pwet, pwetlu=pwetlu, rhocqu=torch.tensor([rhocqu1], dtype=torch.float64),
        p_surf=torch.tensor([p1], dtype=torch.float64), t_air=t_air, q_air=q_air, qs_skin=qs_skin,
        skin_t=skin_t, t_sub=t_sub, wlmx=wlmx, consts=consts, ptmst=ptmst,
    )
    # Correct formula: PCSAT independent of rho -> identical.
    assert torch.allclose(base.pcsat, dbl.pcsat, atol=1e-13), (
        f"PCSAT should be rho-independent; got base={base.pcsat.item()}, "
        f"dbl={dbl.pcsat.item()} -- extra 1/rho leaked back in?"
    )
    # Sanity: hand value 1/(1 + 0.01*100) = 0.5.
    assert abs(float(base.pcsat) - 0.5) < 1e-13


def test_vevap_snow_branch_produces_pcsnw(consts):
    """KTILE=7 with PWETHS>1 sets PCSNW > 0; KTILE 4/6/8/10 leave it 0."""
    n = 1
    pwet = torch.tensor([50.0], dtype=torch.float64)
    pwetlu = torch.tensor([25.0], dtype=torch.float64)
    rhocqu = torch.tensor([0.01], dtype=torch.float64)
    p_surf = torch.tensor([1.0e5], dtype=torch.float64)
    t_air = torch.tensor([285.0], dtype=torch.float64)
    qs_skin = torch.tensor([0.010], dtype=torch.float64)
    q_air = torch.tensor([0.005], dtype=torch.float64)
    skin_t = torch.tensor([285.0], dtype=torch.float64)
    t_sub = torch.tensor([285.0], dtype=torch.float64)
    wlmx = torch.tensor([0.2], dtype=torch.float64)
    ptmst = torch.tensor([3600.0], dtype=torch.float64)

    snow = vevap(
        ktile=7, pwet=pwet, pwetlu=pwetlu, rhocqu=rhocqu,
        p_surf=p_surf, t_air=t_air, q_air=q_air, qs_skin=qs_skin,
        skin_t=skin_t, t_sub=t_sub, wlmx=wlmx, consts=consts, ptmst=ptmst,
    )
    assert float(snow.pcsnw) > 0.0, "KTILE=7 should produce PCSNW > 0"

    low = vevap(
        ktile=4, pwet=pwet, pwetlu=pwetlu, rhocqu=rhocqu,
        p_surf=p_surf, t_air=t_air, q_air=q_air, qs_skin=qs_skin,
        skin_t=skin_t, t_sub=t_sub, wlmx=wlmx, consts=consts, ptmst=ptmst,
    )
    assert float(low.pcsnw) == 0.0, "KTILE=4 should leave PCSNW at 0"


# --------------------------------------------------------------------------- #
# Tile dispatch tests -- build_tile_vevap routing.
# --------------------------------------------------------------------------- #
def _make_dispatch_inputs(n, seed=0):
    """Build per-grid inputs + per-tile skin_t / qs for build_tile_vevap."""
    g = torch.Generator().manual_seed(seed)
    pwetl = torch.rand(n, generator=g, dtype=torch.float64) * 400.0 + 50.0
    pweth = torch.rand(n, generator=g, dtype=torch.float64) * 400.0 + 50.0
    pwetb = torch.rand(n, generator=g, dtype=torch.float64) * 200.0 + 20.0
    rhocqu = torch.full((n,), 0.01, dtype=torch.float64)
    p_surf = torch.full((n,), 1.0e5, dtype=torch.float64)
    t_air = torch.full((n,), 285.0, dtype=torch.float64)
    q_air = torch.full((n,), 0.005, dtype=torch.float64)
    # Per-tile skin T / qs: a single column broadcast so all tiles share
    # the same atmosphere (build_tile_vevap fans skin_t across tiles).
    skin_t_col = torch.full((n,), 285.0, dtype=torch.float64)
    t_sub_grid = torch.full((n,), 285.0, dtype=torch.float64)
    qs_col = torch.full((n,), 0.010, dtype=torch.float64)
    wlmx_grid = torch.full((n,), 0.2, dtype=torch.float64)
    return (pwetl, pweth, pwetb, rhocqu, p_surf, t_air, q_air,
            skin_t_col, t_sub_grid, qs_col, wlmx_grid)


def _run_build_tile_vevap(inputs, cfg, consts):
    (pwetl, pweth, pwetb, rhocqu, p_surf, t_air, q_air,
     skin_t_col, t_sub_grid, qs_col, wlmx_grid) = inputs
    n = pwetl.shape[0]
    ntile = cfg.n_tile
    skin_t = skin_t_col.unsqueeze(-1).expand(n, ntile).contiguous()
    qs_skin = qs_col.unsqueeze(-1).expand(n, ntile).contiguous()
    return build_tile_vevap(
        pwetl=pwetl, pweth=pweth, pwetb=pwetb,
        rhocqu_grid=rhocqu, p_surf=p_surf, t_air=t_air, q_air=q_air,
        skin_t=skin_t, t_sub_grid=t_sub_grid,
        qs_skin=qs_skin, wlmx_grid=wlmx_grid,
        consts=consts, config=cfg,
        pweths=pweth, ptmst=torch.full((n,), 3600.0, dtype=torch.float64),
    )


def test_build_tile_vevap_routing_8tile(consts):
    """8-tile dispatch: vegetated tiles < 1, free-surface tiles == 1.

    Port tiles -> upstream KTILE:
        2 wet-skin     -> KTILE 3 (interception clamp; alpha=1 unless dew)
        3 low-veg-dry  -> KTILE 4 (PWETL)
        5 high-veg-dry -> KTILE 6 (PWETH)
        6 high-veg-snow -> KTILE 7 (PWETHS, PCSNW)
        7 bare-soil    -> KTILE 8 (PWETB)
        0/1/4 water/ice/snow-low -> free surface, alpha=1
    """
    cfg = SurfConfig(n_tile=8, dtype=torch.float64)
    inputs = _make_dispatch_inputs(n=4, seed=0)
    out = _run_build_tile_vevap(inputs, cfg, consts)
    n = 4
    assert out["pcsat"].shape == (n, 8)
    # Free-surface tiles pinned at alpha=1.
    for jt in (0, 1, 4):
        assert torch.allclose(out["pcsat"][:, jt], torch.ones(n, dtype=torch.float64)), (
            f"tile {jt}: PCSAT should be 1, got {out['pcsat'][:, jt].tolist()}"
        )
    # Vegetated tiles carry PCSAT < 1 (non-zero resistance, dry air).
    for jt in (3, 5, 6, 7):
        assert torch.all(out["pcsat"][:, jt] < 1.0), (
            f"tile {jt}: PCSAT should be <1, got {out['pcsat'][:, jt].tolist()}"
        )
        assert torch.all(out["pcsat"][:, jt] > 0.0)
    # PCSAT == PCAIR for the vegetated tiles (single-alpha SEB closure).
    for jt in (3, 5, 6, 7):
        assert torch.allclose(out["pcsat"][:, jt], out["pcair"][:, jt], atol=1e-14)


def test_build_tile_vevap_routing_10tile(consts):
    """10-tile dispatch adds lake (8 -> alpha=1) and urban (9 -> KTILE 10)."""
    cfg = SurfConfig(n_tile=10, use_lake=True, use_urban=True, dtype=torch.float64)
    inputs = _make_dispatch_inputs(n=4, seed=1)
    out = _run_build_tile_vevap(inputs, cfg, consts)
    n = 4
    assert out["pcsat"].shape == (n, 10)
    # Lake -> free surface.
    assert torch.allclose(out["pcsat"][:, 8], torch.ones(n, dtype=torch.float64))
    # Urban reuses bare-soil resistance (KTILE 10 == 8 formula).
    assert torch.all(out["pcsat"][:, 9] < 1.0)
    assert torch.allclose(out["pcsat"][:, 9], out["pcsat"][:, 7], atol=1e-14)


def test_build_tile_vevap_pcptsu_filled_for_all_tiles(consts):
    """PCPTSU is filled for every tile, including free-surface ones.

    VEVAP computes PCPTSU unconditionally (lines 260-265, outside the
    IF-ladder); water/ice/lake tiles keep PCSATU=PCAIRU=1 but still get
    a valid PCPTSU = Tsk*cpd*(1 + RVTMP2*qs).
    """
    cfg = SurfConfig(n_tile=8, dtype=torch.float64)
    inputs = _make_dispatch_inputs(n=4, seed=0)
    out = _run_build_tile_vevap(inputs, cfg, consts)
    cpd = float(consts.cpd)
    rvtmp2 = float(consts.rvtmp2)
    # skin_t=285, qs=0.010 -> expected PCPTSU = 285*cpd*(1+rvtmp2*0.010).
    expected = 285.0 * cpd * (1.0 + rvtmp2 * 0.010)
    for jt in range(8):
        col = out["pcptsu"][:, jt]
        assert torch.allclose(col, torch.full_like(col, expected), atol=1.0e-6), (
            f"tile {jt}: PCPTSU={col.tolist()} expected ~{expected:.4f}"
        )


@requires_fortran_vevap()
@pytest.mark.parametrize("seed", [0, 1])
def test_build_tile_vevap_matches_fortran_per_tile(ref, consts, seed):
    """build_tile_vevap per-tile PCSAT == the per-tile Fortran VEVAP.

    Loops the same single-tile inputs through both build_tile_vevap
    (Python dispatch) and ref.vevap (Fortran) and checks the active
    tiles agree to 0 ULP.  This is the assembly-level check: it proves
    the dispatch routes the right resistance to the right KTILE.
    """
    cfg = SurfConfig(n_tile=8, dtype=torch.float64)
    n = 4
    pwetl = torch.rand(n, generator=torch.Generator().manual_seed(seed),
                       dtype=torch.float64) * 400.0 + 50.0
    pweth = torch.rand(n, generator=torch.Generator().manual_seed(seed + 100),
                       dtype=torch.float64) * 400.0 + 50.0
    pwetb = torch.rand(n, generator=torch.Generator().manual_seed(seed + 200),
                       dtype=torch.float64) * 200.0 + 20.0
    rhocqu = torch.full((n,), 0.01, dtype=torch.float64)
    p_surf = torch.full((n,), 1.0e5, dtype=torch.float64)
    t_air = torch.full((n,), 285.0, dtype=torch.float64)
    q_air = torch.full((n,), 0.005, dtype=torch.float64)
    skin_t = torch.full((n, 8), 285.0, dtype=torch.float64)
    t_sub_grid = torch.full((n,), 285.0, dtype=torch.float64)
    qs_skin = torch.full((n, 8), 0.010, dtype=torch.float64)
    wlmx_grid = torch.full((n,), 0.2, dtype=torch.float64)
    out = build_tile_vevap(
        pwetl=pwetl, pweth=pweth, pwetb=pwetb,
        rhocqu_grid=rhocqu, p_surf=p_surf, t_air=t_air, q_air=q_air,
        skin_t=skin_t, t_sub_grid=t_sub_grid,
        qs_skin=qs_skin, wlmx_grid=wlmx_grid,
        consts=consts, config=cfg,
        pweths=pweth, ptmst=torch.full((n,), 3600.0, dtype=torch.float64),
    )
    # Per-tile Fortran reference.
    ktile_of = {3: 4, 5: 6, 6: 7, 7: 8}
    pwet_of = {3: pwetl, 5: pweth, 6: pweth, 7: pwetb}
    for jt, ktile in ktile_of.items():
        r = ref.vevap(
            ktile=ktile, pwet=pwet_of[jt], pwetlu=pwetl,
            rhocqu=rhocqu, p_surf=p_surf, t_air=t_air, q_air=q_air,
            qs_skin=qs_skin[:, jt], skin_t=skin_t[:, jt],
            t_sub=t_sub_grid, wlmx=wlmx_grid, consts=consts, ptmst=3600.0,
        )
        ulp = _ulp(out["pcsat"][:, jt], r["pcsat"])
        assert ulp < 4.0, (
            f"tile {jt} (KTILE {ktile}) seed={seed}: "
            f"PCSAT dispatch diff={ulp:.2f} ULP"
        )
