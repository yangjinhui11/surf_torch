"""Unit tests for the HTESSEL soil-moisture update
(``surf_pytorch.physics.soil.compute_soil_moisture``).

These tests exercise the Python port of the soil-moisture branch of
``SURFTSTP_CTL`` (section 4): the Van Genuchten hydraulic functions
(SRFWEXC_VG), the VIC / infiltration-limit runoff, the root-extraction
weights, the implicit Thomas solve (SRFWDIF), the new-moisture update
(SRFWINC), and the negative-moisture clipping / redistribution (SRFWNG).

A ULP-level comparison against a self-compiled Fortran reference kernel
(``SURF_REF_SOILW``) is added in M5c; this file covers the *internal*
mathematical correctness of the port so the M5c test only needs to
catch transcription drift rather than re-derive the physics.

Sign conventions
----------------
* ``PEVAPTI`` is positive *downward* (toward the surface).  Negative
  values mean evaporation (moisture loss); positive values mean dew /
  condensation (moisture gain).  This matches the IFS convention used
  in VDFDIFH1S where ``PEVAPTI = PFLH1D/RLVTT`` with PFLH1D positive
  downward.
* ``PROS`` (surface runoff) and ``PROD`` (sub-surface runoff) are
  positive *out of* the soil column.

Test groups:

* ``TestSoilMoistVanGenuchten`` -- K/D clamp at the air-dry floor
  (K=0, D=Dmin), near saturation (K=WCONS, D=Dmax), and the closed-form
  mid-range values.
* ``TestSoilMoistLiquidFraction`` -- ZF=1 below RTF2 (frozen), ZF=0
  above RTF1 (warm liquid), sinusoidal ramp in between.
* ``TestSoilMoistRootExtraction`` -- dew deposition puts all roots in
  layer 1; transpiration distributes via RVROOTSA.
* ``TestSoilMoistRunoff`` -- zero precip gives zero runoff (VIC);
  VIC saturation-excess runoff produces sensible values; the
  infiltration-limit branch gives a different number from VIC.
* ``TestSoilMoistZeroFlux`` -- zero precip + zero evap -> moisture
  nearly unchanged (only internal drainage via K).
* ``TestSoilMoistPrecip`` -- throughfall wets the top layer.
* ``TestSoilMoistDew`` -- positive PEVAPTI (dew) wets the top layer.
* ``TestSoilMoistEvap`` -- negative PEVAPTI (evap) dries the top layer.
* ``TestSoilMoistClipping`` -- forced negative moisture is clipped and
  redistributed to deeper layers.
* ``TestSoilMoistSeaMask`` -- surf points keep their t-1 moisture, zero
  runoff.
* ``TestSoilMoistDifferentiable`` -- autograd runs end-to-end.
* ``TestSoilMoistHelpers`` -- small helper sanity checks.
"""

from __future__ import annotations

import math

import pytest
import torch

from surf_pytorch import SurfConfig, ConstantBundle
from surf_pytorch.physics.soil import (
    SoilMoistInputs,
    SoilMoistOutputs,
    compute_soil_moisture,
    _liquid_fraction,
    _root_extraction_weights,
    _vg_hydraulics,
    _vg_hydraulics_toponly,
    _solve_tridiag,
    _soil_thickness,
)
from surf_pytorch.tables.soil import SoilTable, SoilConstants
from surf_pytorch.tables.veg import VegTable


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def consts() -> ConstantBundle:
    return ConstantBundle(dtype=torch.float64)


@pytest.fixture(scope="session")
def soil_table() -> SoilTable:
    return SoilTable(dtype=torch.float64)


@pytest.fixture(scope="session")
def veg_table() -> VegTable:
    return VegTable(dtype=torch.float64)


@pytest.fixture
def config() -> SurfConfig:
    return SurfConfig(dtype=torch.float64)


def _tile_frac(N: int, T: int, *, dtype=torch.float64) -> torch.Tensor:
    """Equal-tile fractions summing to 1 over the T tiles."""
    return torch.full((N, T), 1.0 / T, dtype=dtype)


# --------------------------------------------------------------------------- #
# Helper: build self-consistent SoilMoistInputs.
# --------------------------------------------------------------------------- #
def _build_inputs(
    *,
    N: int = 2,
    soil_levs: int = 4,
    n_tile: int = 8,
    tsoil: float = 285.0,
    wsoil: float = 0.30,
    ksoty: int = 2,            # Medium texture
    ktvl: int = 3,             # 1-indexed (matches FC RVROOTSA(:,KTVL))
    ktvh: int = 1,             # 1-indexed
    psdor: float = 0.0,
    ptsfc: float = 0.0,        # convective throughfall [kg/m2/s]
    ptsfl: float = 0.0,        # large-scale throughfall [kg/m2/s]
    pmsn: float = 0.0,         # snowmelt [kg/m2/s]
    pemssn: float = 0.0,       # snow-evap mismatch [kg/m2/s]
    pevapsnw: float = 0.0,     # snow-under-forest evap [kg/m2/s]
    evap: float = 0.0,         # tile moisture flux (downward +) [kg/m2/s]
    peintti: float = 0.0,      # interception-layer evap mismatch
    land: bool = True,
    ptmst: float = 3600.0,
    dtype=torch.float64,
) -> SoilMoistInputs:
    """Build a self-consistent SoilMoistInputs slab for unit testing."""
    pwsam1m = torch.full((N, soil_levs), wsoil, dtype=dtype)
    ptsam1m = torch.full((N, soil_levs), tsoil, dtype=dtype)

    pfrti = _tile_frac(N, n_tile, dtype=dtype)
    pevapti = torch.full((N, n_tile), evap, dtype=dtype)
    peintti_t = torch.full((N, n_tile), peintti, dtype=dtype)

    ptsfc_t = torch.full((N,), ptsfc, dtype=dtype)
    ptsfl_t = torch.full((N,), ptsfl, dtype=dtype)
    pmsn_t = torch.full((N,), pmsn, dtype=dtype)
    pemssn_t = torch.full((N,), pemssn, dtype=dtype)
    pevapsnw_t = torch.full((N,), pevapsnw, dtype=dtype)

    ksoty_t = torch.full((N,), ksoty, dtype=torch.long)
    ktvl_t = torch.full((N,), ktvl, dtype=torch.long)
    ktvh_t = torch.full((N,), ktvh, dtype=torch.long)
    psdor_t = torch.full((N,), psdor, dtype=dtype)
    ldland = torch.tensor([land] * N, dtype=torch.bool)
    ptmst_t = torch.tensor(ptmst, dtype=dtype)

    return SoilMoistInputs(
        pwsam1m=pwsam1m, ptsam1m=ptsam1m,
        pfrti=pfrti, pevapti=pevapti,
        ptsfc=ptsfc_t, ptsfl=ptsfl_t,
        pmsn=pmsn_t, pemssn=pemssn_t, pevapsnw=pevapsnw_t, peintti=peintti_t,
        ksoty=ksoty_t, ktvl=ktvl_t, ktvh=ktvh_t, psdor=psdor_t,
        ldland=ldland, ptmst=ptmst_t,
    )


# --------------------------------------------------------------------------- #
# 1. Van Genuchten hydraulics
# --------------------------------------------------------------------------- #
class TestSoilMoistVanGenuchten:
    """K and D clamp at the air-dry floor and near saturation, and the
    closed-form VG expressions are reproduced at mid-range moisture."""

    def test_floor_k_zero_d_is_dmin(
        self, soil_table: SoilTable, consts: ConstantBundle,
    ):
        """W <= 1.001*Wres -> K=0, D=Dmin (Dmin may itself be 0)."""
        ksoty = torch.tensor([2], dtype=torch.long)
        wsat = soil_table.gather("wsat", ksoty)
        wrest = soil_table.gather("wrest", ksoty)
        wconsm = soil_table.gather("wconsm", ksoty)
        mvgalpha = soil_table.gather("mvgalpha", ksoty)
        mfac = soil_table.gather("mfac", ksoty)
        lam = getattr(soil_table, "lambda")[ksoty]
        dmax = soil_table.gather("dmax", ksoty)
        dmin = soil_table.gather("dmin", ksoty)

        w_floor = (1.001 * wrest).clone()
        zk, zd = _vg_hydraulics(
            w_floor, wsat, wrest, wconsm, mvgalpha, mfac, lam, dmax, dmin, 1e-14,
        )
        torch.testing.assert_close(zk, torch.zeros_like(zk))
        torch.testing.assert_close(zd, dmin)

    def test_saturation_k_is_wcons_d_is_dmax(
        self, soil_table: SoilTable, consts: ConstantBundle,
    ):
        """W >= 0.999*Wsat -> K=WCONS, D=Dmax."""
        ksoty = torch.tensor([2], dtype=torch.long)
        wsat = soil_table.gather("wsat", ksoty)
        wrest = soil_table.gather("wrest", ksoty)
        wconsm = soil_table.gather("wconsm", ksoty)
        mvgalpha = soil_table.gather("mvgalpha", ksoty)
        mfac = soil_table.gather("mfac", ksoty)
        lam = getattr(soil_table, "lambda")[ksoty]
        dmax = soil_table.gather("dmax", ksoty)
        dmin = soil_table.gather("dmin", ksoty)

        w_sat = (0.999 * wsat).clone()
        zk, zd = _vg_hydraulics(
            w_sat, wsat, wrest, wconsm, mvgalpha, mfac, lam, dmax, dmin, 1e-14,
        )
        torch.testing.assert_close(zk, wconsm)
        torch.testing.assert_close(zd, dmax)

    def test_saturated_branch_does_not_evaluate_invalid_mid_powers(
        self, soil_table: SoilTable,
    ):
        """Super-saturated W must follow the Fortran saturated branch.

        PyTorch evaluates torch.where branches eagerly; without guarding the
        middle-branch Van Genuchten powers, W > WSAT evaluates fractional
        powers of negative intermediates and produces NaN before selection.
        """
        ksoty = torch.tensor([2], dtype=torch.long)
        wsat = soil_table.gather("wsat", ksoty)
        wrest = soil_table.gather("wrest", ksoty)
        wconsm = soil_table.gather("wconsm", ksoty)
        mvgalpha = soil_table.gather("mvgalpha", ksoty)
        mfac = soil_table.gather("mfac", ksoty)
        lam = getattr(soil_table, "lambda")[ksoty]
        dmax = soil_table.gather("dmax", ksoty)
        dmin = soil_table.gather("dmin", ksoty)

        w_super_sat = 1.05 * wsat
        zk, zd = _vg_hydraulics(
            w_super_sat, wsat, wrest, wconsm, mvgalpha, mfac, lam,
            dmax, dmin, 1e-14,
        )
        zkm, _ = _vg_hydraulics_toponly(
            w_super_sat, wsat, wrest, wconsm, mvgalpha, mfac, lam, 1e-14,
        )

        assert bool(torch.isfinite(zk).all())
        assert bool(torch.isfinite(zd).all())
        assert bool(torch.isfinite(zkm).all())
        torch.testing.assert_close(zk, wconsm)
        torch.testing.assert_close(zd, dmax)
        torch.testing.assert_close(zkm, wconsm)

    def test_midrange_matches_closed_form(
        self, soil_table: SoilTable, consts: ConstantBundle,
    ):
        """Mid-range W (well inside the clamp branches) matches the
        closed-form VG expressions written out by hand."""
        ksoty = torch.tensor([2], dtype=torch.long)
        wsat = soil_table.gather("wsat", ksoty)
        wrest = soil_table.gather("wrest", ksoty)
        wconsm = soil_table.gather("wconsm", ksoty)
        mvgalpha = soil_table.gather("mvgalpha", ksoty)
        mfac = soil_table.gather("mfac", ksoty)
        lam = getattr(soil_table, "lambda")[ksoty]
        dmax = soil_table.gather("dmax", ksoty)
        dmin = soil_table.gather("dmin", ksoty)

        # Mid-range moisture (clear of both clamp branches).
        w = (0.5 * (wrest + wsat)).clone()
        zk, zd = _vg_hydraulics(
            w, wsat, wrest, wconsm, mvgalpha, mfac, lam, dmax, dmin, 1e-14,
        )

        # Closed-form reference.  NOTE: the K closed form is
        #   K = WCONS * Se^lam * (1 - (1 - Se^(1/m))^m)^2
        # and the D closed form is
        #   D = ((1-m)*WCONS / (alpha*m*WFAC)) * Se^(lam-1/m)
        #       * ((1-Se^(1/m))^m + (1-Se^(1/m))^(-m) - 2)
        # The two share the building block (1 - Se^(1/m)) but bracket it
        # differently -- this is the exact form in
        # srfwexc_vg_mod.F90 lines 320-325 and is easy to mistype.
        wfac = wsat - wrest
        se = (w - wrest) / wfac
        rmfac = 1.0 / mfac
        se_rmfac = se ** rmfac
        raw_pow = 1.0 - se_rmfac
        zk_ref = wconsm * (se ** lam) * (1.0 - raw_pow ** mfac) ** 2
        denom = mvgalpha * mfac * wfac       # all positive
        zd_ref = (
            ((1.0 - mfac) * wconsm / denom)
            * (se ** (lam - rmfac))
            * (raw_pow ** mfac + raw_pow ** (-mfac) - 2.0)
        )

        torch.testing.assert_close(zk, zk_ref, rtol=1e-12, atol=1e-18)
        torch.testing.assert_close(zd, zd_ref, rtol=1e-12, atol=1e-18)

    def test_toponly_returns_k_only(
        self, soil_table: SoilTable,
    ):
        """_vg_hydraulics_toponly returns K matching the full hydraulics
        block (ZKM never needs D)."""
        ksoty = torch.tensor([3], dtype=torch.long)
        wsat = soil_table.gather("wsat", ksoty)
        wrest = soil_table.gather("wrest", ksoty)
        wconsm = soil_table.gather("wconsm", ksoty)
        mvgalpha = soil_table.gather("mvgalpha", ksoty)
        mfac = soil_table.gather("mfac", ksoty)
        lam = getattr(soil_table, "lambda")[ksoty]
        dmax = soil_table.gather("dmax", ksoty)
        dmin = soil_table.gather("dmin", ksoty)

        w = (0.5 * (wrest + wsat)).clone()
        zk_full, _ = _vg_hydraulics(
            w, wsat, wrest, wconsm, mvgalpha, mfac, lam, dmax, dmin, 1e-14,
        )
        zk_top, _ = _vg_hydraulics_toponly(
            w, wsat, wrest, wconsm, mvgalpha, mfac, lam, 1e-14,
        )
        torch.testing.assert_close(zk_top, zk_full, rtol=1e-12, atol=1e-18)


# --------------------------------------------------------------------------- #
# 2. Liquid fraction (ZF = frozen-water fraction)
# --------------------------------------------------------------------------- #
class TestSoilMoistLiquidFraction:
    """ZF is the *frozen-water* fraction: 1 below RTF2 (cold/frozen),
    0 above RTF1 (warm/liquid), sinusoidal ramp in between."""

    def test_warm_soil_is_all_liquid(self):
        # T = 285 K is well above RTF1=274.16 -- all liquid.
        t = torch.tensor([[285.0, 290.0, 280.0, 285.0]])
        zf = _liquid_fraction(t, 274.16, 270.16, 272.16, 0.7853981633974483)
        torch.testing.assert_close(zf, torch.zeros_like(zf))

    def test_cold_soil_is_all_frozen(self):
        # T = 268 K is below RTF2=270.16 -- all frozen.
        t = torch.tensor([[268.0, 265.0, 260.0, 268.0]])
        zf = _liquid_fraction(t, 274.16, 270.16, 272.16, 0.7853981633974483)
        torch.testing.assert_close(zf, torch.ones_like(zf))

    def test_freeze_window_is_between_zero_and_one(self):
        # T in [RTF2, RTF1] = [270.16, 274.16] -> ZF in (0, 1).
        t = torch.tensor([[271.0, 272.0, 273.0, 273.5]])
        zf = _liquid_fraction(t, 274.16, 270.16, 272.16, 0.7853981633974483)
        assert (zf > 0.0).all() and (zf < 1.0).all()

    def test_freeze_window_midpoint_is_half(self):
        # T = RTF3 = midpoint -> SIN(0)=0 -> ZF = 0.5.
        t = torch.tensor([[272.16, 272.16, 272.16, 272.16]])
        zf = _liquid_fraction(t, 274.16, 270.16, 272.16, 0.7853981633974483)
        torch.testing.assert_close(zf, torch.full_like(zf, 0.5))


# --------------------------------------------------------------------------- #
# 3. Root extraction weights
# --------------------------------------------------------------------------- #
class TestSoilMoistRootExtraction:
    """Dew deposition (positive PEVAPTI) puts all roots in layer 1;
    transpiration distributes them via RVROOTSA."""

    def test_dew_puts_all_roots_in_layer1(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        # Tile 4 with PEVAPTI > 0 (dew): ZROOTW(:,1,4) = 1, deeper = 0.
        N, L, T = 1, 4, 8
        pfrti = _tile_frac(N, T)
        pevapti = torch.zeros((N, T), dtype=torch.float64)
        pevapti[:, 3] = 1.0e-5         # tile 4 (0-indexed 3) has dew
        pevapsnw = torch.zeros((N,), dtype=torch.float64)
        ksoty = torch.full((N,), 2, dtype=torch.long)
        ktvl = torch.full((N,), 3, dtype=torch.long)
        ktvh = torch.full((N,), 1, dtype=torch.long)
        wpwp = soil_table.gather("wpwp", ksoty)

        zf = torch.ones((N, L), dtype=torch.float64) * 0.0   # all liquid
        wpwp_b = wpwp[:, None].expand(-1, L)
        wcap_b = soil_table.gather("wcap", ksoty)[:, None].expand(-1, L)
        zliq = torch.clamp(
            torch.full((N, L), 0.30, dtype=torch.float64) * (1.0 - zf),
            min=wpwp_b, max=wcap_b,
        )

        zrootw, zext = _root_extraction_weights(
            pfrti, pevapti, pevapsnw, zliq, wpwp, zf,
            ktvl, ktvh, ksoty, veg_table, L,
        )
        # Tile 4 (local 0): layer 1 = 1, others = 0.
        torch.testing.assert_close(zrootw[:, 0, 0], torch.ones(N, dtype=torch.float64))
        torch.testing.assert_close(
            zrootw[:, 1:, 0], torch.zeros((N, L - 1), dtype=torch.float64)
        )
        # ZEXT for tile 4 = 1.
        torch.testing.assert_close(zext[:, 0], torch.ones(N, dtype=torch.float64))

    def test_transpiration_distributes_via_root_profile(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        # Tile 6 with PEVAPTI < 0 (transpiration): ZROOTW follows the
        # KTVH root profile scaled by (ZLIQ - RWPWPM).
        N, L, T = 1, 4, 8
        pfrti = _tile_frac(N, T)
        pevapti = torch.zeros((N, T), dtype=torch.float64)
        pevapti[:, 5] = -1.0e-5        # tile 6 (0-indexed 5) has transpiration
        pevapsnw = torch.zeros((N,), dtype=torch.float64)
        ksoty = torch.full((N,), 2, dtype=torch.long)
        ktvh = torch.full((N,), 1, dtype=torch.long)     # type 1
        ktvl = torch.full((N,), 3, dtype=torch.long)
        wpwp = soil_table.gather("wpwp", ksoty)

        zf = torch.zeros((N, L), dtype=torch.float64)    # all liquid
        wpwp_b = wpwp[:, None].expand(-1, L)
        wcap_b = soil_table.gather("wcap", ksoty)[:, None].expand(-1, L)
        zliq = torch.clamp(
            torch.full((N, L), 0.30, dtype=torch.float64) * (1.0 - zf),
            min=wpwp_b, max=wcap_b,
        )

        zrootw, zext = _root_extraction_weights(
            pfrti, pevapti, pevapsnw, zliq, wpwp, zf,
            ktvl, ktvh, ksoty, veg_table, L,
        )

        # Tile 6 (local 1): ZROOTW = roots_hi * (ZLIQ - WPWP).
        roots_hi = veg_table.gather_roots(ktvh)
        expected = roots_hi * (zliq - wpwp_b)
        torch.testing.assert_close(zrootw[:, :, 1], expected, rtol=1e-12, atol=1e-18)
        # ZEXT = sum over layers of ZROOTW.
        torch.testing.assert_close(
            zext[:, 1], expected.sum(dim=1), rtol=1e-12, atol=1e-18,
        )

    def test_zext_clamped_to_zepsilon(
        self, soil_table: SoilTable, veg_table: VegTable,
    ):
        """ZEXT is clamped to >= ZEPSILON to guard the divide."""
        N, L, T = 1, 4, 8
        # Transpiration on tile 6 but ZLIQ=WPWP (so ZROOTW=0): ZEXT would
        # be 0 without the clamp.
        pfrti = _tile_frac(N, T)
        pevapti = torch.zeros((N, T), dtype=torch.float64)
        pevapti[:, 5] = -1.0e-5
        pevapsnw = torch.zeros((N,), dtype=torch.float64)
        ksoty = torch.full((N,), 2, dtype=torch.long)
        ktvh = torch.full((N,), 1, dtype=torch.long)
        ktvl = torch.full((N,), 3, dtype=torch.long)
        wpwp = soil_table.gather("wpwp", ksoty)

        zf = torch.zeros((N, L), dtype=torch.float64)
        wpwp_b = wpwp[:, None].expand(-1, L)
        # ZLIQ = WPWP exactly -> ZROOTW = roots * 0 = 0.
        zliq = wpwp_b.clone()

        _, zext = _root_extraction_weights(
            pfrti, pevapti, pevapsnw, zliq, wpwp, zf,
            ktvl, ktvh, ksoty, veg_table, L,
        )
        zeps = 100.0 * torch.finfo(torch.float64).eps
        assert (zext >= zeps).all()


# --------------------------------------------------------------------------- #
# 4. Runoff
# --------------------------------------------------------------------------- #
class TestSoilMoistRunoff:
    """Runoff branches."""

    def test_zero_precip_vic_zero_runoff(
        self, soil_table: SoilTable, consts: ConstantBundle,
    ):
        """Zero precipitation -> zero VIC runoff (the raining gate)."""
        from surf_pytorch.physics.soil import _runoff
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        rdaw = _soil_thickness(cfg, torch.float64, torch.device("cpu"))
        N = 2
        inp = _build_inputs(
            N=N, ptsfc=0.0, ptsfl=0.0, pmsn=0.0, psdor=0.0,
        )
        ksoty = inp.ksoty
        wsat = soil_table.gather("wsat", ksoty)
        wrest = soil_table.gather("wrest", ksoty)
        wpwp = soil_table.gather("wpwp", ksoty)
        wconsm = soil_table.gather("wconsm", ksoty)
        dmax = soil_table.gather("dmax", ksoty)
        dmin = soil_table.gather("dmin", ksoty)
        zf = _liquid_fraction(
            inp.ptsam1m, 274.16, 270.16, 272.16, 0.7853981633974483,
        )
        zrot = _runoff(
            inp, ksoty, wsat, wrest, wpwp, wconsm, dmax, dmin, zf,
            rdaw, 0.5, True, 3600.0, 1000.0,
            soil_table.consts.rsigormin, soil_table.consts.rsigormax, 1.0,
        )
        torch.testing.assert_close(zrot, torch.zeros_like(zrot))

    def test_infiltration_limit_lower_bound_zero(
        self, soil_table: SoilTable, consts: ConstantBundle,
    ):
        """Infiltration-limit: low precip with high infiltration capacity
        -> zero runoff."""
        from surf_pytorch.physics.soil import _runoff
        cfg = SurfConfig(dtype=torch.float64, lessro=False)
        rdaw = _soil_thickness(cfg, torch.float64, torch.device("cpu"))
        N = 2
        inp = _build_inputs(
            N=N, ptsfc=0.0, ptsfl=1e-8, pmsn=0.0, wsoil=0.20, psdor=0.0,
        )
        ksoty = inp.ksoty
        wsat = soil_table.gather("wsat", ksoty)
        wrest = soil_table.gather("wrest", ksoty)
        wpwp = soil_table.gather("wpwp", ksoty)
        wconsm = soil_table.gather("wconsm", ksoty)
        dmax = soil_table.gather("dmax", ksoty)
        dmin = soil_table.gather("dmin", ksoty)
        zf = _liquid_fraction(
            inp.ptsam1m, 274.16, 270.16, 272.16, 0.7853981633974483,
        )
        zrot = _runoff(
            inp, ksoty, wsat, wrest, wpwp, wconsm, dmax, dmin, zf,
            rdaw, 0.5, False, 3600.0, 1000.0,
            soil_table.consts.rsigormin, soil_table.consts.rsigormax, 1.0,
        )
        torch.testing.assert_close(zrot, torch.zeros_like(zrot))

    def test_infiltration_limit_high_precip_runs_off(
        self, soil_table: SoilTable, consts: ConstantBundle,
    ):
        """Infiltration-limit: precip exceeding infiltration capacity
        runs off (positive)."""
        from surf_pytorch.physics.soil import _runoff
        cfg = SurfConfig(dtype=torch.float64, lessro=False)
        rdaw = _soil_thickness(cfg, torch.float64, torch.device("cpu"))
        N = 2
        # Large convective precip on a near-saturated top layer.
        inp = _build_inputs(
            N=N, ptsfc=1.0e-2, ptsfl=0.0, pmsn=0.0, wsoil=0.40, psdor=0.0,
        )
        ksoty = inp.ksoty
        wsat = soil_table.gather("wsat", ksoty)
        wrest = soil_table.gather("wrest", ksoty)
        wpwp = soil_table.gather("wpwp", ksoty)
        wconsm = soil_table.gather("wconsm", ksoty)
        dmax = soil_table.gather("dmax", ksoty)
        dmin = soil_table.gather("dmin", ksoty)
        zf = _liquid_fraction(
            inp.ptsam1m, 274.16, 270.16, 272.16, 0.7853981633974483,
        )
        zrot = _runoff(
            inp, ksoty, wsat, wrest, wpwp, wconsm, dmax, dmin, zf,
            rdaw, 0.5, False, 3600.0, 1000.0,
            soil_table.consts.rsigormin, soil_table.consts.rsigormax, 1.0,
        )
        assert (zrot >= 0.0).all()
        assert (zrot > 0.0).any()


# --------------------------------------------------------------------------- #
# 5. Zero-flux steady state
# --------------------------------------------------------------------------- #
class TestSoilMoistZeroFlux:
    """Zero precip + zero evap: top-layer moisture is unchanged up to
    internal drainage (which is small at low W)."""

    def test_zero_flux_low_moisture_top_unchanged(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        """At low W (near WPWP) K is tiny -> drainage is negligible ->
        top layer unchanged over one step."""
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        inp = _build_inputs(evap=0.0, ptsfc=0.0, ptsfl=0.0, wsoil=0.17)
        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)
        # Top layer essentially unchanged (very small K).
        torch.testing.assert_close(
            out.pwsa[:, 0], inp.pwsam1m[:, 0], rtol=1e-3, atol=1e-4,
        )

    def test_zero_flux_no_precip_no_runoff(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        """With zero precip the VIC raining gate forces PROS=0 (no surface
        runoff).  PROD can still be non-zero: by design (SURFTSTP section
        4.6) PROD carries the bottom-layer gravity drainage, which
        continues to drain the column even with no precipitation forcing.
        We assert PROS=0 and that PROD equals exactly the bottom drainage
        (no sub-surface runoff component)."""
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        inp = _build_inputs(evap=0.0, ptsfc=0.0, ptsfl=0.0)
        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)
        torch.testing.assert_close(out.pros, torch.zeros_like(out.pros))
        # PROD = bottom drainage only (PTMST * PSAWGFL(:,KLEVS)).
        ptmst = 3600.0
        expected_prod = out.psawgfl[:, -1] * ptmst
        torch.testing.assert_close(out.prod, expected_prod, rtol=1e-12, atol=1e-12)

    def test_zero_flux_mass_conservation(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        """Zero forcing -> change in column water + runoff = 0 (mass
        conservation).  PROD includes bottom drainage."""
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        rdaw = _soil_thickness(cfg, torch.float64, torch.device("cpu"))
        ptmst = 3600.0
        inp = _build_inputs(evap=0.0, ptsfc=0.0, ptsfl=0.0, wsoil=0.20)
        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)

        col_new = (out.pwsa * rdaw).sum(dim=1) * 1000.0
        col_old = (inp.pwsam1m * rdaw).sum(dim=1) * 1000.0
        evap_sum = (inp.pfrti * inp.pevapti).sum(dim=1)
        zrsfl = inp.ptsfl + inp.pmsn + inp.ptsfc
        inflow = (zrsfl + evap_sum) * ptmst
        outflow = out.pros + out.prod
        torch.testing.assert_close(
            (col_new - col_old) + outflow - inflow,
            torch.zeros_like(col_new), rtol=1e-9, atol=1e-9,
        )


# --------------------------------------------------------------------------- #
# 6. Precip wets the column
# --------------------------------------------------------------------------- #
class TestSoilMoistPrecip:
    """Positive throughfall adds water to the column."""

    def test_precip_adds_column_water(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        rdaw = _soil_thickness(cfg, torch.float64, torch.device("cpu"))
        ptmst = 3600.0
        # Start with a moderately dry top layer.
        inp = _build_inputs(
            evap=0.0, ptsfc=0.0, ptsfl=2.78e-6, wsoil=0.20,
        )
        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)

        # Mass budget.
        col_new = (out.pwsa * rdaw).sum(dim=1) * 1000.0
        col_old = (inp.pwsam1m * rdaw).sum(dim=1) * 1000.0
        evap_sum = (inp.pfrti * inp.pevapti).sum(dim=1)
        zrsfl = inp.ptsfl + inp.pmsn + inp.ptsfc
        inflow = (zrsfl + evap_sum) * ptmst
        outflow = out.pros + out.prod
        torch.testing.assert_close(
            (col_new - col_old) + outflow - inflow,
            torch.zeros_like(col_new), rtol=1e-6, atol=1e-6,
        )


# --------------------------------------------------------------------------- #
# 7. Dew vs evaporation
# --------------------------------------------------------------------------- #
class TestSoilMoistDew:
    """Positive PEVAPTI (dew, downward flux) adds water to the column.

    The top layer may still drain faster than dew wets it (gravity K is
    large at moderate W), so the test checks the *column water budget*:
    dew inflow should appear as added column water (modulo runoff).
    """

    def test_dew_adds_column_water(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        rdaw = _soil_thickness(cfg, torch.float64, torch.device("cpu"))
        ptmst = 3600.0
        inp = _build_inputs(
            evap=+1.0e-5, ptsfc=0.0, ptsfl=0.0, wsoil=0.20,
        )  # positive evap = downward = dew
        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)

        # Mass budget: dcol + runoff - inflow = 0.  Only tiles 3, 4, 6, 7, 8
        # contribute to the soil moisture forcing (tiles 1, 2, 5 are water /
        # ice / low-veg-snow and don't extract soil water).  PROD already
        # includes bottom drainage.
        col_new = (out.pwsa * rdaw).sum(dim=1) * 1000.0
        col_old = (inp.pwsam1m * rdaw).sum(dim=1) * 1000.0
        soil_tile_idx = (2, 3, 5, 6, 7)   # 0-indexed tiles 3, 4, 6, 7, 8
        evap_soil = sum(
            inp.pfrti[:, t] * inp.pevapti[:, t] for t in soil_tile_idx
        )
        inflow = evap_soil * ptmst
        outflow = out.pros + out.prod
        torch.testing.assert_close(
            (col_new - col_old) + outflow - inflow,
            torch.zeros_like(col_new), rtol=1e-6, atol=1e-6,
        )


class TestSoilMoistEvap:
    """Negative PEVAPTI (evap, upward flux) removes water from the column."""

    def test_evap_removes_column_water(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        rdaw = _soil_thickness(cfg, torch.float64, torch.device("cpu"))
        ptmst = 3600.0
        inp = _build_inputs(
            evap=-1.0e-5, ptsfc=0.0, ptsfl=0.0, wsoil=0.30,
        )  # negative evap = upward = evaporation
        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)

        # Mass budget.
        col_new = (out.pwsa * rdaw).sum(dim=1) * 1000.0
        col_old = (inp.pwsam1m * rdaw).sum(dim=1) * 1000.0
        soil_tile_idx = (2, 3, 5, 6, 7)
        evap_soil = sum(
            inp.pfrti[:, t] * inp.pevapti[:, t] for t in soil_tile_idx
        )
        inflow = evap_soil * ptmst
        outflow = out.pros + out.prod
        torch.testing.assert_close(
            (col_new - col_old) + outflow - inflow,
            torch.zeros_like(col_new), rtol=1e-6, atol=1e-6,
        )

    def test_evap_dries_top_layer_at_low_moisture(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        """At low W (near WPWP) K is tiny so drainage is negligible and
        evaporation dominates -> top layer dries."""
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        # W just above WPWP for soil type 2 (0.151).
        inp = _build_inputs(
            evap=-1.0e-5, ptsfc=0.0, ptsfl=0.0, wsoil=0.17,
        )
        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)
        assert (out.pwsa[:, 0] < inp.pwsam1m[:, 0]).all()


# --------------------------------------------------------------------------- #
# 8. Clipping (SRFWNG)
# --------------------------------------------------------------------------- #
class TestSoilMoistClipping:
    """Forced negative moisture is clipped and redistributed downward."""

    def test_clip_and_redistribute_negative_to_zero(
        self, soil_table: SoilTable,
    ):
        from surf_pytorch.physics.soil import _clip_and_redistribute
        N, L = 1, 4
        # Force a deeply negative top layer; keep deeper layers mid-range.
        pwsa = torch.tensor([[-0.05, 0.20, 0.30, 0.40]], dtype=torch.float64)
        pwsam1m = torch.tensor([[0.20, 0.25, 0.30, 0.35]], dtype=torch.float64)
        wsat = torch.tensor([0.439], dtype=torch.float64)
        wrest = torch.tensor([0.01], dtype=torch.float64)
        rdaw = torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=torch.float64)
        land = torch.tensor([1.0], dtype=torch.float64)
        ldland = torch.tensor([True], dtype=torch.bool)
        pros = torch.zeros((N,), dtype=torch.float64)
        pwfsd = torch.zeros((N,), dtype=torch.float64)

        pwsa_out, pros_out, prod_out, pwfsd_out = _clip_and_redistribute(
            pwsa, pwsam1m, pros, pwfsd, wsat, wrest, rdaw,
            land, ldland, 3600.0, 1000.0,
        )
        # Clipped: no negative moisture left.
        assert (pwsa_out >= 0.0).all()
        # The negative volume should be redistributed to layer 2.
        # ZDWD = -0.05, push to JK+1: +(-0.05)*rdaw[0]/rdaw[1] = -0.05*0.07/0.21
        # layer 2 was 0.20 -> 0.20 - 0.05*0.07/0.21 = 0.1833...
        expected_l2 = 0.20 - 0.05 * 0.07 / 0.21
        torch.testing.assert_close(
            pwsa_out[0, 1], torch.tensor(expected_l2, dtype=torch.float64),
            rtol=1e-12, atol=1e-18,
        )

    def test_saturation_excess_becomes_runoff(
        self, soil_table: SoilTable,
    ):
        from surf_pytorch.physics.soil import _clip_and_redistribute
        N, L = 1, 4
        # Force super-saturation in the top layer -> clipped to WSAT and
        # the excess becomes surface runoff.
        wsat_val = 0.439
        pwsa = torch.tensor([[0.50, 0.30, 0.30, 0.30]], dtype=torch.float64)
        pwsam1m = torch.tensor([[0.30, 0.30, 0.30, 0.30]], dtype=torch.float64)
        wsat = torch.tensor([wsat_val], dtype=torch.float64)
        wrest = torch.tensor([0.01], dtype=torch.float64)
        rdaw = torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=torch.float64)
        land = torch.tensor([1.0], dtype=torch.float64)
        ldland = torch.tensor([True], dtype=torch.bool)
        pros = torch.zeros((N,), dtype=torch.float64)
        pwfsd = torch.zeros((N,), dtype=torch.float64)

        pwsa_out, pros_out, prod_out, _ = _clip_and_redistribute(
            pwsa, pwsam1m, pros, pwfsd, wsat, wrest, rdaw,
            land, ldland, 3600.0, 1000.0,
        )
        # Top layer clipped to WSAT.
        torch.testing.assert_close(
            pwsa_out[0, 0], torch.tensor(wsat_val, dtype=torch.float64),
            rtol=1e-12, atol=1e-18,
        )
        # PROS gets the saturation excess * RHOH2O * RDAW(1) -- SRFWNG scales
        # runoff by the layer thickness (kg/m^2), not its reciprocal.
        excess = 0.50 - wsat_val
        expected_pros = 0.07 * 1000.0 * excess
        torch.testing.assert_close(
            pros_out[0], torch.tensor(expected_pros, dtype=torch.float64),
            rtol=1e-12, atol=1e-12,
        )

    def test_deep_saturation_excess_is_prod(
        self, soil_table: SoilTable,
    ):
        """Saturation excess in a deeper layer (JK>1) -> PROD, not PROS."""
        from surf_pytorch.physics.soil import _clip_and_redistribute
        N, L = 1, 4
        wsat_val = 0.439
        pwsa = torch.tensor([[0.30, 0.50, 0.30, 0.30]], dtype=torch.float64)
        pwsam1m = torch.tensor([[0.30, 0.30, 0.30, 0.30]], dtype=torch.float64)
        wsat = torch.tensor([wsat_val], dtype=torch.float64)
        wrest = torch.tensor([0.01], dtype=torch.float64)
        rdaw = torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=torch.float64)
        land = torch.tensor([1.0], dtype=torch.float64)
        ldland = torch.tensor([True], dtype=torch.bool)
        pros = torch.zeros((N,), dtype=torch.float64)
        pwfsd = torch.zeros((N,), dtype=torch.float64)

        _, pros_out, prod_out, _ = _clip_and_redistribute(
            pwsa, pwsam1m, pros, pwfsd, wsat, wrest, rdaw,
            land, ldland, 3600.0, 1000.0,
        )
        # PROS unchanged; PROD gets the layer-2 excess.
        torch.testing.assert_close(pros_out, torch.zeros_like(pros_out))
        excess = 0.50 - wsat_val
        expected_prod = 0.21 * 1000.0 * excess
        torch.testing.assert_close(
            prod_out[0], torch.tensor(expected_prod, dtype=torch.float64),
            rtol=1e-12, atol=1e-12,
        )


# --------------------------------------------------------------------------- #
# 9. Sea mask
# --------------------------------------------------------------------------- #
class TestSoilMoistSeaMask:
    """Surf points keep their t-1 moisture and produce no runoff."""

    def test_sea_points_keep_t_minus_1(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        # Mixed slab: first point land, second point sea.
        inp = _build_inputs(N=2, land=True)
        inp.ldland = torch.tensor([True, False], dtype=torch.bool)
        # Set distinct t-1 moisture so we can see the difference.
        inp.pwsam1m[1] = 0.4321
        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)
        # Sea point unchanged.
        torch.testing.assert_close(out.pwsa[1], inp.pwsam1m[1])
        # Sea point runoff zero.
        torch.testing.assert_close(out.pros[1], torch.tensor(0.0, dtype=torch.float64))
        torch.testing.assert_close(out.prod[1], torch.tensor(0.0, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# 10. Differentiability
# --------------------------------------------------------------------------- #
class TestSoilMoistDifferentiable:
    """Autograd runs end-to-end through the soil-moisture solve."""

    def test_gradient_flows_to_soil_moisture(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        cfg = SurfConfig(dtype=torch.float64, differentiable=True)
        inp = _build_inputs(evap=-1.0e-5, ptsfl=1e-6, wsoil=0.25)
        inp.pwsam1m.requires_grad_(True)

        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)
        loss = out.pwsa.sum()
        loss.backward()

        assert inp.pwsam1m.grad is not None
        assert torch.isfinite(inp.pwsam1m.grad).all()

    def test_gradient_flows_to_temperature(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        """Temperature also affects ZF -> K/D -> moisture."""
        cfg = SurfConfig(dtype=torch.float64, differentiable=True)
        inp = _build_inputs(tsoil=272.0, evap=-1.0e-5, wsoil=0.25)
        inp.ptsam1m.requires_grad_(True)

        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)
        loss = out.pwsa.sum()
        loss.backward()

        assert inp.ptsam1m.grad is not None
        assert torch.isfinite(inp.ptsam1m.grad).any()


# --------------------------------------------------------------------------- #
# 11. Helpers
# --------------------------------------------------------------------------- #
class TestSoilMoistHelpers:
    """Small helper sanity checks."""

    def test_output_dataclass_fields(self):
        """SoilMoistOutputs exposes the documented fields."""
        N, L = 1, 4
        out = SoilMoistOutputs(
            pwsa=torch.zeros((N, L)),
            pros=torch.zeros(N),
            prod=torch.zeros(N),
            pwfsd=torch.zeros(N),
            psawgfl=torch.zeros((N, L)),
        )
        assert out.pwsa.shape == (N, L)
        assert out.pros.shape == (N,)
        assert out.prod.shape == (N,)
        assert out.pwfsd.shape == (N,)
        assert out.psawgfl.shape == (N, L)

    def test_compute_moisture_returns_finite(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        """End-to-end call on a representative slab returns finite values."""
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        inp = _build_inputs(
            evap=-1e-5, ptsfc=1e-6, ptsfl=2e-6, wsoil=0.30, tsoil=285.0,
        )
        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)
        assert torch.isfinite(out.pwsa).all()
        assert torch.isfinite(out.pros).all()
        assert torch.isfinite(out.prod).all()
        assert torch.isfinite(out.pwfsd).all()
        assert torch.isfinite(out.psawgfl).all()
        # New moisture within physical range [0, WSAT].
        assert (out.pwsa >= 0.0).all()
        wsat = soil_table.gather("wsat", inp.ksoty)
        assert (out.pwsa <= wsat[:, None] + 1e-9).all()

    def test_compute_moisture_mass_conservation(
        self, soil_table: SoilTable, veg_table: VegTable, consts: ConstantBundle,
    ):
        """Mass conservation: change in column water + runoff = inflow.

        Inflow = (PTSFL + PMSN + PTSFC) * dt + sum(PFRTI*PEVAPTI) * dt
        (the surface forcing terms).  PROD already includes the bottom-
        layer gravity drainage (SURFTSTP section 4.6), so the budget
        closes without an additional term.
        """
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        rdaw = _soil_thickness(cfg, torch.float64, torch.device("cpu"))
        ptmst = 3600.0
        inp = _build_inputs(
            N=2, evap=-1e-5, ptsfc=1e-6, ptsfl=2e-6, wsoil=0.30,
        )
        out = compute_soil_moisture(inp, soil_table, veg_table, consts, cfg)

        # Column total water [kg/m2].
        col_new = (out.pwsa * rdaw).sum(dim=1) * 1000.0
        col_old = (inp.pwsam1m * rdaw).sum(dim=1) * 1000.0
        dcol = col_new - col_old

        # Surface inflow [kg/m2].  Only tiles 3, 4, 6, 7, 8 (0-indexed
        # 2, 3, 5, 6, 7) exchange water with the soil column: water/ice
        # tiles and the low-veg snow tile do not extract soil water.
        zrsfl = inp.ptsfl + inp.pmsn + inp.ptsfc
        soil_tile_idx = (2, 3, 5, 6, 7)
        evap_sum = sum(
            inp.pfrti[:, t] * inp.pevapti[:, t] for t in soil_tile_idx
        )
        inflow = (zrsfl + evap_sum) * ptmst

        # Outflow: surface runoff + sub-surface runoff (PROD already
        # carries bottom drainage).
        outflow = out.pros + out.prod

        torch.testing.assert_close(
            dcol + outflow - inflow,
            torch.zeros_like(dcol), rtol=1e-6, atol=1e-6,
        )
