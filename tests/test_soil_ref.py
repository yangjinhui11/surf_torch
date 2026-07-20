"""ULP comparison of the PyTorch soil kernels against the Fortran ref.

These tests are the M5d closing piece: they feed identical inputs to
:func:`surf_pytorch.physics.soil.compute_soil_temperature` and the
Fortran :class:`offline_driver.SurfRef.soilt` and demand bit-level
agreement.  Any mismatch is purely an arithmetic transcription error
in either implementation.

The tests are gated on ``libsurf_ref.so`` being built and on the
``fortran_ref`` pytest marker -- they are skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from surf_pytorch import ConstantBundle, SurfConfig
from surf_pytorch.physics.soil import (
    SoilTempInputs,
    SoilTempOutputs,
    SoilMoistInputs,
    SoilMoistOutputs,
    compute_soil_temperature as port_soilt,
    compute_soil_moisture as port_soilw,
)
from surf_pytorch.tables import SoilTable, VegTable

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
def soil_table() -> SoilTable:
    return SoilTable(dtype=torch.float64)


@pytest.fixture(scope="session")
def veg_table() -> VegTable:
    return VegTable(dtype=torch.float64)


@pytest.fixture(scope="session")
def consts() -> ConstantBundle:
    return ConstantBundle(dtype=torch.float64)


@pytest.fixture
def config() -> SurfConfig:
    return SurfConfig(dtype=torch.float64, lesoilcond=False)


def requires_fortran_ref():
    return pytest.mark.skipif(
        not _SURF_REF_AVAILABLE,
        reason="libsurf_ref.so not built -- run offline_driver/build_ref.sh",
    )


# --------------------------------------------------------------------------- #
# Input builder -- mirrors tests/test_soil_temp.py::_build_inputs.
# --------------------------------------------------------------------------- #
def _tile_frac(N: int, T: int, dtype=torch.float64) -> torch.Tensor:
    """Plausible tile fractions summing to ~1 over land."""
    if T == 8:
        # IFS 8-tile: water, ice, wet-skin, low-dry, low-snow,
        # high-dry, high-snow, bare.
        f = torch.tensor([
            [0.00, 0.00, 0.05, 0.30, 0.00, 0.30, 0.00, 0.35],
        ], dtype=dtype).expand(N, T).clone()
    else:
        f = torch.full((N, T), 1.0 / T, dtype=dtype)
    return f


def _build_inputs(
    *,
    N: int = 3,
    soil_levs: int = 4,
    n_tile: int = 8,
    tsoil: float = 285.0,
    wsoil: float = 0.30,
    pcvl: float = 0.3,
    pcvh: float = 0.2,
    ksoty: int = 2,        # Medium texture
    pgsn: float = 0.0,
    ssrfl: float = 200.0,
    slrfl: float = -80.0,
    ahfs: float = 50.0,
    evap: float = 1.0e-5,
    land: bool = True,
    ptmst: float = 3600.0,
    dtype=torch.float64,
) -> SoilTempInputs:
    """Build a self-consistent SoilTempInputs slab for ULP comparison."""
    ptsam1m = torch.full((N, soil_levs), tsoil, dtype=dtype)
    pwsam1m = torch.full((N, soil_levs), wsoil, dtype=dtype)

    pfrti = _tile_frac(N, n_tile, dtype=dtype)
    pssrflti = torch.full((N, n_tile), ssrfl, dtype=dtype)
    pahfsti = torch.full((N, n_tile), ahfs, dtype=dtype)
    pevapti = torch.full((N, n_tile), evap, dtype=dtype)
    pslrfl = torch.full((N,), slrfl, dtype=dtype)
    pgsn_t = torch.full((N,), pgsn, dtype=dtype)

    ksoty_t = torch.full((N,), ksoty, dtype=torch.long)
    pcvl_t = torch.full((N,), pcvl, dtype=dtype)
    pcvh_t = torch.full((N,), pcvh, dtype=dtype)
    ldland = torch.tensor([land] * N, dtype=torch.bool)
    ptmst_t = torch.tensor(ptmst, dtype=dtype)

    return SoilTempInputs(
        ptsam1m=ptsam1m, pwsam1m=pwsam1m,
        pfrti=pfrti, pahfsti=pahfsti, pevapti=pevapti, pssrflti=pssrflti,
        pslrfl=pslrfl, pgsn=pgsn_t,
        ksoty=ksoty_t, pcvl=pcvl_t, pcvh=pcvh_t,
        ldland=ldland, ptmst=ptmst_t,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@requires_fortran_ref()
class TestSoilRefSoilt:
    """ULP comparison of the soil-temperature update port vs the ref."""

    def test_default_slab_matches(
        self, ref, soil_table, veg_table, consts, config,
    ):
        """Default land slab: PTSA / PCTSA / ZSURFL agree to ~1 ULP."""
        inp = _build_inputs()
        out_port = port_soilt(inp, soil_table, consts, config)
        out_ref = ref.soilt(inp, soil_table, consts, config)

        torch.testing.assert_close(out_port.ptsa, out_ref.ptsa,
                                   rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(out_port.pctsa, out_ref.pctsa,
                                   rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(out_port.zsurfl, out_ref.zsurfl,
                                   rtol=1e-10, atol=1e-10)

    def test_freezing_window_matches(
        self, ref, soil_table, veg_table, consts, config,
    ):
        """Soil T inside the freeze window exercises the SRFRCGS term."""
        inp = _build_inputs(tsoil=272.5)
        out_port = port_soilt(inp, soil_table, consts, config)
        out_ref = ref.soilt(inp, soil_table, consts, config)
        # Apparent heat capacity carries the freeze/melt coupling; this
        # is the most arithmetic-sensitive field.
        torch.testing.assert_close(out_port.pctsa, out_ref.pctsa,
                                   rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(out_port.ptsa, out_ref.ptsa,
                                   rtol=1e-10, atol=1e-10)

    def test_sea_points_keep_t_minus_1(
        self, ref, soil_table, veg_table, consts, config,
    ):
        """Sea points (LDLAND=False) keep their t-1 temperature."""
        inp = _build_inputs(N=2, land=False)
        out_port = port_soilt(inp, soil_table, consts, config)
        out_ref = ref.soilt(inp, soil_table, consts, config)
        torch.testing.assert_close(out_port.ptsa, out_ref.ptsa,
                                   rtol=1e-12, atol=1e-12)
        # Both must equal the t-1 temperature.
        torch.testing.assert_close(out_port.ptsa, inp.ptsam1m,
                                   rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Soil-moisture input builder -- mirrors tests/test_soil_moist.py::_build_inputs
# but lives here so the ULP test file is self-contained.
# --------------------------------------------------------------------------- #
def _build_moist_inputs(
    *,
    N: int = 2,
    soil_levs: int = 4,
    n_tile: int = 8,
    tsoil: float = 285.0,
    wsoil: float = 0.30,
    ksoty: int = 2,            # Medium texture
    ktvl: int = 3,             # low-veg type (0-based)
    ktvh: int = 1,             # high-veg type (0-based)
    psdor: float = 0.0,
    ptsfc: float = 0.0,        # convective throughfall [kg/m2/s]
    ptsfl: float = 0.0,        # large-scale throughfall [kg/m2/s]
    pmsn: float = 0.0,         # snowmelt [kg/m2/s]
    pemssn: float = 0.0,
    pevapsnw: float = 0.0,
    evap: float = 0.0,
    peintti: float = 0.0,
    land: bool = True,
    ptmst: float = 3600.0,
    dtype=torch.float64,
) -> SoilMoistInputs:
    """Build a self-consistent SoilMoistInputs slab for ULP comparison."""
    pwsam1m = torch.full((N, soil_levs), wsoil, dtype=dtype)
    ptsam1m = torch.full((N, soil_levs), tsoil, dtype=dtype)

    pfrti = torch.full((N, n_tile), 1.0 / n_tile, dtype=dtype)
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


def _assert_moist_close(out_port: SoilMoistOutputs, out_ref: SoilMoistOutputs):
    """Assert every moisture output field agrees to ~1 ULP."""
    for field in ("pwsa", "pros", "prod", "pwfsd", "psawgfl"):
        torch.testing.assert_close(
            getattr(out_port, field), getattr(out_ref, field),
            rtol=1e-10, atol=1e-10,
        )


@requires_fortran_ref()
class TestSoilRefSoilw:
    """ULP comparison of the soil-moisture update port vs the ref.

    Each test feeds identical inputs to :func:`port_soilw` and
    :meth:`offline_driver.SurfRef.soilw` and demands bit-level agreement
    across all five output fields (PWSA, PROS, PROD, PWFSD, PSAWGFL).
    """

    def test_default_slab_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """Default land slab: medium-texture soil, near-field-capacity W."""
        inp = _build_moist_inputs()
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    def test_freeze_window_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """Soil T inside the freeze window exercises the ZF liquid-fraction
        ramp -- the most arithmetic-sensitive branch of the VG hydraulics."""
        inp = _build_moist_inputs(tsoil=272.5)
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    def test_frozen_soil_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """Fully frozen soil (T < RTF2): ZF=1, all water locked as ice,
        K/D reduced to the frozen-soil branch."""
        inp = _build_moist_inputs(tsoil=265.0)
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    def test_near_saturation_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """W just below WSAT exercises the high-moisture VG branch but no
        SRFWNG saturation-excess clipping yet."""
        inp = _build_moist_inputs(wsoil=0.43)
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    def test_super_saturated_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """W above WSAT triggers SRFWNG saturation-excess runoff split
        (PROS for top layer, PROD for deep layers)."""
        inp = _build_moist_inputs(wsoil=0.45)
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    def test_air_dry_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """W below WPWP exercises the air-dry floor of the VG hydraulics
        (K=0, D=Dmin)."""
        inp = _build_moist_inputs(wsoil=0.05)
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    @pytest.mark.parametrize("ksoty", [1, 3, 5, 7])
    def test_soil_types_match(
        self, ref, soil_table, veg_table, consts, ksoty,
    ):
        """All 7 IFS soil types select distinct per-class VG LUT columns;
        the gather path must match the Fortran 0-based indexing."""
        inp = _build_moist_inputs(ksoty=ksoty)
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    def test_sea_points_match(
        self, ref, soil_table, veg_table, consts,
    ):
        """Sea points (LDLAND=False) keep t-1 moisture, zero runoff."""
        inp = _build_moist_inputs(N=2, land=False)
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)
        # Both must equal the t-1 moisture.
        torch.testing.assert_close(out_port.pwsa, inp.pwsam1m,
                                   rtol=1e-12, atol=1e-12)

    def test_infiltration_limit_dry_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """LESSRO=.FALSE. infiltration-limit runoff: low precip on a dry
        top layer -> near-zero runoff."""
        inp = _build_moist_inputs(wsoil=0.20, ptsfl=1e-8)
        cfg = SurfConfig(dtype=torch.float64, lessro=False)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    def test_infiltration_limit_wet_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """LESSRO=.FALSE. infiltration-limit runoff: large convective
        precip on a near-saturated top layer -> runoff runs off."""
        inp = _build_moist_inputs(wsoil=0.40, ptsfc=1.0e-2)
        cfg = SurfConfig(dtype=torch.float64, lessro=False)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    def test_vic_runoff_flat_orography_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """LESSRO=.TRUE. VIC runoff: PSDOR=0 (flat slab) yields ZROEFF<0
        which clamps to ZBWS=0.01 -- exercises the RSIGORMIN/MAX defaults."""
        inp = _build_moist_inputs(psdor=0.0, ptsfc=1e-3, ptsfl=1e-3)
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    def test_vic_runoff_mountainous_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """LESSRO=.TRUE. VIC runoff: PSDOR=1000 (mountainous) yields
        ZROEFF in the interior of the [0.01, 0.5] clamp."""
        inp = _build_moist_inputs(psdor=1000.0, ptsfc=1e-3, ptsfl=1e-3)
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    def test_snowmelt_forcing_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """Non-zero PMSN adds to ZRSFL (the throughfall + snowmelt forcing
        that drives runoff and surface infiltration)."""
        inp = _build_moist_inputs(pmsn=1.0e-5, ptsfc=0.0, ptsfl=0.0)
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)

    def test_transpiration_matches(
        self, ref, soil_table, veg_table, consts,
    ):
        """Negative evap (transpiration) distributes root extraction across
        all layers via the per-veg-type RVROOTSA profile."""
        inp = _build_moist_inputs(evap=2.0e-5)
        cfg = SurfConfig(dtype=torch.float64, lessro=True)
        out_port = port_soilw(inp, soil_table, veg_table, consts, cfg)
        out_ref = ref.soilw(inp, soil_table, veg_table, consts, cfg)
        _assert_moist_close(out_port, out_ref)
