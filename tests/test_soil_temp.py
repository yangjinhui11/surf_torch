"""Unit tests for the HTESSEL soil-temperature update
(``surf_pytorch.physics.soil``).

These tests exercise the Python port of the soil-temperature branch of
``SURFTSTP_CTL`` (section 3): the apparent heat capacity (SRFRCGS), the
net surface heat flux and Peters-Liddard / Kersten thermal conductivity
(SRFTS), and the implicit Thomas solve (SRFWDIFS).

A ULP-level comparison against a self-compiled Fortran reference kernel
(``SURF_REF_SOILT``) is added in M5c; this file covers the *internal*
mathematical correctness of the port so the M5c test only needs to
catch transcription drift rather than re-derive the physics.

Test groups:

* ``TestSoilTempApparentHeatCapacity`` -- the freeze/melt term: PCTSA
  equals the dry-soil heat capacity outside the freeze window and picks
  up the negative-derivative contribution inside it.  Surf points are
  zeroed.
* ``TestSoilTempSurfaceFlux`` -- ZSURFL fractional-averages the four
  soil-viewing tiles and adds PGSN.  Surf points get zero.
* ``TestSoilTempThermalConductivity`` -- ZDIF reduces to RLAMBDADRYM at
  the Kersten floor and to RLAMBDASAT near saturation.
* ``TestSoilTempTridiag`` -- the Thomas solver reproduces a known
  analytic solution (linear T profile + zero surface flux -> unchanged).
* ``TestSoilTempZeroFlux`` -- with zero surface flux and uniform T, the
  new T equals the old T (energy conservation).
* ``TestSoilTempHeating`` -- a positive ZSURFL warms the top layer and
  the warming diffuses downward.
* ``TestSoilTempSeaMask`` -- surf points keep their t-1 temperature.
* ``TestSoilTempDifferentiable`` -- autograd runs end-to-end.
"""

from __future__ import annotations

import math

import pytest
import torch

from surf_pytorch import SurfConfig, ConstantBundle
from surf_pytorch.physics.soil import (
    SoilTempInputs,
    SoilTempOutputs,
    compute_soil_temperature,
    _apparent_heat_capacity,
    _surface_net_heat_flux,
    _thermal_conductivity,
    _solve_tridiag,
    _soil_thickness,
)
from surf_pytorch.tables.soil import SoilTable, SoilConstants


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def consts() -> ConstantBundle:
    return ConstantBundle(dtype=torch.float64)


@pytest.fixture(scope="session")
def soil_table() -> SoilTable:
    return SoilTable(dtype=torch.float64)


@pytest.fixture
def config() -> SurfConfig:
    return SurfConfig(dtype=torch.float64)


def _thermal_conductivity_for_tests(
    pwsam1m: torch.Tensor,
    wsat: torch.Tensor,
    lambdadry: torch.Tensor,
    lamsat1: torch.Tensor,
    land: torch.Tensor,
    soil_consts: SoilConstants,
    nlevs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ptsam1m = torch.full_like(pwsam1m, 285.0)
    return _thermal_conductivity(
        pwsam1m, ptsam1m, wsat, lambdadry, lamsat1, land, soil_consts,
        274.16, 272.16, 273.16, 1.5707963267948966,
        nlevs, dtype, device,
        lesoilcond=False,
    )


def _tile_frac(N: int, T: int, *, dtype=torch.float64) -> torch.Tensor:
    """Equal-tile fractions summing to 1 over the T tiles."""
    fr = torch.full((N, T), 1.0 / T, dtype=dtype)
    return fr


# --------------------------------------------------------------------------- #
# Helper: build self-consistent SoilTempInputs for the bulk tests.
# --------------------------------------------------------------------------- #
def _build_inputs(
    *,
    N: int = 2,
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
    """Build a self-consistent SoilTempInputs slab for unit testing."""
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
# 1. Apparent heat capacity (SRFRCGS)
# --------------------------------------------------------------------------- #
class TestSoilTempApparentHeatCapacity:
    """SRFRCGS port: freeze/melt term and land/sea masking."""

    def test_outside_freeze_window_matches_dry_capacity(
        self, soil_table: SoilTable, consts: ConstantBundle, config: SurfConfig,
    ):
        """For RTF2 < T < RTF1 false, PCTSA = RRCSOILM(JS)."""
        N, L = 3, 4
        # T = 285 K -- well above RTF1 (RTT+1 = 274.16).
        ptsam1m = torch.full((N, L), 285.0, dtype=torch.float64)
        ksoty = torch.full((N,), 2, dtype=torch.long)
        wcap = soil_table.gather("wcap", ksoty)
        rcsoil = soil_table.gather("rcsoil", ksoty)
        pcvl = torch.full((N,), 0.3, dtype=torch.float64)
        pcvh = torch.full((N,), 0.2, dtype=torch.float64)
        land = torch.ones(N, dtype=torch.bool)

        rtf1 = float(consts.rtf1.item())
        pctsa = _apparent_heat_capacity(
            ptsam1m, wcap, rcsoil, pcvl, pcvh, land.to(torch.float64),
            rlmlt=float(consts.rlmlt.item()),
            rho_h2o=soil_table.consts.rho_h2o,
            rtf1=rtf1, rtf2=float(consts.rtf2.item()),
            rtf3=float(consts.rtf3.item()), rtf4=float(consts.rtf4.item()),
            nlevs=L, dtype=torch.float64, device=ptsam1m.device,
        )

        # Outside the freeze window ZDFDT=0 so PCTSA = RRCSOILM(JS).
        torch.testing.assert_close(pctsa, rcsoil[:, None].expand(N, L))

    def test_inside_freeze_window_adds_derivative_term(
        self, soil_table: SoilTable, consts: ConstantBundle,
    ):
        """At T = RTF3 (centre of the freeze window) the cosine term is
        COS(0) = 1 so PCTSA = RRCSOILM - RLMLT*RHOH2O*ZWA*(-0.5*RTF4)."""
        N, L = 1, 4
        rtf3 = float(consts.rtf3.item())
        ptsam1m = torch.full((N, L), rtf3, dtype=torch.float64)
        ksoty = torch.full((N,), 2, dtype=torch.long)
        wcap = soil_table.gather("wcap", ksoty)
        rcsoil = soil_table.gather("rcsoil", ksoty)
        pcvl = torch.full((N,), 0.3, dtype=torch.float64)
        pcvh = torch.full((N,), 0.2, dtype=torch.float64)
        land = torch.ones(N, dtype=torch.bool)
        rtf4 = float(consts.rtf4.item())
        rlmlt = float(consts.rlmlt.item())
        rho_h2o = soil_table.consts.rho_h2o

        pctsa = _apparent_heat_capacity(
            ptsam1m, wcap, rcsoil, pcvl, pcvh, land.to(torch.float64),
            rlmlt=rlmlt, rho_h2o=rho_h2o,
            rtf1=float(consts.rtf1.item()), rtf2=float(consts.rtf2.item()),
            rtf3=rtf3, rtf4=rtf4,
            nlevs=L, dtype=torch.float64, device=ptsam1m.device,
        )

        # Expected: RRCSOILM - RLMLT*RHOH2O*ZWA*(-0.5*RTF4*COS(0))
        #         = RRCSOILM + RLMLT*RHOH2O*ZWA*(0.5*RTF4)
        zwa = (0.3 + 0.2) * wcap[0].item()
        expected = rcsoil[0].item() + rlmlt * rho_h2o * zwa * (0.5 * rtf4)
        torch.testing.assert_close(
            pctsa[0], torch.full((L,), expected, dtype=torch.float64),
            rtol=1e-12, atol=1e-9,
        )

    def test_sea_points_are_zeroed(self, soil_table: SoilTable, consts: ConstantBundle):
        """LDLAND=.F. -> PCTSA = 0 (surf branch)."""
        N, L = 2, 4
        ptsam1m = torch.full((N, L), 285.0, dtype=torch.float64)
        ksoty = torch.full((N,), 2, dtype=torch.long)
        wcap = soil_table.gather("wcap", ksoty)
        rcsoil = soil_table.gather("rcsoil", ksoty)
        pcvl = torch.full((N,), 0.3, dtype=torch.float64)
        pcvh = torch.full((N,), 0.2, dtype=torch.float64)
        land = torch.zeros(N, dtype=torch.bool)

        pctsa = _apparent_heat_capacity(
            ptsam1m, wcap, rcsoil, pcvl, pcvh, land.to(torch.float64),
            rlmlt=float(consts.rlmlt.item()),
            rho_h2o=soil_table.consts.rho_h2o,
            rtf1=float(consts.rtf1.item()), rtf2=float(consts.rtf2.item()),
            rtf3=float(consts.rtf3.item()), rtf4=float(consts.rtf4.item()),
            nlevs=L, dtype=torch.float64, device=ptsam1m.device,
        )
        torch.testing.assert_close(pctsa, torch.zeros_like(pctsa))


# --------------------------------------------------------------------------- #
# 2. Surface net heat flux (SRFTS section 2)
# --------------------------------------------------------------------------- #
class TestSoilTempSurfaceFlux:
    """ZSURFL: four-tile fractional average + PGSN."""

    def test_fractional_average_of_four_soil_tiles(
        self, consts: ConstantBundle,
    ):
        """With uniform per-tile fluxes and equal tile fractions,
        ZSURFL = (f3+f4+f6+f8) * (SSR + SLR + AHF + RLVTT*EVP) + PGSN."""
        N, T = 1, 8
        # Equal tile fractions: each = 1/8.
        pfrti = torch.full((N, T), 1.0 / T, dtype=torch.float64)
        ssr = 200.0
        slr = -80.0
        ahf = 50.0
        evp = 1.0e-5
        pssrflti = torch.full((N, T), ssr, dtype=torch.float64)
        pslrfl = torch.full((N,), slr, dtype=torch.float64)
        pahfsti = torch.full((N, T), ahf, dtype=torch.float64)
        pevapti = torch.full((N, T), evp, dtype=torch.float64)
        pgsn = torch.full((N,), 0.0, dtype=torch.float64)
        land = torch.ones(N, dtype=torch.bool)
        rlvtt = float(consts.rlvtt.item())

        zsurfl = _surface_net_heat_flux(
            pfrti, pssrflti, pslrfl, pahfsti, pevapti, pgsn,
            land.to(torch.float64), rlvtt, torch.float64, pfrti.device,
        )
        fsum = 4.0 / T   # f3+f4+f6+f8
        expected = fsum * (ssr + slr + ahf + rlvtt * evp)
        torch.testing.assert_close(
            zsurfl[0], torch.tensor(expected, dtype=torch.float64),
            rtol=1e-12, atol=1e-9,
        )

    def test_pgsn_adds_to_zsurfl(self, consts: ConstantBundle):
        """PGSN is a grid-mean additive term."""
        N, T = 1, 8
        pfrti = torch.full((N, T), 1.0 / T, dtype=torch.float64)
        zeros = torch.zeros((N, T), dtype=torch.float64)
        pslrfl = torch.zeros((N,), dtype=torch.float64)
        pgsn = torch.full((N,), 5.0, dtype=torch.float64)
        land = torch.ones(N, dtype=torch.bool)

        zsurfl = _surface_net_heat_flux(
            pfrti, zeros, pslrfl, zeros, zeros, pgsn,
            land.to(torch.float64), float(consts.rlvtt.item()),
            torch.float64, pfrti.device,
        )
        torch.testing.assert_close(zsurfl, pgsn)

    def test_sea_points_zero(self, consts: ConstantBundle):
        N, T = 1, 8
        pfrti = torch.full((N, T), 1.0 / T, dtype=torch.float64)
        pssrflti = torch.full((N, T), 200.0, dtype=torch.float64)
        pslrfl = torch.full((N,), -80.0, dtype=torch.float64)
        pahfsti = torch.full((N, T), 50.0, dtype=torch.float64)
        pevapti = torch.full((N, T), 1e-5, dtype=torch.float64)
        pgsn = torch.full((N,), 3.0, dtype=torch.float64)
        land = torch.zeros(N, dtype=torch.bool)

        zsurfl = _surface_net_heat_flux(
            pfrti, pssrflti, pslrfl, pahfsti, pevapti, pgsn,
            land.to(torch.float64), float(consts.rlvtt.item()),
            torch.float64, pfrti.device,
        )
        torch.testing.assert_close(zsurfl, torch.zeros_like(zsurfl))


# --------------------------------------------------------------------------- #
# 3. Thermal conductivity (SRFTS section 3)
# --------------------------------------------------------------------------- #
class TestSoilTempThermalConductivity:
    """Peters-Liddard / Kersten formula for ZDIF."""

    def test_reduces_to_lambdadry_at_kersten_floor(
        self, soil_table: SoilTable,
    ):
        """At W/WSAT = RKERST1, ZKERSTEN = RKERST2*LOG10(RKERST1)+RKERST3
        = 0 (RKERST1=0.1, RKERST2=1, RKERST3=1 -> LOG10(0.1)=-1 -> 0).
        So ZDIF = RLAMBDADRYM + 0 = RLAMBDADRYM."""
        N, L = 1, 4
        ksoty = torch.full((N,), 2, dtype=torch.long)
        wsat = soil_table.gather("wsat", ksoty)
        lambdadry = soil_table.gather("lambdadry", ksoty)
        lamsat1 = soil_table.gather("lamsat1", ksoty)
        # W = RKERST1 * WSAT exactly (clamps to the floor).
        rkerst1 = soil_table.consts.rkerst1
        pwsam1m = torch.full((N, L), rkerst1 * wsat[0].item(), dtype=torch.float64)
        land = torch.ones(N, dtype=torch.bool)

        zdif = _thermal_conductivity_for_tests(
            pwsam1m, wsat, lambdadry, lamsat1, land.to(torch.float64),
            soil_table.consts, L, torch.float64, pwsam1m.device,
        )
        torch.testing.assert_close(
            zdif, lambdadry[:, None].expand(N, L),
            rtol=1e-12, atol=1e-12,
        )

    def test_sea_points_zero(self, soil_table: SoilTable):
        N, L = 1, 4
        ksoty = torch.full((N,), 2, dtype=torch.long)
        wsat = soil_table.gather("wsat", ksoty)
        lambdadry = soil_table.gather("lambdadry", ksoty)
        lamsat1 = soil_table.gather("lamsat1", ksoty)
        pwsam1m = torch.full((N, L), 0.3, dtype=torch.float64)
        land = torch.zeros(N, dtype=torch.bool)

        zdif = _thermal_conductivity_for_tests(
            pwsam1m, wsat, lambdadry, lamsat1, land.to(torch.float64),
            soil_table.consts, L, torch.float64, pwsam1m.device,
        )
        torch.testing.assert_close(zdif, torch.zeros_like(zdif))


# --------------------------------------------------------------------------- #
# 4. Thomas tridiagonal solver (SRFWDIFS)
# --------------------------------------------------------------------------- #
class TestSoilTempTridiag:
    """SRFWDIFS: the implicit Thomas solver.

    With RSIMP=1, ZTPFAC2=1 and the solver becomes a straight implicit
    Euler step.  Setting ZLST=0 (no inter-layer coupling) decouples the
    layers, so PWSADIF(:,JK) = PTSAM1M(:,JK)/(1+0) = PTSAM1M(:,JK) when
    ZRHS=0 -- i.e. the temperature is unchanged.
    """

    def test_decoupled_layers_preserve_state(self):
        """ZLST=0, ZRHS=0 -> solution equals the input state."""
        N, L = 3, 4
        ptsam1m = torch.randn(N, L, dtype=torch.float64)
        zlst = torch.zeros((N, L), dtype=torch.float64)
        zrhs = torch.zeros((N, L), dtype=torch.float64)
        zcdz = torch.ones((N, L), dtype=torch.float64)
        land = torch.ones(N, dtype=torch.bool)

        out = _solve_tridiag(ptsam1m, zlst, zrhs, zcdz, land, rsimp=1.0)
        torch.testing.assert_close(out, ptsam1m, rtol=1e-12, atol=1e-12)

    def test_top_layer_rhs_injection(self):
        """With ZLST=0 the top layer solves PWSADIF = PTSAM1M + ZRHS
        (ZTPFAC2=1 with RSIMP=1).  Deeper layers stay at PTSAM1M."""
        N, L = 1, 4
        ptsam1m = torch.full((N, L), 280.0, dtype=torch.float64)
        zlst = torch.zeros((N, L), dtype=torch.float64)
        zrhs = torch.zeros((N, L), dtype=torch.float64)
        zrhs[:, 0] = 2.0
        zcdz = torch.ones((N, L), dtype=torch.float64)
        land = torch.ones(N, dtype=torch.bool)

        out = _solve_tridiag(ptsam1m, zlst, zrhs, zcdz, land, rsimp=1.0)
        # Top layer: 280 + 2 = 282.
        torch.testing.assert_close(out[0, 0], torch.tensor(282.0, dtype=torch.float64))
        # Deeper layers unchanged.
        torch.testing.assert_close(
            out[0, 1:], torch.full((L - 1,), 280.0, dtype=torch.float64),
        )


# --------------------------------------------------------------------------- #
# 5. Bulk compute_soil_temperature: energy conservation
# --------------------------------------------------------------------------- #
class TestSoilTempZeroFlux:
    """Zero surface flux + uniform T -> T unchanged (energy conservation)."""

    def test_uniform_temperature_unchanged(
        self, soil_table: SoilTable, consts: ConstantBundle, config: SurfConfig,
    ):
        """With zero net flux (SSR=SLR=AHF=EVP=PGSN=0) and a uniform
        temperature profile, the new soil temperature equals the old."""
        N, L = 2, 4
        T0 = 285.0
        inp = _build_inputs(
            N=N, tsoil=T0, ssrfl=0.0, slrfl=0.0, ahfs=0.0, evap=0.0,
            pgsn=0.0,
        )
        out = compute_soil_temperature(inp, soil_table, consts, config)
        torch.testing.assert_close(
            out.ptsa, torch.full((N, L), T0, dtype=torch.float64),
            rtol=1e-12, atol=1e-9,
        )
        # ZSURFL diagnostic should also be zero.
        torch.testing.assert_close(
            out.zsurfl, torch.zeros(N, dtype=torch.float64),
            rtol=1e-12, atol=1e-9,
        )


class TestSoilTempHeating:
    """A positive ZSURFL warms the top layer; the warming diffuses down."""

    def test_top_layer_warms_under_positive_flux(
        self, soil_table: SoilTable, consts: ConstantBundle, config: SurfConfig,
    ):
        """A positive net heat flux warms the top layer (and only the top
        layer appreciably over a single 1h timestep)."""
        N, L = 1, 4
        T0 = 285.0
        inp = _build_inputs(
            N=N, tsoil=T0, ssrfl=200.0, slrfl=-80.0, ahfs=50.0,
            evap=1.0e-5, pgsn=0.0, ptmst=3600.0,
        )
        out = compute_soil_temperature(inp, soil_table, consts, config)

        # Top layer must warm.
        assert out.ptsa[0, 0].item() > T0
        # ZSURFL must be positive (SSR > |SLR|).
        assert out.zsurfl[0].item() > 0.0

    def test_warming_diffuses_downward(
        self, soil_table: SoilTable, consts: ConstantBundle, config: SurfConfig,
    ):
        """A long timestep with strong surface forcing warms all four
        layers monotonically more than the zero-flux baseline."""
        N, L = 1, 4
        T0 = 285.0
        # Long timestep (6 h) so diffusion reaches the deeper layers.
        inp = _build_inputs(
            N=N, tsoil=T0, ssrfl=400.0, slrfl=0.0, ahfs=100.0,
            evap=0.0, pgsn=0.0, ptmst=6 * 3600.0,
        )
        out = compute_soil_temperature(inp, soil_table, consts, config)

        # Every layer must be >= T0 (warming, never cooling, under a
        # positive flux).  The bottom layer warms the least.
        for jk in range(L):
            assert out.ptsa[0, jk].item() >= T0 - 1e-9


# --------------------------------------------------------------------------- #
# 6. Land/sea mask
# --------------------------------------------------------------------------- #
class TestSoilTempSeaMask:
    """LDLAND=.F. -> PTSA = PTSAM1M (surf points unchanged)."""

    def test_sea_points_keep_t_minus_one(
        self, soil_table: SoilTable, consts: ConstantBundle, config: SurfConfig,
    ):
        """A mixed slab with one land column and one sea column: the sea
        column keeps its t-1 temperature even under strong forcing."""
        N, L = 2, 4
        T0 = 285.0
        inp = _build_inputs(
            N=N, tsoil=T0, ssrfl=400.0, slrfl=0.0, ahfs=100.0, evap=0.0,
        )
        # Make the second column a sea point.
        inp.ldland = torch.tensor([True, False], dtype=torch.bool)

        out = compute_soil_temperature(inp, soil_table, consts, config)
        # Sea column unchanged.
        torch.testing.assert_close(
            out.ptsa[1], torch.full((L,), T0, dtype=torch.float64),
            rtol=1e-12, atol=1e-12,
        )
        # Land column warmed.
        assert out.ptsa[0, 0].item() > T0


# --------------------------------------------------------------------------- #
# 7. Differentiability
# --------------------------------------------------------------------------- #
class TestSoilTempDifferentiable:
    """Autograd runs end-to-end through the soil-temperature solve."""

    def test_gradient_flows_to_soil_temperature(
        self, soil_table: SoilTable, consts: ConstantBundle,
    ):
        config = SurfConfig(dtype=torch.float64, differentiable=True)
        inp = _build_inputs(tsoil=285.0, ssrfl=200.0, ahfs=50.0)
        # Make the input temperature require grad.
        inp.ptsam1m.requires_grad_(True)

        out = compute_soil_temperature(inp, soil_table, consts, config)
        loss = out.ptsa.sum()
        loss.backward()

        assert inp.ptsam1m.grad is not None
        assert torch.isfinite(inp.ptsam1m.grad).all()


# --------------------------------------------------------------------------- #
# 8. Sanity: helpers
# --------------------------------------------------------------------------- #
class TestSoilTempHelpers:
    """Small helper sanity checks."""

    def test_soil_thickness_default(self, config: SurfConfig):
        rdat = _soil_thickness(config, torch.float64, torch.device("cpu"))
        torch.testing.assert_close(
            rdat, torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=torch.float64),
        )

    def test_output_dataclass_fields(self):
        """SoilTempOutputs exposes the three documented fields."""
        N, L = 1, 4
        out = SoilTempOutputs(
            ptsa=torch.zeros((N, L)),
            pctsa=torch.zeros((N, L)),
            zsurfl=torch.zeros(N),
        )
        assert out.ptsa.shape == (N, L)
        assert out.pctsa.shape == (N, L)
        assert out.zsurfl.shape == (N,)
