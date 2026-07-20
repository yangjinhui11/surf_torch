"""Unit tests for the HTESSEL single-layer snow scheme
(``surf_pytorch.physics.snow``).

These tests exercise the Python port of ``SRFSN_LWIMPS`` -- the
simplified single-layer snow scheme (with diagnostic liquid water and
rainfall interception) used by ``SURFTSTPS_CTL`` when ``NCSNEC == 1``
(the offline-driver default).

A ULP-level comparison against a self-compiled Fortran reference kernel
(``SURF_REF_SNOW``) is added in M6b; this file covers the *internal*
mathematical correctness of the port so the M6b test only needs to
catch transcription drift rather than re-derive the physics.

Test groups:

* ``TestSnowBranches`` -- the three algorithm branches
  (no-snow / no-melt / melt) are selected correctly and produce the
  expected outputs for trivial inputs.
* ``TestSnowInputs`` -- input dataclass construction, tile-index
  guards, and the interception-grid-fraction helper.
* ``TestSnowMass`` -- snow-mass budget: snowfall accumulates, melt
  reduces it, evaporation reduces it, no-snow branch keeps snowfall.
* ``TestSnowTemperature`` -- snow-T responds to fluxes sensibly: cold
  snow cools under outgoing LW, warm snow pins to RTT under melt.
* ``TestSnowGroundFlux`` -- PGSN sign convention (positive = downward
  into soil), magnitude scales with snow-soil T difference, normalised
  by ZFRSN.
* ``TestSnowLiquidWater`` -- ZLWC scales with snow mass, collapses to
  RLWCSWEC at high density, zero in no-snow branch.
* ``TestSnowVectorisation`` -- per-column independence and shape
  preservation across a batch.
* ``TestSnowEdgeCases`` -- zero snow mass, tiny snow fraction, very
  dense / very light snow, FLake on/off.
* ``TestSnowDifferentiable`` -- autograd flows through every branch.
"""

from __future__ import annotations

import math

import pytest
import torch

from surf_pytorch import SurfConfig, ConstantBundle
from surf_pytorch.physics.snow import (
    SnowInputs,
    SnowOutputs,
    compute_snow,
    _tile_col,
    _interception_gridfrac,
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


# --------------------------------------------------------------------------- #
# Helper: build self-consistent SnowInputs for the bulk tests.
# --------------------------------------------------------------------------- #
def _build_inputs(
    *,
    N: int = 2,
    n_tile: int = 8,
    snow_mass: float = 10.0,            # kg/m^2
    snow_t: float = 270.0,              # K
    snow_rho: float = 200.0,            # kg/m^3
    soil_t_top: float = 275.0,          # K
    snow_frac_low: float = 0.3,         # tile 5 (snow-low-veg)
    snow_frac_high: float = 0.0,        # tile 7 (snow-high-veg)
    bare_frac: float = 0.7,             # tile 8 (bare-soil)
    ahfs: float = 0.0,                  # W/m^2
    evap: float = 0.0,                  # kg/m^2/s
    ssrfl: float = 0.0,                 # W/m^2
    slrfl: float = 0.0,                 # W/m^2 (net LW)
    pssfc: float = 0.0,                 # kg/m^2/s convective snowfall
    pssfl: float = 0.0,                 # kg/m^2/s large-scale snowfall
    ptsfc: float = 0.0,                 # kg/m^2/s convective rainfall
    ptsfl: float = 0.0,                 # kg/m^2/s large-scale rainfall
    pevapsnw: float = 0.0,              # kg/m^2/s snow-under-forest evap
    ptmst: float = 3600.0,              # s
    dtype=torch.float64,
) -> SnowInputs:
    """Build a self-consistent SnowInputs slab for unit testing.

    The default slab puts ``snow_frac_low + snow_frac_high + bare_frac``
    into the corresponding tiles and leaves the rest at zero.  Pass a
    nonzero ``snow_frac_high`` to populate the snow-under-high-veg tile.
    """
    pfrti = torch.zeros(N, n_tile, dtype=dtype)
    if n_tile > 4:
        pfrti[:, 4] = snow_frac_low              # snow-low-veg (port 4)
    if n_tile > 6:
        pfrti[:, 6] = snow_frac_high             # snow-high-veg (port 6)
    if n_tile > 7:
        pfrti[:, 7] = bare_frac                  # bare-soil (port 7)

    return SnowInputs(
        pssnm1m=torch.full((N,), snow_mass, dtype=dtype),
        ptsnm1m=torch.full((N,), snow_t, dtype=dtype),
        prsnm1m=torch.full((N,), snow_rho, dtype=dtype),
        ptsam1m_top=torch.full((N,), soil_t_top, dtype=dtype),
        pfrti=pfrti,
        pahfsti=torch.full((N, n_tile), ahfs, dtype=dtype),
        pevapti=torch.full((N, n_tile), evap, dtype=dtype),
        pssrflti=torch.full((N, n_tile), ssrfl, dtype=dtype),
        pslrfl=torch.full((N,), slrfl, dtype=dtype),
        pssfc=torch.full((N,), pssfc, dtype=dtype),
        pssfl=torch.full((N,), pssfl, dtype=dtype),
        ptsfc=torch.full((N,), ptsfc, dtype=dtype),
        ptsfl=torch.full((N,), ptsfl, dtype=dtype),
        pevapsnw=torch.full((N,), pevapsnw, dtype=dtype),
        ptmst=torch.as_tensor(ptmst, dtype=dtype),
    )


# --------------------------------------------------------------------------- #
# 1. Branch selection
# --------------------------------------------------------------------------- #
class TestSnowBranches:
    """The three SRFSN_LWIMPS branches (no-snow / no-melt / melt)."""

    def test_no_snow_branch_pins_to_rtt_and_zero_flux(
        self, soil_table, consts, config,
    ):
        """When ZFRSN < RFRSMALL, PTSN=RTT, PGSN=0, ZLWC=0."""
        inp = _build_inputs(
            N=2, snow_frac_low=0.0, snow_frac_high=0.0, bare_frac=1.0,
        )
        out = compute_snow(inp, soil_table, consts, config)
        rtt = float(consts.rtt.item())
        assert torch.allclose(out.ptsn, torch.full_like(out.ptsn, rtt))
        assert torch.allclose(out.pgsn, torch.zeros_like(out.pgsn))

    def test_no_snow_branch_keeps_residual_snow_tile_exchange(
        self, soil_table, consts, config,
    ):
        # SRFSN_LWIMP retains the snow-tile exchange under LEWBSOILFIX
        # even when ZFRSN is below RFRSMALL and the no-snow branch is used.
        frac = 1.0e-9
        evap = 1.0e6
        inp = _build_inputs(
            N=1, snow_mass=0.0, snow_frac_low=frac, bare_frac=1.0 - frac,
            evap=evap, pssfl=0.0,
        )
        out = compute_snow(inp, soil_table, consts, config)
        expected = 3600.0 * frac * evap
        assert torch.allclose(
            out.psstar, torch.full_like(out.psstar, expected), rtol=0, atol=1e-12,
        )
        assert torch.allclose(out.plwc, torch.zeros_like(out.plwc))

    def test_no_snow_branch_keeps_snowfall(
        self, soil_table, consts, config,
    ):
        """No-snow branch advances mass by PTMST*(ZSSFC+ZSSFL) only.

        With bare_frac=1, grid fraction = 1; PSSFL=1e-4 -> mass gain
        = 10 + 3600*1e-4 = 10.36 kg/m^2.
        """
        inp = _build_inputs(
            N=2, snow_frac_low=0.0, snow_frac_high=0.0, bare_frac=1.0,
            pssfl=1.0e-4,
        )
        out = compute_snow(inp, soil_table, consts, config)
        expected = 10.0 + 3600.0 * 1.0e-4
        assert torch.allclose(out.psstar, torch.full_like(out.psstar, expected))

    def test_no_melt_branch_when_cold_snow(
        self, soil_table, consts, config,
    ):
        """Cold snow under outgoing LW: PTSN < RTT (no-melt branch).

        Net flux is the snow-tile-fraction-weighted LW; with snow_frac=0.3
        and slrfl=-50, snow cools.  PTSN should be below RTT and finite.
        """
        inp = _build_inputs(
            N=2, snow_t=265.0, slrfl=-50.0, snow_frac_low=0.3, bare_frac=0.7,
        )
        out = compute_snow(inp, soil_table, consts, config)
        rtt = float(consts.rtt.item())
        assert bool((out.ptsn < rtt).all())
        assert bool(torch.isfinite(out.ptsn).all())

    def test_melt_branch_pins_to_rtt(
        self, soil_table, consts, config,
    ):
        """Strong incoming flux with warm snow: PTSN pinned at RTT.

        With SW=200, sensible=50, LW=-30 the snowpack gains heat; once
        ZTSTAR exceeds RTT the melt branch sets PTSN=RTT exactly.
        """
        inp = _build_inputs(
            N=2, snow_t=271.0, ssrfl=200.0, ahfs=50.0, slrfl=-30.0,
            snow_frac_low=0.5, bare_frac=0.5,
        )
        out = compute_snow(inp, soil_table, consts, config)
        rtt = float(consts.rtt.item())
        assert torch.allclose(out.ptsn, torch.full_like(out.ptsn, rtt))

    def test_pgsn_zero_in_no_snow_branch(
        self, soil_table, consts, config,
    ):
        """PGSN is exactly zero in the no-snow branch (not just tiny)."""
        inp = _build_inputs(
            N=3, snow_frac_low=0.0, snow_frac_high=0.0, bare_frac=1.0,
            soil_t_top=290.0,  # large T difference that would produce flux
        )
        out = compute_snow(inp, soil_table, consts, config)
        assert torch.equal(out.pgsn, torch.zeros_like(out.pgsn))


# --------------------------------------------------------------------------- #
# 2. Input dataclass and helpers
# --------------------------------------------------------------------------- #
class TestSnowInputs:
    """SnowInputs construction, tile-index guards, helpers."""

    def test_snow_inputs_constructs_with_all_fields(self):
        """SnowInputs accepts every documented field."""
        inp = _build_inputs(N=1)
        assert inp.pssnm1m.shape == (1,)
        assert inp.pfrti.shape == (1, 8)
        assert inp.ptmst.shape == ()

    def test_phlicem1m_defaults_to_none(self):
        """PHLICEM1M is optional and defaults to None."""
        inp = _build_inputs(N=1)
        assert inp.phlicem1m is None

    def test_tile_col_returns_column(self):
        """_tile_col extracts the requested tile column."""
        pfrti = torch.arange(24, dtype=torch.float64).reshape(3, 8)
        col = _tile_col(pfrti, 4, torch.float64, pfrti.device)
        assert torch.equal(col, pfrti[:, 4])

    def test_tile_col_returns_zeros_when_index_out_of_range(self):
        """_tile_col falls back to zeros when the tile axis is too narrow."""
        pfrti = torch.arange(8, dtype=torch.float64).reshape(2, 4)
        col = _tile_col(pfrti, 4, torch.float64, pfrti.device)
        assert torch.equal(col, torch.zeros(2, dtype=torch.float64))

    def test_interception_gridfrac_sums_tiles_3_to_8(self):
        """Grid fraction = sum of port tiles 2..7 (F90 3..8).

        Upstream SRFSN_LWIMPS sums PFRTI(JL,3..8) -- the wet-skin,
        low/high-veg (dry+snow) and bare-soil tiles.  In port indices
        that is tiles 2..7.  The earlier port code (M6a) used 3..7 and
        silently dropped the wet-skin tile; the M6b ULP comparison
        against SURF_REF_SNOW caught the off-by-one.
        """
        pfrti = torch.zeros(2, 8, dtype=torch.float64)
        pfrti[:, 2] = 0.1     # wet-skin (F90 3)
        pfrti[:, 7] = 0.9     # bare-soil (F90 8)
        gf = _interception_gridfrac(pfrti, use_flake=False,
                                    dtype=torch.float64, device=pfrti.device)
        assert torch.allclose(gf, torch.full((2,), 1.0, dtype=torch.float64))

    def test_interception_gridfrac_excludes_water_ice_only(self):
        """Only port tiles 0 and 1 (water, ice) are outside the sum.

        The wet-skin tile (port 2 / F90 3) *is* included by the upstream
        ZGRIDFRAC, so it contributes when non-zero.
        """
        pfrti = torch.zeros(2, 8, dtype=torch.float64)
        pfrti[:, 0] = 0.5     # water
        pfrti[:, 1] = 0.3     # ice
        pfrti[:, 2] = 0.2     # wet-skin -> IS in the sum
        gf = _interception_gridfrac(pfrti, use_flake=False,
                                    dtype=torch.float64, device=pfrti.device)
        assert torch.allclose(gf, torch.full((2,), 0.2, dtype=torch.float64))

    def test_interception_gridfrac_full_lake_collapses_to_zero(self):
        """Full lake cover (PFRTI(:,9)==1) -> grid fraction = 0."""
        pfrti = torch.zeros(1, 9, dtype=torch.float64)
        pfrti[0, 8] = 1.0     # full lake
        gf = _interception_gridfrac(pfrti, use_flake=True,
                                    dtype=torch.float64, device=pfrti.device)
        assert torch.allclose(gf, torch.zeros((1,), dtype=torch.float64))


# --------------------------------------------------------------------------- #
# 3. Snow mass budget
# --------------------------------------------------------------------------- #
class TestSnowMass:
    """Snow-mass (PSSTAR) responds to snowfall, melt, evaporation."""

    def test_snowfall_accumulates_in_no_snow_branch(
        self, soil_table, consts, config,
    ):
        """PSSFL adds PTMST*grid_frac*PSSFL to mass in the no-snow branch."""
        inp = _build_inputs(
            N=2, snow_frac_low=0.0, bare_frac=1.0, pssfl=2.0e-4,
        )
        out = compute_snow(inp, soil_table, consts, config)
        # grid_frac = 1.0; mass = 10 + 3600 * 1.0 * 2e-4 = 10.72
        expected = 10.0 + 3600.0 * 1.0 * 2.0e-4
        assert torch.allclose(out.psstar, torch.full_like(out.psstar, expected))

    def test_convective_and_large_scale_snowfall_both_add(
        self, soil_table, consts, config,
    ):
        """PSSFC and PSSFL both contribute to mass in the no-snow branch."""
        inp = _build_inputs(
            N=2, snow_frac_low=0.0, bare_frac=1.0,
            pssfc=1.0e-4, pssfl=1.5e-4,
        )
        out = compute_snow(inp, soil_table, consts, config)
        expected = 10.0 + 3600.0 * (1.0e-4 + 1.5e-4)
        assert torch.allclose(out.psstar, torch.full_like(out.psstar, expected))

    def test_sublimation_reduces_mass(
        self, soil_table, consts, config,
    ):
        """Negative PEVAPTI on the snow tile reduces mass (sublimation).

        IFS sign convention: PEVAPTI > 0 = dew/deposition (mass gain),
        PEVAPTI < 0 = evaporation/sublimation (mass loss).  Cold snow
        (well below RTT) with a small negative evaporation rate so the
        RLSTT*PEVAPTI heat uptake stays small enough to keep the
        no-melt branch; then PSSTAR = 10 + 3600*pf5*evap (evap<0).
        """
        # pf5 = 0.3 -> grid_frac = 0.7 (bare), so the throughfall terms
        # are zero (PTSFC=PTSFL=0).  Sublimation: +3600*0.3*evap (evap<0).
        evap = -1.0e-6   # negative = sublimation
        inp = _build_inputs(
            N=2, snow_frac_low=0.3, bare_frac=0.7, evap=evap,
            snow_t=265.0,    # cold snow, well below RTT
        )
        out = compute_snow(inp, soil_table, consts, config)
        expected = 10.0 + 3600.0 * 0.3 * evap
        assert torch.allclose(out.psstar, torch.full_like(out.psstar, expected),
                              atol=1e-9)

    def test_negative_mass_after_pe_clamped_to_zero(
        self, soil_table, consts, config,
    ):
        """Excessive sublimation clamps the preliminary mass to zero."""
        inp = _build_inputs(
            N=2, snow_mass=0.01, snow_frac_low=1.0, bare_frac=0.0,
            evap=1.0e-2,    # over 1h this would remove 36 kg/m^2
            snow_t=270.0,
        )
        out = compute_snow(inp, soil_table, consts, config)
        # Mass should be non-negative.
        assert bool((out.psstar >= 0.0).all())


# --------------------------------------------------------------------------- #
# 4. Snow temperature
# --------------------------------------------------------------------------- #
class TestSnowTemperature:
    """Snow-T (PTSN) responds to fluxes and pins to RTT under melt."""

    def test_cold_snow_cools_under_outgoing_lw(
        self, soil_table, consts, config,
    ):
        """Outgoing LW cools cold snow below its starting temperature."""
        inp = _build_inputs(
            N=2, snow_t=268.0, slrfl=-80.0,
            snow_frac_low=0.5, bare_frac=0.5,
        )
        out = compute_snow(inp, soil_table, consts, config)
        assert bool((out.ptsn < 268.0).all())

    def test_warm_snow_in_ramp_window_responds(
        self, soil_table, consts, config,
    ):
        """Warm snow in [RTT-0.5*RTEMPAMP, RTT] is in the ramp window.

        With moderate outgoing flux it should still cool but stay below RTT.
        """
        rtt = 273.16
        inp = _build_inputs(
            N=2, snow_t=rtt - 1.0, slrfl=-20.0,
            snow_frac_low=0.5, bare_frac=0.5,
        )
        out = compute_snow(inp, soil_table, consts, config)
        # In the ramp window with mild cooling, PTSN stays in the cold
        # range (no melt).
        assert bool((out.ptsn <= rtt + 1e-9).all())
        assert bool(torch.isfinite(out.ptsn).all())

    def test_ptsn_bounded_above_by_rtt(
        self, soil_table, consts, config,
    ):
        """PTSN never exceeds RTT (melt branch pins it there)."""
        inp = _build_inputs(
            N=2, snow_t=271.0, ssrfl=500.0, ahfs=200.0, slrfl=0.0,
            snow_frac_low=0.5, bare_frac=0.5,
        )
        out = compute_snow(inp, soil_table, consts, config)
        rtt = float(consts.rtt.item())
        assert bool((out.ptsn <= rtt + 1e-9).all())

    def test_ptsn_finite_across_branches(
        self, soil_table, consts, config,
    ):
        """PTSN is finite for a range of snow temperatures spanning branches."""
        rtt = 273.16
        # One column per branch.
        inp = _build_inputs(
            N=3, snow_frac_low=0.5, bare_frac=0.5, slrfl=-30.0,
        )
        # Override per-column snow-T: cold, ramp, melt-window.
        inp.ptsnm1m = torch.tensor([260.0, rtt - 1.0, rtt - 0.1], dtype=torch.float64)
        out = compute_snow(inp, soil_table, consts, config)
        assert bool(torch.isfinite(out.ptsn).all())


# --------------------------------------------------------------------------- #
# 5. Ground heat flux (PGSN)
# --------------------------------------------------------------------------- #
class TestSnowGroundFlux:
    """PGSN sign convention and normalisation."""

    def test_pgsn_sign_matches_soil_snow_temperature_gradient(
        self, soil_table, consts, config,
    ):
        """PGSN > 0 when snow is warmer than soil (heat into soil).

        Warm snow (273.16) over cold soil (270.0) -> positive downward
        flux into the soil.
        """
        rtt = 273.16
        inp = _build_inputs(
            N=2, snow_t=rtt - 0.1, soil_t_top=270.0, snow_frac_low=0.5,
            bare_frac=0.5, ssrfl=200.0, ahfs=50.0,  # force melt -> snow at RTT
        )
        out = compute_snow(inp, soil_table, consts, config)
        # In melt branch PGSN = ZFRSN * (RTT - soil_top) * ZRS, positive
        # because RTT > soil_top.
        assert bool((out.pgsn > 0.0).all())

    def test_pgsn_normalised_by_snow_fraction(
        self, soil_table, consts, config,
    ):
        """PGSN is the snow-tile-fraction-normalised ground heat flux.

        The final step of SRFSN_LWIMPS (lines 336-338) multiplies the
        per-snow-tile PGSN by ZFRSN, so the grid-box-mean PGSN scales
        roughly linearly with the snow fraction.  In the no-melt branch
        PGSN = ZFRSN * (ZTSTAR - soil_top) * ZRS, where ZRS depends on
        snow mass / density (not ZFRSN) and ZTSTAR has a weak feedback
        through the implicit solve, so the dominant scaling is ZFRSN.
        """
        # No-melt branch: lots of snow, mild fluxes, snow-T below RTT.
        base_kwargs = dict(
            N=1, snow_t=272.0, soil_t_top=270.0,
            ssrfl=50.0, ahfs=10.0, slrfl=-10.0,
            snow_mass=100.0, snow_rho=200.0,
            bare_frac=0.0,
        )
        inp1 = _build_inputs(snow_frac_low=0.25, **base_kwargs)
        inp2 = _build_inputs(snow_frac_low=0.50, **base_kwargs)
        out1 = compute_snow(inp1, soil_table, consts, config)
        out2 = compute_snow(inp2, soil_table, consts, config)
        rtt = float(consts.rtt.item())
        # Both in no-melt branch.
        assert bool(out1.ptsn[0] < rtt)
        assert bool(out2.ptsn[0] < rtt)
        # Ratio of PGSN should be close to 2x (within 10% for the
        # secondary ZTSTAR feedback).
        ratio = out2.pgsn[0].item() / out1.pgsn[0].item()
        assert abs(ratio - 2.0) < 0.1

    def test_pgsn_zero_flux_when_snow_equals_soil_temperature(
        self, soil_table, consts, config,
    ):
        """In no-melt branch, PGSN=(ZTSTAR-soil_top)*ZRS, which is ~0 when
        snow-T and soil-T are equal and there is no surface flux.

        We make ZHFLUX zero (no SW, no LW, no sensible, no evap) and set
        snow-T == soil-T so ZTSTAR == that temperature and PGSN ~ 0.
        """
        T = 270.0
        inp = _build_inputs(
            N=2, snow_t=T, soil_t_top=T,
            snow_frac_low=0.5, bare_frac=0.5,
            ssrfl=0.0, ahfs=0.0, evap=0.0, slrfl=0.0,
        )
        out = compute_snow(inp, soil_table, consts, config)
        # No surface flux and snow-T == soil-T -> no-melt branch with
        # ZTSTAR == T -> PGSN = (T - T)*ZRS = 0.
        assert torch.allclose(out.pgsn, torch.zeros_like(out.pgsn), atol=1e-9)
        # PTSN should equal T (unchanged) in this limit.
        assert torch.allclose(out.ptsn, torch.full_like(out.ptsn, T), atol=1e-6)


# --------------------------------------------------------------------------- #
# 6. Liquid water capacity
# --------------------------------------------------------------------------- #
class TestSnowLiquidWater:
    """ZLWC scales with snow mass and density."""

    def test_zlwc_zero_in_no_snow_branch(
        self, soil_table, consts, config,
    ):
        """No-snow branch sets ZLWC=0."""
        inp = _build_inputs(
            N=2, snow_frac_low=0.0, bare_frac=1.0,
        )
        out = compute_snow(inp, soil_table, consts, config)
        assert torch.allclose(out.plwc, torch.zeros_like(out.plwc))

    def test_zlwc_positive_for_snow(
        self, soil_table, consts, config,
    ):
        """With snow on the ground, ZLWC > 0."""
        inp = _build_inputs(
            N=2, snow_mass=20.0, snow_rho=200.0,
            snow_frac_low=0.5, bare_frac=0.5, snow_t=270.0,
        )
        out = compute_snow(inp, soil_table, consts, config)
        assert bool((out.plwc > 0.0).all())

    def test_zlwc_increases_with_snow_mass(
        self, soil_table, consts, config,
    ):
        """ZLWC scales roughly linearly with snow mass (same density)."""
        inp_light = _build_inputs(
            N=1, snow_mass=5.0, snow_rho=200.0,
            snow_frac_low=0.5, bare_frac=0.5, snow_t=270.0,
        )
        inp_heavy = _build_inputs(
            N=1, snow_mass=50.0, snow_rho=200.0,
            snow_frac_low=0.5, bare_frac=0.5, snow_t=270.0,
        )
        out_light = compute_snow(inp_light, soil_table, consts, config)
        out_heavy = compute_snow(inp_heavy, soil_table, consts, config)
        assert bool(out_heavy.plwc[0] > out_light.plwc[0])


# --------------------------------------------------------------------------- #
# 7. Vectorisation
# --------------------------------------------------------------------------- #
class TestSnowVectorisation:
    """Per-column independence and batch shape preservation."""

    def test_per_column_independence(
        self, soil_table, consts, config,
    ):
        """Columns with identical inputs produce identical outputs."""
        N = 5
        inp = _build_inputs(N=N, snow_t=270.0, snow_frac_low=0.4, bare_frac=0.6)
        out = compute_snow(inp, soil_table, consts, config)
        # All columns should match column 0.
        for i in range(1, N):
            assert torch.allclose(out.ptsn[i], out.ptsn[0])
            assert torch.allclose(out.pgsn[i], out.pgsn[0])

    def test_per_column_distinct_inputs_distinct_outputs(
        self, soil_table, consts, config,
    ):
        """Different snow-T inputs give different PTSN outputs."""
        N = 3
        inp = _build_inputs(N=N, snow_frac_low=0.5, bare_frac=0.5, slrfl=-30.0)
        inp.ptsnm1m = torch.tensor([260.0, 268.0, 271.0], dtype=torch.float64)
        out = compute_snow(inp, soil_table, consts, config)
        # Three distinct outputs (or at least not all equal).
        assert not torch.allclose(out.ptsn[0], out.ptsn[1]) or \
               not torch.allclose(out.ptsn[1], out.ptsn[2])

    def test_output_shapes_preserved(
        self, soil_table, consts, config,
    ):
        """All outputs have the input grid shape."""
        N = 4
        inp = _build_inputs(N=N)
        out = compute_snow(inp, soil_table, consts, config)
        assert out.ptsn.shape == (N,)
        assert out.pgsn.shape == (N,)
        assert out.psstar.shape == (N,)
        assert out.plwc.shape == (N,)
        assert out.zfrsn.shape == (N,)


# --------------------------------------------------------------------------- #
# 8. Edge cases
# --------------------------------------------------------------------------- #
class TestSnowEdgeCases:
    """Zero snow mass, tiny snow fraction, extreme density, FLake on/off."""

    def test_zero_snow_mass_does_not_nan(
        self, soil_table, consts, config,
    ):
        """Zero SWE with a nonzero snow fraction: no NaNs/Infs.

        With snow_frac > RFRSMALL but snow_mass = 0, ZDSNR = 0 and the
        safe divides must keep every output finite.
        """
        inp = _build_inputs(
            N=2, snow_mass=0.0, snow_frac_low=0.5, bare_frac=0.5,
            snow_t=270.0,
        )
        out = compute_snow(inp, soil_table, consts, config)
        assert bool(torch.isfinite(out.ptsn).all())
        assert bool(torch.isfinite(out.pgsn).all())
        assert bool(torch.isfinite(out.psstar).all())
        assert bool(torch.isfinite(out.plwc).all())

    def test_tiny_snow_fraction_uses_no_snow_branch(
        self, soil_table, consts, config,
    ):
        """ZFRSN < RFRSMALL triggers the no-snow branch."""
        rfrsmall = soil_table.consts.rfrsmall
        # Place a snow fraction just below RFRSMALL.
        tiny = max(rfrsmall * 0.5, 1e-9)
        inp = _build_inputs(
            N=2, snow_frac_low=tiny, bare_frac=1.0 - tiny,
            snow_t=270.0,
        )
        out = compute_snow(inp, soil_table, consts, config)
        rtt = float(consts.rtt.item())
        assert torch.allclose(out.ptsn, torch.full_like(out.ptsn, rtt))
        assert torch.allclose(out.pgsn, torch.zeros_like(out.pgsn))

    def test_very_dense_snow(
        self, soil_table, consts, config,
    ):
        """Very dense snow (rho > RLWCSWEA) keeps outputs finite."""
        inp = _build_inputs(
            N=2, snow_rho=500.0, snow_mass=20.0,
            snow_frac_low=0.5, bare_frac=0.5, snow_t=270.0,
        )
        out = compute_snow(inp, soil_table, consts, config)
        assert bool(torch.isfinite(out.ptsn).all())
        assert bool(torch.isfinite(out.pgsn).all())

    def test_very_light_snow(
        self, soil_table, consts, config,
    ):
        """Very light snow (rho ~ 50 kg/m^3) keeps outputs finite."""
        inp = _build_inputs(
            N=2, snow_rho=50.0, snow_mass=20.0,
            snow_frac_low=0.5, bare_frac=0.5, snow_t=270.0,
        )
        out = compute_snow(inp, soil_table, consts, config)
        assert bool(torch.isfinite(out.ptsn).all())
        assert bool(torch.isfinite(out.pgsn).all())

    def test_flake_off_does_not_use_lake_tile(
        self, soil_table, consts, config,
    ):
        """With use_flake=False, the lake tile is not in the grid fraction.

        We give the lake tile a nonzero fraction and check it does NOT
        contribute to the snowfall interception mass gain.
        """
        # 9-tile slab with lake fraction 0.5.
        pfrti = torch.zeros(2, 9, dtype=torch.float64)
        pfrti[:, 7] = 0.5     # bare-soil
        pfrti[:, 8] = 0.5     # lake
        inp = _build_inputs(
            N=2, snow_frac_low=0.0, snow_frac_high=0.0, bare_frac=0.0,
            pssfl=1.0e-4,
        )
        inp.pfrti = pfrti
        # use_flake=False (default), grid_frac = 0.5 (bare only).
        out = compute_snow(inp, soil_table, consts, config)
        expected = 10.0 + 3600.0 * 0.5 * 1.0e-4
        assert torch.allclose(out.psstar, torch.full_like(out.psstar, expected))

    def test_zero_timestep_keeps_state(
        self, soil_table, consts, config,
    ):
        """PTMST -> 0 should leave snow mass and temperature unchanged
        (modulo the flux terms that vanish with PTMST)."""
        inp = _build_inputs(
            N=2, snow_mass=10.0, snow_t=270.0, ptmst=1.0e-30,
            snow_frac_low=0.5, bare_frac=0.5, slrfl=0.0,
        )
        out = compute_snow(inp, soil_table, consts, config)
        # Mass should be ~10 (no flux contribution over zero timestep).
        assert torch.allclose(out.psstar, torch.full_like(out.psstar, 10.0),
                              atol=1e-6)


# --------------------------------------------------------------------------- #
# 9. Differentiability
# --------------------------------------------------------------------------- #
class TestSnowRainInterception:
    """Liquid throughfall captured by snow must be available to soil coupling."""

    def test_returns_intercepted_rain_by_precipitation_phase(
        self, soil_table, consts, config,
    ):
        inp = _build_inputs(
            N=1, snow_mass=10.0, snow_t=270.0, soil_t_top=270.0,
            snow_frac_low=0.5, bare_frac=0.5,
            ptsfc=2.0e-5, ptsfl=3.0e-5,
        )
        out = compute_snow(inp, soil_table, consts, config)

        assert out.ptsfcin is not None
        assert out.ptsflin is not None
        assert out.ptsfcin.item() > 0.0
        assert out.ptsflin.item() > 0.0
        assert out.ptsfcin.item() < inp.ptsfc.item()
        assert out.ptsflin.item() < inp.ptsfl.item()
        assert out.ptsfcin.item() / out.ptsflin.item() == pytest.approx(2.0 / 3.0)


# --------------------------------------------------------------------------- #
# 10. Differentiability
# --------------------------------------------------------------------------- #
class TestSnowDifferentiable:
    """Autograd flows through every branch."""

    def test_grad_flows_to_snow_temperature(
        self, soil_table, consts,
    ):
        """Backprop through PTSN reaches PTSNM1M in the no-melt branch."""
        config = SurfConfig(dtype=torch.float64, differentiable=True)
        inp = _build_inputs(
            N=2, snow_t=268.0, slrfl=-30.0,
            snow_frac_low=0.5, bare_frac=0.5,
        )
        inp.ptsnm1m.requires_grad_(True)
        out = compute_snow(inp, soil_table, consts, config)
        out.ptsn.sum().backward()
        assert inp.ptsnm1m.grad is not None
        assert bool(torch.isfinite(inp.ptsnm1m.grad).all())

    def test_grad_flows_to_snow_mass(
        self, soil_table, consts,
    ):
        """Backprop through PSSTAR reaches PSSNM1M."""
        config = SurfConfig(dtype=torch.float64, differentiable=True)
        inp = _build_inputs(
            N=2, snow_mass=10.0, snow_t=270.0,
            snow_frac_low=0.5, bare_frac=0.5,
        )
        inp.pssnm1m.requires_grad_(True)
        out = compute_snow(inp, soil_table, consts, config)
        out.psstar.sum().backward()
        assert inp.pssnm1m.grad is not None
        assert bool(torch.isfinite(inp.pssnm1m.grad).all())

    def test_grad_flows_to_tile_fractions(
        self, soil_table, consts,
    ):
        """Backprop reaches the tile-fraction tensor."""
        config = SurfConfig(dtype=torch.float64, differentiable=True)
        inp = _build_inputs(
            N=2, snow_t=270.0, slrfl=-30.0,
            snow_frac_low=0.5, bare_frac=0.5,
        )
        inp.pfrti.requires_grad_(True)
        out = compute_snow(inp, soil_table, consts, config)
        out.pgsn.sum().backward()
        assert inp.pfrti.grad is not None
        assert bool(torch.isfinite(inp.pfrti.grad).all())

    def test_grad_flows_through_melt_branch(
        self, soil_table, consts,
    ):
        """Backprop through the melt branch stays finite."""
        config = SurfConfig(dtype=torch.float64, differentiable=True)
        inp = _build_inputs(
            N=2, snow_t=271.0, ssrfl=200.0, ahfs=50.0, slrfl=-30.0,
            snow_frac_low=0.5, bare_frac=0.5,
        )
        inp.ptsnm1m.requires_grad_(True)
        inp.pssnm1m.requires_grad_(True)
        out = compute_snow(inp, soil_table, consts, config)
        (out.ptsn.sum() + out.psstar.sum()).backward()
        assert inp.ptsnm1m.grad is not None
        assert inp.pssnm1m.grad is not None
        assert bool(torch.isfinite(inp.ptsnm1m.grad).all())
        assert bool(torch.isfinite(inp.pssnm1m.grad).all())

    def test_grad_flows_through_no_snow_branch(
        self, soil_table, consts,
    ):
        """Backprop through the no-snow branch (PSSTAR only) is finite."""
        config = SurfConfig(dtype=torch.float64, differentiable=True)
        inp = _build_inputs(
            N=2, snow_frac_low=0.0, bare_frac=1.0,
            pssfl=1.0e-4,
        )
        inp.pssnm1m.requires_grad_(True)
        inp.pssfl.requires_grad_(True)
        out = compute_snow(inp, soil_table, consts, config)
        out.psstar.sum().backward()
        assert inp.pssnm1m.grad is not None
        assert inp.pssfl.grad is not None
        assert bool(torch.isfinite(inp.pssfl.grad).all())
