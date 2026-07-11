"""HTESSEL soil temperature update (apparent heat capacity + implicit Thomas).

This module is the Python port of the soil-temperature branch of
``SURFTSTP_CTL`` (``surf/module/surftstp_ctl_mod.F90`` section 3),
comprising three upstream routines:

* ``SRFRCGS`` (``surf/module/srfrcgs_mod.F90``) -- apparent volumetric
  heat capacity.  The "apparent" qualifier is the freeze/melt term: when
  soil water is in the freeze window ``[RTF2, RTF1]`` the heat capacity
  absorbs the latent-heat release/uptake via
  ``PCTSA = ZRCSOIL - RLMLT*RHOH2O*ZWA*ZDFDT`` with
  ``ZDFDT = -0.5*RTF4*COS(RTF4*(T-RTF3))`` (the derivative of the liquid
  fraction ``F(T) = 0.5*(1 - SIN(RTF4*(T-RTF3)))`` from SRFENE).

* ``SRFTS``  (``surf/module/srfts_mod.F90``)     -- assembles the
  tridiagonal coefficients for the soil-temperature solve: net surface
  heat flux ``ZSURFL``, Peters-Liddard / Kersten thermal conductivity
  ``ZDIF``, layer heat capacity ``ZCDZ = PCTSA*RDAT``, and the RHS.

* ``SRFWDIFS`` (``surf/module/srfwdifs_mod.F90``) -- the Thomas
  tridiagonal solver.  Despite the ``W`` (water) prefix, this is a
  generic solver used for both soil temperature (from ``SRFTS``) and
  soil moisture (from the M5b path).  It carries the ``LDALLAYS`` flag
  that lets ``SRFTS`` opt out of solving below the top layer.

The new temperature is closed by
``PTSA = PTSAM1M*ZCONS2 + ZTSA`` with
``ZCONS2 = 1 - 1/RSIMP`` (over-implicit time weight, RSIMP=1 in the
offline driver so ZCONS2=0 and the new T is the Thomas solution
``ZTSA`` itself -- i.e. a fully implicit Euler step).

Steps reproduced (SRFTS section numbers)
----------------------------------------
1.  Computational constants
    ``ZCONS1 = PTMST*RSIMP``, ``ZCONS2 = 1 - 1/RSIMP``.

2.  Net surface heat flux ``ZSURFL`` (SRFTS lines 183-218):
    fractional average of net SW + net LW + (sensible + latent) over the
    four soil-viewing tiles (wet-skin 3, low-veg-dry 4, high-veg-dry 6,
    bare-soil 8) plus the snow-bottom heat flux ``PGSN``.

3.  Per-layer thermal conductivity ``ZDIF`` (SRFTS lines 223-247,
    Peters-Liddard / Kersten formula):
    ``ZLAMBDASAT = RLAMSAT1M * RLAMBDAICE**(WSAT-W) * RLAMBDAWAT**W``,
    ``ZKERSTEN   = RKERST2*LOG10(MAX(RKERST1, W/WSAT)) + RKERST3``,
    ``ZDIF       = RLAMBDADRYM + ZKERSTEN*(ZLAMBDASAT - RLAMBDADRYM)``.

4.  Thomas coefficient assembly (SRFTS lines 252-281):
    ``ZLST(JK) = ZCONS1*(ZDIF(JK)+ZDIF(JK+1))/(RDAT(JK)+RDAT(JK+1))``
    for the inter-layer thermal coupling (half-level JK+1/2), and
    ``ZCDZ(JK) = PCTSA(JK)*RDAT(JK)`` for the layer heat capacity.
    Surface RHS ``ZRHS(:,1) = PTMST*ZSURFL/ZCDZ(:,1)`` injects the net
    surface flux; deeper-layer RHS are zero (no internal sources).
    The bottom layer ``ZLST(:,KLEVS) = 0`` enforces zero-flux lower BC.

5.  Thomas solve (SRFWDIFS): see :func:`_solve_tridiag`.

6.  New temperature ``PTSA = PTSAM1M*ZCONS2 + ZTSA``.

Cold-start simplifications
--------------------------
With RSIMP=1 (offline-driver default) ZCONS2=0, so the solver is a
straightforward fully-implicit step and the per-tile ZSURFL collapses to
the four-tile fractional average.  Snow-bottom heat flux ``PGSN`` is
zero here (M6 snow not yet wired) and snow evaporation mismatch
``PEMSSN`` is zero, matching the cold-start assumption documented in the
M5 design notes.

Sign convention
---------------
``ZSURFL`` is positive *downward* (into the soil), consistent with the
SURFSEB ground-heat-flux convention documented in
:mod:`surf_pytorch.physics.seb`.  Sensible and latent heat fluxes from
``VDFDIFH1S`` follow the same convention; the latent contribution is
``RLVTT*PEVAP`` (evaporation is positive, removing energy from the soil
as latent heat of vapourisation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import (
    TILE_WET_SKIN,
    TILE_LOW_VEG_DRY,
    TILE_HIGH_VEG_DRY,
    TILE_BARE_SOIL,
    SurfConfig,
)
from ..constants import ConstantBundle
from ..tables.soil import SoilTable


# --------------------------------------------------------------------------- #
# Hard-wired constants (yoevdf / yos_soil)
# --------------------------------------------------------------------------- #
# RRCSICE -- volumetric heat capacity of sea ice (sussoil line 467).
# Surf points (LDLAND=.F., LDSICE=.F.) get PCTSA=0 so they never feed the
# temperature solve; we mirror that here by zeroing the whole profile.
_RRCSICE = 1.884e6

# RGH2O -- volumetric heat capacity of liquid water (sussoil line 135),
# used by SRFRCG in the sea-ice branch (ZGICE = 0.5*RGH2O).  Surf points
# are zeroed regardless, but the constant is kept here for parity.
_RGH2O = 4.18e6


# --------------------------------------------------------------------------- #
# Soil-temperature branch (SRFRCGS + SRFTS + SRFWDIFS)
# --------------------------------------------------------------------------- #
@dataclass
class SoilTempInputs:
    """All inputs to the HTESSEL soil-temperature solve.

    Each tensor is ``(Ngrid, Nsoil)`` for per-(grid,layer) fields, per-grid
    fields are ``(Ngrid,)`` and per-tile flux fields are ``(Ngrid, Ntile)``.
    Naming mirrors the Fortran dummy list of ``SRFTS`` so a side-by-side
    audit is easy.

    Cold-start note: ``pgsn`` (PGSN, ground heat flux from the snow deck
    to the soil) is zero until the M6 single-layer snow prognostic is
    wired; passing a non-zero value is harmless and forwards-compatible.
    """

    # ---- Soil state at t-1 ----
    ptsam1m: torch.Tensor            # PTSAM1M  soil temperature    [K]      (Ngrid, Nsoil)
    pwsam1m: torch.Tensor            # PWSAM1M  soil moisture       [m3/m3]  (Ngrid, Nsoil)

    # ---- Per-tile fluxes (from VDFDIFH1S / SURFSEB) ----
    pfrti: torch.Tensor              # PFRTI    tile fractions     [-]      (Ngrid, Ntile)
    pahfsti: torch.Tensor            # PAHFSTI  tile sensible heat  [W/m^2]  (Ngrid, Ntile)
    pevapti: torch.Tensor            # PEVAPTI  tile moisture flux  [kg/m^2/s] (Ngrid, Ntile)
    pssrflti: torch.Tensor           # PSSRFLTI tile net SW         [W/m^2]  (Ngrid, Ntile)

    # ---- Per-grid radiation ----
    pslrfl: torch.Tensor             # PSLRFL   net LW              [W/m^2]  (Ngrid,)

    # ---- Snow-bottom heat flux (M6 forward-compat; zero at cold start) ----
    pgsn: torch.Tensor               # PGSN     ground heat flux from snow  [W/m^2] (Ngrid,)

    # ---- Static descriptors ----
    ksoty: torch.Tensor              # KSOTY    soil texture type  [long]   (Ngrid,)  0..7
    pcvl: torch.Tensor               # PCVL     low-veg  cover     [-]      (Ngrid,)
    pcvh: torch.Tensor               # PCVH     high-veg cover     [-]      (Ngrid,)

    # ---- Masks ----
    ldland: torch.Tensor             # LDLAND   land mask (bool)            (Ngrid,)

    # ---- Timestep ----
    ptmst: torch.Tensor              # PTMST    timestep            [s]     scalar


@dataclass
class SoilTempOutputs:
    """All outputs of the HTESSEL soil-temperature solve."""

    ptsa: torch.Tensor               # PTSA  new soil temperature  [K]     (Ngrid, Nsoil)
    pctsa: torch.Tensor              # PCTSA apparent heat capacity [J/K/m^3] (Ngrid, Nsoil)

    # Surface net heat flux into the top soil layer (positive downward).
    # Surfaced as a diagnostic so the energy-balance closure check can be
    # run from the model.forward() without re-deriving it.
    zsurfl: torch.Tensor             # ZSURFL surface net heat flux [W/m^2] (Ngrid,)


def compute_soil_temperature(
    inp: SoilTempInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    differentiable: Optional[bool] = None,
) -> SoilTempOutputs:
    """Run the HTESSEL soil-temperature solve (SRFRCGS + SRFTS + SRFWDIFS).

    Args
    ----
    inp:
        :class:`SoilTempInputs` -- packed soil state, fluxes and masks.
    soil_table:
        :class:`surf_pytorch.tables.soil.SoilTable` -- per-class LUT
        (wsat, lambdadry, lamsat1, rcsoil) and class-agnostic thermal
        scalars (rsimp, rlambdace, rlambdawat, rkerst1/2/3).
    consts:
        :class:`ConstantBundle` -- provides ``rtt``, ``rlmlt``, ``rlvtt``
        and the freeze-ramp endpoints ``rtf1..rtf4``.
    config:
        :class:`SurfConfig` -- ``n_soil`` selects the soil-layer count
        and ``soil_thickness`` provides ``RDAT``.
    differentiable:
        Override ``config.differentiable`` for this call.

    Returns
    -------
    :class:`SoilTempOutputs` -- new temperature, apparent heat capacity
    and the surface net heat flux diagnostic.
    """
    diff = config.differentiable if differentiable is None else differentiable

    def _core() -> SoilTempOutputs:
        return _soil_temperature_core(inp, soil_table, consts, config)

    if not diff:
        with torch.no_grad():
            return _core()
    return _core()


def _soil_temperature_core(
    inp: SoilTempInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
    config: SurfConfig,
) -> SoilTempOutputs:
    """Internal: assumes the no_grad / enable_grad decision is already made."""
    ngrid = inp.ptsam1m.shape[0]
    nlevs = config.n_soil
    dtype = inp.ptsam1m.dtype
    device = inp.ptsam1m.device

    # Scalar constants (evaluated in the same order as the Fortran so
    # compound expressions match to the last ULP).
    rlmlt = consts.rlmlt
    rlvtt = consts.rlvtt
    rtt = consts.rtt
    rtf1 = consts.rtf1
    rtf2 = consts.rtf2
    rtf3 = consts.rtf3
    rtf4 = consts.rtf4
    rho_h2o = float(soil_table.consts.rho_h2o)

    rsimp = soil_table.consts.rsimp
    ptmst = inp.ptmst

    # Per-class LUT columns indexed by soil type (0..7; index 0 is the
    # water sentinel with all zeros -- gather returns 0 there, which is
    # correct because surf points are masked out below anyway).
    ksoty = inp.ksoty.long().detach()
    wsat = soil_table.gather("wsat", ksoty)             # (Ngrid,)
    lambdadry = soil_table.gather("lambdadry", ksoty)   # (Ngrid,)
    lamsat1 = soil_table.gather("lamsat1", ksoty)       # (Ngrid,)
    rcsoil = soil_table.gather("rcsoil", ksoty)         # (Ngrid,)
    wcap = soil_table.gather("wcap", ksoty)             # (Ngrid,)

    # Soil-layer thicknesses RDAT (sussoil lines 178-181).  Broadcast to
    # (Ngrid, Nsoil) so the per-layer expressions vectorise; the
    # temperature solve uses these as plain per-layer scalars.
    rdat = _soil_thickness(config, dtype, device)       # (Nsoil,)
    rdat_b = rdat.expand(ngrid, nlevs)                  # (Ngrid, Nsoil)

    land = inp.ldland.to(dtype)                         # (Ngrid,)  1 / 0

    # ------------------------------------------------------------------ #
    # Step 1: SRFRCGS -- apparent volumetric heat capacity.
    # ------------------------------------------------------------------ #
    pctsa = _apparent_heat_capacity(
        inp.ptsam1m, wcap, rcsoil, inp.pcvl, inp.pcvh,
        land, rlmlt, rho_h2o, rtf1, rtf2, rtf3, rtf4,
        nlevs, dtype, device,
    )

    # ------------------------------------------------------------------ #
    # Step 2: SRFTS section 2 -- net surface heat flux ZSURFL.
    # ------------------------------------------------------------------ #
    zsurfl = _surface_net_heat_flux(
        inp.pfrti, inp.pssrflti, inp.pslrfl,
        inp.pahfsti, inp.pevapti, inp.pgsn,
        land, rlvtt, dtype, device,
    )

    # ------------------------------------------------------------------ #
    # Step 3: SRFTS section 3 -- per-layer thermal conductivity ZDIF.
    # Peters-Liddard / Kersten formula (vectorised over Nsoil).
    # ------------------------------------------------------------------ #
    zdif = _thermal_conductivity(
        inp.pwsam1m, wsat, lambdadry, lamsat1,
        land, soil_table.consts,
        nlevs, dtype, device,
    )

    # ------------------------------------------------------------------ #
    # Step 4: SRFTS section 4 -- Thomas coefficient assembly.
    # ------------------------------------------------------------------ #
    zcons1 = ptmst * rsimp
    # ZLST is the half-level thermal coupling; Fortran indexes it 1..KLEVS
    # but only uses 1..KLEVS-1 (ZLST(:,KLEVS)=0 enforces zero lower BC).
    zlst = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)
    zcdz = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)
    zrhs = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)

    # Layers 1..KLEVS-1 use the inter-layer coupling
    #   ZLST(JK) = ZCONS1*(ZDIF(JK)+ZDIF(JK+1))/(RDAT(JK)+RDAT(JK+1)).
    # The bottom layer ZLST(:,KLEVS) stays zero (zero-flux lower BC).
    if nlevs > 1:
        zdif_lo = zdif[:, :-1]                          # ZDIF(:,1..KLEVS-1)
        zdif_hi = zdif[:, 1:]                           # ZDIF(:,2..KLEVS)
        rdat_lo = rdat_b[:, :-1]
        rdat_hi = rdat_b[:, 1:]
        zlst[:, :-1] = zcons1 * (zdif_lo + zdif_hi) / (rdat_lo + rdat_hi)

    zcdz = pctsa * rdat_b                              # ZCDZ = PCTSA*RDAT

    # Surface RHS injects the net surface flux into the top layer.
    # ZRHS(:,1) = PTMST*ZSURFL/ZCDZ(:,1).  Guard ZCDZ(:,1) > 0; surf
    # points have ZCDZ=0 and are masked out, but a safe divide keeps the
    # autograd graph finite everywhere.
    zcdz1_safe = torch.where(
        zcdz[:, 0] != 0.0, zcdz[:, 0], torch.ones_like(zcdz[:, 0])
    )
    zrhs[:, 0] = ptmst * zsurfl / zcdz1_safe
    zrhs[:, 0] = torch.where(zcdz[:, 0] != 0.0, zrhs[:, 0], torch.zeros_like(zrhs[:, 0]))

    # ------------------------------------------------------------------ #
    # Step 5: SRFWDIFS -- Thomas tridiagonal solve.
    # ------------------------------------------------------------------ #
    ztsa = _solve_tridiag(inp.ptsam1m, zlst, zrhs, zcdz, land, rsimp)

    # ------------------------------------------------------------------ #
    # Step 6: SRFTS section 7 -- new temperatures.
    #   PTSA = PTSAM1M*ZCONS2 + ZTSA,  ZCONS2 = 1 - 1/RSIMP.
    # Surf points keep their t-1 temperature (LDLAND=.F. branch).
    # ------------------------------------------------------------------ #
    zcons2 = 1.0 - 1.0 / rsimp
    ptsa_land = inp.ptsam1m * zcons2 + ztsa
    ptsa = torch.where(
        land[:, None].to(dtype) != 0.0, ptsa_land, inp.ptsam1m
    )

    return SoilTempOutputs(ptsa=ptsa, pctsa=pctsa, zsurfl=zsurfl)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _soil_thickness(
    config: SurfConfig, dtype: torch.dtype, device,
) -> torch.Tensor:
    """Return RDAT (soil layer thicknesses, m) as a 1-D tensor.

    Mirrors sussoil lines 178-181: (0.07, 0.21, 0.72, 1.89) for the
    4-layer default.  SurfConfig.soil_thickness is the source of truth so
    configurations with non-default layer counts keep working.
    """
    return torch.as_tensor(
        config.soil_thickness, dtype=dtype, device=device
    )[: config.n_soil]


def _apparent_heat_capacity(
    ptsam1m: torch.Tensor,
    wcap: torch.Tensor,        # (Ngrid,) per-class field capacity
    rcsoil: torch.Tensor,      # (Ngrid,) per-class dry-soil heat capacity
    pcvl: torch.Tensor,        # (Ngrid,)
    pcvh: torch.Tensor,        # (Ngrid,)
    land: torch.Tensor,        # (Ngrid,) 1/0
    rlmlt: float,
    rho_h2o: float,
    rtf1: float, rtf2: float, rtf3: float, rtf4: float,
    nlevs: int,
    dtype: torch.dtype, device,
) -> torch.Tensor:
    """SRFRCGS -- apparent volumetric heat capacity (J/K/m^3).

    Reproduces srfrcgs_mod.F90 lines 132-188.  For land points::

        ZDFDT = -0.5*RTF4*COS(RTF4*(T-RTF3))   if RTF2 < T < RTF1 else 0
        ZWA   = (PCVL + PCVH) * RWCAPM(JS)
        PCTSA = RRCSOILM(JS) - RLMLT*RHOH2O*ZWA*ZDFDT

    Surf points (LDLAND=.F.) are zeroed; sea-ice points use RRCSICE on
    the top layer and zero below (not exercised here -- the offline
    driver runs the sea-ice branch through SRFI, and the port routes sea
    ice via a separate path).
    """
    t = ptsam1m
    in_freeze = (t > rtf2) & (t < rtf1)
    zdfdt = -0.5 * rtf4 * torch.cos(rtf4 * (t - rtf3))
    zdfdt = torch.where(in_freeze, zdfdt, torch.zeros_like(zdfdt))

    # ZWA = (PCVL + PCVH) * RWCAPM(JS) -- per-class field capacity scaled
    # by the vegetation cover.  Broadcast (Ngrid,) -> (Ngrid, Nsoil).
    zwa = (pcvl + pcvh) * wcap                       # (Ngrid,)
    zwa_b = zwa[:, None].expand(-1, nlevs)

    pctsa = rcsoil[:, None] - rlmlt * rho_h2o * zwa_b * zdfdt   # (Ngrid, Nsoil)

    # Surf points: zero whole profile (LDLAND=.F., LDSICE=.F. branch).
    land_b = land[:, None].to(dtype)
    pctsa = torch.where(land_b != 0.0, pctsa, torch.zeros_like(pctsa))
    return pctsa


def _surface_net_heat_flux(
    pfrti: torch.Tensor,
    pssrflti: torch.Tensor,
    pslrfl: torch.Tensor,
    pahfsti: torch.Tensor,
    pevapti: torch.Tensor,
    pgsn: torch.Tensor,
    land: torch.Tensor,
    rlvtt: float,
    dtype: torch.dtype, device,
) -> torch.Tensor:
    """SRFTS section 2 -- net surface heat flux ZSURFL (positive downward).

    Reproduces srfts_mod.F90 lines 183-218.  The four soil-viewing tiles
    are wet-skin (3), low-veg-dry (4), high-veg-dry (6), bare-soil (8).
    Their tile fractions sum to ~1 over land (the rest are water / ice /
    snow tiles) so the fractional average is just the tile-weighted sum;
    the snow-bottom heat flux ``PGSN`` is added as a grid-mean term.

    The lake special-case (SRFTS lines 205-213, LEFLAKE branch) is not
    transcribed here: the offline driver has LEFLAKE=.FALSE. and the M5
    port leaves lake tiles routed through a separate path.  Passing a
    non-zero PGSN for snow is M6-forward-compatible.
    """
    f3 = pfrti[:, TILE_WET_SKIN]
    f4 = pfrti[:, TILE_LOW_VEG_DRY]
    f6 = pfrti[:, TILE_HIGH_VEG_DRY]
    f8 = pfrti[:, TILE_BARE_SOIL]

    zssrfl = (
        f3 * pssrflti[:, TILE_WET_SKIN]
        + f4 * pssrflti[:, TILE_LOW_VEG_DRY]
        + f6 * pssrflti[:, TILE_HIGH_VEG_DRY]
        + f8 * pssrflti[:, TILE_BARE_SOIL]
    )
    zslrfl = (f3 + f4 + f6 + f8) * pslrfl
    zthfl = (
        f3 * (pahfsti[:, TILE_WET_SKIN] + rlvtt * pevapti[:, TILE_WET_SKIN])
        + f4 * (pahfsti[:, TILE_LOW_VEG_DRY] + rlvtt * pevapti[:, TILE_LOW_VEG_DRY])
        + f6 * (pahfsti[:, TILE_HIGH_VEG_DRY] + rlvtt * pevapti[:, TILE_HIGH_VEG_DRY])
        + f8 * (pahfsti[:, TILE_BARE_SOIL] + rlvtt * pevapti[:, TILE_BARE_SOIL])
    )

    zsurfl = zssrfl + zslrfl + zthfl + pgsn

    # Surf points: ZSURFL=0 (SRFTS line 216 ELSE branch).  Masked here so
    # the diagnostic carries the land-only value; the tridiag RHS uses
    # this masked field.
    land_mask = land.to(dtype)
    zsurfl = torch.where(land_mask != 0.0, zsurfl, torch.zeros_like(zsurfl))
    return zsurfl


def _thermal_conductivity(
    pwsam1m: torch.Tensor,
    wsat: torch.Tensor,        # (Ngrid,)
    lambdadry: torch.Tensor,   # (Ngrid,)
    lamsat1: torch.Tensor,     # (Ngrid,)
    land: torch.Tensor,
    soil_consts,               # SoilConstants dataclass
    nlevs: int,
    dtype: torch.dtype, device,
) -> torch.Tensor:
    """SRFTS section 3 -- per-layer soil thermal conductivity ZDIF.

    Peters-Liddard / Kersten formula (srfts_mod.F90 lines 223-247)::

        ZWU        = W * (1 - ZFF),    ZFF = 0
        ZLWT       = RLAMBDAWAT ** ZWU
        ZLIC       = RLAMBDAICE ** (WSAT - ZWU)
        ZLAMBDASAT = RLAMSAT1M * ZLIC * ZLWT
        ZCOND      = MAX(RKERST1, W / WSAT)
        ZKERSTEN   = RKERST2 * LOG10(ZCOND) + RKERST3
        ZDIF       = RLAMBDADRYM + ZKERSTEN * (ZLAMBDASAT - RLAMBDADRYM)

    Vectorised over (Ngrid, Nsoil); per-class columns broadcast across
    the soil-axis.  Surf points stay zero (LDLAND=.F. branch).
    """
    w = pwsam1m
    wsat_b = wsat[:, None].expand(-1, nlevs)
    lamsat1_b = lamsat1[:, None].expand(-1, nlevs)
    lambdadry_b = lambdadry[:, None].expand(-1, nlevs)

    rlambdace = float(soil_consts.rlambdace)      # RLAMBDAICE
    rlambdawat = float(soil_consts.rlambdawat)
    rkerst1 = float(soil_consts.rkerst1)
    rkerst2 = float(soil_consts.rkerst2)
    rkerst3 = float(soil_consts.rkerst3)

    zwu = w * (1.0 - 0.0)                          # ZFF = 0
    zlwt = rlambdawat ** zwu
    zlic = rlambdace ** (wsat_b - zwu)
    zlambdasat = lamsat1_b * zlic * zlwt

    zcond = torch.clamp(w / wsat_b, min=rkerst1)   # MAX(RKERST1, W/WSAT)
    zkersten = rkerst2 * torch.log10(zcond) + rkerst3

    zdif = lambdadry_b + zkersten * (zlambdasat - lambdadry_b)

    land_b = land[:, None].to(dtype)
    zdif = torch.where(land_b != 0.0, zdif, torch.zeros_like(zdif))
    return zdif


def _solve_tridiag(
    pwsam1m: torch.Tensor,    # actually PTSAM1M here (generic solver)
    pcfw: torch.Tensor,       # ZLST (lambda-star, half-level coupling)
    prhsw: torch.Tensor,      # ZRHS (right-hand side)
    pcdz: torch.Tensor,       # ZCDZ = PCTSA*RDAT
    land: torch.Tensor,
    rsimp: float,
) -> torch.Tensor:
    """SRFWDIFS -- Thomas tridiagonal solver, batched over Ngrid.

    Despite the ``W`` prefix this is a generic solver used for both soil
    temperature (called from SRFTS) and soil moisture (called from the
    M5b path).  The semantics are identical; the variable names mirror
    the Fortran dummy list so the side-by-side audit is clean.

    Algorithm (srfwdifs_mod.F90 lines 117-176, LDALLAYS=.TRUE.)::

        ZTPFAC2       = 1/RSIMP
        PWSADIF(:,JK) = ZTPFAC2*PWSAM1M(:,JK) + PRHSW(:,JK)

        # Top layer elimination (JK=1)
        ZTCOE       = PCFW(:,1)
        ZQDP        = 1/PCDZ(:,1)
        ZDISC       = 1/(1 + PCFW(:,1)*ZQDP)
        ZEBSW(:,1)  = ZDISC*(PCFW(:,1)*ZQDP)
        PWSADIF(:,1)= ZDISC*PWSADIF(:,1)

        # Middle layers JK=2..KLEVS-1
        ZFAC        = ZTCOE*ZQDP(JK)
        ZTCOE       = PCFW(:,JK)
        ZDISC       = 1/(1 + ZFAC*(1-ZEBSW(:,JK-1)) + PCFW(:,JK)*ZQDP(JK))
        ZEBSW(:,JK) = ZDISC*(PCFW(:,JK)*ZQDP(JK))
        PWSADIF(:,JK)= ZDISC*(PWSADIF(:,JK) + ZFAC*PWSADIF(:,JK-1))

        # Bottom layer JK=KLEVS
        ZFAC        = ZTCOE*ZQDP(KLEVS)
        ZDISC       = 1/(1 + ZFAC*(1-ZEBSW(:,KLEVS-1)) + PCFW(:,KLEVS)*ZQDP(KLEVS))
        PWSADIF(:,KLEVS) = ZDISC*(PWSADIF(:,KLEVS) + ZFAC*PWSADIF(:,KLEVS-1))

        # Back-substitution JK=KLEVS-1..1
        PWSADIF(:,JK) += ZEBSW(:,JK)*PWSADIF(:,JK+1)

    The middle/bottom sweep carries a *column recurrence* (ZTCOE feeds
    the next layer's ZFAC), so it cannot be fully vectorised across
    layers; we unroll it as a Python loop over JK.  With nlevs=4 this is
    three iterations -- trivially cheap and exactly mirrors the Fortran
    control flow.

    Implementation note: the sweep is written *functionally* (per-layer
    solution tensors accumulated in a Python list, joined at the end via
    :func:`torch.stack`) rather than with in-place indexed assignment.
    The in-place form mirrors the Fortran more literally but breaks
    autograd (in-place writes to a leaf of the graph invalidate the
    version counter).  The functional form is bit-identical in the
    forward pass and stays differentiable for ML training.

    Land mask: surf points are never updated (the Fortran guards every
    assignment with ``IF (LDLAND(JL))``).  We carry the land mask
    through the sweep as a multiplicative gate so non-land columns stay
    at their initialised value (zero), matching the Fortran OUT array.
    """
    ngrid, nlevs = pwsam1m.shape
    dtype = pwsam1m.dtype
    device = pwsam1m.device

    ztpfac2 = 1.0 / rsimp

    # 1.1 RHS set-up.
    pwsadif = ztpfac2 * pwsam1m + prhsw

    # Gate non-land columns to zero so they never contribute to the
    # recurrence (the Fortran skips them entirely; we zero them and let
    # the masked update at the end restore the original).
    land_b = land[:, None].to(dtype)
    pwsadif = pwsadif * land_b

    # Per-layer elimination coefficients and the running solution.  Both
    # are accumulated functionally (no in-place writes) so autograd can
    # backprop through the sweep.
    ebsw: list[torch.Tensor] = [torch.empty(0)] * nlevs
    sol: list[torch.Tensor] = [torch.empty(0)] * nlevs

    # 1.2 Top layer elimination (JK=1).
    ztcoe = pcfw[:, 0]                              # PCFW(:,1); a view, never mutated
    zqdp = 1.0 / pcdz[:, 0]
    zdisc = 1.0 / (1.0 + pcfw[:, 0] * zqdp)
    ebsw[0] = zdisc * (pcfw[:, 0] * zqdp)
    sol[0] = zdisc * pwsadif[:, 0]

    # 1.3 Middle layers JK=2..KLEVS-1 (Python loop, exact Fortran order).
    for jk in range(1, nlevs - 1):
        zqdp = 1.0 / pcdz[:, jk]
        zfac = ztcoe * zqdp                         # ZTCOE = PCFW(:,JK-1)
        ztcoe = pcfw[:, jk]                         # advance for next iter
        zdisc = 1.0 / (
            1.0 + zfac * (1.0 - ebsw[jk - 1]) + pcfw[:, jk] * zqdp
        )
        ebsw[jk] = zdisc * (pcfw[:, jk] * zqdp)
        sol[jk] = zdisc * (pwsadif[:, jk] + zfac * sol[jk - 1])

    # 1.4 Bottom layer elimination (JK=KLEVS).  Guard nlevs==1 (single-
    # layer configuration skips the middle loop and reuses JK=1 ZTCOE).
    if nlevs >= 2:
        jk = nlevs - 1
        zqdp = 1.0 / pcdz[:, jk]
        zfac = ztcoe * zqdp
        zdisc = 1.0 / (
            1.0 + zfac * (1.0 - ebsw[jk - 1]) + pcfw[:, jk] * zqdp
        )
        sol[jk] = zdisc * (pwsadif[:, jk] + zfac * sol[jk - 1])

    # 1.5 Back-substitution JK=KLEVS-1..1.  Build a fresh tensor per
    # layer so the graph stays clean (sol[jk+1] is consumed read-only).
    for jk in range(nlevs - 2, -1, -1):
        sol[jk] = sol[jk] + ebsw[jk] * sol[jk + 1]

    return torch.stack(sol, dim=1)


# =========================================================================== #
# Soil-moisture branch (SRFWEXC_VG + SRFWDIF + SRFWINC + SRFWNG)
# =========================================================================== #
"""HTESSEL soil-moisture update with Van Genuchten hydraulics.

This is the Python port of the soil-moisture branch of ``SURFTSTP_CTL``
(``surf/module/surftstp_ctl_mod.F90`` section 4), four upstream routines:

* ``SRFWEXC_VG`` (``surf/module/srfwexc_vg_mod.F90``) -- assembles the
  tridiagonal coefficients for the moisture solve: Van Genuchten
  hydraulic conductivity K(W) and diffusivity D(W), VIC saturation-
  excess runoff *or* infiltration-limit runoff (gated by LESSRO), root
  extraction from the four soil-viewing tiles, and the surface forcing
  (throughfall + snowmelt - runoff + evap).

* ``SRFWDIF`` (``surf/module/srfwdif_mod.F90``) -- the same Thomas
  solver as the temperature path (the generic
  :func:`_solve_tridiag` above).  Despite the ``W`` prefix both paths
  share the algorithm.

* ``SRFWINC`` (``surf/module/srfwinc_mod.F90``) -- converts the Thomas
  solution ``PWSADIF`` to the unfiltered t+1 moisture
  ``PWSA = PWSADIF - ZTPFAC3*PWSAM1M`` and re-applies the gravity flux
  to the drainage diagnostic.

* ``SRFWNG`` (``surf/module/srfwng_mod.F90``) -- negative-moisture
  clipping with downward redistribution.  Excess water in layer JK is
  pushed into JK+1 (column recurrence); saturation excess becomes
  surface runoff (top layer) or sub-surface runoff (deeper layers).

Van Genuchten hydraulics
------------------------
With effective saturation ``Se = (W - Wres) / (Wsat - Wres)``::

    K(Se) = WCONS * Se**lam * (1 - (1 - Se**(1/m))**m)**2
    D(Se) = ((1-m)*WCONS / (alpha*m*WFAC)) * Se**(lam-1/m)
              * ( (1 - Se**(1/m))**m + (1 - Se**(1/m))**(-m) - 2 )

with ``m = 1 - 1/n``, ``WFAC = Wsat - Wres``.  At the air-dry floor
``W <= 1.001*Wres`` the functions clamp to ``K=0, D=Dmin``; near
saturation ``W >= 0.999*Wsat`` they clamp to ``K=WCONS, D=Dmax``.  The
freezing function ``ZF = 0.5*(1-SIN(RTF4*(T-RTF3)))`` further reduces
both (K is driven to zero, D is linearly interpolated toward Dmin) so
frozen soil neither drains nor diffuses appreciably.

Cold-start simplifications
--------------------------
* Snow inputs (PMSN snowmelt, PEMSSN evaporative mismatch, PEVAPSNW
  snow-under-forest evaporation) are all zero at cold start.
* The interception-layer evaporation mismatch PEINTTI is zero (no
  SRFWL call in the offline driver until M6 wires the skin reservoir).
* The lake/urban branches (KTILES > 9) are skipped -- the offline
  driver runs the 8-tile base configuration.

Runoff sign convention
-----------------------
``PROS`` (surface runoff) and ``PROD`` (sub-surface runoff) are
positive *out of* the soil column, in kg/m^2 (accumulated over the
timestep).  Drainage from the bottom layer is added to ``PROD`` by the
caller (SURFTSTP section 4.6).
"""


@dataclass
class SoilMoistInputs:
    """All inputs to the HTESSEL soil-moisture solve.

    Per-(grid,layer) fields are ``(Ngrid, Nsoil)``, per-grid fields are
    ``(Ngrid,)``, per-tile flux fields are ``(Ngrid, Ntile)``.  Naming
    mirrors the Fortran dummy list of ``SRFWEXC_VG``.

    Cold-start note: the snow/interception mismatch inputs (``pmsn``,
    ``pemssn``, ``pevapsnw``, ``peintti``) all default to zero and may
    be left at that until the M6 snow prognostic populates them.
    """

    # ---- Soil state at t-1 ----
    pwsam1m: torch.Tensor            # PWSAM1M  soil moisture     [m3/m3]  (Ngrid, Nsoil)
    ptsam1m: torch.Tensor            # PTSAM1M  soil temperature  [K]      (Ngrid, Nsoil)

    # ---- Per-tile fluxes (from VDFDIFH1S / SURFSEB) ----
    pfrti: torch.Tensor              # PFRTI    tile fractions    [-]      (Ngrid, Ntile)
    pevapti: torch.Tensor            # PEVAPTI  tile moisture flux [kg/m^2/s] (Ngrid, Ntile)

    # ---- Precipitation throughfall (per grid) ----
    ptsfc: torch.Tensor              # PTSFC  convective throughfall [kg/m^2/s]
    ptsfl: torch.Tensor              # PTSFL  large-scale throughfall [kg/m^2/s]

    # ---- Snow inputs (M6 forward-compat; zero at cold start) ----
    pmsn: torch.Tensor               # PMSN     snowmelt              [kg/m^2/s]
    pemssn: torch.Tensor             # PEMSSN   snow evap mismatch    [kg/m^2/s]
    pevapsnw: torch.Tensor           # PEVAPSNW snow-under-forest evap [kg/m^2/s]
    peintti: torch.Tensor            # PEINTTI  interception-layer evap mismatch (Ngrid, Ntile)

    # ---- Static descriptors ----
    ksoty: torch.Tensor              # KSOTY  soil texture type [long]  (Ngrid,)  0..7
    ktvl: torch.Tensor               # KTVL   low-veg type       [long]  (Ngrid,)  0..20
    ktvh: torch.Tensor               # KTVH   high-veg type      [long]  (Ngrid,)  0..20
    psdor: torch.Tensor              # PSDOR  orography param    [m]     (Ngrid,)

    # ---- Masks ----
    ldland: torch.Tensor             # LDLAND  land mask (bool)          (Ngrid,)

    # ---- Timestep ----
    ptmst: torch.Tensor              # PTMST  timestep            [s]     scalar


@dataclass
class SoilMoistOutputs:
    """All outputs of the HTESSEL soil-moisture solve."""

    pwsa: torch.Tensor               # PWSA  new soil moisture  [m3/m3]   (Ngrid, Nsoil)
    pros: torch.Tensor               # PROS  surface runoff     [kg/m^2]  (Ngrid,)
    prod: torch.Tensor               # PROD  sub-surface runoff  [kg/m^2]  (Ngrid,)
    pwfsd: torch.Tensor              # PWFSD flux layer1->layer2 [kg/m^2/s] (Ngrid,)
    psawgfl: torch.Tensor            # PSAWGFL gravity water flux [kg/m^2/s] (Ngrid, Nsoil)


def compute_soil_moisture(
    inp: SoilMoistInputs,
    soil_table: SoilTable,
    veg_table,                       # surf_pytorch.tables.veg.VegTable
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    differentiable: Optional[bool] = None,
) -> SoilMoistOutputs:
    """Run the HTESSEL soil-moisture solve (SRFWEXC_VG + SRFWDIF + SRFWINC + SRFWNG).

    Args
    ----
    inp:
        :class:`SoilMoistInputs` -- packed soil state, fluxes and masks.
    soil_table:
        :class:`surf_pytorch.tables.soil.SoilTable` -- per-class VG LUT
        (wsat, wrest, wcap, wpwp, wconsm, mvgalpha, mfac, lambda,
        dmax, dmin) and class-agnostic scalars (rsimp, rsrdep).
    veg_table:
        :class:`surf_pytorch.tables.veg.VegTable` -- provides the
        per-veg-type root profile ``rvrootsa``.
    consts:
        :class:`ConstantBundle` -- provides ``rtt`` and the freeze-ramp
        endpoints ``rtf1..rtf4``.
    config:
        :class:`SurfConfig` -- ``n_soil``, ``soil_thickness``, and the
        ``lessro`` flag (VIC vs infiltration-limit runoff).
    differentiable:
        Override ``config.differentiable`` for this call.

    Returns
    -------
    :class:`SoilMoistOutputs` -- new moisture, runoff split (surface +
    sub-surface), inter-layer flux and the gravity water-flux profile.
    """
    diff = config.differentiable if differentiable is None else differentiable

    def _core() -> SoilMoistOutputs:
        return _soil_moisture_core(inp, soil_table, veg_table, consts, config)

    if not diff:
        with torch.no_grad():
            return _core()
    return _core()


def _soil_moisture_core(
    inp: SoilMoistInputs,
    soil_table: SoilTable,
    veg_table,
    consts: ConstantBundle,
    config: SurfConfig,
) -> SoilMoistOutputs:
    """Internal: assumes the no_grad / enable_grad decision is already made."""
    ngrid = inp.pwsam1m.shape[0]
    nlevs = config.n_soil
    dtype = inp.pwsam1m.dtype
    device = inp.pwsam1m.device

    # Scalar constants.
    rtf1 = consts.rtf1
    rtf2 = consts.rtf2
    rtf3 = consts.rtf3
    rtf4 = consts.rtf4
    rho_h2o = float(soil_table.consts.rho_h2o)
    rsimp = soil_table.consts.rsimp
    rsrdep = soil_table.consts.rsrdep
    rsigormin = soil_table.consts.rsigormin
    rsigormax = soil_table.consts.rsigormax
    rpsfr = getattr(soil_table.consts, "rpsfr", 1.0)
    ptmst = inp.ptmst
    lessro = getattr(config, "lessro", True)

    # Per-class LUT columns indexed by soil type.
    ksoty = inp.ksoty.long().detach()
    wsat = soil_table.gather("wsat", ksoty)
    wrest = soil_table.gather("wrest", ksoty)
    wcap = soil_table.gather("wcap", ksoty)
    wpwp = soil_table.gather("wpwp", ksoty)
    wconsm = soil_table.gather("wconsm", ksoty)
    mvgalpha = soil_table.gather("mvgalpha", ksoty)
    mfac = soil_table.gather("mfac", ksoty)
    lam = soil_table.gather("lambda", ksoty)
    dmax = soil_table.gather("dmax", ksoty)
    dmin = soil_table.gather("dmin", ksoty)

    rdaw = _soil_thickness(config, dtype, device)         # (Nsoil,)
    rdaw_b = rdaw.expand(ngrid, nlevs)                    # (Ngrid, Nsoil)

    land = inp.ldland.to(dtype)                           # (Ngrid,)

    # ------------------------------------------------------------------ #
    # Step 2.1: liquid fraction ZF (T-dependent freeze function) and the
    # liquid-equivalent soil moisture ZLIQ used for root extraction.
    # ------------------------------------------------------------------ #
    zf = _liquid_fraction(inp.ptsam1m, rtf1, rtf2, rtf3, rtf4)    # (Ngrid, Nsoil)
    wpwp_b = wpwp[:, None].expand(-1, nlevs)
    wcap_b = wcap[:, None].expand(-1, nlevs)
    zliq = torch.clamp(inp.pwsam1m * (1.0 - zf), min=wpwp_b, max=wcap_b)

    # ------------------------------------------------------------------ #
    # Step 2.2: per-tile root extraction weights ZROOTW and normaliser ZEXT.
    # Only tiles 4 (low-veg-dry), 6 (high-veg-dry), 7 (high-veg-snow) have
    # root extraction; for positive evap (dew) all roots are in layer 1.
    # ------------------------------------------------------------------ #
    zrootw, zext = _root_extraction_weights(
        inp.pfrti, inp.pevapti, inp.pevapsnw, zliq, wpwp, zf,
        inp.ktvl, inp.ktvh, ksoty, veg_table, nlevs,
    )

    # ------------------------------------------------------------------ #
    # Step 2.3: top-layer coefficients (Van Genuchten K/D + runoff +
    # tile contributions + ZWSFL surface forcing).
    # ------------------------------------------------------------------ #
    pcfw = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)
    prhsw = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)
    psawgfl = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)
    pros = torch.zeros((ngrid,), dtype=dtype, device=device)
    # Top layer.
    pcfw, prhsw, psawgfl, pros, zsawext_l1, zconds = _top_layer_coeffs(
        inp, ksoty, wsat, wrest, wcap, wpwp, wconsm, mvgalpha, mfac, lam,
        dmax, dmin, zf, zrootw, zext,
        rdaw, land, ptmst, rsimp, rsrdep, rho_h2o, lessro,
        rsigormin, rsigormax, rpsfr,
        pcfw, prhsw, psawgfl, pros,
    )

    # ------------------------------------------------------------------ #
    # Step 2.4: deeper-layer coefficients (JK=2..KLEVS).
    # ------------------------------------------------------------------ #
    pcfw, prhsw, psawgfl = _deep_layer_coeffs(
        inp, ksoty, wsat, wrest, wcap, wpwp, wconsm, mvgalpha, mfac, lam,
        dmax, dmin, zf, zrootw, zext,
        rdaw, land, ptmst, rsimp, rho_h2o,
        pcfw, prhsw, psawgfl,
    )

    # ------------------------------------------------------------------ #
    # Step 4.3: SRFWDIF -- Thomas solve for PWSADIF.
    # ZCDZ here is RDAW (the layer thickness profile).
    # ------------------------------------------------------------------ #
    zcdz = rdaw_b.clone()
    pwsadif = _solve_tridiag(inp.pwsam1m, pcfw, prhsw, zcdz, inp.ldland, rsimp)

    # ------------------------------------------------------------------ #
    # Step 4.4: SRFWINC -- new unfiltered moisture + drainage diagnostic.
    # ------------------------------------------------------------------ #
    pwsa, psawgfl = _soil_moisture_update(
        inp.pwsam1m, pwsadif, pcfw, psawgfl, inp.ldland, rsimp, ptmst, rho_h2o,
    )
    pwfsd = psawgfl[:, 0].clone()

    # ------------------------------------------------------------------ #
    # Step 4.5: SRFWNG -- clip negative moisture, redistribute downward,
    # and split saturation excess into surface / sub-surface runoff.
    # ------------------------------------------------------------------ #
    pwsa, pros, prod, pwfsd = _clip_and_redistribute(
        pwsa, inp.pwsam1m, pros, pwfsd, wsat, wrest, rdaw,
        land, inp.ldland, ptmst, rho_h2o,
    )

    # ------------------------------------------------------------------ #
    # Step 4.6: SURFTSTP_CTL section 4.6 -- bottom-layer gravity drainage
    # is added to the sub-surface runoff diagnostic (so PROD carries the
    # full water *out* of the column).  Without this the water budget is
    # not closed.
    # ------------------------------------------------------------------ #
    prod = prod + psawgfl[:, -1] * ptmst * land

    return SoilMoistOutputs(
        pwsa=pwsa, pros=pros, prod=prod, pwfsd=pwfsd, psawgfl=psawgfl,
    )


# --------------------------------------------------------------------------- #
# Helpers: soil-moisture branch
# --------------------------------------------------------------------------- #
def _liquid_fraction(
    ptsam1m: torch.Tensor,
    rtf1: float, rtf2: float, rtf3: float, rtf4: float,
) -> torch.Tensor:
    """Liquid-water fraction ZF as a function of soil temperature.

    Reproduces srfwexc_vg_mod.F90 lines 219-227::

        ZF = 0.5*(1 - SIN(RTF4*(T-RTF3)))   if RTF2 < T < RTF1
        ZF = 1.0                             if T <= RTF2  (all liquid)
        ZF = 0.0                             if T >= RTF1  (all frozen)

    Outside the freeze window ZF is piecewise-constant, matching the
    upstream SRFENE definition.  Vectorised over (Ngrid, Nsoil).
    """
    t = ptsam1m
    zf = 0.5 * (1.0 - torch.sin(rtf4 * (t - rtf3)))
    zf = torch.where(t > rtf2, zf, torch.ones_like(zf))      # T <= RTF2 -> 1
    zf = torch.where(t < rtf1, zf, torch.zeros_like(zf))     # T >= RTF1 -> 0
    return zf


def _root_extraction_weights(
    pfrti: torch.Tensor,
    pevapti: torch.Tensor,
    pevapsnw: torch.Tensor,
    zliq: torch.Tensor,            # (Ngrid, Nsoil) liquid-equivalent moisture
    wpwp: torch.Tensor,            # (Ngrid,) wilting point
    zf: torch.Tensor,              # (Ngrid, Nsoil) liquid fraction
    ktvl: torch.Tensor, ktvh: torch.Tensor,
    ksoty: torch.Tensor,
    veg_table,
    nlevs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tile root extraction weights ZROOTW and normaliser ZEXT.

    Reproduces srfwexc_vg_mod.F90 lines 268-310 (TILES + LAYERS loops).
    Only tiles 4 (low-veg-dry, KTVL), 6 (high-veg-dry, KTVH) and 7
    (high-veg-snow, KTVH) carry root extraction.  For positive evap
    (dew deposition) all roots are in layer 1; for transpiration the
    weight is ``RVROOTSA(JK, ITYP) * (ZLIQ - RWPWPM + ZEPSILONADD)``.

    Returns
    -------
    zrootw: ``(Ngrid, Nsoil, 3)`` -- per-(grid,layer,tile-in-{4,6,7}).
    zext:   ``(Ngrid, 3)`` -- per-tile normaliser (clamped to >= ZEPSILON).
    """
    ngrid = pfrti.shape[0]
    dtype = pfrti.dtype
    device = pfrti.device
    tile_local = (3, 5, 6)          # 0-indexed for tiles 4, 6, 7

    # Per-veg-type root profiles: (Ngrid, Nsoil) for each of low / high veg.
    roots_lo = veg_table.gather_roots(ktvl.long().detach())
    roots_hi = veg_table.gather_roots(ktvh.long().detach())

    wpwp_b = wpwp[:, None].expand(-1, nlevs)

    zrootw = torch.zeros((ngrid, nlevs, 3), dtype=dtype, device=device)
    zext = torch.zeros((ngrid, 3), dtype=dtype, device=device)

    # Tile 4 (low-veg-dry) -> uses KTVL roots.
    zevap4 = pfrti[:, 3] * pevapti[:, 3]
    dew4 = zevap4 > 0.0
    # Layer 1 = 1 for dew; else roots*(zliq - wpwp).
    w4 = roots_lo * (zliq - wpwp_b)
    zrootw[:, 0, 0] = torch.where(dew4, torch.ones_like(zrootw[:, 0, 0]), w4[:, 0])
    zrootw[:, 1:, 0] = torch.where(
        dew4[:, None].expand(-1, nlevs - 1),
        torch.zeros_like(zrootw[:, 1:, 0]),
        w4[:, 1:],
    )
    zext[:, 0] = zrootw[:, :, 0].sum(dim=1)

    # Tile 6 (high-veg-dry) -> uses KTVH roots.
    zevap6 = pfrti[:, 5] * pevapti[:, 5]
    dew6 = zevap6 > 0.0
    w6 = roots_hi * (zliq - wpwp_b)
    zrootw[:, 0, 1] = torch.where(dew6, torch.ones_like(zrootw[:, 0, 1]), w6[:, 0])
    zrootw[:, 1:, 1] = torch.where(
        dew6[:, None].expand(-1, nlevs - 1),
        torch.zeros_like(zrootw[:, 1:, 1]),
        w6[:, 1:],
    )
    zext[:, 1] = zrootw[:, :, 1].sum(dim=1)

    # Tile 7 (high-veg-snow) -> uses KTVH roots; evap excludes PEVAPSNW.
    zevap7 = pfrti[:, 6] * (pevapti[:, 6] - pevapsnw)
    dew7 = zevap7 > 0.0
    w7 = roots_hi * (zliq - wpwp_b)
    zrootw[:, 0, 2] = torch.where(dew7, torch.ones_like(zrootw[:, 0, 2]), w7[:, 0])
    zrootw[:, 1:, 2] = torch.where(
        dew7[:, None].expand(-1, nlevs - 1),
        torch.zeros_like(zrootw[:, 1:, 2]),
        w7[:, 1:],
    )
    zext[:, 2] = zrootw[:, :, 2].sum(dim=1)

    # Clamp ZEXT to >= ZEPSILON (Fortran MAX(ZEPSILON, ZEXT)).
    zeps = 100.0 * torch.finfo(dtype).eps
    zext = torch.clamp(zext, min=zeps)
    return zrootw, zext


def _vg_hydraulics(
    w: torch.Tensor,             # (Ngrid,) moisture at the half-level
    wsat: torch.Tensor,
    wrest: torch.Tensor,
    wconsm: torch.Tensor,
    mvgalpha: torch.Tensor,
    mfac: torch.Tensor,
    lam: torch.Tensor,
    dmax: torch.Tensor,
    dmin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Van Genuchten hydraulic conductivity K and diffusivity D.

    Reproduces the half-level VG block in srfwexc_vg_mod.F90 (top layer
    lines 318-332, deeper layers 509-523).  The three-branch piecewise
    form clamps K/D at the air-dry floor (W <= 1.001*Wres) and near
    saturation (W >= 0.999*Wsat); in between the closed-form VG
    expressions are used.

    Returns
    -------
    (zk, zd): ``(Ngrid,)`` hydraulic conductivity [m/s] and diffusivity
    [m^2/s] at the given half-level moisture.
    """
    wfac = wsat - wrest
    se = (w - wrest) / wfac
    rmfac = 1.0 / mfac
    se_rmfac = se ** rmfac
    # NOTE the bracketing matches the Fortran closed form exactly.
    # The K formula is
    #   ZKD = ZWCONS * ZSE**ZLAM * (1 - (1 - ZSE**ZRMFAC)**ZMFAC)**2
    # The D formula is
    #   ZDD = ((1-ZMFAC)*ZWCONS / (ZALPHA*ZMFAC*ZWFAC)) * ZSE**(ZLAM-ZRMFAC)
    #         * ((1-ZSE**ZRMFAC)**ZMFAC + (1-ZSE**ZRMFAC)**(-ZMFAC) - 2)
    # The two share the building block (1 - se^rm) but bracket it
    # differently -- do NOT reuse the K-bracket for D.  We precompute the
    # raw power and form each bracket separately.
    raw_pow = 1.0 - se_rmfac                       # (1 - se^rm)
    k_bracket = 1.0 - raw_pow ** mfac              # 1 - (1 - se^rm)^m
    d_summand = raw_pow ** mfac + raw_pow ** (-mfac) - 2.0

    # Mid-branch K and D (closed-form VG).
    zk_mid = wconsm * (se ** lam) * (k_bracket ** 2)
    # D uses SIGN(MAX(ABS(x),eps),x) to protect against zero -- with the
    # default positive alpha/mfac/wfac this is just MAX(ABS(x),eps).
    denom = torch.clamp((mvgalpha * mfac * wfac).abs(), min=eps)
    zd_mid = ((1.0 - mfac) * wconsm / denom) * (se ** (lam - rmfac)) * d_summand

    # Branch selection.
    floor = w <= 1.001 * wrest
    sat = w >= 0.999 * wsat
    mid = ~(floor | sat)

    zk = torch.where(mid, zk_mid, torch.zeros_like(zk_mid))
    zk = torch.where(sat, wconsm, zk)                  # near-sat -> K = WCONS
    # floor -> K = 0 (already zero).

    zd = torch.where(mid, zd_mid, dmin)               # floor -> D = Dmin
    zd = torch.where(sat, dmax, zd)                   # near-sat -> D = Dmax
    return zk, zd


def _top_layer_coeffs(
    inp: SoilMoistInputs,
    ksoty, wsat, wrest, wcap, wpwp, wconsm, mvgalpha, mfac, lam,
    dmax, dmin, zf, zrootw, zext,
    rdaw, land, ptmst, rsimp, rsrdep, rho_h2o, lessro,
    rsigormin, rsigormax, rpsfr,
    pcfw, prhsw, psawgfl, pros,
):
    """SRFWEXC_VG section 2.3 -- top-layer coefficients.

    Returns the updated (pcfw, prhsw, psawgfl, pros) plus the
    diagnostics (zsawext_l1, zconds) needed by SRFWNG.
    """
    ngrid = inp.pwsam1m.shape[0]
    nlevs = rdaw.shape[0]
    dtype = inp.pwsam1m.dtype
    device = inp.pwsam1m.device
    zeps = 100.0 * torch.finfo(dtype).eps
    zhoh2o = 1.0 / rho_h2o

    # ZW = MAX(MAX(W(:,1), W(:,2)), WREST) -- half-level-1 moisture.
    w_top = torch.maximum(
        torch.maximum(inp.pwsam1m[:, 0], inp.pwsam1m[:, 1]), wrest
    )
    zk, zd = _vg_hydraulics(
        w_top, wsat, wrest, wconsm, mvgalpha, mfac, lam, dmax, dmin, zeps,
    )

    # Freezing reduction: ZFF = MIN(ZF(:,1), ZF(:,2)).
    if nlevs >= 2:
        zff = torch.minimum(zf[:, 0], zf[:, 1])
    else:
        zff = zf[:, 0]
    zd = zff * dmin + (1.0 - zff) * zd
    zk = (1.0 - zff) * zk                   # ZK = 0 for frozen soil

    # Runoff: VIC saturation-excess OR infiltration-limit (LESSRO switch).
    zrot = _runoff(
        inp, ksoty, wsat, wrest, wpwp, wconsm, dmax, dmin, zf,
        rdaw, rsrdep, lessro, ptmst, rho_h2o,
        rsigormin, rsigormax, rpsfr,
    )

    # Surface forcing ZWSFL = throughfall + snowmelt + conv - runoff.
    zwsfl = inp.ptsfl + inp.pmsn + inp.ptsfc - zrot

    # Tile contributions (tiles 3, 4, 6, 7, 8).
    # Tile 3 (wet-skin): PFRTI*PEVAPTI - PEINTTI.
    zwsfl = zwsfl + inp.pfrti[:, 2] * inp.pevapti[:, 2] - inp.peintti[:, 2]
    # Tiles 4, 6, 7: root-extraction split into layer 1.
    for it, tile in enumerate((3, 5, 6)):       # tiles 4, 6, 7 (0-indexed 3,5,6)
        if tile == 6:
            evap_tile = inp.pfrti[:, tile] * (inp.pevapti[:, tile] - inp.pevapsnw)
        else:
            evap_tile = inp.pfrti[:, tile] * inp.pevapti[:, tile]
        zextk = (evap_tile - inp.peintti[:, tile]) * zrootw[:, 0, it] / zext[:, it]
        zwsfl = zwsfl + zextk
    # Tile 8 (bare-soil): PFRTI*PEVAPTI + PEMSSN - PEINTTI.
    zwsfl = (
        zwsfl
        + inp.pfrti[:, 7] * inp.pevapti[:, 7]
        + inp.pemssn - inp.peintti[:, 7]
    )

    # PCFW(:,1) and PRHSW(:,1).
    if nlevs >= 2:
        denom_cfw = 0.5 * (rdaw[0] + rdaw[1])
    else:
        denom_cfw = 0.5 * rdaw[0]
    pcfw[:, 0] = zd * ptmst * rsimp / denom_cfw
    prhsw[:, 0] = ptmst * (-zk + zwsfl * zhoh2o) / rdaw[0]

    # Diagnostics.
    psawgfl[:, 0] = rho_h2o * zk
    pros = ptmst * zrot

    # Land mask: zero out non-land columns.
    land_b = land
    pcfw[:, 0] = pcfw[:, 0] * land_b
    prhsw[:, 0] = prhsw[:, 0] * land_b
    psawgfl[:, 0] = psawgfl[:, 0] * land_b
    pros = pros * land_b
    return pcfw, prhsw, psawgfl, pros, None, None


def _runoff(
    inp: SoilMoistInputs,
    ksoty, wsat, wrest, wpwp, wconsm, dmax, dmin, zf,
    rdaw, rsrdep, lessro, ptmst, rho_h2o,
    rsigormin, rsigormax, rpsfr,
) -> torch.Tensor:
    """Surface runoff ZROT (kg/m^2/s, positive out of the soil).

    Two branches gated by ``lessro``:
    * LESSRO = .TRUE.  -> VIC saturation-excess runoff (srfwexc_vg 340-391).
    * LESSRO = .FALSE. -> infiltration-limit runoff (srfwexc_vg 393-416).

    Returns the per-grid runoff rate (kg/m^2/s).  Surf points are zeroed
    by the caller's land mask.
    """
    ngrid = inp.pwsam1m.shape[0]
    nlevs = rdaw.shape[0]
    dtype = inp.pwsam1m.dtype
    device = inp.pwsam1m.device
    invtmst = 1.0 / ptmst
    js = ksoty

    zrsfl = inp.ptsfl + inp.pmsn + inp.ptsfc
    raining = zrsfl > 0.0

    if lessro:
        # VIC: relative saturation in the upper RSRDEP layer.
        # ZFRK(JK) = fraction of layer JK inside the RSRDEP window.
        # rdaw is constant across the batch, so we evaluate ZFRK once as a
        # plain 1-D tensor and broadcast (no per-grid state involved).
        rdaw_py = rdaw.detach().to("cpu", dtype=torch.float64).tolist()
        zfrk_list = []
        for jk in range(nlevs):
            zsum_lo = sum(rdaw_py[:jk]) if jk > 0 else 0.0
            zsum_hi = sum(rdaw_py[: jk + 1])
            val = max(0.0, min(rsrdep, zsum_hi) - zsum_lo) / rdaw_py[jk]
            zfrk_list.append(max(0.0, val))
        zfrk_1d = torch.as_tensor(zfrk_list, dtype=dtype, device=device)
        zfrk = zfrk_1d.expand(ngrid, nlevs)

        # Orographic sub-grid coefficient ZBWS.  RSIGORMIN/MAX are the
        # upstream defaults (100/1000) -- see sussoil_mod.F90:480-481.
        zroeff = (inp.psdor - rsigormin) / (inp.psdor + rsigormax)
        zroeff = torch.clamp(zroeff, min=0.0, max=0.5)
        zroeff = torch.clamp(zroeff, min=0.01)
        zbws = zroeff
        zb1 = 1.0 + zbws
        zbm = 1.0 / zb1

        # ZW = liquid water in upper RSRDEP window.
        zwk = torch.clamp(inp.pwsam1m - wpwp[:, None], min=0.0) * (
            1.0 - zf
        ) + wpwp[:, None]
        rdaw_g = rdaw.expand(ngrid, nlevs)
        zw = (zfrk * zwk * rdaw_g).sum(dim=1)
        zwmk = (wsat[:, None] - wpwp[:, None]) * (1.0 - zf) + wpwp[:, None]
        zwmax = (zfrk * zwmk * rdaw_g).sum(dim=1)

        zconw1 = zwmax * zb1
        zlyeps = torch.clamp(zw - zwmax, min=0.0)
        # ZLIMRS = -ZLYEPS unless ZLYEPS > 0.1*ZWMAX (then 0).
        zlimrs = -zlyeps
        zlimrs = torch.where(
            zlyeps > 0.1 * zwmax, torch.zeros_like(zlimrs), zlimrs
        )
        # ZLYSIC = saturation-area fraction.
        zlysic = torch.clamp((zw - zlyeps) / zwmax, min=0.0, max=1.0)
        # Guard divide-by-zero when ZWMAX = 0.
        zlysic = torch.where(
            zwmax != 0.0, zlysic, torch.zeros_like(zlysic)
        )
        # ZVOL = saturation-excess volume.
        zvol = torch.clamp(
            (1.0 - zlysic) ** zbm - zrsfl / (invtmst * rho_h2o * zconw1),
            min=0.0,
        )
        # ZROS (m): saturation-excess runoff.
        zros = zrsfl / (invtmst * rho_h2o) - torch.maximum(
            zwmax - zw, zlimrs
        )
        zros = torch.where(zvol > 0.0, zros + zwmax * zvol ** zb1, zros)
        zros = torch.clamp(zros, min=0.0)
        zrot = zros * rho_h2o * invtmst
        zrot = torch.where(raining, zrot, torch.zeros_like(zrot))
        return zrot

    # Infiltration-limit runoff.
    zdsurf = dmax
    zksurf = wconsm
    zff1 = zf[:, 0]
    zdsurf = zff1 * dmin + (1.0 - zff1) * dmax
    zksurf = (1.0 - zff1) * wconsm
    zinfmax = (
        zdsurf * (wsat - inp.pwsam1m[:, 0]) / (0.5 * rdaw[0]) + zksurf
    ) * rho_h2o
    # Large-scale precip runoff.
    zrol = torch.clamp(inp.ptsfl + inp.pmsn - zinfmax, min=0.0)
    # Convective precip runoff (RPSFR = 1 default; ZPSFR = 1).
    zroc = torch.clamp(inp.ptsfc - zinfmax, min=0.0)
    zrot = zrol + zroc
    return zrot


def _deep_layer_coeffs(
    inp: SoilMoistInputs,
    ksoty, wsat, wrest, wcap, wpwp, wconsm, mvgalpha, mfac, lam,
    dmax, dmin, zf, zrootw, zext,
    rdaw, land, ptmst, rsimp, rho_h2o,
    pcfw, prhsw, psawgfl,
):
    """SRFWEXC_VG section 2.4 -- coefficients for layers JK=2..KLEVS.

    Each half-level JK+1/2 uses MAX(W(:,JK), W(:,JK+1)) for the K/D
    half-level moisture; the gravity-flux K at the *top* of layer JK
    uses MAX(W(:,JK-1), W(:,JK)) (the ZKM variable in the Fortran).  The
    freezing reduction couples adjacent layers via MIN(ZF(:,JK), ...).
    """
    ngrid = inp.pwsam1m.shape[0]
    nlevs = rdaw.shape[0]
    dtype = inp.pwsam1m.dtype
    device = inp.pwsam1m.device
    zeps = 100.0 * torch.finfo(dtype).eps
    zhoh2o = 1.0 / rho_h2o
    land_b = land

    for jk in range(1, nlevs):
        # Half-level JK+1/2 moisture (for D and the lower K).
        if jk < nlevs - 1:
            w_half = torch.maximum(
                torch.maximum(inp.pwsam1m[:, jk], inp.pwsam1m[:, jk + 1]),
                wrest,
            )
        else:
            w_half = torch.maximum(inp.pwsam1m[:, jk], wrest)
        zk, zd = _vg_hydraulics(
            w_half, wsat, wrest, wconsm, mvgalpha, mfac, lam, dmax, dmin, zeps,
        )

        # ZKM = K at the top of this layer (half-level JK-1/2).
        wm = torch.maximum(
            torch.maximum(inp.pwsam1m[:, jk - 1], inp.pwsam1m[:, jk]), wrest
        )
        zkm, _ = _vg_hydraulics_toponly(
            wm, wsat, wrest, wconsm, mvgalpha, mfac, lam, zeps,
        )

        # Bottom layer: ZD = 0 (no flux below).
        if jk == nlevs - 1:
            zd_layer = torch.zeros_like(zd)
        else:
            zd_layer = zd

        # Freezing reduction.
        if jk < nlevs - 1:
            zff = torch.minimum(zf[:, jk], zf[:, jk + 1])
        else:
            zff = zf[:, jk]
        zffm = torch.minimum(zf[:, jk - 1], zf[:, jk])
        zd_layer = zff * dmin + (1.0 - zff) * zd_layer
        zk = (1.0 - zff) * zk
        zkm = (1.0 - zffm) * zkm

        # Root-extraction contributions to ZWSFL (tiles 4, 6, 7).
        zwsfl = torch.zeros((ngrid,), dtype=dtype, device=device)
        for it, tile in enumerate((3, 5, 6)):
            if tile == 6:
                evap_tile = inp.pfrti[:, tile] * (inp.pevapti[:, tile] - inp.pevapsnw)
            else:
                evap_tile = inp.pfrti[:, tile] * inp.pevapti[:, tile]
            zextk = (evap_tile - inp.peintti[:, tile]) * zrootw[:, jk, it] / zext[:, it]
            zwsfl = zwsfl + zextk

        # PCFW and PRHSW.
        if jk < nlevs - 1:
            denom_cfw = 0.5 * (rdaw[jk] + rdaw[jk + 1])
            pcfw[:, jk] = zd_layer * ptmst * rsimp / denom_cfw
        else:
            pcfw[:, jk] = torch.zeros((ngrid,), dtype=dtype, device=device)
        prhsw[:, jk] = ptmst * (zkm - zk + zwsfl * zhoh2o) / rdaw[jk]

        psawgfl[:, jk] = rho_h2o * zk

        # Land mask.
        pcfw[:, jk] = pcfw[:, jk] * land_b
        prhsw[:, jk] = prhsw[:, jk] * land_b
        psawgfl[:, jk] = psawgfl[:, jk] * land_b
    return pcfw, prhsw, psawgfl


def _vg_hydraulics_toponly(
    w, wsat, wrest, wconsm, mvgalpha, mfac, lam, eps,
):
    """Van Genuchten K only (no D), used for the ZKM top-of-layer K.

    Reproduces srfwexc_vg_mod.F90 lines 529-540 (the ZKM block).  Same
    three-branch piecewise form as :func:`_vg_hydraulics` but returns
    only K (ZKM is never used as a diffusivity).
    """
    wfac = wsat - wrest
    se = (w - wrest) / wfac
    rmfac = 1.0 / mfac
    se_rmfac = se ** rmfac
    raw_pow = 1.0 - se_rmfac
    # NOTE: same bracketing as :func:`_vg_hydraulics` -- the K closed form
    # is (1 - (1 - se^rm)^m)**2, NOT ((1 - se^rm)^m)**2.  See the lengthy
    # comment in _vg_hydraulics for the derivation.
    k_bracket = 1.0 - raw_pow ** mfac

    zkm_mid = wconsm * (se ** lam) * (k_bracket ** 2)

    floor = w <= wrest
    sat = w >= 0.999 * wsat
    mid = ~(floor | sat)
    zkm = torch.where(mid, zkm_mid, torch.zeros_like(zkm_mid))
    zkm = torch.where(sat, wconsm, zkm)
    # Return (zkm, None) so the call site mirrors _vg_hydraulics.
    return zkm, None


def _soil_moisture_update(
    pwsam1m, pwsadif, pcfw, psawgfl, ldland, rsimp, ptmst, rho_h2o,
):
    """SRFWINC -- convert Thomas solution to t+1 moisture + drainage diag.

    Reproduces srfwinc_mod.F90 lines 120-144::

        ZTPFAC3 = 1/RSIMP - 1
        ZCONS   = RHOH2O/PTMST
        PWSA(:,JK) = PWSADIF(:,JK) - ZTPFAC3*PWSAM1M(:,JK)
        PSAWGFL(:,JK) += ZCONS*PCFW(:,JK)*(PWSADIF(:,JK) - PWSADIF(:,JK+1))

    With RSIMP=1, ZTPFAC3=0 and PWSA = PWSADIF (the Thomas solution is
    the new moisture directly).  Surf points keep their t-1 moisture.
    """
    ztpfac3 = 1.0 / rsimp - 1.0
    zcons = rho_h2o / ptmst
    nlevs = pwsam1m.shape[1]
    land_b = ldland.to(pwsam1m.dtype)[:, None]

    pwsa_land = pwsadif - ztpfac3 * pwsam1m
    pwsa = torch.where(land_b != 0.0, pwsa_land, pwsam1m)

    # Update the gravity-flux diagnostic (layers 1..KLEVS-1).
    for jk in range(nlevs - 1):
        psawgfl[:, jk] = psawgfl[:, jk] + zcons * pcfw[:, jk] * (
            pwsadif[:, jk] - pwsadif[:, jk + 1]
        )
    # Surf points: zero the diagnostic.
    psawgfl = torch.where(land_b != 0.0, psawgfl, torch.zeros_like(psawgfl))
    return pwsa, psawgfl


def _clip_and_redistribute(
    pwsa, pwsam1m, pros, pwfsd, wsat, wrest, rdaw,
    land, ldland, ptmst, rho_h2o,
):
    """SRFWNG -- clip negative moisture, redistribute downward, split
    saturation excess into surface / sub-surface runoff.

    Reproduces srfwng_mod.F90 lines 130-182.  The clipping is a column
    recurrence: any negative moisture in layer JK is pushed into JK+1
    (scaled by the layer-thickness ratio), and any saturation excess
    becomes surface runoff (top layer) or sub-surface runoff (deeper
    layers).  The skin-reservoir clipping (PWL > PWLMX) is omitted --
    the offline driver leaves the skin reservoir to a separate path
    (M6 / SRFWL).

    Because the redistribution carries a layer-axis recurrence we
    unroll it as a Python loop over JK; with nlevs=4 this is three
    iterations.  The loop body is written functionally (per-layer
    moisture tensors accumulated in a list) to stay autograd-clean.
    """
    ngrid = pwsa.shape[0]
    nlevs = rdaw.shape[0]
    dtype = pwsa.dtype
    device = pwsa.device
    ztmst = 1.0 / ptmst
    zhoh2o = 1.0 / rho_h2o
    zidi = 1.0 / rdaw                                     # (Nsoil,)

    prod = torch.zeros((ngrid,), dtype=dtype, device=device)
    land_b = land

    # Functional accumulation of the clipped moisture profile.
    sol: list[torch.Tensor] = [pwsa[:, jk] for jk in range(nlevs)]

    for jk in range(nlevs):
        w = sol[jk]
        # Clip negative: ZDWD = MIN(W, 0); push into JK+1.  Note ZDI here
        # is the *layer thickness* (RDAW) -- upstream names it ZDI, with
        # ZIDI = 1/RDAW its reciprocal.  The inter-layer redistribution
        # uses the thickness ratio rdaw[jk]/rdaw[jk+1].
        zdwd = torch.clamp(w, max=0.0)
        w = w - zdwd
        if jk < nlevs - 1:
            sol[jk + 1] = sol[jk + 1] + zdwd * rdaw[jk] * zidi[jk + 1]
        # Saturation excess.  Runoff / drainage carry the excess through
        # the full layer depth, so they scale by rdaw[jk] (kg/m^2).
        zrrri = torch.clamp(w - wsat, min=0.0)
        if jk == 0:
            pros = pros + rdaw[jk] * rho_h2o * zrrri
            pwfsd = pwfsd + zdwd * rho_h2o * rdaw[jk] * ztmst
        else:
            prod = prod + rdaw[jk] * rho_h2o * zrrri
        sol[jk] = w - zrrri

    pwsa_out = torch.stack(sol, dim=1)
    # Land mask: zero runoff for surf points, keep t-1 moisture.
    pwsa_out = torch.where(
        ldland.to(dtype)[:, None] != 0.0, pwsa_out, pwsam1m
    )
    pros = pros * land_b
    prod = prod * land_b
    pwfsd = pwfsd * land_b
    return pwsa_out, pros, prod, pwfsd


__all__ = [
    "SoilTempInputs",
    "SoilTempOutputs",
    "compute_soil_temperature",
    "SoilMoistInputs",
    "SoilMoistOutputs",
    "compute_soil_moisture",
]
