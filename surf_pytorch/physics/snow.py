"""HTESSEL single-layer snow (``SRFSN_LWIMPS``).

This module is the Python port of ``SRFSN_LWIMPS`` -- the simplified
single-layer snow scheme (with diagnostic liquid water and rainfall
interception) used by ``SURFTSTPS_CTL`` when ``NCSNEC == 1`` (the
offline-driver default; ``sudim1s.F90`` line 74).  The richer multi-
layer ``SRFSN_WEBALS`` is *not* ported -- it is gated behind
``LESLML = .FALSE.`` which the offline driver never sets, so it never
runs in our comparison harness.

What the scheme does
--------------------
The single-layer snow prognostic integrates three coupled budgets over
one coupling interval ``PTMST``:

1. **Snow mass** ``PSN`` (water-equivalent, kg/m^2): advanced by the
   large-scale + convective snowfall (``PSSFL/PSSFC``), the snow-tile
   evaporation/sublimation, the snow-under-forest evaporation
   (``PEVAPSNW``) and the throughfall interception / melt.  Sublimation
   is taken from the snow tiles 5 (snow-low-veg) and 7 (snow-high-veg)
   weighted by their tile fractions.

2. **Snow temperature** ``PTSN``: implicit Euler over the snow heat
   capacity and the snow-to-soil conductive coupling.  The "snow heat
   capacity" is the apparent heat capacity that absorbs the freeze/melt
   release via the analytical melt-transition functions ``ZFUNC`` and
   its derivative ``ZDFUNC`` (a sine ramp over ``[RTT-RTEMPAMP, RTT]``).
   The melt branches then redistribute the timestep between heating to
   ``RTT`` and consuming latent heat during melt.

3. **Snow-soil ground heat flux** ``PGSN``: the conductive flux between
   the snow-base (at the new snow temperature ``PTSN``) and the top soil
   layer (at ``PTSAM1M(:,1)``), through the snow thermal resistance
   ``ZSNRES = 0.5*ZDSN/(RLAMICE*(PRSNM1M*ZHOICE)**RALAMSN)`` in series
   with the half-top-soil-layer resistance ``ZSOILRES = 1/RVLAMSK(8)``.
   ``PGSN`` is normalised by the snow tile fraction ``ZFRSN`` so it is a
   grid-box-mean flux ready to feed into ``SRFTS`` as the surface heat
   flux from the snow pack.

Sign convention
---------------
``PGSN`` is *positive downward* (heat flowing from the snow pack into
the soil), matching the ``ZSURFL`` convention in
:mod:`surf_pytorch.physics.soil`.  Sensible and latent heat fluxes from
``VDFDIFH1S`` follow the same downward-positive convention.  The latent
contribution is ``RLSTT*PEVAP`` on the snow-low-veg tile (sublimation)
and ``RLVTT*(PEVAPTI-PEVAPSNW) + RLSTT*PEVAPSNW`` on the snow-high-veg
tile (the under-canopy fraction sublimates, the rest evaporates the
liquid intercepted by the canopy).

Cold-start simplifications
--------------------------
The offline driver cold-starts the snow scheme with:

* ``PEVAPSNW`` (snow-under-forest evaporation) = 0
* ``PSSFC`` / ``PSSFL`` (convective / large-scale snowfall) = the
  solid-precip throughfall; if absent, both are zero (no snow input).
* ``PTSFC`` / ``PTSFL`` (convective / large-scale rainfall throughfall)
  = the liquid-precip throughfall; used by the snow-rainfall
  interception budget.
* ``PHLICEM1M`` (lake-ice thickness) = 0 (FLake stub).

All of these are exposed on :class:`SnowInputs` with explicit defaults
of zero so a caller can build the minimum (state + fluxes + tiles) and
let the scheme treat them as cold-start.

Index convention
----------------
``PFRTI`` / ``PAHFSTI`` / ``PEVAPTI`` / ``PSSRFLTI`` are 0-indexed in
this port.  Fortran tile 5 (snow on low vegetation) maps to
:data:`surf_pytorch.config.TILE_LOW_VEG_SNOW` (4) and tile 7 (snow under
high vegetation) maps to :data:`surf_pytorch.config.TILE_HIGH_VEG_SNOW`
(6).  Tile indices are guarded by ``ntile`` so a smaller tile dimension
(e.g. a 4-tile test slab) does not raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import (
    TILE_WET_SKIN,
    TILE_LOW_VEG_DRY,
    TILE_LOW_VEG_SNOW,
    TILE_HIGH_VEG_DRY,
    TILE_HIGH_VEG_SNOW,
    TILE_BARE_SOIL,
    TILE_LAKE,
    SurfConfig,
)
from ..constants import ConstantBundle
from ..tables.soil import SoilTable


# --------------------------------------------------------------------------- #
# Fortran tile indices used by SRFSN_LWIMPS, mapped to the port tile enum.
# Fortran PFRTI(JL,3..8) -> Python pfrti[:, 2..7]; lake is PFRTI(JL,9) -> 8.
# --------------------------------------------------------------------------- #
# Snow tile-fraction indices (1-based in Fortran).
_F90_TILE_SNOW_LOW = 5      # -> port TILE_LOW_VEG_SNOW (4)
_F90_TILE_SNOW_HIGH = 7     # -> port TILE_HIGH_VEG_SNOW (6)
# Snowfall-interception scaling grid fraction (Fortran tiles 3..8 +
# lake tile 9 if present).  Fortran tile 3 == port TILE_WET_SKIN (2)
# and tile 8 == port TILE_BARE_SOIL (7), so port indices 2..7 (+8 for
# lake).  The earlier code used TILE_LOW_VEG_DRY (port 3 == F90 4),
# which silently dropped the wet-skin tile from the sum -- caught by
# the M6b ULP comparison against SURF_REF_SNOW.
_INTERCEPT_FRAC_LO = TILE_WET_SKIN           # port 2 == F90 3
_INTERCEPT_FRAC_HI = TILE_BARE_SOIL          # port 7 == F90 8


@dataclass
class SnowInputs:
    """All inputs to the HTESSEL single-layer snow solve.

    Each tensor is ``(Ngrid,)`` unless noted.  Per-tile fields are
    ``(Ngrid, Ntile)``.  Naming mirrors the ``SRFSN_LWIMPS`` dummy list
    so a side-by-side audit is easy.

    Cold-start note: the snowfall / throughfall / interception-mismatch
    inputs (``pssfc``, ``pssfl``, ``ptsfc``, ``ptsfl``, ``pevapsnw``)
    all default to zero and may be left at that for a cold start.
    """

    # ---- Snow state at t-1 ----
    pssnm1m: torch.Tensor             # PSSNM1M  snow mass (SWE)      [kg/m^2]    (Ngrid,)
    ptsnm1m: torch.Tensor             # PTSNM1M  snow temperature     [K]         (Ngrid,)
    prsnm1m: torch.Tensor             # PRSNM1M  snow density         [kg/m^3]    (Ngrid,)

    # ---- Soil top-layer temperature (for snow-soil coupling) ----
    ptsam1m_top: torch.Tensor         # PTSAM1M(:,1)  soil T at top  [K]         (Ngrid,)

    # ---- Per-tile fluxes (from VDFDIFH1S / SURFSEB) ----
    pfrti: torch.Tensor               # PFRTI    tile fractions     [-]         (Ngrid, Ntile)
    pahfsti: torch.Tensor             # PAHFSTI  tile sensible heat  [W/m^2]     (Ngrid, Ntile)
    pevapti: torch.Tensor             # PEVAPTI  tile moisture flux  [kg/m^2/s]  (Ngrid, Ntile)
    pssrflti: torch.Tensor            # PSSRFLTI tile net SW         [W/m^2]     (Ngrid, Ntile)

    # ---- Per-grid radiation ----
    pslrfl: torch.Tensor              # PSLRFL   net LW              [W/m^2]     (Ngrid,)

    # ---- Snowfall throughfall (per grid) ----
    pssfc: torch.Tensor               # PSSFC  convective snowfall    [kg/m^2/s]  (Ngrid,)
    pssfl: torch.Tensor               # PSSFL  large-scale snowfall  [kg/m^2/s]  (Ngrid,)

    # ---- Rainfall throughfall (per grid; for snow-rainfall interception) ----
    ptsfc: torch.Tensor               # PTSFC  convective rainfall   [kg/m^2/s]  (Ngrid,)
    ptsfl: torch.Tensor               # PTSFL  large-scale rainfall  [kg/m^2/s]  (Ngrid,)

    # ---- Snow-under-forest evaporation mismatch (per grid) ----
    pevapsnw: torch.Tensor            # PEVAPSNW  snow-under-forest evap [kg/m^2/s] (Ngrid,)

    # ---- Timestep ----
    ptmst: torch.Tensor               # PTMST  timestep              [s]         scalar

    # ---- Lake ice (FLake stub; zero at cold start) ----
    phlicem1m: Optional[torch.Tensor] = None   # PHLICEM1M lake ice thickness [m] (Ngrid,)


@dataclass
class SnowOutputs:
    """All outputs of the HTESSEL single-layer snow solve."""

    # ---- Snow state at t ----
    ptsn: torch.Tensor                # PTSN  new snow temperature   [K]         (Ngrid,)
    pgsn: torch.Tensor                # PGSN  ground heat flux from snow -> soil  [W/m^2] (Ngrid,)

    # ---- Diagnostics (mirror the SRFSN_LWIMPS local variables) ----
    psstar: torch.Tensor              # ZSSTAR  updated snow mass (SWE) [kg/m^2] (Ngrid,)
    plwc: torch.Tensor                # ZLWC   snow liquid water capacity [kg/m^2] (Ngrid,)
    zfrsn: torch.Tensor               # ZFRSN  snow tile fraction (normaliser) [-] (Ngrid,)

    # ---- No-snow mask, exposed so the density-evolution block can ---- #
    # reset density to RHOMINSN at LLNOSNOW points (SRFSN_LWIMP line 491).
    llnosnow: Optional[torch.Tensor] = None   # LLNOSNOW  no-snow mask (bool) (Ngrid,)

    # ---- Melt rate, exposed so the snow-albedo aging block can split ---- #
    # its melt/cold branches on PMSN > 0 (SRFSN line 354).  Zero at no-melt
    # points; positive only where the melt branch is active.
    pmsn: Optional[torch.Tensor] = None       # PMSN  melt rate [kg/m^2/s] (Ngrid,)


def compute_snow(
    inp: SnowInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    differentiable: Optional[bool] = None,
) -> SnowOutputs:
    """Run the HTESSEL single-layer snow solve (``SRFSN_LWIMPS``).

    Args
    ----
    inp:
        :class:`SnowInputs` -- packed snow state, fluxes and masks.
    soil_table:
        :class:`surf_pytorch.tables.soil.SoilTable` -- provides the snow
        constants (rhoice, rhoci, rlamsn, rlwcswe*, rtempamp, rtauf,
        rtaua, rfrsmall, rfrtiny, rdsmax).
    consts:
        :class:`ConstantBundle` -- provides ``rtt``, ``rlmlt``, ``rlvtt``,
        ``rlstt``, ``rday`` and ``pi``.
    config:
        :class:`SurfConfig` -- ``use_flake`` selects the lake branch.
    differentiable:
        Override ``config.differentiable`` for this call.

    Returns
    -------
    :class:`SnowOutputs` -- new snow temperature, snow-soil ground heat
    flux, updated snow mass, liquid-water capacity and the snow tile
    fraction normaliser.
    """
    diff = config.differentiable if differentiable is None else differentiable

    def _core() -> SnowOutputs:
        return _snow_core(inp, soil_table, consts, config)

    if not diff:
        with torch.no_grad():
            return _core()
    return _core()


def _snow_core(
    inp: SnowInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
    config: SurfConfig,
) -> SnowOutputs:
    """Internal: assumes the no_grad / enable_grad decision is already made."""
    ngrid = inp.pssnm1m.shape[0]
    ntile = inp.pfrti.shape[-1]
    dtype = inp.pssnm1m.dtype
    device = inp.pssnm1m.device

    # ------------------------------------------------------------------ #
    # Scalar constants (read in the same order as the Fortran ASSOCIATE).
    # ------------------------------------------------------------------ #
    rtt = consts.rtt
    rlmlt = consts.rlmlt
    rlvtt = consts.rlvtt
    rlstt = consts.rlstt
    rpi = consts.pi
    rday = consts.rday

    sc = soil_table.consts
    rhoice = float(sc.rhoice)
    rlamsn = float(sc.rlamdsn_pow)
    rdsmax = float(sc.rdsmax)
    rhoci = float(sc.rhoci)
    rlamice = float(sc.rlamice)
    rfrsmall = float(sc.rfrsmall)
    rfrtiny = float(sc.rfrtiny)
    rlwcswea = float(sc.rlwcswea)
    rlwcsweb = float(sc.rlwcsweb)
    rlwcswec = float(sc.rlwcswec)
    rtempamp = float(sc.rtempamp)
    rtauf = float(sc.rtauf)
    rtaua = float(sc.rtaua)

    ptmst = inp.ptmst

    # Soil resistance for heat conduction in the half top soil layer.
    # RVLAMSK(8) is the bare-soil (desert) skin-layer conductivity
    # (susveg line 175: 15 W/m^2/K); SRFSN_LWIMPS uses its inverse.
    # The value is class-agnostic in the snow context (only index 8 is
    # ever used), so we hard-wire the desert value to match the Fortran.
    rvlamsk_8 = 15.0
    zsoilres = 1.0 / rvlamsk_8

    # ------------------------------------------------------------------ #
    # Computational constants (Fortran lines 164-174).
    # ------------------------------------------------------------------ #
    zhoice = 1.0 / rhoice
    zt0 = rtt
    ztmst = 1.0 / ptmst
    zexpf = math_exp(-rtauf * ptmst / rday)   # not used by SRFSN_LWIMPS body
                                              # (kept for parity; the offline
                                              # driver computes it but the
                                              # body never references it).
    del zexpf  # silence unused-variable warning
    zlina = rtaua * ptmst / rday              # likewise unused in the body
    del zlina
    zitempamp = 1.0 / rtempamp

    # ZRDSNMAX / ZRDSNRESMAX (Fortran lines 173-174): both 1.0 m.
    zrdsnmax = 1.0
    zrdsnresmax = 1.0

    # ------------------------------------------------------------------ #
    # Per-grid setup (Fortran lines 189-213).
    # ------------------------------------------------------------------ #
    # Snow tile fraction ZFRSN = max(PFRTI(5) + PFRTI(7), RFRTINY).
    pf_low_snow = _tile_col(inp.pfrti, TILE_LOW_VEG_SNOW, dtype, device)
    pf_high_snow = _tile_col(inp.pfrti, TILE_HIGH_VEG_SNOW, dtype, device)
    zfrsn = torch.clamp(pf_low_snow + pf_high_snow, min=rfrtiny)
    zifrsn = 1.0 / zfrsn

    # LLNOSNOW: snow tile fraction below RFRSMALL.
    llnosnow = zfrsn < rfrsmall

    # Snowfall interception scaling: grid fraction covered by tiles
    # 3..8 (Fortran) plus lake tile 9 if FLake is on.
    zgridfrac = _interception_gridfrac(inp.pfrti, config.use_flake, dtype, device)

    # Scaled snowfall / throughfall (lines 208-211).
    zssfc = zgridfrac * inp.pssfc
    zssfl = zgridfrac * inp.pssfl
    ztsfc = zgridfrac * inp.ptsfc
    ztsfl = zgridfrac * inp.ptsfl

    # ------------------------------------------------------------------ #
    # Tile-column extractions (vectorised over Ngrid).
    # ------------------------------------------------------------------ #
    # Net heat flux at the snow surface uses tiles 5 and 7 (1-based).
    pf5 = pf_low_snow                                       # PFRTI(:,5)
    pf7 = pf_high_snow                                      # PFRTI(:,7)
    pahfsti_5 = _tile_col(inp.pahfsti, TILE_LOW_VEG_SNOW, dtype, device)
    pahfsti_7 = _tile_col(inp.pahfsti, TILE_HIGH_VEG_SNOW, dtype, device)
    pevapti_5 = _tile_col(inp.pevapti, TILE_LOW_VEG_SNOW, dtype, device)
    pevapti_7 = _tile_col(inp.pevapti, TILE_HIGH_VEG_SNOW, dtype, device)
    pssrflti_5 = _tile_col(inp.pssrflti, TILE_LOW_VEG_SNOW, dtype, device)
    pssrflti_7 = _tile_col(inp.pssrflti, TILE_HIGH_VEG_SNOW, dtype, device)

    ptsnm1m = inp.ptsnm1m
    pssnm1m = inp.pssnm1m
    prsnm1m = inp.prsnm1m
    ptsam1m_top = inp.ptsam1m_top
    pevapsnw = inp.pevapsnw

    # ------------------------------------------------------------------ #
    # Main loop body, vectorised (Fortran lines 215-331).
    # ------------------------------------------------------------------ #
    # Default outputs (no-snow branch values; overwritten where snow).
    zt0_t = torch.as_tensor(zt0, dtype=dtype, device=device)

    # --- Net heat flux at the snow surface (ZHFLUX) ------------------- #
    # ZHFLUXP = PFRTI(5)*(PAHFSTI(5)+RLSTT*PEVAPTI(5))
    #         + PFRTI(7)*(PAHFSTI(7)+RLVTT*(PEVAPTI(7)-PEVAPSNW)
    #                     +RLSTT*PEVAPSNW)
    #         + PFRTI(5)*PSSRFLTI(5)
    #         + PFRTI(7)*PSSRFLTI(7)
    #         + (PFRTI(5)+PFRTI(7))*PSLRFL
    # ZHFLUX  = ZIFRSN * ZHFLUXP
    zhfluxp = (
        pf5 * (pahfsti_5 + rlstt * pevapti_5)
        + pf7 * (pahfsti_7 + rlvtt * (pevapti_7 - pevapsnw) + rlstt * pevapsnw)
        + pf5 * pssrflti_5
        + pf7 * pssrflti_7
        + (pf5 + pf7) * inp.pslrfl
    )
    zhflux = zifrsn * zhfluxp

    # --- Preliminary snow mass ZSSTAR --------------------------------- #
    # ZSSTAR = PSSNM1M + PTMST*(ZSSFC+ZSSFL+PFRTI(5)*PEVAPTI(5)
    #                          +PFRTI(7)*PEVAPSNW)
    # Negative snow mass after P-E is clamped to zero (lines 240-243).
    zsstar = pssnm1m + ptmst * (
        zssfc + zssfl + pf5 * pevapti_5 + pf7 * pevapsnw
    )
    zsstar = torch.clamp(zsstar, min=0.0)

    # --- Snow liquid water (ZLWC) ------------------------------------- #
    # ZDSNR  = PSSNM1M / (PRSNM1M * ZFRSN)  -- snow depth (real)
    # ZCONSLWC = max(0, (RLWCSWEA - PRSNM1M) / RLWCSWEA)
    # ZLWC   = PSSNM1M * (RLWCSWEB + (RLWCSWEC - RLWCSWEB)*ZCONSLWC)
    # then rescale by min(ZDSNR, ZRDSNMAX)/ZDSNR.
    zdsnr = pssnm1m / (prsnm1m * zfrsn)
    zconslwc = torch.clamp((rlwcswea - prsnm1m) / rlwcswea, min=0.0)
    zlwc = pssnm1m * (rlwcsweb + (rlwcswec - rlwcsweb) * zconslwc)
    zdsn_rp = torch.minimum(zdsnr, torch.as_tensor(zrdsnmax, dtype=dtype, device=device))
    # Guard divide-by-zero: where ZDSNR is zero (no snow) the snow
    # branch is not taken, but the expression must still be finite for
    # autograd.  Use a safe divide and zero out the no-snow points later.
    zdsnr_safe = torch.where(
        zdsnr != 0.0, zdsnr, torch.ones_like(zdsnr)
    )
    zlwc = zlwc * zdsn_rp / zdsnr_safe

    # --- Analytical melt-transition functions ZFUNC / ZDFUNC ---------- #
    # The branch on PTSNM1M selects three windows:
    #   * T < ZT0 - 0.5*RTEMPAMP  -> both 0 (cold snow, no liquid water)
    #   * ZT0 - 0.5*RTEMPAMP <= T < ZT0 -> sinusoidal ramp
    #   * T >= ZT0               -> ZFUNC=0, ZDFUNC=RPI*ZITEMPAMP (melting)
    cold = ptsnm1m < (zt0 - 0.5 * rtempamp)
    melting = ptsnm1m >= zt0
    ramp = ~(cold | melting)

    # Ramp branch:
    #   ZFUNC  = 1 + sin(RPI*(T - ZT0)*ZITEMPAMP)
    #   ZDFUNC = cos(RPI*(T - ZT0)*ZITEMPAMP) * RPI * ZITEMPAMP
    arg = rpi * (ptsnm1m - zt0) * zitempamp
    zfunc_ramp = 1.0 + torch.sin(arg)
    zdfunc_ramp = torch.cos(arg) * rpi * zitempamp

    zfunc = torch.zeros_like(ptsnm1m)
    zdfunc = torch.zeros_like(ptsnm1m)
    zfunc = torch.where(ramp, zfunc_ramp, zfunc)
    zdfunc = torch.where(
        ramp, zdfunc_ramp,
        torch.where(
            melting,
            torch.full_like(ptsnm1m, rpi * zitempamp),
            zdfunc,                                  # cold -> 0
        ),
    )

    # --- Terms for the implicit temperature equation ------------------ #
    # ZCONSB     = RLMLT*ZLWC*ZDFUNC
    # ZCONSAMAX  = (PRSNM1M*RHOCI*ZRDSNMAX)*ZHOICE
    # ZCOND1     = (PSSNM1M*RHOCI*ZHOICE)*ZIFRSN
    # ZCONSA     = min(ZCOND1, ZCONSAMAX)
    # ZCONS2     = PTMST / (ZCONSA + ZCONSB)
    zconsb = rlmlt * zlwc * zdfunc
    zconsamax = (prsnm1m * rhoci * zrdsnmax) * zhoice
    zcond1 = (pssnm1m * rhoci * zhoice) * zifrsn
    zconsa = torch.minimum(zcond1, zconsamax)
    zcons_denom = zconsa + zconsb
    zcons_denom_safe = torch.where(
        zcons_denom != 0.0, zcons_denom, torch.ones_like(zcons_denom)
    )
    zcons2 = ptmst / zcons_denom_safe

    # --- Real snow depth ZDSN and snow-soil resistance ZRS ------------ #
    # ZDSN    = min(PSSNM1M/(PRSNM1M*ZFRSN), ZRDSNRESMAX)
    # ZSNRES  = 0.5*ZDSN / (RLAMICE*(PRSNM1M*ZHOICE)**RALAMSN)
    # ZRS     = 1 / (ZSNRES + ZSOILRES)
    zdsn = torch.minimum(
        pssnm1m / (prsnm1m * zfrsn),
        torch.as_tensor(zrdsnresmax, dtype=dtype, device=device),
    )
    zsnres = 0.5 * zdsn / (
        rlamice * (prsnm1m * zhoice) ** rlamsn
    )
    zrs = 1.0 / (zsnres + zsoilres)

    # --- Throughfall interception (lines 286-299) --------------------- #
    # ZWFLUXRF = (PFRTI(5)+PFRTI(7))*(ZTSFC+ZTSFL)     -- mass advected
    # ZCOND3   = ZWFLUXRF*PTMST
    # ZCOND4   = ZLWC*(1 - ZFUNC)
    # ZWFLUXRF = min(ZCOND3, ZCOND4)
    # ZCOND5   = ZCONSA*(ZT0 - PTSNM1M)*ZTMST + ZT0*ZRS
    # ZCOND6   = RLMLT*ZWFLUXRF*ZTMST
    # ZHFLUXRF1= min(ZCOND5, ZCOND6); ZHFLUXRF1 = max(ZHFLUXRF1, 0)
    # ZWFLUXRF = ZHFLUXRF1 / RLMLT                     -- intercepted mass
    # ZSSTAR   = ZSSTAR + ZWFLUXRF*PTMST
    zwfluxrf_raw = (pf5 + pf7) * (ztsfc + ztsfl)
    zcond3 = zwfluxrf_raw * ptmst
    zcond4 = zlwc * (1.0 - zfunc)
    zwfluxrf = torch.minimum(zcond3, zcond4)
    zcond5 = zconsa * (zt0 - ptsnm1m) * ztmst + zt0 * zrs
    zcond6 = rlmlt * zwfluxrf * ztmst
    zhfluxrf1 = torch.minimum(zcond5, zcond6)
    zhfluxrf1 = torch.clamp(zhfluxrf1, min=0.0)
    zwfluxrf = zhfluxrf1 / rlmlt
    zsstar = zsstar + zwfluxrf * ptmst

    # --- Solve implicit temperature equation -------------------------- #
    # ZTSTARI = 1 / (1 + ZCONS2*ZRS)
    # ZTSTAR  = (PTSNM1M + ZCONS2*(ZHFLUX + ZHFLUXRF1 + PTSAM1M(1)*ZRS))
    #           * ZTSTARI
    ztstari = 1.0 / (1.0 + zcons2 * zrs)
    ztstar = (
        ptsnm1m + zcons2 * (zhflux + zhfluxrf1 + ptsam1m_top * zrs)
    ) * ztstari

    # --- Branches (no-melt vs melt) ----------------------------------- #
    no_melt = ztstar <= zt0
    melt = ~no_melt

    # No-melt branch (lines 304-307):
    #   PTSN = ZTSTAR
    #   PGSN = (ZTSTAR - PTSAM1M(1))*ZRS
    ptsn_no_melt = ztstar
    pgsn_no_melt = (ztstar - ptsam1m_top) * zrs

    # Melt branch (lines 308-327).
    # ZDTHI = 1/(ZHFLUX + ZHFLUXRF1 - (ZT0 - PTSAM1M(1))*ZRS)
    # ZDTH  = ((ZT0 - PTSNM1M)*(ZCONSA + ZCONSB)) * ZDTHI
    # ZDTM  = PTMST - ZDTH
    # ZGSN  = (ZT0 - PTSAM1M(1))*ZRS
    # ZMSN  = (ZHFLUX + ZHFLUXRF1 - ZGSN)/RLMLT
    # ZSSTAR1 = ZSSTAR - ZDTM*ZMSN*ZFRSN
    zdthi_denom = zhflux + zhfluxrf1 - (zt0 - ptsam1m_top) * zrs
    zdthi_safe = torch.where(
        zdthi_denom != 0.0, zdthi_denom, torch.ones_like(zdthi_denom)
    )
    zdthi = 1.0 / zdthi_safe
    zdth = (zt0 - ptsnm1m) * (zconsa + zconsb) * zdthi
    zdtm = ptmst - zdth
    zgsn_melt = (zt0 - ptsam1m_top) * zrs
    zmsn = (zhflux + zhfluxrf1 - zgsn_melt) / rlmlt
    zsstar1 = zsstar - zdtm * zmsn * zfrsn

    # Partial-melt branch (ZSSTAR1 >= 0): snow remains.
    #   PTSN = ZT0; PGSN = (ZT0 - PTSAM1M(1))*ZRS
    ptsn_partial = zt0_t
    pgsn_partial = zgsn_melt

    # Full-melt branch (ZSSTAR1 < 0): all snow melted.
    #   PTSN = ZT0
    #   PGSN = (((ZT0-PTSAM1M(1))*ZRS)*(ZDTM+ZDTH)
    #           + (ZHFLUX+ZHFLUXRF1)*(PTMST-ZDTM-ZDTH)) * ZTMST
    pgsn_full = (
        ((zt0 - ptsam1m_top) * zrs) * (zdtm + zdth)
        + (zhflux + zhfluxrf1) * (ptmst - zdtm - zdth)
    ) * ztmst

    full_melt = zsstar1 < 0.0
    pgsn_melt = torch.where(full_melt, pgsn_full, pgsn_partial)
    ptsn_melt = zt0_t

    # Combine the melt / no-melt branches.
    ptsn_with_snow = torch.where(no_melt, ptsn_no_melt, ptsn_melt)
    pgsn_with_snow = torch.where(no_melt, pgsn_no_melt, pgsn_melt)

    # ------------------------------------------------------------------ #
    # Apply the no-snow branch (Fortran lines 216-220):
    #   ZSSTAR = PSSNM1M + PTMST*(ZSSFC + ZSSFL)  -- snowfall only
    #   PTSN   = ZT0
    #   PGSN   = 0
    #   ZLWC   = 0
    # ------------------------------------------------------------------ #
    zsstar_nosnow = pssnm1m + ptmst * (zssfc + zssfl)
    ptsn = torch.where(llnosnow, zt0_t, ptsn_with_snow)
    pgsn = torch.where(
        llnosnow,
        torch.zeros_like(pgsn_with_snow),
        pgsn_with_snow,
    )
    zsstar_out = torch.where(llnosnow, zsstar_nosnow, zsstar)
    zlwc_out = torch.where(
        llnosnow, torch.zeros_like(zlwc), zlwc
    )

    # ------------------------------------------------------------------ #
    # Normalise PGSN to the grid-square (Fortran lines 336-338):
    #   PGSN = ZFRSN * PGSN
    # ------------------------------------------------------------------ #
    pgsn = zfrsn * pgsn

    # PMSN (melt rate) -- only physically meaningful on the melt branch
    # (SRFSN line 308-327); zero elsewhere so the snow-albedo aging block's
    # melt/cold split on PMSN > 0 (line 354) is well-defined everywhere.
    pmsn_out = torch.where(melt, zmsn, torch.zeros_like(zmsn))

    return SnowOutputs(
        ptsn=ptsn,
        pgsn=pgsn,
        psstar=zsstar_out,
        plwc=zlwc_out,
        zfrsn=zfrsn,
        llnosnow=llnosnow,
        pmsn=pmsn_out,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tile_col(
    pfrti: torch.Tensor, idx: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """Return ``pfrti[:, idx]`` if the tile axis is wide enough.

    Falls back to zeros if the tile dimension is narrower than ``idx + 1``
    (e.g. a 4-tile test slab that omits the snow tiles).  This mirrors
    the ``ntile > TILE_*`` guards used by
    :meth:`LandSurface._update_soil_temperature`.
    """
    ntile = pfrti.shape[-1]
    if ntile > idx:
        return pfrti[..., idx]
    return torch.zeros(pfrti.shape[:-1], dtype=dtype, device=device)


def _interception_gridfrac(
    pfrti: torch.Tensor, use_flake: bool, dtype: torch.dtype, device
) -> torch.Tensor:
    """Snowfall-interception scaling grid fraction (Fortran lines 197-206).

    The grid fraction is the sum of tiles 3..8 (Fortran 1-based), i.e.
    port tiles 2..7 (low-veg-dry .. bare-soil).  If FLake is active and
    the lake tile (port 8) is fully covering (PFRTI(:,9) == 1) the grid
    fraction collapses to zero (no snow over a full lake); otherwise the
    lake fraction is added.
    """
    ntile = pfrti.shape[-1]
    zgridfrac = torch.zeros(pfrti.shape[:-1], dtype=dtype, device=device)
    # Tiles 4..8 in Fortran -> port indices 3..7.
    for port_idx in range(_INTERCEPT_FRAC_LO, _INTERCEPT_FRAC_HI + 1):
        if ntile > port_idx:
            zgridfrac = zgridfrac + pfrti[..., port_idx]

    if use_flake and ntile > TILE_LAKE:
        lake_frac = pfrti[..., TILE_LAKE]
        full_lake = lake_frac == 1.0
        # Full-lake points get zero (no snow there).
        zgridfrac = torch.where(
            full_lake, torch.zeros_like(zgridfrac), zgridfrac + lake_frac
        )
    return zgridfrac


def math_exp(x: float) -> float:
    """``math.exp`` indirection so the import lives at module top."""
    import math
    return math.exp(x)


__all__ = [
    "SnowInputs",
    "SnowOutputs",
    "compute_snow",
]
