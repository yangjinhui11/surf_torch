"""Snow density evolution (``SRFSN_LWIMP`` section 3.1).

This module is the Python port of the snow-density block that lives in
``SRFSN_LWIMP`` (``srfsn_lwimp_mod.F90`` lines 357-377 and 494-527) -- the
single-layer snow scheme with prognostic density.  The simplified single-
layer path the port's :mod:`surf_pytorch.physics.snow` mirrors is
density evolution is done in the sibling ``SRFSN_LWIMP`` (no ``s``) and in
the multi-layer ``SRFSN_RSN`` (called from ``SRFSN_DRIVER``).  This module
supplies the single-layer density evolution used by ``SRFSN_LWIMP``,
so the port's snow prognostic advances ``PRSNM1M`` alongside ``PSNM1M``
and ``PTSNM1M``.

What the scheme does
--------------------
The new snow density ``PRSN`` (kg/m^3) is built from five contributions,
exactly as in ``SRFSN_LWIMP`` lines 494-527:

1. **Fresh-snow density** ``ZRHOMINSN`` -- a linear function of the
   surface temperature and the quarter-power of the surface wind speed,
   floored at ``RHOMINSND`` (50 kg/m^3)::

       ZRHOMINSN = RHOMINSNA + RHOMINSNB*(PTSRF - RTT)
                                 + RHOMINSNC*(U^2 + V^2)^0.25
       ZRHOMINSN = max(ZRHOMINSN, RHOMINSND)

2. **Depth-weighted mixing** of the old pack density and the fresh-snow
   density, weighted on snow *depth* (the post-CY43 formulation, Fortran
   lines 506-508).  ``ZSSFC + ZSSFL`` is the snowfall over the coupling
   interval::

       ZRSTAR = (PSSNM1M + PTMST*(ZSSFC+ZSSFL))
              / (PTMST*(ZSSFC+ZSSFL)/ZRHOMINSN + PSSNM1M/PRSNM1M)

3. **Overburden compaction** (viscous deformation under the snow weight,
   Fortran lines 511-512)::

       ZRSNDTOVER = 0.5*PSSNM1M*RG
                  / (RSNDTOVERA*EXP(RSNDTOVERB*(RTT-PTSN)
                                    + RSNDTOVERC*PRSNM1M))

4. **Destructive metamorphism** (rounding of grains, Fortran lines 514-
   515); the ``RSNDTDESTC`` term damps metamorphism once the density
   exceeds ``RSNDTDESTROI``::

       ZRSNDTDEST = RSNDTDESTA
                  * EXP(-RSNDTDESTB*(RTT-PTSN)
                        - RSNDTDESTC*max(0, PRSNM1M - RSNDTDESTROI))

5. **Melt compaction** (Fortran lines 517-521), the density increase
   driven by internal melt/refreeze.  ``ZPMSNINT`` is the internal melt
   energy released as the snow warms through the melt window; the term
   only acts when ``ZPMSNINT > 0`` and the pack is deep enough
   (``PSSNM1M >= 15`` kg/m^2)::

       ZRSNDTMELT = ZTMST*(ZPMSNINT*PTMST/RLMLT)/(PSSNM1M - ZLW)

   ``ZPMSNINT`` and ``ZLW`` come from the snow liquid-water analytical
   functions (lines 357-377): the liquid-water capacity ``ZLWC``, the
   melt-window shape ``ZFUNC`` and its derivative ``ZDFUNC``, the actual
   liquid water ``ZLW = ZFUNC*ZLWC``, and the internal-melt flux
   ``ZPMSNINT = ZCONSB*ZFRSN*(PTSN - PTSNM1M)/PTMST`` with
   ``ZCONSB = RLMLT*ZLWC*ZDFUNC``.

The new density is then::

    PRSN = ZRSTAR + ZRSTAR*(ZRSNDTOVER + ZRSNDTDEST + ZRSNDTMELT)*PTMST
    PRSN = min(450, PRSN)
    PRSN = max(RHOMINSN, PRSN)

At no-snow points (``LLNOSNOW``) the density is reset to ``RHOMINSN``
(100 kg/m^3), matching the Fortran ``PRSN(JL) = RHOMINSN`` branch.

Sign / unit convention
----------------------
All densities are in kg/m^3.  ``PTSRF`` is the surface (skin) temperature
in K; ``PUSRF``/``PVSRF`` are the surface wind components in m/s.  The
wind enters only via ``(U^2+V^2)^0.25`` so the magnitude is what matters.
``PTSN`` is the *new* snow temperature from the snow solve (the Fortran
uses the just-computed ``PTSN(JL)``, not ``PTSNM1M``, for the
compaction/metamorphism exponentials).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..constants import ConstantBundle
from ..tables.soil import SoilTable


@dataclass
class SnowDensityInputs:
    """Inputs to the single-layer snow-density solve.

    Each tensor is ``(Ngrid,)`` unless noted.  Naming mirrors the
    ``SRFSN_LWIMP`` local / dummy variables so a side-by-side audit is
    easy.
    """

    # ---- Snow state at t-1 ----
    pssnm1m: torch.Tensor       # PSSNM1M  snow mass (SWE)     [kg/m^2]   (Ngrid,)
    prsnm1m: torch.Tensor       # PRSNM1M  snow density        [kg/m^3]   (Ngrid,)
    ptsnm1m: torch.Tensor       # PTSNM1M  snow temperature    [K]        (Ngrid,)

    # ---- New snow state at t (from SRFSN_LWIMP solve) ----
    ptsn: torch.Tensor          # PTSN  new snow temperature   [K]        (Ngrid,)

    # ---- Surface forcing ----
    ptsrf: torch.Tensor         # PTSRF  surface (skin) T      [K]        (Ngrid,)
    pusrf: torch.Tensor         # PUSRF  surface zonal wind    [m/s]      (Ngrid,)
    pvsrf: torch.Tensor         # PVSRF  surface merid wind    [m/s]      (Ngrid,)

    # ---- Snowfall throughfall (per grid, large-scale + convective) ----
    pssfl: torch.Tensor         # PSSFL  large-scale snowfall  [kg/m^2/s] (Ngrid,)
    pssfc: torch.Tensor         # PSSFC  convective snowfall   [kg/m^2/s] (Ngrid,)

    # ---- Snow geometry / mask ----
    zfrsn: torch.Tensor         # ZFRSN  snow tile fraction    [-]        (Ngrid,)
    llnosnow: torch.Tensor      # LLNOSNOW  no-snow mask (bool) (Ngrid,)

    # ---- Timestep ----
    ptmst: torch.Tensor         # PTMST  coupling interval     [s]        scalar


@dataclass
class SnowDensityOutputs:
    """Outputs of the single-layer snow-density solve."""

    prsn: torch.Tensor          # PRSN  new snow density       [kg/m^3]   (Ngrid,)


def compute_snow_density(
    inp: SnowDensityInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
    *,
    differentiable: bool = True,
) -> SnowDensityOutputs:
    """Run the single-layer snow-density evolution (``SRFSN_LWIMP`` 3.1).

    Args
    ----
    inp:
        :class:`SnowDensityInputs` -- packed snow state, surface forcing
        and the snow geometry / mask from the snow solve.
    soil_table:
        :class:`surf_pytorch.tables.soil.SoilTable` -- provides the
        density constants (rhominsn*, rsndtover*, rsndtdest*,
        rhomaxsn_new) and the liquid-water-capacity scalars
        (rlwcswe*, rtempamp, rhoci, rhoice, rlamice, rdsmax, rfrsmall,
        rfrtiny).
    consts:
        :class:`ConstantBundle` -- provides ``rtt``, ``rlmlt``, ``rg``
        and ``pi``.
    differentiable:
        If ``False`` run under ``torch.no_grad()`` for speed.

    Returns
    -------
    :class:`SnowDensityOutputs` -- the new snow density ``PRSN``.
    """

    def _core() -> SnowDensityOutputs:
        return _density_core(inp, soil_table, consts)

    if not differentiable:
        with torch.no_grad():
            return _core()
    return _core()


def _density_core(
    inp: SnowDensityInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
) -> SnowDensityOutputs:
    """Internal: assumes the no_grad / enable_grad decision is already made."""
    dtype = inp.pssnm1m.dtype
    device = inp.pssnm1m.device

    # ------------------------------------------------------------------ #
    # Scalar constants (same read order as the Fortran ASSOCIATE).
    # ------------------------------------------------------------------ #
    rtt = consts.rtt
    rlmlt = consts.rlmlt
    rg = consts.g
    rpi = consts.pi

    sc = soil_table.consts
    # density-evolution constants (rdnml_params.F90:28-65)
    rhominsna = float(sc.rhominsna)
    rhominsnb = float(sc.rhominsnb)
    rhominsnc = float(sc.rhominsnc)
    rhominsnd = float(sc.rhominsnd)
    rsndtovera = float(sc.rsndtovera)
    rsndtoverb = float(sc.rsndtoverb)
    rsndtoverc = float(sc.rsndtoverc)
    rsndtdesta = float(sc.rsndtdesta)
    rsndtdestb = float(sc.rsndtdestb)
    rsndtdestc = float(sc.rsndtdestc)
    rsndtdestroi = float(sc.rsndtdestroi)
    rhomaxsn_new = float(sc.rhomaxsn_new)
    # supporting scalars (sussoil / rdnml_params)
    rhominsn = float(sc.rhominsn)
    rdsmax = float(sc.rdsmax)
    rhoice = float(sc.rhoice)
    rhoci = float(sc.rhoci)
    rlwcswea = float(sc.rlwcswea)
    rlwcsweb = float(sc.rlwcsweb)
    rlwcswec = float(sc.rlwcswec)
    rtempamp = float(sc.rtempamp)
    rfrtiny = float(sc.rfrtiny)

    ptmst = inp.ptmst
    ztmst = 1.0 / ptmst
    zt0 = rtt
    zhoice = 1.0 / rhoice

    # ------------------------------------------------------------------ #
    # Aliases for readability.
    # ------------------------------------------------------------------ #
    pssnm1m = inp.pssnm1m
    prsnm1m = inp.prsnm1m
    ptsnm1m = inp.ptsnm1m
    ptsn = inp.ptsn
    ptsrf = inp.ptsrf
    pusrf = inp.pusrf
    pvsrf = inp.pvsrf
    pssfl = inp.pssfl
    pssfc = inp.pssfc
    zfrsn = inp.zfrsn
    llnosnow = inp.llnosnow

    # ------------------------------------------------------------------ #
    # Snow liquid-water analytical functions (SRFSN_LWIMP lines 357-377).
    # These feed the melt-compaction term ZRSNDTMELT below.  They are the
    # same melt-window functions used by SRFSN_LWIMP, recomputed here so
    # the density block is self-contained.
    # ------------------------------------------------------------------ #
    zeps = 1.0e-12
    zifrsn = 1.0 / torch.clamp(zfrsn, min=rfrtiny)
    # real snow depth at t-1 (capped at RDSNMAX) and the liquid-water
    # capacity fraction ZCONSLWC (collapses to RLWCSWEC at high density).
    zdsnr = pssnm1m * zifrsn / torch.clamp(prsnm1m, min=zeps)
    zdsn_safe = torch.clamp(zdsnr, min=zeps)
    zconslwc = torch.clamp((rlwcswea - prsnm1m) / rlwcswea, min=0.0)
    zlwc = pssnm1m * (rlwcsweb + (rlwcswec - rlwcsweb) * zconslwc)
    # rescale LWC by the effective-depth cap RDSNMAX
    zlwc = zlwc * torch.clamp(zdsnr, max=rdsmax) / zdsn_safe

    # ZFUNC / ZDFUNC: melt-window sine ramp over [RTT-RTEMPAMP, RTT].
    # Below the window ZFUNC=0 (frozen, no liquid water); inside the
    # window ZFUNC ramps 0->1->0 and ZDFUNC is its derivative; above RTT
    # ZFUNC=0 (all liquid) and ZDFUNC = pi/RTEMPAMP.
    t_below = ptsnm1m < (zt0 - 0.5 * rtempamp)
    t_inside = (ptsnm1m >= (zt0 - 0.5 * rtempamp)) & (ptsnm1m < zt0)
    phase = rpi * (ptsnm1m - zt0) / rtempamp
    zfunc_inside = 1.0 + torch.sin(phase)
    zdfunc_inside = torch.cos(phase) * rpi / rtempamp
    zdfunc_above = torch.full_like(ptsnm1m, rpi / rtempamp)
    zfunc = torch.where(t_below, torch.zeros_like(ptsnm1m),
                torch.where(t_inside, zfunc_inside, torch.zeros_like(ptsnm1m)))
    zdfunc = torch.where(t_below, torch.zeros_like(ptsnm1m),
                torch.where(t_inside, zdfunc_inside, zdfunc_above))
    zlw = zfunc * zlwc

    # ZCONSB: apparent heat-capacity contribution from the liquid-water
    # melt window (latent heat released per K of warming).
    zconsb = rlmlt * zlwc * zdfunc
    # ZPMSNINT: internal melt/refreeze energy flux (Fortran line 447).
    zpmsnint = zconsb * zfrsn * (ptsn - ptsnm1m) * ztmst

    # ------------------------------------------------------------------ #
    # 3.1 NEW SNOW DENSITY (SRFSN_LWIMP lines 494-527).
    # ------------------------------------------------------------------ #
    # Fresh-snow density (lines 498-500).
    wind_sq = pusrf ** 2 + pvsrf ** 2
    zrhominsn = rhominsna + rhominsnb * (ptsrf - zt0) \
        + rhominsnc * wind_sq ** 0.25
    zrhominsn = torch.clamp(zrhominsn, min=rhominsnd)

    # Depth-weighted mixing of old pack + fresh snow (lines 506-508).
    zssf = (pssfl + pssfc) * ptmst            # snowfall over the interval
    prsnm1m_safe = torch.clamp(prsnm1m, min=zeps)
    denom = zssf / torch.clamp(zrhominsn, min=zeps) + pssnm1m / prsnm1m_safe
    zrstar = (pssnm1m + zssf) / torch.clamp(denom, min=zeps)

    # Overburden compaction (lines 511-512).
    zrsndtover = 0.5 * pssnm1m * rg \
        / (rsndtovera * torch.exp(rsndtoverb * (zt0 - ptsn)
                                  + rsndtoverc * prsnm1m))

    # Destructive metamorphism (lines 514-515).
    zrsndtdest = rsndtdesta \
        * torch.exp(-rsndtdestb * (zt0 - ptsn)
                    - rsndtdestc * torch.clamp(prsnm1m - rsndtdestroi, min=0.0))

    # Melt compaction (lines 517-521): only where internal melt is
    # positive AND the pack is deep enough (PSSNM1M >= 15 kg/m^2).
    melt_active = (zpmsnint > 0.0) & (pssnm1m >= 15.0)
    pack_safe = torch.clamp(pssnm1m - zlw, min=zeps)
    zrsndtmelt = torch.where(
        melt_active,
        ztmst * (zpmsnint * ptmst / rlmlt) / pack_safe,
        torch.zeros_like(pssnm1m),
    )

    # Total compaction -> new density (lines 523-527).  The single-layer
    # path caps the evolved density at 450 (Fortran line 526); the multi-
    # layer path and the namelist RHOMAXSN_NEW use 500.  We follow the
    # single-layer SRFSN_LWIMP cap of 450.
    zrsndt = zrsndtover + zrsndtdest + zrsndtmelt
    prsn_snow = zrstar + zrstar * zrsndt * ptmst
    prsn_snow = torch.clamp(prsn_snow, max=450.0)
    prsn_snow = torch.clamp(prsn_snow, min=rhominsn)

    # No-snow branch (Fortran line 491): density resets to RHOMINSN.
    prsn = torch.where(llnosnow, torch.full_like(prsn_snow, rhominsn), prsn_snow)

    return SnowDensityOutputs(prsn=prsn)


__all__ = [
    "SnowDensityInputs",
    "SnowDensityOutputs",
    "compute_snow_density",
]
