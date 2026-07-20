"""Skin-reservoir / canopy interception update (``SRFWL``).

This module is the Python port of ``SRFWL`` -- the skin-reservoir
(canopy interception) prognostic update used by ``SURFTSTP_CTL``
(``surftstp_ctl_mod.F90:612``).  It advances the skin-reservoir water
content ``PWL`` (intercepted water, kg/m^2) by one coupling interval
and produces the throughfall fluxes ``PTSFC`` / ``PTSFL`` (convective
/ large-scale rain reaching the soil) together with the per-tile
interception-layer evaporation mismatch ``PEINTTI`` that the soil-
moisture solve consumes.

What the scheme does
--------------------
The skin reservoir represents the water held on leaves and bare
surfaces between rain events.  Over one coupling interval ``PTMST``
the prognostic advances through five vectorisable stages
(``srfwl_mod.F90`` 164-300):

1. **Clamp to capacity** (label 2): ``PWL = clip(PWLM1M, 0, PWLMX)``.
   Any clipping is booked onto ``PEINTTI(:,3)`` (the wet-skin tile) as
   a virtual evaporation that the soil-W solve must see.

2. **Evaporation of the reservoir** (label 3): when the wet-skin tile
   evaporates (``PEVAPTI(:,3) < 0``) the reservoir is drawn down by
   ``PTMST*PFRTI(:,3)*PEVAPTI(:,3)``, floored at zero.  The realised
   flux (after the zero-floor) is booked onto ``PEINTTI(:,3)``.

3. **Dew collection** (label 4): tiles 4/6/7/8/10 (port 3/5/6/7/9) with
   a positive moisture flux deposit dew onto the reservoir up to
   ``PWLMX``.  The snow-under-high-veg tile (F90 7 / port 6) excludes
   the snow-under-canopy sublimation ``PEVAPSNW`` (only the liquid
   fraction is intercepted).  Each realised dew increment is booked
   onto the respective ``PEINTTI`` column.

4. **Budget accumulation** (label 5): ``PFWEL1 = (PWL - PWLM1M)/PTMST``
   (the net skin-reservoir flux tendency, kg/m^2/s).

5. **Interception of precipitation** (label 6): large-scale and
   convective rainfall is intercepted by the vegetation cover
   ``(CVL+CVH)*RVINTER`` up to the remaining capacity ``PWLMX-PWL``.
   The throughfall ``PTSFL`` / ``PTSFC`` is what reaches the soil.

Sign convention
---------------
``PEVAPTI`` is positive *downward* (dew/deposition into the surface),
negative upward (evaporation).  ``PEINTTI`` follows the same
convention: a positive value means the interception reservoir
*absorbed* flux that the soil therefore does *not* see (so the soil-W
solve subtracts it from its surface moisture input), a negative value
means the reservoir *released* flux the soil must see as an extra
input.  The soil-moisture solve consumes
``PEINTTI = PEVAPTI(: realisable) - PEVAPTI(: intercepted)`` per tile.

Cold-start simplifications
--------------------------
* Convective rain ``PRSFC`` is zero on the cold start (the port's
  :class:`Forcing` does not split convective vs large-scale precip),
  so ``PTSFC`` stays zero; only the large-scale branch ``PRSFL`` is
  active.  This matches the convention used for the soil-moisture
  throughfall in :meth:`LandSurface._update_soil_moisture`.
* ``PEVAPSNW`` (snow-under-forest evaporation) is zero until the
  single-layer snow scheme threads it back; the dew-collection branch
  therefore treats the snow-high-veg tile identically to the others.

Index convention
----------------
All per-tile tensors are 0-indexed.  Fortran tile 3 (wet skin) maps to
:data:`surf_pytorch.config.TILE_WET_SKIN` (2); the dew-collection tiles
F90 4/6/7/8/10 map to port 3/5/6/7/9 (low-veg-dry / high-veg-dry /
high-veg-snow / bare-soil / urban).  Tile indices are guarded by
``ntile`` so a narrower slab does not raise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import (
    TILE_WET_SKIN,
    TILE_LOW_VEG_DRY,
    TILE_HIGH_VEG_DRY,
    TILE_HIGH_VEG_SNOW,
    TILE_BARE_SOIL,
    TILE_URBAN,
)
from ..tables.veg import VegTable


# --------------------------------------------------------------------------- #
# Fortran tile indices used by SRFWL, mapped to the port tile enum.
# Fortran PFRTI(JL,3) == port TILE_WET_SKIN (2); dew-collection tiles are
# Fortran 3/4/6/7/8/10 == port 2/3/5/6/7/9.  Indices are guarded by ntile below.
# --------------------------------------------------------------------------- #
# Dew-collection tile set (Fortran tiles 3, 4, 6, 7, 8, 10).  Each entry is the
# (port_index, is_snow_under_high_veg) pair; only the snow-under-high-veg
# tile excludes PEVAPSNW from its intercepted flux.
_DEW_TILES = (
    (TILE_WET_SKIN, False),        # F90 3 -> port 2
    (TILE_LOW_VEG_DRY, False),     # F90 4 -> port 3
    (TILE_HIGH_VEG_DRY, False),    # F90 6 -> port 5
    (TILE_HIGH_VEG_SNOW, True),    # F90 7 -> port 6 (snow-under-high-veg)
    (TILE_BARE_SOIL, False),       # F90 8 -> port 7
    (TILE_URBAN, False),           # F90 10 -> port 9
)


@dataclass
class CanopyInputs:
    """All inputs to the skin-reservoir update.

    Each tensor is ``(Ngrid,)`` unless noted.  Per-tile fields are
    ``(Ngrid, Ntile)``.  Naming mirrors the ``SRFWL`` dummy list.
    """

    # Coupling interval [s].
    ptmst: torch.Tensor             # PTMST  (scalar or (Ngrid,))

    # t-1 skin-reservoir water content [kg/m^2].
    pwlm1m: torch.Tensor            # PWLM1M

    # Vegetation cover fractions (0-1) and interception capacity [kg/m^2].
    pcvl: torch.Tensor              # PCVL   low-veg cover
    pcvh: torch.Tensor              # PCVH   high-veg cover
    pwlmx: torch.Tensor             # PWLMX  max skin-reservoir capacity

    # Per-tile fractions / moisture flux [(Ngrid, Ntile)].
    pfrti: torch.Tensor             # PFRTI  tile fractions (0-1)
    pevapti: torch.Tensor           # PEVAPTI  surface moisture flux [kg/m^2/s]

    # Rainfall fluxes at the surface [kg/m^2/s].
    prsfc: torch.Tensor             # PRSFC  convective rain
    prsfl: torch.Tensor             # PRSFL  large-scale rain

    # Snow-under-forest evaporation [kg/m^2/s] (excluded from the
    # snow-high-veg dew collection); zero on the cold start.
    pevapsnw: torch.Tensor          # PEVAPSNW

    # Land mask (bool).
    ldland: torch.Tensor            # LDLAND
    lewbsoilfix: bool = True         # YDSOIL%LEWBSOILFIX


@dataclass
class CanopyOutputs:
    """Outputs of the skin-reservoir update.

    All ``(Ngrid,)`` except ``peintti`` which is ``(Ngrid, Ntile)``.
    """

    pwl: torch.Tensor               # PWL  t+1 skin-reservoir water [kg/m^2]
    pfwel1: torch.Tensor            # PFWEL1  net skin-reservoir flux [kg/m^2/s]
    ptsfc: torch.Tensor             # PTSFC  convective throughfall [kg/m^2/s]
    ptsfl: torch.Tensor             # PTSFL  large-scale throughfall [kg/m^2/s]
    peintti: torch.Tensor           # PEINTTI  interception-layer evap mismatch


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def update_canopy(
    inp: CanopyInputs,
    veg_table: VegTable,
) -> CanopyOutputs:
    """Advance the skin-reservoir prognostic by one coupling step.

    Vectorised transcription of ``SRFWL`` (``srfwl_mod.F90`` 164-300).
    Every stage is elementwise torch math over the ``(Ngrid,)`` /
    ``(Ngrid, Ntile)`` axes, so a single call reproduces the upstream
    JL loop exactly (including the per-tile dew-collection loop, which
    is column-local).

    Parameters
    ----------
    inp:
        :class:`CanopyInputs` -- t-1 state + fluxes + masks.
    veg_table:
        :class:`VegTable` -- provides ``RVINTER`` (interception
        efficiency, default 0.5).

    Returns
    -------
    :class:`CanopyOutputs`
        The t+1 skin-reservoir water ``PWL``, the net flux tendency
        ``PFWEL1``, the convective / large-scale throughfall
        ``PTSFC`` / ``PTSFL`` and the per-tile interception-layer
        evaporation mismatch ``PEINTTI``.
    """
    device = inp.pwlm1m.device
    dtype = inp.pwlm1m.dtype
    ngrid = inp.pwlm1m.shape[0]
    ntile = inp.pevapti.shape[-1]
    ptmst = inp.ptmst
    # Broadcast a scalar timestep to (Ngrid,) for uniform arithmetic.
    if ptmst.dim() == 0:
        ptmst = ptmst.expand(ngrid).to(dtype=dtype, device=device)
    ztmst = 1.0 / ptmst                          # 1/PTMST

    # Interception efficiency RVINTER (YDVEG%RVINTER, default 0.5).
    rvinter = veg_table.consts.rvinter
    # Convective-precip efficiency factor RPSFR (YDSOIL%RPSFR, default 2.0).
    # Upstream folds PRSFC through RPSFR: ZTPRCP = PRSFC*RPSFR and the
    # intercepted mass is scaled back by 1/RPSFR (ZPSFR = 1/RPSFR).  The
    # veg table does not yet carry RPSFR, so we hard-code the offline-
    # driver default (rdnml_params.F90:299 RPSFR = 2.0).  This mirrors the
    # single value the comparison harness runs with.
    rpsfr = 2.0
    zpsfr = 1.0 / rpsfr

    # Security floor on the reservoir capacity and precip flux
    # (ZEPTINY = 10*TINY(RHOH2O); ZEPPRCP = ZEPTINY).  TINY(float64) ~ 2e-16.
    zeptiny = torch.full((ngrid,), 10.0 * torch.finfo(dtype).tiny,
                         dtype=dtype, device=device)
    zepprcp = zeptiny

    ldland = inp.ldland                       # (Ngrid,) bool
    land = ldland.to(dtype=dtype)             # 1.0 on land, 0.0 on sea

    # ------------------------------------------------------------------ #
    # 1. Initialise PEINTTI and apply the LEWBSOILFIX reservoir clamp.
    # ------------------------------------------------------------------ #
    peintti = torch.zeros((ngrid, ntile), dtype=dtype, device=device)
    pwl_clamped = torch.minimum(inp.pwlmx - zeptiny, inp.pwlm1m)
    pwl_clamped = torch.clamp(pwl_clamped, min=0.0) * land
    if inp.lewbsoilfix:
        pwl = pwl_clamped
    else:
        pwl = inp.pwlm1m * land
    if ntile > TILE_WET_SKIN and inp.lewbsoilfix:
        clamp_flux = (pwl_clamped - inp.pwlm1m) * ztmst
        peintti[:, TILE_WET_SKIN] = clamp_flux * land

    # ------------------------------------------------------------------ #
    # 2. Evaporation of the skin reservoir (label 3, PEVAPTI(:,3) < 0).
    # ------------------------------------------------------------------ #
    # Wet-skin tile is port TILE_WET_SKIN (2) == Fortran tile 3.  Guarded
    # by ntile so a narrower slab does not index out of range.
    evap_wet = torch.zeros((ngrid,), dtype=dtype, device=device)
    if ntile > TILE_WET_SKIN:
        evap_wet = inp.pevapti[:, TILE_WET_SKIN]
    is_evap = evap_wet < 0.0
    # ZQHFLW = PTMST*PFRTI(:,3)*PEVAPTI(:,3)  (negative: draining reservoir).
    zqhf_lw = ptmst * _tile_frac(inp.pfrti, TILE_WET_SKIN, ntile) * evap_wet
    # PWL = max(0, PWL + ZQHFLW); only where evaporating.
    zwl_evap = pwl
    pwl_new = torch.clamp(zwl_evap + zqhf_lw, min=0.0)
    pwl = torch.where(is_evap, pwl_new, pwl)
    # Book the realised (post-floor) flux onto PEINTTI(:,3):
    #   PEINTTI(:,3) += (PWL - ZWL)/PTMST.
    realised_evap = (pwl - zwl_evap) * ztmst
    if ntile > TILE_WET_SKIN:
        peintti[:, TILE_WET_SKIN] = peintti[:, TILE_WET_SKIN] + \
            torch.where(is_evap, realised_evap, torch.zeros_like(realised_evap))

    dew_touched_reservoir = torch.zeros((ngrid,), dtype=torch.bool, device=device)

    # ------------------------------------------------------------------ #
    # 3. Dew collection (label 4): tiles 4/6/7/8/10 with ZELIQUID > 0
    #    deposit dew onto the reservoir up to PWLMX.
    # ------------------------------------------------------------------ #
    for port_idx, is_snow_high in _DEW_TILES:
        if port_idx >= ntile:
            continue
        pevap_tile = inp.pevapti[:, port_idx]
        if is_snow_high:
            # F90 tile 7: ZELIQUID = PFRTI*(PEVAPTI - PEVAPSNW) -- only the
            # liquid fraction is intercepted; the snow-under-canopy
            # sublimation PEVAPSNW bypasses the reservoir.
            zliquid = inp.pfrti[:, port_idx] * (pevap_tile - inp.pevapsnw)
        else:
            zliquid = inp.pfrti[:, port_idx] * pevap_tile
        is_dew = zliquid > 0.0
        # ZQHFLW = PTMST*ZELIQUID; PWL += min(PWLMX - ZEPTINY - PWL, ZQHFLW).
        zqhf = ptmst * zliquid
        zwl_dew = pwl
        headroom = inp.pwlmx - zeptiny - zwl_dew
        inc = torch.minimum(headroom, zqhf)
        pwl = torch.where(is_dew, zwl_dew + inc, pwl)
        # Book the realised dew increment onto PEINTTI(:, tile).
        realised_dew = (pwl - zwl_dew) * ztmst
        dew_touched_reservoir = dew_touched_reservoir | (is_dew & (pwl != zwl_dew))
        peintti[:, port_idx] = peintti[:, port_idx] + \
            torch.where(is_dew, realised_dew, torch.zeros_like(realised_dew))

    # ------------------------------------------------------------------ #
    # 4. Budget accumulation (label 5): PFWEL1 = (PWL - PWLM1M)/PTMST.
    #    Sea points carry PFWEL1 = 0 (reservoir is forced to zero there).
    # ------------------------------------------------------------------ #
    pfwel1 = (pwl - inp.pwlm1m) * ztmst * land

    # ------------------------------------------------------------------ #
    # 5. Interception of precipitation (label 6).
    #    Large-scale: ZMPRCP = min(PWLMX - ZEPTINY - PWL, PTMST*ZIPRCP);
    #                 PTSFL = PRSFL - ZMPRCP/PTMST.
    #    Convective:  scaled by RPSFR (ZTPRCP = PRSFC*RPSFR, mass back by
    #                 1/RPSFR); PTSFC = PRSFC - ZMPRCP/PTMST.
    #    Sea points: throughfall = raw rainfall.
    # ------------------------------------------------------------------ #
    headroom_l = inp.pwlmx - zeptiny - pwl

    # Large-scale rain.
    zziprcp_l = inp.prsfl * (inp.pcvl + inp.pcvh) * rvinter
    zmprcp_l = torch.minimum(headroom_l, ptmst * zziprcp_l)
    has_ls = inp.prsfl > zepprcp
    # ZMPRCP is added back to the reservoir (PWL += ZMPRCP) and removed
    # from the throughfall (PTSFL = PRSFL - ZMPRCP/PTMST).  Both updates
    # only fire where there is large-scale rain (has_ls); elsewhere the
    # reservoir is untouched and the throughfall is zero.
    zmprcp_l = torch.where(has_ls, zmprcp_l, torch.zeros_like(zmprcp_l))
    pwl_before_ls = pwl
    pwl = pwl + zmprcp_l
    ptsfl_land = torch.where(has_ls, inp.prsfl - zmprcp_l * ztmst,
                             torch.zeros_like(inp.prsfl))

    # Convective rain (RPSFR-folded).  Recompute the headroom after the
    # large-scale interception has filled part of the reservoir.
    headroom_c = inp.pwlmx - zeptiny - pwl
    ztprcp_c = inp.prsfc * rpsfr
    zziprcp_c = ztprcp_c * (inp.pcvl + inp.pcvh) * rvinter
    zmprcp_c = torch.minimum(headroom_c, ptmst * zziprcp_c) * zpsfr
    has_cv = inp.prsfc > zepprcp
    zmprcp_c = torch.where(has_cv, zmprcp_c, torch.zeros_like(zmprcp_c))
    pwl_before_cv = pwl
    pwl = pwl + zmprcp_c
    ptsfc_land = torch.where(has_cv, inp.prsfc - zmprcp_c * ztmst,
                             torch.zeros_like(inp.prsfc))

    # Sea points: throughfall = raw rainfall (no interception).
    ptsfl = torch.where(ldland, ptsfl_land, inp.prsfl)
    ptsfc = torch.where(ldland, ptsfc_land, inp.prsfc)

    # Sea points: force the reservoir to zero (label 2 / 3 sea branch).
    pwl = pwl * land

    return CanopyOutputs(
        pwl=pwl,
        pfwel1=pfwel1,
        ptsfc=ptsfc,
        ptsfl=ptsfl,
        peintti=peintti,
    )


def _tile_frac(pfrti: torch.Tensor, port_idx: int, ntile: int) -> torch.Tensor:
    """Return ``pfrti[:, port_idx]`` clamped to zero if the tile is absent.

    Mirrors the ``ntile``-guard pattern used throughout the port so a
    narrower tile slab (e.g. a 4-tile test) does not index out of range.
    """
    ngrid = pfrti.shape[0]
    if port_idx >= ntile:
        return torch.zeros((ngrid,), dtype=pfrti.dtype, device=pfrti.device)
    return pfrti[:, port_idx]
