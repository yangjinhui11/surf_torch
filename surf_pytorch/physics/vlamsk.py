"""Tiled skin layer conductivity ``lambda_skin`` (Fortran ``VLAMSK``).

Direct translation of ``surf/module/vlamsk_mod.F90``.  This routine is
called *before* ``SURFSEB`` (see ``surf/module/surfexcdriver_ctl_mod.F90``
778-789) and produces the ``PLAMSK`` array that ``SURFSEB_CTL`` consumes
verbatim (line 384 of ``surfseb_ctl_mod.F90``: ``ZLAMSK = PLAMSK``).

Tile ordering (1-based in Fortran, 0-based in this port -- see
:mod:`surf_pytorch.config`, in particular :data:`F90_TILE_OF` which is the
single authoritative port_idx -> Fortran JT map)
::

    0 ocean                | 5  vegetation high / snow-free
    1 sea-ice              | 6  snow under high vegetation
    2 wet skin             | 7  bare soil
    3 vegetation low / dry | 8  lake        (optional)
    4 snow on low veg      | 9  urban       (optional)

The per-CASE comments below cite the Fortran JT directly; the Python
slice index is always ``JT - 1`` (see ``F90_TILE_OF``).

Layout
------
All inputs/outputs are ``(Ngrid, Ntile)`` with the tile axis last so the
Penman-Monteith elimination in :mod:`surf_pytorch.physics.seb` can
vectorise over columns and tiles at once.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..config import (
    N_BASE_TILES,
    TILE_WATER,
    TILE_ICE,
    TILE_WET_SKIN,
    TILE_LOW_VEG_DRY,
    TILE_LOW_VEG_SNOW,
    TILE_HIGH_VEG_DRY,
    TILE_HIGH_VEG_SNOW,
    TILE_BARE_SOIL,
    TILE_LAKE,
    TILE_URBAN,
    SurfConfig,
)
from ..constants import ConstantBundle
from ..tables import VegTable


# --------------------------------------------------------------------------- #
# Hard-wired scalars from VLAMSK_MOD / SUSVEG_MOD / SUSSOIL_MOD
# --------------------------------------------------------------------------- #
# Large number that pins skin temperature to the sub-surface temperature
# (Tsk = Tsrf in the limit lambda_skin -> infinity).  Used for ocean tile
# (and forced sea-ice coupling, LSICOUP).
LARGE = 1.0e10

# Tiled-snow stability cap: constrains skin-T variation for melting snow.
LARGE_SN = 50.0

# Wet-skin tile uses the harmonic mean of (lambda_lowveg, lambda_highveg)
# -- 1/20 == 1/10 + 1/10 inverse, capped at 20 to match RVLAMSK defaults.
LARGE_WT = 20.0

# Snow thermal conductivity values (used by the snow tiles 4 and 6).
SNOW = 7.0
SNOW_GLACIER = 8.0          # used for very deep snow (glaciers)
SNOW_HIGH_VEG = 20.0        # snow under high vegetation default

# Slightly below freezing -- triggers snow-melt stability cap.
RTT_MINUS_EPS = 273.16 - 0.2

# Snow depth threshold for stability correction (RQSNCR): 1/15 m by
# default, 1/10 m if LESN09 (cy40+ snow).  We use the cy48r1 default.
RQSNCR_DEFAULT = 1.0 / 15.0
RQSNCR_SN09 = 1.0 / 10.0

# Volumetric heat capacity of snow (J/m^3/K) and ice density (kg/m^3).
# RHOCI = rho_snow * cp_ice = 920 * 2106 ~ 1.94e6; IFS uses 2.05e6.
RHO_CI = 2.05e6
RHO_ICE = 920.0


def vlamsk(
    ptskm1m: torch.Tensor,        # (Ngrid, Ntile) skin T at t-1
    ptsrf: torch.Tensor,          # (Ngrid, Ntile) sub-surface T
    psnm: torch.Tensor,           # (Ngrid,)      snow water eq. [kg/m^2]
    prsn: torch.Tensor,           # (Ngrid,)      snow density     [kg/m^3]
    psntice: Optional[torch.Tensor],  # (Ngrid,)  snow depth on sea ice
    ktvl: torch.Tensor,           # (Ngrid,) long, dominant low veg type
    ktvh: torch.Tensor,           # (Ngrid,) long, dominant high veg type
    veg_table: VegTable,
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    ptstep: Optional[torch.Tensor] = None,    # (Ngrid,) VDF sub-step
    lesn09: bool = False,
    leskti5: bool = False,
    leskti8: bool = False,
    lsicoup: bool = False,
    rurban_c: "torch.Tensor | float" = 20.0,    # urban skin conductivity
) -> torch.Tensor:
    """Compute tiled skin layer conductivity ``PLAMSK`` (W/m^2/K).

    Mirrors ``VLAMSK_MOD:VLAMSK`` line-for-line.  Branches are evaluated
    with ``torch.where`` so the routine is differentiable when the
    surrounding module runs under ``enable_grad`` (gradients are
    zero off the active branch, matching the Fortran piecewise
    behaviour).  Snow / sea-ice stability corrections are applied as
    post-multipliers where the Fortran uses the same expression.

    Args
    ----
    ptskm1m, ptsrf:
        Per-tile skin temperature and sub-surface temperature (K).
        Shape ``(Ngrid, Ntile)``.
    psnm, prsn:
        Grid-mean snow water equivalent and density (the Fortran indexes
        ``PSNM(:,1)``/``PRSN(:,1)`` -- we keep the grid axis only).
    psntice:
        Snow depth on sea ice (m).  Required if the sea-ice tile is
        active and ``lsicoup`` is False; ignored otherwise.
    ktvl, ktvh:
        Dominant low / high vegetation type per column (long, 1..20).
    veg_table:
        :class:`surf_pytorch.tables.VegTable` providing RVLAMSK /
        RVLAMSKS / RVTRSR.
    consts:
        :class:`ConstantBundle` (only ``rtt`` is consumed here).
    config:
        :class:`SurfConfig` -- ``n_tile`` selects how many tile columns
        to fill (extra columns are left untouched).

    Keyword args
    ------------
    ptstep:
        Per-column VDF sub-step (s).  Required for the snow stability
        correction on tile 6 (high-veg snow); may be a 0-d tensor.
    lesn09, leskti5, leskti8:
        Soil-package feature flags from ``YDSOIL%`` -- default to the
        cy48r1 offline-driver values (all False).  Toggle to reproduce
        a namelist override.
    lsicoup:
        Sea-ice coupling flag from ``YDEXC%LSICOUP``.  When True the
        sea-ice tile is forced to ``LARGE`` (skin pinned to SST).
    rurban_c:
        Urban tile skin conductivity (RURBTC).  Default 20.0 reproduces
        the cy48r1 ``SUSURB`` value; override for sensitivity studies.

    Returns
    -------
    plamsk : (Ngrid, Ntile) skin layer conductivity.
    """
    ntile = config.n_tile
    ngrid = ptskm1m.shape[0]
    device = ptskm1m.device
    dtype = ptskm1m.dtype

    plamsk = torch.zeros(ngrid, ntile, dtype=dtype, device=device)

    # Helpers to address a tile slice.
    def _tile(jt: int) -> slice:
        return (slice(None), jt)

    # -- CASE 1, 9 (Fortran): ocean (and ocean-mix in 9).  Always LARGE.
    plamsk[_tile(TILE_WATER)] = LARGE
    if ntile > TILE_LAKE:
        plamsk[_tile(TILE_LAKE)] = LARGE

    # -- CASE 2 (Fortran): sea ice.
    if ntile > TILE_ICE:
        if lsicoup:
            plamsk[_tile(TILE_ICE)] = LARGE
        else:
            # Stable branch: RVLAMSKS(12); unstable: RVLAMSK(12).
            stable_ice = ptskm1m[_tile(TILE_ICE)] > ptsrf[_tile(TILE_ICE)]
            type12 = torch.full_like(ktvl, 12)
            lamsks = veg_table.gather("rvlamsks", type12)
            lamsk = veg_table.gather("rvlamsk", type12)
            plamsk[_tile(TILE_ICE)] = torch.where(
                stable_ice, lamsks.to(dtype=dtype), lamsk.to(dtype=dtype)
            )

    # -- CASE 3 (Fortran): wet skin.  Harmonic mean of low/high veg.
    if ntile > TILE_WET_SKIN:
        plamsk[_tile(TILE_WET_SKIN)] = LARGE_WT

    # -- CASE 4 (Fortran): low vegetation, snow-free.
    if ntile > TILE_LOW_VEG_DRY:
        stable_lv = ptskm1m[_tile(TILE_LOW_VEG_DRY)] > ptsrf[_tile(TILE_LOW_VEG_DRY)]
        lamsks = veg_table.gather("rvlamsks", ktvl)
        lamsk = veg_table.gather("rvlamsk", ktvl)
        plamsk[_tile(TILE_LOW_VEG_DRY)] = torch.where(
            stable_lv, lamsks.to(dtype=dtype), lamsk.to(dtype=dtype)
        )

    # -- CASE 5 (Fortran): snow on low vegetation.
    if ntile > TILE_LOW_VEG_SNOW:
        tsk5 = ptskm1m[_tile(TILE_LOW_VEG_SNOW)]
        srf5 = ptsrf[_tile(TILE_LOW_VEG_SNOW)]
        melt_window = (tsk5 >= srf5) & (tsk5 > RTT_MINUS_EPS)
        deep_snow = psnm > 9000.0
        plamsk[_tile(TILE_LOW_VEG_SNOW)] = torch.where(
            melt_window,
            torch.full_like(tsk5, LARGE_SN),
            torch.where(
                deep_snow,
                torch.full_like(tsk5, SNOW_GLACIER),
                torch.full_like(tsk5, SNOW),
            ),
        )

    # -- CASE 6 (Fortran): high vegetation, snow-free.
    if ntile > TILE_HIGH_VEG_DRY:
        stable_hv = ptskm1m[_tile(TILE_HIGH_VEG_DRY)] > ptsrf[_tile(TILE_HIGH_VEG_DRY)]
        lamsks = veg_table.gather("rvlamsks", ktvh)
        lamsk = veg_table.gather("rvlamsk", ktvh)
        plamsk[_tile(TILE_HIGH_VEG_DRY)] = torch.where(
            stable_hv, lamsks.to(dtype=dtype), lamsk.to(dtype=dtype)
        )

    # -- CASE 7 (Fortran): snow under high vegetation.  Snow stability
    #    correction applied AFTER the SELECT CASE block (lines 219-223).
    if ntile > TILE_HIGH_VEG_SNOW:
        stable_hvs = ptskm1m[_tile(TILE_HIGH_VEG_SNOW)] > ptsrf[_tile(TILE_HIGH_VEG_SNOW)]
        lamsks = veg_table.gather("rvlamsks", ktvh)
        base = torch.where(
            stable_hvs,
            lamsks.to(dtype=dtype),
            torch.full_like(stable_hvs, SNOW_HIGH_VEG, dtype=dtype),
        )
        # Snow stability factor ZSTABEXSN (uses full PSNM/PRSN at col).
        rq = RQSNCR_SN09 if lesn09 else RQSNCR_DEFAULT
        depth = torch.clamp(psnm / prsn, min=rq)
        # rho_cp_eff = RHOCI*PRSN/RHOICE
        rho_cp = RHO_CI * prsn / RHO_ICE
        if ptstep is None:
            stab = torch.zeros_like(depth)
        else:
            ts = ptstep.broadcast_to(depth.shape) if ptstep.dim() else ptstep
            stab = ts / (depth * rho_cp)
        plamsk[_tile(TILE_HIGH_VEG_SNOW)] = base / (1.0 + stab * base)

    # -- CASE 8 (Fortran): bare soil.  Optional cy46+ refinement
    #    (LESKTI8) using soil thermal conductivity -- stubbed to the
    #    classic RVLAMSK(8) lookup.
    if ntile > TILE_BARE_SOIL:
        stable_bs = ptskm1m[_tile(TILE_BARE_SOIL)] > ptsrf[_tile(TILE_BARE_SOIL)]
        eights = torch.full_like(ktvl, 8)
        lamsks = veg_table.gather("rvlamsks", eights)
        lamsk = veg_table.gather("rvlamsk", eights)
        plamsk[_tile(TILE_BARE_SOIL)] = torch.where(
            stable_bs, lamsks.to(dtype=dtype), lamsk.to(dtype=dtype)
        )

    # -- CASE 10 (Fortran): urban.
    if ntile > TILE_URBAN:
        plamsk[_tile(TILE_URBAN)] = float(rurban_c)

    return plamsk


__all__ = [
    "vlamsk",
    "LARGE",
    "LARGE_SN",
    "LARGE_WT",
    "SNOW",
    "SNOW_GLACIER",
    "SNOW_HIGH_VEG",
    "RHO_CI",
    "RHO_ICE",
    "RQSNCR_DEFAULT",
    "RQSNCR_SN09",
]
