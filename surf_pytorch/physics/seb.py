"""Surface energy balance (Penman-Monteith skin-T elimination).

Direct translation of ``surf/module/surfseb_ctl_mod.F90`` (J. Polcher
/ P. Best / A. Beljaars closure).  This is the heart of the land-
surface scheme: it eliminates the skin temperature from the surface
energy balance using a linearisation of qsat around the t-1 skin
temperature, giving a closed-form solution for the new skin temperature
and the tile-mean turbulent fluxes.

Closure summary
---------------
The lowest-model-level dry static energy ``Sl`` and specific humidity
``Ql`` are coupled to the surface through the vertical-diffusion
tridiagonal elimination (the *upward* sweep in
:mod:`surf_pytorch.physics.diffusion`, M4).  That sweep produces a
linear relation::

    Sl = ASL*Js + BSL
    Ql = AQL*Jq + BQL

where ``Js`` and ``Jq`` are the fluxes of dry static energy and
moisture.  Each tile has its own SEB::

    -PSSRFL -PSLRFL + PSTR + PG0 + PSSH + PSLH = 0
    PSSH = Js - cp*Tsk*delta*Jq                (sensible)
    PSLH = lambda * Jq                          (latent, lambda=Lv/Ls)
    PG0  = lambda_skin * (Tsk - Tsrf)           (ground)
    PSTR = PSLRFL + dLW/dT * (Tsk - Tsk_rad)    (linearised LW)

The skin temperature is linearised in qsat through ``qsk(Tsk) ~
qsk_t1 + dqsdT_t1*(Tsk - Tsk_t1)`` so the whole SEB becomes linear in
Tsk, Sl and Ql.  Eliminating Tsk per tile yields the per-tile
coefficients ``ZD{JS,JQ,DSS}*`` (section 4 of the Fortran); tile-
averaging and a 2x2 solve for ``Sl,Ql`` closes the system (sections
5-6); per-tile fluxes and skin temperatures follow (section 7).

Sign convention (IFS surface-package, NOT micrometeorology)
-----------------------------------------------------------
The SEB closure that this routine enforces is::

    PSSRFL + PSTR + PSSH + PSLH - PG0 = 0   (per tile)

i.e. all four surface fluxes share the **positive = downward** (into
the surface) sign, and ``PG0`` (ground heat flux) is **positive =
downward** into the ground:

* ``PSSRFL``, ``PSTR`` (radiation): **positive = downward**.  ``PSTR``
  is the net thermal radiation = ``PSLRFL + dLW/dT*(Tsk - Tskrad)`` and
  ``PSLRFL`` is also positive downward, so a cooling surface (more
  emission than absorption) has ``PSLRFL < 0``.
* ``PSSH`` (sensible heat): **positive = downward** (atmosphere warms
  the surface).  Note this is the *opposite* of the micrometeorological
  sign convention; ``PSSH = PJS - cp*Tsk*delta*PJQ`` with ``PJS`` the
  surface-going dry-static-energy flux already carries the downward
  sign.
* ``PSLH`` (latent heat): **positive = downward** (condensation warms
  the surface).  Again opposite to the micrometeorological sign.
* ``PG0`` (ground heat): **positive = downward**, into the ground (a
  surface warming from above has ``PG0 > 0``).

This is the *output* convention of SURFSEB_CTL (see lines 467-470 of
``surfseb_ctl_mod.F90``); the *elimination* uses the opposite internal
signs (``-PSSRFL``, ``-PSLRFL``, ``+ZLAMSK*PTSRF`` in ZDSS4) which
match the convention above after rearrangement.  This port reproduces
the Fortran elimination verbatim and inherits the same output
convention -- everything is positive downward.

Layout
------
Tensors are ``(Ngrid, Ntile)`` with the tile axis last.  The 2x2 mean
solve reduces the tile axis; outputs ``sl`` / ``ql`` are ``(Ngrid,)``.

Tile index conventions
----------------------
This port keeps the IFS 1-based tile semantics but addresses tensors
0-based (see :mod:`surf_pytorch.config`).  Fortran JT=2 (sea ice) is
this port's ``TILE_ICE=1``; Fortran JT=5 (snow on low veg) is this
port's ``TILE_LOW_VEG_SNOW=4``.  The latent heat selector ``RLSTT``
(sublimation) is used for ice and low-veg-snow tiles -- the offsets are
applied here, not in the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import (
    TILE_ICE,
    TILE_LOW_VEG_SNOW,
    TILE_LAKE,
    SurfConfig,
)
from ..constants import ConstantBundle
from .vlamsk import vlamsk, LARGE


# --------------------------------------------------------------------------- #
# SEB hard-wired constants (surfseb_ctl_mod.F90 lines 209-216)
# --------------------------------------------------------------------------- #
# ZDELTA == RVTMP2 (moist cp correction) -- the offline driver sets it to
# 0.0 via SUPHEC, so PSSH = Js by construction.  We keep the symbolic
# form so a future cp-moist option is a one-line change.
# (consumed from consts.rvtmp2)

# Large number that pins skin T to the sub-surface T (ocean, etc.).
_LARGE_SEB = 1.0e10


@dataclass
class SebInputs:
    """All per-(grid,tile) inputs to SURFSEB_CTL.

    Grouped for clarity; the :func:`surf_seb` call unpacks these.  Each
    tensor is ``(Ngrid, Ntile)`` unless its name ends in ``_grid``
    (``(Ngrid,)``) or it is a scalar field.
    """

    # Skin reservoir at t-1 (linearised in qsat).
    sskm1m: torch.Tensor          # dry static energy of skin    [J/kg]
    tskm1m: torch.Tensor          # skin temperature at t-1      [K]
    qskm1m: torch.Tensor          # saturation q at tskm1m       [kg/kg]
    dqsdT: torch.Tensor           # dqsat/dT at tskm1m           [kg/kg/K]

    # Turbulent exchange: rho * Cd * |U| for heat and moisture.
    rhochu: torch.Tensor          # rho*Ch*|U|                   [kg/m^2/s]
    rhocqu: torch.Tensor          # rho*Cq*|U|                   [kg/m^2/s]

    # Moisture-flux closure coefficients (alpha_l * ql + alpha_s * qs).
    alphal: torch.Tensor          # multiplier of ql             [-]
    alphas: torch.Tensor          # multiplier of qs             [-]

    # Radiation and ground coupling.
    ssrfl: torch.Tensor           # net SW at the surface        [W/m^2]
    slrfl_grid: torch.Tensor      # net LW at the surface        [W/m^2]
    tskrad_grid: torch.Tensor     # mean skin T at rad time      [K]
    emis_grid: torch.Tensor       # surface emissivity           [-]

    # Lower boundary: sub-surface temperature and skin conductivity.
    tsrf: torch.Tensor            # surface T below skin         [K]
    lamsk: torch.Tensor           # skin layer conductivity      [W/m^2/K]

    # Tile fractions and dominant veg types (for lambda Lv/Ls selector).
    frti: torch.Tensor            # tile fractions               [-]
    ktvl_grid: torch.Tensor       # dominant low veg type (long)
    ktvh_grid: torch.Tensor       # dominant high veg type (long)

    # Snow / sea-ice / lake descriptors (used by lambda selector and
    # by the lake Lv/Ls split).
    snowm_grid: Optional[torch.Tensor] = None   # snow water eq. [kg/m^2]
    rsn_grid: Optional[torch.Tensor] = None     # snow density   [kg/m^3]
    thkice_grid: Optional[torch.Tensor] = None  # sea-ice thickness [m]
    sntice_grid: Optional[torch.Tensor] = None  # snow depth on sea ice
    hlice_grid: Optional[torch.Tensor] = None   # lake ice thickness [m]

    # Coupling coefficients from the upward diffusion sweep.
    asl_grid: torch.Tensor = None     # Sl = asl*Js + bsl
    bsl_grid: torch.Tensor = None
    aql_grid: torch.Tensor = None     # Ql = aql*Jq + bql
    bql_grid: torch.Tensor = None

    # Time step.
    ptmst: torch.Tensor = None        # scalar / (Ngrid,)

    # Feature flags.
    lelwdd: bool = True               # LW derivative form
    lelwtl: bool = False              # recompute tiled LW around mean Tsk
    lnemolimthk: bool = False         # NEMO sea-ice thickness source


@dataclass
class SebOutputs:
    """All per-(grid,tile) outputs of SURFSEB_CTL."""

    pjs: torch.Tensor           # flux of dry static energy   [W/m^2]
    pjq: torch.Tensor           # moisture flux               [kg/m^2/s]
    pssk: torch.Tensor          # new dry static energy       [J/kg]
    ptsk: torch.Tensor          # new skin temperature        [K]
    pssh: torch.Tensor          # sensible heat flux          [W/m^2]
    pslh: torch.Tensor          # latent heat flux            [W/m^2]
    pstr: torch.Tensor          # net thermal radiation       [W/m^2]
    pg0: torch.Tensor           # ground heat flux            [W/m^2]

    psl: torch.Tensor           # new lowest-level dry static energy [J/kg]
    pql: torch.Tensor           # new lowest-level specific humidity [kg/kg]


def surf_seb(
    inp: SebInputs,
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    differentiable: Optional[bool] = None,
) -> SebOutputs:
    """Run the Penman-Monteith surface energy balance.

    Args
    ----
    inp:
        :class:`SebInputs` -- packed per-(grid,tile) and per-grid inputs.
    consts:
        :class:`ConstantBundle` -- RCPD, RLSTT, RLVTT, RSIGMA, RTT, RVTMP2.
    config:
        :class:`SurfConfig` -- ``n_tile`` selects which tile columns are
        active (latent heat selector switches JT=ice/low-veg-snow to
        sublimation, JT=9/lake to a Lv/Ls mix).
    differentiable:
        Override ``config.differentiable`` for this call.  Default
        respects the config flag.

    Returns
    -------
    :class:`SebOutputs` -- per-tile fluxes + skin temperature, plus the
    grid-mean lowest-level dry static energy and specific humidity.
    """
    diff = config.differentiable if differentiable is None else differentiable

    def _core() -> SebOutputs:
        return _seb_core(inp, consts, config)

    if not diff:
        with torch.no_grad():
            return _core()
    return _core()


def _seb_core(inp: SebInputs, consts: ConstantBundle, config: SurfConfig) -> SebOutputs:
    """Internal: assumes the no_grad / enable_grad decision is already made."""
    ngrid, ntile = inp.sskm1m.shape

    # Promote per-grid inputs to per-(grid,tile) for vectorised maths.
    def _expand_grid(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(-1).expand(ngrid, ntile)

    slrfl = _expand_grid(inp.slrfl_grid)
    tskrad = _expand_grid(inp.tskrad_grid)
    emis = _expand_grid(inp.emis_grid)
    asl = _expand_grid(inp.asl_grid)
    bsl = _expand_grid(inp.bsl_grid)
    aql = _expand_grid(inp.aql_grid)
    bql = _expand_grid(inp.bql_grid)

    # ------------------------------------------------------------------ #
    # Section 1 -- lake ice mask (only matters if lake tile is active).
    # ------------------------------------------------------------------ #
    if ntile > TILE_LAKE and inp.hlice_grid is not None:
        rh_ice_min = 1.0e-9          # SUSFLAKE line 128
        lake_ice = (inp.hlice_grid > rh_ice_min).to(inp.sskm1m.dtype)
        lake_water = 1.0 - lake_ice
    else:
        # Sentinel: no lake tile -> all-water (the lake Lv/Ls branch is
        # never reached because ntile <= TILE_LAKE).
        lake_water = torch.ones(ngrid, dtype=inp.sskm1m.dtype,
                                device=inp.sskm1m.device)
        lake_ice = torch.zeros_like(lake_water)

    # ------------------------------------------------------------------ #
    # Section 2 -- dLW/dT (tile-independent).
    # ------------------------------------------------------------------ #
    if inp.lelwdd:
        # Linearise LW around the radiation time-level skin T.
        dlwdt_grid = -4.0 * inp.emis_grid * consts.rsigma * inp.tskrad_grid ** 3
    else:
        # Alternative: derive dLW/dT from the supplied net LW.
        dlwdt_grid = 4.0 * (inp.slrfl_grid / inp.tskrad_grid)
    dlwdt = _expand_grid(dlwdt_grid)

    # ------------------------------------------------------------------ #
    # Section 3 -- Js / Jq coefficients in (Sl, Ql, Ssk) (Fortran CJ*).
    # ------------------------------------------------------------------ #
    # ZICPTM1 = TSKM1M / SSKM1M  (inverse of cp*T mapping).  Avoided
    # dividing by SSKM1M where it is zero -- the ocean tile sets
    # SSKM1M = cp*Tsrf so this is well defined there.
    icptm1 = inp.tskm1m / inp.sskm1m
    cjs1 = inp.rhochu
    cjs3 = -inp.rhochu
    cjq2 = inp.rhocqu * inp.alphal
    cjq3 = -inp.rhocqu * inp.alphas * inp.dqsdT * icptm1
    cjq4 = -inp.rhocqu * inp.alphas * (
        inp.qskm1m - inp.dqsdT * inp.tskm1m
    )

    # ------------------------------------------------------------------ #
    # Section 4 -- eliminate Ssk via SEB -> coefficients in (Sl, Ql).
    # ------------------------------------------------------------------ #
    # Latent heat per tile: sublimation (RLSTT) for sea ice / low-veg
    # snow, vaporisation (RLVTT) elsewhere; lake tile is a Lv/Ls mix
    # weighted by lake water/ice mask.  Build per-tile lambda.
    lam = _tile_latent_heat(ngrid, ntile, inp.sskm1m.dtype,
                            inp.sskm1m.device, consts,
                            lake_water, lake_ice)

    # Solar transmission through vegetation per tile (ZFRSR).  Tiles
    # 2/3 (wet skin, low-veg dry) reduce SW by RVTRSR(1)=0.05 and tiles
    # 5/6 (high-veg dry, high-veg snow) by RVTRSR(3)=0.03; the rest see
    # the full net SW.  See :func:`_tile_solar_transmission` for the
    # tile-index mapping (0-based, mirroring surfseb_ctl_mod.F90 CASEs).
    frsr = _tile_solar_transmission(ngrid, ntile, inp.sskm1m.dtype,
                                    inp.sskm1m.device)

    delta = consts.rvtmp2
    coef1 = lam - consts.cpd * inp.tskm1m * delta

    # ZIZZ = 1/ZZ where ZZ = (dlwdt - lamsk)*icptm1 + cjs3 + coef1*cjq3
    zz = (dlwdt - inp.lamsk) * icptm1 + cjs3 + coef1 * cjq3
    zizz = 1.0 / zz

    dss1 = -cjs1 * zizz
    dss2 = -coef1 * cjq2 * zizz
    dss4 = (
        -inp.ssrfl * frsr - slrfl + dlwdt * tskrad
        - coef1 * cjq4 - inp.lamsk * inp.tsrf
    ) * zizz

    djs1 = cjs1 + cjs3 * dss1
    djs2 = cjs3 * dss2
    djs4 = cjs3 * dss4

    djq1 = cjq3 * dss1
    djq2 = cjq2 + cjq3 * dss2
    djq4 = cjq3 * dss4 + cjq4

    # ------------------------------------------------------------------ #
    # Section 5 -- tile-mean coefficients.
    # ------------------------------------------------------------------ #
    frti = inp.frti
    ejs1 = (frti * djs1).sum(dim=-1)
    ejs2 = (frti * djs2).sum(dim=-1)
    ejs4 = (frti * djs4).sum(dim=-1)
    ejq1 = (frti * djq1).sum(dim=-1)
    ejq2 = (frti * djq2).sum(dim=-1)
    ejq4 = (frti * djq4).sum(dim=-1)

    # ------------------------------------------------------------------ #
    # Section 6 -- 2x2 solve for the grid-mean (Sl, Ql).
    # ------------------------------------------------------------------ #
    zz1 = 1.0 - asl[..., 0] * ejs1
    zz2 = aql[..., 0] * ejs2
    zz3 = bsl[..., 0] * ejs1 + bql[..., 0] * ejs2 + ejs4
    zy1 = 1.0 - aql[..., 0] * ejq2
    zy2 = asl[..., 0] * ejq1
    zy3 = bsl[..., 0] * ejq1 + bql[..., 0] * ejq2 + ejq4

    zz_det = zz1 * zy1 - zz2 * zy2
    zizz_det = 1.0 / zz_det
    jq = (zz3 * zy2 + zz1 * zy3) * zizz_det
    js = (zz2 * zy3 + zz3 * zy1) * zizz_det

    sl = asl[..., 0] * js + bsl[..., 0]
    ql = aql[..., 0] * jq + bql[..., 0]

    # ------------------------------------------------------------------ #
    # Section 7 -- per-tile fluxes and skin temperature.
    # ------------------------------------------------------------------ #
    sl_t = sl.unsqueeze(-1)
    ql_t = ql.unsqueeze(-1)

    pssk = dss1 * sl_t + dss2 * ql_t + dss4
    pjs = djs1 * sl_t + djs2 * ql_t + djs4
    pjq = djq1 * sl_t + djq2 * ql_t + djq4
    ptsk = pssk * icptm1

    pssh = pjs - consts.cpd * inp.tskm1m * delta * pjq
    pslh = lam * pjq
    pstr = slrfl + dlwdt * (ptsk - tskrad)
    pg0 = inp.lamsk * (ptsk - inp.tsrf)

    # ------------------------------------------------------------------ #
    # Section 8 -- optional recompute of tiled LW around the gridbox
    # mean skin temperature (LELWTL, energy-conserving correction).
    # ------------------------------------------------------------------ #
    if inp.lelwtl:
        tsk_mean = (frti * ptsk).sum(dim=-1)
        dlwdt_star = -4.0 * emis[..., 0] * consts.rsigma * tsk_mean ** 3
        pstr = inp.slrfl_grid.unsqueeze(-1) + dlwdt_star.unsqueeze(-1) * (
            ptsk - tsk_mean.unsqueeze(-1)
        )

    return SebOutputs(
        pjs=pjs, pjq=pjq, pssk=pssk, ptsk=ptsk,
        pssh=pssh, pslh=pslh, pstr=pstr, pg0=pg0,
        psl=sl, pql=ql,
    )


# --------------------------------------------------------------------------- #
# Per-tile latent heat selector
# --------------------------------------------------------------------------- #
def _tile_latent_heat(
    ngrid: int,
    ntile: int,
    dtype: torch.dtype,
    device,
    consts: ConstantBundle,
    lake_water: torch.Tensor,
    lake_ice: torch.Tensor,
) -> torch.Tensor:
    """Per-tile latent heat ``lambda`` (J/kg).

    Matches surfseb_ctl_mod.F90 lines 271-275 and 385-387:
        JT in {ice, low-veg-snow}  -> RLSTT (sublimation)
        JT == lake                 -> RLVTT*water + RLSTT*ice
        else                       -> RLVTT (vaporisation)

    Returns ``(Ngrid, Ntile)`` tensor.  Tile indices are 0-based
    (``TILE_ICE=1``, ``TILE_LOW_VEG_SNOW=4``, ``TILE_LAKE=8``).
    """
    lam = torch.full((ngrid, ntile), consts.rlvtt,
                     dtype=dtype, device=device)
    if ntile > TILE_ICE:
        lam[:, TILE_ICE] = consts.rlstt
    if ntile > TILE_LOW_VEG_SNOW:
        lam[:, TILE_LOW_VEG_SNOW] = consts.rlstt
    if ntile > TILE_LAKE:
        lam[:, TILE_LAKE] = (
            consts.rlvtt * lake_water
            + consts.rlstt * lake_ice
        )
    return lam


def _tile_solar_transmission(
    ngrid: int,
    ntile: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Per-tile factor ``1 - RVTRSR`` multiplying net SW in the SEB.

    Mirrors the ``ZFRSR`` assignment in ``surfseb_ctl_mod.F90`` SELECT
    CASE block.  Fortran uses 1-based ``JT``; this port is 0-based, so
    the mapping goes through :data:`surf_pytorch.config.F90_TILE_OF`
    (port_idx -> Fortran JT = port_idx + 1).  Tiles whose ``FRSR != 1``:

        port TILE_WET_SKIN     (=2, F90 JT=3) -> 1 - RVTRSR(1) = 0.95
        port TILE_LOW_VEG_DRY  (=3, F90 JT=4) -> 1 - RVTRSR(1) = 0.95
        port TILE_HIGH_VEG_DRY (=5, F90 JT=6) -> 1 - RVTRSR(3) = 0.97
        port TILE_HIGH_VEG_SNOW(=6, F90 JT=7) -> 1 - RVTRSR(3) = 0.97
        all other tiles                          -> 1.0

    The values are *veg-type-independent* constants because the dominant
    veg-type lookup already happened upstream and SURFSEB_CTL uses the
    fixed index 1 or 3 uniformly (RVTRSR(1)=0.05, RVTRSR(3)=0.03 from
    SUSVEG).

    (A previous version of this docstring said "tile 3 (wet skin)" while
    assigning ``frsr[:, 2]`` -- the code was right, the prose mixed the
    port and Fortran indices.  Both are now stated explicitly.)
    """
    from ..config import (
        TILE_WET_SKIN, TILE_LOW_VEG_DRY, TILE_HIGH_VEG_DRY, TILE_HIGH_VEG_SNOW,
    )
    frsr = torch.ones((ngrid, ntile), dtype=dtype, device=device)
    if ntile > TILE_WET_SKIN:           # F90 JT=3, RVTRSR(1)=0.05 -> 0.95
        frsr[:, TILE_WET_SKIN] = 0.95
    if ntile > TILE_LOW_VEG_DRY:        # F90 JT=4, RVTRSR(1)=0.05 -> 0.95
        frsr[:, TILE_LOW_VEG_DRY] = 0.95
    if ntile > TILE_HIGH_VEG_DRY:       # F90 JT=6, RVTRSR(3)=0.03 -> 0.97
        frsr[:, TILE_HIGH_VEG_DRY] = 0.97
    if ntile > TILE_HIGH_VEG_SNOW:      # F90 JT=7, RVTRSR(3)=0.03 -> 0.97
        frsr[:, TILE_HIGH_VEG_SNOW] = 0.97
    return frsr


__all__ = [
    "SebInputs",
    "SebOutputs",
    "surf_seb",
]
