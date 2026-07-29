"""Implicit vertical diffusion -- offline single-column SURFSEB wrapper.

This module is the Python port of ``surf/offline/driver/vdfdifh1s.F90``,
the *offline-driver* version of the implicit vertical-diffusion routine
that closes the surface energy balance.

In the full IFS, ``VDFDIFH`` runs a Thomas-algorithm tridiagonal solve
over the model levels and feeds the upward-sweep coefficients
``(asl, bsl, aql, bql)`` into ``SURFSEB``.  In the offline single-column
driver (``vdfdifh1s.F90``) the entire tridiag block is commented out
with the ``!1s `` prefix and replaced by a Dirichlet boundary
condition (section 1.0 - 1.5 OFFLINE SETUP)::

    asl = 0, aql = 0
    bsl = PSLM1(:,KLEV) + PTMST*PSLGE(:,KLEV)
    bql = PQTM1(:,KLEV) + PTMST*PQTE(:,KLEV)

so this port simply assembles the SURFSEB inputs from the lowest-model-
level forcing and runs ``SURFSEB`` (call #3, the canonical call).  An
optional conditional snow-melt repeat (call #4, gated by the ``LESNML``
flag) handles the explicit snow-melt-under-low-veg branch.  The
optional unstressed-evap diagnostic calls (#1 water, #2 fully
unstressed) are not transcribed here -- they are diagnostic-only and
require extra ``PCPTSTIU/PCAIRTIU/PCSATTIU`` inputs from upstream
VSURF; M4-rem-c will add a ``pjqu`` output when the reference kernel
exercises them.

Steps reproduced (vdfdifh1s.F90 section numbers)
------------------------------------------------
1.0-1.5  Standalone forcing (offline setup)::

     asl = 0,  aql = 0
     bsl = PSLM1(:,KLEV) + PTMST*PSLGE(:,KLEV)
     bql = PQTM1(:,KLEV) + PTMST*PQTE(:,KLEV)

1.6  Per-tile sub-surface temperature ``PTSRF`` (lines 372-388):

       water=PSST, ice=PTICE, wet_skin/low-veg-dry/high-veg-dry/
       bare=PTSM1M, low-veg-snow/high-veg-snow=PTSNOW, lake=PTLICE
       (if ice thicker than 1e-9 m) else PTLWML.

1.6b Exchange-coefficient inversion (lines 409-415)::

       rhoChU = PCFHTI / (RG * RVDIFTS * PTMST)
       rhoCqU = PCFQTI / (RG * RVDIFTS * PTMST)

     where PCFHTI / PCFQTI carry the upstream ``RG*PTSPHY*RVDIFTS``
     rescaling that ``vdfmain1s`` applies (lines 745-746) before
     calling VDFDIFH1S.  In the offline single-column driver
     ``PTMST == PTSPHY`` so the inversion exactly recovers
     ``rho*Ch*|U|`` -- the quantity :mod:`surf_pytorch.physics.seb`
     consumes.

1.7  SURFSEB canonical call (#3) -> per-tile (PJS, PJQ, PSSH, PSLH,
     PSTR, PG0, PTSK) and grid-mean (PSL, PQL).

1.7a Snow-under-forest evaporation (lines 557-558), applied to Fortran
     tile 7 = :data:`surf_pytorch.config.TILE_HIGH_VEG_SNOW`::

       PJQ[:,TILE_HIGH_VEG_SNOW] += PEVAPSNW
       PSLH[:,TILE_HIGH_VEG_SNOW] += PEVAPSNW*RLSTT

1.7c Skin-T extrapolation to t+1 (lines 585-590)::

       PTSKTIP1 = ZTPFAC2*PTSK + ZTPFAC3*PTSKTI
                = (1/RVDIFTS)*PTSK + (1-1/RVDIFTS)*PTSKTI

LESNML snow-melt repeat (call #4, lines 518-547) -- optional, gated by
``DiffusionInputs.lesnml``.  When any column has ``PFRTI(:,5) >= 1e-6``
*and* the call-#3 skin temperature on the low-veg-snow tile is at the
freezing point, the routine pins that tile's ``TSRF`` to ``RTT`` and
``LAMSK`` to 100 W/m^2/K (mimicking a melting snow surface) and re-runs
SURFSEB.  This is a no-op when ``lesnml`` is False.

Sign convention
---------------
Outputs inherit the SURFSEB_CTL convention documented in
:mod:`surf_pytorch.physics.seb`: all four surface fluxes and the ground
heat flux are **positive downward**.  See that module's docstring for
the full closure statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import (
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
from .seb import SebInputs, SebOutputs, surf_seb


# --------------------------------------------------------------------------- #
# VDFDIFH1S hard-wired constants (yoevdf / SUEVDF)
# --------------------------------------------------------------------------- #
# Over-implicit time weight RVDIFTS.  Used both in the exchange-coefficient
# inversion (ZCOEF1 = 1/(RG*RVDIFTS*PTMST)) and in the skin-T extrapolation
# (ZTPFAC2 = 1/RVDIFTS, ZTPFAC3 = 1 - ZTPFAC2).
#
# NOTE: the full IFS uses the default RVDIFTS=1.5 (set in arpifs/phys_ec/
# suvdf.F90), but the *offline single-column driver* overrides it to 1.0 in
# surf/offline/driver/suvdf.F90:57.  Since this port targets the offline
# driver (which we validate against), we pin RVDIFTS=1.0 here.  With this
# value the extrapolation collapses to PTSKTIP1 = PTSK (ZTPFAC2=1, ZTPFAC3=0),
# which matches the driver's step-0 skin-T dump exactly.  Exposed as a module
# constant so tests can pin it.
RVDIFTS = 1.0

# Snow-melt repeat (LESNML branch) thresholds, mirrors vdfdifh1s lines 523-528.
_LESNML_FRAC_MIN = 1.0e-6     # PFRTI threshold to consider tile active
_LESNML_TSK_EPS = 1.0e-6      # TSK - RTT threshold (snow at melting point)
_LESNML_LAMSK_MELT = 100.0    # pinned lambda_skin for melting snow [W/m^2/K]

# Lake ice thickness threshold (vdfdifh1s line 382, SUSFLAKE H_ICE_MIN).
_LAKE_HICE_MIN = 1.0e-9


@dataclass
class DiffusionInputs:
    """All inputs to VDFDIFH1S (offline single-column driver).

    Each tensor is ``(Ngrid, Ntile)`` unless noted otherwise (per-grid
    fields are ``(Ngrid,)``).  The naming mirrors the Fortran dummy
    list of ``vdfdifh1s.F90`` so a side-by-side audit is easy.

    Lowest-model-level forcing
    --------------------------
    ``pslm1_lev`` / ``pqtm1_lev`` are the lowest-level dry static energy
    and specific humidity at t-1; ``pslge_lev`` / ``pqte_lev`` are
    their tendencies.  These produce the Dirichlet boundary condition
    (``bsl``, ``bql``) of the offline driver.

    Rescaled exchange coefficients
    ------------------------------
    ``pcfhti`` / ``pcfqti`` carry the upstream ``RG*PTSPHY*RVDIFTS``
    rescaling applied by ``vdfmain1s`` (lines 745-746); the wrapper
    inverts it via ``ZCOEF1 = 1/(RG*RVDIFTS*PTMST)``.  When called from
    :class:`surf_pytorch.LandSurface` the model multiplies the raw
    ``rho*Ch*|U|`` (from :func:`surf_pytorch.physics.exchange.vexcs`)
    by ``RG*PTSPHY*RVDIFTS`` before populating these fields, mirroring
    the upstream contract.
    """

    # ---- Lowest-model-level forcing at KLEV (per grid) ----
    pslm1_lev: torch.Tensor       # PSLM1(:,KLEV)  dry static energy  [J/kg]
    pqtm1_lev: torch.Tensor       # PQTM1(:,KLEV)  specific humidity  [kg/kg]
    pslge_lev: torch.Tensor       # PSLGE(:,KLEV)  dry-static-energy tendency  [J/kg/s]
    pqte_lev: torch.Tensor        # PQTE(:,KLEV)   humidity tendency  [kg/kg/s]

    # ---- Rescaled turbulent exchange coefficients (per tile) ----
    pcfhti: torch.Tensor          # PCFHTI  heat exch coeff (rescaled)
    pcfqti: torch.Tensor          # PCFQTI  moist exch coeff (rescaled)

    # ---- Lower-boundary temperatures (per grid) for the PTSRF build ----
    psst: torch.Tensor            # PSST  open-sea surface temperature  [K]
    ptice: torch.Tensor           # PTICE sea-ice skin temperature      [K]
    ptsm1m: torch.Tensor          # PTSM1M top soil-layer temperature   [K]
    ptsnow: torch.Tensor          # PTSNOW snow skin temperature        [K]

    # ---- Skin reservoir at t-1 (per tile, from VSURF) ----
    sskm1m: torch.Tensor          # PCPTSTI  cp*Tsk              [J/kg]
    tskm1m: torch.Tensor          # PTSKTI   skin T at t-1       [K]
    qskm1m: torch.Tensor          # PQSTI    qsat(Tsk)           [kg/kg]
    dqsdT: torch.Tensor           # PDQSTI   dqsat/dT at Tsk     [kg/kg/K]

    # ---- Moisture-flux closure coefficients (per tile, from VEVAP) ----
    alphal: torch.Tensor          # PCAIR  multiplier of ql       [-]
    alphas: torch.Tensor          # PCSAT  multiplier of qs       [-]

    # ---- Radiation / emissivity ----
    pssrflti: torch.Tensor        # PSSRFLTI (Ngrid, Ntile) per-tile net SW  [W/m^2]
    pslrfl: torch.Tensor          # PSLRFL   (Ngrid,) net LW               [W/m^2]
    ptskrad: torch.Tensor         # PTSKRAD  (Ngrid,) skin T at rad time   [K]
    pemis: torch.Tensor           # PEMIS    (Ngrid,) surface emissivity   [-]

    # ---- Skin conductivity from VLAMSK (per tile) ----
    plamsk: torch.Tensor          # PLAMSK (Ngrid, Ntile)  [W/m^2/K]

    # ---- Tile fractions and dominant veg types ----
    pfrti: torch.Tensor           # PFRTI (Ngrid, Ntile)
    ktvl: torch.Tensor            # KTVL (Ngrid,) long
    ktvh: torch.Tensor            # KTVH (Ngrid,) long

    # ---- Timestep ----
    ptmst: torch.Tensor           # PTMST scalar  [s]

    # ---- Snow-under-forest evaporation (per grid) ----
    pevapsnw: Optional[torch.Tensor] = None    # PEVAPSNW  [kg/m^2/s]

    # ---- Optional lake tile skin (only used when n_tile > TILE_LAKE) ----
    phlice: Optional[torch.Tensor] = None      # PHLICE (Ngrid,) lake ice thickness [m]
    ptlice: Optional[torch.Tensor] = None      # PTLICE (Ngrid,) lake ice skin T     [K]
    ptlwml: Optional[torch.Tensor] = None      # PTLWML (Ngrid,) lake mixed-layer T  [K]

    # ---- LESNML snow-melt repeat flag (call #4) ----
    lesnml: bool = False

    # ---- Feature flags inherited from SebInputs ----
    lelwdd: bool = True
    lelwtl: bool = False


@dataclass
class DiffusionOutputs:
    """All outputs of VDFDIFH1S (offline single-column driver).

    Inherits the SURFSEB_CTL sign convention -- all four surface fluxes
    and the ground heat flux are positive downward.  See
    :mod:`surf_pytorch.physics.seb` for the closure statement.
    """

    # Canonical SURFSEB outputs (call #3, after snow-under-forest evap).
    pjq: torch.Tensor             # PJQ (Ngrid, Ntile) moisture flux  [kg/m^2/s]
    pssh: torch.Tensor            # PSSH (Ngrid, Ntile) sensible heat [W/m^2]
    pslh: torch.Tensor            # PSLH (Ngrid, Ntile) latent heat   [W/m^2]
    pstr: torch.Tensor            # PSTR (Ngrid, Ntile) net thermal radiation [W/m^2]
    pg0: torch.Tensor             # PG0  (Ngrid, Ntile) ground heat flux     [W/m^2]
    ptsk: torch.Tensor            # PTSK (Ngrid, Ntile) skin T at t+1/2      [K]

    # Skin T extrapolated to t+1 (ZTPFAC2*PTSK + ZTPFAC3*PTSKTI).
    ptsktip1: torch.Tensor        # PTSKTIP1 (Ngrid, Ntile) skin T at t+1    [K]

    # Grid-mean lowest-level update from SURFSEB.
    psl: torch.Tensor             # PSL (Ngrid,) dry static energy at t+1/2  [J/kg]
    pql: torch.Tensor             # PQL (Ngrid,) specific humidity at t+1/2  [kg/kg]


def vdfdifh1s(
    inp: DiffusionInputs,
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    differentiable: Optional[bool] = None,
) -> DiffusionOutputs:
    """Run the offline single-column VDFDIFH1S wrapper around SURFSEB.

    Args
    ----
    inp:
        :class:`DiffusionInputs` -- packed per-(grid,tile) and per-grid
        inputs.  See that dataclass for the rescaled-exchange-coeff
        contract.
    consts:
        :class:`ConstantBundle` -- provides ``RG``, ``RLSTT``, ``RTT``
        and the constants consumed by :func:`surf_seb`.
    config:
        :class:`SurfConfig` -- ``n_tile`` selects which tile columns
        are active.
    differentiable:
        Override ``config.differentiable`` for this call.

    Returns
    -------
    :class:`DiffusionOutputs` -- per-tile fluxes, extrapolated skin
    temperature, and grid-mean lowest-level state.
    """
    diff = config.differentiable if differentiable is None else differentiable

    def _core() -> DiffusionOutputs:
        return _vdfdifh1s_core(inp, consts, config)

    if not diff:
        with torch.no_grad():
            return _core()
    return _core()


def _vdfdifh1s_core(
    inp: DiffusionInputs,
    consts: ConstantBundle,
    config: SurfConfig,
) -> DiffusionOutputs:
    """Internal: assumes the no_grad / enable_grad decision is already made."""
    ngrid = inp.pslm1_lev.shape[0]
    ntile = config.n_tile
    dtype = inp.pslm1_lev.dtype
    device = inp.pslm1_lev.device

    rg = consts.g
    rlstt = consts.rlstt
    rtt = consts.rtt
    ptmst = inp.ptmst  # 0-d tensor

    # ------------------------------------------------------------------ #
    # Step 1.0-1.5: standalone Dirichlet forcing (offline setup).
    # ------------------------------------------------------------------ #
    asl_grid = torch.zeros((ngrid,), dtype=dtype, device=device)
    bsl_grid = inp.pslm1_lev + ptmst * inp.pslge_lev
    aql_grid = torch.zeros((ngrid,), dtype=dtype, device=device)
    bql_grid = inp.pqtm1_lev + ptmst * inp.pqte_lev

    # ------------------------------------------------------------------ #
    # Step 1.6: per-tile sub-surface temperature PTSRF.
    # ------------------------------------------------------------------ #
    tsrf = _build_tsrf(inp, ngrid, ntile, dtype, device)

    # ------------------------------------------------------------------ #
    # Step 1.6b: exchange-coefficient inversion (ZCOEF1).
    # ------------------------------------------------------------------ #
    # PCFHTI / PCFQTI carry the upstream RG*PTSPHY*RVDIFTS rescaling;
    # the SEB consumes the inverted "rho*Ch*|U|" form.
    ptmst_tile = ptmst.unsqueeze(-1) if ptmst.ndim == 1 else ptmst
    zcoef1 = 1.0 / (rg * RVDIFTS * ptmst_tile)
    rhochu = inp.pcfhti * zcoef1
    rhocqu = inp.pcfqti * zcoef1

    # ------------------------------------------------------------------ #
    # Step 1.7: SURFSEB canonical call (#3).
    # ------------------------------------------------------------------ #
    seb_inp = SebInputs(
        sskm1m=inp.sskm1m, tskm1m=inp.tskm1m,
        qskm1m=inp.qskm1m, dqsdT=inp.dqsdT,
        rhochu=rhochu, rhocqu=rhocqu,
        alphal=inp.alphal, alphas=inp.alphas,
        ssrfl=inp.pssrflti,
        slrfl_grid=inp.pslrfl, tskrad_grid=inp.ptskrad, emis_grid=inp.pemis,
        tsrf=tsrf, lamsk=inp.plamsk,
        frti=inp.pfrti, ktvl_grid=inp.ktvl, ktvh_grid=inp.ktvh,
        asl_grid=asl_grid, bsl_grid=bsl_grid,
        aql_grid=aql_grid, bql_grid=bql_grid,
        ptmst=inp.ptmst,
        lelwdd=inp.lelwdd, lelwtl=inp.lelwtl,
    )
    out = surf_seb(seb_inp, consts, config)

    ptsk = out.ptsk           # call-#3 skin T (may be overwritten by #4)
    pstr = out.pstr
    pg0 = out.pg0
    pssh = out.pssh
    pjq = out.pjq.clone() if inp.pevapsnw is not None else out.pjq
    pslh = out.pslh.clone() if inp.pevapsnw is not None else out.pslh

    # ------------------------------------------------------------------ #
    # LESNML snow-melt repeat (call #4): if any column has a melting
    # low-veg-snow tile (Fortran JT=5 = port TILE_LOW_VEG_SNOW=4),
    # re-run SURFSEB with that tile pinned to RTS=RTT, LAMSK=100.
    # ------------------------------------------------------------------ #
    if inp.lesnml and ntile > TILE_LOW_VEG_SNOW:
        ptsk, pstr, pg0, pssh, pjq, pslh = _lesnml_repeat(
            inp, seb_inp, out, ptsk, pstr, pg0, pssh, pjq, pslh,
            consts, config, rtt, dtype, device,
        )

    # ------------------------------------------------------------------ #
    # Step 1.7a: snow-under-forest evaporation.
    # Fortran tile 7 == port TILE_HIGH_VEG_SNOW (6).
    # ------------------------------------------------------------------ #
    if inp.pevapsnw is not None and ntile > TILE_HIGH_VEG_SNOW:
        pjq[:, TILE_HIGH_VEG_SNOW] = pjq[:, TILE_HIGH_VEG_SNOW] + inp.pevapsnw
        pslh[:, TILE_HIGH_VEG_SNOW] = pslh[:, TILE_HIGH_VEG_SNOW] + inp.pevapsnw * rlstt

    # ------------------------------------------------------------------ #
    # Step 1.7c: skin T extrapolation to t+1.
    # ------------------------------------------------------------------ #
    ztpfac2 = 1.0 / RVDIFTS
    ztpfac3 = 1.0 - ztpfac2
    ptsktip1 = ztpfac2 * ptsk + ztpfac3 * inp.tskm1m

    return DiffusionOutputs(
        pjq=pjq, pssh=pssh, pslh=pslh, pstr=pstr, pg0=pg0,
        ptsk=ptsk, ptsktip1=ptsktip1,
        psl=out.psl, pql=out.pql,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_tsrf(
    inp: DiffusionInputs,
    ngrid: int,
    ntile: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Per-tile sub-surface temperature PTSRF (vdfdifh1s lines 372-388).

    Tile mapping (Fortran JT -> port idx):
        1 water      -> PSST
        2 ice        -> PTICE
        3 wet skin   -> PTSM1M
        4 low-veg dry-> PTSM1M
        5 low-veg snow -> PTSNOW
        6 high-veg dry -> PTSM1M
        7 high-veg snow -> PTSNOW
        8 bare soil  -> PTSM1M
        9 lake       -> PTLICE if ice thicker than 1e-9 else PTLWML
       10 urban      -> PTSM1M (not yet wired -- needs LEURBAN flag)
    """
    tsrf = torch.empty((ngrid, ntile), dtype=dtype, device=device)
    tsrf[:, TILE_WATER] = inp.psst
    if ntile > TILE_ICE:
        tsrf[:, TILE_ICE] = inp.ptice
    if ntile > TILE_WET_SKIN:
        tsrf[:, TILE_WET_SKIN] = inp.ptsm1m
    if ntile > TILE_LOW_VEG_DRY:
        tsrf[:, TILE_LOW_VEG_DRY] = inp.ptsm1m
    if ntile > TILE_LOW_VEG_SNOW:
        tsrf[:, TILE_LOW_VEG_SNOW] = inp.ptsnow
    if ntile > TILE_HIGH_VEG_DRY:
        tsrf[:, TILE_HIGH_VEG_DRY] = inp.ptsm1m
    if ntile > TILE_HIGH_VEG_SNOW:
        tsrf[:, TILE_HIGH_VEG_SNOW] = inp.ptsnow
    if ntile > TILE_BARE_SOIL:
        tsrf[:, TILE_BARE_SOIL] = inp.ptsm1m
    if ntile > TILE_LAKE and inp.phlice is not None \
            and inp.ptlice is not None and inp.ptlwml is not None:
        # vdfdifh1s line 382: PHLICE > 1e-9 picks the ice skin T, else
        # the mixed-layer T.
        is_ice = (inp.phlice > _LAKE_HICE_MIN).to(dtype)
        tsrf[:, TILE_LAKE] = is_ice * inp.ptlice + (1.0 - is_ice) * inp.ptlwml
    elif ntile > TILE_LAKE:
        # FLake prognostics not surfaced (use_flake=False stub branch).
        # Fall back to PTSM1M so the lake tile is not left at the
        # ``torch.empty`` allocation (uninitialised memory -- the same
        # cross-column "leak" class as the urban tile below).
        tsrf[:, TILE_LAKE] = inp.ptsm1m
    if ntile > TILE_URBAN:
        # vdfdifh1s line 387: the urban tile's sub-surface T is PTSM1M
        # (same as bare soil).  Without this the column is left at the
        # ``torch.empty`` allocation -- *uninitialised memory*, which is
        # batch-call-dependent and produced a cross-column "leak" that
        # the n_tile=10 batch-independence test catches.
        tsrf[:, TILE_URBAN] = inp.ptsm1m
    return tsrf


def _lesnml_repeat(
    inp: DiffusionInputs,
    seb_inp: SebInputs,
    out: SebOutputs,
    ptsk: torch.Tensor,
    pstr: torch.Tensor,
    pg0: torch.Tensor,
    pssh: torch.Tensor,
    pjq: torch.Tensor,
    pslh: torch.Tensor,
    consts: ConstantBundle,
    config: SurfConfig,
    rtt: float,
    dtype: torch.dtype,
    device,
) -> tuple:
    """LESNML snow-melt repeat (call #4, vdfdifh1s lines 518-547).

    For columns where ``PFRTI(:,TILE_LOW_VEG_SNOW) >= 1e-6`` *and* the
    call-#3 skin T on that tile is at melting (``ZTSK(:,5) >= RTT - 1e-6``),
    pin that tile's ``TSRF`` to ``RTT`` and ``LAMSK`` to 100 W/m^2/K and
    re-run SURFSEB.  The new outputs overwrite the call-#3 outputs on
    *every* tile for those columns (matching the Fortran, which calls
    SURFSEB on the whole KLON/NTILES slab rather than per-column).
    """
    # Build the per-(grid,tile) repeat mask.  LREPEAT in Fortran is
    # OR-reduced across columns; the SURFSEB call then runs on the
    # whole slab.  We replicate that: the repeat happens iff any column
    # triggers.
    active = inp.pfrti[:, TILE_LOW_VEG_SNOW] >= _LESNML_FRAC_MIN
    melting = out.ptsk[:, TILE_LOW_VEG_SNOW] >= rtt - _LESNML_TSK_EPS
    any_repeat = (active & melting).any()
    if not any_repeat:
        return ptsk, pstr, pg0, pssh, pjq, pslh

    # Pin TSRF and LAMSK on the low-veg-snow tile for melting columns.
    tsrf_r = seb_inp.tsrf.clone()
    lamsk_r = seb_inp.lamsk.clone()
    mask = (active & melting).to(dtype)
    # Pin to RTT wherever the mask is 1, keep original elsewhere.
    tsrf_r[:, TILE_LOW_VEG_SNOW] = (
        mask * rtt + (1.0 - mask) * tsrf_r[:, TILE_LOW_VEG_SNOW]
    )
    lamsk_r[:, TILE_LOW_VEG_SNOW] = (
        mask * _LESNML_LAMSK_MELT + (1.0 - mask) * lamsk_r[:, TILE_LOW_VEG_SNOW]
    )

    repeat_inp = SebInputs(
        sskm1m=seb_inp.sskm1m, tskm1m=seb_inp.tskm1m,
        qskm1m=seb_inp.qskm1m, dqsdT=seb_inp.dqsdT,
        rhochu=seb_inp.rhochu, rhocqu=seb_inp.rhocqu,
        alphal=seb_inp.alphal, alphas=seb_inp.alphas,
        ssrfl=seb_inp.ssrfl,
        slrfl_grid=seb_inp.slrfl_grid, tskrad_grid=seb_inp.tskrad_grid,
        emis_grid=seb_inp.emis_grid,
        tsrf=tsrf_r, lamsk=lamsk_r,
        frti=seb_inp.frti, ktvl_grid=seb_inp.ktvl_grid, ktvh_grid=seb_inp.ktvh_grid,
        asl_grid=seb_inp.asl_grid, bsl_grid=seb_inp.bsl_grid,
        aql_grid=seb_inp.aql_grid, bql_grid=seb_inp.bql_grid,
        ptmst=seb_inp.ptmst,
        lelwdd=seb_inp.lelwdd, lelwtl=seb_inp.lelwtl,
    )
    repeat_out = surf_seb(repeat_inp, consts, config)
    return repeat_out.ptsk, repeat_out.pstr, repeat_out.pg0, \
        repeat_out.pssh, repeat_out.pjq, repeat_out.pslh


__all__ = [
    "DiffusionInputs",
    "DiffusionOutputs",
    "vdfdifh1s",
    "RVDIFTS",
]
