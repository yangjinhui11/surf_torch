"""Jarvis stomatal resistance (Fortran ``VSURF`` §2.3 + ``VEVAP``).

Direct translation of the resistance-to-alpha chain that converts the
Jarvis canopy resistance ``PWET*`` arrays into the per-tile moisture-
flux closure coefficients ``alpha`` consumed by
:mod:`surf_pytorch.physics.seb`.  The chain is::

    VSURF  (resistance)  -> PWETL / PWETH / PWETHS / PWETB / PWETLU
    VEVAP  (alpha)       -> PCSAT = PCAIR = 1 / (1 + PCFQ * PWET/rho)

In the offline single-column driver the resistance routine runs only on
tile 4 (low-veg snow-free); tiles 6/7/8 re-use the values computed at
tile 4 (see ``vsurf_mod.F90`` 294-422).  In this port we follow the
same pattern -- :func:`jarvis_resistance` returns *grid-mean* resistances
and :func:`build_tile_alpha` dispatches them onto the active tile set.

Tile -> resistance mapping (port 0-based, see :mod:`surf_pytorch.config`):
::

    0 water        -> alpha = 1.0   (potential evap)
    1 sea-ice      -> alpha = 1.0
    2 wet-skin     -> alpha = 1.0   (skin reservoir)
    3 low-veg dry  -> PWETL
    4 low-veg snow -> 1.0           (snow-covered)
    5 high-veg dry -> PWETH
    6 high-veg sn. -> PWETHS
    7 bare soil    -> PWETB
    8 lake         -> 1.0
    9 urban        -> PWETB         (urban uses bare-soil resistance)

The "alpha = 1.0" tiles correspond to either free water (ocean, ice,
wet skin, snow) or tiles whose resistance is structurally inapplicable;
they bypass the VEVAP formula entirely.

Notes
-----
* Saturation: the upstream ``ZQSAIR`` uses ``FOEEW(PTMLEV)/PAPHMS`` with
  the ``ZCOR = 1/(1-RETV*q)`` factor -- identical to :func:`utils.qsat`
  with no salinity.  We call :func:`utils.qsat` so the saturation
  expression stays in lock-step with the rest of the port.
* Freeze shutdown: when the top soil layer ``PTSAM1M(:,1) <= RTT`` the
  transpiration stress factor F2 collapses to ``RCEPSW`` (vsurf 365-372)
  -- we reproduce this with a ``torch.where`` on the freeze mask.
* All branches are written with ``torch.where`` / ``torch.clamp`` so the
  routine is differentiable on demand under ``enable_grad`` (gradients
  are the local branch derivatives, matching the Fortran piecewise).
"""

from __future__ import annotations

from dataclasses import dataclass
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
from ..tables import SoilTable, VegTable, RootTable
from ..utils import qsat


# --------------------------------------------------------------------------- #
# Module-level scalars (vsurf_mod.F90 module-level + SUSVEG defaults).
# Kept here rather than in ConstantBundle because they are Jarvis-specific
# tuning constants that have no other consumer in the port.
# --------------------------------------------------------------------------- #
# Minimum VPD stress floor (vsurf ZEPSF3 = 1.0E-5).  Caps F3 away from
# the singular EXP(0)=1 limit at saturated air, where the VPD-driven
# stress would otherwise vanish identically.
ZEPSF3 = 1.0e-5

# Canopy unstressed resistance scale (vsurf ZRCLU = 50).  Used only for
# the diagnostic PWETLU; kept here for parity with vsurf.
RCLU = 50.0

# Minimum stomatal resistance over bare soil (vsurf ZRSMINB = 50 s/m).
# Bare soil has no stomata; this is a bulk-surface resistance scaling
# F2B into an absolute resistance for the bare-soil tile.
RSMIN_B = 50.0

# LAI -> resistance sentinel.  When LAI is identically zero the
# transpiring surface has no area; the upstream resistance is set to a
# very large number to switch the tile off.  vsurf uses 1.0E+6.
PWET_INF = 1.0e6

# Humidity-stress floor (susveg RCEPSW = 1.0E-3).  Caps the F2 stress
# factor away from zero so a shut-down canopy still leaks a tiny flux
# (numerical conditioning more than a physical effect).
RCEPSW = 1.0e-3


# --------------------------------------------------------------------------- #
# VEVAP tile-index dispatch (vevap_mod.F90 KTILE values, 1-based upstream).
# The driver calls VEVAP once per tile with KTILE selecting the branch.
# We map the port's 0-based tile indices to the upstream KTILE codes so
# the dispatch table reads in the same order as the Fortran IF-ladder.
# --------------------------------------------------------------------------- #
# Upstream KTILE (1-based) -> port tile index (0-based).
KTILE_WET_SKIN        = 3   # interception reservoir (port TILE_WET_SKIN=2)
KTILE_LOW_VEG_DRY     = 4   # low vegetation, snow-free (port 3)
KTILE_HIGH_VEG_DRY    = 6   # high vegetation, snow-free (port 5)
KTILE_HIGH_VEG_SNOW   = 7   # high vegetation with underlying snow (port 6)
KTILE_BARE_SOIL       = 8   # bare soil (port 7)
KTILE_URBAN           = 10  # urban (port 9)


@dataclass
class VevapOutputs:
    """Per-grid VEVAP outputs for a single tile (vevap_mod.F90).

    All fields have shape ``(Ngrid,)``.  ``pcsat`` / ``pcair`` are the
    moisture-flux closure coefficients consumed by the SEB
    (``alphas = pcsat``, ``alphal = pcair``); the remaining fields are
    diagnostics / inputs for downstream modules (unstressed canopy,
    snow-under-canopy, moist static energy).
    """

    pcsat: torch.Tensor     # multiplier of qs at the surface        [-]
    pcair: torch.Tensor     # multiplier of q at the lowest level     [-]
    pcpts: torch.Tensor     # dry static energy (stressed)           [J/kg]
    pcptsu: torch.Tensor    # dry static energy (unstressed)         [J/kg]
    pcsatu: torch.Tensor    # as pcsat, unstressed low veg            [-]
    pcairu: torch.Tensor    # as pcair, unstressed low veg            [-]
    pcsnw: torch.Tensor     # snow-through-canopy moisture factor    [-]
    pdhvegs: torch.Tensor   # canopy diagnostics (Ngrid, 2, 4)       [various]


# --------------------------------------------------------------------------- #
# Step 1: the Jarvis three-factor resistance.
# --------------------------------------------------------------------------- #
def jarvis_resistance(
    soil_t: torch.Tensor,           # (Ngrid, Nsoil) soil temperature
    soil_w: torch.Tensor,           # (Ngrid, Nsoil) total soil moisture
    soil_type: torch.Tensor,        # (Ngrid,) long, KSOTY
    ktvl: torch.Tensor,             # (Ngrid,) long, dominant low veg type
    ktvh: torch.Tensor,             # (Ngrid,) long, dominant high veg type
    t_air: torch.Tensor,            # (Ngrid,) PTMLEV lowest-model-level T
    q_air: torch.Tensor,            # (Ngrid,) PQMLEV lowest-model-level q
    p_surf: torch.Tensor,           # (Ngrid,) PAPHMS surface pressure
    sw_down: torch.Tensor,          # (Ngrid,) or (Ngrid,Ntile) downward SW
    veg_table: VegTable,
    soil_table: SoilTable,
    root_table: RootTable,
    consts: ConstantBundle,
    *,
    tile_frac: Optional[torch.Tensor] = None,  # (Ngrid, Ntile) for ZBARE
    n_tile: Optional[int] = None,
    lai_low: Optional[torch.Tensor] = None,    # (Ngrid,) PLAIL
    lai_high: Optional[torch.Tensor] = None,   # (Ngrid,) PLAIH
) -> dict:
    """Jarvis canopy resistance (vsurf_mod.F90 294-422).

    Computes the per-column resistance arrays ``PWETL`` / ``PWETH`` /
    ``PWETHS`` / ``PWETB`` / ``PWETLU`` and the three stress factors
    ``F1`` / ``F2`` / ``F3`` for low and high vegetation.  All branches
    follow the Fortran ``IF (KTILE == 4)`` block, which produces values
    that tiles 6/7/8 re-use (see the module docstring).

    Args
    ----
    soil_t, soil_w:
        Per-layer soil temperature (K) and total (liquid + ice) moisture
        content.  Shape ``(Ngrid, Nsoil)``.  ``soil_w`` carries the same
        units as ``PWSAM1M`` (volumetric fraction 0..wsat).
    soil_type:
        Soil texture class per column (long, 0..7).  Indexes the per-
        class scalars ``RWPWPM`` / ``RWCAPM`` / ``RWRESTM`` /
        ``RQWEVAPM`` in the :class:`SoilTable`.
    ktvl, ktvh:
        Dominant low / high vegetation type per column (long, 1..20).
    t_air, q_air, p_surf:
        Lowest-model-level air temperature, specific humidity and
        surface pressure -- the inputs to the F3 VPD calculation.
    sw_down:
        Downward short-wave radiation at the surface (W/m^2). A grid field
        is accepted for legacy callers. The SURF path supplies tile fields,
        from which VSURF uses low-vegetation tile 4 and high-vegetation tile
        6 for their respective Betts radiation-stress factors.
    veg_table, soil_table, root_table:
        Lookup tables -- we read RVRSMIN / RVHSTR / default RVLAI (veg),
        RWPWPM / RWCAPM / RWRESTM / RQWEVAPM (soil), RVROOTSA (roots).
    consts:
        :class:`ConstantBundle` -- needs ``rtt``, ``retv``, ``rd`` and
        the freeze-ramp endpoints ``rtf1..rtf4``.
    tile_frac, n_tile:
        Per-tile fractions (used only for the bare-soil weighted average
        ``ZWPBARE``).  When omitted, ``ZBARE`` defaults to zero so the
        bare-soil wilting point equals the soil-class ``RWPWPM``.
    lai_low, lai_high:
        Optional climatological tile LAI fields ``PLAIL`` / ``PLAIH``.
        When supplied they override the vegetation-table defaults, matching
        the offline driver's monthly LAI interpolation.

    Returns
    -------
    dict with keys ``pwetl``/``pweth``/``pweths``/``pwetb``/``pwetlu``
    (each ``(Ngrid,)``) plus the diagnostic stress factors
    ``f1l``/``f2l``/``f3l``/``lail`` and ``f1h``/``f2h``/``f3h``/``laih``.
    """
    dtype = soil_t.dtype
    device = soil_t.device

    # ----------------------------------------------------------------- #
    # 1. Soil-water liquid fraction (vsurf 301-319): the apparent energy
    #    correction that takes freezing / melting into account.  ZF is
    #    the liquid-water fraction ramp from 1 (T <= RTF2) to 0 (T >= RTF1).
    # ----------------------------------------------------------------- #
    t_soil = soil_t                                   # (Ngrid, Nsoil)
    inside_window = (t_soil > consts.rtf2) & (t_soil < consts.rtf1)
    below_window = t_soil <= consts.rtf2
    zf_inside = 0.5 * (1.0 - torch.sin(consts.rtf4 * (t_soil - consts.rtf3)))
    zf = torch.where(
        inside_window, zf_inside,
        torch.where(below_window, torch.ones_like(t_soil), torch.zeros_like(t_soil)),
    )

    # ----------------------------------------------------------------- #
    # 2. Liquid-equivalent moisture (vsurf 310-313).  Per-class scalars
    #    come from the SoilTable; gather once and broadcast over layers.
    # ----------------------------------------------------------------- #
    wpwp = soil_table.gather("wpwp", soil_type)        # (Ngrid,)
    wcap = soil_table.gather("wcap", soil_type)        # (Ngrid,)
    wrest = soil_table.gather("wrest", soil_type)      # (Ngrid,)
    qwevap = soil_table.gather("qwevap", soil_type)    # (Ngrid,)

    wliq_raw = soil_w * (1.0 - zf)                     # (Ngrid, Nsoil)
    # Broadcast the (Ngrid,) bounds to (Ngrid, Nsoil) via unsqueeze.
    wpwp_b = wpwp.unsqueeze(-1)
    wcap_b = wcap.unsqueeze(-1)
    wrest_b = wrest.unsqueeze(-1)
    zliq = torch.clamp(wliq_raw, min=wpwp_b, max=wcap_b)
    zliqr = torch.clamp(wliq_raw, min=wrest_b, max=wcap_b)

    # ----------------------------------------------------------------- #
    # 3. Per-veg-type table lookups (vsurf 325-330, 384-385).
    # ----------------------------------------------------------------- #
    rsminl = veg_table.gather("rvrsmin", ktvl).to(dtype=dtype)
    rsminh = veg_table.gather("rvrsmin", ktvh).to(dtype=dtype)
    hstrl = veg_table.gather("rvhstr", ktvl).to(dtype=dtype)
    hstrh = veg_table.gather("rvhstr", ktvh).to(dtype=dtype)
    lail = veg_table.gather("rvlai", ktvl).to(dtype=dtype)
    laih = veg_table.gather("rvlai", ktvh).to(dtype=dtype)
    if lai_low is not None:
        lail = lai_low.to(dtype=dtype, device=device)
    if lai_high is not None:
        laih = lai_high.to(dtype=dtype, device=device)

    # ----------------------------------------------------------------- #
    # 4. Bare-soil fraction ZBARE (vsurf 333-336).  Optional -- when the
    #    caller does not supply tile fractions we fall back to zero so
    #    the bare-soil wilting point is the per-class RWPWPM.
    # ----------------------------------------------------------------- #
    if tile_frac is not None and n_tile is not None and n_tile > TILE_BARE_SOIL:
        zbare = tile_frac[..., TILE_BARE_SOIL].to(dtype=dtype).clone()
        if n_tile > TILE_URBAN:
            zbare = zbare + tile_frac[..., TILE_URBAN].to(dtype=dtype)
    else:
        zbare = torch.zeros_like(ktvl, dtype=dtype)

    # ----------------------------------------------------------------- #
    # 5. Root-weighted soil moisture ZWROOT (vsurf 338-343).
    #    RootTable.gather returns (Ngrid, Nsoil); sum over the soil axis.
    # ----------------------------------------------------------------- #
    rootsl = root_table.gather(ktvl).to(dtype=dtype)   # (Ngrid, Nsoil)
    rootsh = root_table.gather(ktvh).to(dtype=dtype)   # (Ngrid, Nsoil)
    zwrootl = (zliq * rootsl).sum(dim=-1)              # (Ngrid,)
    zwrooth = (zliq * rootsh).sum(dim=-1)              # (Ngrid,)

    # ----------------------------------------------------------------- #
    # 6. Bare-soil F2 (vsurf 344-363, LEVGEN branch).  ZWPBARE is the
    #    wilting-point weighted by the bare fraction; ZQWEVAPBARE its
    #    inverse difference against RWCAPM.  F2B clamps to [RCEPSW, 1].
    # ----------------------------------------------------------------- #
    zwpbare = wpwp * (1.0 - zbare) + wrest * zbare
    # vsurf guards JS >= 1 (real soil classes); index 0 ("water") would
    # divide by zero.  We replicate via a where that uses 0 in the
    # denominator -- the clamp on F2B keeps the result finite.
    denom_bare = torch.where(
        soil_type >= 1,
        wcap - zwpbare,
        torch.full_like(zwpbare, 1.0),                  # placeholder; masked below
    )
    qwevapbare = torch.where(
        soil_type >= 1,
        1.0 / denom_bare,
        torch.zeros_like(denom_bare),
    )
    f2b_raw = (zliqr[..., 0] - zwpbare) * qwevapbare
    f2b = torch.clamp(f2b_raw, min=RCEPSW, max=1.0)

    # ----------------------------------------------------------------- #
    # 7. Transpiration F2 (vsurf 365-372).  Freeze shutdown: if the top
    #    soil layer is at/below freezing the canopy is shut (F2 = RCEPSW).
    # ----------------------------------------------------------------- #
    frozen_top = t_soil[..., 0] <= consts.rtt
    f2l_active = torch.clamp((zwrootl - wpwp) * qwevap, min=RCEPSW, max=1.0)
    f2h_active = torch.clamp((zwrooth - wpwp) * qwevap, min=RCEPSW, max=1.0)
    f2l = torch.where(frozen_top, torch.full_like(f2l_active, RCEPSW), f2l_active)
    f2h = torch.where(frozen_top, torch.full_like(f2h_active, RCEPSW), f2h_active)

    # ----------------------------------------------------------------- #
    # 8. Radiation stress F1 (Betts; vsurf 378-381).  Identical for low
    #    and high vegetation.
    # ----------------------------------------------------------------- #
    if sw_down.ndim == 1:
        sw_low = sw_down
        sw_high = sw_down
    elif sw_down.ndim == 2 and sw_down.shape[1] > 5:
        # Fortran tile indices 4 and 6 are zero-based 3 and 5 here.
        sw_low = sw_down[:, 3]
        sw_high = sw_down[:, 5]
    else:
        raise ValueError(
            "sw_down must be (Ngrid,) or a tiled field with at least 6 tiles"
        )
    srfl_low = sw_low / 250.0
    srfl_high = sw_high / 250.0
    f1l = 1.0 / torch.clamp(
        0.81 * (1.0 + srfl_low) / (srfl_low + 0.05), min=1.0,
    )
    f1h = 1.0 / torch.clamp(
        0.81 * (1.0 + srfl_high) / (srfl_high + 0.05), min=1.0,
    )

    # ----------------------------------------------------------------- #
    # 9. Vapour-pressure-deficit stress F3 (vsurf 383-392).  ZQSAIR is
    #    the saturation specific humidity at the lowest-model-level air
    #    temperature, evaluated with FOEEW + the RETV correction (i.e.
    #    utils.qsat with no salinity).
    # ----------------------------------------------------------------- #
    qsat_air = qsat(t_air, p_surf, consts)              # (Ngrid,)
    f3l_raw = torch.exp(-hstrl * (qsat_air - q_air))
    f3h_raw = torch.exp(-hstrh * (qsat_air - q_air))
    f3l = torch.clamp(f3l_raw, min=ZEPSF3, max=1.0)
    f3h = torch.clamp(f3h_raw, min=ZEPSF3, max=1.0)

    # ----------------------------------------------------------------- #
    # 10. Resistances (vsurf 394-410).  PWETL/PWETH use the LAI-weighted
    #     product of the three stress factors; PWETLU is the unstressed
    #     canopy diagnostic; PWETHS = PWETH (snow under high veg reuses
    #     the snow-free high-veg resistance for the canopy term); the
    #     snow-aerodynamic term in VEVAP KTILE==7 is handled separately
    #     inside build_tile_alpha).
    # ----------------------------------------------------------------- #
    pwetl_active = rsminl / (lail * f1l * f2l * f3l)
    pwetl = torch.where(
        lail != 0.0, pwetl_active, torch.full_like(pwetl_active, PWET_INF)
    )
    pwetlu = RCLU / (f1l * f3l)
    pweth_active = rsminh / (laih * f1h * f2h * f3h)
    pweth = torch.where(
        laih != 0.0, pweth_active, torch.full_like(pweth_active, PWET_INF)
    )
    pweths = pweth
    pwetb = RSMIN_B / f2b

    return {
        "pwetl": pwetl, "pweth": pweth, "pweths": pweths,
        "pwetb": pwetb, "pwetlu": pwetlu,
        "f1l": f1l, "f2l": f2l, "f3l": f3l, "lail": lail,
        "f1h": f1h, "f2h": f2h, "f3h": f3h, "laih": laih,
        "f2b": f2b,
    }


# --------------------------------------------------------------------------- #
# Step 2: VEVAP alpha conversion.
# --------------------------------------------------------------------------- #
def resistance_to_alpha(
    pwet: torch.Tensor,             # (Ngrid,) or (Ngrid, Ntile) canopy resistance
    rhocqu: torch.Tensor,           # (Ngrid,) or (Ngrid, Ntile) PCFQ*rho (kg/m^2/s)
    p_surf: torch.Tensor,           # surface pressure [Pa]
    t_air: torch.Tensor,            # lowest-model-level air temperature [K]
    q_air: torch.Tensor,            # lowest-model-level specific humidity
    consts: ConstantBundle,
    *,
    snow_aero_resistance: Optional[torch.Tensor] = None,  # ZRAS (high-veg snow)
) -> torch.Tensor:
    """Convert a canopy resistance to the moisture-flux ``alpha`` (VEVAP).

    Faithful transcription of ``vevap_mod.F90`` 169-242 (tiles 4, 6, 7,
    8 -- the vegetated / bare tiles).  The upstream code computes::

        ZRHO  = PAPHMS / (RD * PTMLEV * (1 + RETV*PQMLEV))
        ZZWET = PWET / ZRHO
        PCSAT = PCAIR = 1 / (1 + PCFQ * ZZWET)

    where ``PCFQ`` is the turbulent exchange coefficient ``rhocqu / ZRHO``
    (note: VEVAP receives ``PCFQ`` as ``rhocqu / rho``; we accept the
    *product* ``rhocqu`` directly because that is what the SEB uses).
    Equivalently ``alpha = 1 / (1 + rhocqu * PWET / rho**2)``; but the
    VEVAP form ``1/(1 + (rhocqu/rho) * (PWET/rho))`` keeps the
    intermediate ``ZZWET`` physical (a conductance ratio).  We follow
    the latter so the snow-under-high-veg branch (KTILE==7) can add its
    extra aerodynamic term cleanly.

    The condensation branch (``PQMLEV > PQS``) leaves ``alpha = 1`` --
    VEVAP only reassigns ``PCSAT``/``PCAIR`` when ``PQMLEV <= PQS``.  We
    expose this through the ``cond_mask`` returned alongside the alpha
    so the caller can route the right tile value.

    Args
    ----
    pwet:
        Canopy resistance in s/m (PWETL / PWETH / PWETHS / PWETB).
        Shape ``(Ngrid,)`` or broadcastable to the alpha shape.
    rhocqu:
        Turbulent moisture exchange coefficient ``rho * CH * |V|`` in
        kg/m^2/s.  Same shape as ``pwet``.
    p_surf, t_air, q_air:
        Surface pressure, lowest-model-level air temperature and
        specific humidity.  Used to evaluate air density ``ZRHO`` and
        the condensation mask ``q_air > qsat(t_air)``.
    snow_aero_resistance:
        Optional snow aerodynamic resistance ``ZRAS`` (s/m) for the
        KTILE==7 (snow under high veg) branch.  When provided the VEVAP
        PCSAT formula becomes ``1/(1 + ZZWET*PCFQ + ZZWET/ZZWETS)``
        with ``ZZWETS = ZRAS/ZRHO``; the snow-under-canopy resistance
        ``PCSNW = 1/(1 + ZZWETS*PCFQ + ZZWETS/ZZWET)`` is also returned
        via the ``snow_alpha`` field of the result dict.  When omitted
        the routine uses the simple ``1/(1 + PCFQ*ZZWET)`` form.

    Returns
    -------
    alpha : ``(Ngrid,)`` tensor.  Equals 1 where the air is condensing
        (``q_air > qsat``), otherwise the resistance-corrected value.
    """
    rho = p_surf / (consts.rd * t_air * (1.0 + consts.retv * q_air))
    rho = rho.to(dtype=pwet.dtype)

    # Conductance ratio ZZWET = PWET / rho (and ZZWETS for the snow branch).
    zwet = pwet / rho
    pcfq = rhocqu / rho                            # vsurf PCFQ = rho*CH*V / rho

    if snow_aero_resistance is not None:
        zwets = snow_aero_resistance / rho
        denom = 1.0 + zwet * pcfq + zwet / zwets
        alpha = 1.0 / denom
    else:
        alpha = 1.0 / (1.0 + pcfq * zwet)

    # Condensation: where q_air > qsat(t_air), VEVAP leaves PCSAT=1.
    qsat_air = qsat(t_air, p_surf, consts)
    cond_mask = q_air > qsat_air
    alpha = torch.where(cond_mask, torch.ones_like(alpha), alpha)
    return alpha


# --------------------------------------------------------------------------- #
# Step 2b: faithful VEVAP transcription (vevap_mod.F90 138-269).
# --------------------------------------------------------------------------- #
def vevap(
    ktile: int,
    pwet: torch.Tensor,              # (Ngrid,) tile-appropriate resistance
    pwetlu: Optional[torch.Tensor],  # (Ngrid,) unstressed low-veg resistance
    rhocqu: torch.Tensor,            # (Ngrid,) PCFQ = rho*Cq*|U|  [kg/m^2/s]
    p_surf: torch.Tensor,            # (Ngrid,) PAPHMS              [Pa]
    t_air: torch.Tensor,             # (Ngrid,) PTMLEV              [K]
    q_air: torch.Tensor,             # (Ngrid,) PQMLEV
    qs_skin: torch.Tensor,           # (Ngrid,) PQS = qsat at skin T
    skin_t: torch.Tensor,            # (Ngrid,) PTSKM1M             [K]
    t_sub: torch.Tensor,             # (Ngrid,) PTSAM1M (sub-surface T) [K]
    wlmx: torch.Tensor,              # (Ngrid,) PWLMX interception capacity [m]
    consts: ConstantBundle,
    *,
    ptmst: torch.Tensor,             # (Ngrid,) time step (for interception clamp)
    rlhaero: float = 15.0,           # aerodynamic skin conductivity (unstable)
    rlhaeros: float = 10.0,          # aerodynamic skin conductivity (stable)
) -> "VevapOutputs":
    """Compute the equivalent evapotranspiration efficiency (VEVAP).

    Faithful transcription of ``vevap_mod.F90`` lines 138-269.  The
    upstream code is called once per tile (``KTILE`` selects the branch);
    this function mirrors that -- pass the upstream ``KTILE`` (1-based)
    and the per-grid inputs and it returns the per-grid VEVAP outputs.

    The moisture-flux closure is::

        ZRHO = PAPHMS / (RD * PTMLEV * (1 + RETV * PQMLEV))
        ZZWET = PWET / ZRHO
        PCSAT = PCAIR = 1 / (1 + PCFQ * ZZWET)

    where ``PCFQ`` is the rho-scaled moisture exchange coefficient
    (``rho * Cq * |U|`` in kg/m^2/s) -- i.e. exactly the SEB's
    ``rhocqu``.  Note this is **not** divided by rho again: the previous
    ``resistance_to_alpha`` recomputed ``pcf q = rhocqu / rho`` and then
    multiplied by ``zwet = pwet / rho``, putting an extra ``1/rho`` in
    the denominator.  That divergence from VEVAP was masked by the test
    structure (both sides used the same Python formula); this routine is
    the faithful reference and the previous one is kept only as a
    deprecated alias.

    Tile branches (vevap_mod.F90 IF-ladder)
    ---------------------------------------
    * KTILE=3 (wet skin): interception-reservoir clamp
      ``PCAIR = MIN(1, ZEMAX/ZEP)`` when ``ZEP < 0 < ZEP - ZEMAX``,
      where ``ZEP = PCFQ*(q_air - qs)`` and ``ZEMAX = -WLMX/dt``.
    * KTILE=4 (low veg dry): ``1/(1 + PCFQ*ZZWET)`` plus the unstressed
      ``1/(1 + PCFQ*ZZWETU)`` with ``ZZWETU = PWETLU/rho``; fills the
      ``PDHVEGS(1, 3..4)`` canopy diagnostics.
    * KTILE=6 (high veg dry): ``1/(1 + PCFQ*ZZWET)``; fills
      ``PDHVEGS(2, 3..4)``.
    * KTILE=7 (high veg snow): the canopy-over-snow coupling with an
      aerodynamic resistance ``ZRAS = 1200/RLHAEROS`` (skin T above
      sub-surface T) or ``1200/RLHAERO`` (otherwise), giving
      ``PCSAT = 1/(1 + ZZWET*PCFQ + ZZWET/ZZWETS)`` and, when the
      canopy resistance is active, the snow-side factor
      ``PCSNW = 1/(1 + ZZWETS*PCFQ + ZZWETS/ZZWET)``.
    * KTILE=8 (bare soil) / KTILE=10 (urban): ``1/(1 + PCFQ*ZZWET)``.

    The condensation mask ``PQMLEV > PQS`` leaves PCSAT/PCAIR at their
    initialised value of 1 (VEVAP only reassigns inside the
    ``PQMLEV <= PQS`` branch).

    Args
    ----
    ktile:
        Upstream tile index (1-based) selecting the branch (3, 4, 6, 7,
        8, or 10).
    pwet:
        The tile-appropriate Jarvis resistance in s/m (PWETL / PWETH /
        PWETHS / PWETB).  Shape ``(Ngrid,)``.
    pwetlu:
        Unstressed low-veg resistance PWETLU (only used by KTILE=4).
        May be ``None`` for other tiles.
    rhocqu:
        Rho-scaled moisture exchange coefficient PCFQ (kg/m^2/s).  This
        is the SEB's ``rhocqu`` directly -- do NOT pre-divide by rho.
    p_surf, t_air, q_air:
        Lowest-model-level pressure (Pa), temperature (K) and specific
        humidity; used to evaluate air density ZRHO and (implicitly via
        the caller-supplied condensation mask) the saturation check.
    qs_skin:
        Saturation specific humidity at the skin temperature (PQS).
    skin_t:
        Skin temperature PTSKM1M (K); used by KTILE=7 (Tsk vs Tsub) and
        by the PCPTS moist-static-energy term.
    t_sub:
        Sub-surface temperature PTSAM1M (K); used by KTILE=7 stability
        selector.
    wlmx:
        Maximum interception reservoir capacity PWLMX (m); used by
        KTILE=3 clamp.
    ptmst:
        Coupling time step (s); used by KTILE=3 ``ZEMAX = -WLMX/dt``.
    rlhaero, rlhaeros:
        Aerodynamic skin conductivities for the KTILE=7 ZRAS selector
        (susveg defaults 15.0 / 10.0).

    Returns
    -------
    :class:`VevapOutputs` with the per-grid PCSAT / PCAIR / PCSATU /
    PCAIRU / PCPTSU / PCSNW / PDHVEGS tensors.
    """
    ngrid = rhocqu.shape[0]
    dtype = rhocqu.dtype
    device = rhocqu.device

    # Section 1 -- initialise all outputs to 1.0 (VEVAP 147-151).  PCSAT
    # / PCSATU / PCAIR / PCAIRU default to 1.0; PCSNW defaults to 0.0
    # but VEVAP only zeroes it inside the KTILE==7 branch, so we start
    # at 0 here and let each branch set what it owns.
    pcsat   = torch.ones((ngrid,), dtype=dtype, device=device)
    pcsatu  = torch.ones((ngrid,), dtype=dtype, device=device)
    pcair   = torch.ones((ngrid,), dtype=dtype, device=device)
    pcairu  = torch.ones((ngrid,), dtype=dtype, device=device)
    pcsnw   = torch.zeros((ngrid,), dtype=dtype, device=device)
    # PDHVEGS(Ngrid, 2, 4) -- only KTILE=4/6 write columns 3..4 of rows
    # 1..2; the rest stays at the VEVAP default (0).
    pdhvegs = torch.zeros((ngrid, 2, 4), dtype=dtype, device=device)

    # Air density ZRHO (vevap 172).  Identical across every branch that
    # needs it, so compute once.  rho = P / (RD*T*(1+RETV*q)).
    rho = p_surf / (consts.rd * t_air * (1.0 + consts.retv * q_air))
    rho = rho.to(dtype=dtype)

    # PCFQ is the rho-scaled exchange coefficient the caller passes
    # directly (rho*Cq*|U|).  VEVAP multiplies ZZWET by PCFQ without any
    # further rho scaling.
    pcfq = rhocqu

    # Condensation mask: VEVAP only reassigns PCSAT/PCAIR inside the
    # PQMLEV <= PQS branch.  We compute the evaporation mask once and
    # gate every branch with it so off-mask columns stay at 1.0.
    evap_mask = q_air <= qs_skin

    cpd = consts.cpd
    rvtmp2 = consts.rvtmp2

    if ktile == KTILE_WET_SKIN:
        # Interception reservoir (vevap 153-164).  Clamp the
        # evaporation so it cannot drain more than the reservoir holds
        # in one step: ZEP = PCFQ*(q_air - qs), ZEMAX = -WLMX/dt; when
        # both are negative and ZEP < ZEMAX, scale PCAIR down.
        zep = pcfq * (q_air - qs_skin)
        zemax = -wlmx / ptmst
        clamp_mask = (zep < 0.0) & (zemax < 0.0) & (zep < zemax)
        # PCAIR = MIN(1, ZEMAX/ZEP) only where the clamp fires and we
        # are evaporating (evap_mask).  Guard the division: where the
        # clamp does not fire we keep PCAIR=1 (use a safe denominator).
        safe_zep = torch.where(clamp_mask, zep, torch.ones_like(zep))
        ratio = zemax / safe_zep
        pcair_new = torch.clamp(ratio, max=1.0)
        pcair = torch.where(clamp_mask & evap_mask, pcair_new, pcair)
        pcsat = pcair
        pcairu = pcair
        pcsatu = pcsat

    elif ktile == KTILE_LOW_VEG_DRY:
        # Low vegetation (vevap 169-185).  PCSAT = PCAIR use PWETL,
        # PCSATU = PCAIRU use PWETLU.  PDHVEGS(:,1,3)=PWETL,
        # PDHVEGS(:,1,4)=rho/PCFQ.
        zwet = pwet / rho
        pcsat_v = 1.0 / (1.0 + pcfq * zwet)
        pcair_v = pcsat_v
        pcsat = torch.where(evap_mask, pcsat_v, pcsat)
        pcair = torch.where(evap_mask, pcair_v, pcair)
        if pwetlu is not None:
            zwetu = pwetlu / rho
            pcsatu_v = 1.0 / (1.0 + pcfq * zwetu)
            pcairu_v = pcsatu_v
            pcsatu = torch.where(evap_mask, pcsatu_v, pcsatu)
            pcairu = torch.where(evap_mask, pcairu_v, pcairu)
        pdhvegs[:, 0, 2] = torch.where(
            evap_mask, pwet, pdhvegs[:, 0, 2])
        # rho/PCFQ -- guard PCFQ==0 (would be a zero-turbulence column).
        safe_pcfq = torch.where(
            pcfq != 0.0, pcfq, torch.ones_like(pcfq))
        pdhvegs[:, 0, 3] = torch.where(
            evap_mask, rho / safe_pcfq, pdhvegs[:, 0, 3])

    elif ktile == KTILE_HIGH_VEG_DRY:
        # High vegetation, snow-free (vevap 189-204).  PCSAT = PCAIR =
        # PCSATU = PCAIRU = 1/(1+PCFQ*ZZWET).  PDHVEGS(:,2,3..4).
        zwet = pwet / rho
        pcsat_v = 1.0 / (1.0 + pcfq * zwet)
        pcsat = torch.where(evap_mask, pcsat_v, pcsat)
        pcair = pcsat
        pcairu = pcair
        pcsatu = pcsat
        pdhvegs[:, 1, 2] = torch.where(
            evap_mask, pwet, pdhvegs[:, 1, 2])
        safe_pcfq = torch.where(
            pcfq != 0.0, pcfq, torch.ones_like(pcfq))
        pdhvegs[:, 1, 3] = torch.where(
            evap_mask, rho / safe_pcfq, pdhvegs[:, 1, 3])

    elif ktile == KTILE_HIGH_VEG_SNOW:
        # High vegetation with underlying snow (vevap 208-230).  ZRAS
        # depends on the skin vs sub-surface temperature contrast;
        # PCSAT couples canopy and snow resistances; PCSNW is the
        # snow-side factor (active when the canopy resistance is real).
        pcsnw_v = torch.zeros((ngrid,), dtype=dtype, device=device)
        # VEVAP uses a strict PTSKM1M > PTSAM1M selector for the
        # snow-under-high-vegetation aerodynamic resistance.
        warm = skin_t > t_sub
        zras = torch.where(
            warm,
            torch.full_like(rho, 1200.0 / rlhaeros),
            torch.full_like(rho, 1200.0 / rlhaero),
        )
        zwet = pwet / rho
        zwets = zras / rho
        denom = 1.0 + zwet * pcfq + zwet / zwets
        pcsat_v = 1.0 / denom
        pcsat = torch.where(evap_mask, pcsat_v, pcsat)
        pcair = pcsat
        # PCSNW only when PWETHS > 1 (real canopy resistance).  Guard
        # zwet==0 (would divide by zero).
        snow_active = evap_mask & (pwet > 1.0)
        safe_zwet = torch.where(
            snow_active, zwet, torch.ones_like(zwet))
        pcsnw_denom = 1.0 + zwets * pcfq + zwets / safe_zwet
        pcsnw_active = 1.0 / pcsnw_denom
        pcsnw_v = torch.where(snow_active, pcsnw_active, pcsnw_v)
        pcsnw = pcsnw_v
        pcairu = pcair
        pcsatu = pcsat

    elif ktile in (KTILE_BARE_SOIL, KTILE_URBAN):
        # Bare soil / urban (vevap 234-256).  Simple PCSAT = PCAIR =
        # PCSATU = PCAIRU = 1/(1+PCFQ*ZZWET).  Urban comment notes
        # RURBRES is not currently used -> reuses PWETB.
        zwet = pwet / rho
        pcsat_v = 1.0 / (1.0 + pcfq * zwet)
        pcsat = torch.where(evap_mask, pcsat_v, pcsat)
        pcair = pcsat
        pcairu = pcair
        pcsatu = pcsat

    else:
        # Tiles without a VEVAP branch (water / ice / snow on low veg /
        # lake) keep the initialised PCSAT=PCAIR=...=1 -- potential
        # evaporation from a free surface.
        pass

    # Section 3 -- stressed and unstressed moist-static-energy terms
    # (vevap 260-265).  The final prognostic SURFSEB call uses PCPTS;
    # PCPTSU belongs to the separate potential-evaporation diagnostic.
    pcpts = skin_t * cpd * (
        1.0 + rvtmp2 * (pcsat * qs_skin + (1.0 - pcair) * q_air)
    )
    pcptsu = skin_t * cpd * (
        1.0 + rvtmp2 * (pcsatu * qs_skin + (1.0 - pcairu) * q_air)
    )

    return VevapOutputs(
        pcsat=pcsat, pcair=pcair, pcpts=pcpts, pcptsu=pcptsu,
        pcsatu=pcsatu, pcairu=pcairu, pcsnw=pcsnw, pdhvegs=pdhvegs,
    )


# --------------------------------------------------------------------------- #
# Step 3: per-tile alpha dispatch.
# --------------------------------------------------------------------------- #
def build_tile_alpha(
    pwetl: torch.Tensor,             # (Ngrid,) low-veg resistance
    pweth: torch.Tensor,             # (Ngrid,) high-veg resistance
    pwetb: torch.Tensor,             # (Ngrid,) bare-soil resistance
    rhocqu_grid: torch.Tensor,       # (Ngrid,) or (Ngrid, Ntile) PCFQ*rho
    p_surf: torch.Tensor,            # (Ngrid,)
    t_air: torch.Tensor,            # (Ngrid,)
    q_air: torch.Tensor,            # (Ngrid,)
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    pweths: Optional[torch.Tensor] = None,
    snow_aero_resistance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the per-tile ``alpha`` tensor ``(Ngrid, Ntile)``.

    Dispatches the per-column Jarvis resistances onto the active tiles
    and converts each to a VEVAP ``alpha``.  Tiles 0/1/2 (water / ice /
    wet-skin) and tile 4 (snow on low veg), 8 (lake) carry ``alpha=1``
    (potential evaporation from a free surface); tiles 3/5/6/7/9 use
    the Jarvis-derived resistances.

    Because ``alpha_l == alpha_s`` in the offline driver (VEVAP sets
    ``PCSAT = PCAIR`` for every vegetated tile -- see ``vevap_mod.F90``
    175-176, 194-197), a single per-tile alpha tensor suffices for the
    SEB closure.

    Args
    ----
    pwetl, pweth, pwetb:
        Per-column canopy / bare-soil resistances from
        :func:`jarvis_resistance`.  Shape ``(Ngrid,)``.
    rhocqu_grid:
        Turbulent moisture exchange coefficient per column (the SEB's
        ``rhocqu``).  Broadcast to every active tile.
    p_surf, t_air, q_air:
        Lowest-model-level state used by :func:`resistance_to_alpha`.
    pweths:
        Optional resistance for the snow-under-high-veg tile (defaults
        to ``pweth`` -- matches vsurf line 409).
    snow_aero_resistance:
        Optional ``ZRAS`` for the KTILE==7 branch (see
        :func:`resistance_to_alpha`).

    Returns
    -------
    alpha : ``(Ngrid, Ntile)`` tensor of moisture-flux closure factors.
    """
    ntile = config.n_tile
    ngrid = rhocqu_grid.shape[0]
    dtype = pwetl.dtype
    device = pwetl.device
    if pweths is None:
        pweths = pweth

    alpha = torch.ones((ngrid, ntile), dtype=dtype, device=device)

    # Accept either a grid-mean (Ngrid,) PCFQ (broadcast across tiles,
    # legacy) or a per-tile (Ngrid, Ntile) PCFQ (per_tile_exchange path,
    # matching the upstream VEVAPS tile loop).
    if rhocqu_grid.dim() == 1:
        rhocqu_t = rhocqu_grid.unsqueeze(-1).expand(ngrid, ntile)
    else:
        rhocqu_t = rhocqu_grid

    # Helper: tile-local resistance -> alpha via VEVAP.  We pull the
    # per-tile slice of pwet/rhocqu, run resistance_to_alpha, and write
    # back into alpha[..., jt].  resistance_to_alpha handles the
    # condensation mask (alpha=1 where q_air > qsat) for us.
    def _set(jt: int, pwet_tile: torch.Tensor, *, snow_res: Optional[torch.Tensor] = None) -> None:
        a = resistance_to_alpha(
            pwet_tile, rhocqu_t[..., jt], p_surf, t_air, q_air, consts,
            snow_aero_resistance=snow_res,
        )
        alpha[..., jt] = a

    # Tile 3 (low-veg dry, JT=4): PWETL.
    if ntile > TILE_LOW_VEG_DRY:
        _set(TILE_LOW_VEG_DRY, pwetl)

    # Tile 5 (high-veg dry, JT=6): PWETH.
    if ntile > TILE_HIGH_VEG_DRY:
        _set(TILE_HIGH_VEG_DRY, pweth)

    # Tile 6 (high-veg snow, JT=7): PWETHS with the snow aerodynamic term.
    if ntile > TILE_HIGH_VEG_SNOW:
        _set(TILE_HIGH_VEG_SNOW, pweths, snow_res=snow_aero_resistance)

    # Tile 7 (bare soil, JT=8): PWETB.
    if ntile > TILE_BARE_SOIL:
        _set(TILE_BARE_SOIL, pwetb)

    # Tile 9 (urban, JT=10): PWETB (vsurf/vevap reuse bare-soil resistance).
    if ntile > TILE_URBAN:
        _set(TILE_URBAN, pwetb)

    return alpha


# --------------------------------------------------------------------------- #
# Step 3b: full per-tile VEVAP assembly (the faithful, complete closure).
# --------------------------------------------------------------------------- #
def build_tile_vevap(
    pwetl: torch.Tensor,             # (Ngrid,) low-veg resistance
    pweth: torch.Tensor,             # (Ngrid,) high-veg resistance
    pwetb: torch.Tensor,             # (Ngrid,) bare-soil resistance
    rhocqu_grid: torch.Tensor,       # (Ngrid,) or (Ngrid, Ntile) PCFQ = rho*Cq*|U|
    p_surf: torch.Tensor,            # (Ngrid,)
    t_air: torch.Tensor,             # (Ngrid,) PTMLEV
    q_air: torch.Tensor,             # (Ngrid,) PQMLEV
    skin_t: torch.Tensor,            # (Ngrid, Ntile) PTSKM1M (per-tile skin T)
    t_sub_grid: torch.Tensor,        # (Ngrid,) or (Ngrid,Ntile) sub-surface T
    qs_skin: torch.Tensor,           # (Ngrid, Ntile) PQS = qsat at skin T
    wlmx_grid: torch.Tensor,         # (Ngrid,) PWLMX interception capacity
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    pwetlu: Optional[torch.Tensor] = None,   # (Ngrid,) unstressed low-veg
    pweths: Optional[torch.Tensor] = None,   # (Ngrid,) snow-under-high-veg
    ptmst: Optional[torch.Tensor] = None,    # (Ngrid,) coupling time step
    rlhaero: float = 15.0,
    rlhaeros: float = 10.0,
) -> dict:
    """Build the per-tile VEVAP closure tensors ``(Ngrid, Ntile)``.

    Dispatches the per-column Jarvis resistances onto the active tiles
    and runs the faithful :func:`vevap` for each, producing the full
    set of per-tile VEVAP outputs the SEB and downstream modules
    consume.  This is the rho-convention-correct successor to
    :func:`build_tile_alpha` (which used the simplified
    ``resistance_to_alpha`` formula with an extra ``1/rho``).

    Per-tile dispatch (port 0-based tile index -> upstream KTILE):
    ::

        0 water / 1 ice / 4 low-veg snow / 8 lake -> alpha = 1
            (no VEVAP branch; free-surface potential evaporation)
        2 wet-skin   -> KTILE=3 (interception-reservoir clamp)
        3 low-veg    -> KTILE=4 (PWETL, unstressed PWETLU)
        5 high-veg   -> KTILE=6 (PWETH)
        6 high-veg-snow -> KTILE=7 (PWETHS + ZRAS, PCSNW)
        7 bare soil  -> KTILE=8 (PWETB)
        9 urban      -> KTILE=10 (PWETB)

    Args
    ----
    pwetl, pweth, pwetb:
        Per-column Jarvis resistances from :func:`jarvis_resistance`,
        shape ``(Ngrid,)``.
    rhocqu_grid:
        Per-column rho-scaled moisture exchange coefficient PCFQ
        (``rho*Cq*|U|`` in kg/m^2/s).  Broadcast across tiles.
    p_surf, t_air, q_air:
        Per-grid lowest-model-level pressure (Pa), temperature (K) and
        specific humidity.
    skin_t:
        Per-tile skin temperature PTSKM1M, shape ``(Ngrid, Ntile)``.
        VEVAP consumes the skin T per tile (the upstream driver calls
        VEVAP once per tile with that tile's PTSKM1M).
    t_sub_grid:
        Per-grid or per-tile sub-surface temperature.  ``SURFEXCDRIVER``
        passes a tile-local ``ZTSA`` to ``VEVAP``; in particular KTILE=7
        receives snow temperature rather than the top soil temperature.
    qs_skin:
        Per-tile saturation specific humidity PQS = qsat at skin T,
        shape ``(Ngrid, Ntile)``.
    wlmx_grid:
        Per-grid interception reservoir capacity PWLMX (m), used by
        the KTILE=3 clamp.
    pwetlu:
        Optional unstressed low-veg resistance (defaults to ``pwetl``
        when omitted -- a reasonable placeholder since the unstressed
        diagnostic is not consumed by the SEB).
    pweths:
        Optional snow-under-high-veg resistance (defaults to ``pweth``).
    ptmst:
        Per-grid coupling time step (s); defaults to a scalar 3600 s
        when omitted (only feeds the KTILE=3 ``ZEMAX`` clamp).
    rlhaero, rlhaeros:
        Aerodynamic skin conductivities for the KTILE=7 ZRAS selector.

    Returns
    -------
    dict with keys ``pcsat`` / ``pcair`` / ``pcsatu`` / ``pcairu`` /
    ``pcptsu`` / ``pcsnw`` / ``pdhvegs`` -- each a per-tile tensor of
    shape ``(Ngrid, Ntile)`` (pdhvegs is ``(Ngrid, Ntile, 2, 4)``).
    The SEB consumes ``pcsat`` (-> alphas) and ``pcair`` (-> alphal);
    the rest are exposed for diagnostics and the M5 canopy update.
    """
    ntile = config.n_tile
    ngrid = rhocqu_grid.shape[0]
    dtype = pwetl.dtype
    device = pwetl.device
    if pweths is None:
        pweths = pweth
    if pwetlu is None:
        pwetlu = pwetl
    if ptmst is None:
        ptmst = torch.full((ngrid,), 3600.0, dtype=dtype, device=device)

    # Per-tile output accumulators.  Initialised to the VEVAP defaults
    # (1.0 for the alpha coefficients, 0.0 for PCSNW, zero for PDHVEGS);
    # tiles without a VEVAP branch keep these defaults.
    pcsat  = torch.ones((ngrid, ntile), dtype=dtype, device=device)
    pcair  = torch.ones((ngrid, ntile), dtype=dtype, device=device)
    pcsatu = torch.ones((ngrid, ntile), dtype=dtype, device=device)
    pcairu = torch.ones((ngrid, ntile), dtype=dtype, device=device)
    pcsnw  = torch.zeros((ngrid, ntile), dtype=dtype, device=device)
    pcpts = torch.zeros((ngrid, ntile), dtype=dtype, device=device)
    pcptsu = torch.zeros((ngrid, ntile), dtype=dtype, device=device)
    pdhvegs = torch.zeros((ngrid, ntile, 2, 4), dtype=dtype, device=device)

    # Accept either a grid-mean (Ngrid,) PCFQ (broadcast across tiles,
    # legacy) or a per-tile (Ngrid, Ntile) PCFQ (per_tile_exchange path,
    # matching the upstream VEVAPS tile loop which receives PCFQTI(:,JT)).
    per_tile_rhocqu = rhocqu_grid.dim() == 2

    def _run_tile(jt: int, ktile: int, pwet_tile: torch.Tensor) -> None:
        """Run vevap for one tile and scatter its outputs into the
        per-tile arrays.  Uses the per-tile skin T / qs columns."""
        rhocqu_jt = rhocqu_grid[:, jt] if per_tile_rhocqu else rhocqu_grid
        t_sub_jt = t_sub_grid[:, jt] if t_sub_grid.dim() == 2 else t_sub_grid
        out = vevap(
            ktile=ktile,
            pwet=pwet_tile,
            pwetlu=pwetlu,
            rhocqu=rhocqu_jt,
            p_surf=p_surf,
            t_air=t_air,
            q_air=q_air,
            qs_skin=qs_skin[:, jt],
            skin_t=skin_t[:, jt],
            t_sub=t_sub_jt,
            wlmx=wlmx_grid,
            consts=consts,
            ptmst=ptmst,
            rlhaero=rlhaero,
            rlhaeros=rlhaeros,
        )
        pcsat[:, jt]  = out.pcsat
        pcair[:, jt]  = out.pcair
        pcpts[:, jt]  = out.pcpts
        pcsatu[:, jt] = out.pcsatu
        pcairu[:, jt] = out.pcairu
        pcptsu[:, jt] = out.pcptsu
        pcsnw[:, jt]  = out.pcsnw
        pdhvegs[:, jt] = out.pdhvegs

    # Dispatch each active tile.  The KTILE codes are the upstream
    # 1-based indices from vevap_mod.F90's IF-ladder.
    if ntile > TILE_WET_SKIN:
        _run_tile(TILE_WET_SKIN, KTILE_WET_SKIN, pwetl)   # PWET unused here
    if ntile > TILE_LOW_VEG_DRY:
        _run_tile(TILE_LOW_VEG_DRY, KTILE_LOW_VEG_DRY, pwetl)
    if ntile > TILE_HIGH_VEG_DRY:
        _run_tile(TILE_HIGH_VEG_DRY, KTILE_HIGH_VEG_DRY, pweth)
    if ntile > TILE_HIGH_VEG_SNOW:
        _run_tile(TILE_HIGH_VEG_SNOW, KTILE_HIGH_VEG_SNOW, pweths)
    if ntile > TILE_BARE_SOIL:
        _run_tile(TILE_BARE_SOIL, KTILE_BARE_SOIL, pwetb)
    if ntile > TILE_URBAN:
        _run_tile(TILE_URBAN, KTILE_URBAN, pwetb)

    # PCPTS / PCPTSU are computed unconditionally per tile in VEVAP (lines
    # 260-265 sit outside the IF-ladder).  Tiles without an alpha
    # branch (water / ice / snow-on-low-veg / lake) keep PCSATU =
    # PCAIRU = 1 but still get PCPTSU = Tsk*cpd*(1 + RVTMP2*(qs +
    # (1-1)*q_air)) = Tsk*cpd*(1 + RVTMP2*qs).  Fill the tiles we did
    # not run above so every tile carries a valid PCPTSU.  Vectorised:
    # where pcptsu is still 0, apply the free-surface form with
    # PCSATU=PCAIRU=1.
    cpd = consts.cpd
    rvtmp2 = consts.rvtmp2
    free_mask = pcptsu == 0.0
    pcptsu_free = skin_t * cpd * (1.0 + rvtmp2 * qs_skin)
    pcpts = torch.where(free_mask, pcptsu_free, pcpts)
    pcptsu = torch.where(free_mask, pcptsu_free, pcptsu)

    return {
        "pcsat": pcsat, "pcair": pcair,
        "pcpts": pcpts,
        "pcsatu": pcsatu, "pcairu": pcairu,
        "pcptsu": pcptsu, "pcsnw": pcsnw,
        "pdhvegs": pdhvegs,
    }


# --------------------------------------------------------------------------- #
# Step 4: Farquhar canopy resistance (cotworestress_mod.F90:412-431).
# --------------------------------------------------------------------------- #
def farquhar_resistance(
    forcing,
    skin_t_prev: torch.Tensor,        # (Ngrid, Ntile) per-tile skin T at t-1
    qskm1m: torch.Tensor,             # (Ngrid, Ntile) qsat at skin T
    ssrfl_grid: torch.Tensor,         # (Ngrid,) mean SW radiation [W/m^2]
    ktvl: torch.Tensor,               # (Ngrid,) dominant low-veg type (long)
    ktvh: torch.Tensor,               # (Ngrid,) dominant high-veg type (long)
    jarvis: dict,                     # output of jarvis_resistance (f2*, lai*, pwetb)
    veg_table: VegTable,
    consts: ConstantBundle,
    *,
    land_mask: Optional[torch.Tensor] = None,   # (Ngrid,) bool; defaults all land
    ktile_skin: int = 0,              # which tile's skin T drives photosynthesis
) -> dict:
    """Farquhar canopy resistance, mirroring ``cotworestress_mod.F90`` 412-431.

    Upstream ``LEFARQUHAR`` branch: the Farquhar photosynthesis model
    (``farquhar_mod.F90``) is called once per column to get the canopy
    conductance ``PGSTOT`` (m/s, for H2O); ``cotworestress`` then converts
    it to a bulk resistance ``PWET = 1/max(RCONDCTMIN, PGSTOT)`` (s/m) and
    ``vsurf`` 694-717 distributes ``PWET`` to ``PWETL``/``PWETH``/
    ``PWETHS`` for tiles 4/6/7.  The downstream VEVAP chain is *identical*
    to the Jarvis path -- only the resistance source changes.

    The bare-soil / urban resistances (``PWETB``, tile 8/10) are **not**
    replaced: upstream ``vsurf`` 687 keeps the Jarvis formula
    ``ZRSMINB/ZF2B`` for bare soil because ``COTWORESTRESS`` only computes
    the canopy (vegetation) conductance.  We therefore pass ``jarvis``
    through for ``pwetb`` / ``pwetlu``.

    Args
    ----
    forcing:
        :class:`Forcing` carrying ``co2`` and the lowest-level state.
    skin_t_prev:
        Per-tile skin T ``(Ngrid, Ntile)``; the ``ktile_skin``-th column
        drives photosynthesis (dominant-tile skin, default tile 0).
    qskm1m:
        Per-tile qsat at skin T ``(Ngrid, Ntile)``; the ``ktile_skin``-th
        column feeds ``pqsurf``/``pqs`` (land columns carry no salinity
        factor so ``pqsurf == pqs``).
    ssrfl_grid:
        Mean SW radiation per column ``(Ngrid,)`` [W/m^2].
    ktvl, ktvh:
        Dominant veg types per column ``(Ngrid,)`` long.  ``ktvl`` indexes
        the per-PFT tables (KVTYPE); ``ktvh`` is accepted for parity with
        the Jarvis caller but the Farquhar call uses the low/dominant type
        (matching the existing ``_dominant_veg_types`` cold-start
        approximation where ktvl == ktvh).
    jarvis:
        Output dict of :func:`jarvis_resistance`.  Provides ``f2l`` (the
        soil-moisture stress function) and ``lail`` (LAI) for the Farquhar
        inputs, plus ``pwetb`` / ``pwetlu`` which are returned unchanged.
    veg_table:
        :class:`VegTable` carrying the Farquhar per-PFT tables and the
        ``farquhar_consts`` bundle (incl. ``rcondctmin``).
    consts:
        :class:`ConstantBundle`.
    land_mask:
        Optional ``(Ngrid,)`` bool land mask; defaults to all-land (the
        caller is the land-surface package).
    ktile_skin:
        Which tile column of ``skin_t_prev`` / ``qskm1m`` drives
        photosynthesis (default 0 = dominant/first tile).

    Returns
    -------
    dict with the same keys :func:`build_tile_vevap` consumes:
    ``pwetl`` / ``pweth`` / ``pweths`` (Farquhar-derived), ``pwetb`` /
    ``pwetlu`` (Jarvis passthrough, upstream vsurf:687).  All ``(Ngrid,)``.
    """
    from .farquhar import farquhar as _farquhar, FarquharInputs

    ngrid = ssrfl_grid.shape[0]
    dtype = ssrfl_grid.dtype
    device = ssrfl_grid.device

    if land_mask is None:
        land_mask = torch.ones((ngrid,), dtype=torch.bool, device=device)

    rcondctmin = veg_table.farquhar_consts.rcondctmin

    def _tile_farquhar(tile: int, kvtype: torch.Tensor, lai: torch.Tensor, pf2: torch.Tensor):
        # VSURF calls COTWORESTRESS inside the tile loop; use the tile-local
        # skin temperature and qsat rather than a fixed grid column.
        skin_col = skin_t_prev[:, tile]
        qs_col = qskm1m[:, tile]
        kco2typ = veg_table.gather("nco2typ", kvtype).to(device=device)
        inp = FarquharInputs(
            ptskm=skin_col,
            ptm=forcing.t,
            ptsoil=skin_col,             # TODO: thread PTSAM1M(:,3) for exact acclimation T.
            paphm=forcing.p_surf,
            pqm=forcing.q,
            pqsurf=qs_col,
            pqs=qs_col,
            pco2=forcing.co2,
            lai=lai,
            kvtype=kvtype,
            kco2typ=kco2typ,
            pf2=pf2,
            ldland=land_mask,
            psrfd=ssrfl_grid,
        )
        fq = _farquhar(inp, veg_table, consts)
        pwet = 1.0 / torch.clamp(fq.pgstot, min=rcondctmin)
        return pwet, fq

    pwetl, fq_low = _tile_farquhar(
        TILE_LOW_VEG_DRY, ktvl, jarvis["lail"], jarvis["f2l"])
    pweth, fq_high = _tile_farquhar(
        TILE_HIGH_VEG_DRY, ktvh, jarvis["laih"], jarvis["f2h"])
    pweths, fq_high_snow = _tile_farquhar(
        TILE_HIGH_VEG_SNOW, ktvh, jarvis["laih"], jarvis["f2h"])

    # Bare-soil / urban resistances are not replaced by COTWORESTRESS
    # (vsurf:687).  Diagnostics keep the low-veg Farquhar fields for
    # backward compatibility with SurfDiag.
    return {
        "pwetl": pwetl,
        "pweth": pweth,
        "pweths": pweths,
        "pwetb": jarvis["pwetb"],
        "pwetlu": jarvis["pwetlu"],
        "farquhar_pgstot": fq_low.pgstot,
        "farquhar_pgpp":   fq_low.pgpp,
        "farquhar_prd":    fq_low.prd,
        "farquhar_pan":    fq_low.pan,
        "farquhar_pgstot_high": fq_high.pgstot,
        "farquhar_pgstot_high_snow": fq_high_snow.pgstot,
    }


__all__ = [
    "jarvis_resistance",
    "resistance_to_alpha",
    "vevap",
    "VevapOutputs",
    "build_tile_alpha",
    "build_tile_vevap",
    "farquhar_resistance",
    "ZEPSF3",
    "RCLU",
    "RSMIN_B",
    "PWET_INF",
    "RCEPSW",
]
