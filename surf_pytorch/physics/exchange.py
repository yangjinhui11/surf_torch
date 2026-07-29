"""Monin-Obukhov surface-layer exchange coefficients (Fortran ``VEXCS``).

Direct translation of ``surf/module/vexcs_mod.F90`` (48r1).  Solves the
implicit relation between the bulk Richardson number and the stability
parameter ``eta = z/L`` iteratively, then evaluates the scaled transfer
coefficients::

    PCFM = rho * Cm * |U|     [kg/m^2/s]
    PCFH = rho * Ch * |U|     [kg/m^2/s]
    PCFQ = rho * Cq * |U|     [kg/m^2/s]
    PKH  = Ch * |U|           [m/s]
    PKC  = Cc * |U|           [m/s]   (tracer, between lev and z0m)

The algorithm (Beljaars 1990) has four sections, transcribed in order:

1. *Basic quantities* (sec. 2 of vexcs_mod.F90): the bulk Richardson
   number ``ZRICLS``, the effective wind-shear ``ZDU2`` (incl. the W*
   convective velocity scale), the neutral log-profile ratios and the
   ``Z1DZ1D`` factor that appears in the back-substitution.
2. *Initial guess* (sec. 3.1): if ``ldinit`` is True the routine builds
   its own two-point initial guess -- the RITBL lookup for stable Ri,
   the simple log-ratio for unstable.  Otherwise it uses the previous
   step's ``PZDL``.
3. *Secant iteration* (sec. 3.2-3.3): one back-substitution produces a
   second approximation, then ``kitt`` linear-interpolation (Newton with
   the derivative approximated from successive iterates) iterations.
   A sign-flip safety keeps the iterates on the right side of zero.
4. *Transfer coefficients* (sec. 4): a very-stable cap at
   ``RCHBHDL * L`` truncates the profile to a finite NBL height, then
   the scaled transfer coefficients follow from the integrated PSI
   functions.

Differentiability
-----------------
Every branch (stable vs unstable, capping region) is evaluated with
``torch.where`` so the routine is end-to-end differentiable under
``enable_grad``.  Off-branch contributions are zero -- matching the
piecewise Fortran semantics.  The RITBL initial-guess is detached
(it is a constant table, never a function of the inputs).

Layout
------
All inputs/outputs are ``(Ngrid,)`` -- ``VEXCS`` is grid-mean.  The
per-tile tiling in the upstream driver (``VDFEXCS``) sits above this
kernel and is not implemented here; the port's SEB stage consumes the
grid-mean coefficients broadcast across tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..constants import ConstantBundle
from ..tables import RitblTable
from ..tables.ritbl import (
    JPRITBL,
    RIMAX,
    RCHETA,
    RCHETB,
    RCHBA,
    RCHBB,
    RCHBC,
    RCHBD,
    RCHB23A,
    RCHBBCD,
    RCHBCD,
    RCDHALF,
    RCDHPI2,
)


# --------------------------------------------------------------------------- #
# Constants from YOS_EXC / YOS_EXCS that are not in PhysicalConstants.
# --------------------------------------------------------------------------- #
# Wind-speed floor for ZDU2 (REPDU2 in YOS_EXC). OpenIFS SURF
# initialises this as (0.1 m/s)^2 in suvexc_mod.F90; using a much
# smaller floor over-amplifies stable Richardson numbers in calm cases.
REPDU2 = 1.0e-2

# von Karman constant (RKAP in YOS_EXC).  48r1 default.
RKAP = 0.4

# Boundary-layer height used to scale the W* contribution (RPARZI).
RPARZI = 1000.0

# Capping ratio for very-stable stratification: the column is truncated
# to RCHBHDL * L (RCHBHDL = 5).  From YOS_EXCS.
RCHBHDL = 5.0

# Iteration count for the secant iteration (KITT).  The upstream
# SURFEXCDRIVER call uses 3 by default when LDINIT=.T.
DEFAULT_KITT = 3


# --------------------------------------------------------------------------- #
# PSI stability functions (statement functions in fcsvdfs.h).
#
# These are vectorised torch translations of the JPRD (double) statement
# functions.  Each takes an ``eta`` tensor that may contain both stable
# (positive) and unstable (negative) values; the calling site selects
# the branch via ``torch.where``.
# --------------------------------------------------------------------------- #
def psimu(eta: torch.Tensor) -> torch.Tensor:
    """Unstable momentum PSI (Dyer-Hicks).

    ``PSIMU(eta) = log((1 + sqrt(sqrt(1 - 16*eta)))^2
                       * (1 + sqrt(1 - 16*eta)) / 8)
                   - 2 * atan(sqrt(sqrt(1 - 16*eta))) + RCDHPI2``
    """
    z = torch.clamp(1.0 - RCDHALF * eta, min=0.0)   # protect sqrt arg
    if torch.is_grad_enabled():
        z = torch.clamp(z, min=torch.finfo(z.dtype).eps)
    sqrt_z = torch.sqrt(z)
    fourth_z = torch.sqrt(sqrt_z)
    # NB: the upstream statement function writes ``sqrt(1-16*eta)`` here,
    # which mathematically equals ``sqrt_z``.  We must reuse the *clamped*
    # ``sqrt_z`` (not a fresh ``sqrt(1-16*eta)``) so that on the off-branch
    # -- i.e. when ``eta > 1/16`` and this routine is masked out by
    # ``_psi_m``'s ``torch.where`` -- the unselected branch stays finite.
    # An unclamped ``sqrt`` of a negative argument yields NaN in the forward
    # pass, and although ``where`` masks the output, the NaN propagates
    # through the backward pass (``0 * nan = nan``), corrupting gradients of
    # callers that feed stable (``eta > 0``) columns into the kernel (e.g.
    # the per-tile VEXCS path, where small-z0 water/ice tiles push ``z/L``
    # past ``1/16``).
    inner = sqrt_z
    return (
        torch.log((1.0 + fourth_z) ** 2 * (1.0 + inner) * 0.125)
        - 2.0 * torch.atan(fourth_z)
        + RCDHPI2
    )


def psihu(eta: torch.Tensor) -> torch.Tensor:
    """Unstable heat/moisture PSI (Dyer-Hicks).

    ``PSIHU(eta) = 2 * log((1 + sqrt(1 - 16*eta)) / 2)``
    """
    arg = torch.clamp(1.0 - RCDHALF * eta, min=0.0)
    # The selected unstable branch has ``arg > 0``. On the unselected
    # stable branch, autograd still differentiates through this expression
    # and sqrt(0) has an infinite derivative. The epsilon is used only
    # while constructing a gradient graph, preserving operational forward
    # values and the Fortran calculation.
    if torch.is_grad_enabled():
        arg = torch.clamp(arg, min=torch.finfo(arg.dtype).eps)
    inner = torch.sqrt(arg)
    return 2.0 * torch.log((1.0 + inner) * 0.5)


def psims(eta: torch.Tensor) -> torch.Tensor:
    """Stable momentum PSI (Holtslag-de Bruin).

    ``PSIMS(eta) = -RCHBB*(eta - RCHBCD)*exp(-RCHBD*eta)
                   - RCHBA*eta - RCHBBCD``
    """
    return (
        -RCHBB * (eta - RCHBCD) * torch.exp(-RCHBD * eta)
        - RCHBA * eta
        - RCHBBCD
    )


def psihs(eta: torch.Tensor) -> torch.Tensor:
    """Stable heat/moisture PSI (Holtslag-de Bruin with critical-Ri cap).

    ``PSIHS(eta) = -RCHBB*(eta - RCHBCD)*exp(-RCHBD*eta)
                   - (1 + RCHB23A*eta)*sqrt(1 + RCHB23A*eta)
                   - RCHBBCD + 1``
    """
    inner = 1.0 + RCHB23A * eta
    return (
        -RCHBB * (eta - RCHBCD) * torch.exp(-RCHBD * eta)
        - inner * torch.sqrt(torch.clamp(inner, min=0.0))
        - RCHBBCD + 1.0
    )


def _psi_m(eta: torch.Tensor) -> torch.Tensor:
    """Momentum PSI on a mixed-sign eta (branch by sign)."""
    return torch.where(eta > 0.0, psims(eta), psimu(eta))


def _psi_h(eta: torch.Tensor) -> torch.Tensor:
    """Heat/moisture PSI on a mixed-sign eta (branch by sign)."""
    return torch.where(eta > 0.0, psihs(eta), psihu(eta))


# --------------------------------------------------------------------------- #
# Output container
# --------------------------------------------------------------------------- #
@dataclass
class ExchangeOutputs:
    """Scaled Monin-Obukhov transfer coefficients (VEXCS outputs).

    All fields have shape ``(Ngrid,)``.  ``pcfm``/``pcfh``/``pcfq`` are
    the rho-scaled coefficients that feed the SEB; ``pkh``/``pkc`` are
    the unscaled (no rho) heat / tracer exchange velocities.
    """

    pcfm: torch.Tensor     # rho*Cm*|U|  [kg/m^2/s]
    pcfh: torch.Tensor     # rho*Ch*|U|  [kg/m^2/s]
    pcfq: torch.Tensor     # rho*Cq*|U|  [kg/m^2/s]
    pkh: torch.Tensor      # Ch*|U|      [m/s]
    pkc: torch.Tensor      # Cc*|U|      [m/s]
    pzdl: torch.Tensor     # z/L at lowest model level


# --------------------------------------------------------------------------- #
# VEXCS
# --------------------------------------------------------------------------- #
def vexcs(
    # lowest-level state
    pu: torch.Tensor,            # (Ngrid,)  u at lowest level
    pv: torch.Tensor,            # (Ngrid,)  v at lowest level
    pt: torch.Tensor,            # (Ngrid,)  T at lowest level [K]
    pq: torch.Tensor,            # (Ngrid,)  q at lowest level [kg/kg]
    paphms: torch.Tensor,        # (Ngrid,)  surface pressure [Pa]
    pgeomlev: torch.Tensor,      # (Ngrid,)  geopotential at lowest level [m2/s2]
    pcptgzlev: torch.Tensor,     # (Ngrid,)  dry-static-energy at lowest level [m2/s2]
    pcpts: torch.Tensor,         # (Ngrid,)  dry-static-energy at surface   [m2/s2]
    pqsam: torch.Tensor,         # (Ngrid,)  surface specific humidity     [kg/kg]
    pz0mm: torch.Tensor,         # (Ngrid,)  momentum roughness length [m]
    pz0hm: torch.Tensor,         # (Ngrid,)  heat roughness length      [m]
    pz0qm: torch.Tensor,         # (Ngrid,)  moisture roughness length [m]
    pzdl0: Optional[torch.Tensor],  # (Ngrid,) initial guess z/L (None => ldinit)
    pbuom: torch.Tensor,         # (Ngrid,)  surface buoyancy flux
    pucurr: torch.Tensor,        # (Ngrid,)  ocean current u [m/s]
    pvcurr: torch.Tensor,        # (Ngrid,)  ocean current v [m/s]
    pi10fgcv: torch.Tensor,      # (Ngrid,)  convective gust [m/s]
    consts: ConstantBundle,
    ritbl: RitblTable,
    *,
    kitt: int = DEFAULT_KITT,
    ldinit: bool = True,
    repdu2: float = REPDU2,
    rkap: float = RKAP,
    rparzi: float = RPARZI,
    rchbhdl: float = RCHBHDL,
) -> ExchangeOutputs:
    """Compute Monin-Obukhov exchange coefficients (port of VEXCS).

    The keyword-only signature matches the upstream ``VEXCS`` dummy
    argument list one-for-one; constants that the upstream pulls from
    ``YDCST`` / ``YDEXC`` are taken from ``consts`` and the defaults at
    the top of this module.

    Args
    ----
    pzdl0:
        Previous-step ``z/L`` initial guess.  If ``None`` or if
        ``ldinit`` is True the routine builds its own first guess from
        the RITBL lookup / log-ratio (sec. 3.1, ``LDINIT=.T.`` branch).
    kitt:
        Number of secant iterations after the back-substitution.  The
        upstream default is 3 (cold start) / 0..1 (warm start).
    ldinit:
        Forces the internal first-guess path even when ``pzdl0`` is
        provided; mirrors the upstream ``LDINIT`` flag.
    """
    # ------------------------------------------------------------------ #
    # 1. Constants and bookkeeping
    # ------------------------------------------------------------------ #
    # Use 0-d tensors directly (not .item()) so torch.compile can trace.
    rd = consts.rd
    retv = consts.retv
    rg = consts.g
    rvtmp2 = consts.rvtmp2

    # Upstream section-1 scalars (vexcs_mod.F90 183-189).
    zcon1 = rvtmp2 - retv       # contribution of humidity to buoyancy
    zcon2 = 2.0 / 3.0           # exponent for (W*)^2
    zcon3 = rkap ** 2           # kappa^2 in neutral log-profiles
    zipbl = rparzi              # PBL height for W* scaling

    # ------------------------------------------------------------------ #
    # 2. Basic quantities: W*, ZDU2, ZRICLS, log-profile ratios.
    # ------------------------------------------------------------------ #
    # Convective velocity scale (W*)^2 -- zero when buoyancy flux <= 0
    # (stable/neutral).  Buoyancy flux is *positive* upward in IFS, so a
    # negative PBUOM means stable stratification.
    pos_buom = torch.clamp(pbuom, min=0.0)
    wstar_base = pos_buom * zipbl
    if torch.is_grad_enabled():
        # ``x**(2/3)`` has an undefined derivative at x=0. The Fortran
        # branch remains exactly zero; this only makes the inactive branch
        # well-defined for PyTorch autograd.
        wstar_base = torch.clamp(wstar_base, min=torch.finfo(wstar_base.dtype).eps)
    zwst2 = torch.where(
        pbuom > 0.0,
        wstar_base ** zcon2,
        torch.zeros_like(pbuom),
    )

    # Effective wind-shear squared at the surface (vexcs 207-208).
    zdu2 = torch.clamp(
        (pu - pucurr) ** 2 + (pv - pvcurr) ** 2 + zwst2,
        min=repdu2,
    )

    # Bulk Richardson number ingredients (vexcs 209-212).
    zdroro = (
        2.0 * (pcptgzlev - pcpts)
        / (pcptgzlev + pcpts - pgeomlev)
        - zcon1 * (pq - torch.clamp(pqsam, max=0.1))
    )
    zricls = pgeomlev * zdroro / zdu2

    # Neutral log-profile ratios (vexcs 217-224).  ZNLEV is the height
    # of the lowest model level above the displacement (z0m).
    znlev = pgeomlev / rg + pz0mm
    z1dz0m = znlev / pz0mm
    z1dz0h = znlev / pz0hm
    z1dz0q = znlev / pz0qm
    z1dz1d = z1dz0m / (z1dz0m - 1.0)
    zxlnm = torch.log(z1dz0m)
    zxlnh = torch.log(z1dz0h)
    zxlnq = torch.log(z1dz0q)

    # Reciprocals used by the back-substitution (vexcs 294-297).
    z1dz0md = 1.0 / z1dz0m
    z1dz0hd = 1.0 / z1dz0h

    # ------------------------------------------------------------------ #
    # 3.1 First / second guess
    # ------------------------------------------------------------------ #
    if ldinit or pzdl0 is None:
        # Stable branch: interpolate the RITBL.
        zrib1 = zricls * zxlnm / zxlnh
        zeta1_ritbl = ritbl.initial_guess(zrib1)
        # Unstable branch: simple log-ratio (vexcs 321).
        zeta1_unstable = zrib1
        zeta1 = torch.where(zricls > 0.0, zeta1_ritbl, zeta1_unstable)

        # Second guess: multiply by the log-ratio factor (vexcs 327).
        zeta2 = zeta1 * zxlnm * z1dz1d
    else:
        zeta2 = pzdl0.clone()

    # ------------------------------------------------------------------ #
    # 3.2 Back-substitution: produces the second iterate (zeta3) and the
    #     first residual (zf2).
    # ------------------------------------------------------------------ #
    # PSI at the lowest level (zeta2) and at the roughness length
    # (zeta2 * z0 / znlev = zeta2 / z1dz0*).
    zprh1 = _psi_h(zeta2)
    zprh0 = _psi_h(zeta2 * z1dz0hd)
    zprm1 = _psi_m(zeta2)
    zprm0 = _psi_m(zeta2 * z1dz0md)

    zprh = zxlnh - zprh1 + zprh0
    zprm = zxlnm - zprm1 + zprm0
    zhu = zprm ** 2 * z1dz1d / zprh
    zeta3 = zricls * zhu
    zf2 = zricls - zeta2 / zhu

    # ------------------------------------------------------------------ #
    # 3.3 KITT secant iterations (vexcs 369-406).
    # ------------------------------------------------------------------ #
    for _ in range(kitt):
        zprh1 = _psi_h(zeta3)
        zprh0 = _psi_h(zeta3 * z1dz0hd)
        zprm1 = _psi_m(zeta3)
        zprm0 = _psi_m(zeta3 * z1dz0md)

        zprh = zxlnh - zprh1 + zprh0
        zprm = zxlnm - zprm1 + zprm0
        zf3 = zricls - zeta3 * zprh / (z1dz1d * zprm ** 2)

        # Secant step (vexcs 394).  When the residual difference is
        # negligible, hold the previous iterate (vexcs 396).
        denom = zf3 - zf2
        tiny = torch.full_like(denom, 1.0e-25)
        step = torch.where(
            torch.abs(denom) > 1.0e-25,
            zf2 * (zeta3 - zeta2) / torch.where(
                torch.abs(denom) > 1.0e-25, denom, tiny
            ),
            torch.zeros_like(denom),
        )
        zeta_new = torch.where(
            torch.abs(denom) > 1.0e-25,
            zeta2 - step,
            zeta3,
        )

        # Sign-flip safety: if the secant overshoots past zero, halve
        # the new iterate toward zeta3 (vexcs 401).
        zeta_new = torch.where(
            zeta_new * zeta3 < 0.0,
            zeta3 * 0.5,
            zeta_new,
        )

        # Roll the iterates forward (vexcs 402-404).
        zeta2 = zeta3
        zf2 = zf3
        zeta3 = zeta_new

    # ------------------------------------------------------------------ #
    # 4. Transfer coefficients.
    # ------------------------------------------------------------------ #
    # Very-stable cap: assume maximum NBL height = RCHBHDL * L and
    # uniform profiles from z0 to znlev (vexcs 424-434).
    over_cap = zeta3 > rchbhdl
    zl_safe = znlev / torch.where(zeta3 != 0.0, zeta3, torch.ones_like(zeta3))
    znlev_cap = rchbhdl * zl_safe + torch.maximum(pz0mm, pz0qm)
    zeta3_cap = znlev_cap / zl_safe

    z1dz0m_c = torch.where(over_cap, znlev_cap / pz0mm, z1dz0m)
    z1dz0h_c = torch.where(over_cap, znlev_cap / pz0hm, z1dz0h)
    z1dz0q_c = torch.where(over_cap, znlev_cap / pz0qm, z1dz0q)
    zxlnm_c = torch.where(over_cap, torch.log(z1dz0m_c), zxlnm)
    zxlnh_c = torch.where(over_cap, torch.log(z1dz0h_c), zxlnh)
    zxlnq_c = torch.where(over_cap, torch.log(z1dz0q_c), zxlnq)
    zeta3_c = torch.where(over_cap, zeta3_cap, zeta3)

    z1dz0md_c = 1.0 / z1dz0m_c
    z1dz0hd_c = 1.0 / z1dz0h_c
    z1dz0qd_c = 1.0 / z1dz0q_c

    # Final PSI evaluation at the (possibly capped) iterate.
    zprm1 = _psi_m(zeta3_c)
    zprm0 = _psi_m(zeta3_c * z1dz0md_c)
    zprh1 = _psi_h(zeta3_c)
    zprh0 = _psi_h(zeta3_c * z1dz0hd_c)
    zprq0 = _psi_h(zeta3_c * z1dz0qd_c)
    zprc0 = _psi_h(zeta3_c * z1dz0md_c)

    zprm = zxlnm_c - zprm1 + zprm0
    zprh = zxlnh_c - zprh1 + zprh0
    zprq = zxlnq_c - zprh1 + zprq0
    zprc = zxlnm_c - zprh1 + zprc0

    rho = paphms / (rd * pt * (1.0 + retv * pq))
    zuabs = torch.sqrt(zdu2)
    zaux2 = zuabs * rho * zcon3

    pcfm = zaux2 / (zprm ** 2)
    pcfh = zaux2 / (zprm * zprh)
    pcfq = zaux2 / (zprm * zprq)
    pkh = pcfh / rho
    pkc = zuabs * zcon3 / (zprm * zprc)

    return ExchangeOutputs(
        pcfm=pcfm, pcfh=pcfh, pcfq=pcfq,
        pkh=pkh, pkc=pkc, pzdl=zeta3,
    )


__all__ = [
    "ExchangeOutputs",
    "vexcs",
    "psimu", "psihu", "psims", "psihs",
    "REPDU2", "RKAP", "RPARZI", "RCHBHDL", "DEFAULT_KITT",
]

def vexcss(
    pu: torch.Tensor, pv: torch.Tensor, pt: torch.Tensor, pq: torch.Tensor,
    paphms: torch.Tensor, pgeomlev: torch.Tensor, pcptgzlev: torch.Tensor,
    pcpts: torch.Tensor, pqsam: torch.Tensor, pz0mm: torch.Tensor,
    pz0hm: torch.Tensor, pz0qm: torch.Tensor, pbuom: torch.Tensor,
    consts: ConstantBundle, ptmst: torch.Tensor, rvdifts: float = 1.0,
) -> ExchangeOutputs:
    """Offline OSM ``VEXCSS`` exchange coefficients.

    Fortran ``VEXCSS`` forms C-star coefficients already scaled by
    ``RG * PTMST * PRVDIFTS`` for the VDF driver.  The public Python
    exchange contract is instead the physical ``rho*C*|U|`` coefficient,
    shared with :func:`vexcs`; the VDF scaling is applied once later when
    building ``DiffusionInputs``.
    """
    rlpbb = 5.0
    rlpcc = 5.0
    rlpdd = 5.0
    g = consts.g
    rd = consts.rd
    retv = consts.retv
    rvtmp2 = consts.rvtmp2
    zdu2 = torch.maximum(
        pu.square() + pv.square()
        + torch.where(pbuom > 0.0, (pbuom * RPARZI).clamp_min(0.0).pow(2.0 / 3.0), torch.zeros_like(pbuom)),
        torch.full_like(pu, REPDU2),
    )
    qsam = torch.minimum(pqsam, torch.full_like(pqsam, 0.1))
    zdiv1 = 1.0 / (pcptgzlev + pcpts - pgeomlev)
    zdro = 2.0 * (pcptgzlev - pcpts) * zdiv1 - (rvtmp2 - retv) * (pq - qsam)
    zri = pgeomlev * zdro / zdu2
    znlev = pgeomlev / g + pz0mm
    lnm = torch.log(znlev / pz0mm)
    lnh = torch.log(znlev / pz0hm)
    lnq = torch.log(znlev / pz0qm)
    cdnm = 1.0 / lnm.square()
    cdnh = 1.0 / (lnm * lnh)
    cdnq = 1.0 / (lnm * lnq)
    stable = zri > 0.0
    z2s_st = torch.sqrt(1.0 + rlpdd * zri)
    fm_st = cdnm * z2s_st / (z2s_st + 2.0 * rlpbb * zri)
    fh_st = cdnh / (1.0 + 3.0 * rlpbb * zri * z2s_st)
    fq_st = cdnq / (1.0 + 3.0 * rlpbb * zri * z2s_st)
    z2s_un = torch.sqrt(torch.maximum(-zri, torch.full_like(zri, 1.0e-3)))
    zcm = rlpcc * torch.sqrt(znlev / pz0mm)
    zch = rlpcc * torch.sqrt(znlev / pz0hm)
    zcq = rlpcc * torch.sqrt(znlev / pz0qm)
    zrk3b = RKAP**2 * 3.0 * rlpbb
    fm_un = cdnm * (1.0 - 2.0 * rlpbb * zri / (1.0 + zrk3b * cdnm * zcm * z2s_un))
    fh_un = cdnh * (1.0 - 3.0 * rlpbb * zri / (1.0 + zrk3b * cdnh * zch * z2s_un))
    fq_un = cdnq * (1.0 - 3.0 * rlpbb * zri / (1.0 + zrk3b * cdnq * zcq * z2s_un))
    # Fortran VEXCSS computes
    #   ZAUX2 = (PRVDIFTS*PTMST*RG) * rho * |U| * RKAP**2
    # and multiplies PCFM/PCFH/PCFQ by ZAUX2.  The rest of this port
    # expects physical rho*C*|U| and applies the VDF scaling exactly once
    # in model.py, so undo the C-star time/gravity scaling here.
    aux = (rvdifts * ptmst * g / rd * paphms) / (pt * (1.0 + retv * pq))
    cstar_scale = aux * torch.sqrt(zdu2) * RKAP**2
    scale = cstar_scale / (rvdifts * ptmst * g)
    pcfm = scale * torch.where(stable, fm_st, fm_un)
    pcfh = scale * torch.where(stable, fh_st, fh_un)
    pcfq = scale * torch.where(stable, fq_st, fq_un)
    zero = torch.zeros_like(pcfm)
    return ExchangeOutputs(pcfm=pcfm, pcfh=pcfh, pcfq=pcfq, pkh=zero, pkc=zero, pzdl=zero)
