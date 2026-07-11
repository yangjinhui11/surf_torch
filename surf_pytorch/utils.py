"""Thermodynamic helpers and differentiable utilities.

This module is a direct translation of two Fortran include files:

* ``surf/offline/function/fcttre.h`` -- statement functions
  ``FOEDELTA``, ``FOEALFA``, ``FOEEW``, ``FOEEWM``, ``FOEDE(SU)``,
  ``FOEDEM``, ``FOELH(M)``, ``FOELDCP(M)``.
* the in-line saturation diagnostics scattered through the surf package
  (e.g. ``vsurf_mod.F90`` lines 280/386-388) that produce the saturation
  specific humidity ``qsat = e_s / (p * (1 - RETV*q_tilde))``.

Everything here is *element-wise* over tensors and respects the
``differentiable`` flag of the calling module: lookup-style switches
(``foedelta``) are implemented with ``torch.where`` so that the gradient
graph stays piecewise differentiable, and the actual ``min/max/sign``
switches happen with detached constants where Fortran would branch.

The qsat definition follows the surf-package convention (also used by
``abl`` and the offline driver):

    ZQS   = FOEEW(T) / P                 ! "partial-pressure" q
    ZCOR  = 1.0 / (1.0 - RETV * ZQS)
    QSAT  = ZQS * ZCOR

which is the same as

    QSAT  = FOEEW(T) / (P - RETV * FOEEW(T))

since ``FOEEW`` already carries the ``R2ES`` factor.  Note: IFS surface
uses ``FOEEW`` (no mixed phase), NOT ``FOEEWM`` -- we expose both for
completeness but the model default path is ``foeew``.
"""

from __future__ import annotations

from typing import Optional

import torch

from .constants import ConstantBundle


# --------------------------------------------------------------------------- #
# Phase selectors (fcttre.h FOEDELTA / FOEALFA)
# --------------------------------------------------------------------------- #
def foedelta(t: torch.Tensor, c: ConstantBundle) -> torch.Tensor:
    """Water/ice phase selector (no mixed phase).

    ``fcttre.h``::

        FOEDELTA(PTARE) = MAX(0.0_JPRB, SIGN(1.0_JPRB, PTARE-RTT))

    Returns 1.0 for water (T >= RTT) and 0.0 for ice (T < RTT), with a
    branch at exactly ``RTT`` matching ``SIGN(1.0, 0.0) == 0.0`` hence
    ``MAX(0,0) == 0`` -- i.e. ice at the triple point.  Implemented with
    ``torch.where`` so the derivative is exactly zero on the boundary,
    matching the statement-function semantics.
    """
    diff = t - c.rtt
    sign = torch.where(diff > 0, torch.ones_like(t), torch.where(diff < 0, -torch.ones_like(t), torch.zeros_like(t)))
    return torch.clamp(sign, min=0.0)


def foealfa(t: torch.Tensor, c: ConstantBundle) -> torch.Tensor:
    """Mixed-phase fraction (fcttre.h FOEALFA).

    ``fcttre.h``::

        FOEALFA(PTARE) = MIN(1.0_JPRB,
            ((MAX(RTICE, MIN(RTWAT, PTARE)) - RTICE) * RTWAT_RTICE_R)**2)

    1.0 above RTWAT (water), 0.0 below RTICE (ice), quadratic ramp
    in-between.  Implemented with clamps so the gradient is well-defined
    away from the breakpoints.
    """
    clamped = torch.clamp(t, min=c.rtice, max=c.rtwat)
    ratio = (clamped - c.rtice) * c.rtwat_rtice_r
    return torch.clamp(ratio * ratio, max=1.0)


# --------------------------------------------------------------------------- #
# Saturation vapour pressure and its derivatives
# --------------------------------------------------------------------------- #
def foeew(t: torch.Tensor, c: ConstantBundle) -> torch.Tensor:
    """Saturation vapour pressure (Pa), no mixed phase (fcttre.h FOEEW).

    ``fcttre.h``::

        FOEEW(PTARE) = R2ES * EXP(
            (R3LES*FOEDELTA + R3IES*(1-FOEDELTA)) * (PTARE-RTT)
          / (PTARE - (R4LES*FOEDELTA + R4IES*(1-FOEDELTA))))

    FOEDELTA=1 (water) selects the (R3LES,R4LES) coefficients,
    FOEDELTA=0 (ice) selects (R3IES,R4IES).
    """
    delta = foedelta(t, c)
    r3 = c.r3les * delta + c.r3ies * (1.0 - delta)
    r4 = c.r4les * delta + c.r4ies * (1.0 - delta)
    arg = r3 * (t - c.rtt) / (t - r4)
    return c.r2es * torch.exp(arg)


def foeewm(t: torch.Tensor, c: ConstantBundle) -> torch.Tensor:
    """Saturation vapour pressure (Pa), with mixed phase (fcttre.h FOEEWM).

    ``fcttre.h``::

        FOEEWM(PTARE) = R2ES * (
            FOEALFA(PTARE) * EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))
          + (1-FOEALFA(PTARE)) * EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES)))
    """
    alfa = foealfa(t, c)
    water = torch.exp(c.r3les * (t - c.rtt) / (t - c.r4les))
    ice = torch.exp(c.r3ies * (t - c.rtt) / (t - c.r4ies))
    return c.r2es * (alfa * water + (1.0 - alfa) * ice)


def foede(t: torch.Tensor, c: ConstantBundle) -> torch.Tensor:
    """d(es)/es/dT simplified form (fcttre.h FOEDE).

    ``fcttre.h``::

        FOEDE(PTARE) = (FOEDELTA*R5ALVCP + (1-FOEDELTA)*R5ALSCP)
                     / (PTARE - (R4LES*FOEDELTA + R4IES*(1-FOEDELTA)))**2
    """
    delta = foedelta(t, c)
    num = delta * c.r5alvcp + (1.0 - delta) * c.r5alscp
    r4 = c.r4les * delta + c.r4ies * (1.0 - delta)
    return num / (t - r4) ** 2


def foedesu(t: torch.Tensor, c: ConstantBundle) -> torch.Tensor:
    """Alternative FOEDE form used at the surface (fcttre.h FOEDESU).

    ``fcttre.h``::

        FOEDESU(PTARE) = (FOEDELTA*R5LES + (1-FOEDELTA)*R5IES)
                       / (PTARE - (R4LES*FOEDELTA + R4IES*(1-FOEDELTA)))**2

    This is the variant called by the surface driver
    (``surfexcdrivers_ctl_mod.F90`` lines 283-284) when building the
    skin-temperature Jacobian of qsat.
    """
    delta = foedelta(t, c)
    num = delta * c.r5les + (1.0 - delta) * c.r5ies
    r4 = c.r4les * delta + c.r4ies * (1.0 - delta)
    return num / (t - r4) ** 2


def foedem(t: torch.Tensor, c: ConstantBundle) -> torch.Tensor:
    """FOEDE with mixed phase (fcttre.h FOEDEM)."""
    alfa = foealfa(t, c)
    return (
        alfa * c.r5alvcp / (t - c.r4les) ** 2
        + (1.0 - alfa) * c.r5alscp / (t - c.r4ies) ** 2
    )


def foelh(t: torch.Tensor, c: ConstantBundle) -> torch.Tensor:
    """Latent heat (vap/subl) selector, no mixed phase (fcttre.h FOELH)."""
    delta = foedelta(t, c)
    return delta * c.rlvtt + (1.0 - delta) * c.rlstt


def foeldcp(t: torch.Tensor, c: ConstantBundle) -> torch.Tensor:
    """L/cp selector, no mixed phase (fcttre.h FOELDCP)."""
    delta = foedelta(t, c)
    return delta * c.ralvdcp + (1.0 - delta) * c.ralsdcp


# --------------------------------------------------------------------------- #
# Saturation specific humidity and its T-derivative
# --------------------------------------------------------------------------- #
def qsat(
    t: torch.Tensor,
    p: torch.Tensor,
    c: ConstantBundle,
    *,
    mixed_phase: bool = False,
    salinity: "torch.Tensor | float | None" = None,
) -> torch.Tensor:
    """Saturation specific humidity.

    Reproduces the in-line definition used throughout the surf package
    (``vsurf_mod.F90`` 278-282)::

        PQS  = FOEEW(T) / P
        ZCOR = ZSALIN / (1.0 - RETV * PQS)
        QSAT = PQS * ZCOR

    The closed form is ``ZSALIN * es / (P - RETV*es)``.  Pass
    ``mixed_phase=True`` to use ``FOEEWM`` instead of ``FOEEW`` (cloud-
    path convention; the land surface always uses the no-mixed-phase
    form).

    ``salinity`` mirrors the upstream ``ZSALIN`` factor
    (``vsurf_mod.F90`` 262-265): ``0.98`` over the ocean when the
    ``LEOCSA`` namelist switch is active, ``1.0`` everywhere else (the
    offline-driver default).  ``None`` is treated as ``1.0`` so
    pre-existing call sites are unchanged; a tensor is broadcast
    element-wise (use this for per-tile ocean masking).
    """
    zsalin = 1.0 if salinity is None else salinity
    es = foeewm(t, c) if mixed_phase else foeew(t, c)
    zqs = es / p
    zcor = zsalin / (1.0 - c.retv * zqs)
    return zqs * zcor


def dqsat_dt(
    t: torch.Tensor,
    p: torch.Tensor,
    c: ConstantBundle,
    *,
    mixed_phase: bool = False,
    salinity: "torch.Tensor | float | None" = None,
) -> torch.Tensor:
    """Exact derivative ``d(qsat)/dT`` (Pa unchanged).

    This is the mathematically exact derivative of :func:`qsat` wrt T.
    With ``qs = ZSALIN*es/(p - RETV*es)`` we have

        dqs/dT = ZSALIN * p / (p - RETV*es)^2 * des/dT

    where ``des/dT = es * d(ln es)/dT = es * FOEDESU(T)`` (the relative
    derivative of es wrt T; see fcttre.h FOEDESU).  Note: we use FOEDESU
    (not FOEDE) because FOEDE carries an extra L/cp factor used by the
    cp-aware moist thermodynamics elsewhere.

    .. warning::

        This is **NOT** the form the upstream surface driver uses.  The
        real ``PDQS`` (``vsurf_mod.F90`` 284) is the small-RETV
        linearisation reproduced by :func:`dqsat_dt_ifs`; the SEB should
        consume that one to stay bit-faithful to Fortran.  This exact
        form is retained only as a reference / cross-check.
    """
    zsalin = 1.0 if salinity is None else salinity
    es = foeewm(t, c) if mixed_phase else foeew(t, c)
    dlnes_dt = foedem(t, c) if mixed_phase else foedesu(t, c)
    desdt = es * dlnes_dt
    denom = p - c.retv * es
    return zsalin * p / (denom * denom) * desdt


def dqsat_dt_approx(
    t: torch.Tensor,
    p: torch.Tensor,
    c: ConstantBundle,
    *,
    mixed_phase: bool = False,
    salinity: "torch.Tensor | float | None" = None,
) -> torch.Tensor:
    """Small-RETV linearisation of ``dqsat/dT``.

    Kept for backwards compatibility; this is mathematically the same
    family as :func:`dqsat_dt_ifs` but evaluates ``ZCOR`` at the
    *post-salinity* ``qs`` rather than at the upstream ``PQS = es/P``.
    Prefer :func:`dqsat_dt_ifs` for Fortran-faithful SEB inputs.
    """
    zsalin = 1.0 if salinity is None else salinity
    qs = qsat(t, p, c, mixed_phase=mixed_phase, salinity=zsalin)
    zcor = 1.0 / (1.0 - c.retv * qs)
    return qs * zcor * foedesu(t, c)


def dqsat_dt_ifs(
    t: torch.Tensor,
    p: torch.Tensor,
    c: ConstantBundle,
    *,
    mixed_phase: bool = False,
    salinity: "torch.Tensor | float | None" = None,
) -> torch.Tensor:
    """The upstream ``PDQS`` from ``vsurf_mod.F90`` (284).

    Faithful transcription of the surf-package saturation derivative::

        PQS   = FOEEW(T) / P
        ZCOR  = ZSALIN / (1 - RETV * PQS)        ! PQS *before* salinity
        PQS   = PQS * ZCOR
        PDQS  = PQS * ZCOR * FOEDESU(T)

    i.e. ``PDQS = (es/P) * ZCOR**2 * FOEDESU(T)`` with the **same**
    ``ZCOR`` used twice and evaluated at the pre-salinity partial
    pressure ``es/P``.  The two ``ZCOR`` factors come from the Fortran
    applying ``ZCOR`` to ``PQS`` and then *again* in the ``PDQS``
    expression -- a detail the previous :func:`dqsat_dt_approx` got
    subtly wrong (it re-evaluated ``ZCOR`` at the post-salinity ``qs``).

    This is the entry point the SEB (:func:`surf_pytorch.physics.seb`)
    should consume so the skin-temperature Jacobian matches the Fortran
    driver to last-ULP.  It differs from the exact :func:`dqsat_dt` by
    the (small) ``RETV*es`` term dropped from the denominator -- order
    ``RETV * es / p ~ 1e-4``.

    ``salinity`` mirrors ``ZSALIN`` (``vsurf_mod.F90`` 262-265):
    ``0.98`` for the ocean tile under ``LEOCSA=.T.``, ``1.0`` elsewhere.
    """
    zsalin = 1.0 if salinity is None else salinity
    es = foeewm(t, c) if mixed_phase else foeew(t, c)
    dlnes_dt = foedesu(t, c) if not mixed_phase else foedem(t, c)
    zqs0 = es / p                                 # PQS before salinity
    zcor = zsalin / (1.0 - c.retv * zqs0)         # single ZCOR, upstream form
    return zqs0 * zcor * zcor * dlnes_dt


# --------------------------------------------------------------------------- #
# Differentiable helpers (the "differentiable on demand" plumbing)
# --------------------------------------------------------------------------- #
def detach_index(idx: torch.Tensor) -> torch.Tensor:
    """Return an integer index tensor detached from any grad graph.

    Lookup indices (tile selection, soil-class lookup, root-layer
    gather) are non-differentiable by definition; we always detach them
    so the rest of the model can stay under ``enable_grad`` without
    producing a stray "index that requires grad" error.
    """
    return idx.detach().to(dtype=torch.long)


def safe_where(
    cond: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    differentiable: bool,
) -> torch.Tensor:
    """``torch.where`` with explicit no-grad branch when ``differentiable`` is False.

    Most surface branches are piecewise (snow/no-snow, water/ice, stable
    /unstable).  When we are *not* differentiating, the faster path is a
    plain where; when we are, we still use where (gradient is the local
    derivative) but the caller can wrap the whole block in ``enable_grad``.
    The flag exists so call sites document intent and so a future fused
    kernel can swap in a straight masked-select when grad is off.
    """
    if not differentiable:
        with torch.no_grad():
            return torch.where(cond, a, b)
    return torch.where(cond, a, b)


def maybe_no_grad(fn, differentiable: bool):
    """Decorator-style context: run ``fn`` under no_grad unless differentiable.

    Used to wrap whole physics sub-steps so the default fast path never
    builds a graph.  ``fn`` is expected to be a zero-arg callable.
    """
    if differentiable:
        return fn()
    with torch.no_grad():
        return fn()


def scatter_to_tiles(
    per_tile: torch.Tensor,
    tile_frac: torch.Tensor,
) -> torch.Tensor:
    """Tile -> grid aggregation: ``sum_tiles(frac * value)``.

    Args:
        per_tile:  (..., Ngrid, Ntile) tensor of per-tile quantities
        tile_frac: (..., Ngrid, Ntile) tile fractions PFRTI (sum to 1)

    Returns:
        (..., Ngrid) grid-mean quantity.
    """
    return (per_tile * tile_frac).sum(dim=-1)


def expand_as_tiles(
    grid: torch.Tensor,
    ntile: int,
) -> torch.Tensor:
    """Broadcast a (..., Ngrid) quantity to (..., Ngrid, Ntile)."""
    return grid.unsqueeze(-1).expand(*grid.shape, ntile)


def tridiag_solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    *,
    differentiable: bool,
) -> torch.Tensor:
    """Solve ``b_i x_i + a_i x_{i-1} + c_i x_{i+1} = d_i`` for x.

    Land-surface vertical diffusion produces a tridiagonal system along
    the soil/snow vertical axis; the lower-diagonal element ``a[0]`` and
    upper-diagonal element ``c[-1]`` are unused.  Layout is
    ``(Ngrid, Nlev)`` (batched over grid columns).

    Uses ``torch.linalg.solve_triangular`` on the packed-batch LU so the
    solve is autograd-friendly when ``differentiable=True``; otherwise we
    run the same code under ``no_grad`` for speed.

    Note: when we eventually fuse with the implicit surface-energy-
    balance closure, the first row's ``a[0]*x[-1]`` term carries the
    PASL/PBSL lower-boundary coupling -- handle that by overwriting
    ``b[0]``/``d[0]`` before calling this routine.
    """
    if not differentiable:
        with torch.no_grad():
            return _tridiag_solve_core(a, b, c, d)
    return _tridiag_solve_core(a, b, c, d)


def _tridiag_solve_core(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Thomas algorithm, batched over leading dims, autograd-safe.

    Implemented as a recurrence along the level axis (``-1``) so we never
    materialise a dense ``(Nlev, Nlev)`` matrix.  Forward sweep +
    backward substitution; numerically stable for diagonally-dominant
    diffusion systems (which all surf-package systems are by
    construction).
    """
    nlev = b.shape[-1]
    # Work on clones so input tensors are untouched.
    cp = torch.empty_like(c)
    dp = torch.empty_like(d)
    # Forward elimination.
    cp[..., 0] = c[..., 0] / b[..., 0]
    dp[..., 0] = d[..., 0] / b[..., 0]
    for k in range(1, nlev):
        denom = b[..., k] - a[..., k] * cp[..., k - 1]
        # The last cp is never read by the back-substitution; we zero it
        # so that a stray NaN/inf in c[..., -1] cannot poison the result.
        if k < nlev - 1:
            cp[..., k] = c[..., k] / denom
        else:
            cp[..., k] = torch.zeros_like(c[..., k])
        dp[..., k] = (d[..., k] - a[..., k] * dp[..., k - 1]) / denom
    # Back substitution.
    x = torch.empty_like(d)
    x[..., -1] = dp[..., -1]
    for k in range(nlev - 2, -1, -1):
        x[..., k] = dp[..., k] - cp[..., k] * x[..., k + 1]
    return x


__all__ = [
    # phase selectors
    "foedelta",
    "foealfa",
    # saturation pressure
    "foeew",
    "foeewm",
    "foede",
    "foedesu",
    "foedem",
    "foelh",
    "foeldcp",
    # qsat family
    "qsat",
    "dqsat_dt",
    "dqsat_dt_approx",
    "dqsat_dt_ifs",
    # differentiable plumbing
    "detach_index",
    "safe_where",
    "maybe_no_grad",
    "scatter_to_tiles",
    "expand_as_tiles",
    "tridiag_solve",
]
