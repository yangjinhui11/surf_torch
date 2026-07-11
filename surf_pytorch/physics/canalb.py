"""Urban canyon albedo (``CANALB``, ``canalb_mod.F90`` lines 82-181).

This module ports the ECMWF urban canyon broadband albedo -- the
Masson (2000) shadowing scheme combined with the Harman (2004) matrix
exchange formulation and the Porson (2010) shape factors.  It is the
algorithm ``SURFRAD`` calls at ``surfrad_ctl_mod.F90:520`` to get the
canyon albedo for the urban tile (tile 10).

What the scheme does
--------------------
For each grid column the routine builds three canyon "surfaces" --
sky (1), road (2), wall (3) -- and solves a 3x3 reflection problem to
find the albedo of the canyon as a whole.  The steps are:

1. **Solar partition** (Masson 2000 eq 13-14, lines 88-99).  From the
   solar zenith cosine ``PMU0`` and the canyon aspect ratio ``RHWR``
   (height-to-road-width), compute the fractions of incident SW scattered
   onto the road (``x_r``) and walls (``x_w``)::

       var1   = tan(acos(mu0))                 # limited away from 0
       omega0 = asin(min(1, 1/(var1*RHWR)))
       x_r    = (2/pi)*(omega0 - RHWR*var1*(1 - cos(omega0)))
       x_w    = (1 - x_r)/(2*RHWR)

   ``x_f = 0.3`` is the sky-scattered fraction (line 85, hard-coded).

2. **Shape factors** (Harman 2004 eq 2-5 + Porson 2010 eq 12-14, lines
   101-116).  From ``RHWR`` build the view-factor matrix ``shf`` that
   distributes outgoing flux among the three surfaces.

3. **Reflection matrix** (lines 118-127).  A 3x3 system coupling
   incoming and reflected flux, using the road albedo ``RROAALB`` and
   the wall albedo ``RWALALB``.

4. **Inverse** (lines 130-144).  The reflection matrix is inverted
   analytically by cofactors (reproduced here verbatim, rather than via
   ``torch.linalg.inv``, so the result matches the Fortran bit-for-bit
   when the inputs match).

5. **Reference flux + assembly** (Porson 2010 eq 10, lines 146-170)::

       ref = [RROAALB*(1-x_f)*x_r, RWALALB*(1-x_f)*x_w, x_f]
       OUT = rinv @ ref
       INC = shf @ OUT
       INC(1) += (1-x_f)*x_r - OUT(1)
       INC(2) += (1-x_f)*x_w - OUT(2)
       PCANALB = clamp(1 - (INC(1) + 2*RHWR*INC(2)), 0, 1)

   ``INC(3)`` is computed but never used -- matching the Fortran (the
   wall self-absorption is folded into the ``2*RHWR`` weighting of the
   wall flux).

Inputs / outputs
----------------
``mu0`` is the cosine of the solar zenith angle ``(Ngrid,)``.  The three
material/geometry fields (``rwalalb``, ``rroaalb``, ``rhwr``) are
``(Ngrid,)`` tensors (per-column) or scalars broadcast across the grid;
the upstream defaults are RWALALB=0.6, RROAALB=0.05, RHWR=1.0.  The
output ``pcanalb`` is the broadband canyon albedo ``(Ngrid,)`` in [0,1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class CanalbInputs:
    """Inputs to the urban canyon albedo solve.

    Each field is ``(Ngrid,)`` and broadcasts naturally against the
    others.  Naming mirrors the ``CANALB`` dummy variables / the
    ``YOS_URB`` members so a side-by-side audit is easy.
    """

    # ---- Solar geometry ----
    mu0: torch.Tensor              # PMU0  cosine solar zenith [-]  (Ngrid,)

    # ---- Canyon material albedos (YOS_URB) ----
    rwalalb: torch.Tensor          # RWALALB  wall albedo  [-]       (Ngrid,)
    rroaalb: torch.Tensor          # RROAALB  road albedo  [-]       (Ngrid,)

    # ---- Canyon geometry (YOS_URB) ----
    rhwr: torch.Tensor             # RHWR  height-to-roadwidth ratio [-] (Ngrid,)


@dataclass
class CanalbOutputs:
    """Outputs of the urban canyon albedo solve."""

    pcanalb: torch.Tensor          # PCANALB  canyon albedo [-]      (Ngrid,)


def compute_canalb(
    inp: CanalbInputs,
    *,
    differentiable: bool = True,
) -> CanalbOutputs:
    """Run the urban canyon albedo (``canalb_mod.F90`` 82-181).

    Args
    ----
    inp:
        :class:`CanalbInputs` -- packed solar geometry and canyon
        material / geometry fields.
    differentiable:
        If ``False`` run under ``torch.no_grad()`` for speed.

    Returns
    -------
    :class:`CanalbOutputs` -- the canyon albedo ``PCANALB`` in [0, 1].
    """

    def _core() -> CanalbOutputs:
        return _canalb_core(inp)

    if not differentiable:
        with torch.no_grad():
            return _core()
    return _core()


def _canalb_core(inp: CanalbInputs) -> CanalbOutputs:
    """Internal: assumes the no_grad / enable_grad decision is already made."""
    # ------------------------------------------------------------------ #
    # Scalar constants (Fortran lines 85-86).
    # ------------------------------------------------------------------ #
    x_f = 0.3
    pi = 4.0 * math.atan(1.0)

    mu0 = inp.mu0
    rwalalb = inp.rwalalb
    rroaalb = inp.rroaalb
    rhwr = inp.rhwr

    # ------------------------------------------------------------------ #
    # Solar partition (Masson 2000 eq 13-14, lines 88-99).
    # tan(acos(mu0)); limit away from 0 (line 92-94).
    # ------------------------------------------------------------------ #
    # acos in [0, pi/2] for mu0 in [0,1]; tan blows up at mu0=0.
    var1 = torch.tan(torch.acos(torch.clamp(mu0, min=1e-12, max=1.0)))
    var1 = torch.where(var1 == 0.0, torch.full_like(var1, 1e-4), var1)
    omega0 = 1.0 / (var1 * rhwr)
    omega0 = torch.asin(torch.clamp(omega0, max=1.0))      # line 96-97
    x_r = (2.0 / pi) * (omega0 - rhwr * torch.sin(omega0) * (1.0 - torch.cos(omega0)))
    x_w = (1.0 - x_r) / (2.0 * rhwr)

    # ------------------------------------------------------------------ #
    # Shape factors (Harman 2004 eq 2-5 + Porson 2010 eq 12-14, 101-116).
    # Build the 3x3 view-factor matrix `shf` as (Ngrid, 3, 3).
    # ------------------------------------------------------------------ #
    aa = torch.sqrt(1.0 + rhwr ** 2) - rhwr
    bb = torch.sqrt(1.0 + (1.0 / rhwr) ** 2) - (1.0 / rhwr)
    cc = 1.0 - aa
    dd = (1.0 - bb) / 2.0
    zero = torch.zeros_like(rhwr)
    # Row-major (i,j): shf[i, j]
    shf = torch.stack([
        torch.stack([zero, cc, aa], dim=-1),   # row 1: sky
        torch.stack([dd, bb, dd], dim=-1),     # row 2: road
        torch.stack([aa, cc, zero], dim=-1),   # row 3: wall
    ], dim=-2)                                 # (Ngrid, 3, 3)

    # ------------------------------------------------------------------ #
    # Reflection matrix (lines 118-127).
    # ------------------------------------------------------------------ #
    one = torch.ones_like(rhwr)
    rfl = torch.stack([
        torch.stack([one, -rroaalb * cc, -rroaalb * aa], dim=-1),  # row 1
        torch.stack([-rwalalb * dd, one - rwalalb * bb, -rwalalb * dd], dim=-1),
        torch.stack([zero, zero, one], dim=-1),                    # row 3
    ], dim=-2)                                                     # (Ngrid, 3, 3)

    # ------------------------------------------------------------------ #
    # Inverse by cofactors (Fortran lines 127-135).
    #
    # The Fortran assigns rinv in three column-groups (rinv(1,j)..rinv(3,j)
    # for j=1,2,3), which is the **adjugate** (transposed cofactor matrix)
    # -- the standard closed-form inverse.  We build rinv row-by-row here
    # (each outer stack entry is a row; inner stack entries are columns),
    # matching rinv(i,j) = (-1)^(i+j) * minor(j,i) / det.  The port
    # originally read the Fortran groups as rows, producing the transpose
    # of the inverse; this is the corrected, audited form.  See the
    # TestMatrixInverse / test_reference_value_recomputed_independently
    # tests, which cross-check rinv against torch.linalg.solve.
    # ------------------------------------------------------------------ #
    r11 = rfl[..., 0, 0]; r12 = rfl[..., 0, 1]; r13 = rfl[..., 0, 2]
    r21 = rfl[..., 1, 0]; r22 = rfl[..., 1, 1]; r23 = rfl[..., 1, 2]
    r31 = rfl[..., 2, 0]; r32 = rfl[..., 2, 1]; r33 = rfl[..., 2, 2]

    detinv = 1.0 / (
        r11 * r22 * r33 - r11 * r23 * r32
        - r12 * r21 * r33 + r12 * r23 * r31
        + r13 * r21 * r32 - r13 * r22 * r31
    )

    # Row 1: rinv(1,1)=+C11, rinv(1,2)=-C12, rinv(1,3)=+C13
    # Row 2: rinv(2,1)=-C21, rinv(2,2)=+C22, rinv(2,3)=-C23
    # Row 3: rinv(3,1)=+C31, rinv(3,2)=-C32, rinv(3,3)=+C33
    rinv = torch.stack([
        torch.stack([
             detinv * (r22 * r33 - r23 * r32),    # rinv(1,1) = +minor(1,1)
            -detinv * (r12 * r33 - r13 * r32),    # rinv(1,2) = -minor(1,2)
             detinv * (r12 * r23 - r13 * r22),    # rinv(1,3) = +minor(1,3)
        ], dim=-1),
        torch.stack([
            -detinv * (r21 * r33 - r23 * r31),    # rinv(2,1) = -minor(2,1)
             detinv * (r11 * r33 - r13 * r31),    # rinv(2,2) = +minor(2,2)
            -detinv * (r11 * r23 - r13 * r21),    # rinv(2,3) = -minor(2,3)
        ], dim=-1),
        torch.stack([
             detinv * (r21 * r32 - r22 * r31),    # rinv(3,1) = +minor(3,1)
            -detinv * (r11 * r32 - r12 * r31),    # rinv(3,2) = -minor(3,2)
             detinv * (r11 * r22 - r12 * r21),    # rinv(3,3) = +minor(3,3)
        ], dim=-1),
    ], dim=-2)   # (Ngrid, 3, 3)

    # ------------------------------------------------------------------ #
    # Reference flux (Porson 2010 eq 10, lines 146-149).
    # ------------------------------------------------------------------ #
    ref = torch.stack([
        rroaalb * (1.0 - x_f) * x_r,       # road
        rwalalb * (1.0 - x_f) * x_w,       # wall
        torch.full_like(x_f if torch.is_tensor(x_f) else rhwr, x_f),
    ], dim=-1)                              # (Ngrid, 3)

    # ------------------------------------------------------------------ #
    # Assembly (lines 151-170).
    #   OUT = rinv @ ref          (Ngrid,3)
    #   INC = shf @ OUT           (Ngrid,3)
    # ------------------------------------------------------------------ #
    out = torch.einsum("nij,nj->ni", rinv, ref)
    inc = torch.einsum("nij,nj->ni", shf, out)
    inc = inc.clone()   # avoid in-place op on autograd graph
    inc = torch.stack([
        inc[..., 0] + (1.0 - x_f) * x_r - out[..., 0],
        inc[..., 1] + (1.0 - x_f) * x_w - out[..., 1],
        inc[..., 2],
    ], dim=-1)

    # ------------------------------------------------------------------ #
    # Canyon albedo (line 168-170) + bounds clamp (173-179).
    # ------------------------------------------------------------------ #
    pcanalb = 1.0 - (inc[..., 0] + 2.0 * rhwr * inc[..., 1])
    pcanalb = torch.clamp(pcanalb, min=0.0, max=1.0)

    return CanalbOutputs(pcanalb=pcanalb)


__all__ = [
    "CanalbInputs",
    "CanalbOutputs",
    "compute_canalb",
]
