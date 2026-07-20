"""Offline OSM lower-boundary VDF closure.

This module collects the part of ``VDFMAIN1S`` needed by the standalone
land driver. It deliberately does *not* replace the atmospheric VDF in a
coupled OpenIFS integration: coupled callers receive lower-boundary fluxes
from the atmospheric model.

The offline ``VDFDIFM1S`` source disables its vertical tridiagonal solve.
Together with ``VDFINCR`` this reduces the lowest-level momentum flux to
``tau_u = PCFM * (u - u_current)`` and ``tau_v = PCFM * (v - v_current)``,
where ``PCFM`` is the area-weighted tile exchange coefficient from VEXCS.
``VDFDIFH1S`` remains responsible for tiled sensible and moisture fluxes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OfflineVDFMomentum:
    """Lowest-level VDFMAIN1S momentum-flux outputs."""

    stress_u: torch.Tensor
    stress_v: torch.Tensor


def vdfdifm1s_surface_stress(
    pcfm_tile: torch.Tensor,
    tile_fraction: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    u_current: torch.Tensor | None = None,
    v_current: torch.Tensor | None = None,
) -> OfflineVDFMomentum:
    """Compute offline ``VDFDIFM1S``/``VDFINCR`` surface stress.

    ``pcfm_tile`` is physical ``rho*Cm*|U|``. VDFMAIN1S applies and
    VDFINCR removes the ``g*dt*RVDIFTS`` scaling, leaving this relation.
    """
    if pcfm_tile.ndim != 2 or tile_fraction.shape != pcfm_tile.shape:
        raise ValueError("pcfm_tile and tile_fraction must both be (Ngrid, Ntile)")
    ngrid = pcfm_tile.shape[0]
    if u.shape != (ngrid,) or v.shape != (ngrid,):
        raise ValueError("u and v must have shape (Ngrid,)")

    zeros = torch.zeros_like(u)
    u_current = zeros if u_current is None else u_current.to(dtype=u.dtype, device=u.device)
    v_current = zeros if v_current is None else v_current.to(dtype=v.dtype, device=v.device)
    if u_current.shape != (ngrid,) or v_current.shape != (ngrid,):
        raise ValueError("u_current and v_current must have shape (Ngrid,)")

    pcfm = torch.sum(pcfm_tile * tile_fraction, dim=-1)
    return OfflineVDFMomentum(
        stress_u=pcfm * (u - u_current),
        stress_v=pcfm * (v - v_current),
    )


__all__ = ["OfflineVDFMomentum", "vdfdifm1s_surface_stress"]
