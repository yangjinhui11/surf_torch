"""Monin-Obukhov stability lookup table ``RITBL``.

Faithful translation of ``SUVEXCS`` (``surf/module/suvexcs_mod.F90``)
plus the statement functions from ``surf/function/fcsvdfs.h``.

What the table encodes
----------------------
For stable stratification the bulk-Richardson number ``Ri`` and the
stability parameter ``eta = z/L`` are linked implicitly by::

    Ri = eta * PHIHS(eta) / PHIMS(eta)^2

with the *stable* gradient functions (Holtslag-de Bruin family)::

    PHIMS(eta) = 1 + RCHETA*eta            ; RCHETA = 5
    PHIHS(eta) = (1 + RCHETB*eta)^2        ; RCHETB = 4

``SUVEXCS`` tabulates the inverse function ``eta(Ri)`` on the grid
``Ri = 0 .. RIMAX`` with uniform spacing ``DRITBL = RIMAX/(JPRITBL-1)``
using Newton iteration with a finite-difference Jacobian.

Usage at run time (see ``physics/exchange.py``) is only as an *initial
guess* for the (unstable or stable) PSI-secant iteration; the table is
therefore a constant buffer and its gradient is never needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


JPRITBL = 101
RIMAX = 10.0

# Stable PSI-function constants (suvexcs lines 84-96)
RCHETA = 5.0
RCHETB = 4.0
RCHBA = 1.0
RCHBB = 2.0 / 3.0
RCHBC = 5.0
RCHBD = 0.35
RCHB23A = (2.0 / 3.0) * RCHBA
RCHBBCD = RCHBB * RCHBC / RCHBD
RCHBCD = RCHBC / RCHBD

# Unstable (Dyer-Hicks)
RCDHALF = 16.0
RCDHPI2 = float(2.0 * np.arctan(1.0))

# Louis-simplified-physics constants (suvexcs 137-139)
RLPBB = 5.0
RLPCC = 5.0
RLPDD = 5.0


# --------------------------------------------------------------------------- #
# Stable gradient functions PHIMS / PHIHS  (fcsvdfs.h lines 51-53)
# --------------------------------------------------------------------------- #
def _phims(eta: np.ndarray) -> np.ndarray:
    return 1.0 + RCHETA * eta


def _phihs(eta: np.ndarray) -> np.ndarray:
    return (1.0 + RCHETB * eta) ** 2


def build_ritbl() -> np.ndarray:
    """Return ``RITBL(1:JPRITBL)`` exactly as ``SUVEXCS`` produces it."""
    dritbl = RIMAX / (JPRITBL - 1)
    ritbl = np.zeros(JPRITBL, dtype=np.float64)

    for jjp in range(2, JPRITBL + 1):           # Fortran loop 2..JPRITBL
        zrib = (jjp - 1) * dritbl
        zeta = ritbl[jjp - 2]                    # RITBL(JJP-1) initial guess
        itmax = 5 if zrib < 0.5 else 3

        for _ in range(itmax):
            zeps = (zeta + 1.0) * 0.001
            zetap = zeta + zeps
            zetam = zeta - zeps
            zfxp = zrib - zetap * _phihs(zetap) / _phims(zetap) ** 2
            zfxm = zrib - zetam * _phihs(zetam) / _phims(zetam) ** 2
            zeta = zeta - zeps * (zfxp + zfxm) / (zfxp - zfxm)

        ritbl[jjp - 1] = zeta

    return ritbl


@dataclass(frozen=True)
class RiTableConstants:
    """Scalars exposed alongside ``RITBL`` for the exchange module."""

    jpritbl: int = JPRITBL
    rimax: float = RIMAX
    dritbl: float = RIMAX / (JPRITBL - 1)

    # stable PSI-function constants (mirror suvexcs for self-containment)
    rcheta: float = RCHETA
    rchetb: float = RCHETB
    rchba: float = RCHBA
    rchbb: float = RCHBB
    rchbc: float = RCHBC
    rchbd: float = RCHBD
    rchb23a: float = RCHB23A
    rchbbcd: float = RCHBBCD
    rchbcd: float = RCHBCD

    # unstable
    rcdhalf: float = RCDHALF
    rcdhpi2: float = RCDHPI2


class RitblTable(nn.Module):
    """Constant buffer holding ``RITBL`` and its bookkeeping constants.

    Because the table is used only as an initial guess for the
    M-O iteration, the buffer carries no gradient.
    """

    def __init__(
        self,
        *,
        consts: RiTableConstants | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.consts = consts or RiTableConstants()
        self.dtype = dtype
        self.device = device
        self.register_buffer(
            "ritbl",
            torch.as_tensor(build_ritbl(), dtype=dtype, device=device),
        )
        self.register_buffer(
            "dritbl",
            torch.as_tensor(self.consts.dritbl, dtype=dtype, device=device),
        )

    def initial_guess(self, zrib1: torch.Tensor) -> torch.Tensor:
        """Tabulated ``eta`` initial guess for a stable bulk Ri ``zrib1``.

        Mirrors the use-site in ``vexcs`` (lines 309-322): linear
        interpolation between table entries with end-capping.  Indices
        are detached -- the table is a constant initial guess only.

        ``zrib1`` is clamped to ``[0, RIMAX]`` before lookup so that
        out-of-range Ri values saturate at the first/last table entry
        rather than extrapolating (the stable iteration only runs for
        ``0 <= Ri <= RIMAX``; anything larger is treated as
        ``Ri = RIMAX``).
        """
        dritbl = self.dritbl
        ri = torch.clamp(zrib1.detach(), min=0.0, max=RIMAX)
        irib = torch.clamp(
            (ri / dritbl).long(), min=0, max=JPRITBL - 2
        )
        za = (irib + 1) * dritbl - ri           # weight on ritbl[irib]
        za = za / dritbl
        zb = 1.0 - za
        a = self.ritbl.gather(0, irib)
        b = self.ritbl.gather(0, irib + 1)
        return za * a + zb * b


__all__ = [
    "RitblTable",
    "RiTableConstants",
    "build_ritbl",
    "JPRITBL",
    "RIMAX",
]
