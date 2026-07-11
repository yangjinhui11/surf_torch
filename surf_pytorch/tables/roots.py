"""Root-fraction table ``RVROOTSA`` (Zeng et al. 1998).

Faithful translation of ``SRFROOTFR`` (``surf/module/srfrootfr_mod.F90``).

The Fortran computes, per vegetation type, a fraction of roots in each
soil layer using the double-exponential integrated-root formula of
Canadell 1996 / Jackson 1996 / Zeng 1998::

    ZISD(k)   = sum of layer thicknesses 1..k
    rootf(1)  = 1 - 0.5 * (exp(-aa*z1) + exp(-bb*z1))
    rootf(k)  = 0.5 * ( exp(-aa*z(k-1)) + exp(-bb*z(k-1))
                      - exp(-aa*z(k))   - exp(-bb*z(k)) )    for 1<k<N
    rootf(N)  = 1 - sum(rootf(1..N-1))

Vegetation types whose ``aa <= 0.01`` (the +/* notes in the source) are
left at zero, and then "funny" classes 8,12,14,15,20 are overwritten so
their entire root mass sits in the top layer.

Layer thickness
---------------
The standard IFS discretisation is 4 layers of thickness
``[0.07, 0.21, 0.72, 1.89]`` m (cumulative ``[0.07, 0.28, 1.00, 2.89]`` m),
matching ``sussoil_mod``.  ``compute_roots`` accepts an arbitrary
thickness vector so a future 9/10-layer soil config still works.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


# ZAA / ZBB from srfrootfr_mod.F90 lines 123-130 (index 1..20)
_ZAA = np.array([
    5.558, 10.739, 6.706, 7.066, 5.990,
    7.344, 8.235, 4.372, 8.992, 5.558,
    4.372, 0.0,    7.344, 0.0,    0.0,
    6.326, 6.326, 4.453, 4.453, 0.0,
], dtype=np.float64)

_ZBB = np.array([
    2.614, 2.608, 2.175, 1.953, 1.955,
    1.303, 1.627, 0.978, 8.992, 2.614,
    0.978, 0.0,   1.303, 0.0,   0.0,
    1.567, 1.567, 1.631, 1.631, 0.0,
], dtype=np.float64)

# Funny classes: root mass concentrated in the top layer (Fortran 169-182).
_FUNNY_TYPES = (8, 12, 14, 15, 20)


def compute_roots(layer_thickness: np.ndarray, *, nvtypes: int = 20) -> np.ndarray:
    """Reproduce ``SRFROOTFR`` for an arbitrary soil discretisation.

    Parameters
    ----------
    layer_thickness
        1-D array of layer thicknesses [m], length = number of soil layers.
    nvtypes
        Number of vegetation types (must be 20).

    Returns
    -------
    ndarray of shape ``(nlevs, nvtypes)`` with ``PROOTF`` semantics.
    """
    if nvtypes != 20:
        raise ValueError("SRFROOTFR assumes NVTYPES=20")
    z = np.asarray(layer_thickness, dtype=np.float64).ravel()
    nlevs = z.size

    aa = _ZAA
    bb = _ZBB
    zisd = np.cumsum(z)

    prootf = np.zeros((nlevs, nvtypes), dtype=np.float64)

    for jt in range(nvtypes):
        if aa[jt] <= 0.01:
            # odd types left at zero; funny classes handled later
            continue
        # layer 1
        prootf[0, jt] = 1.0 - 0.5 * (
            np.exp(-aa[jt] * zisd[0]) + np.exp(-bb[jt] * zisd[0])
        )
        # middle layers 2..nlevs-1
        for k in range(1, nlevs - 1):
            prootf[k, jt] = 0.5 * (
                np.exp(-aa[jt] * zisd[k - 1]) + np.exp(-bb[jt] * zisd[k - 1])
                - np.exp(-aa[jt] * zisd[k]) - np.exp(-bb[jt] * zisd[k])
            )
        # bottom layer: residual
        prootf[nlevs - 1, jt] = 1.0 - np.sum(prootf[: nlevs - 1, jt])

    # Redefine funny classes: all roots in top layer
    for jt in _FUNNY_TYPES:
        prootf[:, jt - 1] = 0.0
        prootf[0, jt - 1] = 1.0

    return prootf


# standard IFS 4-layer soil thickness (sussoil RDAT(1:4))
STD_4LAYER_THICKNESS = np.array([0.07, 0.21, 0.72, 1.89], dtype=np.float64)


class RootTable(nn.Module):
    """Root-fraction LUT ``RVROOTSA`` of shape ``(nsoil, 0..20)``.

    Index 0 mirrors type 8 (desert) exactly as ``susveg_mod`` does
    (``RVROOTSA(JS,0) = RVROOTSA(JS,8)``); since type 8 has all roots in
    the top layer, the sentinel is ``[1, 0, 0, 0]`` for the 4-layer grid.
    """

    def __init__(
        self,
        *,
        layer_thickness: np.ndarray | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        thickness = layer_thickness if layer_thickness is not None else STD_4LAYER_THICKNESS
        prootf = compute_roots(thickness)            # (nsoil, 20)
        nsoil = prootf.shape[0]

        full = np.zeros((nsoil, 21), dtype=np.float64)
        full[:, 1:] = prootf
        full[:, 0] = full[:, 8]                       # sentinel mirrors desert

        self.nsoil = nsoil
        self.nvtypes = 20
        self.dtype = dtype
        self.device = device
        self.register_buffer(
            "rvrootsa", torch.as_tensor(full, dtype=dtype, device=device)
        )

    def gather(self, vegtype: torch.Tensor) -> torch.Tensor:
        """Return ``( *batch, nsoil )`` root fractions for each veg-type.

        ``vegtype`` may be any shape; the result has a trailing soil
        dimension.  Indexing is detached (veg types are categorical).
        """
        idx = vegtype.long().detach()                 # (...,)
        roots = self.rvrootsa                         # (nsoil, 21)
        # Pick one column per veg-type -> (..., nsoil) by indexing the
        # veg axis (dim=1) of roots with idx broadcast over the soil
        # axis (dim=0).
        gathered = roots[:, idx]                      # (nsoil, ...)
        # Move soil axis to the trailing position to match the contract.
        gathered = gathered.movedim(0, -1)            # (..., nsoil)
        return gathered


__all__ = ["RootTable", "compute_roots", "STD_4LAYER_THICKNESS"]
