"""PyTorch port of the OpenIFS 48r1 land-surface package.

Public API
----------
* :class:`LandSurface` -- the top-level ``nn.Module`` to plug into ML
  pipelines.  ``LandSurface().forward(state, forcing, static)`` advances
  the surface state by one coupling interval, batched over grid columns.
* :class:`SurfConfig` -- dimensions, feature flags, dtype.
* :class:`Forcing`, :class:`Static`, :class:`SurfState`, :class:`SurfDiag`
  -- input/output data containers.
* :class:`PhysicalConstants`, :class:`ConstantBundle` -- the yomcst /
  yoethf physical constants transcribed from ``sucst.F90`` and
  ``suphec.F90`` (offline-driver overrides applied).
* :mod:`surf_pytorch.tables` -- the four lookup tables (soil / veg /
  roots / ritbl).
* :mod:`surf_pytorch.utils` -- saturation functions (``foeew``,
  ``foedelta``, ``qsat`` ...) and differentiable helpers.

Example
-------
::

    import torch
    from surf_pytorch import LandSurface, SurfConfig

    model = LandSurface(SurfConfig(dtype=torch.float64))
    # state, forcing, static = ... # see state.py for the dataclasses
    # next_state, diag = model(state, forcing, static)

Status
------
M1-M6 are complete: the full surface physics core (Monin-Obukhov
exchange, implicit vertical diffusion / VDFDIFH1S wrapper, HTESSEL soil
temperature and moisture, single-layer snow) is ported and validated to
bit level against the standalone Fortran reference
(``offline_driver/libsurf_ref.so``). An idealized diurnal-cycle forcing
generator + multi-step column harness (M2c) and the snowfall-
throughfall wiring (M2d/M2e) are layered on top for end-to-end
physical-closure tests. See ``LandSurface.forward`` docstring for the
remaining cold-start placeholders.
"""

from .config import SurfConfig
from .constants import ConstantBundle, PhysicalConstants
from .model import LandSurface
from .state import Forcing, Static, SurfState, SurfDiag
from .forcing import (
    IdealizedDiurnalConfig,
    solar_mu0,
    build_idealized_diurnal,
)
from .column_harness import ColumnHarness, RunResult
from . import tables
from . import utils

__version__ = "0.1.0-m6"

__all__ = [
    "LandSurface",
    "SurfConfig",
    "PhysicalConstants",
    "ConstantBundle",
    "Forcing",
    "Static",
    "SurfState",
    "SurfDiag",
    "IdealizedDiurnalConfig",
    "solar_mu0",
    "build_idealized_diurnal",
    "ColumnHarness",
    "RunResult",
    "tables",
    "utils",
    "__version__",
]
