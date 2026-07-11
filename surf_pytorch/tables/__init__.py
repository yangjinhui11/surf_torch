"""Lookup tables for the land-surface package.

Each table is a small ``nn.Module`` that holds the table as a buffer
(non-learnable by default) or a parameter (learnable).  Tables are
built *exactly* as in the Fortran initialisation routines so a
side-by-side comparison reproduces the gold values to the last ULP:

* :class:`SoilTable`   - Van Genuchten soil hydrology (7 USDA classes)
                         plus the derived scalars from ``SUSSOIL``.
* :class:`VegTable`    - 20 surface cover classes from ``SUSVEG``.
* :class:`RootTable`   - Zeng (1998) root fractions from ``SRFROOTFR``.
* :class:`RitblTable`  - Monin-Obukhov stability table from ``SUVEXCS``.
"""

from .soil import (
    SoilConstants,
    SoilTable,
    raw_soil_table,
    RLAMBDAICE,
    RLAMBDAWAT,
    RKERST1,
    RKERST2,
    RKERST3,
)
from .veg import VegConstants, VegTable
from .roots import RootTable, compute_roots, STD_4LAYER_THICKNESS
from .ritbl import RitblTable, RiTableConstants, build_ritbl, JPRITBL, RIMAX

__all__ = [
    # soil
    "SoilConstants", "SoilTable", "raw_soil_table",
    "RLAMBDAICE", "RLAMBDAWAT", "RKERST1", "RKERST2", "RKERST3",
    # veg
    "VegConstants", "VegTable",
    # roots
    "RootTable", "compute_roots", "STD_4LAYER_THICKNESS",
    # ritbl
    "RitblTable", "RiTableConstants", "build_ritbl", "JPRITBL", "RIMAX",
]
