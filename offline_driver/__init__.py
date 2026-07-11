"""Offline-driver bridge to the standalone Fortran surface reference.

The ``offline_driver`` package is the M2 comparison harness: it wraps a
*self-contained* gfortran build of the two OpenIFS surface kernels
(``libsurf_ref.so``) and exposes them through :class:`SurfRef` so the
PyTorch port can be validated at ULP level against a Fortran reference
that shares no code with the port itself.

Build the library with ``offline_driver/build_ref.sh`` (needs only
gfortran -- no MPI / NetCDF / ecbuild), then::

    from offline_driver import SurfRef
    ref = SurfRef()
    out = ref.run(seb_inputs, veg_table, consts, config)

See ``tests/test_fortran_ref.py`` for the ULP-level comparison tests.
"""

from .surf_ref import SurfRef

__all__ = ["SurfRef"]
