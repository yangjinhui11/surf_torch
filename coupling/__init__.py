"""OpenIFS SURF coupling package.

This package bridges the PyTorch land-surface port
(:mod:`surf_pytorch`) and the OpenIFS full SURF subsystem
(``surf/module/surftstp_ctl_mod.F90``).

Phase 1 -- offline inference coupling
-------------------------------------
The Fortran offline driver (``surf/offline/driver/callpar1s.F90``) is
wired to call into the PyTorch model via an in-process CPython
embedding (Option A of the coupling design):

* ``coupling/bindings/pyoifs_shim.py`` -- a *direction-neutral* flat
  ABI layer.  A single ``pyoifs_step`` callable takes column-major
  pointers and ``VALUE`` scalars (the same contract
  ``offline_driver/libsurf_ref.so`` established) and runs one
  ``LandSurface.forward`` step under ``torch.no_grad``.
* ``coupling/bindings/pyoifs_c.c``    -- the C glue (P1.2): holds the
  GIL, ``dlopen``s libpython, forwards the call.
* ``coupling/bindings/pyoifs_f.F90``  -- the Fortran thunk (P1.3):
  ``BIND(C)`` wrapper called from a ``LEPYOIFS`` branch in
  ``callpar1s``.

The shim is unit-tested without any Fortran/Python-embedding present:
``tests/test_pyoifs_marshalling.py`` feeds it raw numpy buffers (the
exact memory layout the C glue will hand in) and verifies the
pointer->tensor->pointer round-trip is bit-exact and that a real
``forward`` step runs against synthetic column-major buffers.
"""

from .bindings.pyoifs_shim import (
    PyOifsShim,
    PyoifsStepResult,
    PYOIFS_ABI_VERSION,
)

__all__ = ["PyOifsShim", "PyoifsStepResult", "PYOIFS_ABI_VERSION"]
