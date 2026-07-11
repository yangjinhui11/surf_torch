"""Direction-neutral flat ABI layer: OpenIFS SURF <-> PyTorch port.

This is the Python side of the Phase-1 offline coupling (Option A:
in-process CPython embedding).  It exposes a *single* flat callable,
:func:`pyoifs_step` (wrapped by :class:`PyOifsShim`), that mirrors the
argument contract the standalone Fortran reference
(``offline_driver/libsurf_ref.so``) already proved out, but runs in the
opposite direction: OpenIFS hands us raw column-major buffers and we
run one :meth:`surf_pytorch.LandSurface.forward` step.

ABI contract (mirrors surf_ref.F90 / callpar1s.F90 SURFTSTP_CTL inputs)
---------------------------------------------------------------------
Every array argument is a **Fortran column-major** pointer of shape
``(KLON, NTILES)`` / ``(KLON, NSWL)`` / ``(KLON,)``.  Scalars are passed
``BY VALUE`` (``INTEGER, VALUE`` / ``REAL, VALUE``) -- the C glue
(P1.2) lifts them out of the Fortran call.  This is the exact shape of
contract :mod:`surf_pytorch.offline_driver.surf_ref` uses to *call into*
Fortran, so the same ctypes type aliases apply and the round-trip test
in ``tests/test_pyoifs_marshalling.py`` can re-use them.

Why zero-copy
-------------
OpenIFS calls the surface driver once per grid block per coupling
interval; a copy of every forcing/state array on every step would
dominate the wall-clock of a global run.  ``np.ctypeslib.as_array`` gives
us a writable numpy view *onto the Fortran buffer itself* -- no copy on
the way in.  The leading dimension of every IFS array is ``KLON``, which
matches the PyTorch port's batch axis, so the view is a direct alias.
Torch wraps that numpy view with ``torch.from_numpy`` (again zero-copy),
runs the step, and we copy *only the updated prognostics* back into the
same OUT pointers (transpose to column-major via ``np.asfortranarray``
where the step produced a C-contiguous tensor).

Soil thickness is *not* a per-call argument: it is part of the static
parameterisation owned by :class:`SurfConfig` (the cy48r1 default
``(0.07, 0.21, 0.72, 1.89)``).  This matches the offline driver, where
``sudim1s`` fixes the layer depths and never re-reads them per step.

Thread / GIL
------------
The Python embedding runs on the OpenIFS master thread under the GIL
(the C glue in P1.2 acquires it before calling).  ``forward`` itself is
single-threaded CPU under ``torch.no_grad``; OpenMP/MPI parallelism
lives above the surface call, so there is no thread-safety contract to
honour inside ``pyoifs_step``.

Constants & tables
------------------
SoilTable / VegTable / RootTable / RitblTable are bit-identical
hard-coded cy48r1 values owned by the Python side (see
:mod:`surf_pytorch.tables`).  The physical constants come from
:class:`surf_pytorch.PhysicalConstants` with the offline-driver
``YOETHF`` overrides (the same object the bit-level reference tests
use).  No shared namelist is needed -- the shim owns the parameter
space end-to-end, which is what makes the coupling *drop-in*.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from surf_pytorch import (
    LandSurface,
    SurfConfig,
    Forcing,
    Static,
    SurfState,
)
from surf_pytorch.config import N_BASE_TILES


# --------------------------------------------------------------------------- #
# ABI version -- bumped whenever the argument list of pyoifs_step changes.
# The C glue (pyoifs_c.c) asserts this at init so a stale shim and a
# rebuilt driver fail loudly instead of mis-reading memory.
# --------------------------------------------------------------------------- #
PYOIFS_ABI_VERSION: int = 1


# --------------------------------------------------------------------------- #
# ctypes type aliases (identical to offline_driver/surf_ref.py so the two
# sides of the bridge agree on the wire format).
# --------------------------------------------------------------------------- #
_F64 = ctypes.c_double
_I32 = ctypes.c_int
_P_F64 = ctypes.POINTER(ctypes.c_double)
_P_I32 = ctypes.POINTER(ctypes.c_int)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class PyoifsStepResult:
    """Outcome of a single ``pyoifs_step`` call (returned for testability).

    The C glue ignores this struct -- it only cares that the OUT
    pointers were written.  Python tests and the drift logger (P1.6)
    inspect it to verify what was actually pushed back.

    Fields
    ------
    klon, ktiles, ksoil:
        Dimensions as seen by the shim (sanity echoed back).
    n_fortran_arrays:
        Number of Fortran buffers that were aliased (zero-copy) on the
        way in -- equals the count of non-None pointer arguments.
    n_outputs_written:
        Number of OUT pointers the step wrote prognostic/diagnostic
        data into.
    elapsed_s:
        Wall-clock of the ``forward`` call (excluding marshalling), for
        the drift / overhead log.
    """

    klon: int
    ktiles: int
    ksoil: int
    n_fortran_arrays: int
    n_outputs_written: int
    elapsed_s: float


# --------------------------------------------------------------------------- #
# Buffer view helpers -- the inverse of surf_ref.py's _as_fortran_2d.
#
# surf_ref.py takes a torch tensor and produces a Fortran-order numpy
# buffer to *hand to* Fortran.  Here we receive a Fortran-order buffer
# *from* Fortran and produce a torch tensor that aliases it.  The key
# insight: IFS arrays are (KLON, ...) with leading dim KLON, which is
# exactly the PyTorch port's batch axis, so a column-major view is a
# *direct* alias (no transpose, no copy).
# --------------------------------------------------------------------------- #
def _view_f64_1d(ptr, klon: int) -> np.ndarray:
    """Zero-copy writable view of a Fortran ``(KLON,)`` f64 buffer."""
    return np.ctypeslib.as_array(ptr, shape=(klon,))


def _view_f64_2d_fortran(ptr, klon: int, kdim: int) -> np.ndarray:
    """Zero-copy writable Fortran-order view of a ``(KLON, KDIM)`` f64 buffer.

    Fortran stores 2-D arrays column-major: element ``(JL, JD)`` (1-based)
    is at byte offset ``(JL-1 + (JD-1)*KLON) * 8``.  ``np.ctypeslib.as_array``
    on a raw pointer with a 2-D shape produces a *C-contiguous* array
    (row-major strides ``(kdim*8, 8)``), which would transpose / mis-read
    the IFS buffer.  We instead build the ndarray directly from the
    pointer address with column-major strides ``(8, klon*8)`` -- the same
    layout ``surf_ref._as_fortran_2d`` produces going the other way -- so
    the view is a true zero-copy alias of the Fortran memory.

    This is the load-bearing line of the whole coupling: a wrong stride
    here silently swaps tiles / soil layers and the bit-level reference
    tests never catch it because they only ever see the (correct)
    Python-built tensors.  ``tests/test_pyoifs_marshalling.py`` round-
    trips a deliberately column-major Fortran buffer through this view
    and asserts byte-exact agreement to guard exactly that.
    """
    addr = ctypes.addressof(ptr.contents)
    return np.ndarray(
        shape=(klon, kdim),
        dtype=np.float64,
        buffer=(ctypes.c_double * (klon * kdim)).from_address(addr),
        order="F",
    )


def _view_i32_1d(ptr, klon: int) -> np.ndarray:
    """Zero-copy writable view of a Fortran ``(KLON,)`` int32 buffer."""
    return np.ctypeslib.as_array(ptr, shape=(klon,))


def _wrap_tensor_f64(
    arr: np.ndarray,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Wrap a numpy f64 view as a torch tensor (zero-copy when possible).

    ``torch.from_numpy`` is zero-copy but requires the numpy array to be
    a *native* (little-endian on x86) f64 -- which the Fortran REAL(8)
    buffers are.  If the caller asked for float64 we keep the alias; if
    float32 was requested we let torch copy (the only unavoidable copy
    in the hot path, and only when the port is run in single precision).
    """
    t = torch.from_numpy(arr)
    if t.dtype != dtype:
        t = t.to(dtype)
    return t


def _to_int64(arr: np.ndarray) -> torch.Tensor:
    """Cast an int32 numpy view to the int64 tensor the LUT indexers want.

    The soil / veg type tables are indexed with ``torch.long`` (the port
    follows the IFS 0..7 / 0..20 convention); a cast is unavoidable here
    because IFS hands us ``INTEGER(KIND=4)`` buffers.  Cheap (KLON ints).
    """
    return torch.from_numpy(arr.astype(np.int64))


# --------------------------------------------------------------------------- #
# The shim
# --------------------------------------------------------------------------- #
class PyOifsShim:
    """Stateful wrapper holding the built :class:`LandSurface` model.

    Constructed **once** per OpenIFS run (the C glue's ``pyoifs_init``,
    P1.2) and called once per grid block per coupling interval.  The
    model and its tables are immutable across calls -- only the
    per-step forcing/state buffers change, which is exactly the
    ``LandSurface.forward`` contract.

    Parameters
    ----------
    n_tile, n_soil:
        Dimensions.  ``n_tile`` defaults to 8 (base tiles); pass 10 for
        the lake+urban topology.  These must match what OpenIFS allocates
        upstream -- the shim does not silently pad.
    dtype:
        ``torch.float64`` (default) to bit-match the Fortran double
        reference.  ``torch.float32`` halves memory at the cost of one
        copy on the wrap (see :func:`_wrap_tensor_f64`).
    config_overrides:
        Optional dict of :class:`SurfConfig` field overrides (e.g.
        ``{"use_lake": True, "use_urban": True}`` for the 10-tile run).
        Merged onto the inference defaults
        (``differentiable=False, validate=False``).
    """

    def __init__(
        self,
        n_tile: int = N_BASE_TILES,
        n_soil: int = 4,
        *,
        dtype: torch.dtype = torch.float64,
        config_overrides: Optional[dict] = None,
    ) -> None:
        cfg = SurfConfig(
            n_soil=n_soil,
            n_tile=n_tile,
            # Inference coupling: no autograd, no per-step assertions.
            # validate=False is essential because the stub tiles (lake /
            # urban) legitimately fall outside the invariant bands the
            # validator checks -- see test_ntile10.py for that contract.
            differentiable=False,
            validate=False,
            dtype=dtype,
        )
        if config_overrides:
            for k, v in config_overrides.items():
                setattr(cfg, k, v)
        # Re-derive the active tile count after overrides so a
        # use_lake/use_urban flip is consistent with n_tile.
        cfg.n_active_tiles()
        self.config = cfg
        self.model = LandSurface(cfg)
        self._dtype = dtype

    # ----------------------------------------------------------------- #
    # Step
    # ----------------------------------------------------------------- #
    def step(
        self,
        klon: int,
        # ---- Forcing (lowest-level state + radiation + precip + dt) ----
        pu: np.ndarray, pv: np.ndarray, pt: np.ndarray,
        pq: np.ndarray, pco2: np.ndarray,
        pp_surf: np.ndarray, pgeom1: np.ndarray,
        pfr_so: np.ndarray, pfr_th: np.ndarray, pmu0: np.ndarray,
        prain: np.ndarray, psnow: np.ndarray,
        pdt: float,
        # ---- Static (tile frac, masks, soil/veg type) ----
        ptile_frac: np.ndarray,       # (klon, n_tile)
        pland_frac: np.ndarray,       # (klon,)
        porography: np.ndarray,       # (klon,)
        psoil_type: np.ndarray,       # (klon,) int32
        pveg_type: np.ndarray,        # (klon,) int32
        # ---- Prognostic state (in) ----
        pskin_t: np.ndarray,          # (klon, n_tile)
        psoil_t: np.ndarray,          # (klon, ksoil)
        psoil_w_liq: np.ndarray,      # (klon, ksoil)
        psnow_w: np.ndarray,          # (klon,)
        psnow_frac: np.ndarray,       # (klon,)
        pskin_water: np.ndarray,      # (klon,)
        # Optional companion prognostics -- required for bit-exact
        # agreement on the water / snow tiles (tile 0 uses psst, tiles
        # 4 & 6 use psnow_t / psnow_rho).  They are part of the IFS
        # SURFTSTP_CTL prognostic set; passing them keeps the shim
        # bit-identical to the native forward path.
        psst: np.ndarray,             # (klon,) sea-surface temperature
        psnow_t: np.ndarray,          # (klon,) snow skin temperature
        psnow_rho: np.ndarray,        # (klon,) snow density
        # ---- OUT buffers (written in place) ----
        out_skin_t: np.ndarray,       # (klon, n_tile)
        out_soil_t: np.ndarray,       # (klon, ksoil)
        out_soil_w_liq: np.ndarray,   # (klon, ksoil)
        out_snow_w: np.ndarray,       # (klon,)
        out_snow_frac: np.ndarray,    # (klon,)
        out_skin_water: np.ndarray,   # (klon,)
        out_h_flux_tile: np.ndarray,  # (klon, n_tile)
        out_e_flux_tile: np.ndarray,  # (klon, n_tile)
        out_str_tile: np.ndarray,     # (klon, n_tile)
        out_g0_tile: np.ndarray,      # (klon, n_tile)
        out_slh_tile: np.ndarray,     # (klon, n_tile)
    ) -> PyoifsStepResult:
        """Run one surface step; write updated prognostics + diag to OUT.

        All ``np.ndarray`` arguments are *aliases* of the Fortran buffers
        (produced by :func:`_view_f64_1d` / :func:`_view_f64_2d_fortran`
        in the ctypes entry point, or by the test harness directly).  We
        do not copy them on the way in.  Only the OUT arrays are written,
        and only the fields the port actually advances (see the
        ``LandSurface.forward`` docstring for the cold-start placeholders
        that are *not* updated).

        Soil thickness is intentionally absent: it lives on
        :class:`SurfConfig` (the cy48r1 default), matching the offline
        driver where ``sudim1s`` fixes it once and never re-reads it.

        The three companion prognostics (``psst`` / ``psnow_t`` /
        ``psnow_rho``) are *optional in the Python dataclass* but
        *required here*: without them the water tile (0) and the two
        snow tiles (4, 6) fall back to cold-start defaults and diverge
        from the native forward path.  Carrying them in the ABI keeps
        the shim bit-identical to the native model.
        """
        n_tile = self.config.n_tile
        ktiles = n_tile
        ksoil = self.config.n_soil

        # ---- Wrap inputs as torch tensors (zero-copy for f64) -------- #
        # Each input may arrive as either:
        #   * a numpy ndarray  (from the pure-Python pyoifs_step, which
        #     pre-applies _view_f64_*), or
        #   * a raw ctypes POINTER(c_double) / POINTER(c_int) (from the C
        #     glue's pyoifs_step, which casts the address and passes the
        #     pointer straight through -- it does not know the shape).
        # _coerce_1d / _coerce_2d normalise both to a Fortran-correct
        # numpy array before _wrap_tensor_f64 aliases it.  Detecting a
        # ctypes pointer by its lack of __array_interface__ is the
        # robust test (POINTER objects expose the buffer protocol but
        # not the ndarray interface).
        def _is_ctypes_ptr(x):
            return not hasattr(x, "__array_interface__")

        def _coerce_1d(a):
            if _is_ctypes_ptr(a):
                a = np.ctypeslib.as_array(a, shape=(klon,))
            return _wrap_tensor_f64(np.ascontiguousarray(a), dtype=self._dtype)

        def _coerce_2d(a, kdim):
            if _is_ctypes_ptr(a):
                # ctypes pointer from the C glue: build a Fortran-order
                # view directly so strides match the IFS column-major
                # layout.  This mirrors _view_f64_2d_fortran exactly.
                a = _view_f64_2d_fortran(a, klon, kdim)
            return _wrap_tensor_f64(np.ascontiguousarray(a), dtype=self._dtype)

        def _coerce_int1d(a):
            if _is_ctypes_ptr(a):
                a = np.ctypeslib.as_array(a, shape=(klon,))
            return _to_int64(a)

        # _t1 / _t2 short-hands for the common shapes.
        def _t1(a):
            return _coerce_1d(a)

        def _t2_tile(a):
            return _coerce_2d(a, n_tile)

        def _t2_soil(a):
            return _coerce_2d(a, ksoil)

        forcing = Forcing(
            u=_t1(pu), v=_t1(pv), t=_t1(pt), q=_t1(pq), co2=_t1(pco2),
            p_surf=_t1(pp_surf), geom1=_t1(pgeom1),
            fr_so=_t1(pfr_so), fr_th=_t1(pfr_th), mu0=_t1(pmu0),
            rain=_t1(prain), snow=_t1(psnow),
            dt=torch.tensor(float(pdt), dtype=self._dtype),
        )
        # soil_thickness comes from SurfConfig (cy48r1 default), matching
        # the offline driver where sudim1s fixes the layer depths once and
        # never re-reads them per step.  Static requires the field, so we
        # materialise it as a tensor here.
        soil_thickness = torch.tensor(
            self.config.soil_thickness, dtype=self._dtype,
        )
        static = Static(
            tile_frac_base=_t2_tile(ptile_frac),
            land_frac=_t1(pland_frac),
            orography=_t1(porography),
            soil_type=_coerce_int1d(psoil_type),
            veg_type=_coerce_int1d(pveg_type),
            soil_thickness=soil_thickness,
        )
        state = SurfState(
            skin_t=_t2_tile(pskin_t),
            soil_t=_t2_soil(psoil_t),
            soil_w_liq=_t2_soil(psoil_w_liq),
            snow_w=_t1(psnow_w),
            snow_frac=_t1(psnow_frac),
            skin_water=_t1(pskin_water),
            sst=_t1(psst),
            snow_t=_t1(psnow_t),
            snow_rho=_t1(psnow_rho),
        )

        n_in = 24  # pointer args (forcing+static+state incl. sst/snow_t/rho)

        # ---- Run one step under no_grad (inference coupling) -------- #
        t0 = time.perf_counter()
        with torch.no_grad():
            new_state, diag = self.model(state, forcing, static)
        elapsed = time.perf_counter() - t0

        # ---- Copy outputs back into the Fortran OUT buffers ---------- #
        # Each OUT argument is a *destination* the caller owns: either a
        # Fortran-order numpy array (pure-Python pyoifs_step path) or a
        # ctypes POINTER(c_double) (C glue path).  We normalise both to a
        # writable Fortran-order ndarray view, then copy the torch output
        # (C-contiguous) into it via np.asfortranarray so it lands
        # column-major -- the IFS layout the caller reads back.
        n_out = 0

        def _dst_1d(dst):
            if _is_ctypes_ptr(dst):
                return np.ctypeslib.as_array(dst, shape=(klon,))
            return dst

        def _dst_2d(dst, kdim):
            if _is_ctypes_ptr(dst):
                return _view_f64_2d_fortran(dst, klon, kdim)
            return dst

        def _write_f64_2d(dst, src: torch.Tensor, kdim: int) -> None:
            nonlocal n_out
            view = _dst_2d(dst, kdim)
            arr = src.detach().to("cpu", torch.float64).numpy()
            view[...] = np.asfortranarray(arr, dtype=np.float64)
            n_out += 1

        def _write_f64_1d(dst, src: torch.Tensor) -> None:
            nonlocal n_out
            view = _dst_1d(dst)
            arr = src.detach().to("cpu", torch.float64).numpy().ravel()
            view[...] = np.ascontiguousarray(arr, dtype=np.float64)
            n_out += 1

        _write_f64_2d(out_skin_t, new_state.skin_t, n_tile)
        _write_f64_2d(out_soil_t, new_state.soil_t, ksoil)
        _write_f64_2d(out_soil_w_liq, new_state.soil_w_liq, ksoil)
        _write_f64_1d(out_snow_w, new_state.snow_w)
        _write_f64_1d(out_snow_frac, new_state.snow_frac)
        _write_f64_1d(out_skin_water, new_state.skin_water)

        # Diagnostics -- only the fields forward() actually populates.
        # h_flux/e_flux/net_rad/ground_heat are grid-mean and live in
        # SurfDiag but are NOT in the OUT contract here (the OpenIFS
        # SURFTSTP_CTL output set is tile-level fluxes); they remain
        # available on `diag` for the drift logger if needed.
        if diag.h_flux_tile is not None:
            _write_f64_2d(out_h_flux_tile, diag.h_flux_tile, n_tile)
        if diag.e_flux_tile is not None:
            _write_f64_2d(out_e_flux_tile, diag.e_flux_tile, n_tile)
        if diag.str_tile is not None:
            _write_f64_2d(out_str_tile, diag.str_tile, n_tile)
        if diag.g0_tile is not None:
            _write_f64_2d(out_g0_tile, diag.g0_tile, n_tile)
        if diag.slh_tile is not None:
            _write_f64_2d(out_slh_tile, diag.slh_tile, n_tile)

        return PyoifsStepResult(
            klon=klon, ktiles=ktiles, ksoil=ksoil,
            n_fortran_arrays=n_in,
            n_outputs_written=n_out,
            elapsed_s=elapsed,
        )


# --------------------------------------------------------------------------- #
# Flat ctypes entry point (the symbol the C glue will dlsym in P1.2).
#
# This is kept separate from PyOifsShim.step so that:
#   * the Python tests can exercise the marshalling without a live C
#     bridge (they call .step() with raw numpy buffers);
#   * the C glue has a single stable symbol (pyoifs_step) whose arg
#     list is the ABI contract; bumping PYOIFS_ABI_VERSION is the only
#     thing that needs to stay in sync with pyoifs_c.c.
# --------------------------------------------------------------------------- #
# Module-level slot for the last exception text (the C glue reads this
# via a second dlsym'd accessor in P1.2 to produce a usable diagnostic).
_last_error: list = [""]


def pyoifs_step(
    shim: PyOifsShim,
    klon: int,
    # Forcing
    pu, pv, pt, pq, pco2,
    pp_surf, pgeom1,
    pfr_so, pfr_th, pmu0,
    prain, psnow,
    pdt: float,
    # Static (soil_thickness omitted -- owned by SurfConfig)
    ptile_frac, pland_frac, porography,
    psoil_type, pveg_type,
    # State (in)
    pskin_t, psoil_t, psoil_w_liq,
    psnow_w, psnow_frac, pskin_water,
    psst, psnow_t, psnow_rho,
    # OUT
    out_skin_t, out_soil_t, out_soil_w_liq,
    out_snow_w, out_snow_frac, out_skin_water,
    out_h_flux_tile, out_e_flux_tile, out_str_tile,
    out_g0_tile, out_slh_tile,
) -> int:
    """ctypes-friendly entry: views raw ptrs, delegates to shim.step.

    Returns 0 on success, non-zero error code on failure (the C glue
    surfaces these as ``PYOIFS_E_*`` constants).  Every ``pu`` / ``pv``
    / ... argument is a ``POINTER(c_double)`` (or ``POINTER(c_int)`` for
    the type-index arrays) as produced by the Fortran ``BIND(C)`` thunk.
    """
    n_tile = shim.config.n_tile
    n_soil = shim.config.n_soil

    try:
        shim.step(
            klon,
            _view_f64_1d(pu, klon), _view_f64_1d(pv, klon),
            _view_f64_1d(pt, klon), _view_f64_1d(pq, klon),
            _view_f64_1d(pco2, klon),
            _view_f64_1d(pp_surf, klon), _view_f64_1d(pgeom1, klon),
            _view_f64_1d(pfr_so, klon), _view_f64_1d(pfr_th, klon),
            _view_f64_1d(pmu0, klon),
            _view_f64_1d(prain, klon), _view_f64_1d(psnow, klon),
            float(pdt),
            _view_f64_2d_fortran(ptile_frac, klon, n_tile),
            _view_f64_1d(pland_frac, klon),
            _view_f64_1d(porography, klon),
            _view_i32_1d(psoil_type, klon),
            _view_i32_1d(pveg_type, klon),
            _view_f64_2d_fortran(pskin_t, klon, n_tile),
            _view_f64_2d_fortran(psoil_t, klon, n_soil),
            _view_f64_2d_fortran(psoil_w_liq, klon, n_soil),
            _view_f64_1d(psnow_w, klon),
            _view_f64_1d(psnow_frac, klon),
            _view_f64_1d(pskin_water, klon),
            _view_f64_1d(psst, klon),
            _view_f64_1d(psnow_t, klon),
            _view_f64_1d(psnow_rho, klon),
            _view_f64_2d_fortran(out_skin_t, klon, n_tile),
            _view_f64_2d_fortran(out_soil_t, klon, n_soil),
            _view_f64_2d_fortran(out_soil_w_liq, klon, n_soil),
            _view_f64_1d(out_snow_w, klon),
            _view_f64_1d(out_snow_frac, klon),
            _view_f64_1d(out_skin_water, klon),
            _view_f64_2d_fortran(out_h_flux_tile, klon, n_tile),
            _view_f64_2d_fortran(out_e_flux_tile, klon, n_tile),
            _view_f64_2d_fortran(out_str_tile, klon, n_tile),
            _view_f64_2d_fortran(out_g0_tile, klon, n_tile),
            _view_f64_2d_fortran(out_slh_tile, klon, n_tile),
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - surface to C as error code
        # The C glue logs ``repr(exc)`` via the embedding's stderr.
        _last_error[0] = repr(exc)
        return 1


def pyoifs_last_error() -> str:
    """Return the text of the most recent ``pyoifs_step`` failure."""
    return _last_error[0]


__all__ = [
    "PYOIFS_ABI_VERSION",
    "PyOifsShim",
    "PyoifsStepResult",
    "pyoifs_step",
    "pyoifs_last_error",
    "_view_f64_1d",
    "_view_f64_2d_fortran",
    "_view_i32_1d",
    "_wrap_tensor_f64",
    "_to_int64",
]
