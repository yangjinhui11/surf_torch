"""Marshalling tests for the OpenIFS coupling shim (P1.1).

The Phase-1 coupling hands the PyTorch port raw **Fortran column-major**
buffers from the OpenIFS SURF subsystem (``callpar1s`` /
``surftstp_ctl``).  These tests pin the marshalling contract that makes
that hand-off bit-exact, *without* needing the C glue (P1.2) or a live
Fortran build.

Two failure classes these guard against
----------------------------------------
1. **Stride bugs in ``_view_f64_2d_fortran``.**  ``np.ctypeslib.as_array``
   on a raw pointer with a 2-D shape silently produces a *C-contiguous*
   (row-major) view, which transposes / mis-reads the leading dimension
   of an IFS buffer.  A wrong stride here swaps tiles or soil layers and
   the existing per-column bit-level tests *never* catch it -- they only
   ever see Python-built tensors that are already correct.  The round-
   trip tests below feed a deliberately column-major buffer and assert
   byte-exact agreement, which is the only way to surface this class.

2. **Contract drift between the shim and the SURFTSTP_CTL output set.**
   The shim writes a fixed set of OUT pointers (skin/soil prognostics +
   per-tile fluxes).  If the model stops populating a field, or the
   shim's OUT list grows/shrinks, the ``n_outputs_written`` assertion
   catches it.  If a field's shape stops matching ``(KLON, NTILES)`` the
   shape assertions catch it.

Strategy
--------
* Build heterogeneous synthetic batches (re-uses the
  ``_heterogeneous_inputs`` builder from the batch-independence suite
  so the inputs are physically meaningful, not all-equal).
* Materialise them as Fortran-order numpy buffers (the exact memory
  layout the C glue will hand in), run them through the shim, and
  compare against a native ``LandSurface.forward`` call on the *same*
  tensors.  float64, bit-level (rtol=0, atol=0): the shim does no
  arithmetic of its own, so any disagreement is a marshalling bug.
"""

from __future__ import annotations

import ctypes

import numpy as np
import torch

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.state import Forcing, Static, SurfState

from coupling import PyOifsShim, PyoifsStepResult, PYOIFS_ABI_VERSION
from coupling.bindings.pyoifs_shim import (
    _view_f64_1d,
    _view_f64_2d_fortran,
    _view_i32_1d,
    pyoifs_step,
    pyoifs_last_error,
)


# --------------------------------------------------------------------------- #
# Synthetic input builder (mirrors test_batch_independence but returns
# the *numpy* buffers the shim consumes, in Fortran column-major order).
# --------------------------------------------------------------------------- #
def _build_fortran_buffers(*, N: int = 3, n_tile: int = 8, n_soil: int = 4):
    """Return a dict of Fortran-order numpy buffers + the matching native
    (Forcing/Static/SurfState) tensors, so the shim path and the native
    forward path see identical inputs.

    Every 2-D buffer is ``np.asfortranarray`` so its memory layout is the
    same as an IFS ``(KLON, KDIM)`` array -- this is what makes the
    stride test meaningful.
    """
    dtype = torch.float64

    # Per-column heterogeneous values (monotone in column index so a
    # cross-column leak would shift outputs detectably).
    skin = torch.tensor([270.0, 288.0, 305.0], dtype=dtype)[:N]
    tair = torch.tensor([268.0, 290.0, 303.0], dtype=dtype)[:N]
    tsoil = torch.tensor([275.0, 283.0, 295.0], dtype=dtype)[:N]
    wsoil = torch.tensor([0.18, 0.30, 0.40], dtype=dtype)[:N]
    snow_w = torch.tensor([40.0, 5.0, 0.0], dtype=dtype)[:N]
    snow_t = torch.tensor([265.0, 271.0, 273.0], dtype=dtype)[:N]
    qair = torch.tensor([0.002, 0.008, 0.015], dtype=dtype)[:N]
    fr_so = torch.tensor([50.0, 200.0, 400.0], dtype=dtype)[:N] * 3600.0
    fr_th = torch.tensor([-400.0, -250.0, -100.0], dtype=dtype)[:N] * 3600.0
    snowfl = torch.tensor([3e-4, 1e-4, 0.0], dtype=dtype)[:N]

    skin_t_t = skin.reshape(N, 1).expand(N, n_tile).contiguous()
    soil_t_t = tsoil.reshape(N, 1).expand(N, n_soil).contiguous()
    soil_w_t = wsoil.reshape(N, 1).expand(N, n_soil).contiguous()

    # Tile fractions (uniform across columns -- leakage is expected in
    # state/forcing, not in frac, matching test_batch_independence).
    frac = torch.zeros((N, n_tile), dtype=dtype)
    frac[:, 2] = 0.05
    frac[:, 3] = 0.20
    frac[:, 4] = 0.15
    frac[:, 5] = 0.20
    frac[:, 6] = 0.10
    frac[:, 7] = 0.30

    state = SurfState(
        skin_t=skin_t_t, soil_t=soil_t_t, soil_w_liq=soil_w_t,
        snow_w=snow_w,
        snow_frac=torch.tensor([0.30, 0.10, 0.0], dtype=dtype)[:N],
        skin_water=torch.zeros(N, dtype=dtype),
        sst=torch.full((N,), 290.0, dtype=dtype),
        snow_t=snow_t,
        snow_rho=torch.full((N,), 250.0, dtype=dtype),
    )
    forcing = Forcing(
        u=torch.tensor([3.0, 6.0, 9.0], dtype=dtype)[:N],
        v=torch.tensor([0.0, 1.0, 2.0], dtype=dtype)[:N],
        t=tair, q=qair,
        co2=torch.full((N,), 6e-4, dtype=dtype),
        p_surf=torch.full((N,), 1.0e5, dtype=dtype),
        geom1=torch.tensor([20.0, 50.0, 90.0], dtype=dtype)[:N],
        fr_so=fr_so, fr_th=fr_th,
        mu0=torch.tensor([0.2, 0.5, 0.8], dtype=dtype)[:N],
        rain=torch.zeros(N, dtype=dtype), snow=snowfl,
        dt=torch.tensor(3600.0, dtype=dtype),
    )
    static = Static(
        tile_frac_base=frac,
        land_frac=torch.ones(N, dtype=dtype),
        orography=torch.tensor([10.0, 50.0, 200.0], dtype=dtype)[:N],
        soil_type=torch.tensor([3, 5, 1][:N], dtype=torch.long),
        veg_type=torch.tensor([3, 7, 13][:N], dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
    )

    # ---- Materialise as Fortran-order numpy buffers ------------------ #
    def f2(t):
        return np.asfortranarray(t.numpy(), dtype=np.float64)

    def f1(t):
        return np.ascontiguousarray(t.numpy(), dtype=np.float64)

    def i1(t):
        return np.ascontiguousarray(t.numpy().astype(np.int32), dtype=np.int32)

    bufs = dict(
        # Forcing
        pu=f1(forcing.u), pv=f1(forcing.v), pt=f1(forcing.t),
        pq=f1(forcing.q), pco2=f1(forcing.co2),
        pp_surf=f1(forcing.p_surf), pgeom1=f1(forcing.geom1),
        pfr_so=f1(forcing.fr_so), pfr_th=f1(forcing.fr_th),
        pmu0=f1(forcing.mu0),
        prain=f1(forcing.rain), psnow=f1(forcing.snow),
        pdt=float(forcing.dt.item()),
        # Static
        ptile_frac=f2(frac),
        pland_frac=f1(static.land_frac),
        porography=f1(static.orography),
        psoil_type=i1(static.soil_type),
        pveg_type=i1(static.veg_type),
        # State
        pskin_t=f2(state.skin_t),
        psoil_t=f2(state.soil_t),
        psoil_w_liq=f2(state.soil_w_liq),
        psnow_w=f1(state.snow_w),
        psnow_frac=f1(state.snow_frac),
        pskin_water=f1(state.skin_water),
        psst=f1(state.sst),
        psnow_t=f1(state.snow_t),
        psnow_rho=f1(state.snow_rho),
    )
    return bufs, state, forcing, static


def _alloc_outs(*, klon: int, n_tile: int, n_soil: int) -> dict:
    """Allocate zeroed Fortran-order OUT buffers (what the C glue hands in)."""
    def z2(a, b):
        return np.zeros((a, b), dtype=np.float64, order="F")

    def z1(a):
        return np.zeros((a,), dtype=np.float64)

    return dict(
        out_skin_t=z2(klon, n_tile),
        out_soil_t=z2(klon, n_soil),
        out_soil_w_liq=z2(klon, n_soil),
        out_snow_w=z1(klon),
        out_snow_frac=z1(klon),
        out_skin_water=z1(klon),
        out_h_flux_tile=z2(klon, n_tile),
        out_e_flux_tile=z2(klon, n_tile),
        out_str_tile=z2(klon, n_tile),
        out_g0_tile=z2(klon, n_tile),
        out_slh_tile=z2(klon, n_tile),
    )


# --------------------------------------------------------------------------- #
# Tests: ABI / construction sanity
# --------------------------------------------------------------------------- #
class TestShimConstruction:
    """The shim builds cleanly and exposes the documented ABI."""

    def test_abi_version_is_positive_int(self):
        assert isinstance(PYOIFS_ABI_VERSION, int)
        assert PYOIFS_ABI_VERSION >= 1

    def test_default_shim_is_8_tile_float64_inference(self):
        shim = PyOifsShim()
        assert shim.config.n_tile == 8
        assert shim.config.n_soil == 4
        assert shim.config.differentiable is False
        assert shim.config.validate is False
        assert shim._dtype == torch.float64
        assert isinstance(shim.model, LandSurface)

    def test_10_tile_shim_accepts_lake_urban_overrides(self):
        shim = PyOifsShim(
            n_tile=10,
            config_overrides={"use_lake": True, "use_urban": True},
        )
        assert shim.config.n_tile == 10
        assert shim.config.use_lake is True
        assert shim.config.use_urban is True


# --------------------------------------------------------------------------- #
# Tests: the load-bearing stride correctness of _view_f64_2d_fortran.
#
# This is the single most important guard in the coupling suite: a wrong
# stride here silently swaps tiles/soil layers and no other test catches
# it.  We build a deliberately column-major buffer with a known pattern,
# wrap it, and check the view reads the *Fortran* (not C) layout.
# --------------------------------------------------------------------------- #
class TestFortran2DViewStride:
    """``_view_f64_2d_fortran`` reads column-major memory as column-major.

    The pattern encodes the (row, col) index into the value itself so a
    transposed read produces obviously wrong numbers (e.g. 1030 instead
    of 3010) -- a self-checking sentinel.
    """

    @staticmethod
    def _sentinel_buf(klon: int, kdim: int) -> np.ndarray:
        """Column-major buffer where element [r,c] == r*1000 + c.

        Stored Fortran-order so the raw bytes match what IFS would hand
        us.  A C-contiguous mis-read would transpose r/c.
        """
        arr = np.empty((klon, kdim), dtype=np.float64, order="F")
        for r in range(klon):
            for c in range(kdim):
                arr[r, c] = float(r * 1000 + c)
        return arr

    def test_view_reads_fortran_layout_not_c(self):
        klon, kdim = 3, 8
        buf = self._sentinel_buf(klon, kdim)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        view = _view_f64_2d_fortran(ptr, klon, kdim)

        # Shape and dtype.
        assert view.shape == (klon, kdim)
        assert view.dtype == np.float64
        # The sentinel: view[r,c] must equal r*1000+c.  A transposed
        # (C-order) read would give c*1000+r, which differs for r!=c.
        for r in range(klon):
            for c in range(kdim):
                expected = float(r * 1000 + c)
                assert view[r, c] == expected, (
                    f"stride bug: view[{r},{c}]={view[r, c]} expected "
                    f"{expected} (a C-order read would give {c*1000+r})"
                )

    def test_view_is_zero_copy_alias_of_source_buffer(self):
        """Writes through the view must land in the source buffer.

        Zero-copy is the whole point of the coupling (per-step array
        copies would dominate a global run).  We mutate one element via
        the view and confirm the underlying buffer sees it.
        """
        klon, kdim = 2, 4
        buf = self._sentinel_buf(klon, kdim)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        view = _view_f64_2d_fortran(ptr, klon, kdim)

        view[1, 2] = -999.0
        assert buf[1, 2] == -999.0, "view is not a zero-copy alias of buf"

    def test_view_matches_surf_ref_round_trip_layout(self):
        """A buffer produced by np.asfortranarray (the surf_ref.py
        direction) must round-trip through our view unchanged.

        This ties the two sides of the bridge together: what
        ``surf_ref._as_fortran_2d`` writes, ``_view_f64_2d_fortran``
        reads identically.
        """
        klon, kdim = 4, 8
        src = np.random.default_rng(0).standard_normal((klon, kdim))
        fbuf = np.asfortranarray(src, dtype=np.float64)
        ptr = fbuf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        view = _view_f64_2d_fortran(ptr, klon, kdim)
        np.testing.assert_array_equal(view, src)


class TestFortran1DViews:
    """1-D f64 / int32 views are zero-copy and dtype-correct."""

    def test_f64_1d_view_round_trips(self):
        src = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
        ptr = src.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        view = _view_f64_1d(ptr, len(src))
        np.testing.assert_array_equal(view, src)
        assert view.dtype == np.float64

    def test_i32_1d_view_round_trips(self):
        src = np.array([3, 5, 1], dtype=np.int32)
        ptr = src.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        view = _view_i32_1d(ptr, len(src))
        np.testing.assert_array_equal(view, src)
        assert view.dtype == np.int32


# --------------------------------------------------------------------------- #
# Tests: end-to-end shim step against native forward (the headline test).
# --------------------------------------------------------------------------- #
class TestShimMatchesNativeForward:
    """``shim.step`` on Fortran buffers == ``model.forward`` on tensors.

    The shim does no arithmetic of its own -- it only marshals -- so on
    float64 the two paths must agree bit-for-bit.  Any difference is a
    marshalling bug (wrong stride, wrong dtype, wrong field wiring).
    """

    @staticmethod
    def _run_both(N=3, n_tile=8, n_soil=4):
        bufs, state, forcing, static = _build_fortran_buffers(
            N=N, n_tile=n_tile, n_soil=n_soil,
        )

        # Native reference path.
        cfg = SurfConfig(
            n_soil=n_soil, n_tile=n_tile,
            differentiable=False, validate=False, dtype=torch.float64,
        )
        model = LandSurface(cfg)
        ns_ref, diag_ref = model(state, forcing, static)

        # Shim path.
        shim = PyOifsShim(n_tile=n_tile, n_soil=n_soil)
        outs = _alloc_outs(klon=N, n_tile=n_tile, n_soil=n_soil)
        res = shim.step(N, **bufs, **outs)
        return res, outs, ns_ref, diag_ref

    def test_step_returns_well_formed_result(self):
        res, outs, ns_ref, diag_ref = self._run_both()
        assert isinstance(res, PyoifsStepResult)
        assert res.klon == 3
        assert res.ktiles == 8
        assert res.ksoil == 4
        assert res.elapsed_s >= 0.0
        # 6 prognostic OUTs + 5 diagnostic OUTs (forward populates all 5
        # per-tile fluxes for an 8-tile land batch).
        assert res.n_outputs_written == 11, res.n_outputs_written
        assert res.n_fortran_arrays == 24

    def test_out_prognostics_match_native_bit_exact(self):
        res, outs, ns_ref, diag_ref = self._run_both()

        # 2-D prognostics: shim wrote column-major, compare against the
        # native tensor (torch tensors are row-major; np comparison is
        # layout-agnostic so this is a value-only check, which is what
        # we want).
        torch.testing.assert_close(
            torch.from_numpy(np.array(outs["out_skin_t"], dtype=np.float64)),
            ns_ref.skin_t, rtol=0.0, atol=0.0,
            msg="shim skin_t disagrees with native forward",
        )
        torch.testing.assert_close(
            torch.from_numpy(np.array(outs["out_soil_t"], dtype=np.float64)),
            ns_ref.soil_t, rtol=0.0, atol=0.0,
            msg="shim soil_t disagrees with native forward",
        )
        torch.testing.assert_close(
            torch.from_numpy(np.array(outs["out_soil_w_liq"], dtype=np.float64)),
            ns_ref.soil_w_liq, rtol=0.0, atol=0.0,
            msg="shim soil_w_liq disagrees with native forward",
        )
        torch.testing.assert_close(
            torch.from_numpy(outs["out_snow_w"].copy()),
            ns_ref.snow_w, rtol=0.0, atol=0.0,
        )
        torch.testing.assert_close(
            torch.from_numpy(outs["out_snow_frac"].copy()),
            ns_ref.snow_frac, rtol=0.0, atol=0.0,
        )
        torch.testing.assert_close(
            torch.from_numpy(outs["out_skin_water"].copy()),
            ns_ref.skin_water, rtol=0.0, atol=0.0,
        )

    def test_out_diag_tile_fluxes_match_native_bit_exact(self):
        res, outs, ns_ref, diag_ref = self._run_both()

        for name, ref in [
            ("out_h_flux_tile", diag_ref.h_flux_tile),
            ("out_e_flux_tile", diag_ref.e_flux_tile),
            ("out_str_tile", diag_ref.str_tile),
            ("out_g0_tile", diag_ref.g0_tile),
            ("out_slh_tile", diag_ref.slh_tile),
        ]:
            assert ref is not None, f"native diag {name} is None"
            got = torch.from_numpy(np.array(outs[name], dtype=np.float64))
            torch.testing.assert_close(
                got, ref, rtol=0.0, atol=0.0,
                msg=f"shim {name} disagrees with native forward",
            )

    def test_outs_are_finite_and_in_physical_band(self):
        """Sanity: the shim's outputs are finite and roughly physical.

        Catches a class of marshalling bug where a stride error produces
        NaN / inf / wildly out-of-band values that happen to compare
        equal to an equally-broken reference.  Band checks are loose.
        """
        res, outs, ns_ref, diag_ref = self._run_both()

        for name in ("out_skin_t", "out_soil_t"):
            arr = np.array(outs[name], dtype=np.float64)
            assert np.all(np.isfinite(arr)), f"{name} has non-finite entries"
            assert arr.min() > 200.0 and arr.max() < 400.0, (
                f"{name} out of physical band: [{arr.min()}, {arr.max()}]"
            )
        # Soil moisture is a volumetric fraction in [0, ~0.5].
        w = np.array(outs["out_soil_w_liq"], dtype=np.float64)
        assert np.all(np.isfinite(w))
        assert w.min() >= -1e-9 and w.max() < 1.0


# --------------------------------------------------------------------------- #
# Tests: the flat pyoifs_step entry point (what the C glue will dlsym).
# --------------------------------------------------------------------------- #
class TestFlatPyoifsStep:
    """``pyoifs_step`` (the ctypes entry symbol) delegates correctly and
    returns 0 on success / 1 + last_error text on failure."""

    def _ptrs_from_buffers(self, bufs: dict, outs: dict) -> dict:
        """Lift numpy arrays to ctypes pointers (what Fortran BIND(C) hands
        the C glue, and what the C glue hands pyoifs_step)."""
        P_F64 = ctypes.POINTER(ctypes.c_double)
        P_I32 = ctypes.POINTER(ctypes.c_int)

        def p64(arr):
            return arr.ctypes.data_as(P_F64)

        def pi32(arr):
            return arr.ctypes.data_as(P_I32)

        ptrs = dict(
            pu=p64(bufs["pu"]), pv=p64(bufs["pv"]), pt=p64(bufs["pt"]),
            pq=p64(bufs["pq"]), pco2=p64(bufs["pco2"]),
            pp_surf=p64(bufs["pp_surf"]), pgeom1=p64(bufs["pgeom1"]),
            pfr_so=p64(bufs["pfr_so"]), pfr_th=p64(bufs["pfr_th"]),
            pmu0=p64(bufs["pmu0"]),
            prain=p64(bufs["prain"]), psnow=p64(bufs["psnow"]),
            pdt=bufs["pdt"],
            ptile_frac=p64(bufs["ptile_frac"]),
            pland_frac=p64(bufs["pland_frac"]),
            porography=p64(bufs["porography"]),
            psoil_type=pi32(bufs["psoil_type"]),
            pveg_type=pi32(bufs["pveg_type"]),
            pskin_t=p64(bufs["pskin_t"]),
            psoil_t=p64(bufs["psoil_t"]),
            psoil_w_liq=p64(bufs["psoil_w_liq"]),
            psnow_w=p64(bufs["psnow_w"]),
            psnow_frac=p64(bufs["psnow_frac"]),
            pskin_water=p64(bufs["pskin_water"]),
            psst=p64(bufs["psst"]),
            psnow_t=p64(bufs["psnow_t"]),
            psnow_rho=p64(bufs["psnow_rho"]),
            out_skin_t=p64(outs["out_skin_t"]),
            out_soil_t=p64(outs["out_soil_t"]),
            out_soil_w_liq=p64(outs["out_soil_w_liq"]),
            out_snow_w=p64(outs["out_snow_w"]),
            out_snow_frac=p64(outs["out_snow_frac"]),
            out_skin_water=p64(outs["out_skin_water"]),
            out_h_flux_tile=p64(outs["out_h_flux_tile"]),
            out_e_flux_tile=p64(outs["out_e_flux_tile"]),
            out_str_tile=p64(outs["out_str_tile"]),
            out_g0_tile=p64(outs["out_g0_tile"]),
            out_slh_tile=p64(outs["out_slh_tile"]),
        )
        return ptrs

    def test_flat_step_returns_zero_and_writes_outputs(self):
        bufs, state, forcing, static = _build_fortran_buffers(N=3)
        shim = PyOifsShim()
        outs = _alloc_outs(klon=3, n_tile=8, n_soil=4)
        ptrs = self._ptrs_from_buffers(bufs, outs)

        rc = pyoifs_step(shim, 3, **ptrs)
        assert rc == 0, f"pyoifs_step failed: {pyoifs_last_error()}"
        # Confirm at least one OUT was actually written (not all-zero).
        assert np.any(outs["out_skin_t"] != 0.0)

    def test_flat_step_failure_sets_last_error(self):
        """A deliberately wrong KLON (smaller than the buffers) must trip
        the view's bounds and surface as rc=1 + a non-empty error string."""
        bufs, state, forcing, static = _build_fortran_buffers(N=3)
        shim = PyOifsShim()
        outs = _alloc_outs(klon=3, n_tile=8, n_soil=4)
        ptrs = self._ptrs_from_buffers(bufs, outs)

        # KLON=99 vs buffers sized for 3 -> the 2-D Fortran view will
        # build a (99, 8) ndarray over a 24-element buffer; the forward
        # call then reads out of bounds and either crashes or the view
        # construction itself raises.  We assert rc != 0 and that an
        # error string is recorded (rather than asserting a specific
        # exception class -- the failure mode is platform-dependent).
        rc = pyoifs_step(shim, 99, **ptrs)
        assert rc != 0
        assert pyoifs_last_error() != ""
