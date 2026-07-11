"""Batch-independence of :meth:`LandSurface.forward`.

The headline guarantee of the "batched differentiable nn.Module" port
is that one forward pass over a batch of ``N`` columns reproduces the
result of running each column as its own length-1 batch -- i.e. columns
do not leak into one another.  This is the property that lets ML
training feed whole mini-batches and trust the outputs.

It is *invisible* to the per-column Fortran bit-level tests: those
inherently exercise one column at a time, so a stray cross-column
reduction (a ``torch.mean`` over dim 0, a dropped ``dim=`` argument, a
broadcast that accidentally aligns across the batch axis) would pass
every existing test yet corrupt a real batch.  These tests pin that
property explicitly.

Strategy
--------
Build a *heterogeneous* batch -- deliberately different columns, so any
leakage has something to corrupt -- then compare the single batched
call against one call per column, and against arbitrary sub-batches.
Bit-level (rtol=0, atol=0) on float64, because a correct batched
implementation produces identical floating-point ops regardless of
batch size.
"""

from __future__ import annotations

import dataclasses

import torch

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.state import Forcing, Static, SurfState


# --------------------------------------------------------------------------- #
# Input builder: a heterogeneous batch (the precondition for leakage to show)
# --------------------------------------------------------------------------- #
def _heterogeneous_inputs(
    *, N: int = 3, dtype: torch.dtype = torch.float64,
) -> tuple[SurfState, Forcing, Static, SurfConfig]:
    """Build a batch whose columns differ in every state / forcing field.

    Column ``i`` gets a distinct skin / air temperature, soil moisture
    and snow load, so a cross-column reduction would visibly change the
    output of at least one column.  Snowfall is non-zero so the snow
    branch (the most recently wired path, M6c/M2d) is also exercised.
    """
    cfg = SurfConfig(dtype=dtype, differentiable=True)
    ntile, nsoil = cfg.n_tile, cfg.n_soil

    # Per-column values -- monotone in the column index so any leak shifts
    # outputs in a detectable direction.
    skin = torch.tensor([270.0, 288.0, 305.0], dtype=dtype)[:N]
    tair = torch.tensor([268.0, 290.0, 303.0], dtype=dtype)[:N]
    tsoil = torch.tensor([275.0, 283.0, 295.0], dtype=dtype)[:N]
    wsoil = torch.tensor([0.18, 0.30, 0.40], dtype=dtype)[:N]
    snow_w = torch.tensor([40.0, 5.0, 0.0], dtype=dtype)[:N]   # col 2 no snow
    snow_t = torch.tensor([265.0, 271.0, 273.0], dtype=dtype)[:N]
    qair = torch.tensor([0.002, 0.008, 0.015], dtype=dtype)[:N]
    fr_so = torch.tensor([ 50.0, 200.0, 400.0], dtype=dtype)[:N] * 3600.0
    fr_th = torch.tensor([-400.0, -250.0, -100.0], dtype=dtype)[:N] * 3600.0
    snowfl = torch.tensor([3e-4, 1e-4, 0.0], dtype=dtype)[:N]

    state = SurfState(
        skin_t=skin.reshape(N, 1).expand(N, ntile).contiguous(),
        soil_t=tsoil.reshape(N, 1).expand(N, nsoil).contiguous(),
        soil_w_liq=wsoil.reshape(N, 1).expand(N, nsoil).contiguous(),
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
        t=tair,
        q=qair,
        co2=torch.full((N,), 6e-4, dtype=dtype),
        p_surf=torch.full((N,), 1.0e5, dtype=dtype),
        geom1=torch.tensor([20.0, 50.0, 90.0], dtype=dtype)[:N],
        fr_so=fr_so,
        fr_th=fr_th,
        mu0=torch.tensor([0.2, 0.5, 0.8], dtype=dtype)[:N],
        rain=torch.zeros(N, dtype=dtype),
        snow=snowfl,
        dt=torch.tensor(3600.0, dtype=dtype),
    )
    frac = torch.zeros((N, ntile), dtype=dtype)
    # Identical tile layout per column (frac is not where leakage is
    # expected -- state/forcing are -- so a uniform layout is fine).
    frac[:, 2] = 0.05
    frac[:, 3] = 0.20
    frac[:, 4] = 0.15
    frac[:, 5] = 0.20
    frac[:, 6] = 0.10
    frac[:, 7] = 0.30
    static = Static(
        tile_frac_base=frac,
        land_frac=torch.ones(N, dtype=dtype),
        orography=torch.tensor([10.0, 50.0, 200.0], dtype=dtype)[:N],
        soil_type=torch.tensor([3, 5, 1][:N], dtype=torch.long),
        veg_type=torch.tensor([3, 7, 13][:N], dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.full((N,), 0.99, dtype=dtype),
    )
    return state, forcing, static, cfg


def _slice_col(obj, i: int):
    """Return a length-1 batch holding only column ``i`` of ``obj``.

    Walks the dataclass fields; for tensor fields takes ``[i:i+1]`` so
    the leading axis stays a batch dim (length 1), matching what a
    standalone forward expects.  ``None`` / non-tensor fields pass
    through untouched.
    """
    return _slice_batch(obj, i, i + 1)


def _is_batched(v: torch.Tensor) -> bool:
    """A tensor carries a column axis iff it has >= 1 dim.

    ``forcing.dt`` is a 0-dim scalar -- the physical timestep is shared
    across the whole batch, not per-column -- so it must NOT be sliced
    (it has no leading axis to index).  This distinction is itself part
    of the batched-port contract these tests pin.
    """
    return v.dim() >= 1


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestBatchIndependence:
    """forward over an N-batch == one forward per column, bit-exact."""

    def test_batched_forward_matches_per_column_outputs(self):
        """The defining property: ``forward([A,B,C])`` reproduces the
        concatenation of ``forward([A])``, ``forward([B])``, ``forward([C])``.

        Bit-level (rtol=0, atol=0): a correct batched implementation
        performs identical floating-point ops regardless of how many
        columns sit in the batch, so any difference is a real leak bug
        (a cross-column reduction, a wrong ``dim=``, a stray broadcast).
        """
        state, forcing, static, cfg = _heterogeneous_inputs(N=3)
        model = LandSurface(cfg)

        # One batched call over all 3 columns.
        new_state_b, diag_b = model(state, forcing, static)

        # One call per column, then stack along dim 0.
        per_col_states = []
        per_col_diags = []
        for i in range(3):
            ns_i, diag_i = model(
                _slice_col(state, i),
                _slice_col(forcing, i),
                _slice_col(static, i),
            )
            per_col_states.append(ns_i)
            per_col_diags.append(diag_i)

        # Field-by-field compare new_state and diag, bit-exact.  Stack the
        # per-column length-1 outputs back to (3, ...) along the batch axis.
        for container_name, batched, per_list in [
            ("new_state", new_state_b, per_col_states),
            ("diag", diag_b, per_col_diags),
        ]:
            for f in dataclasses.fields(batched):
                tb = getattr(batched, f.name)
                tp = [getattr(s, f.name) for s in per_list]
                if not isinstance(tb, torch.Tensor):
                    continue  # optional/None fields handled below
                stacked = torch.cat(tp, dim=0)
                torch.testing.assert_close(
                    tb, stacked, rtol=0.0, atol=0.0,
                    msg=f"{container_name}.{f.name}: batched != per-column",
                )

    def test_arbitrary_sub_batch_split_matches_full_batch(self):
        """Splitting a batch at any boundary gives identical results.

        ``forward([A,B,C])`` == cat(forward([A,B]), forward([C])) ==
        cat(forward([A]), forward([B,C])).  This rules out a class of bug
        where independence holds for length-1 vs length-N but not for an
        intermediate split (e.g. a kernel that hardcodes a power-of-two
        or a reduction that only mishandles even/odd counts).
        """
        state, forcing, static, cfg = _heterogeneous_inputs(N=3)
        model = LandSurface(cfg)
        ns_full, diag_full = model(state, forcing, static)

        # Split [0:2] + [2:3].
        ns_a, diag_a = model(
            _slice_batch(state, 0, 2),
            _slice_batch(forcing, 0, 2),
            _slice_batch(static, 0, 2),
        )
        ns_b, diag_b = model(
            _slice_batch(state, 2, 3),
            _slice_batch(forcing, 2, 3),
            _slice_batch(static, 2, 3),
        )
        for container_name, full, parts in [
            ("new_state", ns_full, [ns_a, ns_b]),
            ("diag", diag_full, [diag_a, diag_b]),
        ]:
            for f in dataclasses.fields(full):
                tf = getattr(full, f.name)
                if not isinstance(tf, torch.Tensor):
                    continue
                stacked = torch.cat(
                    [getattr(p, f.name) for p in parts], dim=0,
                )
                torch.testing.assert_close(
                    tf, stacked, rtol=0.0, atol=0.0,
                    msg=f"{container_name}.{f.name}: full != split",
                )

    def test_outputs_are_independent_of_batch_size(self):
        """A single column's output does not change when more columns
        are appended to its batch.

        Slices column 0 out of three different batch sizes (1, 2, 3) and
        checks its slice of the output is identical every time.  This is
        the strongest form of the independence guarantee and the most
        direct regression guard: if any future change makes a column's
        result depend on its neighbours, this fails.
        """
        state, forcing, static, cfg = _heterogeneous_inputs(N=3)
        model = LandSurface(cfg)

        outs = []
        for size in (1, 2, 3):
            ns, diag = model(
                _slice_batch(state, 0, size),
                _slice_batch(forcing, 0, size),
                _slice_batch(static, 0, size),
            )
            outs.append((ns, diag))

        for container_name, getter in [("new_state", lambda o: o[0]),
                                       ("diag", lambda o: o[1])]:
            ref = getter(outs[-1])  # batch size 3, take column 0
            for (ns, diag) in outs[:-1]:
                cur = getter((ns, diag))
                for f in dataclasses.fields(ref):
                    rf = getattr(ref, f.name)
                    cf = getattr(cur, f.name)
                    if not isinstance(rf, torch.Tensor):
                        continue
                    torch.testing.assert_close(
                        cf[0:1], rf[0:1], rtol=0.0, atol=0.0,
                        msg=(f"{container_name}.{f.name}: column 0 output "
                             f"depends on batch size"),
                    )


# --------------------------------------------------------------------------- #
# float32 smoke (the ML default dtype, previously zero coverage)
# --------------------------------------------------------------------------- #
class TestFloat32Smoke:
    """float32 is the ML training default -- verify the port runs and the
    validate safety-net stays out of its way (no spurious asserts from
    float32 rounding at the band edges)."""

    def test_float32_forward_runs_and_finite(self):
        state, forcing, static, _ = _heterogeneous_inputs(N=3)
        # Rebuild in float32 (helper hardcodes float64 tensors).
        state32 = _to_dtype(state, torch.float32)
        forcing32 = _to_dtype(forcing, torch.float32)
        static32 = _to_dtype(static, torch.float32)
        cfg32 = SurfConfig(dtype=torch.float32, differentiable=True)
        model = LandSurface(cfg32)

        new_state, diag = model(state32, forcing32, static32)
        assert torch.isfinite(new_state.skin_t).all()
        assert torch.isfinite(new_state.soil_t).all()
        assert torch.isfinite(new_state.soil_w_liq).all()
        assert torch.isfinite(new_state.snow_w).all()

    def test_float32_matches_float64_within_reasonable_tolerance(self):
        """float32 forward should agree with float64 to float32 precision.

        Catches a porting bug where a hardcoded constant or an integer
        division silently degrades float32 accuracy far beyond the
        expected ~1e-5 relative error.  Tolerance is loose (rtol=1e-3):
        the point is "same answer to float32 precision", not bit-exact.
        """
        state, forcing, static, _ = _heterogeneous_inputs(N=3)
        m64 = LandSurface(SurfConfig(dtype=torch.float64))
        ns64, _ = m64(state, forcing, static)

        state32 = _to_dtype(state, torch.float32)
        forcing32 = _to_dtype(forcing, torch.float32)
        static32 = _to_dtype(static, torch.float32)
        m32 = LandSurface(SurfConfig(dtype=torch.float32))
        ns32, _ = m32(state32, forcing32, static32)

        for f in dataclasses.fields(ns64):
            t64 = getattr(ns64, f.name)
            t32 = getattr(ns32, f.name)
            if not isinstance(t64, torch.Tensor):
                continue
            torch.testing.assert_close(
                t32, t64.to(torch.float32), rtol=1e-3, atol=1e-3,
                msg=f"float32 vs float64 disagree beyond float32 precision "
                    f"on {f.name}",
            )


# --------------------------------------------------------------------------- #
# Small slice helpers
# --------------------------------------------------------------------------- #
def _slice_batch(obj, start: int, stop: int):
    """Return ``obj[start:stop]`` as a fresh container (batch dim sliced).

    0-dim scalar tensors (e.g. ``forcing.dt``) are passed through
    unchanged -- they are batch-shared, not per-column.
    """
    kw = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, torch.Tensor) and _is_batched(v):
            kw[f.name] = v[start:stop]
        else:
            kw[f.name] = v
    return type(obj)(**kw)


def _stack_field(obj_list, field_name: str) -> torch.Tensor:
    """Stack a tensor field across a list of equal-shape containers."""
    return torch.cat([getattr(o, field_name) for o in obj_list], dim=0)


def _to_dtype(obj, dtype: torch.dtype):
    """Cast every tensor field of a container to ``dtype``."""
    kw = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        kw[f.name] = v.to(dtype) if isinstance(v, torch.Tensor) else v
    return type(obj)(**kw)
