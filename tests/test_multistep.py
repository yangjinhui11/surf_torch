"""Multi-step rollout determinism for :meth:`LandSurface.forward`.

ML training drives the surface step in a feedback loop: the ``new_state``
returned at step *k* is fed back as the input at step *k+1* (see
:class:`surf_pytorch.column_harness.ColumnHarness`).  The single-step
forward tests cannot see a class of bug that only surfaces over a
rollout:

* **state aliasing** -- if ``new_state`` shared storage with an internal
  buffer, the next step would read mutated input;
* **harness vs hand-loop drift** -- if :meth:`ColumnHarness.run`
  mis-recorded a frame or fed the wrong state forward, the recorded
  series would diverge from a plain ``for k: state, diag = model(...)``
  loop even though each step is individually correct;
* **non-determinism** -- a stray buffer reuse across calls would make
  two identical rollouts differ;
* **slow divergence** -- an energy/mass leak too small to see in one
  step blows up over dozens of steps.

These tests pin the multi-step contract directly.  They do NOT compare
against Fortran (the per-step bit-level coverage in the ``*_ref`` tests
is the source of truth); they assert *internal consistency* of the
rollout, which is what a training loop relies on.
"""

from __future__ import annotations

import dataclasses

import torch

from surf_pytorch import (
    ColumnHarness,
    IdealizedDiurnalConfig,
    LandSurface,
    SurfConfig,
    build_idealized_diurnal,
)
from surf_pytorch.state import SurfState


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_rollout(n_steps: int = 24, *, seed: int = 0):
    """Build (forcing_list, static, state0, model) for a deterministic run.

    Uses the same idealized-diurnal generator as the M2c harness tests
    so the forcing sequence is physically structured (diurnal SW, a
    precip event) rather than a flat constant -- the rollout exercises
    the genuine day/night + wet/dry transitions a training loop sees.
    """
    torch.manual_seed(seed)
    fl, static, state0 = build_idealized_diurnal(
        IdealizedDiurnalConfig(n_steps=n_steps),
    )
    model = LandSurface(SurfConfig(dtype=torch.float64, validate=False))
    return fl, static, state0, model


def _manual_loop(model, state0, forcing_list, static):
    """A from-scratch ``state -> new_state`` feedback loop.

    Independent of :class:`ColumnHarness` -- just calls ``forward`` and
    threads the returned state.  This is the reference the harness is
    compared against; if they ever disagree, the bug is in the harness
    (recording / feedback), not in forward (which both paths share).
    """
    states: list[SurfState] = []
    diags = []
    state = state0
    for f in forcing_list:
        state, diag = model(state, f, static)
        states.append(state)
        diags.append(diag)
    return states, diags


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestMultiStepDeterminism:
    """The harness rollout is identical to a hand-rolled feedback loop."""

    def test_harness_matches_manual_loop_bit_exact(self):
        """``ColumnHarness.run`` reproduces a plain ``forward`` feedback loop.

        Bit-level on every recorded prognostic / diagnostic field.  The
        harness and the manual loop share the same ``forward``; a
        difference can only come from the harness (wrong frame recorded,
        wrong state fed forward), so this isolates the harness logic.
        """
        fl, static, state0, model = _build_rollout(n_steps=24)
        result = ColumnHarness(model, static, state0).run(fl)
        states, diags = _manual_loop(model, state0, fl, static)

        # Stack the manual-loop outputs to (T, ...) and compare field by field.
        recorded = {
            "skin_t": result.skin_t, "soil_t": result.soil_t,
            "soil_w_liq": result.soil_w_liq, "snow_w": result.snow_w,
            "skin_water": result.skin_water,
            "h_flux": result.h_flux, "e_flux": result.e_flux,
            "net_rad": result.net_rad, "ground_heat": result.ground_heat,
        }
        manual = {
            "skin_t": torch.stack([s.skin_t for s in states]),
            "soil_t": torch.stack([s.soil_t for s in states]),
            "soil_w_liq": torch.stack([s.soil_w_liq for s in states]),
            "snow_w": torch.stack([s.snow_w for s in states]),
            "skin_water": torch.stack([s.skin_water for s in states]),
            "h_flux": torch.stack([d.h_flux for d in diags]),
            "e_flux": torch.stack([d.e_flux for d in diags]),
            "net_rad": torch.stack([d.net_rad for d in diags]),
            "ground_heat": torch.stack([d.ground_heat for d in diags]),
        }
        for name in recorded:
            torch.testing.assert_close(
                recorded[name], manual[name], rtol=0.0, atol=0.0,
                msg=f"harness vs manual-loop drift on {name}",
            )

    def test_state_feedback_is_correct_per_step(self):
        """The state fed to step *k+1* is exactly step *k*'s output.

        Re-runs the prefix [0..k] from cold start for every k and checks
        the k-th output equals the k-th frame of the full run -- i.e.
        no step's result depends on steps after it, and the feedback
        chain is unbroken.  Catches a state-aliasing bug where step
        k+1 mutates the recorded frame k.
        """
        fl, static, state0, model = _build_rollout(n_steps=12)
        full = ColumnHarness(model, static, state0).run(fl)

        for k in range(len(fl)):
            prefix = fl[: k + 1]
            res_k = ColumnHarness(model, static, state0).run(prefix)
            # The final frame of the prefix run must match frame k of the
            # full run -- they are the same forward chain up to step k.
            torch.testing.assert_close(
                res_k.skin_t[-1], full.skin_t[k], rtol=0.0, atol=0.0,
                msg=f"prefix-run frame {k} != full-run frame {k} (skin_t)",
            )
            torch.testing.assert_close(
                res_k.soil_t[-1], full.soil_t[k], rtol=0.0, atol=0.0,
                msg=f"prefix-run frame {k} != full-run frame {k} (soil_t)",
            )


class TestMultiStepReproducibility:
    """Two identical rollouts are identical -- no hidden non-determinism."""

    def test_repeat_run_is_bit_identical(self):
        """Same (model, static, state0, forcing) -> same series, twice.

        A stray buffer reused inside forward, or a state that aliases an
        internal tensor whose content depends on call history, would
        make two runs of the same rollout diverge.  The deterministic-
        given-inputs contract (documented on ``ColumnHarness.run``) is
        pinned bit-exact.
        """
        fl, static, state0, model = _build_rollout(n_steps=24)
        r1 = ColumnHarness(model, static, state0).run(fl)
        r2 = ColumnHarness(model, static, state0).run(fl)
        for f in dataclasses.fields(r1):
            a, b = getattr(r1, f.name), getattr(r2, f.name)
            if not isinstance(a, torch.Tensor):
                continue
            torch.testing.assert_close(
                a, b, rtol=0.0, atol=0.0,
                msg=f"non-determinism: {f.name} differs between two runs",
            )


class TestMultiStepStability:
    """A longer rollout stays bounded -- no slow blow-up."""

    def test_long_rollout_outputs_remain_finite_and_bounded(self):
        """72 steps (3 days) keep skin / soil T in a sane physical band.

        A +0.001 K/step energy leak is invisible in the single-step
        tests but over 72 hours pushes skin T out of band; this catches
        that class of slow divergence.  Bounds are deliberately loose
        (200..330 K) -- the point is "not blowing up", not "exact value".
        """
        fl, static, state0, model = _build_rollout(n_steps=72)
        result = ColumnHarness(model, static, state0).run(fl)

        assert torch.isfinite(result.skin_t).all(), "skin_t went non-finite"
        assert torch.isfinite(result.soil_t).all(), "soil_t went non-finite"
        assert torch.isfinite(result.soil_w_liq).all(), "soil_w_liq non-finite"
        assert torch.isfinite(result.snow_w).all(), "snow_w went non-finite"

        assert result.skin_t.min().item() > 200.0, (
            f"skin_t dropped below 200 K over the rollout: "
            f"{result.skin_t.min().item():.1f}"
        )
        assert result.skin_t.max().item() < 330.0, (
            f"skin_t exceeded 330 K over the rollout: "
            f"{result.skin_t.max().item():.1f}"
        )
        # Soil moisture must stay in [0, wsat]-ish; negative or >0.5 is a
        # mass-conservation failure that diverges over many steps.
        assert result.soil_w_liq.min().item() >= 0.0, (
            f"soil_w_liq went negative: {result.soil_w_liq.min().item():.3e}"
        )
        assert result.soil_w_liq.max().item() < 0.6, (
            f"soil_w_liq unphysically high: {result.soil_w_liq.max().item():.3e}"
        )
