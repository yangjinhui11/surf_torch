"""Pin the step-0 skin-temperature advance against the Fortran single-step.

This is the regression guard for the **per-tile exchange** fix (P1.6-4b).
The original grid-mean port gave a step-0 grid-mean skin-T delta of only
~-3.6 K against the offline driver's documented -6.04 K -- a -2.4 K
single-step physics gap that compounded into the multi-step drift.  The
root cause was that the port broadcast one grid-mean ``rho*Ch*|U|`` across
every tile, whereas the upstream offline driver (``surfexcdrivers_ctl_mod.
F90:388-398``) calls ``VEXCS`` once per tile with tile-specific roughness
lengths (``vupdz0s_mod.F90``) and tile-specific surface state.  Driving
``VEXCS`` per-tile (a single vectorised call on the flattened
``(Ngrid, ntile)`` axis -- the kernel is shape-agnostic) closes most of
the gap.

The Fortran reference
---------------------
The offline driver's step 0 is the first half-hour [0, 1 h] UTC of day 172
at lon=0 (a documented night-time point: skin 288 K, tair ~282.6 K,
SW=0).  The driver records the skin-T after step 0 as 281.962 K, so the
Fortran single-step delta is::

    o_gg[1] - o_gg[0] = 281.962 - 288.0 = -6.038 K

The dominant term in that step is the net longwave flux.  ``callpar1s``
(:812-813) computes it as ``PFRTH = ZEMIS*(PSRTHD - sigma*PTL^4)`` with
``ZEMIS=0.99`` (the offline-driver default ``REMISS``, ``surdi1s.F90:12``),
``PTL=skin=288`` and the forcing LW-down ``PSRTHD`` read from the NetCDF.
Feeding that *exact* net-LW (``-117.6312 W/m^2``) into the port's
``LandSurface.forward`` currently reproduces the Fortran single step to
~0.25 K, while the old grid-mean exchange path remains ~1.94 K away.

The residual is **not** a port defect by itself: the radiation assembly is
bit-exact (``scripts/diag_step0_residual.py`` shows ``pslrfl``/``ptskrad``/
``pemis`` match Fortran) and the kernel is bit-equal to ``ref.vdfdifh``
(``test_forward_e2e.py``: <2000 ULP).  The gap is the difference between
the differentiable single-step ``forward()`` (vdfdifh1s + skin-T
extrapolation only) and the full offline-driver step 0, which also runs
``cpg1s`` radiation and the full-model lowest-level Dirichlet coupling.
See ``docs/PROGRESS.md`` P1.6-4c for the full investigation.

These tests are self-contained: they do not need the coupled Fortran run
(``OIFS_OGG_PATH``).  They use the port's own ``LandSurface.forward`` with
the verified Fortran net-LW, so they run on every commit.
"""

from __future__ import annotations

import torch

from surf_pytorch import IdealizedDiurnalConfig, build_idealized_diurnal
from surf_pytorch.constants import PhysicalConstants
from surf_pytorch.model import LandSurface
from surf_pytorch.config import SurfConfig


# --------------------------------------------------------------------------- #
# The documented offline-driver step-0 reference values.
# --------------------------------------------------------------------------- #
# Net longwave the Fortran applies at step 0 (callpar1s:812-813).
FORTRAN_STEP0_FRTH_WM2 = -117.6312

# Cold-start skin T and the driver-recorded post-step-0 skin T (o_gg[1]).
STEP0_INPUT_SKIN = 288.0
STEP0_OGG_OUT_SKIN = 281.962

# Fortran single-step grid-mean skin-T delta (o_gg[1] - o_gg[0]).
FORTRAN_STEP0_DELTA = STEP0_OGG_OUT_SKIN - STEP0_INPUT_SKIN   # = -6.038 K

# Per-tile fix residual: the port reproduces the Fortran single step to
# ~0.25 K.  Investigation (P1.6-4c, docs/PROGRESS.md) shows the radiation
# assembly is bit-exact and the kernel matches ref.vdfdifh to <2000 ULP,
# so this is the single-step-forward vs full-offline-driver-step gap
# (cpg1s radiation + full-model Dirichlet coupling), not a port defect.
# Allow a comfortable band that still catches a regression to the old
# grid-mean ~-3.6 K behaviour (which would miss by >2 K).
PER_TILE_RESIDUAL_K = 0.30


def _matching_config() -> IdealizedDiurnalConfig:
    """The step-0 forcing matching the offline driver's documented point."""
    return IdealizedDiurnalConfig(
        n_grid=1, dt_seconds=3600.0, n_steps=48,
        lat_deg=45.0, lon_deg=0.0, day_of_year=172,
        tair_mean=290.0, tair_amp=8.0, wind_speed=5.0, rh=0.7,
        precip_events=[(10, 14, 1e-4)],
        soil_t_init=285.0, soil_w_init=0.30,
        skin_t_init=STEP0_INPUT_SKIN, sst_init=290.0,
    )


def _step0_delta(fr_th_wm2: float, *, per_tile: bool) -> float:
    """Run one forward() step with a forced net-LW; return grid-mean Δskin.

    Rebuilds the idealized-diurnal step-0 forcing, overrides the net
    longwave to the supplied rate (cumulative = rate * dt), runs one
    ``LandSurface.forward`` step, and returns the tile-fraction-weighted
    grid-mean skin-T delta (K).  ``per_tile`` selects the exchange mode.
    """
    cfg = _matching_config()
    forcing_list, static, init_state = build_idealized_diurnal(cfg)
    f0 = forcing_list[0]
    dt = float(f0.dt.item())
    f0_fr = type(f0)(
        **{**f0.__dict__,
           "fr_th": torch.full_like(f0.fr_th, fr_th_wm2 * dt)},
    )
    surf_cfg = SurfConfig(
        dtype=torch.float64, device=torch.device("cpu"),
        per_tile_exchange=per_tile,
    )
    model = LandSurface(surf_cfg, consts=PhysicalConstants())
    with torch.no_grad():
        new_state, _ = model(init_state, f0_fr, static)
    skin = new_state.skin_t[0].cpu().numpy()           # (8,)
    frac = static.tile_frac_base[0].cpu().numpy()      # (8,)
    gm = float((skin * frac).sum() / frac.sum())
    return gm - cfg.skin_t_init


class TestStep0Physics:
    """The per-tile exchange closes the single-step skin-T gap."""

    def test_per_tile_matches_fortran_step0(self):
        # With per-tile exchange + the Fortran-exact net-LW, the port's
        # step-0 grid-mean skin-T delta must reproduce the Fortran
        # -6.038 K to within the documented residual (~0.25 K, the
        # single-step-forward vs full-driver-step gap -- see module
        # docstring + docs/PROGRESS.md P1.6-4c).
        delta = _step0_delta(FORTRAN_STEP0_FRTH_WM2, per_tile=True)
        gap = abs(delta - FORTRAN_STEP0_DELTA)
        assert gap < PER_TILE_RESIDUAL_K, (
            f"per-tile step-0 Δskin = {delta:+.4f} K, "
            f"Fortran = {FORTRAN_STEP0_DELTA:+.4f} K, "
            f"gap = {gap:.4f} K exceeds {PER_TILE_RESIDUAL_K} K residual "
            f"-- a regression in the per-tile exchange."
        )

    def test_per_tile_beats_grid_mean(self):
        # The per-tile exchange must materially improve on the grid-mean
        # path: the grid-mean delta is ~-3.6 K (a >2 K gap), whereas the
        # per-tile delta is within 0.1 K.  Pin the improvement so a
        # future change that silently disables per-tile is caught.
        delta_per = _step0_delta(FORTRAN_STEP0_FRTH_WM2, per_tile=True)
        delta_grid = _step0_delta(FORTRAN_STEP0_FRTH_WM2, per_tile=False)
        gap_per = abs(delta_per - FORTRAN_STEP0_DELTA)
        gap_grid = abs(delta_grid - FORTRAN_STEP0_DELTA)
        assert gap_per < 0.5 * gap_grid, (
            f"per-tile gap ({gap_per:.3f} K) did not halve the grid-mean "
            f"gap ({gap_grid:.3f} K) -- per-tile exchange is not improving "
            f"the single-step physics as expected."
        )

    def test_grid_mean_legacy_path_is_documented_poor(self):
        # The grid-mean (legacy) path leaves a material single-step gap
        # (~1.9 K under the Fortran-exact net-LW).  Pin it explicitly so
        # the magnitude of the per-tile fix is recorded: this is the
        # "before" baseline.  If the grid-mean path ever starts matching
        # Fortran too, this test will fail loudly as the prompt to
        # revisit the per-tile dispatch.
        delta_grid = _step0_delta(FORTRAN_STEP0_FRTH_WM2, per_tile=False)
        gap_grid = abs(delta_grid - FORTRAN_STEP0_DELTA)
        assert gap_grid > 1.0, (
            f"grid-mean step-0 Δskin = {delta_grid:+.4f} K, gap = "
            f"{gap_grid:.3f} K -- the grid-mean legacy path is no longer "
            f"materially off; re-evaluate whether per_tile_exchange is "
            f"still needed as the default."
        )
