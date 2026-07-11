"""Multi-step column-by-column run harness for the land-surface model.

Drives :meth:`surf_pytorch.LandSurface.forward` over a forcing time
series with full state feedback (the ``new_state`` returned at step *k*
is the input state at step *k+1*), collecting the prognostic-state and
diagnostic time series into stacked tensors.

The harness is *pure forward* -- it does not differentiate.  Call
:func:`torch.enable_grad` around :meth:`ColumnHarness.run` if a
downstream loss needs to backprop through the unrolled trajectory
(the model itself respects :attr:`SurfConfig.differentiable`).

Outputs
-------
:meth:`ColumnHarness.run` returns a :class:`RunResult` holding the
stacked prognostic time series (``skin_t``, ``soil_t``, ``soil_w_liq``,
``snow_w``, ``skin_water``), the grid-mean diagnostic fluxes
(``h_flux``, ``e_flux``, ``net_rad``, ``ground_heat``) and the time
axis in seconds.  The final :class:`SurfState` is retained so runs can
be chained.  Two exporters cover the validation use-cases without
pulling any new dependency:

* :meth:`RunResult.export_csv` -- a parseable ``numpy.savetxt`` grid-
  mean time series (top-soil T/W, skin T, fluxes) for eyeballing a run;
* :meth:`RunResult.export_snapshot` -- a ``numpy.savez_compressed``
  dump of the full stacked arrays for golden regression (deterministic
  given a fixed seed / config).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .model import LandSurface
from .state import Forcing, Static, SurfState


# --------------------------------------------------------------------------- #
# Run result
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    """Stacked prognostic + diagnostic time series from a column run.

    Every leading axis is *time* (``T`` steps), then grid (``N``), then
    any vertical / tile axis.  ``time`` is in seconds since the run
    start (mid-interval: ``time[k] = (k+0.5)*dt``), matching the
    forcing convention in :mod:`surf_pytorch.forcing`.
    """

    # Time axis (T,) in seconds.
    time: torch.Tensor

    # Prognostic stacks.
    skin_t: torch.Tensor      # (T, N, Ntile)
    soil_t: torch.Tensor      # (T, N, Nsoil)
    soil_w_liq: torch.Tensor  # (T, N, Nsoil)
    snow_w: torch.Tensor      # (T, N)
    skin_water: torch.Tensor  # (T, N)

    # Grid-mean diagnostic flux stacks (positive downward into surface).
    h_flux: torch.Tensor      # (T, N) [W/m^2]
    e_flux: torch.Tensor      # (T, N) [kg/m^2/s]
    net_rad: torch.Tensor     # (T, N) [W/m^2]
    ground_heat: torch.Tensor  # (T, N) [W/m^2]

    # Final state, for chained runs.
    final_state: SurfState

    # Coupling interval (scalar, seconds) -- assumed constant across the run.
    dt: float

    def export_csv(self, path: str) -> None:
        """Write the grid-mean time series to ``path`` as CSV.

        One column per quantity, one row per timestep.  Quantities:
        ``time_s, skin_t_K, soil_t0_K, soil_w0, snow_w_kgm2,
        skin_water_kgm2, h_flux_Wm2, e_flux_kgm2s, net_rad_Wm2,
        ground_heat_Wm2``.  When ``N > 1`` the columns are the grid-mean
        (averaged over columns) -- use :meth:`export_snapshot` for the
        full per-column arrays.
        """
        time = self.time.cpu().numpy()
        skin = self.skin_t.to(torch.float64).mean(dim=-1).cpu().numpy()
        soil_t0 = self.soil_t[..., 0].to(torch.float64).cpu().numpy()
        soil_w0 = self.soil_w_liq[..., 0].to(torch.float64).cpu().numpy()
        snow = self.snow_w.to(torch.float64).cpu().numpy()
        skinw = self.skin_water.to(torch.float64).cpu().numpy()
        h = self.h_flux.to(torch.float64).cpu().numpy()
        e = self.e_flux.to(torch.float64).cpu().numpy()
        nr = self.net_rad.to(torch.float64).cpu().numpy()
        gh = self.ground_heat.to(torch.float64).cpu().numpy()

        # Grid-mean for the 2-D (T, N) diagnostics.
        def _mean1d(x):
            return x.mean(axis=-1) if x.ndim >= 2 else x

        data = np.column_stack([
            time,
            _mean1d(skin),
            _mean1d(soil_t0),
            _mean1d(soil_w0),
            _mean1d(snow),
            _mean1d(skinw),
            _mean1d(h),
            _mean1d(e),
            _mean1d(nr),
            _mean1d(gh),
        ])
        header = ("time_s,skin_t_K,soil_t0_K,soil_w0,snow_w_kgm2,"
                  "skin_water_kgm2,h_flux_Wm2,e_flux_kgm2s,"
                  "net_rad_Wm2,ground_heat_Wm2")
        # comments="#" (default) keeps the header line a comment, so
        # np.loadtxt skips it automatically on round-trip.
        np.savetxt(path, data, delimiter=",", header=header)

    def export_snapshot(self, path: str) -> None:
        """Dump the full stacked arrays to ``path`` as ``.npz``.

        Deterministic given a fixed seed / config; intended as a golden
        regression artifact.  ``np.load(path)`` round-trips back to the
        same arrays.
        """
        np.savez_compressed(
            path,
            time=self.time.cpu().numpy(),
            skin_t=self.skin_t.cpu().numpy(),
            soil_t=self.soil_t.cpu().numpy(),
            soil_w_liq=self.soil_w_liq.cpu().numpy(),
            snow_w=self.snow_w.cpu().numpy(),
            skin_water=self.skin_water.cpu().numpy(),
            h_flux=self.h_flux.cpu().numpy(),
            e_flux=self.e_flux.cpu().numpy(),
            net_rad=self.net_rad.cpu().numpy(),
            ground_heat=self.ground_heat.cpu().numpy(),
            dt=self.dt,
        )


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
class ColumnHarness:
    """Drive :class:`LandSurface.forward` over a forcing time series.

    Parameters
    ----------
    model:
        A built :class:`LandSurface`.
    static:
        The fixed per-column static descriptor (carried unchanged across
        all steps).
    initial_state:
        Cold-start (or warm-start) :class:`SurfState` for step 0.

    Notes
    -----
    The harness stores ``model`` / ``static`` / ``initial_state`` by
    reference; :meth:`run` does not mutate them (``forward`` returns a
    fresh state each step, so the caller's ``initial_state`` is safe).
    """

    def __init__(
        self,
        model: LandSurface,
        static: Static,
        initial_state: SurfState,
    ) -> None:
        self.model = model
        self.static = static
        self.initial_state = initial_state

    def run(self, forcing_list: List[Forcing]) -> RunResult:
        """Advance ``n_steps == len(forcing_list)`` coupling intervals.

        Returns a :class:`RunResult` with the stacked prognostic /
        diagnostic time series.  The run is deterministic given the
        model, static, initial state and forcing list -- no internal
        randomness.
        """
        if not forcing_list:
            raise ValueError("forcing_list must be non-empty")

        dt = float(forcing_list[0].dt.flatten()[0].item())
        n_steps = len(forcing_list)
        ngrid = self.static.tile_frac_base.shape[0]

        # Pre-allocate the stacks from the first state to fix shapes / dtype.
        skin_t0 = self.initial_state.skin_t
        soil_t0 = self.initial_state.soil_t
        ntile = skin_t0.shape[-1]
        nsoil = soil_t0.shape[-1]
        dtype = skin_t0.dtype
        device = skin_t0.device

        skin_t = torch.empty((n_steps, ngrid, ntile), dtype=dtype,
                             device=device)
        soil_t = torch.empty((n_steps, ngrid, nsoil), dtype=dtype,
                             device=device)
        soil_w_liq = torch.empty((n_steps, ngrid, nsoil), dtype=dtype,
                                 device=device)
        snow_w = torch.empty((n_steps, ngrid), dtype=dtype, device=device)
        skin_water = torch.empty((n_steps, ngrid), dtype=dtype, device=device)
        h_flux = torch.empty((n_steps, ngrid), dtype=dtype, device=device)
        e_flux = torch.empty((n_steps, ngrid), dtype=dtype, device=device)
        net_rad = torch.empty((n_steps, ngrid), dtype=dtype, device=device)
        ground_heat = torch.empty((n_steps, ngrid), dtype=dtype, device=device)

        state = self.initial_state
        for k, f in enumerate(forcing_list):
            state, diag = self.model(state, f, self.static)
            skin_t[k] = state.skin_t
            soil_t[k] = state.soil_t
            soil_w_liq[k] = state.soil_w_liq
            snow_w[k] = state.snow_w
            skin_water[k] = state.skin_water
            h_flux[k] = diag.h_flux
            e_flux[k] = diag.e_flux
            net_rad[k] = diag.net_rad
            ground_heat[k] = diag.ground_heat

        time = (torch.arange(n_steps, dtype=dtype, device=device) + 0.5) * dt

        return RunResult(
            time=time,
            skin_t=skin_t, soil_t=soil_t, soil_w_liq=soil_w_liq,
            snow_w=snow_w, skin_water=skin_water,
            h_flux=h_flux, e_flux=e_flux,
            net_rad=net_rad, ground_heat=ground_heat,
            final_state=state,
            dt=dt,
        )


__all__ = ["ColumnHarness", "RunResult"]
