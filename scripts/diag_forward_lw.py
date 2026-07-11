"""P1.6-3h: confirm the net-LW sensitivity on the FULL forward() path.

``diag_lw_sensitivity.py`` proved the sensitivity at the *kernel* level by
spying on ``compute_vdfdifh1s`` and re-running it with a patched ``pslrfl``.
This script goes one layer up: it runs the production
``LandSurface.forward()`` with two different ``fr_th`` values built into the
forcing, confirming the 0.77 K net-LW effect survives the model's own
``_build_diffusion_inputs`` assembly (which divides ``fr_th`` by ``dt`` and
threads it through ``slrfl_grid``).

Result table (expected, from the kernel-level diag):
  Cd=1e-3 + fr_th(Tair)   -> gridmean Δskin ~ -4.55 K   (current default)
  Cd=1e-3 + fr_th(skin)   -> gridmean Δskin ~ -5.32 K   (Fortran convention)

The latter is what the Fortran driver computes (callpar1s.F90:812-813,
``PFRTH = ZEMIS*(PSRTHD - sigma*PTL^4)`` with PTL = skin 288 K).
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.constants import PhysicalConstants
from surf_pytorch.forcing import build_idealized_diurnal, IdealizedDiurnalConfig


def _matching_config():
    """Match the step-0 forcing used by TestSingleStepPhysics."""
    return IdealizedDiurnalConfig(
        n_grid=1, dt_seconds=3600.0, n_steps=48,
        lat_deg=45.0, lon_deg=0.0, day_of_year=172,
        tair_mean=290.0, tair_amp=8.0, wind_speed=5.0, rh=0.7,
        precip_events=[(10, 14, 1e-4)],
        soil_t_init=285.0, soil_w_init=0.30,
        skin_t_init=288.0, sst_init=290.0,
    )


def _step0_fr_th(cfg: IdealizedDiurnalConfig, lw_basis: str) -> float:
    """Compute the step-0 net-LW flux (W/m^2) under a given basis.

    ``tair``  -- matches forcing.py ``_net_longwave`` (skin proxy = Tair).
    ``skin``  -- matches the Fortran callpar1s convention (PTL = skin 288).

    Step 0 is the first half-hour [0, 1 h] UTC of day 172 at lon=0, so the
    mid-interval local solar hour is 0.5 h -- well before the 14:00 peak,
    giving the documented night-time Tair ~ 282.6 K.
    """
    bundle = PhysicalConstants()
    sigma = float(bundle.rsigma)
    dt = float(cfg.dt_seconds)
    # mid-interval local solar hour for step 0 (k=0): (0+0.5)*dt/3600 + lon/15
    hour_utc = (0 + 0.5) * dt / 3600.0
    hour_local = hour_utc + cfg.lon_deg / 15.0
    phase = 2.0 * np.pi * (hour_local - cfg.tair_peak_local_hour) / 24.0
    tair = cfg.tair_mean + cfg.tair_amp * np.cos(phase)
    tskin = cfg.skin_t_init  # 288.0
    tb = tair if lw_basis == "tair" else tskin
    lw_down = cfg.atm_emis * sigma * tb ** 4
    lw_up = cfg.emis_sfc * sigma * tb ** 4
    return lw_down - lw_up  # W/m^2 (down-positive)


def _run_forward(fr_th_wm2: float) -> float:
    """Run one forward() step with a forced net-LW and return the
    tile-fraction-weighted grid-mean skin-T delta (K).

    Rebuilds the forcing so step 0 carries the requested net-LW, then runs
    ``LandSurface.forward`` once.  The grid-mean is weighted by the
    PyTorch default tile fractions (matching TestSingleStepPhysics).
    """
    cfg = _matching_config()
    forcing_list, static, init_state = build_idealized_diurnal(cfg)
    f0 = forcing_list[0]

    # Override the step-0 net-LW (cumulative = rate * dt).
    dt = float(f0.dt.item())
    f0_fr = type(f0)(
        **{**f0.__dict__, "fr_th": torch.full_like(f0.fr_th, fr_th_wm2 * dt)},
    )

    model = LandSurface(SurfConfig(dtype=torch.float64, device=torch.device('cpu')),
                        consts=PhysicalConstants())
    with torch.no_grad():
        new_state, _diag = model(init_state, f0_fr, static)

    skin = new_state.skin_t[0].cpu().numpy()           # (8,)
    frac = static.tile_frac_base[0].cpu().numpy()      # (8,)
    gm = float((skin * frac).sum() / frac.sum())
    return gm - cfg.skin_t_init


def main():
    print("P1.6-3h: net-LW sensitivity on the FULL forward() path")
    print("=" * 64)
    cfg = _matching_config()

    fr_th_tair = _step0_fr_th(cfg, "tair")
    fr_th_skin = _step0_fr_th(cfg, "skin")
    print(f"step-0 Tair basis -> fr_th = {fr_th_tair:.4f} W/m^2  (cumulative {fr_th_tair*3600:.2f} J/m^2)")
    print(f"step-0 skin basis -> fr_th = {fr_th_skin:.4f} W/m^2  (cumulative {fr_th_skin*3600:.2f} J/m^2)")
    print(f"  net-LW divergence = {fr_th_skin - fr_th_tair:+.4f} W/m^2")
    print()

    # The Fortran's ACTUAL step-0 value: callpar1s:812-813
    #   PFRTH = ZEMIS*(PSRTHD - sigma*PTL^4)  with ZEMIS=0.99, PTL=skin 288,
    #   PSRTHD (forcing LWdown) = 271.2858 from the NetCDF.
    # This is NOT the same as the generator's "skin basis" (-0.24*sigma*288^4)
    # because the Fortran applies emissivity to the WHOLE net, not separately
    # to LWdown (0.75) and LWup (0.99).  Test the exact Fortran value too.
    fr_th_fortran = -117.6312

    print(f"{'basis':<16} {'fr_th W/m^2':>12} {'gridmean skin':>14} {'Δskin K':>10}")
    print("-" * 56)
    for basis, fr_th in [("tair (gen)", fr_th_tair),
                         ("skin (gen)", fr_th_skin),
                         ("Fortran exact", fr_th_fortran)]:
        dskin = _run_forward(fr_th)
        skin = cfg.skin_t_init + dskin
        print(f"{basis:<16} {fr_th:12.4f} {skin:14.4f} {dskin:+10.4f}")

    print()
    d_tair = _run_forward(fr_th_tair)
    d_skin = _run_forward(fr_th_skin)
    d_ft = _run_forward(fr_th_fortran)
    print(f"generator skin-basis effect (skin - tair)   = {d_skin - d_tair:+.4f} K")
    print(f"Fortran-value effect (exact - tair)         = {d_ft - d_tair:+.4f} K")
    print(f"Fortran step-0 delta (o_gg[1] - o_gg[0])     = {-6.038:+.4f} K")
    print(f"residual gap after Fortran-value correction  = {d_ft - (-6.038):+.4f} K")


if __name__ == "__main__":
    main()
