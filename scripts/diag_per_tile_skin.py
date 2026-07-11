"""Per-tile skin-T decomposition + SEB flux budget at step 0.

Goal: understand WHY the gridpoint-mean Δskin is only -3.6K in PyTorch
but -6.04K in Fortran, even though port==Fortran on identical inputs.

Approach: capture the step-0 DiffusionInputs and OUT per-tile skin-T,
then print the SEB per tile (radiation, sensible, latent, ground) to see
whether every tile cools uniformly or whether one tile dominates the mean.
"""
from __future__ import annotations
import os, sys
from dataclasses import replace
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.constants import PhysicalConstants
from surf_pytorch.forcing import build_idealized_diurnal, IdealizedDiurnalConfig
from surf_pytorch.tables.veg import VegTable
import surf_pytorch.model as model_mod
from surf_pytorch.physics import diffusion as diff_mod


def _matching_config():
    return IdealizedDiurnalConfig(n_grid=1, dt_seconds=3600.0, n_steps=48,
        lat_deg=45.0, lon_deg=0.0, day_of_year=172, tair_mean=290.0,
        tair_amp=8.0, wind_speed=5.0, rh=0.7, precip_events=[(10, 14, 1e-4)],
        soil_t_init=285.0, soil_w_init=0.30, skin_t_init=288.0, sst_init=290.0)


def run_capture(static_label, static):
    """Run one step with DiffusionInputs capture, return (inp, out_skin_t)."""
    cfg = _matching_config()
    fl, _b, init = build_idealized_diurnal(cfg)
    f = fl[0]
    dtype = torch.float64
    device = torch.device('cpu')
    model = LandSurface(SurfConfig(dtype=dtype, device=device),
                        consts=PhysicalConstants())
    cap = {}
    orig = model_mod.compute_vdfdifh1s

    def spy(inp, consts, config):
        cap['inp'] = inp
        out = orig(inp, consts, config)
        cap['out'] = out
        return out

    model_mod.compute_vdfdifh1s = spy
    try:
        with torch.no_grad():
            ns, _ = model(init, f, static)
    finally:
        model_mod.compute_vdfdifh1s = orig
    return cap['inp'], cap['out'], init, model.consts, model.config


def main():
    dtype = torch.float64
    device = torch.device('cpu')
    cfg = _matching_config()
    _fl, st, _i = build_idealized_diurnal(cfg)

    # --- Two static configs: Cd=1e-3 (default, z0=None) and VEXCS (z0 set) ---
    veg = VegTable(dtype=dtype, device=device)
    z0m = veg.gather('rvz0m', st.veg_type).to(dtype).to(device)
    z0h = veg.gather('rvz0h', st.veg_type).to(dtype).to(device)
    st_vex = replace(st, z0m=z0m, z0h=z0h, z0q=z0h.clone())

    sig = 5.670374419e-8

    for label, static in [("Cd=1e-3 (default z0=None)", st),
                           ("VEXCS (z0 from veg table)", st_vex)]:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        inp, out, init, cb, cf = run_capture(label, static)
        skin_in = init.skin_t[0].cpu().numpy()
        skin_out = out.ptsktip1[0].cpu().numpy()
        dskin = skin_out - skin_in
        frac = static.tile_frac_base[0].cpu().numpy()
        names = ["1:water", "2:ice", "3:wet", "4:dryLow", "5:snowLow",
                 "6:dryHigh", "7:snowHigh", "8:bare"]

        print(f"\n{'tile':<12} {'frac':>6} {'Tin':>8} {'Tout':>8} {'dT':>8}")
        print("-" * 48)
        for j in range(8):
            star = " <-- nonzero frac" if frac[j] > 1e-6 else ""
            print(f"{names[j]:<12} {frac[j]:6.3f} {skin_in[j]:8.3f} "
                  f"{skin_out[j]:8.3f} {dskin[j]:+8.3f}{star}")
        tot = float((frac).sum())
        gm_in = float((skin_in * frac).sum() / tot)
        gm_out = float((skin_out * frac).sum() / tot)
        print(f"\ngridpoint-mean skin: {gm_in:.3f} -> {gm_out:.3f} "
              f"(Δ={gm_out-gm_in:+.3f} K)")

        # SEB per tile from DiffusionInputs
        print(f"\n--- SEB per tile (from DiffusionInputs) ---")
        pslrfl = inp.pslrfl[0].cpu().numpy()
        print(f"  pslrfl (net LW per tile): {pslrfl}")
        if hasattr(inp, 'pssrfl'):
            print(f"  pssrfl (net SW per tile): {inp.pssrfl[0].cpu().numpy()}")

        # print all DiffusionInputs fields with shapes
        print(f"\n--- DiffusionInputs fields ---")
        for fname in sorted(inp.__dataclass_fields__):
            v = getattr(inp, fname)
            if hasattr(v, 'shape') and v.numel() <= 16:
                print(f"  {fname}: {v.cpu().numpy().ravel().tolist()}")


if __name__ == "__main__":
    main()
