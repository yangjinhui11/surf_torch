"""Direct SEB sensitivity to net-LW: run vdfdifh1s with pslrfl = -86.8 vs -117.6
and compare per-tile skin-T.  Resolves whether the bisect's pslrfl perturbation
actually propagated, and quantifies the true sensitivity.
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


def capture_inp(static):
    cfg = _matching_config(); fl, _b, init = build_idealized_diurnal(cfg); f = fl[0]
    dtype = torch.float64; device = torch.device('cpu')
    model = LandSurface(SurfConfig(dtype=dtype, device=device),
                        consts=PhysicalConstants())
    cap = {}
    orig = model_mod.compute_vdfdifh1s
    def spy(inp, consts, config):
        cap['inp'] = inp; out = orig(inp, consts, config); cap['out'] = out; return out
    model_mod.compute_vdfdifh1s = spy
    try:
        with torch.no_grad(): ns, _ = model(init, f, static)
    finally: model_mod.compute_vdfdifh1s = orig
    return cap['inp'], cap['out'], init, model.consts, model.config


def main():
    dtype = torch.float64; device = torch.device('cpu')
    cfg = _matching_config(); _fl, st, _i = build_idealized_diurnal(cfg)
    veg = VegTable(dtype=dtype, device=device)
    z0m = veg.gather('rvz0m', st.veg_type).to(dtype).to(device)
    z0h = veg.gather('rvz0h', st.veg_type).to(dtype).to(device)
    st_vex = replace(st, z0m=z0m, z0h=z0h, z0q=z0h.clone())

    print("=== Cd=1e-3 path (default) ===")
    inp, out, init, cb, cf = capture_inp(st)
    frac = st.tile_frac_base[0].cpu().numpy()
    skin_base = out.ptsktip1[0].cpu().numpy()
    gmin = float(init.skin_t[0, 0].item())
    names = ["1:water","2:ice","3:wet","4:dryLow","5:snowLow","6:dryHigh","7:snowHigh","8:bare"]

    for lwval, label in [(-86.81144412600162, "PyTorch (Tair basis)"),
                          (-117.6312, "Fortran (skin 288 basis)")]:
        ip = replace(inp, pslrfl=torch.full_like(inp.pslrfl, lwval))
        with torch.no_grad():
            o = diff_mod.vdfdifh1s(ip, cb, cf)
        sk = o.ptsktip1[0].cpu().numpy()
        print(f"\npslrfl = {lwval:.3f} ({label})")
        print(f"{'tile':<12} {'Tout':>8} {'dT':>8}  {'LWout':>8}")
        for j in range(8):
            pstr = o.pstr[0, j].item() if hasattr(o, 'pstr') else float('nan')
            print(f"{names[j]:<12} {sk[j]:8.3f} {sk[j]-gmin:+8.3f}  {pstr:8.3f}")
        tot = float(frac.sum())
        gm = float((sk*frac).sum()/tot)
        print(f"  gridpoint-mean Δskin = {gm-gmin:+.4f} K")

    print("\n\n=== VEXCS path ===")
    inp, out, init, cb, cf = capture_inp(st_vex)
    frac = st_vex.tile_frac_base[0].cpu().numpy()
    gmin = float(init.skin_t[0, 0].item())

    for lwval, label in [(-86.81144412600162, "PyTorch (Tair basis)"),
                          (-117.6312, "Fortran (skin 288 basis)")]:
        ip = replace(inp, pslrfl=torch.full_like(inp.pslrfl, lwval))
        with torch.no_grad():
            o = diff_mod.vdfdifh1s(ip, cb, cf)
        sk = o.ptsktip1[0].cpu().numpy()
        print(f"\npslrfl = {lwval:.3f} ({label})")
        print(f"{'tile':<12} {'Tout':>8} {'dT':>8}")
        for j in range(8):
            print(f"{names[j]:<12} {sk[j]:8.3f} {sk[j]-gmin:+8.3f}")
        tot = float(frac.sum())
        gm = float((sk*frac).sum()/tot)
        print(f"  gridpoint-mean Δskin = {gm-gmin:+.4f} K")

    # Also dump the pg0 (ground heat) and pslh (latent) to see SEB balance
    print("\n\n=== SEB flux budget (Cd=1e-3, pslrfl=-117.6) ===")
    inp2, _, _, cb2, cf2 = capture_inp(st)
    ip = replace(inp2, pslrfl=torch.full_like(inp2.pslrfl, -117.6312))
    with torch.no_grad():
        o = diff_mod.vdfdifh1s(ip, cb2, cf2)
    print(f"{'tile':<12} {'pssh(H)':>9} {'pslh(LE)':>9} {'pstr(LW)':>9} {'pg0(G)':>9}")
    for j in range(8):
        if frac[j] > 1e-6:
            print(f"{names[j]:<12} {o.pssh[0,j]:9.3f} {o.pslh[0,j]:9.3f} "
                  f"{o.pstr[0,j]:9.3f} {o.pg0[0,j]:9.3f}")


if __name__ == "__main__":
    main()
