"""Bisect which DiffusionInputs field drives the skin-T gap.

forward() with VEXCS gives Δskin=-3.61; Fortran coupled gives -6.04.
Same DiffusionInputs -> port==Fortran exactly (0.0). So the gap is in
an input VALUE.  This script takes the captured step-0 DiffusionInputs
and re-runs SURFSEB while perturbing each field toward plausible
Fortran-side values, to find which one moves Δskin toward -6.04.

Strategy: for each candidate field, set it to a "Fortran-like" value and
measure the Δskin response.  The field whose change closes the gap is
the culprit.
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

FORTRAN_DELTA = -6.038

def _matching_config():
    return IdealizedDiurnalConfig(n_grid=1, dt_seconds=3600.0, n_steps=48,
        lat_deg=45.0, lon_deg=0.0, day_of_year=172, tair_mean=290.0,
        tair_amp=8.0, wind_speed=5.0, rh=0.7, precip_events=[(10,14,1e-4)],
        soil_t_init=285.0, soil_w_init=0.30, skin_t_init=288.0, sst_init=290.0)

def gm(t, frac):
    a=t.detach().cpu().numpy(); fr=frac.detach().cpu().numpy()
    if a.ndim==2: return float(((a*fr).sum(-1)/fr.sum(-1)).reshape(-1)[0])
    return float(np.asarray(a).reshape(-1)[0])

def capture_inp(static, dt_default):
    cfg=_matching_config(); fl,_b,init=build_idealized_diurnal(cfg); f=fl[0]
    dtype=torch.float64; device=torch.device('cpu')
    model=LandSurface(SurfConfig(dtype=dtype,device=device),consts=PhysicalConstants())
    cap={}
    orig=model_mod.compute_vdfdifh1s
    def spy(inp,consts,config):
        cap['inp']=inp; return orig(inp,consts,config)
    model_mod.compute_vdfdifh1s=spy
    try:
        with torch.no_grad(): ns,_=model(init,f,static)
    finally: model_mod.compute_vdfdifh1s=orig
    return cap['inp'], init, model.consts, model.config, static.tile_frac_base

def main():
    dtype=torch.float64; device=torch.device('cpu')
    cfg=_matching_config(); _fl,st,_i=build_idealized_diurnal(cfg)
    veg=VegTable(dtype=dtype,device=device)
    z0m=veg.gather('rvz0m',st.veg_type).to(dtype).to(device)
    z0h=veg.gather('rvz0h',st.veg_type).to(dtype).to(device)
    st_vex=replace(st,z0m=z0m,z0h=z0h,z0q=z0h.clone())

    inp, init, cb, cf, frac = capture_inp(st_vex, None)
    gmin=gm(init.skin_t,frac)
    print(f"baseline (VEXCS) Δskin = {gm(_run(inp,cb,cf),frac)-gmin:+.3f}")
    print(f"target Fortran       Δskin = {FORTRAN_DELTA:+.3f}")
    print()

    # Candidate perturbations toward "Fortran-like" values.
    sig=5.670374419e-8
    candidates = []

    # 1. net-LW: Fortran uses PTL for LWup (colder -> stronger cooling)
    emis=0.99; lwdown=271.2793464123928; ptl=288.0
    pfrth = emis*(lwdown - sig*ptl**4)
    candidates.append(("pslrfl = Fortran PFRTH (-117.6)", "pslrfl",
        torch.full_like(inp.pslrfl, pfrth)))

    # 2. plamsk: tile fractions are [0,0,0,0.27,0,0.63,0,0.10]; high-veg tile
    #    (5) dominates with 0.63. What if plamsk there is larger (more ground
    #    coupling -> skin pulled toward colder 285K soil)?
    # bare-soil veg type 8 -> lamsk=15; check current plamsk per tile:
    pm = inp.plamsk[0].tolist()
    print("current plamsk per tile:", [f"{x:.2f}" for x in pm])
    print("tile fractions        :", [f"{x:.2f}" for x in frac[0].tolist()])
    print()

    # 3. rhoc_u much smaller (like Cd=1e-3 -> rhoc~0.006) but applied uniformly
    rg=9.80665; dt=3600.0; RVDIFTS=1.5
    for rho in [0.0062, 0.05, 0.2, 0.9]:
        scaled = rho*rg*dt*RVDIFTS
        candidates.append((f"rhoc_u=rhoc_q={rho}", "pcfqti",
            torch.full_like(inp.pcfqti, scaled)))

    # 4. alphas/alphal: maybe Fortran uses different stomatal closure
    #    (dry night: alphas small). Try alphas=alphal=0.0 (no unstressed)
    candidates.append(("alphas=alphal=0.0", None,
        {"alphas":torch.zeros_like(inp.alphas),"alphal":torch.zeros_like(inp.alphal)}))

    # 5. tskrad colder (radiation base T = tair 282.6 instead of skin 288)
    candidates.append(("ptskrad=282.6 (tair)", "ptskrad",
        torch.full_like(inp.ptskrad, 282.6)))

    print(f"{'perturbation':<32} {'Δskin':>8}  {'toward target?':>14}")
    print("-"*58)
    for name, key, val in candidates:
        if key is None:
            ip = replace(inp, **val)
        else:
            ip = replace(inp, **{key: val})
        d = gm(_run(ip,cb,cf),frac) - gmin
        arrow = "<-- closes gap" if abs(d-FORTRAN_DELTA)<abs(gm(_run(inp,cb,cf),frac)-gmin-FORTRAN_DELTA) else ""
        print(f"{name:<32} {d:+8.3f}  {arrow:>14}")

def _run(inp, cb, cf):
    return diff_mod.vdfdifh1s(inp, cb, cf).ptsktip1

if __name__=="__main__": main()
