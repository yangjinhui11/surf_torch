"""Focused diagnostic: single C3 broadleaf column, compare port vs ref at ULP.

Run:  python offline_driver/_dbg_farquhar.py
This is a developer tool, not part of the test suite.  It exercises the
real Farquhar port against the independent Fortran reference on a single
well-posed C3 broadleaf-forest column and prints the per-grid scalars
and per-level assimilation/conductance so a regression in the port can
be localised quickly.
"""
from __future__ import annotations
import torch
from surf_pytorch import ConstantBundle
from surf_pytorch.constants import PhysicalConstants
from surf_pytorch.tables import VegTable
from surf_pytorch.physics.farquhar import FarquharInputs, farquhar
from surf_ref import SurfRef

torch.manual_seed(0)
# Single broadleaf-forest column (C3), daytime, warm, well-watered.
sw = torch.tensor([700.0], dtype=torch.float64)
t_air = torch.tensor([298.0], dtype=torch.float64)
skin_t = torch.tensor([299.0], dtype=torch.float64)
t_soil = torch.tensor([295.0], dtype=torch.float64)
q_air = torch.tensor([0.010], dtype=torch.float64)
qs_skin = torch.tensor([0.014], dtype=torch.float64)
p_surf = torch.tensor([101325.0], dtype=torch.float64)
co2 = torch.tensor([390.0e-6], dtype=torch.float64)
lai = torch.tensor([4.0], dtype=torch.float64)
f2 = torch.tensor([0.9], dtype=torch.float64)
kvtype = torch.tensor([5], dtype=torch.long)        # broadleaf C3
kco2typ = torch.tensor([1], dtype=torch.long)       # C3
ldland = torch.tensor([True], dtype=torch.bool)

veg = VegTable(dtype=torch.float64)
consts = ConstantBundle(PhysicalConstants(), dtype=torch.float64)
ref = SurfRef()

out_ref = ref.farquhar(
    sw_down=sw, p_surf=p_surf, qs_skin=qs_skin, qsurf=qs_skin, q_air=q_air,
    t_air=t_air, skin_t=skin_t, t_soil=t_soil, f2=f2, co2=co2, lai=lai,
    kvtype=kvtype, kco2typ=kco2typ, ldland=ldland, veg=veg, consts=consts,
    lskin_temp=True,
)
print("REF:  PGSTOT=", out_ref["pgstot"][0].item(), " PGPP=", out_ref["pgpp"][0].item())

inp = FarquharInputs(
    ptskm=skin_t, ptm=t_air, ptsoil=t_soil, paphm=p_surf, pqm=q_air,
    pqsurf=qs_skin, pqs=qs_skin, pco2=co2, lai=lai, kvtype=kvtype,
    kco2typ=kco2typ, pf2=f2, ldland=ldland, psrfd=sw,
)
dbg = {}
out_port = farquhar(inp, veg, consts, debug=dbg)
print("PORT: PGSTOT=", out_port.pgstot[0].item(), " PGPP=", out_port.pgpp[0].item())
print("ratio port/ref PGSTOT:", out_port.pgstot[0].item()/out_ref["pgstot"][0].item())

print("\n--- Port per-grid scalars ---")
for k in ["zvcmax25", "zt_vcmax", "zt_jmax", "zvc", "zvj", "zgm", "zg0var",
          "zkmc", "zkmo", "zsco", "zgamma_star", "zfvpd", "zgbco2", "zvpd",
          "zco2"]:
    v = dbg[k]
    print(f"  {k:14s}= {v[0].item():.8e}")

print("\n--- Port per-level ZASSIMI / ZGS (active levels) ---")
la = dbg["level_active"][:, 0]
for j in range(la.shape[0]):
    if la[j]:
        a = dbg["zassimi_lev"][j, 0].item()
        g = dbg["zgs_lev"][j, 0].item()
        rd = dbg["zrd_lev"][j, 0].item()
        dl = dbg["dlai"][j].item()
        print(f"  jlevel={j+1:2d}  ZASSIMI={a:14.6e}  ZGS={g:14.6e}  "
              f"ZRD={rd:14.6e}  dlai={dl:.6f}")
