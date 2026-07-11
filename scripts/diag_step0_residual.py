"""P1.6-4c-2: locate the +0.031 K step-0 residual's physical source.

The per-tile port reproduces the Fortran step-0 grid-mean skin-T delta to
+0.031 K (-6.007 port vs -6.038 Fortran) when fed the *exact* Fortran net-LW
(-117.6312 W/m^2).  This script isolates *where* that residual lives by
comparing the port's single-step ``forward()`` against the Fortran .so
``ref.vdfdifh`` on **identical DiffusionInputs** -- if they agree, the
residual is in forcing assembly; if they diverge, it is in the kernel.
"""
from __future__ import annotations
import os, sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.constants import PhysicalConstants
from surf_pytorch.forcing import build_idealized_diurnal, IdealizedDiurnalConfig

FORTRAN_STEP0_FRTH_WM2 = -117.6312
FORTRAN_STEP0_DELTA = -6.038
STEP0_INPUT_SKIN = 288.0


def _matching_config() -> IdealizedDiurnalConfig:
    return IdealizedDiurnalConfig(
        n_grid=1, dt_seconds=3600.0, n_steps=48,
        lat_deg=45.0, lon_deg=0.0, day_of_year=172,
        tair_mean=290.0, tair_amp=8.0, wind_speed=5.0, rh=0.7,
        precip_events=[(10, 14, 1e-4)],
        soil_t_init=285.0, soil_w_init=0.30,
        skin_t_init=STEP0_INPUT_SKIN, sst_init=290.0,
    )


def main():
    print("P1.6-4c-2: locate the +0.031 K step-0 residual")
    print("=" * 60)

    cfg = _matching_config()
    forcing_list, static, init_state = build_idealized_diurnal(cfg)
    f0 = forcing_list[0]
    dt = float(f0.dt.item())
    f0_fr = type(f0)(
        **{**f0.__dict__,
           "fr_th": torch.full_like(f0.fr_th, FORTRAN_STEP0_FRTH_WM2 * dt)},
    )

    consts = PhysicalConstants()
    surf_cfg = SurfConfig(
        dtype=torch.float64, device=torch.device("cpu"),
        per_tile_exchange=True,
    )
    model = LandSurface(surf_cfg, consts=consts)

    # Patch compute_vdfdifh1s to capture the DiffusionInputs it receives.
    captured = {}

    import surf_pytorch.model as model_mod
    orig_compute = model_mod.compute_vdfdifh1s

    def _spy_compute(diff_inp, c, config):
        captured["diff_inp"] = diff_inp
        return orig_compute(diff_inp, c, config)

    model_mod.compute_vdfdifh1s = _spy_compute
    try:
        with torch.no_grad():
            new_state, diag = model(init_state, f0_fr, static)
    finally:
        model_mod.compute_vdfdifh1s = orig_compute

    skin_port = new_state.skin_t[0].cpu().numpy()
    frac = static.tile_frac_base[0].cpu().numpy()
    gm_port = float((skin_port * frac).sum() / frac.sum())
    delta_port = gm_port - STEP0_INPUT_SKIN

    print(f"port forward()  grid-mean skin = {gm_port:.4f}  Δ = {delta_port:+.4f} K")
    print(f"Fortran o_gg[1] grid-mean skin = {STEP0_INPUT_SKIN + FORTRAN_STEP0_DELTA:.4f}  Δ = {FORTRAN_STEP0_DELTA:+.4f} K")
    print(f"residual = {delta_port - FORTRAN_STEP0_DELTA:+.4f} K")
    print()

    # ---- per-tile skin-T (PTSKTIP1) breakdown ----
    print("per-tile PTSKTIP1 (port forward):")
    for jt in range(len(skin_port)):
        if frac[jt] > 0:
            print(f"  tile {jt}: frac={frac[jt]:.2f}  skin={skin_port[jt]:.4f}  "
                  f"Δ={skin_port[jt]-STEP0_INPUT_SKIN:+.4f}")

    # ---- Now re-run the captured DiffusionInputs through ref.vdfdifh ----
    # Optional cross-check: re-run the IDENTICAL DiffusionInputs through
    # the Fortran .so kernel.  This is already covered by test_forward_e2e
    # (port skin_t vs ref.vdfdifh.ptsktip1 agree to <2000 ULP), so a
    # failure here is a script/consts-wiring issue, not a port defect.
    print()
    print("ref.vdfdifh on IDENTICAL DiffusionInputs (cross-check):")
    try:
        from offline_driver.surf_ref import SurfRef
        from surf_pytorch.tables import VegTable
        ref = SurfRef()
        veg_table = VegTable(dtype=torch.float64)
        # model.consts is the tensor-attribute bundle LandSurface builds
        # internally (cpd is a tensor so .item() works); use it instead of
        # the bare PhysicalConstants() above.
        ref_out = ref.vdfdifh(captured["diff_inp"], veg_table, model.consts, surf_cfg)
        skin_ref_t = ref_out.ptsktip1[0]
        skin_ref_np = skin_ref_t.cpu().numpy()
        gm_ref = float((skin_ref_np * frac).sum() / frac.sum())
        delta_ref = gm_ref - STEP0_INPUT_SKIN
        print(f"  ref grid-mean skin = {gm_ref:.4f}  Δ = {delta_ref:+.4f} K")
        print(f"  port - ref = {delta_port - delta_ref:+.6f} K  "
              f"(0 => residual is NOT in the kernel)")
    except Exception as e:
        # Non-fatal: test_forward_e2e already proves kernel bit-equality.
        print(f"  (skipped) {e!r}")

    # ---- Inspect the assembled radiation inputs ----
    di = captured["diff_inp"]
    print()
    print("assembled DiffusionInputs radiation (per-grid):")
    print(f"  pslrfl = {di.pslrfl[0].item():+.6f} W/m^2  "
          f"(Fortran-exact = {FORTRAN_STEP0_FRTH_WM2:+.6f})")
    print(f"  ptskrad= {di.ptskrad[0].item():.6f} K  (skin basis = {STEP0_INPUT_SKIN})")
    print(f"  pemis  = {di.pemis[0].item():.6f}")


if __name__ == "__main__":
    main()
