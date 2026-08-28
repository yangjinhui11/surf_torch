import torch

from surf_pytorch.physics.offline_vdf import vdfdifm1s_surface_stress


def test_vdfdifm1s_surface_stress_matches_offline_vdfincr_reduction():
    pcfm = torch.tensor([[0.03, 0.01], [0.02, 0.06]], dtype=torch.float64)
    frac = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float64)
    u = torch.tensor([5.0, -3.0], dtype=torch.float64)
    v = torch.tensor([2.0, 4.0], dtype=torch.float64)
    ut = torch.tensor([1e-4, -2e-4], dtype=torch.float64)
    vt = torch.tensor([3e-4, 1e-4], dtype=torch.float64)

    out = vdfdifm1s_surface_stress(
        pcfm, frac, u, v, dt=1800.0, rg=9.80665,
        u_tendency=ut, v_tendency=vt,
    )

    pcfm_lev = torch.zeros(2, dtype=torch.float64)
    for jt in range(2):
        pcfm_lev = pcfm_lev + pcfm[:, jt] * frac[:, jt]
    zcfm = ((pcfm_lev * 9.80665) * 1800.0) * 1.0
    zudif = (1.0 / 1.0) * u + 1800.0 * ut
    zvdif = (1.0 / 1.0) * v + 1800.0 * vt
    zcons2 = 1.0 / (9.80665 * 1800.0)
    torch.testing.assert_close(out.stress_u, (zcons2 * zcfm) * zudif, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out.stress_v, (zcons2 * zcfm) * zvdif, rtol=0.0, atol=0.0)

    # F90 链与化简形式 PCFM*(U+dt*PVOM) 代数等价，仅差几个 ulp。
    simplified = pcfm_lev * (u + 1800.0 * ut)
    torch.testing.assert_close(out.stress_u, simplified, rtol=1e-13, atol=0.0)


def test_vdfdifm1s_surface_stress_zero_tendency_matches_legacy_short_chain():
    pcfm = torch.tensor([[0.03, 0.01]], dtype=torch.float64)
    frac = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    u = torch.tensor([5.0], dtype=torch.float64)
    v = torch.tensor([2.0], dtype=torch.float64)

    out = vdfdifm1s_surface_stress(pcfm, frac, u, v, dt=1800.0, rg=9.80665)
    pcfm_lev = pcfm[:, 0] * frac[:, 0] + pcfm[:, 1] * frac[:, 1]
    torch.testing.assert_close(out.stress_u, pcfm_lev * u, rtol=1e-13, atol=0.0)
    torch.testing.assert_close(out.stress_v, pcfm_lev * v, rtol=1e-13, atol=0.0)
