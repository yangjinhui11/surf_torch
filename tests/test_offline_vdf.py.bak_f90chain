import torch

from surf_pytorch.physics.offline_vdf import vdfdifm1s_surface_stress


def test_vdfdifm1s_surface_stress_matches_offline_vdfincr_reduction():
    pcfm = torch.tensor([[0.03, 0.01], [0.02, 0.06]], dtype=torch.float64)
    frac = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float64)
    u = torch.tensor([5.0, -3.0], dtype=torch.float64)
    v = torch.tensor([2.0, 4.0], dtype=torch.float64)
    ucur = torch.tensor([1.0, -1.0], dtype=torch.float64)
    vcur = torch.tensor([0.5, 1.0], dtype=torch.float64)

    out = vdfdifm1s_surface_stress(pcfm, frac, u, v, ucur, vcur)
    coef = torch.sum(pcfm * frac, dim=-1)
    torch.testing.assert_close(out.stress_u, coef * (u - ucur), rtol=0.0, atol=0.0)
    torch.testing.assert_close(out.stress_v, coef * (v - vcur), rtol=0.0, atol=0.0)
