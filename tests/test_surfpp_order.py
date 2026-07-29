import torch

from surf_pytorch import LandSurface, SurfConfig


def test_surfpp_flux_mean_uses_fortran_tile_order():
    """SURFPP_CTL sums tile fluxes in ascending tile order."""
    model = LandSurface(SurfConfig(dtype=torch.float64, validate=False))
    frac = torch.tensor(
        [[0.0, 0.0, 0.0, 0.49500000000000005, 0.0,
          0.045000000000000005, 0.0, 0.4599999999999999]],
        dtype=torch.float64,
    )
    pssh = torch.tensor(
        [[-150.0455138227626, -29.52756057822137, 3.219980375005889,
          -12.000373353368104, 107.88911038101742, -12.482951978969709,
          -23.267199048151582, -12.948947729426777]],
        dtype=torch.float64,
    )
    pjq = torch.zeros_like(pssh)

    pssh_out, _ = model._surfpp_zero_tile_fluxes(pssh, pjq, frac)

    # This is the corresponding SURFPP_CTL ZAHFSM value from Fortran.
    assert pssh_out[0, 4].item() == -12.458433604507166
