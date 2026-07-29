"""Reference checks for the offline VUPDZ0 warm-start closure."""

from __future__ import annotations

import torch

from surf_pytorch import LandSurface, SurfConfig
from surf_pytorch.state import Forcing, SurfState


def test_warm_start_matches_real_fortran_unstable_column():
    """Reproduce a real CMFD high-vegetation VUPDZ0 trace."""
    dtype = torch.float64
    cfg = SurfConfig(dtype=dtype, validate=False)
    model = LandSurface(cfg)
    ntile = cfg.n_tile

    state = SurfState(
        skin_t=torch.full((1, ntile), 292.14867204929652, dtype=dtype),
        soil_t=torch.full((1, cfg.n_soil), 292.0, dtype=dtype),
        soil_w_liq=torch.full((1, cfg.n_soil), 0.2, dtype=dtype),
        snow_w=torch.zeros(1, dtype=dtype),
        snow_frac=torch.zeros(1, dtype=dtype),
        skin_water=torch.zeros(1, dtype=dtype),
    )
    forcing = Forcing(
        u=torch.tensor([0.051998138427734375], dtype=dtype),
        v=torch.zeros(1, dtype=dtype),
        t=torch.tensor([292.1300048828125], dtype=dtype),
        q=torch.tensor([0.01255200058221817], dtype=dtype),
        co2=torch.tensor([6.0e-4], dtype=dtype),
        p_surf=torch.tensor([94718.0], dtype=dtype),
        geom1=torch.tensor([98.0665], dtype=dtype),
        fr_so=torch.zeros(1, dtype=dtype),
        fr_th=torch.zeros(1, dtype=dtype),
        mu0=torch.zeros(1, dtype=dtype),
        rain=torch.zeros(1, dtype=dtype),
        snow=torch.zeros(1, dtype=dtype),
        dt=torch.tensor(10800.0, dtype=dtype),
    )

    model._last_pustrti = torch.full(
        (1, ntile), 1.4944065215554065e-3, dtype=dtype
    )
    model._last_pvstrti = torch.zeros((1, ntile), dtype=dtype)
    model._last_pahfsti = torch.full(
        (1, ntile), 2.6585277333142585, dtype=dtype
    )
    model._last_pevapti = torch.full(
        (1, ntile), -1.6285433981019196e-5, dtype=dtype
    )
    pzdl, pbuom = model._vupdz0_warm_start(forcing, state, ntile)

    z0m = torch.full((1, ntile), 2.0, dtype=dtype)
    z0q = torch.full((1, ntile), 2.0, dtype=dtype)
    praqti = model._vupdz0_praqti(forcing, state, z0m, z0q, pzdl)

    tile = 5  # Fortran tile 6, zero-based in the port.
    assert torch.isclose(
        pbuom.reshape(1, ntile)[0, tile],
        torch.tensor(7.3530022898104131e-6, dtype=dtype),
        rtol=0.0,
        atol=2.0e-16,
    )
    assert torch.isclose(
        pzdl.reshape(1, ntile)[0, tile],
        torch.tensor(-5.5622814967798506e-1, dtype=dtype),
        rtol=0.0,
        atol=1.0e-10,
    )
    assert torch.isclose(
        praqti[0, tile],
        torch.tensor(5.5925728785348646e1, dtype=dtype),
        rtol=0.0,
        atol=5.0e-10,
    )
