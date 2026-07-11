"""Ocean mixed-layer model stub (``OCEAN_ML_DRIVER`` / KPP).

This module is the placeholder port of ``OCEAN_ML_DRIVER`` -- the
K-profile-parameterization ocean mixed-layer model (Large, McWilliams &
Doney 1994) invoked by ``SURFTSTP_CTL`` when the per-point mask
``LDOCN_KPP`` is ``.TRUE.``.  The offline driver cold-starts with
``LEOCSA = .FALSE.`` (*su0phy1s.F90:107*), which forces
``LDOCN_KPP = .FALSE.`` everywhere (surfbc_ctl_mod.F90:262), so the
KPP driver never runs and the SST is prescribed instead.

What the real KPP driver does
-----------------------------
``OCEAN_ML_DRIVER`` integrates a 1-D column ocean over a fixed depth
grid (``KLEVO`` levels, default 8 -- *sudim1s.F90*).  It advances the
prognostic temperature, salinity and (optionally) horizontal currents
under the surface fluxes (net SW / LW, sensible / latent heat, momentum
stress, convective + large-scale rain and snow), using the KPP mixing
scheme (Large et al. 1994) for the surface-boundary-layer eddy
diffusivities and the Richardson-number-based interior mixing.

The prognostic state lives on the IFS state as ``PTO0`` (temperature),
``PSO0`` (salinity), ``PUO0``/``PVO0`` (currents) at the ocean levels,
plus the depth grid ``PZO`` and layer thicknesses ``PDO``.  The driver
returns the t+1 state plus the SW-penetration diagnostic.

What this stub does
-------------------
The function :func:`ocean_ml_driver` reproduces the offline-driver
default behaviour: it returns the t-1 SST unchanged (identity update),
ignoring the surface forcing.  This is exactly what the offline driver
does when ``LEOCSA = .FALSE.`` (the prescribed-SST path), so flipping
``SurfConfig.use_kpp`` from ``False`` (the current state, which calls
this stub) to ``True`` later will be a *pure* drop-in once the full KPP
transcription lands.

A future milestone will port the real ``OCEAN_ML_DRIVER`` body
(ocean_ml_driver_mod.F90 + the kpp_* companions) into this file.  Until
then the identity update is bit-exact with the upstream
``LEOCSA = .FALSE.`` branch and is exercised by the unit test
:mod:`tests.test_forward_m6`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
# Input / output containers
# --------------------------------------------------------------------------- #
@dataclass
class OceanMlInputs:
    """t-1 ocean state + surface forcing (OCEAN_ML_DRIVER dummy list).

    The prognostic tensors are ``(Ngrid, Nlev_ocean)``; the forcing and
    mask tensors are ``(Ngrid,)``.  The full 3-D state (temperature,
    salinity, currents) is accepted but ignored by the stub -- it is
    present so the signature matches the eventual real driver.  Only
    ``pto0[:, 0]`` (the skin-layer temperature, which feeds the water
    tile skin) is propagated into :class:`OceanMlOutputs`.
    """

    # t-1 prognostic state.
    pto0: torch.Tensor         # ocean temperature [K], (Ngrid, Nlev)
    pso0: torch.Tensor         # ocean salinity [psu], (Ngrid, Nlev)
    puo0: torch.Tensor         # ocean u-current [m/s], (Ngrid, Nlev)
    pvo0: torch.Tensor         # ocean v-current [m/s], (Ngrid, Nlev)

    # Static grid.
    pzo: torch.Tensor          # ocean level depths [m], (Ngrid, Nlev+1)
    pdo: torch.Tensor          # ocean layer thicknesses [m], (Ngrid, Nlev)

    # Forcing (accepted, ignored by the stub).
    pssrflti: torch.Tensor     # net SW radiation at surface [W/m^2]
    pslrfl:   torch.Tensor     # net LW radiation at surface [W/m^2]
    pahfsti:  torch.Tensor     # sensible heat flux downwards [W/m^2]
    pevapti:  torch.Tensor     # latent heat flux downwards [W/m^2]
    pustrti:  torch.Tensor     # x-momentum flux [N/m^2]
    pvstrti:  torch.Tensor     # y-momentum flux [N/m^2]

    # Masks / scalars.
    ld_ocn_kpp: torch.Tensor   # ocean-ML active mask (bool)
    ptstep:    float           # coupling interval [s]


@dataclass
class OceanMlOutputs:
    """t+1 ocean state (the upstream ``PTOE1`` / ``PSOE1`` locals).

    Only the surface temperature ``pto0_new[:, 0]`` is consumed by the
    forward path (it pins the water-tile skin).  The deeper levels and
    the salinity / currents are returned for completeness so the
    real-driver transcription can populate them without changing the
    output contract.
    """

    pto0_new: torch.Tensor     # t+1 ocean temperature [K], (Ngrid, Nlev)
    pso0_new: torch.Tensor     # t+1 ocean salinity [psu], (Ngrid, Nlev)
    puoe1:    torch.Tensor     # t+1 u-current [m/s], (Ngrid, Nlev)
    pvoe1:    torch.Tensor     # t+1 v-current [m/s], (Ngrid, Nlev)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def ocean_ml_driver(inp: OceanMlInputs) -> OceanMlOutputs:
    """Advance the ocean mixed-layer state by one coupling step.

    **Stub implementation** -- reproduces the offline-driver default
    (``LEOCSA = .FALSE.``): every prognostic is held at its t-1 value.
    The surface forcing is accepted but unused.

    Once the real KPP body is transcribed this function is the only
    thing that changes -- the signature, the dataclasses and the call
    site in :meth:`LandSurface._update_ocean_ml` stay put.
    """
    return OceanMlOutputs(
        pto0_new=inp.pto0.clone(),
        pso0_new=inp.pso0.clone(),
        puoe1=inp.puo0.clone(),
        pvoe1=inp.pvo0.clone(),
    )
