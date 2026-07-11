"""FLake lake model stub (``FLAKE_DRIVER``).

This module is the placeholder port of ``FLAKE_DRIVER`` -- the
fresh-water lake model used by ``SURFTSTP_CTL`` (section 4.7,
surftstp_ctl_mod.F90:795-818) when the namelist switch ``LEFLAKE`` is
``.TRUE.``.  The offline driver cold-starts with ``LEFLAKE = .FALSE.``
(*sudim1s.F90*), so the upstream ``ELSE`` branch at lines 808-818 runs
instead: every prognostic is held at its t-1 value (identity update).

What the real FLake does
------------------------
FLake (Mironov 2008) is a two-layer parametrization of the lake mixed
layer and thermocline.  It integrates seven prognostics per lake point:

* ``PTLICE``  -- temperature at the air-ice / snow-ice interface [K]
* ``PTLMNW``  -- mean temperature of the water column [K]
* ``PTLWML``  -- mixed-layer temperature [K]
* ``PTLBOT``  -- lake-bottom temperature [K]
* ``PTLSF``   -- skin temperature at the lower surface of the ice [K]
* ``PHLICE``  -- lake-ice thickness [m]
* ``PHLML``   -- mixed-layer depth [m]

The driver takes the surface forcing (net SW / LW, sensible / latent
heat, momentum stress) plus the lake depth and latitude, and returns
the t+1 prognostics.  It is gated behind a lake-point mask (``LDLAKE``)
so non-lake points are skipped entirely.

What this stub does
-------------------
The function :func:`flake_driver` reproduces the upstream ``ELSE``
branch only: it copies every t-1 prognostic into the t+1 output and
returns it, ignoring the surface forcing.  This is exactly what the
offline-driver default does, so flipping ``SurfConfig.use_flake`` from
``False`` (the current state, which calls this stub) to ``True`` later
will be a *pure* drop-in once the full FLake transcription lands -- the
inputs / outputs / call site will not move.

A future milestone will port the real ``FLAKE_DRIVER`` body
(flake_driver_mod.F90 + the flakeene / flakerad companions) into this
file.  Until then, the identity update is bit-exact with the upstream
``LEFLAKE = .FALSE.`` branch and is exercised by the unit test
:mod:`tests.test_forward_m6`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
# Input / output containers
# --------------------------------------------------------------------------- #
@dataclass
class LakeInputs:
    """t-1 lake prognostics + surface forcing (FLAKE_DRIVER dummy list).

    All tensors are ``(Ngrid,)``.  The prognostic fields mirror the
    ``PT* M1M`` / ``PH* M1M`` arguments of ``FLAKE_DRIVER``; the forcing
    fields mirror ``PSSRFLTI`` etc.  The forcing is accepted but ignored
    by the stub -- it is present so the signature matches the eventual
    real driver.
    """

    # t-1 prognostics (the upstream P* M1M arguments).
    ptlice_m1m: torch.Tensor   # air-ice interface T [K]
    ptlmnw_m1m: torch.Tensor   # mean water-column T [K]
    ptlwml_m1m: torch.Tensor   # mixed-layer T [K]
    ptlbot_m1m: torch.Tensor   # lake-bottom T [K]
    ptlsf_m1m:  torch.Tensor   # lower-ice-surface skin T [K]
    phlice_m1m: torch.Tensor   # lake-ice thickness [m]
    phlml_m1m:  torch.Tensor   # mixed-layer depth [m]

    # Forcing (accepted, ignored by the stub).
    pssrflti: torch.Tensor     # net SW radiation at surface [W/m^2]
    pslrfl:   torch.Tensor     # net LW radiation at surface [W/m^2]
    pahfsti:  torch.Tensor     # sensible heat flux downwards [W/m^2]
    pevapti:  torch.Tensor     # latent heat flux downwards [W/m^2]
    pustrti:  torch.Tensor     # x-momentum flux [N/m^2]
    pvstrti:  torch.Tensor     # y-momentum flux [N/m^2]

    # Static descriptors.
    pldepth:  torch.Tensor     # lake depth [m]
    pgemu:    torch.Tensor     # sin(latitude)
    ld_lake:  torch.Tensor     # lake-point mask (bool)
    ptstep:   float            # coupling interval [s]


@dataclass
class LakeOutputs:
    """t+1 lake prognostics (the upstream ``ZT*`` / ``ZH*`` locals)."""

    ptlice: torch.Tensor       # air-ice interface T [K]
    ptlmnw: torch.Tensor       # mean water-column T [K]
    ptlwml: torch.Tensor       # mixed-layer T [K]
    ptlbot: torch.Tensor       # lake-bottom T [K]
    ptlsf:  torch.Tensor       # lower-ice-surface skin T [K]
    phlice: torch.Tensor       # lake-ice thickness [m]
    phlml:  torch.Tensor       # mixed-layer depth [m]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def flake_driver(inp: LakeInputs) -> LakeOutputs:
    """Advance the FLake prognostics by one coupling step.

    **Stub implementation** -- reproduces the upstream ``ELSE`` branch of
    ``SURFTSTP_CTL`` (surftstp_ctl_mod.F90:808-818): every prognostic is
    held at its t-1 value.  Bit-exact with the offline-driver default
    (``LEFLAKE = .FALSE.``); the surface forcing is accepted but unused.

    Once the real FLake body is transcribed this function is the only
    thing that changes -- the signature, the dataclasses and the call
    site in :meth:`LandSurface._update_lake` stay put.
    """
    return LakeOutputs(
        ptlice=inp.ptlice_m1m.clone(),
        ptlmnw=inp.ptlmnw_m1m.clone(),
        ptlwml=inp.ptlwml_m1m.clone(),
        ptlbot=inp.ptlbot_m1m.clone(),
        ptlsf=inp.ptlsf_m1m.clone(),
        phlice=inp.phlice_m1m.clone(),
        phlml=inp.phlml_m1m.clone(),
    )
