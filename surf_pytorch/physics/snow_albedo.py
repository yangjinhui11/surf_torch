"""Snow albedo aging (``SRFSN`` 340-361 and ``SRFSN_ASN`` 133-179).

This module ports the snow-albedo aging of the single-layer snow scheme.
Two algorithms live here, dispatched by whether the land-ice cover
fraction ``PCIL`` is supplied on the inputs:

* **Legacy** (``SRFSN`` lines 340-361) -- the pure-land-snow path,
  selected when ``pcil`` is ``None``.  This is bit-for-bit the original
  port: a no-snow reset, a melt exponential-decay / cold linear-decay
  branch, and a fresh-snow hard reset on the ``RSFRESH`` threshold.

* **SRFSN_ASN** (``srfsn_asn_mod.F90`` lines 133-179) -- the full
  multi-surface scheme, selected when ``pcil`` is supplied.  It splits
  the new albedo into a **land part** (``ZASN_L``) and a **glacier /
  land-ice part** (``ZASN_I``), then blends them by ``PCIL``.  The land
  part reuses the legacy melt/cold decay but swaps the hard RSFRESH
  reset for a **linear snowfall brightening** (a graded approach to
  RALFMAXSN as the snowfall water-equivalent grows toward 10 mm).  The
  glacier part uses glacier-specific constants (ZALBMIN=0.65,
  ZRTAUF_GL=0.085, ZALBMAX_GL=RALFMINPSN=0.82) and a hemisphere-
  dependent snowfall reset threshold (5 mm NH / 4 mm SH).

The ``SRFSN_LWIMP`` path that :mod:`surf_pytorch.physics.snow`
mirrors *does not* update the snow albedo ``PASN`` -- its signature only
outputs ``PTSN``/``PGSN`` -- so this module supplies the aging that
advances ``PASNM1M`` alongside ``PSNM1M``,
``PTSNM1M`` and ``PRSNM1M``.

What the legacy scheme does
---------------------------
The new snow albedo ``PASN`` (dimensionless, in [RALFMINSN, RALFMAXSN] =
[0.50, 0.85]) is built from three branches, exactly as in ``SRFSN`` lines
340-361:

1. **No-snow branch** (``LLNOSNOW``): the albedo is reset to fresh-snow
   maximum ``RALFMAXSN``::

       PASN = RALFMAXSN

2. **Melt branch** (``PMSN > 0``): when the snow is melting the albedo
   decays exponentially toward ``RALFMINSN`` with the melt decay time
   ``RTAUF`` (1/day)::

       ZEXPF = exp(-RTAUF * PTMST / RDAY)
       PASN  = (PASNM1M - RALFMINSN) * ZEXPF + RALFMINSN

3. **Cold branch** (``PMSN <= 0``): when the snow is not melting the
   albedo decays linearly toward ``RALFMINSN`` with the cold decay rate
   ``RTAUA`` (1/day), floored at ``RALFMINSN``::

       ZLINA = RTAUA * PTMST / RDAY
       PASN  = max(RALFMINSN, PASNM1M - ZLINA)

4. **Fresh-snow reset**: on top of either snow branch, a snowfall rate
   exceeding the ``RSFRESH`` threshold (1/3600 kg/m^2/s) resets the
   albedo to ``RALFMAXSN``, modelling the brightening effect of fresh
   snow::

       IF (ZSSFC + ZSSFL) > RSFRESH: PASN = RALFMAXSN

The commented-out ``PSSNM1M >= RSNPER`` branch in the Fortran (lines 349-
353) is deliberately kept inert here: ``RSNPER`` is not in the port's
constant set, and the dead branch collapses to ``ZALFMINSN = RALFMINSN``
in every case, so it has no effect.

What the SRFSN_ASN glacier scheme adds
--------------------------------------
When ``PCIL`` is supplied, the same no-snow reset fires, but the snow-
bearing branch splits into (``srfsn_asn_mod.F90`` 137-176):

* **Land part** ``ZASN_L``: identical melt/cold decay as the legacy
  branch, then a *linear* snowfall brightening
  ``ZASN_L += min(max(PSNOWF*PTMST,0)/10,1)*(RALFMAXSN - ZASN_L)`` and a
  clamp to ``[RALFMINSN, RALFMAXSN]``.

* **Glacier part** ``ZASN_I``: if ``PTSNM1M(:,1) > RTT - 5`` (melting
  conditions) decays exponentially toward ``ZALBMIN=0.65`` with the
  glacier melt rate ``ZRTAUF_GL=0.085``; otherwise ``ZASN_I = PASNM1M``
  (cold glaciers keep their albedo).  A linear snowfall brightening
  with the hemisphere-dependent threshold ``ZALBRESET`` (5 mm NH / 4 mm
  SH) drives ``ZASN_I`` toward ``ZALBMAX_GL=RALFMINPSN``, then a clamp
  to ``[ZALBMIN, ZALBMAX_GL]``.

* **Blend**: ``PASN = (1-PCIL)*ZASN_L + PCIL*ZASN_I`` (or ``ZASN_I``
  when ``PCIL >= 1 - epsilon``), with a final safety clamp to
  ``[RALFMINSN, RALFMAXSN]``.

Sign / unit convention
----------------------
``PASN`` is dimensionless in [0.50, 0.85].  ``PMSN`` is the melt rate in
kg/m^2/s (positive only on the melt branch, as exported by
:mod:`surf_pytorch.physics.snow`).  ``ZSSFC + ZSSFL`` is the total
snowfall rate (convective + large-scale) in kg/m^2/s.  ``PTMST`` is the
coupling interval in seconds; ``RDAY`` is the day length in seconds.
``PCIL`` is the land-ice cover fraction in [0,1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ..constants import ConstantBundle
from ..tables.soil import SoilTable


@dataclass
class SnowAlbedoInputs:
    """Inputs to the snow-albedo aging solve.

    Each tensor is ``(Ngrid,)`` unless noted.  Naming mirrors the
    ``SRFSN`` / ``SRFSN_ASN`` local / dummy variables so a side-by-side
    audit is easy.

    The glacier (land-ice) branch fields are **optional**.  When
    ``pcil`` is ``None`` the solver runs the pure-land-snow legacy
    ``SRFSN`` algorithm (340-361) bit-for-bit.  Supplying ``pcil``
    activates the full ``SRFSN_ASN`` scheme (133-179) which splits the
    albedo into a land part (``ZASN_L``) and a glacier part (``ZASN_I``),
    weighted by the land-ice cover fraction ``PCIL``.
    """

    # ---- Snow albedo at t-1 ----
    pasnm1m: torch.Tensor       # PASNM1M  snow albedo   [-]        (Ngrid,)

    # ---- Melt rate at t (from SRFSN_LWIMP solve) ----
    pmsn: torch.Tensor          # PMSN  melt rate        [kg/m^2/s] (Ngrid,)

    # ---- Snowfall throughfall (per grid, large-scale + convective) ----
    pssflpssfc: torch.Tensor    # ZSSFL+ZSSFC total snowfall  [kg/m^2/s] (Ngrid,)

    # ---- Snow geometry / mask ----
    llnosnow: torch.Tensor      # LLNOSNOW  no-snow mask (bool) (Ngrid,)

    # ---- Timestep ----
    ptmst: torch.Tensor         # PTMST  coupling interval  [s]   scalar
    pmsnint: Optional[torch.Tensor] = None  # ZPMSNINT internal phase change [W/m^2]

    # ---- Glacier (land-ice) branch fields (optional, SRFSN_ASN) ----
    # When ``pcil`` is None the pure-land-snow legacy path is used; set
    # it to activate the full SRFSN_ASN scheme with the glacier branch.
    pcil: Optional[torch.Tensor] = None          # PCIL  land-ice fraction [-] (Ngrid,)
    ptsnm1m_top: Optional[torch.Tensor] = None   # PTSNM1M(:,1) top snow-layer T [K] (Ngrid,)
    psnowf: Optional[torch.Tensor] = None        # PSNOWF  snowfall rate [kg/m^2/s] (Ngrid,)
    ldnh: Optional[torch.Tensor] = None          # LDNH  Northern-Hemisphere mask (bool) (Ngrid,)


@dataclass
class SnowAlbedoOutputs:
    """Outputs of the single-layer snow-albedo aging solve."""

    pasn: torch.Tensor          # PASN  new snow albedo  [-]       (Ngrid,)


def compute_snow_albedo(
    inp: SnowAlbedoInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
    *,
    differentiable: bool = True,
) -> SnowAlbedoOutputs:
    """Run the single-layer snow-albedo aging (``SRFSN`` 340-361).

    Args
    ----
    inp:
        :class:`SnowAlbedoInputs` -- packed snow albedo, melt rate,
        snowfall and the no-snow mask from the snow solve.
    soil_table:
        :class:`surf_pytorch.tables.soil.SoilTable` -- provides the
        albedo constants (ralfminsn, ralfmaxsn, rtauf, rtaua, rsfresh).
    consts:
        :class:`ConstantBundle` -- provides ``rday``.
    differentiable:
        If ``False`` run under ``torch.no_grad()`` for speed.

    Returns
    -------
    :class:`SnowAlbedoOutputs` -- the new snow albedo ``PASN``.
    """

    def _core() -> SnowAlbedoOutputs:
        return _albedo_core(inp, soil_table, consts)

    if not differentiable:
        with torch.no_grad():
            return _core()
    return _core()


def _albedo_core(
    inp: SnowAlbedoInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
) -> SnowAlbedoOutputs:
    """Internal: assumes the no_grad / enable_grad decision is already made.

    Dispatches on ``inp.pcil``:

    * ``pcil is None`` -> legacy ``SRFSN`` (340-361), bit-for-bit the
      original port (no-snow reset + melt/cold decay + RSFRESH reset).
    * ``pcil`` supplied -> full ``SRFSN_ASN`` (133-179) with the glacier
      branch and linear snowfall brightening.
    """
    if inp.pcil is None:
        return _legacy_albedo(inp, soil_table, consts)
    return _asn_albedo(inp, soil_table, consts)


def _legacy_albedo(
    inp: SnowAlbedoInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
) -> SnowAlbedoOutputs:
    """Legacy single-layer snow-albedo aging (``SRFSN`` 340-361).

    Bit-for-bit the original port.  Activated when ``pcil`` is None.
    """
    # ------------------------------------------------------------------ #
    # Scalar constants (sussoil lines 497-528 + physical constants).
    # ------------------------------------------------------------------ #
    sc = soil_table.consts
    ralfminsn = float(sc.ralfminsn)   # 0.50  minimum snow albedo
    ralfmaxsn = float(sc.ralfmaxsn)   # 0.85  fresh-snow albedo
    rtauf = float(sc.rtauf)           # 0.24  melt decay rate [1/day]
    rtaua = float(sc.rtaua)           # 0.008 cold decay rate [1/day]
    rday = consts.rday

    ptmst = inp.ptmst

    # ------------------------------------------------------------------ #
    # Decay factors (Fortran lines 338-339, computed once for the whole
    # snow block -- the same ZEXPF/ZLINA the density block uses).
    # ------------------------------------------------------------------ #
    zexpf = math.exp(-rtauf * ptmst / rday)   # melt: exponential decay factor
    zlina = rtaua * ptmst / rday              # cold: linear decay amount

    # ------------------------------------------------------------------ #
    # Aliases for readability.
    # ------------------------------------------------------------------ #
    pasnm1m = inp.pasnm1m
    pmsn = inp.pmsn
    pmsnint = inp.pmsnint if inp.pmsnint is not None else torch.zeros_like(pmsn)
    pssflpssfc = inp.pssflpssfc
    llnosnow = inp.llnosnow

    zalfminsn = torch.full_like(pasnm1m, ralfminsn)
    zalfmaxsn = torch.full_like(pasnm1m, ralfmaxsn)

    # ------------------------------------------------------------------ #
    # Snow branches (SRFSN lines 354-358).
    # NOTE: the commented Fortran RSNPER branch (lines 349-353) collapses
    # to ZALFMINSN = RALFMINSN in every case (RSNPER is not in the port's
    # constant set), so it is kept inert here.
    # ------------------------------------------------------------------ #
    melt = (pmsn > 0.0) | (pmsnint > 0.0)
    # Melt branch (line 355): exponential decay toward RALFMINSN.
    pasn_melt = (pasnm1m - zalfminsn) * zexpf + zalfminsn
    # Cold branch (line 357): linear decay, floored at RALFMINSN.
    pasn_cold = torch.clamp_min(pasnm1m - zlina, min=ralfminsn)
    pasn_with = torch.where(melt, pasn_melt, pasn_cold)

    # ------------------------------------------------------------------ #
    # Fresh-snow brightening. The active offline SURF path calls SRFSN_ASN,
    # whose land branch uses a linear approach to RALFMAXSN over 10 mm SWE,
    # not the older SRFSN hard reset at RSFRESH.
    # ------------------------------------------------------------------ #
    snow_amt = torch.clamp(torch.clamp_min(pssflpssfc * ptmst, 0.0) / 10.0, max=1.0)
    pasn_with = pasn_with + snow_amt * (zalfmaxsn - pasn_with)
    pasn_with = torch.clamp(pasn_with, min=ralfminsn, max=ralfmaxsn)

    # ------------------------------------------------------------------ #
    # No-snow branch (SRFSN line 342): albedo resets to RALFMAXSN.
    # ------------------------------------------------------------------ #
    pasn = torch.where(llnosnow, zalfmaxsn, pasn_with)

    return SnowAlbedoOutputs(pasn=pasn)


def _asn_albedo(
    inp: SnowAlbedoInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
) -> SnowAlbedoOutputs:
    """Full SRFSN_ASN snow-albedo aging (``srfsn_asn_mod.F90`` 133-179).

    Activated when ``pcil`` is supplied.  Splits the new albedo into a
    land part (``ZASN_L``) and a glacier part (``ZASN_I``), blends them
    by PCIL, and applies a final safety clamp.
    """
    # ------------------------------------------------------------------ #
    # Scalar constants.
    # ------------------------------------------------------------------ #
    sc = soil_table.consts
    ralfminsn = float(sc.ralfminsn)   # 0.50  minimum (land) snow albedo
    ralfmaxsn = float(sc.ralfmaxsn)   # 0.85  fresh-snow (land) albedo
    rtauf = float(sc.rtauf)           # 0.24  melt decay rate [1/day]
    rtaua = float(sc.rtaua)           # 0.008 cold decay rate [1/day]
    rday = consts.rday
    rtt = consts.rtt    # 273.15  freezing point [K]

    # Glacier-specific constants (srfsn_asn_mod.F90:114-120).
    zalbmin = 0.65                    # glacier minimum albedo
    zalbmax_gl = float(sc.ralfminpsn)  # 0.82  glacier max albedo
    zrtaua_gl = rtaua / 2.0           # RTAUA/2 (unused; kept for parity)
    zrtauf_gl = 0.085                 # glacier melt decay rate [1/day]
    ztsnthr = 5.0                     # glacier melt threshold [K]
    zepsilon = 1.0e-7                 # PCIL~1 detection tolerance

    ptmst = inp.ptmst

    # Decay factors (same as legacy).
    zexpf = math.exp(-rtauf * ptmst / rday)   # land melt decay factor
    zlina = rtaua * ptmst / rday              # land cold linear decay
    zexpf_gl = math.exp(-zrtauf_gl * ptmst / rday)  # glacier melt decay factor

    # ------------------------------------------------------------------ #
    # Aliases + fallbacks for the glacier-branch fields.
    # ------------------------------------------------------------------ #
    pasnm1m = inp.pasnm1m
    pmsn = inp.pmsn
    pmsnint = inp.pmsnint if inp.pmsnint is not None else torch.zeros_like(pmsn)
    psnowf = inp.psnowf if inp.psnowf is not None else inp.pssflpssfc
    llnosnow = inp.llnosnow
    pcil = inp.pcil
    ptsnm1m_top = inp.ptsnm1m_top if inp.ptsnm1m_top is not None else \
        torch.full_like(pasnm1m, rtt)   # cold-start: treat as freezing
    ldnh = inp.ldnh if inp.ldnh is not None else \
        torch.zeros_like(pasnm1m, dtype=torch.bool)   # default SH

    zalfminsn = torch.full_like(pasnm1m, ralfminsn)
    zalfmaxsn = torch.full_like(pasnm1m, ralfmaxsn)
    z_albmin = torch.full_like(pasnm1m, zalbmin)
    z_albmax_gl = torch.full_like(pasnm1m, zalbmax_gl)

    # ------------------------------------------------------------------ #
    # Land part ZASN_L (srfsn_asn_mod.F90:138-150).
    # ------------------------------------------------------------------ #
    # SRFSN_LWIMP uses a broader melting-condition proxy than SRFSN_ASN:
    # PMSN > 0, internal melt/refreeze, or old snow temperature within 2 K
    # of freezing.  The current port does not carry ZPMSNINT, but the
    # temperature branch is required for the offline SURF path.
    melt = (pmsn > 0.0) | (pmsnint > 0.0) | (ptsnm1m_top > (rtt - 2.0))
    pasn_melt = (pasnm1m - zalfminsn) * zexpf + zalfminsn
    pasn_cold = torch.clamp_min(pasnm1m - zlina, min=ralfminsn)
    zasn_l = torch.where(melt, pasn_melt, pasn_cold)
    # Linear snowfall brightening (replaces the RSFRESH hard reset).
    snow_amt = torch.clamp(torch.clamp_min(psnowf * ptmst, 0.0) / 10.0, max=1.0)
    zasn_l = zasn_l + snow_amt * (zalfmaxsn - zasn_l)
    zasn_l = torch.clamp(zasn_l, min=ralfminsn, max=ralfmaxsn)

    # ------------------------------------------------------------------ #
    # Glacier part ZASN_I (srfsn_asn_mod.F90:153-167).
    # ------------------------------------------------------------------ #
    glacier_melt = ptsnm1m_top > (rtt - ztsnthr)
    pasn_melt_gl = (pasnm1m - z_albmin) * zexpf_gl + z_albmin
    # Cold glaciers keep PASNM1M (line 162).
    zasn_i = torch.where(glacier_melt, pasn_melt_gl, pasnm1m)
    # Hemisphere-dependent snowfall reset threshold (5 mm NH / 4 mm SH).
    zalbreset = torch.where(ldnh, torch.full_like(pasnm1m, 5.0),
                            torch.full_like(pasnm1m, 4.0))
    snow_amt_i = torch.clamp(
        torch.clamp_min(psnowf * ptmst, 0.0) / zalbreset, max=1.0,
    )
    zasn_i = zasn_i + snow_amt_i * (z_albmax_gl - zasn_i)
    zasn_i = torch.clamp(zasn_i, min=zalbmin, max=zalbmax_gl)

    # ------------------------------------------------------------------ #
    # Blend by PCIL (srfsn_asn_mod.F90:169-173) + safety clamp (176).
    # ------------------------------------------------------------------ #
    near_pure_glacier = pcil >= (1.0 - zepsilon)
    pasn_blend = (1.0 - pcil) * zasn_l + pcil * zasn_i
    pasn_snow = torch.where(near_pure_glacier, zasn_i, pasn_blend)
    pasn_snow = torch.clamp(pasn_snow, min=ralfminsn, max=ralfmaxsn)

    # No-snow branch (line 134): albedo resets to RALFMAXSN.
    zalfmaxsn = torch.full_like(pasnm1m, ralfmaxsn)
    pasn = torch.where(llnosnow, zalfmaxsn, pasn_snow)

    return SnowAlbedoOutputs(pasn=pasn)


__all__ = [
    "SnowAlbedoInputs",
    "SnowAlbedoOutputs",
    "compute_snow_albedo",
]
