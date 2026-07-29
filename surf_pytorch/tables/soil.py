"""Van-Genuchten soil parameter table (NSOTY = 7 texture classes).

Faithful translation of ``SUSSOIL`` (``surf/module/sussoil_mod.F90``).
Index convention matches the Fortran: soil types are 1..7, and a dummy
index 0 ("water", all zeros) is prepended so ``gather`` with a raw
``soiltype`` tensor works without subtracting one.

What is here
------------
* ``raw_soil_table()``  -- the 6 base columns read straight from the
  Fortran ``DATA`` statements (alpha, ksat, nfac, lambda, wsat, wres).
* ``SoilTable``         -- a ``nn.Module``-style bundle of derived
  hydraulic + thermal parameters, registered as a non-learnable buffer
  on a chosen device/dtype.

Derived quantities reproduced exactly (``sussoil_mod`` lines 340-433):

  ZPSIPWP = -15  * ZFAC            (classic permanent wilting point)
  ZPSICAP = -0.10 * ZFAC           (LEVGEN=.TRUE. sandy-soil value)
  ZFAC    = 100/9.8
  RCWPSISM = 1/alpha
  RMFACM   = 1 - 1/nfac
  RWSATM, WRES as given
  RWCAPM   = wres + (wsat-wres) * (1/(1+(|alpha*psicap|)**n))**mfac
  RWPWPM   = wres + (wsat-wres) * (1/(1+(|alpha*psipwp|)**n))**mfac
  RWRESTM  = wres
  RDMAXM, RDMINM   -- bounding diffusivities (JPRD arithmetic)
  RQWEVAPM = 1/(wcap-wpwp)
  RQWSBCRM = pi/(1.6*wcap)
  RLAMBDADRYM, RLAMSAT1M, RRCSOILM -- thermal constants

Thermal constants (common to all classes) live here too because they are
needed alongside the per-class LUT when soil temperature is assembled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn

from ..constants import RHOH2O, RPI


# --------------------------------------------------------------------------- #
# Base values straight from sussoil_mod.F90 lines 289-296
# Index 0..7 ; index 0 == WATER (all zeros, lines 299-320)
# --------------------------------------------------------------------------- #
# columns: alpha, ksat[m/s], nfac, lambda, wsat, wres
_BASE_1TO7 = np.array([
    [3.83,  6.94e-6, 1.3774,  1.2500, 0.403, 0.025],   # 1 Coarse
    [3.14,  1.16e-6, 1.1804, -2.3421, 0.439, 0.010],   # 2 Medium
    [0.83,  2.63e-7, 1.2539, -0.5884, 0.430, 0.010],   # 3 Medium Fine
    [3.67,  2.87e-6, 1.1012, -1.9772, 0.520, 0.010],   # 4 Fine
    [2.65,  1.74e-6, 1.1033,  2.5000, 0.614, 0.010],   # 5 Very Fine
    [1.300, 0.93e-6, 1.2039,  0.4000, 0.766, 0.010],   # 6 Trop. organic
    [3.14,  1.16e-6, 1.1804, -2.3421, 0.439, 0.010],   # 7 Trop. organic (medium-like)
], dtype=np.float64)


def raw_soil_table() -> np.ndarray:
    """Return the 8x6 base table with the water sentinel at index 0."""
    out = np.zeros((8, 6), dtype=np.float64)
    out[1:] = _BASE_1TO7
    return out


# --------------------------------------------------------------------------- #
# Thermal-constant building blocks (sussoil lines 406-417, 426-433)
# --------------------------------------------------------------------------- #
_ZRHOSM = 2700.0
_ZLAMBDAQ = 7.7
_ZLAMBDAO = 2.0
_ZQ = 0.4
_ZLAMBDASM = _ZLAMBDAQ**_ZQ * _ZLAMBDAO ** (1.0 - _ZQ)
_ZRCGDRY = 1.6e6
# SUSSOIL uses ``0.947*ZRHOD`` without a ``_JPRB`` suffix.  That literal is
# rounded as default real before promotion in the double-precision build.
_ZDRY_DENOM_COEFF = float(np.float32(0.947))

# class-agnostic soil constants (RLAMBDAICE / RLAMBDAWAT / RKERST*)
RLAMBDAICE = 2.2
RLAMBDAWAT = 0.57
RKERST1 = 0.1
RKERST2 = 1.0
RKERST3 = 1.0


def _eps_machine() -> float:
    # ZEPSILON = 100 * EPSILON(ZEPSILON)  (Fortran intrinsic, double)
    eps = 1.0
    while 1.0 + eps * 0.5 != 1.0:
        eps *= 0.5
    return 100.0 * eps


def _derived_soil(levgen: bool = True) -> Dict[str, np.ndarray]:
    """Reproduce the ``DO JS=1,NSOTY`` loop in sussoil (lines 340-381, 426-433).

    Returns dict of 1-D arrays indexed 0..7 (index 0 == 0).
    """
    base = raw_soil_table()
    nsoty = 7
    alpha = base[1:, 0].copy()
    wconsm = base[1:, 1].copy()
    nfac = base[1:, 2].copy()
    lam = base[1:, 3].copy()
    wsat = base[1:, 4].copy()
    wres = base[1:, 5].copy()

    mfac = 1.0 - 1.0 / nfac

    zfac = 100.0 / 9.8
    zpsipwp = -15.0 * zfac
    zpsicap = (-0.10 if levgen else -0.33) * zfac

    rcwpsism = 1.0 / alpha
    wcapm = wres + (wsat - wres) * (
        1.0 / (1.0 + np.abs(alpha * zpsicap) ** nfac)
    ) ** mfac
    wpwpm = wres + (wsat - wres) * (
        1.0 / (1.0 + np.abs(alpha * zpsipwp) ** nfac)
    ) ** mfac
    wrestm = wres.copy()

    eps = _eps_machine()

    # RDMAXM / RDMINM -- mirror the JPRD arithmetic of sussoil (lines 357-371)
    # Fortran SIGN(MAX(ABS(WSAT-WREST), EPS), WSAT-WREST).
    # np.sign's second positional argument is ``out``, not a sign source.
    wfac = np.copysign(
        np.maximum(np.abs(wsat - wrestm), eps), wsat - wrestm
    )
    zwmax = 0.999 * wsat
    zsemax = (zwmax - wrestm) / wfac
    semax_term = (1.0 - zsemax ** (1.0 / mfac)) ** mfac
    zdmax_d = (
        zsemax ** (lam - 1.0 / mfac)
        * (semax_term + (1.0 - zsemax ** (1.0 / mfac)) ** (-mfac) - 2.0)
    )
    rdmaxm = ((1.0 - mfac) * wconsm / (alpha * mfac * wfac)) * zdmax_d

    zwmin = 1.001 * wrestm
    zsemin = (zwmin - wrestm) / wfac
    semin_term = (1.0 - zsemin ** (1.0 / mfac)) ** mfac
    zdmin_d = (
        zsemin ** (lam - 1.0 / mfac)
        * (semin_term + (1.0 - zsemin ** (1.0 / mfac)) ** (-mfac) - 2.0)
    )
    rdminm = ((1.0 - mfac) * wconsm / (alpha * mfac * wfac)) * zdmin_d

    rqwevapm = 1.0 / (wcapm - wpwpm)
    rqwsbcrm = RPI / (1.6 * wcapm)

    # thermal constants (sussoil 426-433)
    rhod = (1.0 - wsat) * _ZRHOSM
    rlambdadrym = (0.135 * rhod + 64.7) / (_ZRHOSM - _ZDRY_DENOM_COEFF * rhod)
    rlamsat1m = _ZLAMBDASM ** (1.0 - wsat)
    rrcsoilm = (1.0 - wsat) * _ZRCGDRY + wcapm * 4.18e6  # RGH2O = 4.18e6

    def _pad(v: np.ndarray) -> np.ndarray:
        out = np.zeros(v.shape[0] + 1, dtype=np.float64)
        out[1:] = v
        return out

    return {
        "rcwpsism": _pad(rcwpsism),
        "mvgalpha": _pad(alpha),
        "wconsm": _pad(wconsm),
        "mfac": _pad(mfac),
        "nfac": _pad(nfac),
        "lambda": _pad(lam),
        "wsat": _pad(wsat),
        "wcap": _pad(wcapm),
        "wpwp": _pad(wpwpm),
        "wrest": _pad(wrestm),
        "dmax": _pad(rdmaxm),
        "dmin": _pad(rdminm),
        "qwevap": _pad(rqwevapm),
        "qwsbcr": _pad(rqwsbcrm),
        "lambdadry": _pad(rlambdadrym),
        "lamsat1": _pad(rlamsat1m),
        "rcsoil": _pad(rrcsoilm),
    }


@dataclass(frozen=True)
class SoilConstants:
    """Class-agnostic soil / thermal / snow scalars from SUSSOIL."""

    rlamice: float = 2.22
    rhoice: float = 920.0
    rhoci: float = 2.05e6
    rlamdsn_pow: float = 1.88       # RALAMSN
    rdsmax: float = 1.0             # snow depth cap
    rqsmcr: float = 1.0 / 15.0      # snow-cover fraction denominator
    rthrfri_ti: float = 1.0e-6      # RTHRFRTI default (tile fraction floor)
    rfrsmall: float = 1.0e-6
    rfrtiny: float = 1.0e-7
    rwlmax: float = 0.2             # skin reservoir scaling [kg/m^2]
    rsimp: float = 1.0              # implicit time weight
    rsrdep: float = 0.5             # VIC representative depth [m]
    # VIC surface-runoff sub-grid orography bounds (sussoil 480-481).
    # ZROEFF = clamp((PSDOR - RSIGORMIN)/(PSDOR + RSIGORMAX), 0, 0.5) drives
    # the saturation-area exponent ZBWS; the defaults 100/1000 mean a flat
    # slab (PSDOR=0) yields ZROEFF = 0 and ZBWS = 0.01.
    rsigormin: float = 100.0
    rsigormax: float = 1000.0

    # snow albedo / density (sussoil lines 497-528)
    ralfminsn: float = 0.50
    ralfmaxsn: float = 0.85
    rhominsn: float = 100.0
    rhomaxsn: float = 300.0
    rtauf: float = 0.24
    rtaua: float = 0.008
    rsfresh: float = 1.0 / 3600.0

    # snow albedo over land ice / glacier (suphec.F90:176).  Maximum
    # albedo for the glacier (land-ice) branch of the SRFSN_ASN aging
    # block; the glacier branch is a weighted (by PCIL) companion to the
    # land-snow branch when the land-ice cover fraction is supplied.
    ralfminpsn: float = 0.82

    # snow density evolution (rdnml_params.F90:28-65; the SRFSN_LWIMP /
    # SRFSN_RSN single- and multi-layer density blocks).  The fresh-snow
    # density is RHOMINSNA + RHOMINSNB*(Tsurf - RTT) + RHOMINSNC*|U|^0.5,
    # floored at RHOMINSND.  Overburden compaction uses RSNDTOVERA/B/C in
    # the exponential viscous-deformation rate; destructive metamorphism
    # uses RSNDTDESTA/B/C/ROI (the C term damps metamorphism once the
    # density exceeds RSNDTDESTROI).  RHOMAXSN_NEW is the 450/500 cap on
    # the evolved density (single-layer path hard-codes 450; the multi-
    # layer path and the namelist default use 500).
    rhominsna: float = 109.0
    rhominsnb: float = 6.0
    rhominsnc: float = 26.0
    rhominsnd: float = 50.0
    rsndtovera: float = 3.7e7
    rsndtoverb: float = 8.1e-2
    rsndtoverc: float = 1.8e-2
    rsndtdesta: float = 2.8e-6
    rsndtdestb: float = 4.2e-2
    rsndtdestc: float = 460.0
    rsndtdestroi: float = 150.0
    rhomaxsn_new: float = 500.0

    # single-layer snow (SRFSN_LWIMP) liquid-water-capacity + melt window
    # (sussoil lines 520-523).  RLWCSWEA is the SWE scale above which the
    # liquid-water-capacity fraction collapses toward RLWCSWEC; RLWCSWEB /
    # RLWCSWEC are the low-density / high-density capacity fractions.
    # RTEMPAMP is the half-width of the sinusoidal melt transition around
    # RTT used by the analytical ZFUNC / ZDFUNC.
    rlwcswea: float = 200.0
    rlwcsweb: float = 0.03
    rlwcswec: float = 0.1
    rtempamp: float = 4.0

    # soil freezing function (sussoil 456-459); RTT injected at build time
    # RTF1 = RTT+1, RTF2 = RTT-3, RTF3 = (RTF1+RTF2)/2, RTF4 = pi/(RTF1-RTF2)

    # thermal constants
    rlambdace: float = RLAMBDAICE
    rlambdawat: float = RLAMBDAWAT
    rkerst1: float = RKERST1
    rkerst2: float = RKERST2
    rkerst3: float = RKERST3
    rho_h2o: float = RHOH2O


class SoilTable(nn.Module):
    """Per-class soil LUT + class-agnostic scalars.

    All per-class arrays are stored with a leading water sentinel so
    ``soiltype`` tensors (values 0..7) index directly.  They are
    registered as ``buffer``s -- not ``Parameter``s -- by default; pass
    ``learnable=...`` (a mapping of array-name -> bool) to expose
    selected columns as optimisable parameters for ML fine-tuning.
    """

    PER_CLASS_FIELDS = (
        "rcwpsism", "mvgalpha", "wconsm", "mfac", "nfac", "lambda",
        "wsat", "wcap", "wpwp", "wrest", "dmax", "dmin",
        "qwevap", "qwsbcr", "lambdadry", "lamsat1", "rcsoil",
    )

    def __init__(
        self,
        *,
        levgen: bool = True,
        consts: SoilConstants | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
        learnable=None,
    ) -> None:
        super().__init__()
        derived = _derived_soil(levgen=levgen)
        self.consts = consts or SoilConstants()
        self.dtype = dtype
        self.device = device
        self.levgen = bool(levgen)

        learnable = learnable or {}
        for name in self.PER_CLASS_FIELDS:
            arr = torch.as_tensor(derived[name], dtype=dtype, device=device)
            if learnable.get(name, False):
                self.register_parameter(name, nn.Parameter(arr))
            else:
                self.register_buffer(name, arr)

    # convenience ---------------------------------------------------------- #
    def gather(self, name: str, soiltype: torch.Tensor) -> torch.Tensor:
        """Index a per-class column by a long ``soiltype`` tensor.

        ``soiltype`` may be any shape; the result follows that shape.
        Gradients never flow back to the index (the index is detached),
        matching the Fortran where the soil type selects fixed LUT rows.
        """
        lut = getattr(self, name)
        idx = soiltype.long().detach()
        return lut[idx]


__all__ = [
    "SoilConstants",
    "SoilTable",
    "raw_soil_table",
    "RLAMBDAICE",
    "RLAMBDAWAT",
    "RKERST1",
    "RKERST2",
    "RKERST3",
]
