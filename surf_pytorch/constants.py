"""Physical constants of the IFS / OpenIFS land-surface package.

All values are transcribed verbatim from the offline driver's ``SUCST``
routine (``surf/offline/driver/sucst.F90``) and its thermodynamic
``yoethf`` companions.  The intent is *bit-level* agreement with the
Fortran single-column driver so that ``tests/reference`` gold files can
be reproduced.

Precision
---------
``SUCST`` uses ``REAL(KIND=JPRB)``; in the offline driver ``JPRB`` is
the working precision which is double by default for offline runs, so
every literal below is written at full IEEE-754 double precision.

The numbers are computed in exactly the same order as the Fortran so
that compound expressions (e.g. ``RSIGMA``) match to the last ULP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# --------------------------------------------------------------------------- #
# Fundamental constants (SUCST section 1)
# --------------------------------------------------------------------------- #
RPI = 2.0 * math.asin(1.0)            # pi
RCLUM = 299792458.0                   # speed of light  [m/s]
RHPLA = 6.6260755e-34                 # Planck constant [J s]
RKBOL = 1.380658e-23                  # Boltzmann       [J/K]
RNAVO = 6.0221367e23                  # Avogadro        [1/mol]

# --------------------------------------------------------------------------- #
# Astronomical constants (SUCST section 2)
# --------------------------------------------------------------------------- #
RDAY = 86400.0                        # length of day   [s]
REA = 149597870000.0                  # semi-major axis of Earth orbit [m]
REPSM = 0.409093                      # mean obliquity  [rad]
RSIYEA = 365.25 * RDAY * 2.0 * RPI / 6.283076   # sidereal year [s]
RSIDAY = RDAY / (1.0 + RDAY / RSIYEA)           # sidereal day  [s]
ROMEGA = 2.0 * RPI / RSIDAY                     # angular velocity [1/s]

# --------------------------------------------------------------------------- #
# Geoide (SUCST section 3)
# --------------------------------------------------------------------------- #
RG = 9.80665                          # gravity         [m/s^2]
RA = 6371229.0                        # Earth radius    [m]
R1SA = 1.0 / RA                       # inverse Earth radius

# --------------------------------------------------------------------------- #
# Radiation (SUCST section 4)
# --------------------------------------------------------------------------- #
# Keep the double-precision expression used in SUCST (the JPRM branch is
# only an fp32 re-association that we do not need here).
RSIGMA = 2.0 * RPI**5 * RKBOL**4 / (15.0 * RCLUM**2 * RHPLA**3)   # Stefan-Boltzmann
RI0 = 1370.0                          # solar constant  [W/m^2]

# --------------------------------------------------------------------------- #
# Thermodynamic gas phase (SUCST section 5)
# --------------------------------------------------------------------------- #
R = RNAVO * RKBOL                      # universal gas constant
RMD = 28.9644                          # molar mass dry air     [g/mol]
RMCO2 = 44.0095                        # molar mass CO2         [g/mol]
RMV = 18.0153                          # molar mass water vap   [g/mol]
RMO3 = 47.9942                         # molar mass ozone       [g/mol]
RD = 1000.0 * R / RMD                  # gas const. dry air     [J/kg/K]
RV = 1000.0 * R / RMV                  # gas const. water vap   [J/kg/K]
RCPD = 3.5 * RD                        # cp dry air             [J/kg/K]
RCVD = RCPD - RD                       # cv dry air
RCPV = 4.0 * RV                        # cp water vapour
RCVV = RCPV - RV                       # cv water vapour
RKAPPA = RD / RCPD                     # kappa = Rd/cp
RETV = RV / RD - 1.0                   # (Rv/Rd - 1)

# --------------------------------------------------------------------------- #
# Liquid / solid phase heat capacities (SUCST sections 6, 7)
# --------------------------------------------------------------------------- #
RCW = 4218.0                           # cp liquid water        [J/kg/K]
RCS = 2106.0                           # cp ice                 [J/kg/K]

# --------------------------------------------------------------------------- #
# Phase transition (SUCST section 8)
# --------------------------------------------------------------------------- #
RTT = 273.16                           # triple point of water  [K]
RDT = 11.82                            # RTT - Tx(ew-ei)
RLVTT = 2.5008e6                       # latent heat vap.  at RTT [J/kg]
RLSTT = 2.8345e6                       # latent heat subl. at RTT [J/kg]
RLVZER = RLVTT + RTT * (RCW - RCPV)    # L_v(0K)
RLSZER = RLSTT + RTT * (RCS - RCPV)    # L_s(0K)
RLMLT = RLSTT - RLVTT                  # latent heat of fusion
RATM = 100000.0                        # normal pressure        [Pa]

# --------------------------------------------------------------------------- #
# YOETHF derived constants (offline SUPHEC, surf/offline/driver/suphec.F90).
# IMPORTANT: the offline driver overrides the canonical values of RVTMP2,
# R2ES and the saturation-curve slopes -- those overrides are applied here.
# --------------------------------------------------------------------------- #
# SUPHEC line 137: RVTMP2 = 0   (neglect cp,moist).  This is the value the
# offline single-column driver actually uses, NOT RCPV/RCPD-1.
RVTMP2 = 0.0
RHOH2O = RATM / 100.0                  # density of water       [kg/m^3]

# Saturation-mixing-ratio coefficients (SUPHEC lines 139-145).
R2ES = 611.21 * RD / RV
R3LES = 17.502
R3IES = 22.587
R4LES = 32.19
R4IES = -0.7
R5LES = R3LES * (RTT - R4LES)
R5IES = R3IES * (RTT - R4IES)
R5ALVCP = R5LES * RLVTT / RCPD
R5ALSCP = R5IES * RLSTT / RCPD
RALVDCP = RLVTT / RCPD
RALSDCP = RLSTT / RCPD
RALFDCP = RLMLT / RCPD
RTWAT = RTT
RTBER = RTT - 5.0
RTICE = RTT - 23.0
RTICECU = RTT - 23.0
RTWAT_RTICE_R = 1.0 / (RTWAT - RTICE)
RTWAT_RTICECU_R = 1.0 / (RTWAT - RTICECU)

# Soil-freeze ramp (SUSSOIL 456-459): liquid-water fraction ZF transitions
# from 1 (all-liquid, T <= RTF2) to 0 (all-frozen, T >= RTF1) across the
# window [RTF2, RTF1] via the sine 0.5*(1 - SIN(RTF4*(T-RTF3))).
RTF1 = RTT + 1.0
RTF2 = RTT - 3.0
RTF3 = 0.5 * (RTF1 + RTF2)
RTF4 = RPI / (RTF1 - RTF2)

# --------------------------------------------------------------------------- #
# Saturation vapour pressure curve (SUCST section 9)
#   es(T) coefficients -- see fcttre.h FOEEW/FOEEQS.  These are the
#   constants used inside the qsat() linearisation, kept here so that
#   ``utils.qsat`` reproduces ``FCEW``/``FDES`` exactly.
# --------------------------------------------------------------------------- #
RESTT = 611.14                         # es at triple point     [Pa]
RGAMW = (RCW - RCPV) / RV
RBETW = RLVTT / RV + RGAMW * RTT
RALPW = math.log(RESTT) + RBETW / RTT + RGAMW * math.log(RTT)
RGAMS = (RCS - RCPV) / RV
RBETS = RLSTT / RV + RGAMS * RTT
RALPS = math.log(RESTT) + RBETS / RTT + RGAMS * math.log(RTT)
RGAMD = RGAMS - RGAMW
RBETD = RBETS - RBETW
RALPD = RALPS - RALPW

# --------------------------------------------------------------------------- #
# Miscellaneous surface-package constants (yoesoil / yoevdf)
# --------------------------------------------------------------------------- #
# RVTMP2 is set above in the YOETHF block (offline driver uses 0.0).


@dataclass(frozen=True)
class PhysicalConstants:
    """Frozen container of every scalar constant above.

    Provided so a ``LandSurface`` module can hold a single tensor-backed
    bundle (via :class:`ConstantBundle`) and so tests can sweep values.
    """

    pi: float = RPI
    g: float = RG
    rd: float = RD
    rv: float = RV
    cpd: float = RCPD
    cpv: float = RCPV
    retv: float = RETV
    rtt: float = RTT
    rlvtt: float = RLVTT
    rlstt: float = RLSTT
    rlmlt: float = RLMLT
    rsigma: float = RSIGMA
    rho_h2o: float = RHOH2O
    rday: float = RDAY

    # saturation-curve coefficients (FOEEW/FOEEQS path)
    ralpw: float = RALPW
    rbetw: float = RBETW
    rgamw: float = RGAMW
    ralps: float = RALPS
    rbets: float = RBETS
    rgams: float = RGAMS
    rgamd: float = RGAMD
    rbetd: float = RBETD
    ralpd: float = RALPD

    # YOETHF saturation-mixing-ratio coefficients (SUPHEC overrides).
    r2es: float = R2ES
    r3les: float = R3LES
    r3ies: float = R3IES
    r4les: float = R4LES
    r4ies: float = R4IES
    r5les: float = R5LES
    r5ies: float = R5IES
    r5alvcp: float = R5ALVCP
    r5alscp: float = R5ALSCP
    ralvdcp: float = RALVDCP
    ralsdcp: float = RALSDCP
    ralfdcp: float = RALFDCP
    rvtmp2: float = RVTMP2
    rtwat: float = RTWAT
    rtice: float = RTICE
    rticecu: float = RTICECU
    rtwat_rtice_r: float = RTWAT_RTICE_R

    # Soil-freeze ramp endpoints (SUSSOIL 456-459); used by stomata F2 freeze
    # shutdown and by the freeze-correction in the soil-water kernel.
    rtf1: float = RTF1
    rtf2: float = RTF2
    rtf3: float = RTF3
    rtf4: float = RTF4

    # Jarvis stomatal-resistance scalars (vsurf_mod.F90 module scalars).
    #   ZEPSF3 -- minimum VPD stress floor (vsurf ZEPSF3 = 1.0E-5)
    #   RCLU    -- canopy light-use coefficient, unused in the offline
    #              Betts-light branch but kept for parity with vsurf
    #   RSMIN_B -- minimum stomatal resistance over bare soil [s/m]
    zepsf3: float = 1.0e-5
    rclu: float = 50.0
    rsmin_b: float = 50.0


def _scalar_tensor(value: float, *, dtype: torch.dtype, device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


class ConstantBundle:
    """Tensor-backed view of :class:`PhysicalConstants`.

    All physics modules receive one of these so constants live on the
    correct device/dtype without global state.  Scalars are 0-d tensors;
    autograd is never attached (constants are leaves with
    ``requires_grad=False``).
    """

    __slots__ = tuple(PhysicalConstants.__dataclass_fields__) + ("dtype", "device")

    def __init__(
        self,
        consts: PhysicalConstants | None = None,
        *,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        c = consts or PhysicalConstants()
        self.dtype = dtype
        self.device = device
        for name in PhysicalConstants.__dataclass_fields__:
            val = getattr(c, name)
            tensor = _scalar_tensor(val, dtype=dtype, device=device)
            setattr(self, name, tensor)

    def to(self, *, dtype=None, device=None) -> "ConstantBundle":
        return ConstantBundle(
            PhysicalConstants(), dtype=dtype or self.dtype, device=device or self.device
        )
