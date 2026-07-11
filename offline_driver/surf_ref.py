"""ctypes bridge to the standalone Fortran surface reference kernels.

This module wraps ``libsurf_ref.so`` (built by ``build_ref.sh``) and
exposes its two entry points -- ``surf_ref_vlamsk`` and ``surf_ref_seb``
-- as Python callables that take the *same* tensors the PyTorch port
uses.  That lets the comparison tests in ``tests/test_fortran_ref.py``
feed identical inputs to both implementations and assert ULP-level
agreement.

Layout notes
------------
The Fortran kernels declare every array as ``(KLON, NTILES)`` with
column-major (Fortran) storage.  numpy / torch default to row-major
(C order), so the bridge hands the library **Fortran-contiguous**
buffers via ``np.asfortranarray`` on every 2-D array before taking its
``.ctypes.data_as`` pointer.  Scalars passed ``BY VALUE`` in Fortran
(``REAL, VALUE`` / ``INTEGER, VALUE`` / ``LOGICAL(C_BOOL), VALUE``)
map to plain ``c_double`` / ``c_int`` / ``c_bool`` arguments.

Constants
---------
All physical constants are passed in from the same
:class:`~surf_pytorch.constants.ConstantBundle` that the PyTorch port
consumes, so the two implementations share an identical
parameterisation; any numerical mismatch is then *purely* arithmetic
(rounding / re-association), which is exactly what the ULP-level test
is meant to surface.

The four VLAMSK scalars that are not in :class:`PhysicalConstants`
(``RHOICE`` / ``RHOCI`` / ``RQSNCR`` / ``RURBTC``) are pulled from
the same literals :mod:`surf_pytorch.physics.vlamsk` hard-wires, so
the two stay in sync.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

import numpy as np
import torch

from surf_pytorch import ConstantBundle, SurfConfig
from surf_pytorch.physics.seb import SebInputs, SebOutputs
from surf_pytorch.physics import vlamsk as vlamsk_mod
from surf_pytorch.physics import exchange as exchange_mod
from surf_pytorch.tables import VegTable
from surf_pytorch.tables.ritbl import (
    build_ritbl,
    JPRITBL,
    RIMAX,
    RCHETA,
    RCHETB,
    RCHBA,
    RCHBB,
    RCHBC,
    RCHBD,
    RCHB23A,
    RCHBBCD,
    RCHBCD,
    RCDHALF,
    RCDHPI2,
)


# --------------------------------------------------------------------------- #
# VLAMSK scalars that aren't in PhysicalConstants (mirrors vlamsk.py).
# --------------------------------------------------------------------------- #
RHO_ICE = vlamsk_mod.RHO_ICE          # 920.0 kg/m^3
RHO_CI = vlamsk_mod.RHO_CI            # 2.05e6 J/m^3/K (RHOCI)
RQSNCR_DEFAULT = vlamsk_mod.RQSNCR_DEFAULT     # 1/15 m (cy48r1 default)
RURBTC_DEFAULT = 20.0                 # urban tile skin conductivity


# --------------------------------------------------------------------------- #
# ctypes type aliases
# --------------------------------------------------------------------------- #
_F64 = ctypes.c_double
_I32 = ctypes.c_int
_BOOL = ctypes.c_bool
_P_F64 = ctypes.POINTER(_F64)
_P_I32 = ctypes.POINTER(_I32)
_P_BOOL = ctypes.POINTER(_BOOL)


class _LoadedLib:
    """Holds the loaded shared library and its configured FuncPtrs."""

    __slots__ = ("lib", "vlamsk", "seb", "vexcs", "dqsdt", "stomata", "vevap",
                 "vdfdifh", "soilt", "snow", "soilw", "farquhar")

    def __init__(self, lib, vlamsk, seb, vexcs, dqsdt, stomata, vevap, vdfdifh,
                 soilt, snow, soilw, farquhar) -> None:
        self.lib = lib
        self.vlamsk = vlamsk
        self.seb = seb
        self.vexcs = vexcs
        self.dqsdt = dqsdt
        self.stomata = stomata
        self.vevap = vevap
        self.vdfdifh = vdfdifh
        self.soilt = soilt
        self.snow = snow
        self.soilw = soilw
        self.farquhar = farquhar


def _default_so_path() -> str:
    """Locate ``libsurf_ref.so`` relative to this file.

    The build script writes the .so next to this module, so the
    sibling lookup is the common case.  We also honour an explicit
    ``SURF_REF_SO`` environment variable for out-of-tree builds.
    """
    env = os.environ.get("SURF_REF_SO")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "libsurf_ref.so")


def _configure_vlamsk(step) -> None:
    """argtypes for SURF_REF_VLAMSK -- mirrors the .F90 dummy order."""
    step.restype = None
    step.argtypes = [
        _I32, _I32,                              # KLON_C, NTILES_C
        _P_F64, _P_F64,                          # TSKM1M, TSRF (2D)
        _P_F64, _P_F64,                          # SNM, RSN (1D)
        _P_I32, _P_I32,                          # KTVL, KTVH (1D)
        _P_F64, _P_F64,                          # RVLAMSK, RVLAMSKS (0:20)
        _F64, _F64, _F64, _F64, _F64,            # RHOICE,RHOCI,RQSNCR,RURBTC,PTMST
        _BOOL,                                   # LSICOUP
        _P_F64,                                  # PLAMSK (2D out)
    ]


def _configure_seb(step) -> None:
    """argtypes for SURF_REF_SEB -- mirrors the .F90 dummy order."""
    step.restype = None
    step.argtypes = [
        _I32, _I32,                              # KLON_C, NTILES_C
        _P_F64, _P_F64, _P_F64, _P_F64,          # SSKM1M,TSKM1M,QSKM1M,DQSDT (2D)
        _P_F64, _P_F64, _P_F64, _P_F64,          # RHOCHU,RHOCQU,ALPHAL,ALPHAS (2D)
        _P_F64, _P_F64, _P_F64, _P_F64,          # SSRFL(2D),SLRFL,TSKRAD,EMIS (1D)
        _P_F64, _P_F64,                          # TSRF (2D), PLAMSK (2D)
        _P_F64,                                  # FRTI (2D)
        _P_F64, _P_F64, _P_F64, _P_F64,          # ASL,BSL,AQL,BQL (1D)
        _F64,                                    # PTMST (value)
        _P_F64,                                  # RVTRSR (0:20)
        _F64, _F64, _F64, _F64, _F64, _F64,      # RCPD,RLSTT,RLVTT,RSIGMA,RTT,RVTMP2
        _BOOL, _BOOL,                            # LELWDD, LELWTL
        _P_F64, _P_F64, _P_F64, _P_F64,          # PJS,PJQ,PSSK,PTSK (2D out)
        _P_F64, _P_F64, _P_F64, _P_F64,          # PSSH,PSLH,PSTR,PG0 (2D out)
        _P_F64, _P_F64,                          # PSL,PQL (1D out)
    ]


def _configure_vexcs(step) -> None:
    """argtypes for SURF_REF_VEXCS -- mirrors the .F90 dummy order."""
    step.restype = None
    step.argtypes = [
        _I32, _I32, _BOOL,                       # KLON_C, KITT_C, LDINIT
        _P_F64, _P_F64, _P_F64, _P_F64,          # PUMLEV,PVMLEV,PTMLEV,PQMLEV
        _P_F64, _P_F64,                          # PAPHMS, PGEOMLEV
        _P_F64, _P_F64, _P_F64,                  # PCPTGZLEV,PCPTS,PQSAM
        _P_F64, _P_F64, _P_F64,                  # PZ0MM,PZ0HM,PZ0QM
        _P_F64,                                  # PZDL (inout)
        _P_F64, _P_F64, _P_F64, _P_F64,          # PBUOM,PUCURR,PVCURR,PI10FGCV
        _P_F64,                                  # RITBL (JPRITBL)
        _F64, _F64, _F64, _F64, _F64, _F64,      # RD,RETV,RG,RVTMP2,RKAP,REPDU2
        _F64, _F64, _F64,                        # RPARZI,RCHBHDL,DRITBL
        _F64, _F64, _F64, _F64, _F64, _F64,      # RCHETA,RCHETB,RCHBA,RCHBB,RCHBC,RCHBD
        _F64, _F64, _F64, _F64, _F64,            # RCHB23A,RCHBBCD,RCHBCD,RCDHALF,RCDHPI2
        _I32,                                    # JPRITBL_C
        _P_F64, _P_F64, _P_F64,                  # PCFM,PCFH,PCFQ (1D out)
        _P_F64, _P_F64,                          # PKH,PKC (1D out)
    ]


def _configure_dqsdt(step) -> None:
    """argtypes for SURF_REF_DQSDT -- mirrors the .F90 dummy order."""
    step.restype = None
    step.argtypes = [
        _I32,                                    # KLON_C
        _P_F64, _P_F64, _P_F64, _P_F64,          # PTSKM1M,PAPHMS,PFRTI7,ZSALIN (1D)
        _BOOL,                                   # LCLAMP7
        _F64, _F64,                              # ZEPSILON, RETV_C
        _F64, _F64, _F64, _F64, _F64,            # R2ES,R3LES,R3IES,R4LES,R4IES
        _F64, _F64, _F64,                        # R5LES,R5IES,RTT
        _P_F64,                                  # PDQS (1D out)
    ]


def _configure_stomata(step) -> None:
    """argtypes for SURF_REF_STOMATA -- mirrors the .F90 dummy order."""
    step.restype = None
    step.argtypes = [
        _I32, _I32, _I32, _I32,                  # KLON_C, KLEVS_C, NSOTY_C, NTILES_C
        _P_F64, _P_F64,                          # PTSAM1M,PWSAM1M (2D: KLON,KLEVS)
        _P_I32, _P_I32, _P_I32,                  # KSOTY,KTVL,KTVH (1D)
        _P_F64, _P_F64, _P_F64, _P_F64,          # PTMLEV,PQMLEV,PAPHMS,PSRFD (1D)
        _P_F64, _P_F64,                          # PLAIL,PLAIH (1D)
        _P_F64,                                  # PFRTI (2D: KLON,NTILES)
        _P_F64, _P_F64,                          # RVRSMIN,RVHSTR (0:20)
        _P_F64,                                  # RVROOTSA (KLEVS,0:20)
        _P_F64, _P_F64, _P_F64, _P_F64,          # RWPWPM,RWCAPM,RWRESTM,RQWEVAPM (0:NSOTY)
        _F64, _F64, _F64, _F64,                  # RCEPSW,RCLU,RSMINB,ZEPSF3
        _F64, _F64, _F64, _F64, _F64, _F64, _F64,# RTF1..RTF4,RTT,RETV,RD
        _F64, _F64, _F64, _F64, _F64,            # R2ES,R3LES,R3IES,R4LES,R4IES
        _F64, _F64,                              # R5LES,R5IES
        _P_F64, _P_F64, _P_F64, _P_F64, _P_F64,  # PWETL,PWETH,PWETHS,PWETB,PWETLU (1D out)
        _P_F64,                                  # PEXDIAG (2D: KLON,10 out)
    ]


def _configure_vevap(step) -> None:
    """argtypes for SURF_REF_VEVAP -- mirrors the .F90 dummy order."""
    step.restype = None
    step.argtypes = [
        _I32, _I32,                              # KLON_C, KTILE_C
        _F64,                                    # PTMST (value)
        _P_F64,                                  # PWLMX (1D)
        _P_F64, _P_F64, _P_F64,                  # PTMLEV,PQMLEV,PAPHMS (1D)
        _P_F64, _P_F64,                          # PTSKM1M,PTSAM1M (1D)
        _P_F64, _P_F64,                          # PQS,PCFQ (1D)
        _P_F64, _P_F64,                          # PWETB,PWETL (1D)
        _P_F64, _P_F64, _P_F64,                  # PWETLU,PWETH,PWETHS (1D)
        _F64, _F64, _F64, _F64,                  # RCPD,RD,RETV,RVTMP2
        _F64, _F64,                              # RLHAERO,RLHAEROS
        _P_F64, _P_F64, _P_F64,                  # PCPTS,PCSAT,PCAIR (1D out)
        _P_F64, _P_F64, _P_F64,                  # PCPTSU,PCSATU,PCAIRU (1D out)
        _P_F64,                                  # PCSNW (1D out)
        _P_F64,                                  # PDHVEGS (3D: KLON,2,4 out)
    ]


def _configure_vdfdifh(step) -> None:
    """argtypes for SURF_REF_VDFDIFH -- mirrors the .F90 dummy order."""
    step.restype = None
    step.argtypes = [
        _I32, _I32,                              # KLON_C, NTILES_C
        _P_F64, _P_F64, _P_F64, _P_F64,          # PSLM1_LEV,PQTM1_LEV,PSLGE_LEV,PQTE_LEV (1D)
        _P_F64, _P_F64,                          # PCFHTI,PCFQTI (2D)
        _P_F64, _P_F64, _P_F64, _P_F64,          # PSST,PTICE,PTSM1M,PTSNOW (1D)
        _P_F64, _P_F64, _P_F64, _P_F64,          # SSKM1M,TSKM1M,QSKM1M,DQSDT (2D)
        _P_F64, _P_F64,                          # ALPHAL,ALPHAS (2D)
        _P_F64, _P_F64, _P_F64, _P_F64,          # PSSRFLTI(2D),PSLRFL,PTSKRAD,PEMIS (1D)
        _P_F64, _P_F64,                          # PLAMSK(2D),PFRTI(2D)
        _P_I32, _P_I32,                          # KTVL,KTVH (1D)
        _F64,                                    # PTMST (value)
        _P_F64,                                  # PEVAPSNW (1D)
        _P_F64, _P_F64, _P_F64,                  # PHLICE,PTLICE,PTLWML (1D)
        _P_F64,                                  # RVTRSR (0:20)
        _F64, _F64, _F64, _F64, _F64, _F64,      # RCPD,RLSTT,RLVTT,RSIGMA,RTT,RVTMP2
        _F64, _F64,                              # RG,RVDIFTS
        _BOOL, _BOOL, _BOOL, _BOOL,              # LESNML,HAS_EVAPSNW,LELWDD,LELWTL
        _P_F64, _P_F64, _P_F64, _P_F64, _P_F64,  # PJQ,PSSH,PSLH,PSTR,PG0 (2D out)
        _P_F64, _P_F64,                          # PTSK,PTSKTIP1 (2D out)
        _P_F64, _P_F64,                          # PSL,PQL (1D out)
    ]


def _configure_soilt(step) -> None:
    """argtypes for SURF_REF_SOILT -- mirrors the .F90 dummy order."""
    step.restype = None
    step.argtypes = [
        _I32, _I32, _I32, _I32,                  # KLON_C,KLEVS_C,NTILES_C,NSOTY_C
        _P_F64, _P_F64,                          # PTSAM1M,PWSAM1M (2D KLON,KLEVS)
        _P_F64, _P_F64, _P_F64, _P_F64,          # PFRTI,PAHFSTI,PEVAPTI,PSSRFLTI (2D KLON,NTILES)
        _P_F64, _P_F64,                          # PSLRFL,PGSN (1D)
        _P_I32,                                  # KSOTY (1D)
        _P_F64, _P_F64,                          # PCVL,PCVH (1D)
        _P_BOOL,                                 # LDLAND (1D)
        _P_F64,                                  # RDAT (KLEVS)
        _P_F64, _P_F64, _P_F64, _P_F64, _P_F64,  # RWSATM,RLAMBDADRYM,RLAMSAT1M,RRCSOILM,RWCAPM (NSOTY)
        _F64, _F64, _F64, _F64, _F64,            # PTMST,RSIMP,RLMLT,RHOH2O,RLVTT
        _F64, _F64, _F64, _F64,                  # RTF1,RTF2,RTF3,RTF4
        _F64, _F64,                              # RLAMBDAICE,RLAMBDAWAT
        _F64, _F64, _F64,                        # RKERST1,RKERST2,RKERST3
        _P_F64, _P_F64, _P_F64,                  # PTSA,PCTSA (2D out),PZSURFL (1D out)
    ]


def _configure_snow(step) -> None:
    """argtypes for SURF_REF_SNOW -- mirrors the .F90 dummy order."""
    step.restype = None
    step.argtypes = [
        _I32, _I32,                              # KLON_C, NTILES_C
        _P_F64, _P_F64, _P_F64, _P_F64,          # PSSNM1M,PTSNM1M,PRSNM1M,PTSAM1M_TOP (1D)
        _P_F64, _P_F64, _P_F64, _P_F64,          # PFRTI,PAHFSTI,PEVAPTI,PSSRFLTI (2D KLON,NTILES)
        _P_F64,                                  # PSLRFL (1D)
        _P_F64, _P_F64, _P_F64, _P_F64, _P_F64,  # PSSFC,PSSFL,PTSFC,PTSFL,PEVAPSNW (1D)
        _BOOL,                                   # LEFLAKE_C (by value)
        _F64,                                    # PTMST
        _F64, _F64, _F64, _F64, _F64, _F64,      # RTT,RLMLT,RLVTT,RLSTT,RPI,RDAY
        _F64, _F64, _F64, _F64, _F64,            # RHOICE,RLAMICE,RHOCI,RALAMSN,RDSNMAX
        _F64, _F64,                              # RFRSMALL,RFRTINY
        _F64, _F64, _F64, _F64,                  # RLWCSWEA,RLWCSWEB,RLWCSWEC,RTEMPAMP
        _F64, _F64,                              # RTAUF,RTAUA
        _F64,                                    # RVLAMSK_8
        _P_F64, _P_F64, _P_F64, _P_F64, _P_F64,  # PTSN,PGSN,PSSTAR,PLWC,PZFRSN (1D out)
    ]


def _configure_soilw(step) -> None:
    """argtypes for SURF_REF_SOILW -- mirrors the .F90 dummy order."""
    step.restype = None
    step.argtypes = [
        _I32, _I32, _I32, _I32, _I32,              # KLON_C,KLEVS_C,NTILES_C,NSOTY_C,NV_C
        _P_F64, _P_F64,                            # PWSAM1M,PTSAM1M (2D KLON,KLEVS)
        _P_F64, _P_F64, _P_F64,                    # PFRTI,PEVAPTI,PEINTTI (2D KLON,NTILES)
        _P_F64, _P_F64, _P_F64, _P_F64, _P_F64,    # PTSFC,PTSFL,PMSN,PEMSSN,PEVAPSNW (1D)
        _P_F64, _P_I32, _P_I32, _P_I32, _P_BOOL,   # PSDOR,KSOTY,KTVL,KTVH,LDLAND (1D)
        _P_F64, _P_F64,                            # RDAT (KLEVS), RVROOTSA (KLEVS,NV)
        _P_F64, _P_F64, _P_F64, _P_F64, _P_F64,    # RWSATM,RWRESTM,RWCAPM,RWPWPM,RWCONSM (NSOTY)
        _P_F64, _P_F64, _P_F64, _P_F64, _P_F64,    # RMVGALPHA,RMFACM,RLAMBDAM,RDMAXM,RDMINM (NSOTY)
        _F64, _F64, _F64,                          # RSIGORMAX,RSIGORMIN,RPSFR (value)
        _F64, _F64, _F64, _F64,                    # PTMST,RSIMP,RSRDEP,RHOH2O (value)
        _F64, _F64, _F64, _F64,                    # RTF1,RTF2,RTF3,RTF4 (value)
        _BOOL,                                     # LESSRO (value)
        _P_F64, _P_F64, _P_F64, _P_F64, _P_F64,    # PWSA (2D out),PROS,PROD,PWFSD (1D),PSAWGFL (2D out)
    ]


def _configure_farquhar(step) -> None:
    """argtypes for SURF_REF_FARQUHAR -- mirrors the .F90 dummy order.

    The signature is flattened: every path-vector (RARJV(2) etc.) appears
    as a pair of VALUE scalars (RARJV1, RARJV2).  All scalars are passed
    ``BY VALUE`` (REAL/INTEGER/LOGICAL, VALUE) so they map to ``c_double``
    / ``c_int`` / ``c_bool``; only the per-column and per-PFT-table arrays
    are passed by pointer.  Order is transcribed verbatim from the
    ``SURF_REF_FARQUHAR`` dummy list in surf_ref.F90.
    """
    step.restype = None
    step.argtypes = [
        _I32, _I32,                                  # KLON_C, NLAI_C
        _P_BOOL, _P_I32, _P_I32,                     # LDLAND, KVTYPE, KCO2TYP (1D)
        _P_F64, _P_F64,                              # PSRFD, PAPHM (1D)
        _P_F64, _P_F64, _P_F64,                      # PQS, PQSURF, PQM (1D)
        _P_F64, _P_F64, _P_F64,                      # PTM, PTSKM, PTSOIL (1D)
        _P_F64, _P_F64, _P_F64,                      # PF2, PCO2, PLAI (1D)
        # 8 per-PFT tables (0:20, 2), Fortran column-major.
        _P_F64, _P_F64, _P_F64, _P_F64, _P_F64,      # RVCMAX25,RHUMREL,RA1,RB1,RG0
        _P_F64, _P_F64, _P_F64,                      # RGM25, RE_VCMAX, RE_JMAX
        # Flattened path-pair VALUE scalars (2*NPATH each, C3 col then C4 col).
        _F64, _F64, _F64, _F64,                      # RARJV1/2, RBRJV1/2
        _F64, _F64, _F64, _F64,                      # RASJ1/2, RBSJ1/2
        _F64, _F64, _F64, _F64,                      # RASV1/2, RBSV1/2
        _F64, _F64, _F64, _F64,                      # RD_VCMAX1/2, RD_JMAX1/2
        _F64,                                        # RE_GAMMA_STAR
        _F64, _F64, _F64, _F64,                      # RE_GM1/2, RS_GM1/2
        _F64, _F64, _F64, _F64,                      # RS_JMAX1/2, RD_GM1/2
        # Class-agnostic VALUE scalars (sufarquhar_mod.F90).
        _F64, _F64, _F64, _F64,                      # RE_KMC, RE_KMO, RE_RD, RE_SCO
        _F64,                                        # RGAMMA_STAR25
        _F64, _F64, _F64, _F64,                      # RKMC25_1/2, RKMO25_1/2
        _F64, _F64,                                  # RSCO25_1/2
        _F64, _F64, _F64,                            # RSTRESS_GM/GS/VCMAX
        _F64, _F64, _F64,                            # RLAI_LEVEL_DEPTH, RLAIMAX, REXT_COEFF
        _F64, _F64, _F64, _F64,                      # RGB_REF, RTP_00, RPB_STD, RRG_TO_PAR
        _F64, _F64, _F64,                            # RW_TO_MOL, RALPHA_LL, RTHETA
        _F64, _F64, _F64, _F64,                      # RFPSIR, RFQ, RFPSEUDO, RH_PROTONS
        _F64, _F64, _F64, _F64,                      # RGBS, ROI, RALPHA, RKP
        _F64, _F64, _F64,                            # RTETENS_1/2, RREF_TEMP
        _F64, _F64,                                  # RMOL_TO_M_1, RRATIO_H2O_TO_CO2
        _F64,                                        # RN_VERT_ATT
        _F64, _F64,                                  # RPI_C, RR_C
        _F64, _F64,                                  # RMAIR_C, RMCO2_C
        _BOOL,                                       # LSKIN_TEMP_C
        # Outputs (1D).
        _P_F64, _P_F64, _P_F64, _P_F64,              # PGSTOT, PGPP, PRD, PAN
    ]


def _load_lib(path: Optional[str] = None) -> _LoadedLib:
    """Load the Fortran shared library and configure argtypes."""
    so_path = path or _default_so_path()
    if not os.path.exists(so_path):
        raise FileNotFoundError(
            f"libsurf_ref.so not found at {so_path!r}.  "
            "Build it with:  offline_driver/build_ref.sh"
        )
    lib = ctypes.CDLL(so_path)
    vlamsk_fn = lib.surf_ref_vlamsk
    seb_fn = lib.surf_ref_seb
    vexcs_fn = lib.surf_ref_vexcs
    # SURF_REF_DQSDT was added to the .so after the first M2 build; a
    # .so built from an older surf_ref.F90 will not export the symbol,
    # so degrade gracefully (tests that need it skip via is_available).
    dqsdt_fn = getattr(lib, "surf_ref_dqsdt", None)
    # SURF_REF_STOMATA likewise degrades gracefully if the .so predates
    # the Jarvis reference entry (see tests/test_stomata_ref.py).
    stomata_fn = getattr(lib, "surf_ref_stomata", None)
    # SURF_REF_VEVAP degrades gracefully if the .so predates the VEVAP
    # reference entry (see tests/test_vevap_ref.py).
    vevap_fn = getattr(lib, "surf_ref_vevap", None)
    # SURF_REF_VDFDIFH degrades gracefully if the .so predates the
    # offline-driver SURFSEB-wrapper reference entry (M4-rem-c).
    vdfdifh_fn = getattr(lib, "surf_ref_vdfdifh", None)
    # SURF_REF_SOILT degrades gracefully if the .so predates the M5c
    # soil-temperature reference entry.
    soilt_fn = getattr(lib, "surf_ref_soilt", None)
    # SURF_REF_SNOW degrades gracefully if the .so predates the M6b
    # single-layer-snow reference entry.
    snow_fn = getattr(lib, "surf_ref_snow", None)
    # SURF_REF_SOILW degrades gracefully if the .so predates the M5c-soilw
    # soil-moisture reference entry.
    soilw_fn = getattr(lib, "surf_ref_soilw", None)
    # SURF_REF_FARQUHAR degrades gracefully if the .so predates the P2.1
    # Farquhar photosynthesis reference entry (see tests/test_farquhar_ref.py).
    farquhar_fn = getattr(lib, "surf_ref_farquhar", None)
    _configure_vlamsk(vlamsk_fn)
    _configure_seb(seb_fn)
    _configure_vexcs(vexcs_fn)
    if dqsdt_fn is not None:
        _configure_dqsdt(dqsdt_fn)
    if stomata_fn is not None:
        _configure_stomata(stomata_fn)
    if vevap_fn is not None:
        _configure_vevap(vevap_fn)
    if vdfdifh_fn is not None:
        _configure_vdfdifh(vdfdifh_fn)
    if soilt_fn is not None:
        _configure_soilt(soilt_fn)
    if snow_fn is not None:
        _configure_snow(snow_fn)
    if soilw_fn is not None:
        _configure_soilw(soilw_fn)
    if farquhar_fn is not None:
        _configure_farquhar(farquhar_fn)
    return _LoadedLib(
        lib=lib, vlamsk=vlamsk_fn, seb=seb_fn, vexcs=vexcs_fn,
        dqsdt=dqsdt_fn, stomata=stomata_fn, vevap=vevap_fn,
        vdfdifh=vdfdifh_fn, soilt=soilt_fn, snow=snow_fn, soilw=soilw_fn,
        farquhar=farquhar_fn,
    )


# Module-level singleton: loading the .so is ~1 ms and idempotent.
_LIB_CACHE: dict[str, _LoadedLib] = {}


def _get_lib(path: Optional[str] = None) -> _LoadedLib:
    key = path or _default_so_path()
    if key not in _LIB_CACHE:
        _LIB_CACHE[key] = _load_lib(key)
    return _LIB_CACHE[key]


# --------------------------------------------------------------------------- #
# Buffer helpers -- promote torch tensors to Fortran-order numpy views.
# --------------------------------------------------------------------------- #
def _as_fortran_2d(t: torch.Tensor) -> np.ndarray:
    """Return a float64, Fortran-contiguous ``(KLON, NTILES)`` view.

    The Fortran kernel indexes ``X(JL, JT)`` with the leading
    dimension ``KLON``, so the buffer *must* be column-major.  We
    detach + cpu + numpy and let ``asfortranarray`` copy if needed.
    """
    arr = t.detach().to("cpu", torch.float64).numpy()
    return np.asfortranarray(arr, dtype=np.float64)


def _as_contig_1d(t: torch.Tensor) -> np.ndarray:
    """Return a float64, C-contiguous ``(KLON,)`` view."""
    arr = t.detach().to("cpu", torch.float64).numpy().ravel()
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_int_1d(t: torch.Tensor) -> np.ndarray:
    """Return an int32, C-contiguous ``(KLON,)`` view (for KTVL/KTVH)."""
    arr = t.detach().to("cpu").to(torch.int32).numpy().ravel()
    return np.ascontiguousarray(arr, dtype=np.int32)


def _ptr_f64(arr: np.ndarray):
    return arr.ctypes.data_as(_P_F64)


def _ptr_i32(arr: np.ndarray):
    return arr.ctypes.data_as(_P_I32)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class SurfRef:
    """Stateful wrapper around the two Fortran reference kernels.

    Construct once (loads the .so), call :meth:`vlamsk` / :meth:`seb`
    per comparison.  Cheap to instantiate because the .so load is
    cached at module level.
    """

    def __init__(self, so_path: Optional[str] = None) -> None:
        self._loaded = _get_lib(so_path)

    @staticmethod
    def is_available(so_path: Optional[str] = None) -> bool:
        """True iff the .so exists at the expected path.

        Tests use this to skip cleanly when the Fortran reference has
        not been built (e.g. on CI without gfortran).
        """
        return os.path.exists(so_path or _default_so_path())

    # ------------------------------------------------------------------ #
    # VLAMSK
    # ------------------------------------------------------------------ #
    def vlamsk(
        self,
        ptskm1m: torch.Tensor,
        ptsrf: torch.Tensor,
        psnm: torch.Tensor,
        prsn: torch.Tensor,
        ktvl: torch.Tensor,
        ktvh: torch.Tensor,
        veg: VegTable,
        consts: ConstantBundle,
        config: SurfConfig,
        *,
        ptstep: float = 3600.0,
        lsicoup: bool = False,
        rhoice: float = RHO_ICE,
        rhoci: float = RHO_CI,
        rqsmcr: float = RQSNCR_DEFAULT,
        rurban_c: float = RURBTC_DEFAULT,
    ) -> torch.Tensor:
        """Run VLAMSK via the Fortran kernel; return PLAMSK (Ngrid, Ntile).

        Argument order mirrors :func:`surf_pytorch.physics.vlamsk.vlamsk`
        so the test can feed identical tensors to both.  Returns a
        torch tensor with the same dtype/device as ``ptskm1m``.
        """
        ngrid, ntile = ptskm1m.shape
        tskm1m = _as_fortran_2d(ptskm1m)
        tsrf = _as_fortran_2d(ptsrf)
        snm = _as_contig_1d(psnm)
        rsn = _as_contig_1d(prsn)
        ktvl_arr = _as_int_1d(ktvl)
        ktvh_arr = _as_int_1d(ktvh)
        rvlamsk = _as_contig_1d(veg.rvlamsk)
        rvlamsks = _as_contig_1d(veg.rvlamsks)
        plamsk = np.zeros((ngrid, ntile), dtype=np.float64, order="F")

        self._loaded.vlamsk(
            _I32(ngrid), _I32(ntile),
            _ptr_f64(tskm1m), _ptr_f64(tsrf),
            _ptr_f64(snm), _ptr_f64(rsn),
            _ptr_i32(ktvl_arr), _ptr_i32(ktvh_arr),
            _ptr_f64(rvlamsk), _ptr_f64(rvlamsks),
            _F64(rhoice), _F64(rhoci), _F64(rqsmcr), _F64(rurban_c),
            _F64(ptstep), _BOOL(lsicoup),
            _ptr_f64(plamsk),
        )
        return (
            torch.from_numpy(plamsk.copy())
            .to(ptskm1m.dtype).to(ptskm1m.device)
        )

    # ------------------------------------------------------------------ #
    # SURFSEB_CTL
    # ------------------------------------------------------------------ #
    def seb(
        self,
        inp: SebInputs,
        veg: VegTable,
        consts: ConstantBundle,
        config: SurfConfig,
        *,
        lamsk: Optional[torch.Tensor] = None,
    ) -> SebOutputs:
        """Run SURFSEB_CTL via the Fortran kernel.

        Args mirror :func:`surf_pytorch.physics.seb.surf_seb`.  ``lamsk``
        overrides ``inp.lamsk`` (the per-tile skin conductivity produced
        by VLAMSK) -- the comparison test passes the same VLAMSK output
        to both implementations through this argument.

        Returns a :class:`SebOutputs` with the same field layout as the
        port's.
        """
        ngrid, ntile = inp.sskm1m.shape
        lamsk_t = lamsk if lamsk is not None else inp.lamsk

        sskm1m = _as_fortran_2d(inp.sskm1m)
        tskm1m = _as_fortran_2d(inp.tskm1m)
        qskm1m = _as_fortran_2d(inp.qskm1m)
        dqsdT = _as_fortran_2d(inp.dqsdT)
        rhochu = _as_fortran_2d(inp.rhochu)
        rhocqu = _as_fortran_2d(inp.rhocqu)
        alphal = _as_fortran_2d(inp.alphal)
        alphas = _as_fortran_2d(inp.alphas)
        # ssrfl is per-(grid,tile) in SebInputs; expand only if a caller
        # passed a 1-D tensor.
        ssrfl_2d = inp.ssrfl if inp.ssrfl.dim() == 2 else \
            inp.ssrfl.unsqueeze(-1).expand(ngrid, ntile)
        ssrfl_arr = _as_fortran_2d(ssrfl_2d)
        tsrf = _as_fortran_2d(inp.tsrf)
        plamsk = _as_fortran_2d(lamsk_t)
        frti = _as_fortran_2d(inp.frti)

        slrfl = _as_contig_1d(inp.slrfl_grid)
        tskrad = _as_contig_1d(inp.tskrad_grid)
        emis = _as_contig_1d(inp.emis_grid)
        asl = _as_contig_1d(inp.asl_grid)
        bsl = _as_contig_1d(inp.bsl_grid)
        aql = _as_contig_1d(inp.aql_grid)
        bql = _as_contig_1d(inp.bql_grid)

        rvtrsr = _as_contig_1d(veg.rvtrsr)

        rcpd = float(consts.cpd.item())
        rlstt = float(consts.rlstt.item())
        rlvtt = float(consts.rlvtt.item())
        rsigma = float(consts.rsigma.item())
        rtt = float(consts.rtt.item())
        rvtmp2 = float(consts.rvtmp2.item())
        ptmst = float(inp.ptmst.item()) if inp.ptmst is not None else 3600.0

        pjs = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        pjq = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        pssk = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        ptsk = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        pssh = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        pslh = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        pstr = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        pg0 = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        psl = np.zeros(ngrid, dtype=np.float64)
        pql = np.zeros(ngrid, dtype=np.float64)

        self._loaded.seb(
            _I32(ngrid), _I32(ntile),
            _ptr_f64(sskm1m), _ptr_f64(tskm1m), _ptr_f64(qskm1m), _ptr_f64(dqsdT),
            _ptr_f64(rhochu), _ptr_f64(rhocqu), _ptr_f64(alphal), _ptr_f64(alphas),
            _ptr_f64(ssrfl_arr), _ptr_f64(slrfl), _ptr_f64(tskrad), _ptr_f64(emis),
            _ptr_f64(tsrf), _ptr_f64(plamsk),
            _ptr_f64(frti),
            _ptr_f64(asl), _ptr_f64(bsl), _ptr_f64(aql), _ptr_f64(bql),
            _F64(ptmst),
            _ptr_f64(rvtrsr),
            _F64(rcpd), _F64(rlstt), _F64(rlvtt), _F64(rsigma), _F64(rtt),
            _F64(rvtmp2),
            _BOOL(bool(inp.lelwdd)), _BOOL(bool(inp.lelwtl)),
            _ptr_f64(pjs), _ptr_f64(pjq), _ptr_f64(pssk), _ptr_f64(ptsk),
            _ptr_f64(pssh), _ptr_f64(pslh), _ptr_f64(pstr), _ptr_f64(pg0),
            _ptr_f64(psl), _ptr_f64(pql),
        )

        out_dtype = inp.sskm1m.dtype
        out_device = inp.sskm1m.device

        def _t2(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        def _t1(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        return SebOutputs(
            pjs=_t2(pjs),
            pjq=_t2(pjq),
            pssk=_t2(pssk),
            ptsk=_t2(ptsk),
            pssh=_t2(pssh),
            pslh=_t2(pslh),
            pstr=_t2(pstr),
            pg0=_t2(pg0),
            psl=_t1(psl),
            pql=_t1(pql),
        )

    # ------------------------------------------------------------------ #
    # VEXCS
    # ------------------------------------------------------------------ #
    def vexcs(
        self,
        pu: torch.Tensor,
        pv: torch.Tensor,
        pt: torch.Tensor,
        pq: torch.Tensor,
        paphms: torch.Tensor,
        pgeomlev: torch.Tensor,
        pcptgzlev: torch.Tensor,
        pcpts: torch.Tensor,
        pqsam: torch.Tensor,
        pz0mm: torch.Tensor,
        pz0hm: torch.Tensor,
        pz0qm: torch.Tensor,
        pbuom: torch.Tensor,
        pucurr: torch.Tensor,
        pvcurr: torch.Tensor,
        pi10fgcv: torch.Tensor,
        consts: ConstantBundle,
        *,
        pzdl0: Optional[torch.Tensor] = None,
        kitt: int = exchange_mod.DEFAULT_KITT,
        ldinit: bool = True,
    ) -> "exchange_mod.ExchangeOutputs":
        """Run VEXCS via the Fortran kernel; return scaled transfer coeffs.

        All inputs are ``(Ngrid,)``.  Argument names mirror the upstream
        ``VEXCS`` dummy list (see vexcs_mod.F90) and the port's
        :func:`surf_pytorch.physics.exchange.vexcs`.  Returns an
        :class:`~surf_pytorch.physics.exchange.ExchangeOutputs` whose
        tensors share ``pu``'s dtype/device.

        The RITBL lookup table is rebuilt here (cheap, ~101 Newton
        iterations) and passed to the kernel; the constants bundle and
        the ``exchange_mod`` PSI scalars provide everything else.
        """
        ngrid = pu.shape[0]
        pumlev = _as_contig_1d(pu)
        pvmlev = _as_contig_1d(pv)
        ptmlev = _as_contig_1d(pt)
        pqmlev = _as_contig_1d(pq)
        paphms_a = _as_contig_1d(paphms)
        pgeomlev_a = _as_contig_1d(pgeomlev)
        pcptgzlev_a = _as_contig_1d(pcptgzlev)
        pcpts_a = _as_contig_1d(pcpts)
        pqsam_a = _as_contig_1d(pqsam)
        pz0mm_a = _as_contig_1d(pz0mm)
        pz0hm_a = _as_contig_1d(pz0hm)
        pz0qm_a = _as_contig_1d(pz0qm)
        pbuom_a = _as_contig_1d(pbuom)
        pucurr_a = _as_contig_1d(pucurr)
        pvcurr_a = _as_contig_1d(pvcurr)
        pi10fgcv_a = _as_contig_1d(pi10fgcv)

        # PZDL is INOUT -- seed with the caller's guess or zero.
        if pzdl0 is not None:
            pzdl_a = _as_contig_1d(pzdl0).copy()
        else:
            pzdl_a = np.zeros(ngrid, dtype=np.float64)

        ritbl = np.ascontiguousarray(build_ritbl(), dtype=np.float64)

        # Scalar constants.
        rd = float(consts.rd.item())
        retv = float(consts.retv.item())
        rg = float(consts.g.item())
        rvtmp2 = float(consts.rvtmp2.item())

        pcfm = np.zeros(ngrid, dtype=np.float64)
        pcfh = np.zeros(ngrid, dtype=np.float64)
        pcfq = np.zeros(ngrid, dtype=np.float64)
        pkh = np.zeros(ngrid, dtype=np.float64)
        pkc = np.zeros(ngrid, dtype=np.float64)

        self._loaded.vexcs(
            _I32(ngrid), _I32(kitt), _BOOL(ldinit),
            _ptr_f64(pumlev), _ptr_f64(pvmlev), _ptr_f64(ptmlev),
            _ptr_f64(pqmlev),
            _ptr_f64(paphms_a), _ptr_f64(pgeomlev_a),
            _ptr_f64(pcptgzlev_a), _ptr_f64(pcpts_a), _ptr_f64(pqsam_a),
            _ptr_f64(pz0mm_a), _ptr_f64(pz0hm_a), _ptr_f64(pz0qm_a),
            _ptr_f64(pzdl_a),
            _ptr_f64(pbuom_a), _ptr_f64(pucurr_a),
            _ptr_f64(pvcurr_a), _ptr_f64(pi10fgcv_a),
            _ptr_f64(ritbl),
            _F64(rd), _F64(retv), _F64(rg), _F64(rvtmp2),
            _F64(exchange_mod.RKAP), _F64(exchange_mod.REPDU2),
            _F64(exchange_mod.RPARZI), _F64(exchange_mod.RCHBHDL),
            _F64(RIMAX / (JPRITBL - 1)),                 # DRITBL
            # PSI-function constants (shared with the port via ritbl.py).
            _F64(RCHETA), _F64(RCHETB),
            _F64(RCHBA), _F64(RCHBB), _F64(RCHBC), _F64(RCHBD),
            _F64(RCHB23A), _F64(RCHBBCD), _F64(RCHBCD),
            _F64(RCDHALF), _F64(RCDHPI2),
            _I32(JPRITBL),
            _ptr_f64(pcfm), _ptr_f64(pcfh), _ptr_f64(pcfq),
            _ptr_f64(pkh), _ptr_f64(pkc),
        )

        out_dtype = pu.dtype
        out_device = pu.device

        def _t1(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        return exchange_mod.ExchangeOutputs(
            pcfm=_t1(pcfm), pcfh=_t1(pcfh), pcfq=_t1(pcfq),
            pkh=_t1(pkh), pkc=_t1(pkc),
            pzdl=_t1(pzdl_a),
        )

    # ------------------------------------------------------------------ #
    # DQSDT (vsurf_mod.F90 saturation Jacobian)
    # ------------------------------------------------------------------ #
    def dqsdt(
        self,
        ptskm1m: torch.Tensor,
        paphms: torch.Tensor,
        consts: ConstantBundle,
        *,
        salinity: "torch.Tensor | float | None" = None,
        pfrti7: "torch.Tensor | None" = None,
        lclamp7: bool = False,
        zepsilon: float = 0.0,
    ) -> torch.Tensor:
        """Run the vsurf_mod.F90 (278-285) PDQS computation in Fortran.

        This is the *independent* reference for the saturation derivative
        that ``utils.dqsat_dt_ifs`` produces on the Python side: the
        Fortran kernel re-derives PDQS from the upstream ``fcttre.h``
        statement functions (FOEEW / FOEDESU / FOEDELTA) rather than
        consuming a Python-computed value, so a comparison test between
        the two is a genuine check of the dqs/dT *formula*, not a
        self-transcription (which ``SURF_REF_SEB`` -- taking DQSDT as an
        input -- cannot provide).

        Args mirror :func:`surf_pytorch.utils.dqsat_dt_ifs`.  All inputs
        are ``(Ngrid,)``.  ``salinity`` is the per-column ZSALIN (0.98 for
        ocean under LEOCSA, 1.0 elsewhere); it broadcasts against the grid.
        ``pfrti7`` / ``lclamp7`` gate the KTILE==7 Tsk clamp (vsurf
        273-277); the offline default is no clamp (``pfrti7`` ignored).

        Returns a torch tensor with ``ptskm1m``'s dtype/device.
        """
        ngrid = ptskm1m.shape[0]
        ptskm1m_a = _as_contig_1d(ptskm1m)
        paphms_a = _as_contig_1d(paphms)
        if isinstance(salinity, torch.Tensor):
            zsalin_a = _as_contig_1d(salinity)
        else:
            zsalin_a = np.full(ngrid, 1.0 if salinity is None else float(salinity),
                               dtype=np.float64)
        if pfrti7 is not None:
            pfrti7_a = _as_contig_1d(pfrti7)
        else:
            # Clamp branch never fires when LCLAMP7 is False; pass zeros.
            pfrti7_a = np.zeros(ngrid, dtype=np.float64)

        # fcttre.h YOETHF coefficients, pulled from the same ConstantBundle
        # the port uses so the two share an identical parameterisation.
        r2es = float(consts.r2es.item())
        r3les = float(consts.r3les.item())
        r3ies = float(consts.r3ies.item())
        r4les = float(consts.r4les.item())
        r4ies = float(consts.r4ies.item())
        r5les = float(consts.r5les.item())
        r5ies = float(consts.r5ies.item())
        rtt = float(consts.rtt.item())
        retv = float(consts.retv.item())

        pdqs = np.zeros(ngrid, dtype=np.float64)

        self._loaded.dqsdt(
            _I32(ngrid),
            _ptr_f64(ptskm1m_a), _ptr_f64(paphms_a),
            _ptr_f64(pfrti7_a), _ptr_f64(zsalin_a),
            _BOOL(bool(lclamp7)),
            _F64(float(zepsilon)), _F64(retv),
            _F64(r2es), _F64(r3les), _F64(r3ies), _F64(r4les), _F64(r4ies),
            _F64(r5les), _F64(r5ies), _F64(rtt),
            _ptr_f64(pdqs),
        )
        return torch.from_numpy(pdqs.copy()).to(ptskm1m.dtype).to(ptskm1m.device)

    # ------------------------------------------------------------------ #
    # STOMATA (vsurf_mod.F90 Jarvis resistance, KTILE==4 block)
    # ------------------------------------------------------------------ #
    def stomata(
        self,
        soil_t: torch.Tensor,
        soil_w: torch.Tensor,
        soil_type: torch.Tensor,
        ktvl: torch.Tensor,
        ktvh: torch.Tensor,
        t_air: torch.Tensor,
        q_air: torch.Tensor,
        p_surf: torch.Tensor,
        sw_down: torch.Tensor,
        veg: VegTable,
        soil_table,
        root_table,
        consts: ConstantBundle,
        config: SurfConfig,
        *,
        tile_frac: "torch.Tensor | None" = None,
    ) -> dict:
        """Run the vsurf_mod.F90 (294-422) Jarvis resistance in Fortran.

        Independent reference for :func:`surf_pytorch.physics.stomata.
        jarvis_resistance`: the Fortran kernel re-derives PWETL/PWETH/
        PWETHS/PWETB/PWETLU from the upstream vsurf block, so a
        comparison test against the port is a genuine check of the
        resistance *formula* (including the freeze shutdown, root-
        weighted moisture, Betts radiation stress and VPD F3).

        Argument names mirror the port's ``jarvis_resistance`` so the
        test can feed identical tensors to both.  Returns a dict with
        the same keys (``pwetl`` / ``pweth`` / ``pweths`` / ``pwetb`` /
        ``pwetlu``) plus the 10-column diagnostic ``exdiag`` tensor
        (vsurf 412-421: PWETL, F1L, F2L, F3L, LAIL, PWETH, F1H, F2H,
        F3H, LAIH).

        ``tile_frac`` defaults to a zero array (so ZBARE=0, matching
        vsurf's behaviour when only the per-column fractions are used
        for tile weighting rather than the bare-soil F2 blend).  Pass
        the model's per-tile fractions for a full vsurf match.
        """
        if self._loaded.stomata is None:
            raise RuntimeError(
                "SURF_REF_STOMATA not exported by this .so -- rebuild with "
                "offline_driver/build_ref.sh after adding the stomata entry."
            )
        ngrid, nsoil = soil_t.shape
        ntile = config.n_tile
        nsoty = config.n_soil_types

        # 2-D arrays need Fortran column-major storage (KLON leading).
        ptsam1m = _as_fortran_2d(soil_t)
        pwsam1m = _as_fortran_2d(soil_w)
        if tile_frac is None:
            pfrti = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        else:
            pfrti = _as_fortran_2d(tile_frac)

        ksoty = _as_int_1d(soil_type)
        ktvl_a = _as_int_1d(ktvl)
        ktvh_a = _as_int_1d(ktvh)
        ptmlev = _as_contig_1d(t_air)
        pqmlev = _as_contig_1d(q_air)
        paphms = _as_contig_1d(p_surf)
        psrfd = _as_contig_1d(sw_down)

        # LAI per column -- pulled from the VegTable so the ref and the
        # port share identical LAI values (vsurf reads PLAIL/PLAIH, which
        # the offline driver fills from the RVLAI column at KTVL/KTVH).
        plail = veg.gather("rvlai", ktvl).detach().to("cpu", torch.float64).numpy()
        plaih = veg.gather("rvlai", ktvh).detach().to("cpu", torch.float64).numpy()
        plail = np.ascontiguousarray(plail.ravel(), dtype=np.float64)
        plaih = np.ascontiguousarray(plaih.ravel(), dtype=np.float64)

        # LUTs.
        rvrsmin = _as_contig_1d(veg.rvrsmin)
        rvhstr = _as_contig_1d(veg.rvhstr)
        # RVROOTSA stored as (nsoil, 0:20) on the RootTable; vsurf indexes
        # RVROOTSA(JK, KTVL) so we hand the (KLEVS, 0:20) buffer directly.
        rvrootsa = root_table.rvrootsa.detach().to("cpu", torch.float64).numpy()
        rvrootsa = np.asfortranarray(rvrootsa, dtype=np.float64)
        rwpwpm = _as_contig_1d(soil_table.wpwp)
        rwcapm = _as_contig_1d(soil_table.wcap)
        rwrestm = _as_contig_1d(soil_table.wrest)
        rqwevapm = _as_contig_1d(soil_table.qwevap)

        # Jarvis scalars + physical constants.
        from surf_pytorch.physics import stomata as stomata_mod
        rcepsw = stomata_mod.RCEPSW
        rclu = stomata_mod.RCLU
        rsminb = stomata_mod.RSMIN_B
        zepsf3 = stomata_mod.ZEPSF3

        rtf1 = float(consts.rtf1.item())
        rtf2 = float(consts.rtf2.item())
        rtf3 = float(consts.rtf3.item())
        rtf4 = float(consts.rtf4.item())
        rtt = float(consts.rtt.item())
        retv = float(consts.retv.item())
        rd = float(consts.rd.item())
        r2es = float(consts.r2es.item())
        r3les = float(consts.r3les.item())
        r3ies = float(consts.r3ies.item())
        r4les = float(consts.r4les.item())
        r4ies = float(consts.r4ies.item())
        r5les = float(consts.r5les.item())
        r5ies = float(consts.r5ies.item())

        pwetl = np.zeros(ngrid, dtype=np.float64)
        pweth = np.zeros(ngrid, dtype=np.float64)
        pweths = np.zeros(ngrid, dtype=np.float64)
        pwetb = np.zeros(ngrid, dtype=np.float64)
        pwetlu = np.zeros(ngrid, dtype=np.float64)
        pexdiag = np.zeros((ngrid, 10), dtype=np.float64, order="F")

        self._loaded.stomata(
            _I32(ngrid), _I32(nsoil), _I32(nsoty), _I32(ntile),
            _ptr_f64(ptsam1m), _ptr_f64(pwsam1m),
            _ptr_i32(ksoty), _ptr_i32(ktvl_a), _ptr_i32(ktvh_a),
            _ptr_f64(ptmlev), _ptr_f64(pqmlev), _ptr_f64(paphms), _ptr_f64(psrfd),
            _ptr_f64(plail), _ptr_f64(plaih),
            _ptr_f64(pfrti),
            _ptr_f64(rvrsmin), _ptr_f64(rvhstr),
            _ptr_f64(rvrootsa),
            _ptr_f64(rwpwpm), _ptr_f64(rwcapm), _ptr_f64(rwrestm), _ptr_f64(rqwevapm),
            _F64(rcepsw), _F64(rclu), _F64(rsminb), _F64(zepsf3),
            _F64(rtf1), _F64(rtf2), _F64(rtf3), _F64(rtf4), _F64(rtt), _F64(retv), _F64(rd),
            _F64(r2es), _F64(r3les), _F64(r3ies), _F64(r4les), _F64(r4ies),
            _F64(r5les), _F64(r5ies),
            _ptr_f64(pwetl), _ptr_f64(pweth), _ptr_f64(pweths), _ptr_f64(pwetb), _ptr_f64(pwetlu),
            _ptr_f64(pexdiag),
        )

        out_dtype = soil_t.dtype
        out_device = soil_t.device

        def _t1(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        return {
            "pwetl": _t1(pwetl), "pweth": _t1(pweth), "pweths": _t1(pweths),
            "pwetb": _t1(pwetb), "pwetlu": _t1(pwetlu),
            "exdiag": _t1(pexdiag),
        }

    # ------------------------------------------------------------------ #
    # VEVAP (equivalent evapotranspiration efficiency per tile)
    # ------------------------------------------------------------------ #
    def vevap(
        self,
        ktile: int,
        pwet: torch.Tensor,
        pwetlu: torch.Tensor,
        rhocqu: torch.Tensor,
        p_surf: torch.Tensor,
        t_air: torch.Tensor,
        q_air: torch.Tensor,
        qs_skin: torch.Tensor,
        skin_t: torch.Tensor,
        t_sub: torch.Tensor,
        wlmx: torch.Tensor,
        consts: ConstantBundle,
        *,
        ptmst: float = 3600.0,
        rlhaero: float = 15.0,
        rlhaeros: float = 10.0,
    ) -> dict:
        """Run vevap_mod.F90 (138-269) for a single tile in Fortran.

        Independent reference for :func:`surf_pytorch.physics.stomata.
        vevap`: the Fortran kernel re-derives PCSAT/PCAIR/PCSATU/PCAIRU/
        PCPTSU/PCSNW/PDHVEGS from the upstream VEVAP block, so a
        comparison test against the port is a genuine check of the
        *evapotranspiration-efficiency formula* (including the
        interception-reservoir clamp, the snow-under-canopy ZRAS branch
        and the rho convention).

        Argument names mirror the port's single-tile ``vevap()`` so the
        test can feed identical tensors to both.  ``ktile`` is the
        upstream 1-based tile selector (3/4/6/7/8/10).  Returns a dict
        with keys ``pcsat`` / ``pcair`` / ``pcsatu`` / ``pcairu`` /
        ``pcptsu`` / ``pcpts`` / ``pcsnw`` / ``pdhvegs``.
        """
        if self._loaded.vevap is None:
            raise RuntimeError(
                "SURF_REF_VEVAP not exported by this .so -- rebuild with "
                "offline_driver/build_ref.sh after adding the vevap entry."
            )
        ngrid = rhocqu.shape[0]
        out_dtype = rhocqu.dtype
        out_device = rhocqu.device

        # 1-D inputs (KLON).
        pwlmx_a = _as_contig_1d(wlmx)
        ptmlev = _as_contig_1d(t_air)
        pqmlev = _as_contig_1d(q_air)
        paphms = _as_contig_1d(p_surf)
        ptskm1m = _as_contig_1d(skin_t)
        ptsam1m = _as_contig_1d(t_sub)
        pqs = _as_contig_1d(qs_skin)
        pcfq = _as_contig_1d(rhocqu)
        pwetb_a = _as_contig_1d(pwet)        # PWETB slot (unused unless KTILE 8/10)
        pwetl_a = _as_contig_1d(pwet)        # PWETL slot (unused unless KTILE 4)
        pwetlu_a = _as_contig_1d(pwetlu)
        pweth_a = _as_contig_1d(pwet)        # PWETH slot (unused unless KTILE 6)
        pweths_a = _as_contig_1d(pwet)       # PWETHS slot (unused unless KTILE 7)

        rcpd = float(consts.cpd.item()) if torch.is_tensor(consts.cpd) else float(consts.cpd)
        rd = float(consts.rd.item())
        retv = float(consts.retv.item())
        # rvtmp2 lives directly on ConstantBundle (yos_thf scalar).
        rvtmp2 = float(consts.rvtmp2.item())

        # Output buffers.
        pcpts = np.zeros(ngrid, dtype=np.float64)
        pcsat = np.zeros(ngrid, dtype=np.float64)
        pcair = np.zeros(ngrid, dtype=np.float64)
        pcptsu = np.zeros(ngrid, dtype=np.float64)
        pcsatu = np.zeros(ngrid, dtype=np.float64)
        pcairu = np.zeros(ngrid, dtype=np.float64)
        pcsnw = np.zeros(ngrid, dtype=np.float64)
        pdhvegs = np.zeros((ngrid, 2, 4), dtype=np.float64, order="F")

        self._loaded.vevap(
            _I32(ngrid), _I32(int(ktile)),
            _F64(float(ptmst)),
            _ptr_f64(pwlmx_a),
            _ptr_f64(ptmlev), _ptr_f64(pqmlev), _ptr_f64(paphms),
            _ptr_f64(ptskm1m), _ptr_f64(ptsam1m),
            _ptr_f64(pqs), _ptr_f64(pcfq),
            _ptr_f64(pwetb_a), _ptr_f64(pwetl_a),
            _ptr_f64(pwetlu_a), _ptr_f64(pweth_a), _ptr_f64(pweths_a),
            _F64(rcpd), _F64(rd), _F64(retv), _F64(rvtmp2),
            _F64(float(rlhaero)), _F64(float(rlhaeros)),
            _ptr_f64(pcpts), _ptr_f64(pcsat), _ptr_f64(pcair),
            _ptr_f64(pcptsu), _ptr_f64(pcsatu), _ptr_f64(pcairu),
            _ptr_f64(pcsnw),
            _ptr_f64(pdhvegs),
        )

        def _t1(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        return {
            "pcsat": _t1(pcsat), "pcair": _t1(pcair),
            "pcsatu": _t1(pcsatu), "pcairu": _t1(pcairu),
            "pcptsu": _t1(pcptsu), "pcpts": _t1(pcpts),
            "pcsnw": _t1(pcsnw),
            "pdhvegs": _t1(pdhvegs),
        }

    # ------------------------------------------------------------------ #
    # VDFDIFH (offline single-column SURFSEB wrapper)
    # ------------------------------------------------------------------ #
    def vdfdifh(
        self,
        inp: "DiffusionInputs",
        veg: VegTable,
        consts: ConstantBundle,
        config: SurfConfig,
    ) -> "DiffusionOutputs":
        """Run the offline-driver VDFDIFH1S in Fortran.

        Independent reference for
        :func:`surf_pytorch.physics.diffusion.vdfdifh1s`: the Fortran
        kernel re-derives the standalone Dirichlet forcing, exchanges
        the inversion (ZCOEF1), per-tile PTSRF, the canonical SURFSEB
        call #3, the optional LESNML snow-melt repeat (#4), the snow-
        under-forest evaporation addition (tile 7), and the skin-T
        extrapolation to t+1, so a comparison test against the port is
        a genuine check of the *packaging* logic (not just the inner
        SURFSEB, which is already validated by ``seb()``).

        Args mirror :func:`surf_pytorch.physics.diffusion.vdfdifh1s`
        and consume the same :class:`DiffusionInputs` dataclass so the
        test can feed identical tensors to both.  Returns a
        :class:`DiffusionOutputs` with the same field layout.
        """
        if self._loaded.vdfdifh is None:
            raise RuntimeError(
                "SURF_REF_VDFDIFH not exported by this .so -- rebuild with "
                "offline_driver/build_ref.sh after adding the vdfdifh entry."
            )
        from surf_pytorch.physics.diffusion import (
            DiffusionOutputs, RVDIFTS,
        )

        ngrid, ntile = inp.sskm1m.shape

        # 1-D inputs (per grid).
        pslm1_lev = _as_contig_1d(inp.pslm1_lev)
        pqtm1_lev = _as_contig_1d(inp.pqtm1_lev)
        pslge_lev = _as_contig_1d(inp.pslge_lev)
        pqte_lev = _as_contig_1d(inp.pqte_lev)
        psst = _as_contig_1d(inp.psst)
        ptice = _as_contig_1d(inp.ptice)
        ptsm1m = _as_contig_1d(inp.ptsm1m)
        ptsnow = _as_contig_1d(inp.ptsnow)
        pslrfl = _as_contig_1d(inp.pslrfl)
        ptskrad = _as_contig_1d(inp.ptskrad)
        pemis = _as_contig_1d(inp.pemis)
        ktvl = _as_int_1d(inp.ktvl)
        ktvh = _as_int_1d(inp.ktvh)

        # PEVAPSNW may be None (HAS_EVAPSNW=False path).  The kernel
        # still needs a buffer; pass zeros and a False flag so it is
        # ignored.
        has_evapsnw = inp.pevapsnw is not None and ntile > 6
        if inp.pevapsnw is not None:
            pevapsnw = _as_contig_1d(inp.pevapsnw)
        else:
            pevapsnw = np.zeros(ngrid, dtype=np.float64)

        # Lake tile skin -- only meaningful when n_tile > TILE_LAKE.
        if inp.phlice is not None and inp.ptlice is not None \
                and inp.ptlwml is not None:
            phlice = _as_contig_1d(inp.phlice)
            ptlice = _as_contig_1d(inp.ptlice)
            ptlwml = _as_contig_1d(inp.ptlwml)
        else:
            phlice = np.zeros(ngrid, dtype=np.float64)
            ptlice = np.full(ngrid, 273.16, dtype=np.float64)
            ptlwml = np.full(ngrid, 273.16, dtype=np.float64)

        # 2-D inputs (per grid, per tile).  Fortran column-major.
        sskm1m = _as_fortran_2d(inp.sskm1m)
        tskm1m = _as_fortran_2d(inp.tskm1m)
        qskm1m = _as_fortran_2d(inp.qskm1m)
        dqsdT = _as_fortran_2d(inp.dqsdT)
        alphal = _as_fortran_2d(inp.alphal)
        alphas = _as_fortran_2d(inp.alphas)
        pssrflti = _as_fortran_2d(inp.pssrflti)
        plamsk = _as_fortran_2d(inp.plamsk)
        pfrti = _as_fortran_2d(inp.pfrti)
        pcfhti = _as_fortran_2d(inp.pcfhti)
        pcfqti = _as_fortran_2d(inp.pcfqti)

        rvtrsr = _as_contig_1d(veg.rvtrsr)

        # Scalars.
        ptmst = float(inp.ptmst.item()) if torch.is_tensor(inp.ptmst) \
            else float(inp.ptmst)
        rcpd = float(consts.cpd.item())
        rlstt = float(consts.rlstt.item())
        rlvtt = float(consts.rlvtt.item())
        rsigma = float(consts.rsigma.item())
        rtt = float(consts.rtt.item())
        rvtmp2 = float(consts.rvtmp2.item())
        rg = float(consts.g.item())

        # Output buffers.
        pjq = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        pssh = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        pslh = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        pstr = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        pg0 = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        ptsk = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        ptsktip1 = np.zeros((ngrid, ntile), dtype=np.float64, order="F")
        psl = np.zeros(ngrid, dtype=np.float64)
        pql = np.zeros(ngrid, dtype=np.float64)

        self._loaded.vdfdifh(
            _I32(ngrid), _I32(ntile),
            _ptr_f64(pslm1_lev), _ptr_f64(pqtm1_lev),
            _ptr_f64(pslge_lev), _ptr_f64(pqte_lev),
            _ptr_f64(pcfhti), _ptr_f64(pcfqti),
            _ptr_f64(psst), _ptr_f64(ptice),
            _ptr_f64(ptsm1m), _ptr_f64(ptsnow),
            _ptr_f64(sskm1m), _ptr_f64(tskm1m),
            _ptr_f64(qskm1m), _ptr_f64(dqsdT),
            _ptr_f64(alphal), _ptr_f64(alphas),
            _ptr_f64(pssrflti), _ptr_f64(pslrfl),
            _ptr_f64(ptskrad), _ptr_f64(pemis),
            _ptr_f64(plamsk), _ptr_f64(pfrti),
            _ptr_i32(ktvl), _ptr_i32(ktvh),
            _F64(ptmst),
            _ptr_f64(pevapsnw),
            _ptr_f64(phlice), _ptr_f64(ptlice), _ptr_f64(ptlwml),
            _ptr_f64(rvtrsr),
            _F64(rcpd), _F64(rlstt), _F64(rlvtt), _F64(rsigma), _F64(rtt),
            _F64(rvtmp2), _F64(rg), _F64(RVDIFTS),
            _BOOL(bool(inp.lesnml)), _BOOL(bool(has_evapsnw)),
            _BOOL(bool(inp.lelwdd)), _BOOL(bool(inp.lelwtl)),
            _ptr_f64(pjq), _ptr_f64(pssh), _ptr_f64(pslh),
            _ptr_f64(pstr), _ptr_f64(pg0),
            _ptr_f64(ptsk), _ptr_f64(ptsktip1),
            _ptr_f64(psl), _ptr_f64(pql),
        )

        out_dtype = inp.sskm1m.dtype
        out_device = inp.sskm1m.device

        def _t2(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        def _t1(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        return DiffusionOutputs(
            pjq=_t2(pjq), pssh=_t2(pssh), pslh=_t2(pslh),
            pstr=_t2(pstr), pg0=_t2(pg0),
            ptsk=_t2(ptsk), ptsktip1=_t2(ptsktip1),
            psl=_t1(psl), pql=_t1(pql),
        )

    # ------------------------------------------------------------------ #
    # SOILT (soil-temperature update: SRFRCGS + SRFTS + SRFWDIFS)
    # ------------------------------------------------------------------ #
    def soilt(
        self,
        inp: "SoilTempInputs",
        soil_table: "SoilTable",
        consts: ConstantBundle,
        config: "SurfConfig",
    ) -> "SoilTempOutputs":
        """Run the soil-temperature solve in Fortran.

        Independent reference for
        :func:`surf_pytorch.physics.soil.compute_soil_temperature`:
        the Fortran kernel re-derives SRFRCGS (apparent heat capacity
        with the freeze/melt term), SRFTS (Peters-Liddard / Kersten
        thermal conductivity + net surface heat flux + Thomas
        coefficient assembly) and SRFWDIFS (the Thomas solve) so a
        comparison test against the port validates the full temperature
        update including the apparent-heat-capacity coupling.

        Args mirror :func:`surf_pytorch.physics.soil.compute_soil_temperature`
        and consume the same :class:`SoilTempInputs` dataclass so the
        test can feed identical tensors to both.  Returns a
        :class:`SoilTempOutputs` with the same field layout.
        """
        if self._loaded.soilt is None:
            raise RuntimeError(
                "SURF_REF_SOILT not exported by this .so -- rebuild with "
                "offline_driver/build_ref.sh after adding the soilt entry."
            )
        from surf_pytorch.physics.soil import SoilTempOutputs

        ngrid, nlevs = inp.ptsam1m.shape
        ntile = inp.pfrti.shape[1]

        # 2-D inputs (per grid, per soil layer / per tile).  Fortran
        # column-major so the buffer is indexed X(JL, JK).
        ptsam1m = _as_fortran_2d(inp.ptsam1m)
        pwsam1m = _as_fortran_2d(inp.pwsam1m)
        pfrti = _as_fortran_2d(inp.pfrti)
        pahfsti = _as_fortran_2d(inp.pahfsti)
        pevapti = _as_fortran_2d(inp.pevapti)
        pssrflti = _as_fortran_2d(inp.pssrflti)

        # 1-D inputs.
        pslrfl = _as_contig_1d(inp.pslrfl)
        pgsn = _as_contig_1d(inp.pgsn)
        pcvl = _as_contig_1d(inp.pcvl)
        pcvh = _as_contig_1d(inp.pcvh)
        ksoty = _as_int_1d(inp.ksoty)
        # LDLAND: pass as a bool array (Fortran LOGICAL(C_BOOL)).
        ldland_np = inp.ldland.detach().to("cpu").numpy().astype(np.bool_)
        ldland = np.ascontiguousarray(ldland_np, dtype=np.bool_)

        # Soil-layer thicknesses RDAT (KLEVS).  Built by the same helper
        # the port uses so the values are bit-identical.
        from surf_pytorch.physics.soil import _soil_thickness
        rdat = _as_contig_1d(_soil_thickness(
            config, torch.float64, torch.device("cpu")
        ))

        # Per-soil-type LUT columns.  We pass the *full* 8-element buffer
        # (water sentinel + 7 soil types) so indexing with ksoty=0..7 in
        # the Fortran kernel matches the Python SoilTable.gather() path
        # exactly.  NSOTY_C below is therefore 8.
        wsat = np.ascontiguousarray(
            soil_table.wsat.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        lambdadry = np.ascontiguousarray(
            soil_table.lambdadry.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        lamsat1 = np.ascontiguousarray(
            soil_table.lamsat1.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        rcsoil = np.ascontiguousarray(
            soil_table.rcsoil.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        wcap = np.ascontiguousarray(
            soil_table.wcap.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        nsoty_c = wsat.shape[0]

        # Scalars.
        ptmst = float(inp.ptmst.item()) if torch.is_tensor(inp.ptmst) \
            else float(inp.ptmst)
        rsimp = float(soil_table.consts.rsimp)
        rlmlt = float(consts.rlmlt.item())
        rho_h2o = float(soil_table.consts.rho_h2o)
        rlvtt = float(consts.rlvtt.item())
        rtf1 = float(consts.rtf1.item())
        rtf2 = float(consts.rtf2.item())
        rtf3 = float(consts.rtf3.item())
        rtf4 = float(consts.rtf4.item())
        rlambdace = float(soil_table.consts.rlambdace)   # RLAMBDAICE
        rlambdawat = float(soil_table.consts.rlambdawat)
        rkerst1 = float(soil_table.consts.rkerst1)
        rkerst2 = float(soil_table.consts.rkerst2)
        rkerst3 = float(soil_table.consts.rkerst3)

        # Output buffers.
        ptsa = np.zeros((ngrid, nlevs), dtype=np.float64, order="F")
        pctsa = np.zeros((ngrid, nlevs), dtype=np.float64, order="F")
        pzsurfl = np.zeros(ngrid, dtype=np.float64)

        self._loaded.soilt(
            _I32(ngrid), _I32(nlevs), _I32(ntile), _I32(nsoty_c),
            _ptr_f64(ptsam1m), _ptr_f64(pwsam1m),
            _ptr_f64(pfrti), _ptr_f64(pahfsti),
            _ptr_f64(pevapti), _ptr_f64(pssrflti),
            _ptr_f64(pslrfl), _ptr_f64(pgsn),
            _ptr_i32(ksoty),
            _ptr_f64(pcvl), _ptr_f64(pcvh),
            ldland.ctypes.data_as(_P_BOOL),
            _ptr_f64(rdat),
            _ptr_f64(wsat), _ptr_f64(lambdadry), _ptr_f64(lamsat1),
            _ptr_f64(rcsoil), _ptr_f64(wcap),
            _F64(ptmst), _F64(rsimp), _F64(rlmlt), _F64(rho_h2o), _F64(rlvtt),
            _F64(rtf1), _F64(rtf2), _F64(rtf3), _F64(rtf4),
            _F64(rlambdace), _F64(rlambdawat),
            _F64(rkerst1), _F64(rkerst2), _F64(rkerst3),
            _ptr_f64(ptsa), _ptr_f64(pctsa), _ptr_f64(pzsurfl),
        )

        out_dtype = inp.ptsam1m.dtype
        out_device = inp.ptsam1m.device

        def _t2(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        def _t1(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        return SoilTempOutputs(
            ptsa=_t2(ptsa), pctsa=_t2(pctsa), zsurfl=_t1(pzsurfl),
        )

    # ------------------------------------------------------------------ #
    # SNOW (single-layer snow: SRFSN_LWIMPS)
    # ------------------------------------------------------------------ #
    def snow(
        self,
        inp: "SnowInputs",
        soil_table: "SoilTable",
        consts: ConstantBundle,
        config: "SurfConfig",
    ) -> "SnowOutputs":
        """Run the single-layer snow solve in Fortran.

        Independent reference for
        :func:`surf_pytorch.physics.snow.compute_snow`: the Fortran kernel
        re-derives SRFSN_LWIMPS (NCSNEC=1) -- the no-snow / no-melt /
        melt branches, the snow-soil resistance, the implicit snow
        temperature solve, the snowfall-interception throughfall, the
        liquid-water capacity and the snow-mass update -- so a
        comparison test against the port validates the full single-layer
        snow update.

        Args mirror :func:`surf_pytorch.physics.snow.compute_snow` and
        consume the same :class:`SnowInputs` dataclass so the test can
        feed identical tensors to both.  Returns a :class:`SnowOutputs`
        with the same field layout.
        """
        if self._loaded.snow is None:
            raise RuntimeError(
                "SURF_REF_SNOW not exported by this .so -- rebuild with "
                "offline_driver/build_ref.sh after adding the snow entry."
            )
        from surf_pytorch.physics.snow import SnowOutputs

        ngrid = inp.pssnm1m.shape[0]
        ntile = inp.pfrti.shape[1]

        # 1-D snow-state inputs (KLON).
        pssnm1m = _as_contig_1d(inp.pssnm1m)
        ptsnm1m = _as_contig_1d(inp.ptsnm1m)
        prsnm1m = _as_contig_1d(inp.prsnm1m)
        ptsam1m_top = _as_contig_1d(inp.ptsam1m_top)

        # 2-D per-tile fluxes (KLON, NTILES), Fortran column-major so the
        # buffer is indexed X(JL, JT).
        pfrti = _as_fortran_2d(inp.pfrti)
        pahfsti = _as_fortran_2d(inp.pahfsti)
        pevapti = _as_fortran_2d(inp.pevapti)
        pssrflti = _as_fortran_2d(inp.pssrflti)

        # 1-D radiation + snowfall/rainfall throughfall + evap-mismatch.
        pslrfl = _as_contig_1d(inp.pslrfl)
        pssfc = _as_contig_1d(inp.pssfc)
        pssfl = _as_contig_1d(inp.pssfl)
        ptsfc = _as_contig_1d(inp.ptsfc)
        ptsfl = _as_contig_1d(inp.ptsfl)
        pevapsnw = _as_contig_1d(inp.pevapsnw)

        # Scalars.
        ptmst = float(inp.ptmst.item()) if torch.is_tensor(inp.ptmst) \
            else float(inp.ptmst)
        leflake_c = bool(config.use_flake)
        rtt = float(consts.rtt.item())
        rlmlt = float(consts.rlmlt.item())
        rlvtt = float(consts.rlvtt.item())
        rlstt = float(consts.rlstt.item())
        rpi = float(consts.pi.item())
        rday = float(consts.rday.item())
        rhoice = float(soil_table.consts.rhoice)
        rlamice = float(soil_table.consts.rlamice)
        rhoci = float(soil_table.consts.rhoci)
        ralamsn = float(soil_table.consts.rlamdsn_pow)
        rdsmax = float(soil_table.consts.rdsmax)
        rfrsmall = float(soil_table.consts.rfrsmall)
        rfrtiny = float(soil_table.consts.rfrtiny)
        rlwcswea = float(soil_table.consts.rlwcswea)
        rlwcsweb = float(soil_table.consts.rlwcsweb)
        rlwcswec = float(soil_table.consts.rlwcswec)
        rtempamp = float(soil_table.consts.rtempamp)
        rtauf = float(soil_table.consts.rtauf)
        rtaua = float(soil_table.consts.rtaua)
        # RVLAMSK(8): desert bare-skin conductivity 15 W/m^2/K used by
        # ZSOILRES = 1 / RVLAMSK(8); see susveg_mod.F90.
        rvlamsk_8 = 15.0

        # Output buffers.
        ptsn = np.zeros(ngrid, dtype=np.float64)
        pgsn = np.zeros(ngrid, dtype=np.float64)
        psstar = np.zeros(ngrid, dtype=np.float64)
        plwc = np.zeros(ngrid, dtype=np.float64)
        pzfrsn = np.zeros(ngrid, dtype=np.float64)

        self._loaded.snow(
            _I32(ngrid), _I32(ntile),
            _ptr_f64(pssnm1m), _ptr_f64(ptsnm1m),
            _ptr_f64(prsnm1m), _ptr_f64(ptsam1m_top),
            _ptr_f64(pfrti), _ptr_f64(pahfsti),
            _ptr_f64(pevapti), _ptr_f64(pssrflti),
            _ptr_f64(pslrfl),
            _ptr_f64(pssfc), _ptr_f64(pssfl),
            _ptr_f64(ptsfc), _ptr_f64(ptsfl),
            _ptr_f64(pevapsnw),
            _BOOL(leflake_c),
            _F64(ptmst),
            _F64(rtt), _F64(rlmlt), _F64(rlvtt), _F64(rlstt),
            _F64(rpi), _F64(rday),
            _F64(rhoice), _F64(rlamice), _F64(rhoci), _F64(ralamsn),
            _F64(rdsmax), _F64(rfrsmall), _F64(rfrtiny),
            _F64(rlwcswea), _F64(rlwcsweb), _F64(rlwcswec), _F64(rtempamp),
            _F64(rtauf), _F64(rtaua),
            _F64(rvlamsk_8),
            _ptr_f64(ptsn), _ptr_f64(pgsn), _ptr_f64(psstar),
            _ptr_f64(plwc), _ptr_f64(pzfrsn),
        )

        out_dtype = inp.pssnm1m.dtype
        out_device = inp.pssnm1m.device

        def _t1(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        return SnowOutputs(
            ptsn=_t1(ptsn), pgsn=_t1(pgsn), psstar=_t1(psstar),
            plwc=_t1(plwc), zfrsn=_t1(pzfrsn),
        )

    # ------------------------------------------------------------------ #
    # SOILW (soil-moisture update: SRFWEXC_VG + SRFWDIF + SRFWINC + SRFWNG)
    # ------------------------------------------------------------------ #
    def soilw(
        self,
        inp: "SoilMoistInputs",
        soil_table: "SoilTable",
        veg_table,
        consts: ConstantBundle,
        config: "SurfConfig",
    ) -> "SoilMoistOutputs":
        """Run the HTESSEL soil-moisture solve in Fortran.

        Independent reference for
        :func:`surf_pytorch.physics.soil.compute_soil_moisture`: the
        Fortran kernel re-derives SRFWEXC_VG (Van Genuchten hydraulic
        K/D, VIC saturation-excess or infiltration-limit runoff, root
        extraction, tridiagonal coefficient assembly), SRFWDIF (Thomas
        solve), SRFWINC (t+1 unfiltered moisture + drainage diagnostic)
        and SRFWNG (clip / redistribute / runoff split) -- so a
        comparison test against the port validates the full moisture
        update including the runoff split and the gravity-water flux.

        Args mirror :func:`surf_pytorch.physics.soil.compute_soil_moisture`
        and consume the same :class:`SoilMoistInputs` dataclass so the
        test can feed identical tensors to both.  Returns a
        :class:`SoilMoistOutputs` with the same field layout.
        """
        if self._loaded.soilw is None:
            raise RuntimeError(
                "SURF_REF_SOILW not exported by this .so -- rebuild with "
                "offline_driver/build_ref.sh after adding the soilw entry."
            )
        from surf_pytorch.physics.soil import SoilMoistOutputs

        ngrid, nlevs = inp.pwsam1m.shape
        ntile = inp.pfrti.shape[1]

        # 2-D inputs (per grid, per soil layer / per tile).  Fortran
        # column-major so the buffer is indexed X(JL, JK).
        pwsam1m = _as_fortran_2d(inp.pwsam1m)
        ptsam1m = _as_fortran_2d(inp.ptsam1m)
        pfrti = _as_fortran_2d(inp.pfrti)
        pevapti = _as_fortran_2d(inp.pevapti)
        peintti = _as_fortran_2d(inp.peintti)

        # 1-D inputs.
        ptsfc = _as_contig_1d(inp.ptsfc)
        ptsfl = _as_contig_1d(inp.ptsfl)
        pmsn = _as_contig_1d(inp.pmsn)
        pemssn = _as_contig_1d(inp.pemssn)
        pevapsnw = _as_contig_1d(inp.pevapsnw)
        psdor = _as_contig_1d(inp.psdor)
        ksoty = _as_int_1d(inp.ksoty)
        ktvl = _as_int_1d(inp.ktvl)
        ktvh = _as_int_1d(inp.ktvh)
        # LDLAND: pass as a bool array (Fortran LOGICAL(C_BOOL)).
        ldland_np = inp.ldland.detach().to("cpu").numpy().astype(np.bool_)
        ldland = np.ascontiguousarray(ldland_np, dtype=np.bool_)

        # Soil-layer thicknesses RDAT (KLEVS).  Built by the same helper
        # the port uses so the values are bit-identical.
        from surf_pytorch.physics.soil import _soil_thickness
        rdat = _as_contig_1d(_soil_thickness(
            config, torch.float64, torch.device("cpu")
        ))

        # RVROOTSA (KLEVS, NV).  Stored on VegTable as (n_soil, 21); pass
        # through as Fortran-column-major so the kernel indexes
        # RVROOTSA(JK, ITYP).
        rvrootsa = np.asfortranarray(
            veg_table.rvrootsa.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        nv_c = rvrootsa.shape[1]

        # Per-soil-type LUT columns.  We pass the *full* 8-element buffer
        # (water sentinel + 7 soil types) so indexing with ksoty=0..7 in
        # the Fortran kernel matches the Python SoilTable.gather() path
        # exactly.  NSOTY_C below is therefore 8.
        wsat = np.ascontiguousarray(
            soil_table.wsat.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        wrest = np.ascontiguousarray(
            soil_table.wrest.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        wcap = np.ascontiguousarray(
            soil_table.wcap.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        wpwp = np.ascontiguousarray(
            soil_table.wpwp.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        wconsm = np.ascontiguousarray(
            soil_table.wconsm.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        mvgalpha = np.ascontiguousarray(
            soil_table.mvgalpha.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        mfac = np.ascontiguousarray(
            soil_table.mfac.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        lam = np.ascontiguousarray(
            getattr(soil_table, "lambda")
            .detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        dmax = np.ascontiguousarray(
            soil_table.dmax.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        dmin = np.ascontiguousarray(
            soil_table.dmin.detach().to("cpu", torch.float64).numpy(),
            dtype=np.float64,
        )
        nsoty_c = wsat.shape[0]

        # Scalars.  RSIGORMAX/MIN and RPSFR are sub-grid orography bounds
        # for the VIC runoff branch; taken from SoilConstants (defaults
        # 100/1000 -- sussoil_mod.F90:480-481).
        rsigormax = float(soil_table.consts.rsigormax)
        rsigormin = float(soil_table.consts.rsigormin)
        rpsfr = float(getattr(soil_table.consts, "rpsfr", 1.0))
        ptmst = float(inp.ptmst.item()) if torch.is_tensor(inp.ptmst) \
            else float(inp.ptmst)
        rsimp = float(soil_table.consts.rsimp)
        rsrdep = float(soil_table.consts.rsrdep)
        rho_h2o = float(soil_table.consts.rho_h2o)
        rtf1 = float(consts.rtf1.item())
        rtf2 = float(consts.rtf2.item())
        rtf3 = float(consts.rtf3.item())
        rtf4 = float(consts.rtf4.item())
        lessro = bool(getattr(config, "lessro", True))

        # Output buffers.
        pwsa = np.zeros((ngrid, nlevs), dtype=np.float64, order="F")
        pros = np.zeros(ngrid, dtype=np.float64)
        prod = np.zeros(ngrid, dtype=np.float64)
        pwfsd = np.zeros(ngrid, dtype=np.float64)
        psawgfl = np.zeros((ngrid, nlevs), dtype=np.float64, order="F")

        self._loaded.soilw(
            _I32(ngrid), _I32(nlevs), _I32(ntile), _I32(nsoty_c), _I32(nv_c),
            _ptr_f64(pwsam1m), _ptr_f64(ptsam1m),
            _ptr_f64(pfrti), _ptr_f64(pevapti), _ptr_f64(peintti),
            _ptr_f64(ptsfc), _ptr_f64(ptsfl),
            _ptr_f64(pmsn), _ptr_f64(pemssn), _ptr_f64(pevapsnw),
            _ptr_f64(psdor),
            _ptr_i32(ksoty), _ptr_i32(ktvl), _ptr_i32(ktvh),
            ldland.ctypes.data_as(_P_BOOL),
            _ptr_f64(rdat), _ptr_f64(rvrootsa),
            _ptr_f64(wsat), _ptr_f64(wrest), _ptr_f64(wcap),
            _ptr_f64(wpwp), _ptr_f64(wconsm),
            _ptr_f64(mvgalpha), _ptr_f64(mfac), _ptr_f64(lam),
            _ptr_f64(dmax), _ptr_f64(dmin),
            _F64(rsigormax), _F64(rsigormin), _F64(rpsfr),
            _F64(ptmst), _F64(rsimp), _F64(rsrdep), _F64(rho_h2o),
            _F64(rtf1), _F64(rtf2), _F64(rtf3), _F64(rtf4),
            _BOOL(lessro),
            _ptr_f64(pwsa), _ptr_f64(pros), _ptr_f64(prod),
            _ptr_f64(pwfsd), _ptr_f64(psawgfl),
        )

        out_dtype = inp.pwsam1m.dtype
        out_device = inp.pwsam1m.device

        def _t2(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        def _t1(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        return SoilMoistOutputs(
            pwsa=_t2(pwsa), pros=_t1(pros), prod=_t1(prod),
            pwfsd=_t1(pwfsd), psawgfl=_t2(psawgfl),
        )

    # ------------------------------------------------------------------ #
    # FARQUHAR (coupled photosynthesis + Ball-Berry conductance)
    # ------------------------------------------------------------------ #
    def farquhar(
        self,
        sw_down: torch.Tensor,
        p_surf: torch.Tensor,
        qs_skin: torch.Tensor,
        qsurf: torch.Tensor,
        q_air: torch.Tensor,
        t_air: torch.Tensor,
        skin_t: torch.Tensor,
        t_soil: torch.Tensor,
        f2: torch.Tensor,
        co2: torch.Tensor,
        lai: torch.Tensor,
        kvtype: torch.Tensor,
        kco2typ: torch.Tensor,
        ldland: torch.Tensor,
        veg: VegTable,
        consts: ConstantBundle,
        *,
        lskin_temp: bool = True,
    ) -> dict:
        """Run FARQUHAR (coupled photosynthesis + conductance) in Fortran.

        Independent reference for :func:`surf_pytorch.physics.farquhar.
        farquhar`: the Fortran kernel re-derives the full A-gs solve --
        the per-PFT table lookup, the Arrhenius (modified) scalings, the
        Kattge&Knorr temperature acclimation, the C3/C4 cubic stomatal
        closure (Yin 2009), the 20-LAI-level Beer-Lambert canopy
        integration and the Ball-Berry conductance -- so a comparison
        test against the port is a genuine check of the *photosynthesis
        formulation* (not a self-transcription).

        Args mirror the per-column inputs the port's ``farquhar()`` will
        consume: all ``(Ngrid,)`` tensors.  The per-PFT tables and the
        class-agnostic scalars come from :class:`VegTable` and its
        ``farquhar_consts`` (sufarquhar_mod.F90 defaults), so the two
        implementations share an identical parameterisation.

        Returns a dict with keys ``pgstot`` (canopy conductance, m/s),
        ``pgpp`` (GPP, kgC/m^2/s), ``prd`` (dark respiration) and ``pan``
        (net assimilation).
        """
        if self._loaded.farquhar is None:
            raise RuntimeError(
                "SURF_REF_FARQUHAR not exported by this .so -- rebuild with "
                "offline_driver/build_ref.sh after adding the farquhar entry."
            )
        from surf_pytorch.tables.veg import FarquharConstants

        ngrid = sw_down.shape[0]
        out_dtype = sw_down.dtype
        out_device = sw_down.device

        # 1-D inputs (KLON).
        psrfd = _as_contig_1d(sw_down)
        paphm = _as_contig_1d(p_surf)
        pqs = _as_contig_1d(qs_skin)
        pqsurf = _as_contig_1d(qsurf)
        pqm = _as_contig_1d(q_air)
        ptm = _as_contig_1d(t_air)
        ptskm = _as_contig_1d(skin_t)
        ptsoil = _as_contig_1d(t_soil)
        pf2 = _as_contig_1d(f2)
        pco2 = _as_contig_1d(co2)
        plai = _as_contig_1d(lai)
        kvtype_a = _as_int_1d(kvtype)
        kco2typ_a = _as_int_1d(kco2typ)
        ldland_np = ldland.detach().to("cpu").numpy().astype(np.bool_)
        ldland = np.ascontiguousarray(ldland_np, dtype=np.bool_)

        # Per-PFT tables (0:20, 2): col 0 = C3, col 1 = C4.  The VegTable
        # stores these as (21, 2) row-major; hand the kernel a Fortran
        # column-major view so X(ITYPE, ICTYPE) matches Fortran storage.
        rvmax25 = _as_fortran_2d(veg.rvmax25)
        rhumrel = _as_fortran_2d(veg.rhumrel)
        ra1 = _as_fortran_2d(veg.ra1)
        rb1 = _as_fortran_2d(veg.rb1)
        rg0 = _as_fortran_2d(veg.rg0)
        rgm25 = _as_fortran_2d(veg.rgm25)
        re_vcmax = _as_fortran_2d(veg.re_vcmax)
        re_jmax = _as_fortran_2d(veg.re_jmax)

        # FarquharConstants -- pull every scalar from the VegTable's
        # frozen FarquharConstants so ref and port stay bit-identical.
        fc = veg.farquhar_consts
        if not isinstance(fc, FarquharConstants):
            fc = FarquharConstants()

        # Path-vector scalars (C3, C4) -- flattened to pairs.
        rarjv1, rarjv2 = fc.rarJV
        rbrjv1, rbrjv2 = fc.rbrJV
        rasj1, rasj2 = fc.rasJ
        rbsj1, rbsj2 = fc.rbsJ
        rasv1, rasv2 = fc.rasV
        rbsv1, rbsv2 = fc.rbsV
        rd_vcmax1, rd_vcmax2 = fc.rd_vcmax
        rd_jmax1, rd_jmax2 = fc.rd_jmax
        re_gm1, re_gm2 = fc.re_gm
        rs_gm1, rs_gm2 = fc.rs_gm
        rs_jmax1, rs_jmax2 = fc.rs_jmax
        rd_gm1, rd_gm2 = fc.rd_gm
        rkmc25_1, rkmc25_2 = fc.rkmc25
        rkmo25_1, rkmo25_2 = fc.rkmo25
        rsco25_1, rsco25_2 = fc.rsco25

        # Gas constants.  R = RNAVO*RKBOL (sucst.F90:148); pulled from the
        # same ConstantBundle source so the ref and port agree exactly.
        from surf_pytorch.constants import R as RGAS
        rr_c = float(RGAS)
        rpi_c = float(consts.pi.item())
        rmair_c = float(fc.rmair)
        rmco2_c = float(fc.rmco2)

        # Output buffers.
        pgstot = np.zeros(ngrid, dtype=np.float64)
        pgpp = np.zeros(ngrid, dtype=np.float64)
        prd = np.zeros(ngrid, dtype=np.float64)
        pan = np.zeros(ngrid, dtype=np.float64)

        self._loaded.farquhar(
            _I32(ngrid), _I32(int(fc.nlai)),
            ldland.ctypes.data_as(_P_BOOL),
            _ptr_i32(kvtype_a), _ptr_i32(kco2typ_a),
            _ptr_f64(psrfd), _ptr_f64(paphm),
            _ptr_f64(pqs), _ptr_f64(pqsurf), _ptr_f64(pqm),
            _ptr_f64(ptm), _ptr_f64(ptskm), _ptr_f64(ptsoil),
            _ptr_f64(pf2), _ptr_f64(pco2), _ptr_f64(plai),
            _ptr_f64(rvmax25), _ptr_f64(rhumrel),
            _ptr_f64(ra1), _ptr_f64(rb1), _ptr_f64(rg0),
            _ptr_f64(rgm25), _ptr_f64(re_vcmax), _ptr_f64(re_jmax),
            # Flattened path-pair scalars.
            _F64(rarjv1), _F64(rarjv2), _F64(rbrjv1), _F64(rbrjv2),
            _F64(rasj1), _F64(rasj2), _F64(rbsj1), _F64(rbsj2),
            _F64(rasv1), _F64(rasv2), _F64(rbsv1), _F64(rbsv2),
            _F64(rd_vcmax1), _F64(rd_vcmax2), _F64(rd_jmax1), _F64(rd_jmax2),
            _F64(fc.re_gamma_star),
            _F64(re_gm1), _F64(re_gm2), _F64(rs_gm1), _F64(rs_gm2),
            _F64(rs_jmax1), _F64(rs_jmax2), _F64(rd_gm1), _F64(rd_gm2),
            _F64(fc.re_kmc), _F64(fc.re_kmo), _F64(fc.re_rd), _F64(fc.re_sco),
            _F64(fc.rgamma_star25),
            _F64(rkmc25_1), _F64(rkmc25_2), _F64(rkmo25_1), _F64(rkmo25_2),
            _F64(rsco25_1), _F64(rsco25_2),
            _F64(fc.rstress_gm), _F64(fc.rstress_gs), _F64(fc.rstress_vcmax),
            _F64(fc.rlai_level_depth), _F64(fc.rlaimax), _F64(fc.rext_coeff),
            _F64(fc.rgb_ref), _F64(fc.rtp_00), _F64(fc.rpb_std), _F64(fc.rrg_to_par),
            _F64(fc.rw_to_mol), _F64(fc.ralpha_ll), _F64(fc.rtheta),
            _F64(fc.rfpsir), _F64(fc.rfq), _F64(fc.rfpseudo), _F64(fc.rh_protons),
            _F64(fc.rgbs), _F64(fc.roi), _F64(fc.ralpha), _F64(fc.rkp),
            _F64(fc.rtetens_1), _F64(fc.rtetens_2), _F64(fc.rref_temp),
            _F64(fc.rmol_to_m_1), _F64(fc.rratio_h2o_to_co2),
            _F64(fc.rn_vert_att),
            _F64(rpi_c), _F64(rr_c),
            _F64(rmair_c), _F64(rmco2_c),
            _BOOL(bool(lskin_temp)),
            _ptr_f64(pgstot), _ptr_f64(pgpp), _ptr_f64(prd), _ptr_f64(pan),
        )

        def _t1(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(out_dtype).to(out_device)

        return {
            "pgstot": _t1(pgstot), "pgpp": _t1(pgpp),
            "prd": _t1(prd), "pan": _t1(pan),
        }


__all__ = ["SurfRef"]
