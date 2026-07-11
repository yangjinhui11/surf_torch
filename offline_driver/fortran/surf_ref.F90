! surf_ref.F90 -- Minimal standalone Fortran reference for the OpenIFS
!                 land-surface SEB + skin-conductivity kernels.
!
! Purpose
! -------
! This module provides a *self-contained* Fortran reimplementation of the
! two OpenIFS 48r1 surface kernels that the PyTorch port (surf_pytorch)
! currently implements:
!
!   * VLAMSK       (surf/module/vlamsk_mod.F90)        -- tiled skin
!                                                         conductivity
!   * SURFSEB_CTL  (surf/module/surfseb_ctl_mod.F90)   -- Penman-Monteith
!                                                         surface energy
!                                                         balance
!
! It is deliberately *standalone*: no PARKIND1, no YOMHOOK, no derived
! types, no MPI, no NetCDF.  Every constant is passed in by the caller
! (the Python ctypes wrapper builds them from the offline-driver SUPHEC /
! SUCST values, exactly as surf_pytorch.constants does).  This makes the
! library trivial to compile with a bare gfortran and to call from
! Python via ctypes / numpy.
!
! Faithfulness
! ------------
! The arithmetic below is transcribed line-for-line from the upstream
! Fortran.  The two intentional simplifications, both matching the
! surf_pytorch defaults, are:
!
!   1. Classic VLAMSK path only -- the optional cy46+ refinements
!      (LESKTI5 / LESKTI8 snow/soil thermal-conductivity lookups) are
!      omitted; the offline driver leaves both flags .FALSE. by default.
!   2. Sea-ice coupling LSICOUP is exposed as an argument (default
!      .FALSE.) but, when .FALSE., follows the same RVLAMSK(12) lookup
!      the upstream code uses.
!
! Entry points
! ------------
! The library mirrors the *port's* structure: three BIND(C) subroutines,
! one per kernel, so the comparison tests can feed identical inputs to
! both implementations:
!
!   * SURF_REF_VLAMSK -- per-tile skin conductivity PLAMSK.
!     Mirrors surf_pytorch.physics.vlamsk.vlamsk().
!   * SURF_REF_SEB    -- Penman-Monteith SEB; takes PLAMSK as input.
!     Mirrors surf_pytorch.physics.seb.surf_seb().
!   * SURF_REF_VEXCS  -- Beljaars Monin-Obukhov exchange coefficients.
!     Mirrors surf_pytorch.physics.exchange.vexcs().
!   * SURF_REF_DQSDT  -- skin-T saturation Jacobian PDQS, transcribed
!     from vsurf_mod.F90 (278-285) using the fcttre.h statement
!     functions FOEEW / FOEDESU / FOEDELTA.  Mirrors
!     surf_pytorch.utils.dqsat_dt_ifs().  This entry exists so the
!     saturation-derivative can be validated *independently* of the
!     Python transcription: SURF_REF_SEB takes DQSDT as an input, so
!     without this entry the comparison tests could not distinguish a
!     wrong dqs/dT formula from a correct one.
!   * SURF_REF_STOMATA -- Jarvis three-factor stomatal resistance
!     (vsurf 296-421), outputs PWETL/PWETH/PWETB/PWETLU/PWETHS + diag.
!     Mirrors surf_pytorch.physics.stomata.jarvis_resistance().
!   * SURF_REF_VEVAP   -- equivalent evapotranspiration efficiency per
!     tile (vevap_mod.F90 138-269), outputs PCSAT/PCAIR/PCSATU/PCAIRU/
!     PCPTSU/PCSNW/PDHVEGS.  Called once per tile with a KTILE selector.
!     Mirrors surf_pytorch.physics.stomata.vevap() /
!     build_tile_vevap().  This entry is the rho-convention reference:
!     the port's old build_tile_alpha divided by rho twice; this kernel
!     proves the corrected single-rho form.
!   * SURF_REF_SOILT   -- HTESSEL soil-temperature update (SRFRCGS +
!     SRFTS + SRFWDIFS), outputs PTSA/PCTSA/ZSURFL.  Mirrors
!     surf_pytorch.physics.soil.compute_soil_temperature().
!   * SURF_REF_SNOW    -- single-layer snow scheme (SRFSN_LWIMPS):
!     implicit snow-T, snow-soil ground heat flux, diagnostic liquid
!     water and rainfall interception, with the no-snow / no-melt /
!     melt branches.  Outputs PTSN/PGSN plus diagnostics PSSTAR/PLWC/
!     ZFRSN.  Mirrors surf_pytorch.physics.snow.compute_snow().
!
! The qsat diagnostic is computed on the Python side
! (surf_pytorch.utils.qsat) and passed in as arrays, so the
! Fortran ref stays free of the saturation-curve code.  The RITBL
! stability lookup table (101 doubles from SUVEXCS) is likewise passed
! in by the caller so the .so has no module-data initialisation step.
!
! All arrays are (KLON, NTILES) with Fortran column-major storage; the
! ctypes wrapper passes them as flattened buffers reshaped with the
! leading dimension KLON.  Integer arguments are C-interoperable.
!
! Author: surf-pytorch reference driver (M2).  Algorithm (c) ECMWF.

MODULE SURF_REF
  USE ISO_C_BINDING, ONLY: C_INT, C_DOUBLE, C_BOOL
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: SURF_REF_VLAMSK, SURF_REF_SEB, SURF_REF_VEXCS, SURF_REF_DQSDT
  PUBLIC :: SURF_REF_STOMATA, SURF_REF_VEVAP, SURF_REF_VDFDIFH
  PUBLIC :: SURF_REF_SOILT, SURF_REF_SNOW, SURF_REF_SOILW
  PUBLIC :: SURF_REF_FARQUHAR

  ! JPRB alias so the transcription reads identically to the upstream.
  INTEGER, PARAMETER :: JPRB = C_DOUBLE
  ! JPRD alias -- upstream uses double for the PSI statement functions.
  INTEGER, PARAMETER :: JPRD = C_DOUBLE

  ! PSI-function constants.  Populated from the VALUE scalars on entry
  ! to SURF_REF_VEXCS so the module-level PSI helpers can read them via
  ! host association -- mirrors the fcsvdfs.h statement-function form
  ! while staying BIND(C)-compatible (statement functions cannot live
  ! in a BIND(C) procedure).
  REAL(JPRD) :: RCHBB_M, RCHBCD_M, RCHBD_M, RCHBA_M, RCHBBCD_M
  REAL(JPRD) :: RCHB23A_M, RCDHALF_M, RCDHPI2_M
  REAL(JPRD), PARAMETER :: ONE_M = 1.0_JPRD, TWO_M = 2.0_JPRD
  REAL(JPRD), PARAMETER :: HALF_M = 0.5_JPRD, EIGHTH_M = 0.125_JPRD

  ! fcttre.h statement-function host variables (set by SURF_REF_DQSDT).
  ! Mirrors YOETHF / YOMCST scalars FOEEW / FOEDESU / FOEDELTA need.
  REAL(JPRB) :: R2ES_M, R3LES_M, R3IES_M, R4LES_M, R4IES_M
  REAL(JPRB) :: R5LES_M, R5IES_M, RTT_M

  ! Jarvis stomatal-resistance host scalars (set by SURF_REF_STOMATA).
  ! RTF1..RTF4 are the soil-freeze ramp endpoints (SUSSOIL); RETV_M and
  ! RD_M mirror YOMCST entries that vsurf uses in the F3 / rho formulas.
  REAL(JPRB) :: RTF1_M, RTF2_M, RTF3_M, RTF4_M
  REAL(JPRB) :: RETV_M, RD_M

CONTAINS

  !---------------------------------------------------------------------!
  ! fcttre.h statement-function equivalents (thermodynamic saturation). !
  ! Module-level so they read line-for-line with fcttre.h; the YOETHF   !
  ! constants are set by SURF_REF_DQSDT before the body runs.           !
  ! --------------------------------------------------------------------!
  REAL(JPRB) FUNCTION FOEDELTA_M(PTARE)
    REAL(JPRB), INTENT(IN) :: PTARE
    ! fcttre.h: FOEDELTA(PTARE) = MAX(0.0,SIGN(1.0,PTARE-RTT))
    FOEDELTA_M = MAX(0.0_JPRB, SIGN(1.0_JPRB, PTARE - RTT_M))
  END FUNCTION FOEDELTA_M

  REAL(JPRB) FUNCTION FOEEW_M(PTARE)
    REAL(JPRB), INTENT(IN) :: PTARE
    REAL(JPRB) :: ZDEL
    ! fcttre.h: FOEEW(PTARE) = R2ES*EXP((R3LES*DEL+R3IES*(1-DEL))*(T-RTT)
    !                                / (T - (R4LES*DEL+R4IES*(1-DEL))))
    ZDEL = FOEDELTA_M(PTARE)
    FOEEW_M = R2ES_M * EXP((R3LES_M*ZDEL + R3IES_M*(1.0_JPRB-ZDEL)) &
        &   * (PTARE-RTT_M) / (PTARE-(R4LES_M*ZDEL+R4IES_M*(1.0_JPRB-ZDEL))))
  END FUNCTION FOEEW_M

  REAL(JPRB) FUNCTION FOEDESU_M(PTARE)
    REAL(JPRB), INTENT(IN) :: PTARE
    REAL(JPRB) :: ZDEL
    ! fcttre.h: FOEDESU(PTARE) = (DEL*R5LES+(1-DEL)*R5IES)
    !                            / (T-(R4LES*DEL+R4IES*(1-DEL)))**2
    ZDEL = FOEDELTA_M(PTARE)
    FOEDESU_M = (ZDEL*R5LES_M + (1.0_JPRB-ZDEL)*R5IES_M) &
        &   / (PTARE-(R4LES_M*ZDEL+R4IES_M*(1.0_JPRB-ZDEL)))**2
  END FUNCTION FOEDESU_M

  !---------------------------------------------------------------------!
  ! PSI statement-function equivalents (fcsvdfs.h).  Module-level so   !
  ! they read line-for-line with fcsvdfs.h; the constants are set by   !
  ! SURF_REF_VEXCS before the body runs.                                !
  !---------------------------------------------------------------------!
  REAL(JPRD) FUNCTION PSIMU(DPETA)
    REAL(JPRD), INTENT(IN) :: DPETA
    PSIMU = LOG((ONE_M + SQRT(SQRT(ONE_M - RCDHALF_M * DPETA)))**2 &
        &    * (ONE_M + SQRT(ONE_M - RCDHALF_M * DPETA)) * EIGHTH_M) &
        &    - TWO_M * ATAN(SQRT(SQRT(ONE_M - RCDHALF_M * DPETA))) &
        &    + RCDHPI2_M
  END FUNCTION PSIMU

  REAL(JPRD) FUNCTION PSIHU(DPETA)
    REAL(JPRD), INTENT(IN) :: DPETA
    PSIHU = TWO_M * LOG((ONE_M + SQRT(ONE_M - RCDHALF_M * DPETA)) * HALF_M)
  END FUNCTION PSIHU

  REAL(JPRD) FUNCTION PSIMS(DPETA)
    REAL(JPRD), INTENT(IN) :: DPETA
    PSIMS = -RCHBB_M * (DPETA - RCHBCD_M) * EXP(-RCHBD_M * DPETA) &
        &   - RCHBA_M * DPETA - RCHBBCD_M
  END FUNCTION PSIMS

  REAL(JPRD) FUNCTION PSIHS(DPETA)
    REAL(JPRD), INTENT(IN) :: DPETA
    PSIHS = -RCHBB_M * (DPETA - RCHBCD_M) * EXP(-RCHBD_M * DPETA) &
        &   - (ONE_M + RCHB23A_M * DPETA) &
        &      * SQRT(ONE_M + RCHB23A_M * DPETA) &
        &   - RCHBBCD_M + ONE_M
  END FUNCTION PSIHS

  !=====================================================================!
  ! SURF_REF_VLAMSK -- tiled skin conductivity PLAMSK (classic path).
  !
  ! Direct transcription of vlamsk_mod.F90 SELECT CASE block + the
  ! post-SELECT JT=7 stability factor.  The 1-based JT indexing matches
  ! the upstream Fortran; the surf_pytorch port uses 0-based (see
  ! surf_pytorch.config) but the arithmetic is identical.
  !=====================================================================!
  SUBROUTINE SURF_REF_VLAMSK( &
      &   KLON_C, NTILES_C, &
      &   TSKM1M, TSRF, SNM, RSN, KTVL, KTVH, &
      &   RVLAMSK, RVLAMSKS, &
      &   RHOICE, RHOCI, RQSNCR, RURBTC, PTMST, LSICOUP, &
      &   PLAMSK &
      & ) BIND(C, NAME="surf_ref_vlamsk")

    !--- dimensions ------------------------------------------------------
    INTEGER(C_INT), VALUE, INTENT(IN) :: KLON_C, NTILES_C
    !--- inputs ----------------------------------------------------------
    REAL(JPRB), INTENT(IN)  :: TSKM1M(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: TSRF  (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: SNM   (KLON_C)
    REAL(JPRB), INTENT(IN)  :: RSN   (KLON_C)
    INTEGER(C_INT), INTENT(IN) :: KTVL(KLON_C)
    INTEGER(C_INT), INTENT(IN) :: KTVH(KLON_C)
    ! veg LUTs dimensioned 0:20; entry 0 unused (port-side sentinel
    ! mirrors type 8 but VLAMSK never looks up index 0).
    REAL(JPRB), INTENT(IN)  :: RVLAMSK (0:20)
    REAL(JPRB), INTENT(IN)  :: RVLAMSKS(0:20)
    REAL(JPRB), VALUE, INTENT(IN) :: RHOICE, RHOCI, RQSNCR, RURBTC, PTMST
    LOGICAL(C_BOOL), VALUE, INTENT(IN) :: LSICOUP
    !--- output ----------------------------------------------------------
    REAL(JPRB), INTENT(OUT) :: PLAMSK(KLON_C, NTILES_C)

    !--- locals (vlamsk_mod.F90 lines 114-121) ---------------------------
    REAL(JPRB), PARAMETER :: ZLARGE = 1.0E10_JPRB
    REAL(JPRB), PARAMETER :: ZLARGESN = 50.0_JPRB
    REAL(JPRB), PARAMETER :: ZLARGEWT = 20.0_JPRB
    REAL(JPRB), PARAMETER :: ZRTTMEPS_DELTA = 0.2_JPRB   ! RTT - ZRTTMEPS
    REAL(JPRB), PARAMETER :: ZSNOW = 7.0_JPRB
    REAL(JPRB), PARAMETER :: ZSNOW_GLACIER = 8.0_JPRB
    REAL(JPRB), PARAMETER :: ZSNOWHVEG = 20.0_JPRB
    REAL(JPRB) :: ZRTTMEPS, ZSTABEXSN, ZDEPTH
    INTEGER :: KLON, NTILES, JL, JT
    LOGICAL :: LSICOUP_F

    KLON   = KLON_C
    NTILES = NTILES_C
    LSICOUP_F = LSICOUP
    ZRTTMEPS = 273.16_JPRB - ZRTTMEPS_DELTA   ! matches RTT (273.16) - 0.2

    PLAMSK = 0.0_JPRB

    DO JT = 1, NTILES
      SELECT CASE (JT)
      CASE (1)            ! ocean (Fortran JT=1) -> LARGE
        PLAMSK(:, JT) = ZLARGE
      CASE (2)            ! sea ice (Fortran JT=2)
        IF (LSICOUP_F) THEN
          PLAMSK(:, JT) = ZLARGE
        ELSE
          DO JL = 1, KLON
            IF (TSKM1M(JL, JT) > TSRF(JL, JT)) THEN
              PLAMSK(JL, JT) = RVLAMSKS(12)
            ELSE
              PLAMSK(JL, JT) = RVLAMSK(12)
            END IF
          END DO
        END IF
      CASE (3)            ! wet skin (Fortran JT=3) -> LARGEWT
        PLAMSK(:, JT) = ZLARGEWT
      CASE (4)            ! low veg dry (Fortran JT=4)
        DO JL = 1, KLON
          IF (TSKM1M(JL, JT) > TSRF(JL, JT)) THEN
            PLAMSK(JL, JT) = RVLAMSKS(KTVL(JL))
          ELSE
            PLAMSK(JL, JT) = RVLAMSK(KTVL(JL))
          END IF
        END DO
      CASE (5)            ! snow on low veg (Fortran JT=5)
        DO JL = 1, KLON
          IF (TSKM1M(JL, JT) >= TSRF(JL, JT) .AND. &
            & TSKM1M(JL, JT) > ZRTTMEPS) THEN
            PLAMSK(JL, JT) = ZLARGESN
          ELSE IF (SNM(JL) > 9000.0_JPRB) THEN
            ! vlamsk_mod.F90 uses SUM(PSNM(:,1:...))>9000; offline driver
            ! runs a single snow column so SNM(JL) alone is the sum.
            PLAMSK(JL, JT) = ZSNOW_GLACIER
          ELSE
            PLAMSK(JL, JT) = ZSNOW
          END IF
        END DO
      CASE (6)            ! high veg dry (Fortran JT=6)
        DO JL = 1, KLON
          IF (TSKM1M(JL, JT) > TSRF(JL, JT)) THEN
            PLAMSK(JL, JT) = RVLAMSKS(KTVH(JL))
          ELSE
            PLAMSK(JL, JT) = RVLAMSK(KTVH(JL))
          END IF
        END DO
      CASE (7)            ! snow under high veg (Fortran JT=7)
        DO JL = 1, KLON
          IF (TSKM1M(JL, JT) > TSRF(JL, JT)) THEN
            PLAMSK(JL, JT) = RVLAMSKS(KTVH(JL))
          ELSE
            PLAMSK(JL, JT) = ZSNOWHVEG
          END IF
        END DO
      CASE (8)            ! bare soil (Fortran JT=8)
        DO JL = 1, KLON
          IF (TSKM1M(JL, JT) > TSRF(JL, JT)) THEN
            PLAMSK(JL, JT) = RVLAMSKS(8)
          ELSE
            PLAMSK(JL, JT) = RVLAMSK(8)
          END IF
        END DO
      CASE (9)            ! lake (Fortran JT=9) -> LARGE
        PLAMSK(:, JT) = ZLARGE
      CASE (10)           ! urban (Fortran JT=10) -> RURBTC
        PLAMSK(:, JT) = RURBTC
      END SELECT
    END DO

    ! Stability factor for forest-snow mix (Fortran JT=7, post-SELECT).
    ! Matches surf_pytorch.physics.vlamsk exactly:
    !   stab = PTMST / (max(RQSNCR, SNM/RSN) * RHOCI * RSN / RHOICE)
    !   PLAMSK(JT=7) /= (1 + stab * PLAMSK(JT=7))
    JT = 7
    IF (NTILES >= JT) THEN
      DO JL = 1, KLON
        ZDEPTH = MAX(RQSNCR, SNM(JL) / RSN(JL))
        ZSTABEXSN = PTMST / (ZDEPTH * RHOCI * RSN(JL) / RHOICE)
        PLAMSK(JL, JT) = PLAMSK(JL, JT) / (1.0_JPRB + ZSTABEXSN * PLAMSK(JL, JT))
      END DO
    END IF
  END SUBROUTINE SURF_REF_VLAMSK


  !=====================================================================!
  ! SURF_REF_SEB -- Penman-Monteith skin-T elimination.
  !
  ! Direct transcription of surfseb_ctl_mod.F90 sections 2-8.  PLAMSK
  ! is an INPUT here (computed by SURF_REF_VLAMSK), mirroring how the
  ! port's surf_seb() consumes the output of vlamsk().
  !=====================================================================!
  SUBROUTINE SURF_REF_SEB( &
      &   KLON_C, NTILES_C, &
      &   SSKM1M, TSKM1M, QSKM1M, DQSDT, &
      &   RHOCHU, RHOCQU, ALPHAL, ALPHAS, &
      &   SSRFL, SLRFL, TSKRAD, EMIS, &
      &   TSRF, PLAMSK, &
      &   FRTI, &
      &   ASL, BSL, AQL, BQL, &
      &   PTMST, &
      &   RVTRSR, &
      &   RCPD, RLSTT, RLVTT, RSIGMA, RTT, RVTMP2, &
      &   LELWDD, LELWTL, &
      &   PJS, PJQ, PSSK, PTSK, PSSH, PSLH, PSTR, PG0, PSL, PQL  &
      & ) BIND(C, NAME="surf_ref_seb")

    !--- dimensions ------------------------------------------------------
    INTEGER(C_INT), VALUE, INTENT(IN) :: KLON_C, NTILES_C
    !--- inputs ----------------------------------------------------------
    REAL(JPRB), INTENT(IN)  :: SSKM1M(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: TSKM1M(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: QSKM1M(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: DQSDT (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: RHOCHU(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: RHOCQU(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: ALPHAL(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: ALPHAS(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: SSRFL (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: SLRFL (KLON_C)
    REAL(JPRB), INTENT(IN)  :: TSKRAD(KLON_C)
    REAL(JPRB), INTENT(IN)  :: EMIS  (KLON_C)
    REAL(JPRB), INTENT(IN)  :: TSRF  (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: PLAMSK(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: FRTI  (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN)  :: ASL(KLON_C), BSL(KLON_C)
    REAL(JPRB), INTENT(IN)  :: AQL(KLON_C), BQL(KLON_C)
    REAL(JPRB), VALUE, INTENT(IN) :: PTMST
    REAL(JPRB), INTENT(IN)  :: RVTRSR(0:20)
    REAL(JPRB), VALUE, INTENT(IN) :: RCPD, RLSTT, RLVTT, RSIGMA, RTT, RVTMP2
    LOGICAL(C_BOOL), VALUE, INTENT(IN) :: LELWDD, LELWTL

    !--- outputs ---------------------------------------------------------
    REAL(JPRB), INTENT(OUT) :: PJS (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PJQ (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PSSK(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PTSK(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PSSH(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PSLH(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PSTR(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PG0 (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PSL (KLON_C)
    REAL(JPRB), INTENT(OUT) :: PQL (KLON_C)

    !--- locals ----------------------------------------------------------
    REAL(JPRB) :: ZDLWDT(KLON_C), ZDLWDTSTAR, ZTSKMEAN
    REAL(JPRB) :: ZICPTM1(KLON_C, NTILES_C)
    REAL(JPRB) :: ZCJS1, ZCJS3, ZCJQ2, ZCJQ3, ZCJQ4
    REAL(JPRB) :: ZLAM, ZFRSR, ZCOEF1, ZZ, ZIZZ
    REAL(JPRB) :: ZDSS1(KLON_C, NTILES_C), ZDSS2(KLON_C, NTILES_C), &
      &           ZDSS4(KLON_C, NTILES_C)
    REAL(JPRB) :: ZDJS1(KLON_C, NTILES_C), ZDJS2(KLON_C, NTILES_C), &
      &           ZDJS4(KLON_C, NTILES_C)
    REAL(JPRB) :: ZDJQ1(KLON_C, NTILES_C), ZDJQ2(KLON_C, NTILES_C), &
      &           ZDJQ4(KLON_C, NTILES_C)
    REAL(JPRB) :: ZEJS1(KLON_C), ZEJS2(KLON_C), ZEJS4(KLON_C)
    REAL(JPRB) :: ZEJQ1(KLON_C), ZEJQ2(KLON_C), ZEJQ4(KLON_C)
    REAL(JPRB) :: ZZ1, ZZ2, ZZ3, ZY1, ZY2, ZY3, ZZDET, ZJQ, ZJS
    REAL(JPRB) :: ZSL, ZQL
    INTEGER :: KLON, NTILES, JL, JT
    LOGICAL :: LELWDD_F, LELWTL_F

    KLON   = KLON_C
    NTILES = NTILES_C
    LELWDD_F  = LELWDD
    LELWTL_F  = LELWTL

    ! Section 2 -- dLW/dT (tile-independent).  (lines 239-247)
    IF (LELWDD_F) THEN
      DO JL = 1, KLON
        ZDLWDT(JL) = -4.0_JPRB * EMIS(JL) * RSIGMA * TSKRAD(JL) ** 3
      END DO
    ELSE
      DO JL = 1, KLON
        ZDLWDT(JL) = 4.0_JPRB * (SLRFL(JL) / TSKRAD(JL))
      END DO
    END IF

    ! Sections 3-4 -- Js/Jq coefficients, eliminate Ssk.  (lines 252-403)
    DO JT = 1, NTILES
      ! Latent heat selector (lines 271-275): sea-ice (JT=2) and
      ! snow-on-low-veg (JT=5) use RLSTT (sublimation); else RLVTT.
      IF (JT == 2 .OR. JT == 5) THEN
        ZLAM = RLSTT
      ELSE
        ZLAM = RLVTT
      END IF
      ! Solar transmission ZFRSR (lines 320,329,346,355).
      ! Mirrors surf_pytorch.physics.seb._tile_solar_transmission:
      !   * wet skin (JT=3), low veg dry (JT=4)  -> 1 - RVTRSR(1) = 0.95
      !   * high veg dry (JT=6), high veg snow (JT=7) -> 1 - RVTRSR(3) = 0.97
      ZFRSR = 1.0_JPRB
      IF (JT == 3 .OR. JT == 4) ZFRSR = 1.0_JPRB - RVTRSR(1)
      IF (JT == 6 .OR. JT == 7) ZFRSR = 1.0_JPRB - RVTRSR(3)

      DO JL = 1, KLON
        ZICPTM1(JL, JT) = TSKM1M(JL, JT) / SSKM1M(JL, JT)
        ZCJS1 = RHOCHU(JL, JT)
        ZCJS3 = -RHOCHU(JL, JT)
        ZCJQ2 = RHOCQU(JL, JT) * ALPHAL(JL, JT)
        ZCJQ3 = -RHOCQU(JL, JT) * ALPHAS(JL, JT) * DQSDT(JL, JT) * ZICPTM1(JL, JT)
        ZCJQ4 = -RHOCQU(JL, JT) * ALPHAS(JL, JT) * &
          &      (QSKM1M(JL, JT) - DQSDT(JL, JT) * TSKM1M(JL, JT))

        ZCOEF1 = ZLAM - RCPD * TSKM1M(JL, JT) * RVTMP2
        ZZ = (ZDLWDT(JL) - PLAMSK(JL, JT)) * ZICPTM1(JL, JT) + ZCJS3 + ZCOEF1 * ZCJQ3
        ZIZZ = 1.0_JPRB / ZZ

        ZDSS1(JL, JT) = -ZCJS1 * ZIZZ
        ZDSS2(JL, JT) = -ZCOEF1 * ZCJQ2 * ZIZZ
        ZDSS4(JL, JT) = (-SSRFL(JL, JT) * ZFRSR - SLRFL(JL) + &
          &              ZDLWDT(JL) * TSKRAD(JL) - ZCOEF1 * ZCJQ4 - &
          &              PLAMSK(JL, JT) * TSRF(JL, JT)) * ZIZZ
        ZDJS1(JL, JT) = ZCJS1 + ZCJS3 * ZDSS1(JL, JT)
        ZDJS2(JL, JT) = ZCJS3 * ZDSS2(JL, JT)
        ZDJS4(JL, JT) = ZCJS3 * ZDSS4(JL, JT)
        ZDJQ1(JL, JT) = ZCJQ3 * ZDSS1(JL, JT)
        ZDJQ2(JL, JT) = ZCJQ2 + ZCJQ3 * ZDSS2(JL, JT)
        ZDJQ4(JL, JT) = ZCJQ3 * ZDSS4(JL, JT) + ZCJQ4
      END DO
    END DO

    ! Section 5 -- tile-mean coefficients.  (lines 408-426)
    ZEJS1 = 0.0_JPRB; ZEJS2 = 0.0_JPRB; ZEJS4 = 0.0_JPRB
    ZEJQ1 = 0.0_JPRB; ZEJQ2 = 0.0_JPRB; ZEJQ4 = 0.0_JPRB
    DO JT = 1, NTILES
      DO JL = 1, KLON
        ZEJS1(JL) = ZEJS1(JL) + FRTI(JL, JT) * ZDJS1(JL, JT)
        ZEJS2(JL) = ZEJS2(JL) + FRTI(JL, JT) * ZDJS2(JL, JT)
        ZEJS4(JL) = ZEJS4(JL) + FRTI(JL, JT) * ZDJS4(JL, JT)
        ZEJQ1(JL) = ZEJQ1(JL) + FRTI(JL, JT) * ZDJQ1(JL, JT)
        ZEJQ2(JL) = ZEJQ2(JL) + FRTI(JL, JT) * ZDJQ2(JL, JT)
        ZEJQ4(JL) = ZEJQ4(JL) + FRTI(JL, JT) * ZDJQ4(JL, JT)
      END DO
    END DO

    ! Section 6 -- 2x2 solve for grid-mean (Sl, Ql).  (lines 430-446)
    DO JL = 1, KLON
      ZZ1 = 1.0_JPRB - ASL(JL) * ZEJS1(JL)
      ZZ2 = AQL(JL) * ZEJS2(JL)
      ZZ3 = BSL(JL) * ZEJS1(JL) + BQL(JL) * ZEJS2(JL) + ZEJS4(JL)
      ZY1 = 1.0_JPRB - AQL(JL) * ZEJQ2(JL)
      ZY2 = ASL(JL) * ZEJQ1(JL)
      ZY3 = BSL(JL) * ZEJQ1(JL) + BQL(JL) * ZEJQ2(JL) + ZEJQ4(JL)

      ZZDET = ZZ1 * ZY1 - ZZ2 * ZY2
      ZIZZ = 1.0_JPRB / ZZDET
      ZJQ = (ZZ3 * ZY2 + ZZ1 * ZY3) * ZIZZ
      ZJS = (ZZ2 * ZY3 + ZZ3 * ZY1) * ZIZZ

      ZSL = ASL(JL) * ZJS + BSL(JL)
      ZQL = AQL(JL) * ZJQ + BQL(JL)
      PSL(JL) = ZSL
      PQL(JL) = ZQL
    END DO

    ! Section 7 -- per-tile fluxes and skin temperature.  (lines 450-472)
    DO JT = 1, NTILES
      IF (JT == 2 .OR. JT == 5) THEN
        ZLAM = RLSTT
      ELSE
        ZLAM = RLVTT
      END IF
      DO JL = 1, KLON
        PSSK(JL, JT) = ZDSS1(JL, JT) * PSL(JL) + ZDSS2(JL, JT) * PQL(JL) + ZDSS4(JL, JT)
        PJS (JL, JT) = ZDJS1(JL, JT) * PSL(JL) + ZDJS2(JL, JT) * PQL(JL) + ZDJS4(JL, JT)
        PJQ (JL, JT) = ZDJQ1(JL, JT) * PSL(JL) + ZDJQ2(JL, JT) * PQL(JL) + ZDJQ4(JL, JT)
        PTSK(JL, JT) = PSSK(JL, JT) * ZICPTM1(JL, JT)

        PSSH(JL, JT) = PJS(JL, JT) - RCPD * TSKM1M(JL, JT) * RVTMP2 * PJQ(JL, JT)
        PSLH(JL, JT) = ZLAM * PJQ(JL, JT)
        PSTR(JL, JT) = SLRFL(JL) + ZDLWDT(JL) * (PTSK(JL, JT) - TSKRAD(JL))
        PG0 (JL, JT) = PLAMSK(JL, JT) * (PTSK(JL, JT) - TSRF(JL, JT))
      END DO
    END DO

    ! Section 8 -- optional re-compute of tiled LW around grid-box mean.
    ! (lines 478-495)  Energy-conserving correction; runs only when
    ! LELWTL is set (offline driver default is .FALSE.).
    IF (LELWTL_F) THEN
      DO JL = 1, KLON
        ZTSKMEAN = 0.0_JPRB
        DO JT = 1, NTILES
          ZTSKMEAN = ZTSKMEAN + FRTI(JL, JT) * PTSK(JL, JT)
        END DO
        ZDLWDTSTAR = -4.0_JPRB * EMIS(JL) * RSIGMA * ZTSKMEAN ** 3
        DO JT = 1, NTILES
          PSTR(JL, JT) = SLRFL(JL) + ZDLWDTSTAR * (PTSK(JL, JT) - ZTSKMEAN)
        END DO
      END DO
    END IF
  END SUBROUTINE SURF_REF_SEB


  !=====================================================================!
  ! SURF_REF_VEXCS -- Beljaars (1990) Monin-Obukhov exchange coeffs.
  !
  ! Direct transcription of vexcs_mod.F90 (48r1) sections 2-4 (the
  ! K_VMASS<=0 branch -- the IBM MASS vector path is not needed for the
  ! reference).  The RITBL lookup table is passed in by the caller
  ! (built by surf_pytorch.tables.ritbl.build_ritbl()) so the module
  ! has no initialisation step.
  !
  ! Constants consumed (from YOS_CST / YOS_EXC / YOS_EXCS):
  !   RD, RETV, RG, RVTMP2, RKAP, REPDU2, RPARZI, RCHBHDL, DRITBL,
  !   RCHETA, RCHETB, RCHBA, RCHBB, RCHBC, RCHBD, RCHB23A, RCHBBCD,
  !   RCHBCD, RCDHALF, RCDHPI2, JPRITBL.
  ! All are passed in by the caller.
  !=====================================================================!
  SUBROUTINE SURF_REF_VEXCS( &
      &   KLON_C, KITT_C, LDINIT, &
      &   PUMLEV, PVMLEV, PTMLEV, PQMLEV, PAPHMS, PGEOMLEV, &
      &   PCPTGZLEV, PCPTS, PQSAM, &
      &   PZ0MM, PZ0HM, PZ0QM, PZDL, PBUOM, PUCURR, PVCURR, PI10FGCV, &
      &   RITBL, &
      &   RD_C, RETV_C, RG_C, RVTMP2_C, RKAP_C, REPDU2_C, RPARZI_C, &
      &   RCHBHDL_C, DRITBL_C, &
      &   RCHETA_C, RCHETB_C, RCHBA_C, RCHBB_C, RCHBC_C, RCHBD_C, &
      &   RCHB23A_C, RCHBBCD_C, RCHBCD_C, RCDHALF_C, RCDHPI2_C, &
      &   JPRITBL_C, &
      &   PCFM, PCFH, PCFQ, PKH, PKC &
      & ) BIND(C, NAME="surf_ref_vexcs")

    !--- dimensions ------------------------------------------------------
    INTEGER(C_INT), VALUE, INTENT(IN) :: KLON_C, KITT_C, JPRITBL_C
    LOGICAL(C_BOOL),  VALUE, INTENT(IN) :: LDINIT
    !--- inputs ----------------------------------------------------------
    REAL(JPRB), INTENT(IN)  :: PUMLEV(KLON_C), PVMLEV(KLON_C)
    REAL(JPRB), INTENT(IN)  :: PTMLEV(KLON_C), PQMLEV(KLON_C)
    REAL(JPRB), INTENT(IN)  :: PAPHMS(KLON_C), PGEOMLEV(KLON_C)
    REAL(JPRB), INTENT(IN)  :: PCPTGZLEV(KLON_C), PCPTS(KLON_C), PQSAM(KLON_C)
    REAL(JPRB), INTENT(IN)  :: PZ0MM(KLON_C), PZ0HM(KLON_C), PZ0QM(KLON_C)
    REAL(JPRB), INTENT(IN)  :: PBUOM(KLON_C), PUCURR(KLON_C), PVCURR(KLON_C)
    REAL(JPRB), INTENT(IN)  :: PI10FGCV(KLON_C)
    REAL(JPRB), INTENT(IN)  :: RITBL(JPRITBL_C)
    !--- in/out ----------------------------------------------------------
    REAL(JPRB), INTENT(INOUT) :: PZDL(KLON_C)
    !--- scalars (BY VALUE) ----------------------------------------------
    REAL(JPRB), VALUE, INTENT(IN) :: RD_C, RETV_C, RG_C, RVTMP2_C, RKAP_C
    REAL(JPRB), VALUE, INTENT(IN) :: REPDU2_C, RPARZI_C, RCHBHDL_C, DRITBL_C
    REAL(JPRB), VALUE, INTENT(IN) :: RCHETA_C, RCHETB_C, RCHBA_C, RCHBB_C
    REAL(JPRB), VALUE, INTENT(IN) :: RCHBC_C, RCHBD_C, RCHB23A_C
    REAL(JPRB), VALUE, INTENT(IN) :: RCHBBCD_C, RCHBCD_C, RCDHALF_C, RCDHPI2_C
    !--- outputs ---------------------------------------------------------
    REAL(JPRB), INTENT(OUT) :: PCFM(KLON_C), PCFH(KLON_C), PCFQ(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PKH(KLON_C),  PKC(KLON_C)

    !--- locals (vexcs_mod.F90 142-166) ----------------------------------
    INTEGER :: JIT, JL, IRIB
    REAL(JPRB) :: Z1DZ0M(KLON_C), Z1DZ0H(KLON_C), Z1DZ0Q(KLON_C)
    REAL(JPRB) :: Z1DZ0MD(KLON_C), Z1DZ0HD(KLON_C), Z1DZ0QD(KLON_C)
    REAL(JPRB) :: ZXLNM(KLON_C), ZXLNH(KLON_C), ZXLNQ(KLON_C)
    REAL(JPRB) :: ZF2(KLON_C), ZF3(KLON_C), ZDU2(KLON_C), ZNLEV(KLON_C)
    REAL(JPRB) :: Z1DZ1D(KLON_C), ZRICLS(KLON_C), ZETA2(KLON_C), ZETA3(KLON_C)
    REAL(JPRB) :: ZAUX1, ZAUX2, ZCON1, ZCON2, ZCON3, ZDRORO
    REAL(JPRB) :: ZETA, ZETA1, ZHUI, ZIPBL, ZL
    REAL(JPRB) :: ZPRH, ZPRH0, ZPRH1, ZPRM, ZPRM0, ZPRM1
    REAL(JPRB) :: ZPRQ, ZPRQ0, ZPRC, ZPRC0, ZRHO, ZRIB1, ZUABS, ZWST2
    REAL(JPRB) :: ZX2, ZA, ZB
    REAL(JPRD) :: ZDUMMY
    LOGICAL :: LDINIT_F

    !--------------------------------------------------------------------!
    ! Initialise module-level PSI-function constants (read by PSIMU /    !
    ! PSIHU / PSIMS / PSIHS at module scope).                            !
    !--------------------------------------------------------------------!
    RCHBB_M   = RCHBB_C
    RCHBCD_M  = RCHBCD_C
    RCHBD_M   = RCHBD_C
    RCHBA_M   = RCHBA_C
    RCHBBCD_M = RCHBBCD_C
    RCHB23A_M = RCHB23A_C
    RCDHALF_M = RCDHALF_C
    RCDHPI2_M = RCDHPI2_C
    LDINIT_F  = LDINIT

    !--------------------------------------------------------------!
    ! 1. INITIALIZE CONSTANTS                                       !
    !--------------------------------------------------------------!
      ZCON1 = RVTMP2_C - RETV_C
      ZCON2 = 2.0_JPRB / 3.0_JPRB
      ZCON3 = RKAP_C ** 2
      ZIPBL = RPARZI_C

      !--------------------------------------------------------------!
      ! 2. COMPUTATION OF BASIC QUANTITIES                           !
      !--------------------------------------------------------------!
      DO JL = 1, KLON_C
        IF (PBUOM(JL) <= 0.0_JPRB) THEN
          ZWST2 = 0.0_JPRB
        ELSE
          ZWST2 = (PBUOM(JL) * ZIPBL) ** ZCON2
        ENDIF
        ZDU2(JL) = MAX(REPDU2_C, (PUMLEV(JL) - PUCURR(JL)) ** 2 &
            &   + (PVMLEV(JL) - PVCURR(JL)) ** 2 + ZWST2)
        ZDRORO = 2.0_JPRB * (PCPTGZLEV(JL) - PCPTS(JL)) &
            &   / (PCPTGZLEV(JL) + PCPTS(JL) - PGEOMLEV(JL)) &
            &   - ZCON1 * (PQMLEV(JL) - MIN(PQSAM(JL), 0.1_JPRB))
        ZRICLS(JL) = PGEOMLEV(JL) * ZDRORO / ZDU2(JL)

        ZNLEV(JL)  = PGEOMLEV(JL) / RG_C + PZ0MM(JL)
        Z1DZ0M(JL) = ZNLEV(JL) / PZ0MM(JL)
        Z1DZ0H(JL) = ZNLEV(JL) / PZ0HM(JL)
        Z1DZ0Q(JL) = ZNLEV(JL) / PZ0QM(JL)
        Z1DZ1D(JL) = Z1DZ0M(JL) / (Z1DZ0M(JL) - 1.0_JPRB)
        ZXLNM(JL)  = LOG(Z1DZ0M(JL))
        ZXLNH(JL)  = LOG(Z1DZ0H(JL))
        ZXLNQ(JL)  = LOG(Z1DZ0Q(JL))
        Z1DZ0MD(JL) = 1.0_JPRB / Z1DZ0M(JL)
        Z1DZ0HD(JL) = 1.0_JPRB / Z1DZ0H(JL)
      ENDDO

      !--------------------------------------------------------------!
      ! 3.1 FIRST AND SECOND GUESS                                    !
      !--------------------------------------------------------------!
      IF (LDINIT_F) THEN
        DO JL = 1, KLON_C
          IF (ZRICLS(JL) > 0.0_JPRB) THEN
            ZRIB1 = ZRICLS(JL) * ZXLNM(JL) / ZXLNH(JL)
            IRIB  = INT(ZRIB1 / DRITBL_C) + 1
            IF (IRIB >= JPRITBL_C) THEN
              ZETA1 = RITBL(JPRITBL_C)
            ELSE
              ZX2   = IRIB * DRITBL_C
              ZA    = (ZX2 - ZRIB1) / DRITBL_C
              ZB    = 1.0_JPRB - ZA
              ZETA1 = ZA * RITBL(IRIB) + ZB * RITBL(IRIB + 1)
            ENDIF
          ELSE
            ZETA1 = ZRICLS(JL) * ZXLNM(JL) / ZXLNH(JL)
          ENDIF
          ZETA2(JL) = ZETA1 * ZXLNM(JL) * Z1DZ1D(JL)
        ENDDO
      ELSE
        DO JL = 1, KLON_C
          ZETA2(JL) = PZDL(JL)
        ENDDO
      ENDIF

      !--------------------------------------------------------------!
      ! 3.2 BACK SUBSTITUTION                                         !
      !--------------------------------------------------------------!
      DO JL = 1, KLON_C
        IF (ZETA2(JL) > 0.0_JPRB) THEN
          ZDUMMY = ZETA2(JL)
          ZPRH1 = PSIHS(ZDUMMY)
          ZDUMMY = ZETA2(JL) * Z1DZ0HD(JL)
          ZPRH0 = PSIHS(ZDUMMY)
          ZDUMMY = ZETA2(JL)
          ZPRM1 = PSIMS(ZDUMMY)
          ZDUMMY = ZETA2(JL) * Z1DZ0MD(JL)
          ZPRM0 = PSIMS(ZDUMMY)
        ELSE
          ZDUMMY = ZETA2(JL)
          ZPRH1 = PSIHU(ZDUMMY)
          ZDUMMY = ZETA2(JL) * Z1DZ0HD(JL)
          ZPRH0 = PSIHU(ZDUMMY)
          ZDUMMY = ZETA2(JL)
          ZPRM1 = PSIMU(ZDUMMY)
          ZDUMMY = ZETA2(JL) * Z1DZ0MD(JL)
          ZPRM0 = PSIMU(ZDUMMY)
        ENDIF
        ZPRH = ZXLNH(JL) - ZPRH1 + ZPRH0
        ZPRM = ZXLNM(JL) - ZPRM1 + ZPRM0
        ZHUI = ZPRM ** 2 * Z1DZ1D(JL) / ZPRH
        ZETA3(JL) = ZRICLS(JL) * ZHUI
        ZF2(JL)   = ZRICLS(JL) - ZETA2(JL) / ZHUI
      ENDDO

      !--------------------------------------------------------------!
      ! 3.3 ITERATE KITT TIMES                                        !
      !--------------------------------------------------------------!
      DO JIT = 1, KITT_C
        DO JL = 1, KLON_C
          IF (ZETA3(JL) > 0.0_JPRB) THEN
            ZDUMMY = ZETA3(JL)
            ZPRH1 = PSIHS(ZDUMMY)
            ZDUMMY = ZETA3(JL) * Z1DZ0HD(JL)
            ZPRH0 = PSIHS(ZDUMMY)
            ZDUMMY = ZETA3(JL)
            ZPRM1 = PSIMS(ZDUMMY)
            ZDUMMY = ZETA3(JL) * Z1DZ0MD(JL)
            ZPRM0 = PSIMS(ZDUMMY)
          ELSE
            ZDUMMY = ZETA3(JL)
            ZPRH1 = PSIHU(ZDUMMY)
            ZDUMMY = ZETA3(JL) * Z1DZ0HD(JL)
            ZPRH0 = PSIHU(ZDUMMY)
            ZDUMMY = ZETA3(JL)
            ZPRM1 = PSIMU(ZDUMMY)
            ZDUMMY = ZETA3(JL) * Z1DZ0MD(JL)
            ZPRM0 = PSIMU(ZDUMMY)
          ENDIF
          ZPRH = ZXLNH(JL) - ZPRH1 + ZPRH0
          ZPRM = ZXLNM(JL) - ZPRM1 + ZPRM0
          ZF3(JL) = ZRICLS(JL) - ZETA3(JL) * ZPRH &
              &   / (Z1DZ1D(JL) * ZPRM ** 2)
          IF (ABS(ZF3(JL) - ZF2(JL)) > 1.0E-25_JPRB) THEN
            ZETA = ZETA2(JL) - ZF2(JL) * (ZETA3(JL) - ZETA2(JL)) &
                & / (ZF3(JL) - ZF2(JL))
          ELSE
            ZETA = ZETA3(JL)
          ENDIF
          ! SAFETY PROVISION FOR DIVERGING ITERATIONS
          IF (ZETA * ZETA3(JL) < 0.0_JPRB) ZETA = ZETA3(JL) * 0.5_JPRB
          ZETA2(JL) = ZETA3(JL)
          ZF2(JL)   = ZF3(JL)
          ZETA3(JL) = ZETA
        ENDDO
      ENDDO

      !--------------------------------------------------------------!
      ! 3.4 COPY ZNLEV/L TO OUTPUT ARRAY                              !
      !--------------------------------------------------------------!
      DO JL = 1, KLON_C
        PZDL(JL) = ZETA3(JL)
      ENDDO

      !--------------------------------------------------------------!
      ! 4. COMPUTE EXCHANGE COEFFICIENTS                              !
      !--------------------------------------------------------------!
      DO JL = 1, KLON_C
        IF (ZETA3(JL) > RCHBHDL_C) THEN
          ZL = ZNLEV(JL) / ZETA3(JL)
          ZNLEV(JL) = RCHBHDL_C * ZL + MAX(PZ0MM(JL), PZ0QM(JL))
          ZETA3(JL) = ZNLEV(JL) / ZL
          Z1DZ0M(JL) = ZNLEV(JL) / PZ0MM(JL)
          Z1DZ0H(JL) = ZNLEV(JL) / PZ0HM(JL)
          Z1DZ0Q(JL) = ZNLEV(JL) / PZ0QM(JL)
          ZXLNM(JL)  = LOG(Z1DZ0M(JL))
          ZXLNH(JL)  = LOG(Z1DZ0H(JL))
          ZXLNQ(JL)  = LOG(Z1DZ0Q(JL))
        ENDIF
      ENDDO

      DO JL = 1, KLON_C
        Z1DZ0MD(JL) = 1.0_JPRB / Z1DZ0M(JL)
        Z1DZ0HD(JL) = 1.0_JPRB / Z1DZ0H(JL)
        Z1DZ0QD(JL) = 1.0_JPRB / Z1DZ0Q(JL)
      ENDDO

      DO JL = 1, KLON_C
        IF (ZETA3(JL) > 0.0_JPRB) THEN
          ZDUMMY = ZETA3(JL)
          ZPRM1 = PSIMS(ZDUMMY)
          ZDUMMY = ZETA3(JL) * Z1DZ0MD(JL)
          ZPRM0 = PSIMS(ZDUMMY)
          ZDUMMY = ZETA3(JL)
          ZPRH1 = PSIHS(ZDUMMY)
          ZDUMMY = ZETA3(JL) * Z1DZ0HD(JL)
          ZPRH0 = PSIHS(ZDUMMY)
          ZDUMMY = ZETA3(JL) * Z1DZ0QD(JL)
          ZPRQ0 = PSIHS(ZDUMMY)
          ZDUMMY = ZETA3(JL) * Z1DZ0MD(JL)
          ZPRC0 = PSIHS(ZDUMMY)
        ELSE
          ZDUMMY = ZETA3(JL)
          ZPRM1 = PSIMU(ZDUMMY)
          ZDUMMY = ZETA3(JL) * Z1DZ0MD(JL)
          ZPRM0 = PSIMU(ZDUMMY)
          ZDUMMY = ZETA3(JL)
          ZPRH1 = PSIHU(ZDUMMY)
          ZDUMMY = ZETA3(JL) * Z1DZ0HD(JL)
          ZPRH0 = PSIHU(ZDUMMY)
          ZDUMMY = ZETA3(JL) * Z1DZ0QD(JL)
          ZPRQ0 = PSIHU(ZDUMMY)
          ZDUMMY = ZETA3(JL) * Z1DZ0MD(JL)
          ZPRC0 = PSIHU(ZDUMMY)
        ENDIF
        ZPRM = ZXLNM(JL) - ZPRM1 + ZPRM0
        ZPRH = ZXLNH(JL) - ZPRH1 + ZPRH0
        ZPRQ = ZXLNQ(JL) - ZPRH1 + ZPRQ0
        ZPRC = ZXLNM(JL) - ZPRH1 + ZPRC0
        ZRHO  = PAPHMS(JL) / (RD_C * PTMLEV(JL) &
            &   * (1.0_JPRB + RETV_C * PQMLEV(JL)))
        ZUABS = SQRT(ZDU2(JL))
        ZAUX2 = ZUABS * ZRHO * ZCON3
        PCFM(JL) = ZAUX2 / (ZPRM ** 2)
        PCFH(JL) = ZAUX2 / (ZPRM * ZPRH)
        PCFQ(JL) = ZAUX2 / (ZPRM * ZPRQ)
      PKH(JL)  = PCFH(JL) / ZRHO
      PKC(JL)  = ZUABS * ZCON3 / (ZPRM * ZPRC)
    ENDDO
  END SUBROUTINE SURF_REF_VEXCS

  !=====================================================================!
  ! SURF_REF_DQSDT -- skin-temperature saturation Jacobian PDQS.
  !
  ! Direct transcription of vsurf_mod.F90 (262-285), using the fcttre.h
  ! statement functions FOEEW / FOEDESU / FOEDELTA (mirrored above as
  ! module functions FOEEW_M / FOEDESU_M / FOEDELTA_M).  This is the
  ! *independent* Fortran computation of the value SURF_REF_SEB consumes
  ! as DQSDT; the surf_pytorch port computes the same quantity via
  ! utils.dqsat_dt_ifs().  Comparing the two (see test_dqsdt.py) is the
  ! only way to validate the dqs/dT *formula itself*, since SURF_REF_SEB
  ! takes DQSDT as an input and so cannot catch a wrong Jacobian.
  !
  ! Faithfulness notes
  ! ------------------
  !   * ZSALIN is passed in per-column (vsurf uses a tile-scalar, but the
  !     only variation is ocean=0.98 under LEOCSA vs 1.0 elsewhere); the
  !     caller sets it to match the tile being validated.
  !   * The KTILE==7 Tsk clamp (vsurf 273-277) is applied when LCLAMP7 is
  !     true, with the same ZEPSILON and RTT guards.  Upstream uses
  !     EPSILON(1.0); a hard 0.0 (no clamp) is the offline default since
  !     tile fractions are passed in fully.
  !   * The arithmetic is byte-identical to vsurf_mod.F90 278-285:
  !       PQS  = FOEEW(ZTSK)/P
  !       ZCOR = ZSALIN/(1-RETV*PQS)
  !       PQS  = PQS*ZCOR
  !       PDQS = PQS*ZCOR*FOEDESU(ZTSK)
  !     i.e. ZCOR is used twice -> PDQS = (es/P)*ZCOR**2*FOEDESU.
  !=====================================================================!
  SUBROUTINE SURF_REF_DQSDT( &
      &   KLON_C, &
      &   PTSKM1M, PAPHMS, PFRTI7, ZSALIN, &
      &   LCLAMP7, ZEPSILON, RETV_C, &
      &   R2ES, R3LES, R3IES, R4LES, R4IES, R5LES, R5IES, RTT, &
      &   PDQS &
      & ) BIND(C, NAME="surf_ref_dqsdt")

    !--- dimensions ------------------------------------------------------
    INTEGER(C_INT), VALUE, INTENT(IN) :: KLON_C

    !--- inputs ----------------------------------------------------------
    REAL(JPRB), INTENT(IN)  :: PTSKM1M(KLON_C)     ! skin T per column
    REAL(JPRB), INTENT(IN)  :: PAPHMS  (KLON_C)    ! surface pressure
    REAL(JPRB), INTENT(IN)  :: PFRTI7  (KLON_C)    ! tile-7 fraction (KTILE==7)
    REAL(JPRB), INTENT(IN)  :: ZSALIN  (KLON_C)    ! salinity factor (0.98/1.0)
    LOGICAL(C_BOOL), VALUE, INTENT(IN) :: LCLAMP7  ! apply KTILE==7 clamp
    REAL(JPRB), VALUE, INTENT(IN) :: ZEPSILON
    REAL(JPRB), VALUE, INTENT(IN) :: RETV_C
    REAL(JPRB), VALUE, INTENT(IN) :: R2ES, R3LES, R3IES, R4LES, R4IES
    REAL(JPRB), VALUE, INTENT(IN) :: R5LES, R5IES, RTT

    !--- output ----------------------------------------------------------
    REAL(JPRB), INTENT(OUT) :: PDQS(KLON_C)

    !--- locals ----------------------------------------------------------
    INTEGER  :: JL
    REAL(JPRB) :: ZTSK, ZPQS, ZCOR

    ! Publish the YOETHF scalars so the fcttre module functions see them.
    R2ES_M = R2ES; R3LES_M = R3LES; R3IES_M = R3IES
    R4LES_M = R4LES; R4IES_M = R4IES
    R5LES_M = R5LES; R5IES_M = R5IES; RTT_M = RTT

    DO JL = 1, KLON_C
      ZTSK = PTSKM1M(JL)
      ! vsurf 273-277: clamp spurious tile-7 overshoots to RTT.
      IF (LCLAMP7) THEN
        IF (PFRTI7(JL) <= ZEPSILON .AND. ZTSK > RTT + 100.0_JPRB*ZEPSILON) THEN
          ZTSK = RTT
        ENDIF
      ENDIF
      ! vsurf 279-284 (verbatim).
      ZPQS = FOEEW_M(ZTSK) / PAPHMS(JL)
      ZCOR = ZSALIN(JL) / (1.0_JPRB - RETV_C * ZPQS)
      ZPQS = ZPQS * ZCOR
      PDQS(JL) = ZPQS * ZCOR * FOEDESU_M(ZTSK)
    ENDDO
  END SUBROUTINE SURF_REF_DQSDT


  !=====================================================================!
  ! SURF_REF_STOMATA -- Jarvis stomatal resistance (vsurf_mod.F90       !
  !                     §2.3, lines 294-422).                            !
  !                                                                      !
  ! Direct transcription of the IF (KTILE == 4) block that produces the  !
  ! per-column canopy resistances PWETL / PWETH / PWETHS / PWETB /       !
  ! PWETLU.  Tiles 6/7/8 re-use these values in the upstream driver;     !
  ! the surf_pytorch port follows the same convention (see               !
  ! physics.stomata.jarvis_resistance).                                  !
  !                                                                      !
  ! Independent of SURF_REF_DQSDT in the sense that the alpha conversion !
  ! happens in VEVAP (surf_pytorch.physics.stomata.resistance_to_alpha). !
  ! This entry returns the resistances and the stress factors; the alpha !
  ! assembly for the SEB is built on the Python side by build_tile_alpha.!
  !                                                                      !
  ! Inputs (per-column unless noted):                                    !
  !   PTSAM1M(KLON, KLEVS) -- soil temperature profile                   !
  !   PWSAM1M(KLON, KLEVS) -- soil moisture (liquid + ice)               !
  !   KSOTY   (KLON)       -- soil texture class                         !
  !   KTVL, KTVH(KLON)     -- dominant low / high veg type               !
  !   PTMLEV, PQMLEV, PAPHMS -- lowest-model-level state + sfc pressure  !
  !   PSRFD   (KLON)       -- downward short-wave radiation              !
  !   PLAIL, PLAIH(KLON)   -- LAI for low / high veg                     !
  !   PFRTI   (KLON, NTILES) -- tile fractions (used for ZBARE)          !
  !   RVRSMIN, RVHSTR, RVROOTSA -- veg / root LUTs                       !
  !   RWPWPM, RWCAPM, RWRESTM, RQWEVAPM (0:NSOTY) -- soil LUT            !
  !   RCEPSW, RCLU, RSMINB, ZEPSF3 -- Jarvis scalars                     !
  !   RTF1..RTF4, RTT, RETV, RD, R2ES..R5IES -- physical constants       !
  !                                                                      !
  ! Outputs: PWETL / PWETH / PWETHS / PWETB / PWETLU (KLON) plus         !
  !          diagnostics PEXDIAG(KLON, 10) mirroring vsurf 412-421.      !
  !=====================================================================!
  SUBROUTINE SURF_REF_STOMATA( &
      &   KLON_C, KLEVS_C, NSOTY_C, NTILES_C, &
      &   PTSAM1M, PWSAM1M, KSOTY, KTVL, KTVH, &
      &   PTMLEV, PQMLEV, PAPHMS, PSRFD, &
      &   PLAIL, PLAIH, PFRTI, &
      &   RVRSMIN, RVHSTR, RVROOTSA, &
      &   RWPWPM, RWCAPM, RWRESTM, RQWEVAPM, &
      &   RCEPSW, RCLU_C, RSMINB, ZEPSF3, &
      &   RTF1, RTF2, RTF3, RTF4, RTT, RETV_C, RD_C, &
      &   R2ES, R3LES, R3IES, R4LES, R4IES, R5LES, R5IES, &
      &   PWETL, PWETH, PWETHS, PWETB, PWETLU, PEXDIAG &
      & ) BIND(C, NAME="surf_ref_stomata")

    !--- dimensions ------------------------------------------------------
    INTEGER(C_INT), VALUE, INTENT(IN) :: KLON_C, KLEVS_C, NSOTY_C, NTILES_C

    !--- inputs ----------------------------------------------------------
    REAL(JPRB), INTENT(IN)  :: PTSAM1M(KLON_C, KLEVS_C)
    REAL(JPRB), INTENT(IN)  :: PWSAM1M(KLON_C, KLEVS_C)
    INTEGER(C_INT), INTENT(IN) :: KSOTY(KLON_C)
    INTEGER(C_INT), INTENT(IN) :: KTVL(KLON_C), KTVH(KLON_C)
    REAL(JPRB), INTENT(IN)  :: PTMLEV(KLON_C), PQMLEV(KLON_C), PAPHMS(KLON_C)
    REAL(JPRB), INTENT(IN)  :: PSRFD (KLON_C)
    REAL(JPRB), INTENT(IN)  :: PLAIL (KLON_C), PLAIH(KLON_C)
    REAL(JPRB), INTENT(IN)  :: PFRTI (KLON_C, NTILES_C)
    ! veg / root LUTs indexed 0:20 (veg) and 1:NSOTY (soil).
    REAL(JPRB), INTENT(IN)  :: RVRSMIN(0:20), RVHSTR(0:20)
    REAL(JPRB), INTENT(IN)  :: RVROOTSA(KLEVS_C, 0:20)
    REAL(JPRB), INTENT(IN)  :: RWPWPM(0:NSOTY_C), RWCAPM(0:NSOTY_C)
    REAL(JPRB), INTENT(IN)  :: RWRESTM(0:NSOTY_C), RQWEVAPM(0:NSOTY_C)
    ! scalars
    REAL(JPRB), VALUE, INTENT(IN) :: RCEPSW, RCLU_C, RSMINB, ZEPSF3
    REAL(JPRB), VALUE, INTENT(IN) :: RTF1, RTF2, RTF3, RTF4, RTT, RETV_C, RD_C
    REAL(JPRB), VALUE, INTENT(IN) :: R2ES, R3LES, R3IES, R4LES, R4IES
    REAL(JPRB), VALUE, INTENT(IN) :: R5LES, R5IES

    !--- outputs ---------------------------------------------------------
    REAL(JPRB), INTENT(OUT) :: PWETL (KLON_C), PWETH (KLON_C)
    REAL(JPRB), INTENT(OUT) :: PWETHS(KLON_C), PWETB(KLON_C), PWETLU(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PEXDIAG(KLON_C, 10)

    !--- locals (vsurf_mod.F90 296-330) ----------------------------------
    INTEGER  :: JL, JK, JS
    REAL(JPRB) :: ZF(KLEVS_C), ZLIQ(KLEVS_C), ZLIQR(KLEVS_C)
    REAL(JPRB) :: ZRSMINL, ZRSMINH, ZLAIL, ZLAIH, ZBARE
    REAL(JPRB) :: ZWROOTL, ZWROOTH, ZWPWP, ZQWEVAP
    REAL(JPRB) :: ZWPBARE, ZQWEVAPBARE, ZF2B
    REAL(JPRB) :: ZF2L, ZF2H, ZSRFL, ZF1L, ZF1H
    REAL(JPRB) :: ZHSTRL, ZHSTRH, ZQSAIR, ZCOR
    REAL(JPRB) :: ZF3L, ZF3H
    REAL(JPRB), PARAMETER :: ZPWET_INF = 1.0E+6_JPRB

    ! Publish YOETHF / YOMCST scalars so the fcttre module functions see them.
    R2ES_M = R2ES; R3LES_M = R3LES; R3IES_M = R3IES
    R4LES_M = R4LES; R4IES_M = R4IES
    R5LES_M = R5LES; R5IES_M = R5IES; RTT_M = RTT
    RTF1_M = RTF1; RTF2_M = RTF2; RTF3_M = RTF3; RTF4_M = RTF4
    RETV_M = RETV_C; RD_M = RD_C

    DO JL = 1, KLON_C
      !* -- Liquid fraction of soil water (vsurf 301-319) -------------- *
      DO JK = 1, KLEVS_C
        IF (PTSAM1M(JL, JK) < RTF1 .AND. PTSAM1M(JL, JK) > RTF2) THEN
          ZF(JK) = 0.5_JPRB * (1.0_JPRB - SIN(RTF4 * (PTSAM1M(JL, JK) - RTF3)))
        ELSEIF (PTSAM1M(JL, JK) <= RTF2) THEN
          ZF(JK) = 1.0_JPRB
        ELSE
          ZF(JK) = 0.0_JPRB
        ENDIF
        ! LEVGEN branch (offline driver default).
        JS = KSOTY(JL)
        ZLIQ (JK) = MAX(RWPWPM(JS), MIN(RWCAPM(JS), PWSAM1M(JL, JK) * (1.0_JPRB - ZF(JK))))
        ZLIQR(JK) = MAX(RWRESTM(JS), MIN(RWCAPM(JS), PWSAM1M(JL, JK) * (1.0_JPRB - ZF(JK))))
      ENDDO

      !* -- Per-column veg lookups (vsurf 325-330, 384-385) ------------ *
      ZRSMINL = RVRSMIN(KTVL(JL))
      ZRSMINH = RVRSMIN(KTVH(JL))
      ZLAIL   = PLAIL(JL)
      ZLAIH   = PLAIH(JL)
      ZHSTRL  = RVHSTR(KTVL(JL))
      ZHSTRH  = RVHSTR(KTVH(JL))

      !* -- Bare-soil fraction ZBARE (vsurf 333-336) ------------------- *
      ZBARE = PFRTI(JL, 8)
      IF (NTILES_C > 9) ZBARE = ZBARE + PFRTI(JL, 9)

      !* -- Root-weighted soil moisture (vsurf 338-343) ---------------- *
      ZWROOTL = 0.0_JPRB
      ZWROOTH = 0.0_JPRB
      DO JK = 1, KLEVS_C
        ZWROOTL = ZWROOTL + ZLIQ(JK) * RVROOTSA(JK, KTVL(JL))
        ZWROOTH = ZWROOTH + ZLIQ(JK) * RVROOTSA(JK, KTVH(JL))
      ENDDO

      !* -- Bare-soil F2 (vsurf 344-363, LEVGEN branch) ---------------- *
      JS = KSOTY(JL)
      ZWPWP = RWPWPM(JS)
      ZQWEVAP = RQWEVAPM(JS)
      ZWPBARE = RWPWPM(JS) * (1.0_JPRB - ZBARE) + RWRESTM(JS) * ZBARE
      IF (JS >= 1) THEN
        ZQWEVAPBARE = 1.0_JPRB / (RWCAPM(JS) - ZWPBARE)
      ELSE
        ZQWEVAPBARE = 0.0_JPRB
      ENDIF
      ZF2B = MAX(RCEPSW, MIN(1.0_JPRB, (ZLIQR(1) - ZWPBARE) * ZQWEVAPBARE))

      !* -- Transpiration F2 with freeze shutdown (vsurf 365-372) ------ *
      IF (PTSAM1M(JL, 1) <= RTT) THEN
        ZF2L = RCEPSW
        ZF2H = RCEPSW
      ELSE
        ZF2L = MAX(RCEPSW, MIN(1.0_JPRB, (ZWROOTL - ZWPWP) * ZQWEVAP))
        ZF2H = MAX(RCEPSW, MIN(1.0_JPRB, (ZWROOTH - ZWPWP) * ZQWEVAP))
      ENDIF

      !* -- Radiation stress F1 (Betts; vsurf 378-381) ----------------- *
      ZSRFL = PSRFD(JL) / 250.0_JPRB
      ZF1L = 1.0_JPRB / MAX(1.0_JPRB, 0.81_JPRB * (1.0_JPRB + ZSRFL) / (ZSRFL + 0.05_JPRB))
      ZF1H = ZF1L

      !* -- VPD stress F3 (vsurf 383-392) ------------------------------ *
      ZQSAIR = FOEEW_M(PTMLEV(JL)) / PAPHMS(JL)
      ZCOR = 1.0_JPRB / (1.0_JPRB - RETV_C * ZQSAIR)
      ZQSAIR = ZQSAIR * ZCOR
      ZF3L = EXP(-ZHSTRL * (ZQSAIR - PQMLEV(JL)))
      ZF3H = EXP(-ZHSTRH * (ZQSAIR - PQMLEV(JL)))
      ZF3L = MAX(ZEPSF3, MIN(1.0_JPRB, ZF3L))
      ZF3H = MAX(ZEPSF3, MIN(1.0_JPRB, ZF3H))

      !* -- Resistances (vsurf 394-410) -------------------------------- *
      IF (ZLAIL /= 0.0_JPRB) THEN
        PWETL(JL) = ZRSMINL / (ZLAIL * ZF1L * ZF2L * ZF3L)
      ELSE
        PWETL(JL) = ZPWET_INF
      ENDIF
      PWETLU(JL) = RCLU_C / (ZF1L * ZF3L)
      IF (ZLAIH /= 0.0_JPRB) THEN
        PWETH(JL) = ZRSMINH / (ZLAIH * ZF1H * ZF2H * ZF3H)
      ELSE
        PWETH(JL) = ZPWET_INF
      ENDIF
      PWETHS(JL) = PWETH(JL)
      PWETB(JL) = RSMINB / ZF2B

      !* -- Diagnostics (vsurf 412-421) -------------------------------- *
      PEXDIAG(JL, 1)  = PWETL(JL)
      PEXDIAG(JL, 2)  = ZF1L
      PEXDIAG(JL, 3)  = ZF2L
      PEXDIAG(JL, 4)  = ZF3L
      PEXDIAG(JL, 5)  = ZLAIL
      PEXDIAG(JL, 6)  = PWETH(JL)
      PEXDIAG(JL, 7)  = ZF1H
      PEXDIAG(JL, 8)  = ZF2H
      PEXDIAG(JL, 9)  = ZF3H
      PEXDIAG(JL, 10) = ZLAIH
    ENDDO
  END SUBROUTINE SURF_REF_STOMATA

  !=====================================================================!
  ! SURF_REF_VEVAP -- equivalent evapotranspiration efficiency per tile.
  !
  ! Faithful, line-for-line transcription of vevap_mod.F90 (138-269).
  ! Called once per tile with a single KTILE selector; the Python ctypes
  ! bridge loops over the active tiles, exactly as the upstream driver
  ! (surfexcdriver) calls VEVAP inside its per-tile loop.
  !
  ! Rho convention: PCFQ here is the SEB's rho-scaled exchange coefficient
  ! (rhocqu = rho*Cq*|U|, kg/m^2/s), matching the upstream ASSOCIATE
  ! PCFQ=>... that comes from VEXCS.  The ZRHO factor is recomputed inside
  ! the vegetated branches to convert PWET (kg/m^2/s ... /rho = s/m) back
  ! to the alpha denominator -- so the final PCSAT = 1/(1 + Cq*|U|*PWET)
  ! carries NO extra 1/rho.  This is the fidelity fix: the old
  ! surf_pytorch build_tile_alpha used resistance_to_alpha which divided
  ! by rho twice.
  !
  ! Outputs:
  !   PCPTS   -- dry static energy at surface (uses stressed PCSAT/PCAIR)
  !   PCPTSU  -- dry static energy, unstressed (uses PCSATU/PCAIRU)
  !   PCSAT   -- multiplier of qs at the surface          -> alpha_s
  !   PCAIR   -- multiplier of q at the lowest level       -> alpha_l
  !   PCSATU  -- as PCSAT, unstressed low veg
  !   PCAIRU  -- as PCAIR, unstressed low veg
  !   PCSNW   -- snow-through-canopy moisture factor (KTILE=7 only)
  !   PDHVEGS -- canopy diagnostics (KLON, 2, 4)
  !=====================================================================!
  SUBROUTINE SURF_REF_VEVAP( &
      &   KLON_C, KTILE_C, &
      &   PTMST, PWLMX, PTMLEV, PQMLEV, PAPHMS, PTSKM1M, PTSAM1M, &
      &   PQS, PCFQ, PWETB, PWETL, PWETLU, PWETH, PWETHS, &
      &   RCPD_C, RD_C, RETV_C, RVTMP2_C, RLHAERO_C, RLHAEROS_C, &
      &   PCPTS, PCSAT, PCAIR, PCPTSU, PCSATU, PCAIRU, PCSNW, PDHVEGS &
      & ) BIND(C, NAME="surf_ref_vevap")

    !--- dimensions ------------------------------------------------------
    INTEGER(C_INT), VALUE, INTENT(IN) :: KLON_C, KTILE_C

    !--- inputs ----------------------------------------------------------
    REAL(JPRB), VALUE, INTENT(IN) :: PTMST
    REAL(JPRB), INTENT(IN)  :: PWLMX (KLON_C)
    REAL(JPRB), INTENT(IN)  :: PTMLEV(KLON_C), PQMLEV(KLON_C), PAPHMS(KLON_C)
    REAL(JPRB), INTENT(IN)  :: PTSKM1M(KLON_C), PTSAM1M(KLON_C)
    REAL(JPRB), INTENT(IN)  :: PQS   (KLON_C)
    REAL(JPRB), INTENT(IN)  :: PCFQ  (KLON_C)
    REAL(JPRB), INTENT(IN)  :: PWETB (KLON_C), PWETL (KLON_C)
    REAL(JPRB), INTENT(IN)  :: PWETLU(KLON_C), PWETH (KLON_C), PWETHS(KLON_C)
    ! scalars (YDCST / YDVEG ASSOCIATE targets)
    REAL(JPRB), VALUE, INTENT(IN) :: RCPD_C, RD_C, RETV_C, RVTMP2_C
    REAL(JPRB), VALUE, INTENT(IN) :: RLHAERO_C, RLHAEROS_C

    !--- outputs ---------------------------------------------------------
    REAL(JPRB), INTENT(OUT) :: PCPTS (KLON_C), PCSAT (KLON_C)
    REAL(JPRB), INTENT(OUT) :: PCAIR (KLON_C), PCPTSU(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PCSATU(KLON_C), PCAIRU(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PCSNW (KLON_C)
    REAL(JPRB), INTENT(OUT) :: PDHVEGS(KLON_C, 2, 4)

    !--- locals (vevap_mod.F90 121-124, 134-140) -------------------------
    INTEGER  :: JL, KTILE
    REAL(JPRB) :: ZZWET, ZZWETS, ZRAS, ZEP, ZEMAX, ZRHO, ZZWETU

    KTILE = KTILE_C

    !* -- 2.  Equivalent efficiency for evaporation (vevap 144-150) ----- *
    PCSAT (1:KLON_C) = 1.0_JPRB
    PCSATU(1:KLON_C) = 1.0_JPRB
    PCAIR (1:KLON_C) = 1.0_JPRB
    PCAIRU(1:KLON_C) = 1.0_JPRB
    PCSNW (1:KLON_C) = 0.0_JPRB

    !* -- interception reservoir (KTILE=3, vevap 154-165) --------------- *
    IF (KTILE == 3) THEN
      DO JL = 1, KLON_C
        ZEP   = PCFQ(JL) * (PQMLEV(JL) - PQS(JL))
        ZEMAX = -PWLMX(JL) / PTMST
        IF (ZEP < 0.0_JPRB .AND. ZEMAX < 0.0_JPRB .AND. ZEP < ZEMAX) THEN
          PCAIR (JL) = MIN(1.0_JPRB, ZEMAX/ZEP)
          PCSAT (JL) = PCAIR(JL)
          PCAIRU(JL) = PCAIR(JL)
          PCSATU(JL) = PCSAT(JL)
        ENDIF
      ENDDO
    ENDIF

    !* -- low vegetation (KTILE=4, vevap 169-185) ----------------------- *
    IF (KTILE == 4) THEN
      DO JL = 1, KLON_C
        IF (PQMLEV(JL) <= PQS(JL)) THEN
          ZRHO = PAPHMS(JL) / (RD_C*PTMLEV(JL)*(1.0_JPRB + RETV_C*PQMLEV(JL)))
          ZZWET  = PWETL (JL) / ZRHO
          ZZWETU = PWETLU(JL) / ZRHO
          PCSAT (JL) = 1.0_JPRB / (1.0_JPRB + PCFQ(JL)*ZZWET)
          PCAIR (JL) = 1.0_JPRB / (1.0_JPRB + PCFQ(JL)*ZZWET)
          PCSATU(JL) = 1.0_JPRB / (1.0_JPRB + PCFQ(JL)*ZZWETU)
          PCAIRU(JL) = 1.0_JPRB / (1.0_JPRB + PCFQ(JL)*ZZWETU)

          PDHVEGS(JL,1,3) = PWETL(JL)
          PDHVEGS(JL,1,4) = ZRHO / PCFQ(JL)
        ENDIF
      ENDDO
    ENDIF

    !* -- high vegetation, snow-free (KTILE=6, vevap 189-203) ----------- *
    IF (KTILE == 6) THEN
      DO JL = 1, KLON_C
        IF (PQMLEV(JL) <= PQS(JL)) THEN
          ZRHO = PAPHMS(JL) / (RD_C*PTMLEV(JL)*(1.0_JPRB + RETV_C*PQMLEV(JL)))
          ZZWET = PWETH(JL) / ZRHO
          PCSAT (JL) = 1.0_JPRB / (1.0_JPRB + PCFQ(JL)*ZZWET)
          PCAIR (JL) = PCSAT(JL)
          PCAIRU(JL) = PCAIR(JL)
          PCSATU(JL) = PCSAT(JL)

          PDHVEGS(JL,2,3) = PWETH(JL)
          PDHVEGS(JL,2,4) = ZRHO / PCFQ(JL)
        ENDIF
      ENDDO
    ENDIF

    !* -- high vegetation with underlying snow (KTILE=7, vevap 207-227) - *
    IF (KTILE == 7) THEN
      DO JL = 1, KLON_C
        IF (PQMLEV(JL) <= PQS(JL)) THEN
          IF (PTSKM1M(JL) > PTSAM1M(JL)) THEN
            ZRAS = 1200._JPRB / RLHAEROS_C
          ELSE
            ZRAS = 1200._JPRB / RLHAERO_C
          ENDIF
          ZRHO = PAPHMS(JL) / (RD_C*PTMLEV(JL)*(1.0_JPRB + RETV_C*PQMLEV(JL)))
          ZZWET  = PWETHS(JL) / ZRHO
          ZZWETS = ZRAS / ZRHO
          PCSAT(JL) = 1.0_JPRB / (1.0_JPRB + ZZWET*PCFQ(JL) + ZZWET/ZZWETS)
          PCAIR(JL) = PCSAT(JL)
          IF (PWETHS(JL) > 1.0_JPRB) &
           & PCSNW(JL) = 1.0_JPRB / (1.0_JPRB + ZZWETS*PCFQ(JL) + ZZWETS/ZZWET)
          PCAIRU(JL) = PCAIR(JL)
          PCSATU(JL) = PCSAT(JL)
        ENDIF
      ENDDO
    ENDIF

    !* -- bare soil (KTILE=8, vevap 231-242) ---------------------------- *
    IF (KTILE == 8) THEN
      DO JL = 1, KLON_C
        IF (PQMLEV(JL) <= PQS(JL)) THEN
          ZRHO = PAPHMS(JL) / (RD_C*PTMLEV(JL)*(1.0_JPRB + RETV_C*PQMLEV(JL)))
          ZZWET = PWETB(JL) / ZRHO
          PCSAT (JL) = 1.0_JPRB / (1.0_JPRB + PCFQ(JL)*ZZWET)
          PCAIR (JL) = PCSAT(JL)
          PCAIRU(JL) = PCAIR(JL)
          PCSATU(JL) = PCSAT(JL)
        ENDIF
      ENDDO
    ENDIF

    !* -- urban (KTILE=10, vevap 246-257) ------------------------------- *
    IF (KTILE == 10) THEN
      DO JL = 1, KLON_C
        IF (PQMLEV(JL) <= PQS(JL)) THEN
          ZRHO = PAPHMS(JL) / (RD_C*PTMLEV(JL)*(1.0_JPRB + RETV_C*PQMLEV(JL)))
          ZZWET = PWETB(JL) / ZRHO   ! RURBRES not currently used
          PCSAT (JL) = 1.0_JPRB / (1.0_JPRB + PCFQ(JL)*ZZWET)
          PCAIR (JL) = PCSAT(JL)
          PCAIRU(JL) = PCAIR(JL)
          PCSATU(JL) = PCSAT(JL)
        ENDIF
      ENDDO
    ENDIF

    !* -- dry static energy (vevap 260-265; unconditional per tile) ----- *
    DO JL = 1, KLON_C
      PCPTS(JL) = PTSKM1M(JL) * RCPD_C * (1.0_JPRB + RVTMP2_C * &
       &   (PCSAT(JL)*PQS(JL) + (1.0_JPRB - PCAIR(JL))*PQMLEV(JL)))
      PCPTSU(JL) = PTSKM1M(JL) * RCPD_C * (1.0_JPRB + RVTMP2_C * &
       &   (PCSATU(JL)*PQS(JL) + (1.0_JPRB - PCAIRU(JL))*PQMLEV(JL)))
    ENDDO
  END SUBROUTINE SURF_REF_VEVAP


  !=====================================================================!
  ! SURF_REF_VDFDIFH -- offline single-column VDFDIFH1S wrapper.
  !
  ! Direct transcription of surf/offline/driver/vdfdifh1s.F90.  The
  ! offline driver comments out the entire Thomas-algorithm tridiagonal
  ! solve (!1s prefix) and replaces it with a Dirichlet boundary
  ! condition:
  !
  !   asl = 0,  aql = 0
  !   bsl = PSLM1(:,KLEV) + PTMST*PSLGE(:,KLEV)
  !   bql = PQTM1(:,KLEV) + PTMST*PQTE(:,KLEV)
  !
  ! so this wrapper simply packages the SURFSEB inputs from the lowest-
  ! model-level forcing, inverts the exchange coefficients (ZCOEF1 =
  ! 1/(RG*RVDIFTS*PTMST)), builds the per-tile PTSRF, calls SURF_REF_SEB
  ! (the canonical call #3), optionally repeats for LESNML snow melt
  ! (call #4), adds the snow-under-forest evaporation to tile 7, and
  ! extrapolates the skin T to t+1.
  !
  ! Calling SURF_REF_SEB from within this module procedure is valid:
  ! BIND(C) only affects the external ABI; from Fortran's perspective
  ! it is a regular module procedure callable via host association.
  !=====================================================================!
  SUBROUTINE SURF_REF_VDFDIFH( &
      &   KLON_C, NTILES_C, &
      &   PSLM1_LEV, PQTM1_LEV, PSLGE_LEV, PQTE_LEV, &
      &   PCFHTI, PCFQTI, &
      &   PSST, PTICE, PTSM1M, PTSNOW, &
      &   SSKM1M, TSKM1M, QSKM1M, DQSDT, &
      &   ALPHAL, ALPHAS, &
      &   PSSRFLTI, PSLRFL, PTSKRAD, PEMIS, &
      &   PLAMSK, PFRTI, KTVL, KTVH, &
      &   PTMST, PEVAPSNW, &
      &   PHLICE, PTLICE, PTLWML, &
      &   RVTRSR, &
      &   RCPD, RLSTT, RLVTT, RSIGMA, RTT, RVTMP2, RG, RVDIFTS, &
      &   LESNML, HAS_EVAPSNW, LELWDD, LELWTL, &
      &   PJQ, PSSH, PSLH, PSTR, PG0, PTSK, PTSKTIP1, PSL, PQL &
      & ) BIND(C, NAME="surf_ref_vdfdifh")

    !--- dimensions ------------------------------------------------------
    INTEGER(C_INT), VALUE, INTENT(IN) :: KLON_C, NTILES_C
    !--- lowest-level forcing (1D) ---------------------------------------
    REAL(JPRB), INTENT(IN) :: PSLM1_LEV(KLON_C), PQTM1_LEV(KLON_C)
    REAL(JPRB), INTENT(IN) :: PSLGE_LEV(KLON_C), PQTE_LEV(KLON_C)
    !--- rescaled exchange coefficients (2D) -----------------------------
    REAL(JPRB), INTENT(IN) :: PCFHTI(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: PCFQTI(KLON_C, NTILES_C)
    !--- lower-boundary temperatures (1D) --------------------------------
    REAL(JPRB), INTENT(IN) :: PSST(KLON_C), PTICE(KLON_C)
    REAL(JPRB), INTENT(IN) :: PTSM1M(KLON_C), PTSNOW(KLON_C)
    !--- skin reservoir (2D) ---------------------------------------------
    REAL(JPRB), INTENT(IN) :: SSKM1M(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: TSKM1M(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: QSKM1M(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: DQSDT (KLON_C, NTILES_C)
    !--- moisture closure (2D) -------------------------------------------
    REAL(JPRB), INTENT(IN) :: ALPHAL(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: ALPHAS(KLON_C, NTILES_C)
    !--- radiation -------------------------------------------------------
    REAL(JPRB), INTENT(IN) :: PSSRFLTI(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: PSLRFL(KLON_C)
    REAL(JPRB), INTENT(IN) :: PTSKRAD(KLON_C), PEMIS(KLON_C)
    !--- skin conductivity / tile fractions / veg (2D + 1D) --------------
    REAL(JPRB), INTENT(IN) :: PLAMSK(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: PFRTI  (KLON_C, NTILES_C)
    INTEGER(C_INT), INTENT(IN) :: KTVL(KLON_C), KTVH(KLON_C)
    !--- timestep / snow-under-forest evap / lake (1D) -------------------
    REAL(JPRB), VALUE, INTENT(IN) :: PTMST
    REAL(JPRB), INTENT(IN) :: PEVAPSNW(KLON_C)
    REAL(JPRB), INTENT(IN) :: PHLICE(KLON_C), PTLICE(KLON_C), PTLWML(KLON_C)
    !--- LUT + scalars ---------------------------------------------------
    REAL(JPRB), INTENT(IN)  :: RVTRSR(0:20)
    REAL(JPRB), VALUE, INTENT(IN) :: RCPD, RLSTT, RLVTT, RSIGMA, RTT, RVTMP2
    REAL(JPRB), VALUE, INTENT(IN) :: RG, RVDIFTS
    LOGICAL(C_BOOL), VALUE, INTENT(IN) :: LESNML, HAS_EVAPSNW, LELWDD, LELWTL

    !--- outputs ---------------------------------------------------------
    REAL(JPRB), INTENT(OUT) :: PJQ (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PSSH(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PSLH(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PSTR(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PG0 (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PTSK(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PTSKTIP1(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(OUT) :: PSL (KLON_C)
    REAL(JPRB), INTENT(OUT) :: PQL (KLON_C)

    !--- locals ----------------------------------------------------------
    REAL(JPRB) :: ZASL(KLON_C), ZBSL(KLON_C), ZAQL(KLON_C), ZBQL(KLON_C)
    REAL(JPRB) :: ZTSRF(KLON_C, NTILES_C)
    REAL(JPRB) :: ZTSRF_R(KLON_C, NTILES_C), ZLAMSK_R(KLON_C, NTILES_C)
    REAL(JPRB) :: ZRHOCHU(KLON_C, NTILES_C), ZRHOCQU(KLON_C, NTILES_C)
    REAL(JPRB) :: ZJS(KLON_C, NTILES_C), ZSSK(KLON_C, NTILES_C)
    REAL(JPRB) :: ZTSK(KLON_C, NTILES_C)
    REAL(JPRB) :: ZCOEF1, ZTPFAC2, ZTPFAC3
    REAL(JPRB), PARAMETER :: ZHICE_MIN = 1.0E-9_JPRB
    REAL(JPRB), PARAMETER :: ZFRAC_MIN = 1.0E-6_JPRB
    REAL(JPRB), PARAMETER :: ZTSK_EPS  = 1.0E-6_JPRB
    REAL(JPRB), PARAMETER :: ZLAMSK_MELT = 100.0_JPRB
    LOGICAL :: LREPEAT, LESNML_F, HAS_EVAPSNW_F, LELWDD_F, LELWTL_F
    INTEGER :: JL, JT

    LESNML_F     = LESNML
    HAS_EVAPSNW_F = HAS_EVAPSNW
    LELWDD_F     = LELWDD
    LELWTL_F     = LELWTL

    !--- Step 1.0-1.5: Dirichlet boundary condition (offline setup) -----
    DO JL = 1, KLON_C
      ZASL(JL) = 0.0_JPRB
      ZBSL(JL) = PSLM1_LEV(JL) + PTMST * PSLGE_LEV(JL)
      ZAQL(JL) = 0.0_JPRB
      ZBQL(JL) = PQTM1_LEV(JL) + PTMST * PQTE_LEV(JL)
    END DO

    !--- Step 1.6: per-tile sub-surface temperature PTSRF ---------------
    ! (vdfdifh1s.F90 lines 372-388).  Fortran 1-based JT.
    ZTSRF(:, 1) = PSST(1:KLON_C)
    IF (NTILES_C >= 2) ZTSRF(:, 2) = PTICE(1:KLON_C)
    IF (NTILES_C >= 3) ZTSRF(:, 3) = PTSM1M(1:KLON_C)
    IF (NTILES_C >= 4) ZTSRF(:, 4) = PTSM1M(1:KLON_C)
    IF (NTILES_C >= 5) ZTSRF(:, 5) = PTSNOW(1:KLON_C)
    IF (NTILES_C >= 6) ZTSRF(:, 6) = PTSM1M(1:KLON_C)
    IF (NTILES_C >= 7) ZTSRF(:, 7) = PTSNOW(1:KLON_C)
    IF (NTILES_C >= 8) ZTSRF(:, 8) = PTSM1M(1:KLON_C)
    IF (NTILES_C >= 9) THEN
      DO JL = 1, KLON_C
        IF (PHLICE(JL) > ZHICE_MIN) THEN
          ZTSRF(JL, 9) = PTLICE(JL)
        ELSE
          ZTSRF(JL, 9) = PTLWML(JL)
        END IF
      END DO
    END IF

    !--- Step 1.6b: exchange-coefficient inversion (ZCOEF1) -------------
    ZCOEF1 = 1.0_JPRB / (RG * RVDIFTS * PTMST)
    DO JT = 1, NTILES_C
      DO JL = 1, KLON_C
        ZRHOCHU(JL, JT) = PCFHTI(JL, JT) * ZCOEF1
        ZRHOCQU(JL, JT) = PCFQTI(JL, JT) * ZCOEF1
      END DO
    END DO

    !--- Step 1.7: SURFSEB canonical call (#3) --------------------------
    CALL SURF_REF_SEB(KLON_C, NTILES_C, &
      &   SSKM1M, TSKM1M, QSKM1M, DQSDT, &
      &   ZRHOCHU, ZRHOCQU, ALPHAL, ALPHAS, &
      &   PSSRFLTI, PSLRFL, PTSKRAD, PEMIS, &
      &   ZTSRF, PLAMSK, PFRTI, &
      &   ZASL, ZBSL, ZAQL, ZBQL, &
      &   PTMST, RVTRSR, &
      &   RCPD, RLSTT, RLVTT, RSIGMA, RTT, RVTMP2, &
      &   LELWDD, LELWTL, &
      &   ZJS, PJQ, ZSSK, ZTSK, PSSH, PSLH, PSTR, PG0, PSL, PQL)

    !--- LESNML snow-melt repeat (call #4, lines 518-547) ---------------
    ! When any column has PFRTI(:,5) >= 1e-6 AND ZTSK(:,5) >= RTT-1e-6,
    ! pin that tile's TSRF to RTT and LAMSK to 100 for those columns
    ! and re-run SURFSEB on the whole slab.
    IF (LESNML_F .AND. NTILES_C >= 5) THEN
      LREPEAT = .FALSE.
      DO JL = 1, KLON_C
        IF (PFRTI(JL, 5) >= ZFRAC_MIN .AND. &
          & ZTSK(JL, 5) >= RTT - ZTSK_EPS) THEN
          LREPEAT = .TRUE.
          EXIT
        END IF
      END DO
      IF (LREPEAT) THEN
        DO JT = 1, NTILES_C
          DO JL = 1, KLON_C
            ZTSRF_R(JL, JT)  = ZTSRF(JL, JT)
            ZLAMSK_R(JL, JT) = PLAMSK(JL, JT)
          END DO
        END DO
        DO JL = 1, KLON_C
          IF (PFRTI(JL, 5) >= ZFRAC_MIN .AND. &
            & ZTSK(JL, 5) >= RTT - ZTSK_EPS) THEN
            ZTSRF_R(JL, 5)  = RTT
            ZLAMSK_R(JL, 5) = ZLAMSK_MELT
          END IF
        END DO
        CALL SURF_REF_SEB(KLON_C, NTILES_C, &
          &   SSKM1M, TSKM1M, QSKM1M, DQSDT, &
          &   ZRHOCHU, ZRHOCQU, ALPHAL, ALPHAS, &
          &   PSSRFLTI, PSLRFL, PTSKRAD, PEMIS, &
          &   ZTSRF_R, ZLAMSK_R, PFRTI, &
          &   ZASL, ZBSL, ZAQL, ZBQL, &
          &   PTMST, RVTRSR, &
          &   RCPD, RLSTT, RLVTT, RSIGMA, RTT, RVTMP2, &
          &   LELWDD, LELWTL, &
          &   ZJS, PJQ, ZSSK, ZTSK, PSSH, PSLH, PSTR, PG0, PSL, PQL)
      END IF
    END IF

    !--- Step 1.7a: snow-under-forest evaporation (tile JT=7) -----------
    ! (vdfdifh1s lines 557-558)
    IF (HAS_EVAPSNW_F .AND. NTILES_C >= 7) THEN
      DO JL = 1, KLON_C
        PJQ (JL, 7) = PJQ (JL, 7) + PEVAPSNW(JL)
        PSLH(JL, 7) = PSLH(JL, 7) + PEVAPSNW(JL) * RLSTT
      END DO
    END IF

    !--- Step 1.7c: skin T extrapolation to t+1 -------------------------
    ! (vdfdifh1s lines 585-590)
    ZTPFAC2 = 1.0_JPRB / RVDIFTS
    ZTPFAC3 = 1.0_JPRB - ZTPFAC2
    DO JT = 1, NTILES_C
      DO JL = 1, KLON_C
        PTSK(JL, JT)     = ZTSK(JL, JT)
        PTSKTIP1(JL, JT) = ZTPFAC2 * ZTSK(JL, JT) + ZTPFAC3 * TSKM1M(JL, JT)
      END DO
    END DO
  END SUBROUTINE SURF_REF_VDFDIFH

  !=====================================================================!
  ! SURF_REF_SOILT -- HTESSEL soil-temperature update.                  !
  !                                                                     !
  ! Standalone transcription of the soil-temperature branch of          !
  ! SURFTSTP_CTL section 3, comprising three upstream routines:         !
  !   * SRFRCGS  (apparent volumetric heat capacity incl. freeze/melt)  !
  !   * SRFTS    (tridiagonal coefficient assembly + Peters-Liddard     !
  !              thermal conductivity + net surface heat flux)          !
  !   * SRFWDIFS (Thomas tridiagonal solver)                            !
  !                                                                     !
  ! Mirrors surf_pytorch.physics.soil.compute_soil_temperature() so     !
  ! the comparison test (tests/test_soil_ref.py::TestSoilRefSoilt) can  !
  ! feed identical inputs to both implementations and compare ULP.      !
  !                                                                     !
  ! Cold-start simplification: RSIMP=1 -> ZCONS2=0 (fully implicit),    !
  ! ZFF=0 (no freezing reduction on the conductivity).  The lake and   !
  ! sea-ice branches are not transcribed (LEFLAKE=.FALSE. in the        !
  ! offline driver; sea ice routes through SRFI).                       !
  !                                                                     !
  ! Inputs (KLON_C x KLEVS_C = Ngrid x Nsoil):                          !
  !   state   PTSAM1M, PWSAM1M (soil T, moisture at t-1)                !
  !   fluxes  PFRTI, PAHFSTI, PEVAPTI, PSSRFLTI (KLON x NTILES)         !
  !           PSLRFL (KLON), PGSN (KLON)                                !
  !   static  KSOTY (KLON), PCVL (KLON), PCVH (KLON), LDLAND (KLON)     !
  !   LUT     per-soil-type columns gathered by the caller:             !
  !             RWSATM, RLAMBDADRYM, RLAMSAT1M, RRCSOILM, RWCAPM (NSOTY)!
  !   scalars PTMST, RSIMP, RLMLT, RHOH2O, RLVTT, RTF1..RTF4,           !
  !           RLAMBDAICE, RLAMBDAWAT, RKERST1..3, NTILES_C, NSOTY_C     !
  ! Outputs:                                                            !
  !   PTSA (KLON x KLEVS), PCTSA (KLON x KLEVS), PZSURFL (KLON)         !
  !                                                                     !
  ! Author: surf-pytorch reference driver (M5c).  Algorithm (c) ECMWF.  !
  !=====================================================================!
  SUBROUTINE SURF_REF_SOILT( &
      &   KLON_C, KLEVS_C, NTILES_C, NSOTY_C, &
      &   PTSAM1M, PWSAM1M, &
      &   PFRTI, PAHFSTI, PEVAPTI, PSSRFLTI, &
      &   PSLRFL, PGSN, &
      &   KSOTY, PCVL, PCVH, LDLAND, &
      &   RDAT, &
      &   RWSATM, RLAMBDADRYM, RLAMSAT1M, RRCSOILM, RWCAPM, &
      &   PTMST, RSIMP, RLMLT, RHOH2O, RLVTT, RTF1, RTF2, RTF3, RTF4, &
      &   RLAMBDAICE, RLAMBDAWAT, RKERST1, RKERST2, RKERST3, &
      &   PTSA, PCTSA, PZSURFL &
      & ) BIND(C, NAME="surf_ref_soilt")

    !--- dimensions ------------------------------------------------------
    INTEGER(C_INT), VALUE, INTENT(IN) :: KLON_C, KLEVS_C, NTILES_C, NSOTY_C
    !--- soil state at t-1 (KLON x KLEVS) --------------------------------
    REAL(JPRB), INTENT(IN) :: PTSAM1M(KLON_C, KLEVS_C)
    REAL(JPRB), INTENT(IN) :: PWSAM1M(KLON_C, KLEVS_C)
    !--- per-tile fluxes (KLON x NTILES) ---------------------------------
    REAL(JPRB), INTENT(IN) :: PFRTI  (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: PAHFSTI(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: PEVAPTI(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: PSSRFLTI(KLON_C, NTILES_C)
    !--- per-grid radiation / snow-bottom flux (KLON) --------------------
    REAL(JPRB), INTENT(IN) :: PSLRFL(KLON_C), PGSN(KLON_C)
    !--- static descriptors ----------------------------------------------
    INTEGER(C_INT), INTENT(IN) :: KSOTY(KLON_C)
    REAL(JPRB), INTENT(IN) :: PCVL(KLON_C), PCVH(KLON_C)
    LOGICAL(C_BOOL), INTENT(IN) :: LDLAND(KLON_C)
    !--- soil-layer thicknesses (KLEVS) ----------------------------------
    REAL(JPRB), INTENT(IN) :: RDAT(KLEVS_C)
    !--- per-soil-type LUT columns (NSOTY; 1-indexed) --------------------
    REAL(JPRB), INTENT(IN) :: RWSATM(NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RLAMBDADRYM(NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RLAMSAT1M(NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RRCSOILM(NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RWCAPM(NSOTY_C)
    !--- scalars ---------------------------------------------------------
    REAL(JPRB), VALUE, INTENT(IN) :: PTMST, RSIMP, RLMLT, RHOH2O, RLVTT
    REAL(JPRB), VALUE, INTENT(IN) :: RTF1, RTF2, RTF3, RTF4
    REAL(JPRB), VALUE, INTENT(IN) :: RLAMBDAICE, RLAMBDAWAT
    REAL(JPRB), VALUE, INTENT(IN) :: RKERST1, RKERST2, RKERST3
    !--- outputs ---------------------------------------------------------
    REAL(JPRB), INTENT(OUT) :: PTSA(KLON_C, KLEVS_C)
    REAL(JPRB), INTENT(OUT) :: PCTSA(KLON_C, KLEVS_C)
    REAL(JPRB), INTENT(OUT) :: PZSURFL(KLON_C)

    !--- locals ----------------------------------------------------------
    REAL(JPRB) :: ZCONS1, ZCONS2, ZSURFL, ZCDZ1
    REAL(JPRB) :: ZDFDT, ZWA, ZWU, ZLWT, ZLIC, ZLAMBDASAT
    REAL(JPRB) :: ZCOND, ZKERSTEN, ZDIF_TMP
    REAL(JPRB) :: ZDIF(KLEVS_C), ZCDZ(KLEVS_C), ZLST(KLEVS_C)
    REAL(JPRB) :: ZRHS(KLEVS_C), ZTSA(KLEVS_C)
    REAL(JPRB) :: ZTPFAC2, ZTCOE, ZQDP, ZDISC, ZFAC
    REAL(JPRB) :: ZEBSW(KLEVS_C), ZWSDIF(KLEVS_C)
    REAL(JPRB) :: ZSSRFL, ZSLRFL, ZTHFL
    REAL(JPRB) :: ZF3, ZF4, ZF6, ZF8
    INTEGER(C_INT) :: JL, JK, JS
    LOGICAL :: LLAND

    ! Constants reused inside the per-point loop.
    ZCONS1 = PTMST * RSIMP
    ZCONS2 = 1.0_JPRB - 1.0_JPRB / RSIMP

    ! Tile indices are 1-based here: wet-skin=3, low-veg-dry=4,
    ! high-veg-dry=6, bare-soil=8 -- matching IFS tile numbering.

    DO JL = 1, KLON_C
      LLAND = LDLAND(JL)
      ! KSOTY is 0-based in the Python SoilTable (0 = water sentinel,
      ! 1..7 = soil types); the per-class LUT columns passed in are the
      ! same 8-element buffers.  BIND(C) arrays are 1-based in Fortran,
      ! so we shift by +1 to index them with the Python convention.
      JS = KSOTY(JL) + 1

      IF (LLAND) THEN
        !-- SRFRCGS: apparent heat capacity PCTSA(JK) ------------------
        ! ZWA = (PCVL+PCVH) * RWCAPM(JS) is per-grid (no JK dependence).
        ZWA = (PCVL(JL) + PCVH(JL)) * RWCAPM(JS)
        DO JK = 1, KLEVS_C
          IF (PTSAM1M(JL,JK) > RTF2 .AND. PTSAM1M(JL,JK) < RTF1) THEN
            ZDFDT = -0.5_JPRB * RTF4 * COS(RTF4 * (PTSAM1M(JL,JK) - RTF3))
          ELSE
            ZDFDT = 0.0_JPRB
          ENDIF
          PCTSA(JL,JK) = RRCSOILM(JS) - RLMLT*RHOH2O*ZWA*ZDFDT
        END DO

        !-- SRFTS section 2: net surface heat flux ZSURFL --------------
        ZF3 = PFRTI(JL,3); ZF4 = PFRTI(JL,4)
        ZF6 = PFRTI(JL,6); ZF8 = PFRTI(JL,8)
        ZSSRFL = ZF3*PSSRFLTI(JL,3) + ZF4*PSSRFLTI(JL,4) &
           &   + ZF6*PSSRFLTI(JL,6) + ZF8*PSSRFLTI(JL,8)
        ZSLRFL = (ZF3 + ZF4 + ZF6 + ZF8) * PSLRFL(JL)
        ZTHFL = ZF3*(PAHFSTI(JL,3) + RLVTT*PEVAPTI(JL,3)) &
           & + ZF4*(PAHFSTI(JL,4) + RLVTT*PEVAPTI(JL,4)) &
           & + ZF6*(PAHFSTI(JL,6) + RLVTT*PEVAPTI(JL,6)) &
           & + ZF8*(PAHFSTI(JL,8) + RLVTT*PEVAPTI(JL,8))
        ZSURFL = ZSSRFL + ZSLRFL + ZTHFL + PGSN(JL)
      ELSE
        ! Surf points: PCTSA=0, ZSURFL=0 (SRFRCGS / SRFTS ELSE branches).
        DO JK = 1, KLEVS_C
          PCTSA(JL,JK) = 0.0_JPRB
        END DO
        ZSURFL = 0.0_JPRB
      END IF
      PZSURFL(JL) = ZSURFL

      IF (LLAND) THEN
        !-- SRFTS section 3: per-layer thermal conductivity ZDIF -------
        ! Peters-Liddard / Kersten formula.  ZFF=0 at cold start so
        ! ZWU = W*(1-ZFF) = W.
        DO JK = 1, KLEVS_C
          ZWU = PWSAM1M(JL,JK)
          ZLWT = RLAMBDAWAT ** ZWU
          ZLIC = RLAMBDAICE ** (RWSATM(JS) - ZWU)
          ZLAMBDASAT = RLAMSAT1M(JS) * ZLIC * ZLWT
          ZCOND = MAX(RKERST1, PWSAM1M(JL,JK) / RWSATM(JS))
          ZKERSTEN = RKERST2 * LOG10(ZCOND) + RKERST3
          ZDIF(JK) = RLAMBDADRYM(JS) + ZKERSTEN*(ZLAMBDASAT - RLAMBDADRYM(JS))
        END DO

        !-- SRFTS section 4: Thomas coefficient assembly ---------------
        DO JK = 1, KLEVS_C
          ZCDZ(JK) = PCTSA(JL,JK) * RDAT(JK)
          ZLST(JK) = 0.0_JPRB
          ZRHS(JK) = 0.0_JPRB
        END DO
        DO JK = 1, KLEVS_C - 1
          ZLST(JK) = ZCONS1 * (ZDIF(JK) + ZDIF(JK+1)) / (RDAT(JK) + RDAT(JK+1))
        END DO
        ! ZLST(KLEVS) stays 0 (zero-flux lower BC).
        ! Surface RHS injects the net surface flux into the top layer.
        ZCDZ1 = ZCDZ(1)
        IF (ZCDZ1 /= 0.0_JPRB) THEN
          ZRHS(1) = PTMST * ZSURFL / ZCDZ1
        ELSE
          ZRHS(1) = 0.0_JPRB
        END IF

        !-- SRFWDIFS: Thomas tridiagonal solve ------------------------
        ZTPFAC2 = 1.0_JPRB / RSIMP
        DO JK = 1, KLEVS_C
          ZWSDIF(JK) = ZTPFAC2 * PTSAM1M(JL,JK) + ZRHS(JK)
        END DO

        ! Top layer elimination (JK=1).
        ZTCOE = ZLST(1)
        ZQDP = 1.0_JPRB / ZCDZ(1)
        ZDISC = 1.0_JPRB / (1.0_JPRB + ZLST(1)*ZQDP)
        ZEBSW(1) = ZDISC * (ZLST(1)*ZQDP)
        ZWSDIF(1) = ZDISC * ZWSDIF(1)

        ! Middle layers JK=2..KLEVS-1.
        DO JK = 2, KLEVS_C - 1
          ZQDP = 1.0_JPRB / ZCDZ(JK)
          ZFAC = ZTCOE * ZQDP
          ZTCOE = ZLST(JK)
          ZDISC = 1.0_JPRB / (1.0_JPRB + ZFAC*(1.0_JPRB - ZEBSW(JK-1)) &
             &                        + ZLST(JK)*ZQDP)
          ZEBSW(JK) = ZDISC * (ZLST(JK)*ZQDP)
          ZWSDIF(JK) = ZDISC * (ZWSDIF(JK) + ZFAC*ZWSDIF(JK-1))
        END DO

        ! Bottom layer JK=KLEVS.
        IF (KLEVS_C >= 2) THEN
          JK = KLEVS_C
          ZQDP = 1.0_JPRB / ZCDZ(JK)
          ZFAC = ZTCOE * ZQDP
          ZDISC = 1.0_JPRB / (1.0_JPRB + ZFAC*(1.0_JPRB - ZEBSW(JK-1)) &
             &                          + ZLST(JK)*ZQDP)
          ZWSDIF(JK) = ZDISC * (ZWSDIF(JK) + ZFAC*ZWSDIF(JK-1))
        END IF

        ! Back-substitution JK=KLEVS-1..1.
        DO JK = KLEVS_C - 1, 1, -1
          ZWSDIF(JK) = ZWSDIF(JK) + ZEBSW(JK) * ZWSDIF(JK+1)
        END DO

        ! New temperatures.
        DO JK = 1, KLEVS_C
          PTSA(JL,JK) = PTSAM1M(JL,JK) * ZCONS2 + ZWSDIF(JK)
        END DO
      ELSE
        ! Surf points keep their t-1 temperature (LDLAND=.F. branch).
        DO JK = 1, KLEVS_C
          PTSA(JL,JK) = PTSAM1M(JL,JK)
        END DO
      END IF
    END DO
  END SUBROUTINE SURF_REF_SOILT


  !---------------------------------------------------------------------!
  ! SURF_REF_SNOW -- single-layer snow (SRFSN_LWIMPS) reference kernel. !
  ! --------------------------------------------------------------------!
  ! Standalone reimplementation of the SRFSN_LWIMPS single-layer snow   !
  ! scheme used by SURFTSTPS_CTL when NCSNEC=1 (offline-driver default).!
  ! Transcribed line-for-line from surf/module/srfsn_lwimps_mod.F90 so  !
  ! a ULP-level comparison against surf_pytorch.physics.snow.compute_   !
  ! snow() validates the full port: no-snow / no-melt / melt branches,  !
  ! implicit snow-T solve, snow-soil ground heat flux, liquid-water     !
  ! capacity and throughfall interception.                              !
  !                                                                     !
  ! Sign convention: PGSN positive downward (into soil).  PEVAPTI > 0   !
  ! = dew/deposition (mass gain), < 0 = evaporation/sublimation.        !
  !                                                                     !
  ! Tile indices are 1-based here matching IFS tile numbering: 5 = snow !
  ! on low vegetation, 7 = snow under high vegetation.                  !
  !---------------------------------------------------------------------!
  SUBROUTINE SURF_REF_SNOW( &
      &   KLON_C, NTILES_C, &
      &   PSSNM1M, PTSNM1M, PRSNM1M, PTSAM1M_TOP, &
      &   PFRTI, PAHFSTI, PEVAPTI, PSSRFLTI, &
      &   PSLRFL, &
      &   PSSFC, PSSFL, PTSFC, PTSFL, PEVAPSNW, &
      &   LEFLAKE_C, &
      &   PTMST, &
      &   RTT, RLMLT, RLVTT, RLSTT, RPI, RDAY, &
      &   RHOICE, RLAMICE, RHOCI, RALAMSN, RDSNMAX, &
      &   RFRSMALL, RFRTINY, &
      &   RLWCSWEA, RLWCSWEB, RLWCSWEC, RTEMPAMP, &
      &   RTAUF, RTAUA, &
      &   RVLAMSK_8, &
      &   PTSN, PGSN, PSSTAR, PLWC, PZFRSN &
      & ) BIND(C, NAME="surf_ref_snow")

    !--- dimensions ------------------------------------------------------
    INTEGER(C_INT), VALUE, INTENT(IN) :: KLON_C, NTILES_C
    !--- snow state at t-1 (KLON) ----------------------------------------
    REAL(JPRB), INTENT(IN) :: PSSNM1M(KLON_C)
    REAL(JPRB), INTENT(IN) :: PTSNM1M(KLON_C)
    REAL(JPRB), INTENT(IN) :: PRSNM1M(KLON_C)
    REAL(JPRB), INTENT(IN) :: PTSAM1M_TOP(KLON_C)
    !--- per-tile fluxes (KLON x NTILES) ---------------------------------
    REAL(JPRB), INTENT(IN) :: PFRTI  (KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: PAHFSTI(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: PEVAPTI(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: PSSRFLTI(KLON_C, NTILES_C)
    !--- per-grid radiation (KLON) ---------------------------------------
    REAL(JPRB), INTENT(IN) :: PSLRFL(KLON_C)
    !--- snowfall / rainfall throughfall + snow-under-forest evap (KLON) -
    REAL(JPRB), INTENT(IN) :: PSSFC(KLON_C), PSSFL(KLON_C)
    REAL(JPRB), INTENT(IN) :: PTSFC(KLON_C), PTSFL(KLON_C)
    REAL(JPRB), INTENT(IN) :: PEVAPSNW(KLON_C)
    !--- FLake flag ------------------------------------------------------
    LOGICAL(C_BOOL), VALUE, INTENT(IN) :: LEFLAKE_C
    !--- scalar timestep -------------------------------------------------
    REAL(JPRB), VALUE, INTENT(IN) :: PTMST
    !--- physical constants ----------------------------------------------
    REAL(JPRB), VALUE, INTENT(IN) :: RTT, RLMLT, RLVTT, RLSTT, RPI, RDAY
    !--- snow-table constants --------------------------------------------
    REAL(JPRB), VALUE, INTENT(IN) :: RHOICE, RLAMICE, RHOCI, RALAMSN
    REAL(JPRB), VALUE, INTENT(IN) :: RDSNMAX, RFRSMALL, RFRTINY
    REAL(JPRB), VALUE, INTENT(IN) :: RLWCSWEA, RLWCSWEB, RLWCSWEC, RTEMPAMP
    REAL(JPRB), VALUE, INTENT(IN) :: RTAUF, RTAUA
    REAL(JPRB), VALUE, INTENT(IN) :: RVLAMSK_8
    !--- outputs ---------------------------------------------------------
    REAL(JPRB), INTENT(OUT) :: PTSN(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PGSN(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PSSTAR(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PLWC(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PZFRSN(KLON_C)

    !--- locals ----------------------------------------------------------
    LOGICAL :: LLNOSNOW, LEFLAKE
    REAL(JPRB) :: ZFRSN, ZIFRSN, ZGRIDFRAC
    REAL(JPRB) :: ZSSFC, ZSSFL, ZTSFC, ZTSFL
    REAL(JPRB) :: ZHFLUXP, ZHFLUX, ZSSTAR, ZSSTAR1
    REAL(JPRB) :: ZDSNR, ZDSNRP, ZCONSLWC, ZLWC
    REAL(JPRB) :: ZFUNC, ZDFUNC, ZCONSB, ZCONSAMAX, ZCOND1, ZCONSA, ZCONS2
    REAL(JPRB) :: ZDSN, ZSNRES, ZRS, ZSOILRES
    REAL(JPRB) :: ZWFLUXRF, ZCOND3, ZCOND4, ZCOND5, ZCOND6, ZHFLUXRF1
    REAL(JPRB) :: ZTSTARI, ZTSTAR, ZT0, ZTMST, ZHOICE, ZITEMPAMP
    REAL(JPRB) :: ZDTHI, ZDTH, ZDTM, ZGSN, ZMSN
    REAL(JPRB) :: ZRDSNMAX, ZRDSNRESMAX
    REAL(JPRB) :: PF5, PF7
    INTEGER(C_INT) :: JL

    !--- setup constants (Fortran lines 164-174) -------------------------
    LEFLAKE = LEFLAKE_C
    ZHOICE = 1.0_JPRB / RHOICE
    ZSOILRES = 1.0_JPRB / RVLAMSK_8
    ZT0 = RTT
    ZTMST = 1.0_JPRB / PTMST
    ZITEMPAMP = 1.0_JPRB / RTEMPAMP
    ZRDSNMAX = 1.0_JPRB
    ZRDSNRESMAX = 1.0_JPRB

    ! ZEXPF / ZLINA (Fortran lines 168-169) are computed upstream but
    ! never referenced in the SRFSN_LWIMPS body, so we omit them here.

    DO JL = 1, KLON_C
      !--- per-grid setup (Fortran lines 189-213) -----------------------
      ZFRSN = MAX(PFRTI(JL,5) + PFRTI(JL,7), RFRTINY)
      PZFRSN(JL) = ZFRSN
      ZIFRSN = 1.0_JPRB / ZFRSN
      IF (ZFRSN < RFRSMALL) THEN
        LLNOSNOW = .TRUE.
      ELSE
        LLNOSNOW = .FALSE.
      ENDIF

      ! Grid fraction for snowfall interception scaling (tiles 3..8
      ! 1-based; plus lake tile 9 if FLake is on and not full cover).
      ZGRIDFRAC = PFRTI(JL,3) + PFRTI(JL,4) + PFRTI(JL,5) &
                & + PFRTI(JL,6) + PFRTI(JL,7) + PFRTI(JL,8)
      IF (LEFLAKE) THEN
        IF (NTILES_C >= 9) THEN
          IF (PFRTI(JL,9) == 1.0_JPRB) THEN
            ZGRIDFRAC = 0.0_JPRB
          ELSE
            ZGRIDFRAC = ZGRIDFRAC + PFRTI(JL,9)
          ENDIF
        ENDIF
      ENDIF

      ZSSFC = ZGRIDFRAC * PSSFC(JL)
      ZSSFL = ZGRIDFRAC * PSSFL(JL)
      ZTSFC = ZGRIDFRAC * PTSFC(JL)
      ZTSFL = ZGRIDFRAC * PTSFL(JL)

      PF5 = PFRTI(JL,5)
      PF7 = PFRTI(JL,7)

      IF (LLNOSNOW) THEN
        !--- no-snow branch (Fortran lines 216-220) --------------------
        ZSSTAR = PSSNM1M(JL) + PTMST*(ZSSFC + ZSSFL)
        PTSN(JL) = ZT0
        PGSN(JL) = 0.0_JPRB
        ZLWC = 0.0_JPRB
      ELSE
        !--- net heat flux at snow surface (Fortran lines 227-233) -----
        ZHFLUXP = PF5*(PAHFSTI(JL,5) + RLSTT*PEVAPTI(JL,5)) &
           &    + PF7*(PAHFSTI(JL,7) + RLVTT*(PEVAPTI(JL,7) - PEVAPSNW(JL)) &
           &           + RLSTT*PEVAPSNW(JL)) &
           &    + PF5*PSSRFLTI(JL,5) &
           &    + PF7*PSSRFLTI(JL,7) &
           &    + (PF5 + PF7)*PSLRFL(JL)
        ZHFLUX = ZIFRSN * ZHFLUXP

        !--- preliminary snow mass (Fortran lines 236-243) -------------
        ZSSTAR = PSSNM1M(JL) + PTMST*(ZSSFC + ZSSFL &
           &    + PF5*PEVAPTI(JL,5) + PF7*PEVAPSNW(JL))
        IF (ZSSTAR < 0.0_JPRB) ZSSTAR = 0.0_JPRB

        !--- snow liquid water (Fortran lines 247-258) -----------------
        ZDSNR = PSSNM1M(JL) / (PRSNM1M(JL) * ZFRSN)
        ZCONSLWC = MAX(0.0_JPRB, (RLWCSWEA - PRSNM1M(JL)) / RLWCSWEA)
        ZLWC = PSSNM1M(JL) * (RLWCSWEB + (RLWCSWEC - RLWCSWEB)*ZCONSLWC)
        IF (ZDSNR < ZRDSNMAX) THEN
          ZDSNRP = ZDSNR
        ELSE
          ZDSNRP = ZRDSNMAX
        ENDIF
        IF (ZDSNR /= 0.0_JPRB) THEN
          ZLWC = ZLWC * ZDSNRP / ZDSNR
        ELSE
          ZLWC = 0.0_JPRB
        ENDIF

        !--- analytical melt-transition functions (Fortran 260-269) ----
        IF (PTSNM1M(JL) < ZT0 - 0.5_JPRB*RTEMPAMP) THEN
          ZFUNC = 0.0_JPRB
          ZDFUNC = 0.0_JPRB
        ELSEIF (PTSNM1M(JL) < ZT0) THEN
          ZFUNC = 1.0_JPRB + SIN(RPI*(PTSNM1M(JL) - ZT0)*ZITEMPAMP)
          ZDFUNC = COS(RPI*(PTSNM1M(JL) - ZT0)*ZITEMPAMP)*RPI*ZITEMPAMP
        ELSE
          ZFUNC = 0.0_JPRB
          ZDFUNC = RPI*ZITEMPAMP
        ENDIF

        !--- implicit temperature equation terms (Fortran 272-276) -----
        ZCONSB = RLMLT * ZLWC * ZDFUNC
        ZCONSAMAX = (PRSNM1M(JL)*RHOCI*ZRDSNMAX) * ZHOICE
        ZCOND1 = (PSSNM1M(JL)*RHOCI*ZHOICE) * ZIFRSN
        ZCONSA = MIN(ZCOND1, ZCONSAMAX)
        ZCONS2 = PTMST / (ZCONSA + ZCONSB)

        !--- snow depth and snow-soil resistance (Fortran 279-284) -----
        ZDSN = MIN(PSSNM1M(JL)/(PRSNM1M(JL)*ZFRSN), ZRDSNRESMAX)
        ZSNRES = 0.5_JPRB * ZDSN / (RLAMICE*(PRSNM1M(JL)*ZHOICE)**RALAMSN)
        ZRS = 1.0_JPRB / (ZSNRES + ZSOILRES)

        !--- throughfall interception (Fortran 286-299) ----------------
        ZWFLUXRF = (PF5 + PF7)*(ZTSFC + ZTSFL)
        ZCOND3 = ZWFLUXRF * PTMST
        ZCOND4 = ZLWC * (1.0_JPRB - ZFUNC)
        ZWFLUXRF = MIN(ZCOND3, ZCOND4)
        ZCOND5 = ZCONSA*(ZT0 - PTSNM1M(JL))*ZTMST + ZT0*ZRS
        ZCOND6 = RLMLT*ZWFLUXRF*ZTMST
        ZHFLUXRF1 = MIN(ZCOND5, ZCOND6)
        ZHFLUXRF1 = MAX(ZHFLUXRF1, 0.0_JPRB)
        ZWFLUXRF = ZHFLUXRF1 / RLMLT
        ZSSTAR = ZSSTAR + ZWFLUXRF*PTMST

        !--- implicit temperature solve (Fortran 301-303) --------------
        ZTSTARI = 1.0_JPRB / (1.0_JPRB + ZCONS2*ZRS)
        ZTSTAR = (PTSNM1M(JL) + ZCONS2*(ZHFLUX + ZHFLUXRF1 &
           &     + PTSAM1M_TOP(JL)*ZRS)) * ZTSTARI

        IF (ZTSTAR <= ZT0) THEN
          !--- no-melt branch (Fortran 304-307) -----------------------
          PTSN(JL) = ZTSTAR
          PGSN(JL) = (ZTSTAR - PTSAM1M_TOP(JL)) * ZRS
        ELSE
          !--- melt branch (Fortran 308-327) --------------------------
          ZDTHI = 1.0_JPRB / (ZHFLUX + ZHFLUXRF1 - (ZT0 - PTSAM1M_TOP(JL))*ZRS)
          ZDTH = ((ZT0 - PTSNM1M(JL)) * (ZCONSA + ZCONSB)) * ZDTHI
          ZDTM = PTMST - ZDTH
          ZGSN = (ZT0 - PTSAM1M_TOP(JL)) * ZRS
          ZMSN = (ZHFLUX + ZHFLUXRF1 - ZGSN) / RLMLT
          ZSSTAR1 = ZSSTAR - ZDTM*ZMSN*ZFRSN
          IF (ZSSTAR1 >= 0.0_JPRB) THEN
            ! Partial melt: snow remains.
            PTSN(JL) = ZT0
            PGSN(JL) = (ZT0 - PTSAM1M_TOP(JL)) * ZRS
          ELSE
            ! Full melt: all snow melted.
            PTSN(JL) = ZT0
            PGSN(JL) = (((ZT0 - PTSAM1M_TOP(JL))*ZRS)*(ZDTM + ZDTH) &
               &      + (ZHFLUX + ZHFLUXRF1)*(PTMST - ZDTM - ZDTH)) * ZTMST
          ENDIF
        ENDIF
      ENDIF

      PSSTAR(JL) = ZSSTAR
      PLWC(JL) = ZLWC

      !--- normalise PGSN to grid-square (Fortran lines 336-338) -------
      ! PGSN = ZFRSN * PGSN.  ZFRSN was computed at the top of this
      ! iteration and is still in scope.
      PGSN(JL) = ZFRSN * PGSN(JL)
    END DO
  END SUBROUTINE SURF_REF_SNOW

  !=====================================================================!
  ! SURF_REF_SOILW -- HTESSEL soil-moisture update (Van Genuchten).      !
  !                                                                     !
  ! Standalone transcription of the soil-moisture branch of              !
  ! SURFTSTP_CTL section 4, comprising four upstream routines:          !
  !   * SRFWEXC_VG  (VG hydraulic K/D + VIC / infiltration runoff +      !
  !                 root extraction + tridiagonal coefficients)          !
  !   * SRFWDIF     (Thomas tridiagonal solver for PWSADIF)              !
  !   * SRFWINC     (t+1 unfiltered moisture + drainage diagnostic)      !
  !   * SRFWNG      (clip negative / redistribute saturation excess;     !
  !                 split surface / sub-surface runoff)                  !
  ! plus the SURFTSTP_CTL bottom-drainage-to-PROD closure (sec 4.6).     !
  !                                                                     !
  ! Mirrors surf_pytorch.physics.soil.compute_soil_moisture() so the     !
  ! ULP comparison test can feed identical inputs to both and compare.   !
  !                                                                     !
  ! Cold-start simplifications:                                          !
  !   * LEWBSOILFIX = .FALSE.  -> ZEPSILONADD = 0 (no water-balance fix) !
  !   * LEVGEN = .TRUE.        -> per-soil-type RWSAT (RWSATM(JS))       !
  !   * KIDIA=1, KFDIA=KLON_C (whole column)                            !
  !   * no FLake / no urban (KTILES = 8): PCUR/RURB* branches omitted    !
  !   * no skin-reservoir clipping in SRFWNG (PWL not carried)           !
  !                                                                     !
  ! Outputs:                                                            !
  !   PWSA (KLON x KLEVS), PROS (KLON), PROD (KLON),                    !
  !   PWFSD (KLON), PSAWGFL (KLON x KLEVS)                              !
  !                                                                     !
  ! Author: surf-pytorch reference driver (M5c-soilw).  Algorithm (c) ECMWF.!
  !=====================================================================!
  SUBROUTINE SURF_REF_SOILW( &
      &   KLON_C, KLEVS_C, NTILES_C, NSOTY_C, NV_C, &
      &   PWSAM1M, PTSAM1M, &
      &   PFRTI, PEVAPTI, PEINTTI, &
      &   PTSFC, PTSFL, PMSN, PEMSSN, PEVAPSNW, &
      &   PSDOR, KSOTY, KTVL, KTVH, LDLAND, &
      &   RDAT, RVROOTSA, &
      &   RWSATM, RWRESTM, RWCAPM, RWPWPM, RWCONSM, &
      &   RMVGALPHA, RMFACM, RLAMBDAM, RDMAXM, RDMINM, &
      &   RSIGORMAX, RSIGORMIN, RPSFR, &
      &   PTMST, RSIMP, RSRDEP, RHOH2O, RTF1, RTF2, RTF3, RTF4, &
      &   LESSRO, &
      &   PWSA, PROS, PROD, PWFSD, PSAWGFL &
      & ) BIND(C, NAME="surf_ref_soilw")

    USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_BOOL

    !--- dimensions ------------------------------------------------------
    INTEGER(C_INT), VALUE, INTENT(IN) :: KLON_C, KLEVS_C, NTILES_C, NSOTY_C, NV_C
    !--- soil state at t-1 (KLON x KLEVS) --------------------------------
    REAL(JPRB), INTENT(IN) :: PWSAM1M(KLON_C, KLEVS_C)
    REAL(JPRB), INTENT(IN) :: PTSAM1M(KLON_C, KLEVS_C)
    !--- per-tile fluxes (KLON x NTILES) ---------------------------------
    REAL(JPRB), INTENT(IN) :: PFRTI(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: PEVAPTI(KLON_C, NTILES_C)
    REAL(JPRB), INTENT(IN) :: PEINTTI(KLON_C, NTILES_C)
    !--- precipitation throughfall / snowmelt / snow-evap mismatch (KLON) -
    REAL(JPRB), INTENT(IN) :: PTSFC(KLON_C)
    REAL(JPRB), INTENT(IN) :: PTSFL(KLON_C)
    REAL(JPRB), INTENT(IN) :: PMSN(KLON_C)
    REAL(JPRB), INTENT(IN) :: PEMSSN(KLON_C)
    REAL(JPRB), INTENT(IN) :: PEVAPSNW(KLON_C)
    !--- static fields (KLON) --------------------------------------------
    REAL(JPRB), INTENT(IN) :: PSDOR(KLON_C)
    INTEGER(C_INT), INTENT(IN) :: KSOTY(KLON_C)
    INTEGER(C_INT), INTENT(IN) :: KTVL(KLON_C)
    INTEGER(C_INT), INTENT(IN) :: KTVH(KLON_C)
    LOGICAL(C_BOOL), INTENT(IN) :: LDLAND(KLON_C)
    !--- layer thickness (KLEVS) + root fractions table (KLEVS x NV) -----
    !--- RVROOTSA is 0-based on the veg-type axis to match the Python
    !----VegTable.rvrootsa (n_soil, 21) buffer indexed 0..20.  Per-class
    !----soil LUTs below are likewise 0-based (index 0 = water sentinel)
    !----so KSOTY=0..7 maps directly to the Python SoilTable columns.
    REAL(JPRB), INTENT(IN) :: RDAT(KLEVS_C)
    REAL(JPRB), INTENT(IN) :: RVROOTSA(KLEVS_C, 0:NV_C-1)
    !--- per-soil-type VG LUT columns (0:NSOTY_C, 0-based) --------------
    REAL(JPRB), INTENT(IN) :: RWSATM(0:NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RWRESTM(0:NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RWCAPM(0:NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RWPWPM(0:NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RWCONSM(0:NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RMVGALPHA(0:NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RMFACM(0:NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RLAMBDAM(0:NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RDMAXM(0:NSOTY_C)
    REAL(JPRB), INTENT(IN) :: RDMINM(0:NSOTY_C)
    !--- VIC sub-grid orography bounds + convective-precip splitter ------
    REAL(JPRB), VALUE, INTENT(IN) :: RSIGORMAX, RSIGORMIN, RPSFR
    !--- scalars ---------------------------------------------------------
    REAL(JPRB), VALUE, INTENT(IN) :: PTMST, RSIMP, RSRDEP, RHOH2O
    REAL(JPRB), VALUE, INTENT(IN) :: RTF1, RTF2, RTF3, RTF4
    LOGICAL(C_BOOL), VALUE, INTENT(IN) :: LESSRO
    !--- outputs ---------------------------------------------------------
    REAL(JPRB), INTENT(OUT) :: PWSA(KLON_C, KLEVS_C)
    REAL(JPRB), INTENT(OUT) :: PROS(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PROD(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PWFSD(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PSAWGFL(KLON_C, KLEVS_C)

    !--- locals ----------------------------------------------------------
    INTEGER :: JL, JK, JS
    INTEGER :: ITYP(KLON_C)
    REAL(JPRB) :: ZF(KLON_C, KLEVS_C), ZLIQ(KLON_C, KLEVS_C)
    REAL(JPRB) :: ZROOTW(KLON_C, KLEVS_C, 3)     ! tiles 4,6,7 only
    REAL(JPRB) :: ZEXT(KLON_C, 3)
    REAL(JPRB) :: ZSAWEXT(KLON_C, KLEVS_C)
    REAL(JPRB) :: ZCONDS(KLON_C)
    REAL(JPRB) :: PCFW(KLON_C, KLEVS_C), PRHSW(KLON_C, KLEVS_C)
    REAL(JPRB) :: PWSADIF(KLON_C, KLEVS_C)
    REAL(JPRB) :: ZTCOE(KLON_C), ZEBSW(KLON_C, KLEVS_C)
    REAL(JPRB) :: ZDI(KLEVS_C), ZIDI(KLEVS_C), ZRRRI(KLEVS_C)
    REAL(JPRB) :: ZEPSILON, ZHOH2O, ZPSFR, ZINVTMST
    REAL(JPRB) :: ZW, ZSE, ZWFAC, ZLAM, ZMFAC, ZRMFAC, ZWCONS, ZALPHA
    REAL(JPRB) :: ZK, ZD, ZKM, ZKD, ZDD, ZDMAX, ZDMIN, ZFF, ZFFM
    REAL(JPRB) :: ZWM, ZKMD, ZEXTK, ZWSFL, ZDISC, ZQDP, ZFAC
    REAL(JPRB) :: ZTPFAC2, ZTPFAC3, ZCONS
    REAL(JPRB) :: ZRSFL, ZROT, ZROL, ZROC
    REAL(JPRB) :: ZDSURF, ZKSURF, ZINFMAX
    ! VIC locals
    REAL(JPRB) :: ZFRK(KLEVS_C), ZSUM, ZROEFF, ZBWS, ZB1, ZBM
    REAL(JPRB) :: ZWK(KLEVS_C), ZWMK(KLEVS_C), ZWMAX, ZCONW1
    REAL(JPRB) :: ZLYEPS, ZLIMRS, ZLYSIC, ZVOL, ZROS
    ! SRFWNG locals
    REAL(JPRB) :: ZWSAT, ZTMST, ZDWD
    ! misc
    INTEGER :: JTILE4, JTILE6, JTILE7   ! ZROOTW slice indices (1..3)
    LOGICAL :: LLFREEZ, LLAND
    REAL(JPRB), PARAMETER :: ZEPS = 1.0E-12_JPRB

    ! tile mapping: ZROOTW(:,:,1)=tile4, ZROOTW(:,:,2)=tile6, ZROOTW(:,:,3)=tile7
    JTILE4 = 1
    JTILE6 = 2
    JTILE7 = 3

    ZEPSILON = 100.0_JPRB * EPSILON(ZEPSILON)
    ZHOH2O = 1.0_JPRB / RHOH2O
    ZPSFR = 1.0_JPRB / RPSFR
    ZINVTMST = 1.0_JPRB / PTMST
    LLFREEZ = .TRUE.

    PCFW = 0.0_JPRB
    PRHSW = 0.0_JPRB
    PSAWGFL = 0.0_JPRB
    PROS = 0.0_JPRB
    PROD = 0.0_JPRB
    PWFSD = 0.0_JPRB
    ZSAWEXT = 0.0_JPRB
    ZCONDS = 0.0_JPRB
    ZROOTW = 0.0_JPRB
    ZEXT = 0.0_JPRB

    ! ==================================================================
    ! SRFWEXC_VG -- VG coefficients, runoff, root extraction, tridiag
    ! ==================================================================

    ! -- 2.1 liquid fraction ZF + liquid-equivalent ZLIQ -----------------
    DO JK = 1, KLEVS_C
      DO JL = 1, KLON_C
        IF (PTSAM1M(JL,JK) < RTF1 .AND. PTSAM1M(JL,JK) > RTF2) THEN
          ZF(JL,JK) = 0.5_JPRB * (1.0_JPRB - SIN(RTF4*(PTSAM1M(JL,JK)-RTF3)))
        ELSE IF (PTSAM1M(JL,JK) <= RTF2) THEN
          ZF(JL,JK) = 1.0_JPRB
        ELSE
          ZF(JL,JK) = 0.0_JPRB
        END IF
        JS = KSOTY(JL)
        ZLIQ(JL,JK) = MAX(RWPWPM(JS), MIN(RWCAPM(JS), &
                     & PWSAM1M(JL,JK)*(1.0_JPRB - ZF(JL,JK))))
      END DO
    END DO

    ! -- 2.2 root-extraction weights ZROOTW + normaliser ZEXT ------------
    ! tiles 4 (low-veg), 6 (high-veg), 7 (high-veg-snow) only.
    ! JTILE 4 -> ITYP=KTVL ; JTILE 6,7 -> ITYP=KTVH.
    DO JL = 1, KLON_C
      ITYP(JL) = KTVL(JL)
    END DO
    DO JK = 1, KLEVS_C
      DO JL = 1, KLON_C
        JS = KSOTY(JL)
        ! tile 4: dew forces layer-1 weight = 1
        IF (PFRTI(JL,4)*PEVAPTI(JL,4) > 0.0_JPRB) THEN
          IF (JK == 1) ZROOTW(JL,JK,JTILE4) = 1.0_JPRB
        ELSE
          ZROOTW(JL,JK,JTILE4) = RVROOTSA(JK,ITYP(JL)) &
               & *(ZLIQ(JL,JK) - RWPWPM(JS))
        END IF
        ZEXT(JL,JTILE4) = ZEXT(JL,JTILE4) + ZROOTW(JL,JK,JTILE4)
      END DO
    END DO

    DO JL = 1, KLON_C
      ITYP(JL) = KTVH(JL)
    END DO
    DO JK = 1, KLEVS_C
      DO JL = 1, KLON_C
        JS = KSOTY(JL)
        ! tile 6
        IF (PFRTI(JL,6)*PEVAPTI(JL,6) > 0.0_JPRB) THEN
          IF (JK == 1) ZROOTW(JL,JK,JTILE6) = 1.0_JPRB
        ELSE
          ZROOTW(JL,JK,JTILE6) = RVROOTSA(JK,ITYP(JL)) &
               & *(ZLIQ(JL,JK) - RWPWPM(JS))
        END IF
        ZEXT(JL,JTILE6) = ZEXT(JL,JTILE6) + ZROOTW(JL,JK,JTILE6)
        ! tile 7 (evap reduced by snow-under-forest)
        IF (PFRTI(JL,7)*(PEVAPTI(JL,7)-PEVAPSNW(JL)) > 0.0_JPRB) THEN
          IF (JK == 1) ZROOTW(JL,JK,JTILE7) = 1.0_JPRB
        ELSE
          ZROOTW(JL,JK,JTILE7) = RVROOTSA(JK,ITYP(JL)) &
               & *(ZLIQ(JL,JK) - RWPWPM(JS))
        END IF
        ZEXT(JL,JTILE7) = ZEXT(JL,JTILE7) + ZROOTW(JL,JK,JTILE7)
      END DO
    END DO
    DO JL = 1, KLON_C
      ZEXT(JL,JTILE4) = MAX(ZEPSILON, ZEXT(JL,JTILE4))
      ZEXT(JL,JTILE6) = MAX(ZEPSILON, ZEXT(JL,JTILE6))
      ZEXT(JL,JTILE7) = MAX(ZEPSILON, ZEXT(JL,JTILE7))
    END DO

    ! -- 2.3 top-layer coefficients --------------------------------------
    DO JL = 1, KLON_C
      LLAND = LDLAND(JL)
      IF (LLAND) THEN
        JS = KSOTY(JL)
        ZDMAX = RDMAXM(JS)
        ZDMIN = RDMINM(JS)
        ZW = MAX(MAX(PWSAM1M(JL,1), PWSAM1M(JL,2)), RWRESTM(JS))
        ZSE = (ZW - RWRESTM(JS)) / (RWSATM(JS) - RWRESTM(JS))
        ZWFAC = RWSATM(JS) - RWRESTM(JS)
        ZLAM = RLAMBDAM(JS)
        ZMFAC = RMFACM(JS)
        ZRMFAC = 1.0_JPRB / ZMFAC
        ZWCONS = RWCONSM(JS)
        ZALPHA = RMVGALPHA(JS)
        IF (ZW <= 1.001_JPRB*RWRESTM(JS)) THEN
          ZK = 0.0_JPRB
          ZD = ZDMIN
        ELSE IF (ZW <= 0.999_JPRB*RWSATM(JS)) THEN
          ZKD = ZWCONS * ZSE**ZLAM &
               & *(1.0_JPRB - (1.0_JPRB - ZSE**ZRMFAC)**ZMFAC)**2.0_JPRB
          ZDD = ((1.0_JPRB - ZMFAC)*ZWCONS) &
               & / SIGN(MAX(ABS(ZALPHA*ZMFAC*ZWFAC), ZEPSILON), &
               &       ZALPHA*ZMFAC*ZWFAC)
          ZDD = ZDD * (ZSE**(ZLAM - ZRMFAC)) &
               & *((1.0_JPRB - ZSE**ZRMFAC)**ZMFAC &
               &  + (1.0_JPRB - ZSE**ZRMFAC)**(-ZMFAC) - 2.0_JPRB)
          ZK = ZKD
          ZD = ZDD
        ELSE
          ZK = RWCONSM(JS)
          ZD = ZDMAX
        END IF
        IF (LLFREEZ) THEN
          ZFF = MIN(ZF(JL,1), ZF(JL,2))
          ZD = ZFF*ZDMIN + (1.0_JPRB - ZFF)*ZD
          ZK = (1.0_JPRB - ZFF)*ZK
        END IF

        ! -- runoff ----------------------------------------------------
        ZRSFL = PTSFL(JL) + PMSN(JL) + PTSFC(JL)
        IF (LESSRO) THEN
          ! VIC surface runoff (only when raining)
          DO JK = 1, KLEVS_C
            IF (JK > 1) THEN
              ZSUM = SUM(RDAT(1:JK-1))
            ELSE
              ZSUM = 0.0_JPRB
            END IF
            ZFRK(JK) = MAX(0.0_JPRB, (MIN(RSRDEP, SUM(RDAT(1:JK))) - ZSUM) &
                  & / RDAT(JK))
          END DO
          IF (ZRSFL > 0.0_JPRB) THEN
            ZROEFF = MAX(0.0_JPRB, (PSDOR(JL) - RSIGORMIN)) &
                   & / (PSDOR(JL) + RSIGORMAX)
            ZBWS = MAX(MIN(ZROEFF, 0.5_JPRB), 0.01_JPRB)
            ZB1 = 1.0_JPRB + ZBWS
            ZBM = 1.0_JPRB / ZB1
            DO JK = 1, KLEVS_C
              ZWK(JK) = MAX(0.0_JPRB, (PWSAM1M(JL,JK) - RWPWPM(JS))) &
                   & *(1.0_JPRB - ZF(JL,JK)) + RWPWPM(JS)
              ZWMK(JK) = ((RWSATM(JS) - RWPWPM(JS))*(1.0_JPRB - ZF(JL,JK)) &
                   & + RWPWPM(JS))
            END DO
            ZW = SUM(ZFRK(1:KLEVS_C) * ZWK(1:KLEVS_C) * RDAT(1:KLEVS_C))
            ZWMAX = SUM(ZFRK(1:KLEVS_C) * ZWMK(1:KLEVS_C) * RDAT(1:KLEVS_C))
            ZCONW1 = ZWMAX * ZB1
            ZLYEPS = MAX(0.0_JPRB, ZW - ZWMAX)
            ZLIMRS = -1.0_JPRB * ZLYEPS
            IF (ZLYEPS > 0.1_JPRB*ZWMAX) ZLIMRS = 0.0_JPRB
            ZLYSIC = MIN(1.0_JPRB, MAX(0.0_JPRB, (ZW - ZLYEPS)/ZWMAX))
            ZVOL = MAX(0.0_JPRB, (1.0_JPRB - ZLYSIC)**ZBM &
                  & - ZRSFL/(ZINVTMST*RHOH2O*ZCONW1))
            ZROS = ZRSFL/(ZINVTMST*RHOH2O) - MAX(ZWMAX - ZW, ZLIMRS)
            IF (ZVOL > 0.0_JPRB) ZROS = ZROS + ZWMAX * ZVOL**ZB1
            ZROS = MAX(ZROS, 0.0_JPRB)
            ZROT = ZROS * RHOH2O * ZINVTMST
          ELSE
            ZROT = 0.0_JPRB
          END IF
        ELSE
          ! infiltration-limit runoff.  NOTE: uses *separate* locals
          !----ZDSURF/ZKSURF so the top-layer VG ZD/ZK (used below for
          !----PCFW/PRHSW) are NOT clobbered -- matches upstream
          !----srfwexc_vg_mod.F90 lines 398-416 which never reassigns
          !----ZD/ZK here.
          ZDSURF = RDMAXM(JS)
          ZKSURF = RWCONSM(JS)
          IF (LLFREEZ) THEN
            ZFF = ZF(JL,1)
            ZDSURF = ZFF*RDMINM(JS) + (1.0_JPRB - ZFF)*RDMAXM(JS)
            ZKSURF = (1.0_JPRB - ZFF)*RWCONSM(JS)
          END IF
          ZINFMAX = (ZDSURF*(RWSATM(JS) - PWSAM1M(JL,1))/(0.5_JPRB*RDAT(1)) &
               & + ZKSURF)*RHOH2O
          ZROL = MAX(0.0_JPRB, PTSFL(JL) + PMSN(JL) - ZINFMAX)
          ZROC = MAX(0.0_JPRB, RPSFR*PTSFC(JL) - ZINFMAX) * ZPSFR
          ZROT = ZROL + ZROC
        END IF

        ! -- assemble ZWSFL + tile contributions -----------------------
        ZWSFL = PTSFL(JL) + PMSN(JL) + PTSFC(JL) - ZROT
        ! tile 3 (wet skin)
        ZWSFL = ZWSFL + PFRTI(JL,3)*PEVAPTI(JL,3) - PEINTTI(JL,3)
        ! tile 4
        ZEXTK = (PFRTI(JL,4)*PEVAPTI(JL,4) - PEINTTI(JL,4)) &
             & * ZROOTW(JL,1,JTILE4) / ZEXT(JL,JTILE4)
        ZSAWEXT(JL,1) = ZEXTK
        ZCONDS(JL) = ZCONDS(JL) + MAX((PFRTI(JL,4)*PEVAPTI(JL,4) &
               & - PEINTTI(JL,4)), 0.0_JPRB)
        ZWSFL = ZWSFL + ZEXTK
        ! tile 6
        ZEXTK = (PFRTI(JL,6)*PEVAPTI(JL,6) - PEINTTI(JL,6)) &
             & * ZROOTW(JL,1,JTILE6) / ZEXT(JL,JTILE6)
        ZSAWEXT(JL,1) = ZSAWEXT(JL,1) + ZEXTK
        ZCONDS(JL) = ZCONDS(JL) + MAX((PFRTI(JL,6)*PEVAPTI(JL,6) &
               & - PEINTTI(JL,6)), 0.0_JPRB)
        ZWSFL = ZWSFL + ZEXTK
        ! tile 7
        ZEXTK = (PFRTI(JL,7)*(PEVAPTI(JL,7) - PEVAPSNW(JL)) - PEINTTI(JL,7)) &
             & * ZROOTW(JL,1,JTILE7) / ZEXT(JL,JTILE7)
        ZSAWEXT(JL,1) = ZSAWEXT(JL,1) + ZEXTK
        ZCONDS(JL) = ZCONDS(JL) + MAX((PFRTI(JL,7)*(PEVAPTI(JL,7) &
               & - PEVAPSNW(JL)) - PEINTTI(JL,7)), 0.0_JPRB)
        ZWSFL = ZWSFL + ZEXTK
        ! tile 8 (bare soil)
        ZEXTK = PFRTI(JL,8)*PEVAPTI(JL,8) + PEMSSN(JL) - PEINTTI(JL,8)
        ZCONDS(JL) = ZCONDS(JL) + MAX((PFRTI(JL,8)*PEVAPTI(JL,8) + PEMSSN(JL) &
               & - PEINTTI(JL,8)), 0.0_JPRB)
        ZWSFL = ZWSFL + ZEXTK

        ! -- top-layer coefficients -----------------------------------
        PCFW(JL,1) = ZD * PTMST * RSIMP / (0.5_JPRB*(RDAT(1) + RDAT(2)))
        PRHSW(JL,1) = PTMST * (-ZK + ZWSFL*ZHOH2O) / RDAT(1)
        PSAWGFL(JL,1) = RHOH2O * ZK
        PROS(JL) = PTMST * ZROT
      END IF
    END DO

    ! -- 2.4 deeper-layer coefficients (JK = 2..KLEVS) -------------------
    DO JK = 2, KLEVS_C
      DO JL = 1, KLON_C
        LLAND = LDLAND(JL)
        IF (LLAND) THEN
          JS = KSOTY(JL)
          ZDMAX = RDMAXM(JS)
          ZDMIN = RDMINM(JS)
          IF (JK < KLEVS_C) THEN
            ZW = MAX(MAX(PWSAM1M(JL,JK), PWSAM1M(JL,JK+1)), RWRESTM(JS))
          ELSE
            ZW = MAX(PWSAM1M(JL,JK), RWRESTM(JS))
          END IF
          ZSE = (ZW - RWRESTM(JS)) / (RWSATM(JS) - RWRESTM(JS))
          ZWFAC = RWSATM(JS) - RWRESTM(JS)
          ZLAM = RLAMBDAM(JS)
          ZMFAC = RMFACM(JS)
          ZRMFAC = 1.0_JPRB / ZMFAC
          ZWCONS = RWCONSM(JS)
          ZALPHA = RMVGALPHA(JS)
          IF (ZW <= 1.001_JPRB*RWRESTM(JS)) THEN
            ZD = ZDMIN
            ZK = 0.0_JPRB
          ELSE IF (ZW <= 0.999_JPRB*RWSATM(JS)) THEN
            ZKD = ZWCONS * ZSE**ZLAM &
                 & *(1.0_JPRB - (1.0_JPRB - ZSE**ZRMFAC)**ZMFAC)**2.0_JPRB
            ZDD = ((1.0_JPRB - ZMFAC)*ZWCONS) &
                 & / SIGN(MAX(ABS(ZALPHA*ZMFAC*ZWFAC), ZEPSILON), &
                 &       ZALPHA*ZMFAC*ZWFAC)
            ZDD = ZDD * (ZSE**(ZLAM - ZRMFAC)) &
                 & *((1.0_JPRB - ZSE**ZRMFAC)**ZMFAC &
                 &  + (1.0_JPRB - ZSE**ZRMFAC)**(-ZMFAC) - 2.0_JPRB)
            ZK = ZKD
            ZD = ZDD
          ELSE
            ZK = RWCONSM(JS)
            ZD = ZDMAX
          END IF

          ! ZKM from the layer above (j-1,j)
          ZWM = MAX(MAX(PWSAM1M(JL,JK-1), PWSAM1M(JL,JK)), RWRESTM(JS))
          ZSE = (ZWM - RWRESTM(JS)) / (RWSATM(JS) - RWRESTM(JS))
          IF (ZWM <= RWRESTM(JS)) THEN
            ZKM = 0.0_JPRB
          ELSE IF (ZWM < 0.999_JPRB*RWSATM(JS)) THEN
            ZKMD = ZWCONS * ZSE**ZLAM &
                 & *(1.0_JPRB - (1.0_JPRB - ZSE**ZRMFAC)**ZMFAC)**2.0_JPRB
            ZKM = ZKMD
          ELSE
            ZKM = RWCONSM(JS)
          END IF

          IF (JK == KLEVS_C) ZD = 0.0_JPRB

          IF (LLFREEZ) THEN
            IF (JK < KLEVS_C) THEN
              ZFF = MIN(ZF(JL,JK), ZF(JL,JK+1))
              ZD = ZFF*ZDMIN + (1.0_JPRB - ZFF)*ZD
            ELSE
              ZFF = ZF(JL,JK)
            END IF
            ZFFM = MIN(ZF(JL,JK-1), ZF(JL,JK))
            ZK = (1.0_JPRB - ZFF)*ZK
            ZKM = (1.0_JPRB - ZFFM)*ZKM
          END IF

          ! tile root-extraction contributions to the r.h.s.
          ZWSFL = 0.0_JPRB
          ! tile 4
          ZEXTK = (PFRTI(JL,4)*PEVAPTI(JL,4) - PEINTTI(JL,4)) &
               & * ZROOTW(JL,JK,JTILE4) / ZEXT(JL,JTILE4)
          ZSAWEXT(JL,JK) = ZEXTK
          ZWSFL = ZWSFL + ZEXTK
          ! tile 6
          ZEXTK = (PFRTI(JL,6)*PEVAPTI(JL,6) - PEINTTI(JL,6)) &
               & * ZROOTW(JL,JK,JTILE6) / ZEXT(JL,JTILE6)
          ZSAWEXT(JL,JK) = ZSAWEXT(JL,JK) + ZEXTK
          ZWSFL = ZWSFL + ZEXTK
          ! tile 7
          ZEXTK = (PFRTI(JL,7)*(PEVAPTI(JL,7) - PEVAPSNW(JL)) - PEINTTI(JL,7)) &
               & * ZROOTW(JL,JK,JTILE7) / ZEXT(JL,JTILE7)
          ZSAWEXT(JL,JK) = ZSAWEXT(JL,JK) + ZEXTK
          ZWSFL = ZWSFL + ZEXTK

          ! soil-water diffusivity / r.h.s. / drainage diagnostic
          IF (JK < KLEVS_C) THEN
            PCFW(JL,JK) = ZD * PTMST * RSIMP / (0.5_JPRB*(RDAT(JK) + RDAT(JK+1)))
          ELSE
            PCFW(JL,JK) = 0.0_JPRB
          END IF
          PRHSW(JL,JK) = PTMST * (ZKM - ZK + ZWSFL*ZHOH2O) / RDAT(JK)
          PSAWGFL(JL,JK) = RHOH2O * ZK
        END IF
      END DO
    END DO

    ! ==================================================================
    ! SRFWDIF -- Thomas tridiagonal solve -> PWSADIF
    ! ==================================================================
    ZTPFAC2 = 1.0_JPRB / RSIMP
    DO JK = 1, KLEVS_C
      DO JL = 1, KLON_C
        IF (LDLAND(JL)) THEN
          PWSADIF(JL,JK) = ZTPFAC2 * PWSAM1M(JL,JK) + PRHSW(JL,JK)
        END IF
      END DO
    END DO
    ! top-layer elimination
    DO JL = 1, KLON_C
      IF (LDLAND(JL)) THEN
        ZTCOE(JL) = PCFW(JL,1)
        ZQDP = 1.0_JPRB / RDAT(1)
        ZDISC = 1.0_JPRB / (1.0_JPRB + PCFW(JL,1)*ZQDP)
        ZEBSW(JL,1) = ZDISC * (PCFW(JL,1)*ZQDP)
        PWSADIF(JL,1) = ZDISC * PWSADIF(JL,1)
      END IF
    END DO
    ! middle-layer elimination
    DO JK = 2, KLEVS_C - 1
      DO JL = 1, KLON_C
        IF (LDLAND(JL)) THEN
          ZQDP = 1.0_JPRB / RDAT(JK)
          ZFAC = ZTCOE(JL) * ZQDP
          ZTCOE(JL) = PCFW(JL,JK)
          ZDISC = 1.0_JPRB / (1.0_JPRB + ZFAC*(1.0_JPRB - ZEBSW(JL,JK-1)) &
               & + PCFW(JL,JK)*ZQDP)
          ZEBSW(JL,JK) = ZDISC * (PCFW(JL,JK)*ZQDP)
          PWSADIF(JL,JK) = ZDISC * (PWSADIF(JL,JK) + ZFAC*PWSADIF(JL,JK-1))
        END IF
      END DO
    END DO
    ! bottom-layer elimination
    DO JL = 1, KLON_C
      IF (LDLAND(JL)) THEN
        ZQDP = 1.0_JPRB / RDAT(KLEVS_C)
        ZFAC = ZTCOE(JL) * ZQDP
        ZDISC = 1.0_JPRB / (1.0_JPRB + ZFAC*(1.0_JPRB - ZEBSW(JL,KLEVS_C-1)) &
             & + PCFW(JL,KLEVS_C)*ZQDP)
        PWSADIF(JL,KLEVS_C) = ZDISC * (PWSADIF(JL,KLEVS_C) &
             & + ZFAC*PWSADIF(JL,KLEVS_C-1))
      END IF
    END DO
    ! back-substitution
    DO JK = KLEVS_C - 1, 1, -1
      DO JL = 1, KLON_C
        IF (LDLAND(JL)) THEN
          PWSADIF(JL,JK) = PWSADIF(JL,JK) + ZEBSW(JL,JK)*PWSADIF(JL,JK+1)
        END IF
      END DO
    END DO

    ! ==================================================================
    ! SRFWINC -- t+1 unfiltered moisture + drainage diagnostic
    ! ==================================================================
    ZTPFAC3 = ZTPFAC2 - 1.0_JPRB
    ZCONS = RHOH2O / PTMST
    DO JK = 1, KLEVS_C
      DO JL = 1, KLON_C
        IF (LDLAND(JL)) THEN
          PWSA(JL,JK) = PWSADIF(JL,JK) - ZTPFAC3 * PWSAM1M(JL,JK)
          IF (JK < KLEVS_C) THEN
            PSAWGFL(JL,JK) = PSAWGFL(JL,JK) + ZCONS*PCFW(JL,JK) &
                 & *(PWSADIF(JL,JK) - PWSADIF(JL,JK+1))
          END IF
        ELSE
          PWSA(JL,JK) = PWSAM1M(JL,JK)
          PSAWGFL(JL,JK) = 0.0_JPRB
        END IF
      END DO
    END DO
    DO JL = 1, KLON_C
      PWFSD(JL) = PSAWGFL(JL,1)
    END DO

    ! ==================================================================
    ! SRFWNG -- clip negative moisture, redistribute saturation excess,
    ! split surface / sub-surface runoff.  (Skin-reservoir clip skipped.)
    ! ==================================================================
    ZTMST = 1.0_JPRB / PTMST
    DO JK = 1, KLEVS_C
      ZDI(JK) = RDAT(JK)
      ZIDI(JK) = 1.0_JPRB / RDAT(JK)
    END DO
    DO JL = 1, KLON_C
      IF (LDLAND(JL)) THEN
        JS = KSOTY(JL)
        ZWSAT = RWSATM(JS)
        DO JK = 1, KLEVS_C
          ZDWD = MIN(PWSA(JL,JK), 0.0_JPRB)
          PWSA(JL,JK) = PWSA(JL,JK) - ZDWD
          IF (JK < KLEVS_C) THEN
            PWSA(JL,JK+1) = PWSA(JL,JK+1) + ZDWD*ZDI(JK)*ZIDI(JK+1)
          END IF
          ZRRRI(JK) = MAX(0.0_JPRB, PWSA(JL,JK) - ZWSAT)
          IF (JK == 1) THEN
            PROS(JL) = PROS(JL) + ZDI(JK)*RHOH2O*ZRRRI(JK)
            PWFSD(JL) = PWFSD(JL) + ZDWD*RHOH2O*ZDI(JK)*ZTMST
          ELSE
            PROD(JL) = PROD(JL) + ZDI(JK)*RHOH2O*ZRRRI(JK)
          END IF
          PWSA(JL,JK) = PWSA(JL,JK) - ZRRRI(JK)
        END DO
      ELSE
        PROD(JL) = 0.0_JPRB
        PWFSD(JL) = 0.0_JPRB
      END IF
    END DO

    ! ==================================================================
    ! SURFTSTP_CTL section 4.6 -- bottom-layer gravity drainage added to
    ! the sub-surface runoff diagnostic PROD (closes the water budget).
    ! ==================================================================
    DO JL = 1, KLON_C
      IF (LDLAND(JL)) THEN
        PROD(JL) = PROD(JL) + PSAWGFL(JL,KLEVS_C) * PTMST
      END IF
    END DO

  END SUBROUTINE SURF_REF_SOILW



  !---------------------------------------------------------------------!
  ! SURF_REF_FARQUHAR -- Farquhar photosynthesis + Ball-Berry stomatal  !
  !                       conductance (farquhar_mod.F90, Yin et al 2009).!
  ! --------------------------------------------------------------------!
  ! Standalone reimplementation of the FARQUHAR body.  All hardcoded    !
  ! scalars (sufarquhar_mod.F90:84-258) are passed by VALUE so the ref  !
  ! has zero module-data dependency; the eight per-PFT tables are       !
  ! passed as (0:20,2) INTENT(IN) arrays mirroring the VegTable sentinel!
  ! convention (col 1 = C3, col 2 = C4).  The cubic-equation working    !
  ! variables are spelled JPRD to mirror the upstream (which uses       !
  ! REAL(KIND=JPRD) there); here JPRD == JPRB == C_DOUBLE.              !
  !                                                                     !
  ! The upstream is a two-phase vectorised kernel: a JL setup loop that !
  ! precomputes per-column Vcmax/Jmax/gs coefficients, then a JLEVEL    !
  ! loop over the NLAI levels that gathers the "active" columns (those  !
  ! whose PLAI >= ZLAITAB(JLEVEL)) and solves the C3/C4 cubic per       !
  ! level, accumulating the canopy integral.  This scalar ref folds     !
  ! both phases into a single per-column JL loop: for each column that  !
  ! assimilates, we walk the LAI levels in order, summing every level   !
  ! whose ZLAITAB(JLEVEL) <= PLAI -- exactly the upstream's gather,     !
  ! because the active set is a prefix of the level sequence.           !
  !                                                                     !
  ! Faithfulness notes:                                                 !
  !  * LACCLIMATION = .TRUE. (farquhar_mod.F90:386 default).            !
  !  * LDOWNREGULATION_CO2 = .FALSE. (line 388 default).                !
  !  * LOSM_ACCLIMATION = .FALSE. (line 387) -- ZTEMP_GROWTH =          !
  !    PTSOIL - RTP_00 (the soil proxy, line 493).                      !
  !  * LSKIN_TEMP exposed as an argument (caller picks PTSKM vs PTM).   !
  !  * RR_C = YDCST%R = RNAVO*RKBOL = 8.3145112 (matches the offline    !
  !    driver's SUCST, NOT the stale 8.314 in the source comment).      !
  ! Mirrors surf_pytorch.physics.farquhar.farquhar().
  SUBROUTINE SURF_REF_FARQUHAR( &
      &   KLON_C, NLAI_C, &
      &   LDLAND, KVTYPE, KCO2TYP, &
      &   PSRFD, PAPHM, PQS, PQSURF, PQM, PTM, PTSKM, PTSOIL, PF2, &
      &   PCO2, PLAI, &
      &   RVCMAX25, RHUMREL, RA1, RB1, RG0, RGM25, RE_VCMAX, RE_JMAX, &
      &   RARJV1, RARJV2, RBRJV1, RBRJV2, &
      &   RASJ1, RASJ2, RBSJ1, RBSJ2, RASV1, RASV2, RBSV1, RBSV2, &
      &   RD_VCMAX1, RD_VCMAX2, RD_JMAX1, RD_JMAX2, &
      &   RE_GAMMA_STAR, RE_GM1, RE_GM2, RS_GM1, RS_GM2, &
      &   RS_JMAX1, RS_JMAX2, RD_GM1, RD_GM2, &
      &   RE_KMC, RE_KMO, RE_RD, RE_SCO, RGAMMA_STAR25, &
      &   RKMC25_1, RKMC25_2, RKMO25_1, RKMO25_2, RSCO25_1, RSCO25_2, &
      &   RSTRESS_GM, RSTRESS_GS, RSTRESS_VCMAX, &
      &   RLAI_LEVEL_DEPTH, RLAIMAX, REXT_COEFF, RGB_REF, RTP_00, &
      &   RPB_STD, RRG_TO_PAR, RW_TO_MOL, RALPHA_LL, RTHETA, &
      &   RFPSIR, RFQ, RFPSEUDO, RH_PROTONS, RGBS, ROI, RALPHA, RKP, &
      &   RTETENS_1, RTETENS_2, RREF_TEMP, RMOL_TO_M_1, &
      &   RRATIO_H2O_TO_CO2, RN_VERT_ATT, RPI_C, RR_C, &
      &   RMAIR_C, RMCO2_C, LSKIN_TEMP_C, &
      &   PGSTOT, PGPP, PRD, PAN &
      & ) BIND(C, NAME="surf_ref_farquhar")

    !--- dimensions ------------------------------------------------------
    INTEGER(C_INT), VALUE, INTENT(IN) :: KLON_C, NLAI_C

    !--- inputs (per-column state) ---------------------------------------
    LOGICAL(C_BOOL), VALUE, INTENT(IN) :: LSKIN_TEMP_C
    LOGICAL(C_BOOL), INTENT(IN) :: LDLAND(KLON_C)
    INTEGER(C_INT),  INTENT(IN) :: KVTYPE(KLON_C)   ! veg type 0..20
    INTEGER(C_INT),  INTENT(IN) :: KCO2TYP(KLON_C)  ! photosyn pathway (4=C4)
    REAL(JPRB), INTENT(IN) :: PSRFD(KLON_C), PAPHM(KLON_C)
    REAL(JPRB), INTENT(IN) :: PQS(KLON_C), PQSURF(KLON_C), PQM(KLON_C)
    REAL(JPRB), INTENT(IN) :: PTM(KLON_C), PTSKM(KLON_C), PTSOIL(KLON_C)
    REAL(JPRB), INTENT(IN) :: PF2(KLON_C), PCO2(KLON_C), PLAI(KLON_C)

    !--- per-PFT tables, indexed 0:20 (sentinel mirrors type 8) ---------
    ! Column 1 = C3, column 2 = C4 (1-based in Fortran, matching the
    ! upstream RVCMAX25(:,1/2) indexing).
    REAL(JPRB), INTENT(IN) :: RVCMAX25(0:20,2), RHUMREL(0:20,2)
    REAL(JPRB), INTENT(IN) :: RA1(0:20,2), RB1(0:20,2), RG0(0:20,2)
    REAL(JPRB), INTENT(IN) :: RGM25(0:20,2), RE_VCMAX(0:20,2), RE_JMAX(0:20,2)

    !--- scalars (sufarquhar_mod.F90 hardcoded constants), all VALUE ----
    ! (The 2-element path vectors are flattened to pairs of scalars to
    !  keep the ctypes signature simple and POD-only.)
    REAL(JPRB), VALUE, INTENT(IN) :: RARJV1, RARJV2, RBRJV1, RBRJV2
    REAL(JPRB), VALUE, INTENT(IN) :: RASJ1, RASJ2, RBSJ1, RBSJ2
    REAL(JPRB), VALUE, INTENT(IN) :: RASV1, RASV2, RBSV1, RBSV2
    REAL(JPRB), VALUE, INTENT(IN) :: RD_VCMAX1, RD_VCMAX2, RD_JMAX1, RD_JMAX2
    REAL(JPRB), VALUE, INTENT(IN) :: RE_GAMMA_STAR
    REAL(JPRB), VALUE, INTENT(IN) :: RE_GM1, RE_GM2, RS_GM1, RS_GM2
    REAL(JPRB), VALUE, INTENT(IN) :: RS_JMAX1, RS_JMAX2, RD_GM1, RD_GM2
    REAL(JPRB), VALUE, INTENT(IN) :: RE_KMC, RE_KMO, RE_RD, RE_SCO
    REAL(JPRB), VALUE, INTENT(IN) :: RGAMMA_STAR25
    REAL(JPRB), VALUE, INTENT(IN) :: RKMC25_1, RKMC25_2, RKMO25_1, RKMO25_2
    REAL(JPRB), VALUE, INTENT(IN) :: RSCO25_1, RSCO25_2
    REAL(JPRB), VALUE, INTENT(IN) :: RSTRESS_GM, RSTRESS_GS, RSTRESS_VCMAX
    REAL(JPRB), VALUE, INTENT(IN) :: RLAI_LEVEL_DEPTH, RLAIMAX, REXT_COEFF
    REAL(JPRB), VALUE, INTENT(IN) :: RGB_REF, RTP_00, RPB_STD, RRG_TO_PAR
    REAL(JPRB), VALUE, INTENT(IN) :: RW_TO_MOL, RALPHA_LL, RTHETA
    REAL(JPRB), VALUE, INTENT(IN) :: RFPSIR, RFQ, RFPSEUDO, RH_PROTONS
    REAL(JPRB), VALUE, INTENT(IN) :: RGBS, ROI, RALPHA, RKP
    REAL(JPRB), VALUE, INTENT(IN) :: RTETENS_1, RTETENS_2, RREF_TEMP
    REAL(JPRB), VALUE, INTENT(IN) :: RMOL_TO_M_1, RRATIO_H2O_TO_CO2
    REAL(JPRB), VALUE, INTENT(IN) :: RN_VERT_ATT
    REAL(JPRB), VALUE, INTENT(IN) :: RPI_C, RR_C
    REAL(JPRB), VALUE, INTENT(IN) :: RMAIR_C, RMCO2_C

    !--- outputs ---------------------------------------------------------
    REAL(JPRB), INTENT(OUT) :: PGSTOT(KLON_C), PGPP(KLON_C)
    REAL(JPRB), INTENT(OUT) :: PRD(KLON_C), PAN(KLON_C)

    !--- locals ----------------------------------------------------------
    INTEGER :: JL, JLEVEL, LIMIT_PHOTO
    INTEGER :: KVTYPE_TEMPO, ZCTYPE
    REAL(JPRB) :: ZPB, ZCO2, ZVPD, ZWATER_LIM
    REAL(JPRB) :: ZVCMAX25, ZJMAX25, ZT_KMC, ZT_KMO, ZT_SCO, ZT_GAMMA_STAR
    REAL(JPRB) :: ZT_RD, ZT_JMAX, ZT_VCMAX, ZT_GM
    REAL(JPRB) :: ZS_JMAX_ACCLIMTEMP, ZS_VCMAX_ACCLIMTEMP
    REAL(JPRB) :: ZVC, ZVJ, ZGM, ZG0VAR, ZKMC, ZKMO, ZSCO, ZFVPD
    REAL(JPRB) :: ZGAMMA_STAR, ZLOW_GAMMA_STAR, ZGBH2O, ZGBCO2
    REAL(JPRB) :: ZNITRO_VCMAX, ZIABS, ZJMAX, ZTEMP_GROWTH, ZLEAF_TEMP
    REAL(JPRB) :: ZRVEGET, ZGSTOP, ZRDTOT, ZASSIMTOT, ZLEAF_GS_TOP, ZILAI
    REAL(JPRB) :: ZASSIMI, ZGS, ZRD, ZVC2, ZVJ2, ZJJ
    REAL(JPRB) :: RARJV(2), RBRJV(2), RASJ(2), RBSJ(2), RASV(2), RBSV(2)
    REAL(JPRB) :: RD_VCMAX(2), RD_JMAX(2), RE_GM(2), RS_GM(2)
    REAL(JPRB) :: RS_JMAX(2), RD_GM(2), RKMC25(2), RKMO25(2), RSCO25(2)
    REAL(JPRB), PARAMETER :: RUNDEF_SECHIBA = 1.0E+20_JPRB
    ! LAI-level arrays (NLAI1 = NLAI+1 boundaries, NLAI light fractions).
    REAL(JPRB) :: ZLAITAB(NLAI_C+1), ZLIGHT(NLAI_C)

    ! JPRD working variables for the C3/C4 cubic equations (lines 863-1127).
    REAL(JPRD) :: ZA, ZB, ZX1, ZX2, ZX3
    REAL(JPRD) :: ZM, ZF, ZJc, ZG, ZH, ZI, ZL, ZP, ZQ, ZR
    REAL(JPRD) :: ZQQ, ZUU, ZPSI, ZA_1, ZA_1_tmp, ZA_3, ZA_3_tmp
    REAL(JPRD) :: ZFCYC, ZRM, ZCS_STAR, ZJ2, ZZc, ZVPJ2
    REAL(JPRD) :: ZCC, ZLEAF_CI, ZOBS, ZCI_STAR
    ! C3/C4 cubic coefficients (upstream ZC/ZD, JPRD scalars; ZC only in C3).
    REAL(JPRD) :: ZC, ZD

    ! Default configuration flags (farquhar_mod.F90:386-388).
    LOGICAL, PARAMETER :: LACCLIMATION = .TRUE.
    LOGICAL, PARAMETER :: LDOWNREGULATION_CO2 = .FALSE.

    ! Reassemble the 2-element path vectors for clean C3/C4 indexing.
    RARJV(1)=RARJV1; RARJV(2)=RARJV2
    RBRJV(1)=RBRJV1; RBRJV(2)=RBRJV2
    RASJ(1)=RASJ1;   RASJ(2)=RASJ2
    RBSJ(1)=RBSJ1;   RBSJ(2)=RBSJ2
    RASV(1)=RASV1;   RASV(2)=RASV2
    RBSV(1)=RBSV1;   RBSV(2)=RBSV2
    RD_VCMAX(1)=RD_VCMAX1; RD_VCMAX(2)=RD_VCMAX2
    RD_JMAX(1)=RD_JMAX1;   RD_JMAX(2)=RD_JMAX2
    RE_GM(1)=RE_GM1;   RE_GM(2)=RE_GM2
    RS_GM(1)=RS_GM1;   RS_GM(2)=RS_GM2
    RS_JMAX(1)=RS_JMAX1; RS_JMAX(2)=RS_JMAX2
    RD_GM(1)=RD_GM1;   RD_GM(2)=RD_GM2
    RKMC25(1)=RKMC25_1; RKMC25(2)=RKMC25_2
    RKMO25(1)=RKMO25_1; RKMO25(2)=RKMO25_2
    RSCO25(1)=RSCO25_1; RSCO25(2)=RSCO25_2

    ! 1. LAI levels + Beer light profile (lines 506-517) -----------------
    DO JLEVEL = 1, NLAI_C+1
      ZLAITAB(JLEVEL) = RLAIMAX*(EXP(RLAI_LEVEL_DEPTH*REAL(JLEVEL-1,JPRB))-1.0_JPRB)/ &
                        (EXP(RLAI_LEVEL_DEPTH*REAL(NLAI_C,JPRB))-1.0_JPRB)
    ENDDO
    DO JLEVEL = 1, NLAI_C
      ZLIGHT(JLEVEL) = EXP(-REXT_COEFF*ZLAITAB(JLEVEL))
    ENDDO

    ! Initialise outputs (per-column) to zero.
    PGSTOT(:) = 0.0_JPRB
    PGPP(:)   = 0.0_JPRB
    PRD(:)    = 0.0_JPRB
    PAN(:)    = 0.0_JPRB

    !-------------------------------------------------------------------
    ! Main per-column loop.  Folds the upstream JL setup loop (418-829)
    ! and the JLEVEL LAI loop (841-1179) into one scalar pass per column.
    !-------------------------------------------------------------------
    DO JL = 1, KLON_C

      ZASSIMTOT = 0.0_JPRB
      ZRDTOT    = 0.0_JPRB
      ZLEAF_GS_TOP = 0.0_JPRB
      ZGSTOP    = 0.0_JPRB
      ZILAI     = 1.0_JPRB
      ZRVEGET   = RUNDEF_SECHIBA

      ! Default-branch outputs (non-assimilating columns stay zero).
      IF (.NOT. LDLAND(JL)) CYCLE

      ZPB  = PAPHM(JL)/100.0_JPRB
      ZCO2 = REAL(REAL(PCO2(JL),JPRD)*1.0E6_JPRD*RMAIR_C/RMCO2_C, JPRB)

      ! ZCTYPE (lines 443-446): 1=C3, 2=C4 when KCO2TYP==4.
      ZCTYPE = 1
      IF (KCO2TYP(JL) == 4) ZCTYPE = 2

      ! KVTYPE protection (lines 523-527).
      IF (KVTYPE(JL) > 0) THEN
        KVTYPE_TEMPO = KVTYPE(JL)
      ELSE
        KVTYPE_TEMPO = 1
      ENDIF

      ! ZVCMAX25 (lines 530-545) -- downregulation branch never taken.
      ZVCMAX25 = RVCMAX25(KVTYPE_TEMPO, ZCTYPE)

      ! VPD (lines 554-561), in kPa.  PQSURF unused in the active path
      ! (qsat passed via PQS); PQSURF kept in the signature for fidelity.
      ZVPD = (PQS(JL)*ZPB/(RTETENS_1+PQS(JL)*RTETENS_2)) &
           - (PQM(JL)*ZPB/(RTETENS_1+PQM(JL)*RTETENS_2))
      ZVPD = ZVPD/10.0_JPRB

      ! ZTEMP_GROWTH / ZLEAF_TEMP (lines 466-493, 460-464).
      ZTEMP_GROWTH = PTSOIL(JL) - RTP_00
      IF (LSKIN_TEMP_C) THEN
        ZLEAF_TEMP = PTSKM(JL)
      ELSE
        ZLEAF_TEMP = PTM(JL)
      ENDIF

      ! Assimilation gate (line 588): land .AND. PLAI>0.01 .AND. KVTYPE>0.
      IF (.NOT. (PLAI(JL) > 0.01_JPRB .AND. KVTYPE(JL) > 0)) CYCLE

      ZWATER_LIM = MIN(1.0_JPRB, PF2(JL)*RHUMREL(KVTYPE_TEMPO,ZCTYPE))

      ! Day respiration (lines 599-606): computed for all NLAI levels even
      ! at nighttime (Rd is needed for the canopy integral regardless).
      ZT_RD = EXP(((ZLEAF_TEMP-RREF_TEMP)*RE_RD)/(RREF_TEMP*RR_C*ZLEAF_TEMP))

      ! Assimilation preconditions (line 609-612).
      L_ASSIM: IF (PSRFD(JL) > EPSILON(1.0_JPRB) .AND. PF2(JL) > EPSILON(1.0_JPRB) &
            .AND. ZTEMP_GROWTH > -4.0_JPRB .AND. ZTEMP_GROWTH < 55.0_JPRB) THEN

        ! Arrhenius scalings (lines 637-640).
        ZT_KMC        = EXP(((ZLEAF_TEMP-RREF_TEMP)*RE_KMC)/(RREF_TEMP*RR_C*ZLEAF_TEMP))
        ZT_KMO        = EXP(((ZLEAF_TEMP-RREF_TEMP)*RE_KMO)/(RREF_TEMP*RR_C*ZLEAF_TEMP))
        ZT_SCO        = EXP(((ZLEAF_TEMP-RREF_TEMP)*RE_SCO)/(RREF_TEMP*RR_C*ZLEAF_TEMP))
        ZT_GAMMA_STAR = EXP(((ZLEAF_TEMP-RREF_TEMP)*RE_GAMMA_STAR)/(RREF_TEMP*RR_C*ZLEAF_TEMP))

        ZVC = 0.0_JPRB
        ZVJ = 0.0_JPRB

        IF (ZCTYPE == 1) THEN
          !--- C3 (lines 666-716) -------------------------------------
          IF (LACCLIMATION) THEN
            ZS_JMAX_ACCLIMTEMP = RASJ(1) + RBSJ(1)*MAX(11.0_JPRB,MIN(ZTEMP_GROWTH,35.0_JPRB))
            ZT_JMAX = ARRHENIUS_MOD(ZLEAF_TEMP,RREF_TEMP,RE_JMAX(KVTYPE_TEMPO,1), &
                      RD_JMAX(1), ZS_JMAX_ACCLIMTEMP, RR_C)
            ZS_VCMAX_ACCLIMTEMP = RASV(1) + RBSV(1)*MAX(11.0_JPRB,MIN(ZTEMP_GROWTH,35.0_JPRB))
            ZT_VCMAX = ARRHENIUS_MOD(ZLEAF_TEMP,RREF_TEMP,RE_VCMAX(KVTYPE_TEMPO,1), &
                      RD_VCMAX(1), ZS_VCMAX_ACCLIMTEMP, RR_C)
            ZJMAX25 = (RARJV(1)+RBRJV(1)*MAX(11.0_JPRB,MIN(ZTEMP_GROWTH,35.0_JPRB)))*ZVCMAX25
          ELSE
            ZT_JMAX = ARRHENIUS_MOD(ZLEAF_TEMP,RREF_TEMP,RE_JMAX(KVTYPE_TEMPO,1), &
                      RD_JMAX(1), RS_JMAX(1), RR_C)
            ZT_VCMAX = EXP(((ZLEAF_TEMP-RREF_TEMP)*RE_VCMAX(KVTYPE_TEMPO,1))/(RREF_TEMP*RR_C*ZLEAF_TEMP))
            ZJMAX25 = (RARJV(1)+RBRJV(1)*25.0_JPRB)*ZVCMAX25
          ENDIF
          ZVJ = ZJMAX25*ZT_JMAX
          ZVC = ZVCMAX25*ZT_VCMAX
          ZT_GM = ARRHENIUS_MOD(ZLEAF_TEMP,RREF_TEMP,RE_GM(1),RD_GM(1),RS_GM(1),RR_C)
          ZGM = RGM25(KVTYPE_TEMPO,1)*ZT_GM*MAX(1.0_JPRB-RSTRESS_GM, ZWATER_LIM)
          ZG0VAR = RG0(KVTYPE_TEMPO,1)*MAX(1.0_JPRB-RSTRESS_GS, ZWATER_LIM)
          ZKMC = RKMC25(1)*ZT_KMC
          ZKMO = RKMO25(1)*ZT_KMO
          ZSCO = RSCO25(1)*ZT_SCO
          ZFVPD = 1.0_JPRB / (1.0_JPRB/MIN(1.0_JPRB-EPSILON(1.0_JPRB), &
                  MAX(EPSILON(1.0_JPRB),(RA1(KVTYPE_TEMPO,1)-RB1(KVTYPE_TEMPO,1)*ZVPD))) - 1.0_JPRB) &
                * MAX(1.0_JPRB-RSTRESS_GS, ZWATER_LIM)

        ELSEIF (ZCTYPE == 2) THEN
          !--- C4 (lines 717-765) -------------------------------------
          IF (LACCLIMATION) THEN
            ZS_JMAX_ACCLIMTEMP = RASJ(2) + RBSJ(2)*MAX(11.0_JPRB,MIN(ZTEMP_GROWTH,35.0_JPRB))
            ZT_JMAX = ARRHENIUS_MOD(ZLEAF_TEMP,RREF_TEMP,RE_JMAX(KVTYPE_TEMPO,2), &
                      RD_JMAX(2), ZS_JMAX_ACCLIMTEMP, RR_C)
            ZS_VCMAX_ACCLIMTEMP = RASV(2) + RBSV(2)*MAX(11.0_JPRB,MIN(ZTEMP_GROWTH,35.0_JPRB))
            ZT_VCMAX = ARRHENIUS_MOD(ZLEAF_TEMP,RREF_TEMP,RE_VCMAX(KVTYPE_TEMPO,2), &
                      RD_VCMAX(2), ZS_VCMAX_ACCLIMTEMP, RR_C)
            ZVJ = (RARJV(2)+RBRJV(2)*MAX(11.0_JPRB,MIN(ZTEMP_GROWTH,35.0_JPRB)))*ZVCMAX25*ZT_JMAX
            ZVC = ZVCMAX25*ZT_VCMAX
          ELSE
            ZJMAX25 = (RARJV(2)+RBRJV(2)*25.0_JPRB)*ZVCMAX25
            ZVJ = ZJMAX25*ARRHENIUS_MOD(ZLEAF_TEMP,RREF_TEMP,RE_JMAX(KVTYPE_TEMPO,2), &
                 RD_JMAX(2), RS_JMAX(2), RR_C)
            ZVC = ZVCMAX25*EXP(((ZLEAF_TEMP-RREF_TEMP)*RE_VCMAX(KVTYPE_TEMPO,2))/(RREF_TEMP*RR_C*ZLEAF_TEMP))
          ENDIF
          ! gm not used for C4.
          ZG0VAR = RG0(KVTYPE_TEMPO,2)*MAX(1.0_JPRB-RSTRESS_GS, ZWATER_LIM)
          ZKMC = RKMC25(2)*ZT_KMC
          ZKMO = RKMO25(2)*ZT_KMO
          ZSCO = RSCO25(2)*ZT_SCO
          ZFVPD = 1.0_JPRB / (1.0_JPRB/MIN(1.0_JPRB-EPSILON(1.0_JPRB), &
                  MAX(EPSILON(1.0_JPRB),(RA1(KVTYPE_TEMPO,2)-RB1(KVTYPE_TEMPO,2)*ZVPD))) - 1.0_JPRB) &
                * MAX(1.0_JPRB-RSTRESS_GS, ZWATER_LIM)
        ELSE
          ZSCO = RSCO25(2)
        ENDIF

        ZGAMMA_STAR = RGAMMA_STAR25*ZT_GAMMA_STAR
        ZLOW_GAMMA_STAR = 0.5_JPRB/ZSCO
        ZGBH2O = RGB_REF*44.6_JPRB*(RTP_00/PTM(JL))*(ZPB/RPB_STD)
        ZGBCO2 = ZGBH2O/RRATIO_H2O_TO_CO2

        !--- LAI-level loop (upstream 841-1179), scalar per column ------
        ! Walk levels in order; sum every level whose ZLAITAB(JLEVEL) <=
        ! PLAI (the upstream gather's active set is a level prefix).
        DO JLEVEL = 1, NLAI_C
          IF (.NOT. (ZLAITAB(JLEVEL) <= PLAI(JL))) CYCLE

          ! Per-level Rd (lines 600-606).
          ZNITRO_VCMAX = (1.0_JPRB - RN_VERT_ATT*(1.0_JPRB - ZLIGHT(JLEVEL)))
          ZRD = ZVCMAX25*ZNITRO_VCMAX*0.01_JPRB*ZT_RD &
               * MAX(1.0_JPRB-RSTRESS_VCMAX, ZWATER_LIM)

          ! Vc/J scaled into the canopy (lines 786-788).
          ZVC2 = ZVC*ZNITRO_VCMAX*MAX(1.0_JPRB-RSTRESS_VCMAX, ZWATER_LIM)
          ZVJ2 = ZVJ*ZNITRO_VCMAX*MAX(1.0_JPRB-RSTRESS_VCMAX, ZWATER_LIM)

          ! eq. 4 of Yin et al (2009) -- electron transport rate J.
          ZIABS = PSRFD(JL)*RW_TO_MOL*RRG_TO_PAR*REXT_COEFF*ZLIGHT(JLEVEL)
          ZJMAX = ZVJ2
          ZJJ = (RALPHA_LL*ZIABS + ZJMAX - &
                SQRT((RALPHA_LL*ZIABS+ZJMAX)**2 - 4.0_JPRB*RTHETA*ZJMAX*RALPHA_LL*ZIABS)) &
                /(2.0_JPRB*RTHETA)

          IF (ZCTYPE == 2) THEN
            !--- C4 cubic (lines 863-1012) -----------------------------
            ZFCYC = 1.0_JPRD - (4.0_JPRD*(1.0_JPRD-REAL(RFPSIR,JPRD))* &
                   (1.0_JPRD+REAL(RFQ,JPRD))+3.0_JPRD*REAL(RH_PROTONS,JPRD)* &
                   REAL(RFPSEUDO,JPRD))/(3.0_JPRD*REAL(RH_PROTONS,JPRD) - &
                   4.0_JPRD*(1.0_JPRD-REAL(RFPSIR,JPRD)))
            ZRM = REAL(ZRD,JPRD)/2.0_JPRD
            ZCS_STAR = (REAL(RGBS,JPRD)*REAL(ZLOW_GAMMA_STAR,JPRD)*REAL(ROI,JPRD) - &
                   (1.0_JPRD+REAL(ZLOW_GAMMA_STAR,JPRD)*REAL(RALPHA,JPRD)/0.047_JPRD)* &
                   REAL(ZRD,JPRD) + REAL(ZRM,JPRD)) / (REAL(RGBS,JPRD)+REAL(RKP,JPRD))
            ZJ2 = REAL(ZJJ,JPRD)/(1.0_JPRD-REAL(RFPSEUDO,JPRD)/(1.0_JPRD-ZFCYC))
            ZZc = (2.0_JPRD+REAL(RFQ,JPRD)-ZFCYC)/(REAL(RH_PROTONS,JPRD)*(1.0_JPRD-ZFCYC))
            ZVPJ2 = REAL(RFPSIR,JPRD)*ZJ2*ZZc/2.0_JPRD
            ZA_3 = 9999.0_JPRD

            DO LIMIT_PHOTO = 1, 2
              IF (LIMIT_PHOTO == 1) THEN
                ZA = 1.0_JPRD + REAL(RKP,JPRD)/REAL(RGBS,JPRD)
                ZB = 0.0_JPRD
                ZX1 = REAL(ZVC2,JPRD)
                ZX2 = REAL(ZKMC,JPRD)/REAL(ZKMO,JPRD)
                ZX3 = REAL(ZKMC,JPRD)
              ELSE
                ZA = 1.0_JPRD
                ZB = ZVPJ2
                ZX1 = (1.0_JPRD-REAL(RFPSIR,JPRD))*ZJ2*ZZc/3.0_JPRD
                ZX2 = 7.0_JPRD*REAL(ZLOW_GAMMA_STAR,JPRD)/3.0_JPRD
                ZX3 = 0.0_JPRD
              ENDIF
              ZM = REAL(ZFVPD,JPRD) - REAL(ZG0VAR,JPRD)/REAL(ZGBCO2,JPRD)
              ZD = REAL(ZG0VAR,JPRD)*(REAL(ZCO2,JPRD)-ZCS_STAR) + &
                   REAL(ZFVPD,JPRD)*REAL(ZRD,JPRD)
              ZF = (ZB-ZRM-REAL(ZLOW_GAMMA_STAR,JPRD)*REAL(ROI,JPRD)* &
                   REAL(RGBS,JPRD))*ZX1*ZD + ZA*REAL(RGBS,JPRD)*ZX1* &
                   REAL(ZCO2,JPRD)*ZD
              ZJc = (ZB-ZRM+REAL(RGBS,JPRD)*ZX3+ZX2*REAL(RGBS,JPRD)* &
                   REAL(ROI,JPRD))*ZM + (REAL(RALPHA,JPRD)*ZX2/0.047_JPRD-1.0_JPRD)*ZD + &
                   ZA*REAL(RGBS,JPRD)*(REAL(ZCO2,JPRD)*ZM - ZD/REAL(ZGBCO2,JPRD) - &
                   (REAL(ZCO2,JPRD)-ZCS_STAR))
              ZG = (ZB-ZRM-REAL(ZLOW_GAMMA_STAR,JPRD)*REAL(ROI,JPRD)* &
                   REAL(RGBS,JPRD))*ZX1*ZM - (REAL(RALPHA,JPRD)* &
                   REAL(ZLOW_GAMMA_STAR,JPRD)/0.047_JPRD+1.0_JPRD)*ZX1*ZD + &
                   ZA*REAL(RGBS,JPRD)*ZX1*(REAL(ZCO2,JPRD)*ZM - &
                   ZD/REAL(ZGBCO2,JPRD) - (REAL(ZCO2,JPRD)-ZCS_STAR))
              ZH = -((REAL(RALPHA,JPRD)*REAL(ZLOW_GAMMA_STAR,JPRD)/ &
                   0.047_JPRD+1.0_JPRD)*ZX1*ZM + (ZA*REAL(RGBS,JPRD)* &
                   ZX1*(ZM-1.0_JPRD))/REAL(ZGBCO2,JPRD))
              ZI = (ZB-ZRM+REAL(RGBS,JPRD)*ZX3+ZX2*REAL(RGBS,JPRD)* &
                   REAL(ROI,JPRD))*ZD + ZA*REAL(RGBS,JPRD)*REAL(ZCO2,JPRD)*ZD
              ZL = (REAL(RALPHA,JPRD)*ZX2/0.047_JPRD-1.0_JPRD)*ZM - &
                   (ZA*REAL(RGBS,JPRD)*(ZM-1.0_JPRD))/REAL(ZGBCO2,JPRD)
              ZP = (ZJc-(ZH-ZL*REAL(ZRD,JPRD)))/ZL
              ZQ = (ZI+ZJc*REAL(ZRD,JPRD)-ZG)/ZL
              ZR = -(ZF-ZI*REAL(ZRD,JPRD))/ZL
              ZQQ = ((ZP**2.0_JPRD) - 3.0_JPRD*ZQ)/9.0_JPRD
              ZUU = (2.0_JPRD*(ZP**3.0_JPRD) - 9.0_JPRD*ZP*ZQ + 27.0_JPRD*ZR)/54.0_JPRD
              IF ((ZQQ >= 0.0_JPRD) .AND. (ABS(ZUU/(ABS(ZQQ)**1.5_JPRD)) <= 1.0_JPRD)) THEN
                ZPSI = ACOS(MAX(-1.0_JPRD,MIN(ZUU/(ZQQ**1.5_JPRD),1.0_JPRD)))
                ZA_3_tmp = -2.0_JPRD*SQRT(ZQQ)*COS((ZPSI+4.0_JPRD*REAL(RPI_C,JPRD))/3.0_JPRD) &
                        - ZP/3.0_JPRD
                IF (ZA_3_tmp < ZA_3) THEN
                  ZA_3 = ZA_3_tmp
                ELSE
                  ! Re-initialise for Vc-limited assimilation.
                  ZA = 1.0_JPRD + REAL(RKP,JPRD)/REAL(RGBS,JPRD)
                  ZB = 0.0_JPRD
                  ZX1 = REAL(ZVC2,JPRD)
                  ZX2 = REAL(ZKMC,JPRD)/REAL(ZKMO,JPRD)
                  ZX3 = REAL(ZKMC,JPRD)
                ENDIF
              ENDIF

              ! NOTE: upstream farquhar_mod.F90 places the ZASSIMI assignment
              ! and the entire ZGS closure (eqs. 24/22/25 of Yin 2009) INSIDE
              ! the DO LIMIT_PHOTO loop (lines 971-1011), so ZGS is computed
              ! for both limiting processes and the loop-exit value reflects
              ! the 2nd iteration (J-limited).  This is reproduced verbatim:
              ! the structure matters for bit-exact agreement with OpenIFS.
              IF ((ZA_3 == 9999.0_JPRD) .OR. (ZA_3 < (-REAL(ZRD,JPRD)))) THEN
                ZA_3 = -REAL(ZRD,JPRD)
              ENDIF
              ZASSIMI = REAL(ZA_3,JPRB)

              IF (ABS(ZASSIMI + ZRD) < EPSILON(1.0_JPRB)) THEN
                ZGS = ZG0VAR
              ELSE
                ! eq. 24 / 22 of Yin et al. (2009).
                ZOBS = (REAL(RALPHA,JPRD)*REAL(ZASSIMI,JPRD))/(0.047_JPRD*REAL(RGBS,JPRD)) &
                     + REAL(ROI,JPRD)
                ZCC = ((REAL(ZASSIMI,JPRD)+REAL(ZRD,JPRD))*(ZX2*ZOBS+ZX3) + &
                     REAL(ZLOW_GAMMA_STAR,JPRD)*ZOBS*ZX1) / &
                     MAX(EPSILON(1.0_JPRB), ZX1-(REAL(ZASSIMI,JPRD)+REAL(ZRD,JPRD)))
                ZLEAF_CI = (ZCC-(ZB-REAL(ZASSIMI,JPRD)-REAL(ZRM,JPRD))/REAL(RGBS,JPRD))/ZA
                ! eq. 25 of Yin et al. (2009).
                ZGS = REAL(REAL(ZG0VAR,JPRD) + (REAL(ZASSIMI,JPRD)+REAL(ZRD,JPRD))/ &
                     (REAL(ZCO2,JPRD)-ZCS_STAR)*REAL(ZFVPD,JPRD), JPRB)
              ENDIF
            ENDDO  ! LIMIT_PHOTO

          ELSEIF (ZCTYPE == 1) THEN
            !--- C3 cubic (lines 1014-1127) ----------------------------
            ZA_1 = 9999.0_JPRD
            DO LIMIT_PHOTO = 1, 2
              IF (LIMIT_PHOTO == 1) THEN
                ZX1 = REAL(ZVC2,JPRD)
                ZX2 = REAL(ZKMC,JPRD)*(1.0_JPRD+2.0_JPRD*REAL(ZGAMMA_STAR,JPRD)* &
                     REAL(ZSCO,JPRD)/REAL(ZKMO,JPRD))
              ELSE
                ZX1 = REAL(ZJJ,JPRD)/4.0_JPRD
                ZX2 = 2.0_JPRD*REAL(ZGAMMA_STAR,JPRD)
              ENDIF
              ! Appendix B of Yin et al. (2009).
              ZA = REAL(ZG0VAR,JPRD)*(ZX2+REAL(ZGAMMA_STAR,JPRD)) + &
                   (REAL(ZG0VAR,JPRD)/REAL(ZGM,JPRD)+REAL(ZFVPD,JPRD))* &
                   (ZX1-REAL(ZRD,JPRD))
              ZB = REAL(ZCO2,JPRD)*(ZX1-REAL(ZRD,JPRD)) - &
                   REAL(ZGAMMA_STAR,JPRD)*ZX1 - REAL(ZRD,JPRD)*ZX2
              ZC = REAL(ZCO2,JPRD) + ZX2 + (1.0_JPRD/REAL(ZGM,JPRD)+ &
                   1.0_JPRD/REAL(ZGBCO2,JPRD))*(ZX1-REAL(ZRD,JPRD))
              ZD = ZX2 + REAL(ZGAMMA_STAR,JPRD) + (ZX1-REAL(ZRD,JPRD))/REAL(ZGM,JPRD)
              ZM = 1.0_JPRD/REAL(ZGM,JPRD) + (REAL(ZG0VAR,JPRD)/REAL(ZGM,JPRD)+ &
                   REAL(ZFVPD,JPRD))*(1.0_JPRD/REAL(ZGM,JPRD)+1.0_JPRD/REAL(ZGBCO2,JPRD))
              ZP = -(ZD + (ZX1-REAL(ZRD,JPRD))/REAL(ZGM,JPRD) + ZA* &
                   (1.0_JPRD/REAL(ZGM,JPRD)+1.0_JPRD/REAL(ZGBCO2,JPRD)) + &
                   (REAL(ZG0VAR,JPRD)/REAL(ZGM,JPRD)+REAL(ZFVPD,JPRD))*ZC)/ZM
              ZQ = (ZD*(ZX1-REAL(ZRD,JPRD)) + ZA*ZC + (REAL(ZG0VAR,JPRD)/ &
                   REAL(ZGM,JPRD)+REAL(ZFVPD,JPRD))*ZB)/ZM
              ZR = -ZA*ZB/ZM
              ZQQ = ((ZP**2.0_JPRD) - 3.0_JPRD*ZQ)/9.0_JPRD
              ZUU = (2.0_JPRD*(ZP**3.0_JPRD) - 9.0_JPRD*ZP*ZQ + 27.0_JPRD*ZR)/54.0_JPRD
              IF ((ZQQ >= 0.0_JPRD) .AND. (ABS(ZUU/(ZQQ**1.5_JPRD)) <= 1.0_JPRD)) THEN
                ZPSI = ACOS(ZUU/(ZQQ**1.5_JPRD))
                ZA_1_tmp = -2.0_JPRD*SQRT(ZQQ)*COS(ZPSI/3.0_JPRD) - ZP/3.0_JPRD
                IF (ZA_1_tmp < ZA_1) THEN
                  ZA_1 = ZA_1_tmp
                ELSE
                  ! Re-initialise x1/x2 for Vc-limited assimilation.
                  ZX1 = REAL(ZVC2,JPRD)
                  ZX2 = REAL(ZKMC,JPRD)*(1.0_JPRD+2.0_JPRD*REAL(ZGAMMA_STAR,JPRD)* &
                       REAL(ZSCO,JPRD)/REAL(ZKMO,JPRD))
                ENDIF
              ENDIF
            ENDDO  ! LIMIT_PHOTO

            IF ((ZA_1 == 9999.0_JPRD) .OR. (ZA_1 < (-REAL(ZRD,JPRD)))) THEN
              ZA_1 = -REAL(ZRD,JPRD)
            ENDIF
            ZASSIMI = REAL(ZA_1,JPRB)

            IF (ABS(ZASSIMI + ZRD) < EPSILON(1.0_JPRB)) THEN
              ZGS = ZG0VAR
            ELSE
              ! eq. 18 / 17 of Yin et al. (2009).
              ZCC = (REAL(ZGAMMA_STAR,JPRD)*ZX1 + (REAL(ZASSIMI,JPRD)+ &
                   REAL(ZRD,JPRD))*ZX2) / MAX(EPSILON(1.0_JPRB), &
                   ZX1-(REAL(ZASSIMI,JPRD)+REAL(ZRD,JPRD)))
              ZLEAF_CI = ZCC + REAL(ZASSIMI,JPRD)/REAL(ZGM,JPRD)
              ZCI_STAR = REAL(ZGAMMA_STAR,JPRD) - REAL(ZRD,JPRD)/REAL(ZGM,JPRD)
              ! eq. 15 of Yin et al. (2009).
              ZGS = REAL(REAL(ZG0VAR,JPRD) + (REAL(ZASSIMI,JPRD)+REAL(ZRD,JPRD))/ &
                   (ZLEAF_CI-ZCI_STAR)*REAL(ZFVPD,JPRD), JPRB)
            ENDIF

          ELSE
            ZASSIMI = 0.0_JPRB
            ZGS = ZG0VAR
          ENDIF  ! C3/C4

          ! Top-level conductance (line 1149-1151).
          IF (JLEVEL == 1) ZLEAF_GS_TOP = ZGS

          ! Canopy integration (lines 1159-1164).
          ZASSIMTOT = ZASSIMTOT + (ZASSIMI+ZRD)*(ZLAITAB(JLEVEL+1)-ZLAITAB(JLEVEL))
          PGSTOT(JL) = PGSTOT(JL) + ZGS*(ZLAITAB(JLEVEL+1)-ZLAITAB(JLEVEL))
          ZILAI = REAL(JLEVEL,JPRB)
        ENDDO  ! JLEVEL

      ENDIF L_ASSIM  ! assimilation preconditions

      ! Canopy dark-respiration integral (lines 1174-1177): summed over
      ! ALL NLAI levels regardless of the assimilation gate (Rd is needed
      ! at nighttime too).  Mirrors the upstream JL loop outside the gate.
      DO JLEVEL = 1, NLAI_C
        ZNITRO_VCMAX = (1.0_JPRB - RN_VERT_ATT*(1.0_JPRD - ZLIGHT(JLEVEL)))
        ZRD = ZVCMAX25*ZNITRO_VCMAX*0.01_JPRB*ZT_RD &
             * MAX(1.0_JPRB-RSTRESS_VCMAX, ZWATER_LIM)
        ZRDTOT = ZRDTOT + ZRD*(ZLAITAB(JLEVEL+1)-ZLAITAB(JLEVEL))
      ENDDO

      ! If this column never assimilated, PGSTOT stays 0 and the unit
      ! conversions below are skipped (gate on ZLEAF_GS_TOP / ZASSIMTOT).
      IF (ZLEAF_GS_TOP > 0.0_JPRB .OR. ZASSIMTOT /= 0.0_JPRB) THEN
        ! Unit conversions (lines 1216, 1237-1242, 1305-1306).
        PGPP(JL) = ZASSIMTOT*1.0E-6_JPRB*RMCO2_C
        PGSTOT(JL) = RMOL_TO_M_1*(PTM(JL)/RTP_00)*(RPB_STD/ZPB)* &
                     PGSTOT(JL)*RRATIO_H2O_TO_CO2
        ZGSTOP = RMOL_TO_M_1*(PTM(JL)/RTP_00)*(RPB_STD/ZPB)* &
                 ZLEAF_GS_TOP*RRATIO_H2O_TO_CO2*ZLAITAB(NINT(ZILAI)+1)
        IF (ZGSTOP < EPSILON(1.0_JPRB)) THEN
          ZRVEGET = RUNDEF_SECHIBA
        ELSE
          ZRVEGET = 1.0_JPRB/ZGSTOP
        ENDIF
        PAN(JL) = (ZASSIMTOT-ZRDTOT)*1.0E-6_JPRB*RMCO2_C
        PRD(JL) = ZRDTOT*1.0E-6_JPRB*RMCO2_C
      ENDIF

    ENDDO  ! JL

  CONTAINS

    ! Arrhenius (modified) inline helper (farquhar_mod.F90:48-67).
    ! Host-associated scalars come from the parent; result is JPRB.
    REAL(JPRB) FUNCTION ARRHENIUS_MOD(PTEMP,PREF,PEACT,PDEACT,PENTR,PRR)
      REAL(JPRB), INTENT(IN) :: PTEMP, PREF, PEACT, PDEACT, PENTR, PRR
      ARRHENIUS_MOD = EXP(((PTEMP-PREF)*PEACT)/(PREF*PRR*PTEMP)) &
          * (1.0_JPRB + EXP((PREF*PENTR-PDEACT)/(PREF*PRR))) &
          / (1.0_JPRB + EXP((PTEMP*PENTR-PDEACT)/(PRR*PTEMP)))
    END FUNCTION ARRHENIUS_MOD

  END SUBROUTINE SURF_REF_FARQUHAR

END MODULE SURF_REF
