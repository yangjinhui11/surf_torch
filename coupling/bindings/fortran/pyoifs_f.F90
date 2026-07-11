! ========================================================================== !
!  pyoifs_f.F90 -- Fortran thunk between OpenIFS SURF and the PyTorch port.
!
!  This is the P1.3 component of the offline coupling (Option A: in-process
!  CPython embedding).  It is the Fortran face of the three-layer bridge
!  OpenIFS -> libpyoifs.so -> libpython3.10 -> surf_pytorch:
!
!     callpar1s.F90  (LEPYOIFS branch)
!          |  CALL PYOIFS_SURFTSTP(...)
!          v
!     pyoifs_f.F90   (THIS FILE -- MODULE PYOIFS_F)
!          |  BIND(C) interface -> C symbols
!          v
!     pyoifs_c.c     (libpyoifs.so)
!          |  PyGILState_Ensure / PyObject_CallObject
!          v
!     pyoifs_shim.py (PyOifsShim.step -> LandSurface.forward)
!
!  Three PUBLIC entry points (the ABI callpar1s links against):
!    SUBROUTINE PYOIFS_INIT(python_home, n_tile, n_soil, dt_default, ierr)
!    SUBROUTINE PYOIFS_SURFTSTP(klon, ktiles, klevs, <38 args>, ierr)
!    SUBROUTINE PYOIFS_FINALIZE(ierr)
!  plus a diagnostic helper PYOIFS_LAST_ERROR.
!
!  Argument order convention
!  -------------------------
!  The 38 arrays of PYOIFS_SURFTSTP are in the SAME order as the C
!  pyoifs_step() (pyoifs_c.c) and the Python PyOifsShim.step()
!  (pyoifs_shim.py).  All three layers are kept in lockstep; a mismatch
!  is caught by the PYOIFS_ABI_VERSION cross-check at init time.  Do NOT
!  reorder arguments in one layer without bumping the ABI version in all
!  three.  The order is: 12 forcing ptrs, dt, 5 static ptrs, 9 state
!  ptrs, 11 OUT ptrs (see the inline headers below).
!
!  Column-major pass-through
!  -------------------------
!  Every array dummy is assumed-size REAL(C_DOUBLE) :: X(*) (or
!  INTEGER(C_INT) for the two type arrays).  Fortran passes the address
!  of element (1,1) by reference, which is exactly the column-major
!  leading-dim-KLON buffer the C side (double*) and the Python side
!  (np.ctypeslib.as_array, Fortran strides) expect.  No transpose, no
!  copy on the Fortran side -- the whole bridge is zero-copy in.
!
!  Compile
!  -------
!  See ../build_pyoifs_fortran.sh.  Standalone-compileable (no PARKIND1,
!  no YOMHOOK, no MPI) so it can be smoke-tested like the offline
!  libsurf_ref.so: it depends only on ISO_C_BINDING and libgfortran.
! ========================================================================== !
MODULE PYOIFS_F

  USE, INTRINSIC :: ISO_C_BINDING, ONLY : &
    &   C_INT,        &
    &   C_DOUBLE,     &
    &   C_CHAR,       &
    &   C_NULL_CHAR,  &
    &   C_PTR,        &
    &   C_F_POINTER,  &
    &   C_ASSOCIATED

  IMPLICIT NONE

  PRIVATE

  ! ---- Public entry points (the callpar1s branch calls only these) ----------
  PUBLIC :: PYOIFS_INIT
  PUBLIC :: PYOIFS_SURFTSTP
  PUBLIC :: PYOIFS_FINALIZE
  PUBLIC :: PYOIFS_LAST_ERROR
  PRIVATE :: PYOIFS_DISABLE_FP_TRAPS

  ! ---- Module-saved state --------------------------------------------------
  ! The shim is built once (at PYOIFS_INIT) with a fixed tile / soil-layer
  ! count; every PYOIFS_SURFTSTP call must agree.  We save them here so the
  ! thunk can assert at call time -- a mismatched KTILES would otherwise let
  ! the Python side stride off the end of a column-major buffer silently.
  LOGICAL,         SAVE :: L_INITIALISED = .FALSE.
  INTEGER(C_INT),  SAVE :: N_TILE_SAVED  = 0_C_INT
  INTEGER(C_INT),  SAVE :: N_SOIL_SAVED  = 0_C_INT

  ! Reusable scratch for the null-terminated python_home C-string.
  INTEGER, PARAMETER :: JPHOME = 4096

  ! ==========================================================================
  !  BIND(C) interfaces to the C glue (libpyoifs.so).
  !
  !  These mirror pyoifs_c.c line-for-line.  Scalars the C side takes by
  !  value (klon, n_tile, n_soil, dt_default, pdt) are VALUE here; arrays
  !  the C side takes as pointers (double*, int*) are passed by reference
  !  via assumed-size dummies.
  ! ==========================================================================
  INTERFACE

    ! int pyoifs_init(const char *python_home, int n_tile, int n_soil,
    !                 double dt_default);
    FUNCTION C_PYOIFS_INIT(PYTHON_HOME, N_TILE, N_SOIL, DT_DEFAULT) &
    &   BIND(C, NAME='pyoifs_init')
      IMPORT :: C_CHAR, C_INT, C_DOUBLE
      INTEGER(C_INT)               :: C_PYOIFS_INIT
      CHARACTER(KIND=C_CHAR), DIMENSION(*), INTENT(IN) :: PYTHON_HOME
      INTEGER(C_INT),  VALUE                      :: N_TILE
      INTEGER(C_INT),  VALUE                      :: N_SOIL
      REAL(C_DOUBLE),  VALUE                      :: DT_DEFAULT
    END FUNCTION C_PYOIFS_INIT

    ! int pyoifs_step(int klon, double* pu, ..., double pdt, ..., int*,
    !                 ..., double* out_slh_tile);
    ! See pyoifs_c.c:286-305 for the authoritative signature.
    FUNCTION C_PYOIFS_STEP(KLON, &
    &   PU, PV, PT, PQ, PCO2, PP_SURF, PGEOM1, &
    &   PFR_SO, PFR_TH, PMU0, PRAIN, PSNOW, PDT, &
    &   PTILE_FRAC, PLAND_FRAC, POROGRAPHY, PSOIL_TYPE, PVEG_TYPE, &
    &   PSKIN_T, PSOIL_T, PSOIL_W_LIQ, PSNOW_W, PSNOW_FRAC, PSKIN_WATER, &
    &   PSST, PSNOW_T, PSNOW_RHO, &
    &   OUT_SKIN_T, OUT_SOIL_T, OUT_SOIL_W_LIQ, &
    &   OUT_SNOW_W, OUT_SNOW_FRAC, OUT_SKIN_WATER, &
    &   OUT_H_FLUX_TILE, OUT_E_FLUX_TILE, OUT_STR_TILE, &
    &   OUT_G0_TILE, OUT_SLH_TILE) &
    &   BIND(C, NAME='pyoifs_step')
      IMPORT :: C_INT, C_DOUBLE
      INTEGER(C_INT), VALUE                :: KLON
      ! ---- Forcing (12 pointers) ----
      REAL(C_DOUBLE), DIMENSION(*), INTENT(IN) :: PU, PV, PT, PQ, PCO2
      REAL(C_DOUBLE), DIMENSION(*), INTENT(IN) :: PP_SURF, PGEOM1
      REAL(C_DOUBLE), DIMENSION(*), INTENT(IN) :: PFR_SO, PFR_TH, PMU0
      REAL(C_DOUBLE), DIMENSION(*), INTENT(IN) :: PRAIN, PSNOW
      REAL(C_DOUBLE), VALUE                :: PDT
      ! ---- Static (3 double*, 2 int*) ----
      REAL(C_DOUBLE),     DIMENSION(*), INTENT(IN) :: PTILE_FRAC
      REAL(C_DOUBLE),     DIMENSION(*), INTENT(IN) :: PLAND_FRAC, POROGRAPHY
      INTEGER(C_INT),     DIMENSION(*), INTENT(IN) :: PSOIL_TYPE, PVEG_TYPE
      ! ---- Prognostic state, in (9 double*) ----
      REAL(C_DOUBLE),     DIMENSION(*), INTENT(IN) :: PSKIN_T, PSOIL_T
      REAL(C_DOUBLE),     DIMENSION(*), INTENT(IN) :: PSOIL_W_LIQ
      REAL(C_DOUBLE),     DIMENSION(*), INTENT(IN) :: PSNOW_W, PSNOW_FRAC
      REAL(C_DOUBLE),     DIMENSION(*), INTENT(IN) :: PSKIN_WATER
      REAL(C_DOUBLE),     DIMENSION(*), INTENT(IN) :: PSST, PSNOW_T, PSNOW_RHO
      ! ---- OUT (11 double*, written in place) ----
      REAL(C_DOUBLE), DIMENSION(*), INTENT(OUT) :: OUT_SKIN_T, OUT_SOIL_T
      REAL(C_DOUBLE), DIMENSION(*), INTENT(OUT) :: OUT_SOIL_W_LIQ
      REAL(C_DOUBLE), DIMENSION(*), INTENT(OUT) :: OUT_SNOW_W, OUT_SNOW_FRAC
      REAL(C_DOUBLE), DIMENSION(*), INTENT(OUT) :: OUT_SKIN_WATER
      REAL(C_DOUBLE), DIMENSION(*), INTENT(OUT) :: OUT_H_FLUX_TILE
      REAL(C_DOUBLE), DIMENSION(*), INTENT(OUT) :: OUT_E_FLUX_TILE
      REAL(C_DOUBLE), DIMENSION(*), INTENT(OUT) :: OUT_STR_TILE
      REAL(C_DOUBLE), DIMENSION(*), INTENT(OUT) :: OUT_G0_TILE
      REAL(C_DOUBLE), DIMENSION(*), INTENT(OUT) :: OUT_SLH_TILE
      INTEGER(C_INT)                       :: C_PYOIFS_STEP
    END FUNCTION C_PYOIFS_STEP

    ! int pyoifs_finalize(void);
    FUNCTION C_PYOIFS_FINALIZE() BIND(C, NAME='pyoifs_finalize')
      IMPORT :: C_INT
      INTEGER(C_INT) :: C_PYOIFS_FINALIZE
    END FUNCTION C_PYOIFS_FINALIZE

    ! const char *pyoifs_last_error(void);
    ! Returns a C pointer to a static NUL-terminated string.
    FUNCTION C_PYOIFS_LAST_ERROR() BIND(C, NAME='pyoifs_last_error') &
    &   RESULT(C_PTR_ERR)
      IMPORT :: C_PTR
      TYPE(C_PTR) :: C_PTR_ERR
    END FUNCTION C_PYOIFS_LAST_ERROR

  END INTERFACE

  ! ==========================================================================
  !  glibc <fenv.h> FP-exception-mask control (used by PYOIFS_DISABLE_FP_TRAPS).
  !  fedisableexcept masks exceptions in the MXCSR; feclearexcept clears any
  !  pending flags so a stale (already-raised) bit cannot fire on re-enable.
  !  FE_* bit values are the glibc/x86-64 definitions (fenv.h): they are NOT
  !  portable to non-glibc -- this bridge targets the OpenIFS linux stack only.
  ! ==========================================================================
  INTERFACE
    ! int fedisableexcept(int excepts);
    FUNCTION C_FEDISABLEEXCEPT(EXCEPTS) BIND(C, NAME='fedisableexcept') &
    &   RESULT(IRET)
      IMPORT :: C_INT
      INTEGER(C_INT), VALUE :: EXCEPTS
      INTEGER(C_INT)       :: IRET
    END FUNCTION C_FEDISABLEEXCEPT

    ! int feclearexcept(int excepts);
    FUNCTION C_FECLEAREXCEPT(EXCEPTS) BIND(C, NAME='feclearexcept') &
    &   RESULT(IRET)
      IMPORT :: C_INT
      INTEGER(C_INT), VALUE :: EXCEPTS
      INTEGER(C_INT)       :: IRET
    END FUNCTION C_FECLEAREXCEPT
  END INTERFACE

  ! glibc FE_* bit masks (from bits/fenv.h on x86-64).
  INTEGER(C_INT), PARAMETER :: FE_ALL_EXCEPT = 61_C_INT  ! 0x3d = INEXACT|DIVZ|INVALID|OVERFLOW

CONTAINS

  ! --------------------------------------------------------------------------
  !  PYOIFS_DISABLE_FP_TRAPS -- mask all FP exceptions in MXCSR and clear
  !  pending flags.  Called once at PYOIFS_INIT before CPython starts.  This
  !  undoes libfiat/Dr.Hook's feenableexcept so pure-Python FP ops cannot
  !  raise SIGFPE.  Cheap (two syscalls-free libc calls), idempotent.
  ! --------------------------------------------------------------------------
  SUBROUTINE PYOIFS_DISABLE_FP_TRAPS()
    INTEGER(C_INT) :: J_DUMMY
    J_DUMMY = C_FECLEAREXCEPT(FE_ALL_EXCEPT)
    J_DUMMY = C_FEDISABLEEXCEPT(FE_ALL_EXCEPT)
  END SUBROUTINE PYOIFS_DISABLE_FP_TRAPS

  ! ==========================================================================
  !  PYOIFS_INIT -- bring up the Python embedding once, at model start.
  !
  !  Trims the Fortran PYTHON_HOME string, appends a C NUL, and forwards to
  !  the C pyoifs_init (which dlopens libpython, calls Py_Initialize, imports
  !  the shim module, cross-checks the ABI version, and builds the
  !  PyOifsShim singleton).  Saves N_TILE / N_SOIL for the per-call assert.
  !
  !  IERR is 0 on success, non-zero on failure (call PYOIFS_LAST_ERROR for
  !  the human-readable message).  Idempotent: a second call is a no-op.
  ! ==========================================================================
  SUBROUTINE PYOIFS_INIT(PYTHON_HOME, N_TILE, N_SOIL, DT_DEFAULT, IERR)
    CHARACTER(LEN=*), INTENT(IN)  :: PYTHON_HOME
    INTEGER,         INTENT(IN)  :: N_TILE
    INTEGER,         INTENT(IN)  :: N_SOIL
    REAL(C_DOUBLE),  INTENT(IN)  :: DT_DEFAULT
    INTEGER,         INTENT(OUT) :: IERR

    CHARACTER(LEN=JPHOME, KIND=C_CHAR) :: CLHOME_C
    INTEGER :: ILEN, I

    ! ---- Disable hardware FP-exception trapping before entering CPython -----
    ! OpenIFS links ECMWF's libfiat, whose Dr.Hook init calls feenableexcept()
    ! (FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW) to turn FP exceptions into SIGFPE
    ! for Fortran debugging.  That masking is stored in the process-wide MXCSR,
    ! so it is inherited by the embedded CPython interpreter -- which never
    ! expects hardware traps (Python maps /0 to a software ZeroDivisionError).
    ! Result: any FP-inf/nan/0 produced inside pure-Python bytecode (here seen
    ! in PyNumber_Multiply during module import) raises SIGFPE and kills the
    ! process.  DR_HOOK_TRAPFPE=0 is NOT honoured in this fiat build, so we
    ! mask the traps explicitly here, just before the first C/Python call.
    CALL PYOIFS_DISABLE_FP_TRAPS()

    ! Pack the Fortran string into a NUL-terminated C char buffer.  We
    ! blank-fill then drop a single C_NULL_CHAR after the trimmed length;
    ! the C side reads up to the NUL so trailing blanks are irrelevant.
    ILEN = MIN(LEN_TRIM(PYTHON_HOME), JPHOME - 1)
    CLHOME_C = REPEAT(' ', JPHOME)
    DO I = 1, ILEN
      CLHOME_C(I:I) = PYTHON_HOME(I:I)
    END DO
    CLHOME_C(ILEN + 1:ILEN + 1) = C_NULL_CHAR

    IERR = C_PYOIFS_INIT(CLHOME_C, &
    &                    INT(N_TILE, KIND=C_INT), &
    &                    INT(N_SOIL, KIND=C_INT), &
    &                    DT_DEFAULT)

    IF (IERR == 0) THEN
      L_INITIALISED = .TRUE.
      N_TILE_SAVED  = INT(N_TILE, KIND=C_INT)
      N_SOIL_SAVED  = INT(N_SOIL, KIND=C_INT)
    END IF
  END SUBROUTINE PYOIFS_INIT


  ! ==========================================================================
  !  PYOIFS_SURFTSTP -- one surface step, redirected to the PyTorch port.
  !
  !  This is the per-coupling-interval entry point callpar1s calls from its
  !  LEPYOIFS branch.  It asserts the (klon, ktiles, klevs) geometry against
  !  what PYOIFS_INIT configured, then forwards straight to the C pyoifs_step.
  !
  !  KLON is the leading dimension of every array (the number of gridpoints
  !  in the block).  The arrays are column-major Fortran buffers; the C side
  !  receives their base addresses, and the Python side reinterprets them
  !  with explicit Fortran strides.  No data crosses the boundary as a copy.
  !
  !  The 38 array dummies below appear in the LOCKSTEP order (forcing, dt,
  !  static, state, OUT) -- do not reorder without bumping PYOIFS_ABI_VERSION
  !  in all three layers (pyoifs_f.F90, pyoifs_c.c, pyoifs_shim.py).
  !
  !  KTILES / KLEVS are INTENT(IN) only for the geometry assert; they are NOT
  !  forwarded (the C ABI derives tile/soil counts from the singleton built
  !  at init).  IERR is 0 on success.
  ! ==========================================================================
  SUBROUTINE PYOIFS_SURFTSTP(KLON, KTILES, KLEVS, &
  &   PU, PV, PT, PQ, PCO2, PP_SURF, PGEOM1, &
  &   PFR_SO, PFR_TH, PMU0, PRAIN, PSNOW, PDT, &
  &   PTILE_FRAC, PLAND_FRAC, POROGRAPHY, PSOIL_TYPE, PVEG_TYPE, &
  &   PSKIN_T, PSOIL_T, PSOIL_W_LIQ, PSNOW_W, PSNOW_FRAC, PSKIN_WATER, &
  &   PSST, PSNOW_T, PSNOW_RHO, &
  &   OUT_SKIN_T, OUT_SOIL_T, OUT_SOIL_W_LIQ, &
  &   OUT_SNOW_W, OUT_SNOW_FRAC, OUT_SKIN_WATER, &
  &   OUT_H_FLUX_TILE, OUT_E_FLUX_TILE, OUT_STR_TILE, &
  &   OUT_G0_TILE, OUT_SLH_TILE, IERR)

    ! ---- Geometry ----
    INTEGER, INTENT(IN) :: KLON, KTILES, KLEVS

    ! ---- Forcing (lowest-model-level state + radiation + precip) ----
    REAL(C_DOUBLE),    DIMENSION(KLON),            INTENT(IN)  :: PU, PV, PT
    REAL(C_DOUBLE),    DIMENSION(KLON),            INTENT(IN)  :: PQ, PCO2
    REAL(C_DOUBLE),    DIMENSION(KLON),            INTENT(IN)  :: PP_SURF, PGEOM1
    REAL(C_DOUBLE),    DIMENSION(KLON),            INTENT(IN)  :: PFR_SO, PFR_TH
    REAL(C_DOUBLE),    DIMENSION(KLON),            INTENT(IN)  :: PMU0, PRAIN, PSNOW
    REAL(C_DOUBLE),                                  INTENT(IN)  :: PDT

    ! ---- Static (tile frac, masks, soil/veg type) ----
    REAL(C_DOUBLE),    DIMENSION(KLON, KTILES),    INTENT(IN)  :: PTILE_FRAC
    REAL(C_DOUBLE),    DIMENSION(KLON),            INTENT(IN)  :: PLAND_FRAC, POROGRAPHY
    INTEGER(C_INT),    DIMENSION(KLON),            INTENT(IN)  :: PSOIL_TYPE, PVEG_TYPE

    ! ---- Prognostic state (in) ----
    REAL(C_DOUBLE),    DIMENSION(KLON, KTILES),    INTENT(IN)  :: PSKIN_T
    REAL(C_DOUBLE),    DIMENSION(KLON, KLEVS),     INTENT(IN)  :: PSOIL_T, PSOIL_W_LIQ
    REAL(C_DOUBLE),    DIMENSION(KLON),            INTENT(IN)  :: PSNOW_W, PSNOW_FRAC
    REAL(C_DOUBLE),    DIMENSION(KLON),            INTENT(IN)  :: PSKIN_WATER
    REAL(C_DOUBLE),    DIMENSION(KLON),            INTENT(IN)  :: PSST, PSNOW_T, PSNOW_RHO

    ! ---- OUT (updated prognostics + per-tile fluxes, written in place) ----
    REAL(C_DOUBLE),    DIMENSION(KLON, KTILES),    INTENT(OUT) :: OUT_SKIN_T
    REAL(C_DOUBLE),    DIMENSION(KLON, KLEVS),     INTENT(OUT) :: OUT_SOIL_T, OUT_SOIL_W_LIQ
    REAL(C_DOUBLE),    DIMENSION(KLON),            INTENT(OUT) :: OUT_SNOW_W, OUT_SNOW_FRAC
    REAL(C_DOUBLE),    DIMENSION(KLON),            INTENT(OUT) :: OUT_SKIN_WATER
    REAL(C_DOUBLE),    DIMENSION(KLON, KTILES),    INTENT(OUT) :: OUT_H_FLUX_TILE
    REAL(C_DOUBLE),    DIMENSION(KLON, KTILES),    INTENT(OUT) :: OUT_E_FLUX_TILE
    REAL(C_DOUBLE),    DIMENSION(KLON, KTILES),    INTENT(OUT) :: OUT_STR_TILE
    REAL(C_DOUBLE),    DIMENSION(KLON, KTILES),    INTENT(OUT) :: OUT_G0_TILE
    REAL(C_DOUBLE),    DIMENSION(KLON, KTILES),    INTENT(OUT) :: OUT_SLH_TILE

    INTEGER, INTENT(OUT) :: IERR

    ! ---- Geometry assert: the singleton was built for (N_TILE_SAVED,
    !      N_SOIL_SAVED).  A mismatch here means callpar1s's KTILES/KLEVS do
    !      not agree with the PYOIFS_INIT config -- the Python side would
    !      stride off the buffer, so we refuse rather than corrupt memory.
    IF (.NOT. L_INITIALISED) THEN
      WRITE(*,'(A)') '[pyoifs] PYOIFS_SURFTSTP called before PYOIFS_INIT'
      IERR = 1
      RETURN
    END IF
    IF (KTILES /= N_TILE_SAVED) THEN
      WRITE(*,'(A,I0,A,I0)') '[pyoifs] KTILES mismatch: call=', KTILES, &
      &                      ' init=', N_TILE_SAVED
      IERR = 2
      RETURN
    END IF
    IF (KLEVS /= N_SOIL_SAVED) THEN
      WRITE(*,'(A,I0,A,I0)') '[pyoifs] KLEVS mismatch: call=', KLEVS, &
      &                      ' init=', N_SOIL_SAVED
      IERR = 3
      RETURN
    END IF

    ! Forward to the C glue.  KLON and PDT are VALUE; the arrays go by
    ! reference (Fortran default) and land as double*/int* on the C side.
    IERR = C_PYOIFS_STEP(INT(KLON, KIND=C_INT), &
    &   PU, PV, PT, PQ, PCO2, PP_SURF, PGEOM1, &
    &   PFR_SO, PFR_TH, PMU0, PRAIN, PSNOW, PDT, &
    &   PTILE_FRAC, PLAND_FRAC, POROGRAPHY, PSOIL_TYPE, PVEG_TYPE, &
    &   PSKIN_T, PSOIL_T, PSOIL_W_LIQ, PSNOW_W, PSNOW_FRAC, PSKIN_WATER, &
    &   PSST, PSNOW_T, PSNOW_RHO, &
    &   OUT_SKIN_T, OUT_SOIL_T, OUT_SOIL_W_LIQ, &
    &   OUT_SNOW_W, OUT_SNOW_FRAC, OUT_SKIN_WATER, &
    &   OUT_H_FLUX_TILE, OUT_E_FLUX_TILE, OUT_STR_TILE, &
    &   OUT_G0_TILE, OUT_SLH_TILE)
  END SUBROUTINE PYOIFS_SURFTSTP


  ! ==========================================================================
  !  PYOIFS_FINALIZE -- tear down the Python embedding at model end.
  !
  !  Only meaningful if WE initialised the interpreter (the C side tracks
  !  this with s_we_initialized; when the host process is itself Python,
  !  e.g. under the pytest harness, Py_Finalize would rip out the host's
  !  running interpreter and is correctly skipped by the C glue).
  ! ==========================================================================
  SUBROUTINE PYOIFS_FINALIZE(IERR)
    INTEGER, INTENT(OUT) :: IERR
    IERR = C_PYOIFS_FINALIZE()
    L_INITIALISED = .FALSE.
  END SUBROUTINE PYOIFS_FINALIZE


  ! ==========================================================================
  !  PYOIFS_LAST_ERROR -- fetch the most recent C-side error string.
  !
  !  The C glue keeps the last exception text in a static buffer; this
  !  copies it out into a Fortran string for the driver to log / abort on.
  !  Returns a blank string if no error is recorded.
  ! ==========================================================================
  SUBROUTINE PYOIFS_LAST_ERROR(CLMSG)
    CHARACTER(LEN=*), INTENT(OUT) :: CLMSG
    TYPE(C_PTR)          :: CPTR_ERR
    CHARACTER(LEN=1, KIND=C_CHAR), POINTER :: FPC(:)
    INTEGER :: I, NMAX

    CLMSG = REPEAT(' ', LEN(CLMSG))
    CPTR_ERR = C_PYOIFS_LAST_ERROR()
    IF (.NOT. C_ASSOCIATED(CPTR_ERR)) RETURN

    NMAX = LEN(CLMSG)
    CALL C_F_POINTER(CPTR_ERR, FPC, [NMAX])
    DO I = 1, NMAX
      IF (FPC(I) == C_NULL_CHAR) EXIT
      CLMSG(I:I) = FPC(I)
    END DO
  END SUBROUTINE PYOIFS_LAST_ERROR

END MODULE PYOIFS_F
