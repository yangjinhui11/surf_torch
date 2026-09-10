#!/usr/bin/env python3
"""Add the day/night-asymmetric relaxation times to the driver-level slab
closure in cpg1s.F90 (on top of the PAHFSG heat-flux fix).  Two new
environment variables, SURF_SLAB_TAU_DAY_H and SURF_SLAB_TAU_NIGHT_H
(hours), switch the scalar beta of the recurrence to the shortwave-blended
per-point form

    w     = max(FSSRD,0) / (max(FSSRD,0) + 50 W m-2)
    beta  = exp(-dt (w/tau_day + (1-w)/tau_night))

matching the torch implementation (modis_slab_asym_train.py).  With either
variable absent the original tied scalar beta is kept, and with
SURF_SLAB_LAMBDA absent/0 the patch remains a bitwise no-op."""

from pathlib import Path

CPG1S = Path("/home/qixiang/yangjinhui/openifs/openifs-48r1/ifs-source"
             "/surf/offline/driver/cpg1s.F90")


def replace_once(text: str, old: str, new: str, where: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{where}: expected 1 occurrence, found {count}: {old!r}")
    return text.replace(old, new, 1)


cpg = CPG1S.read_text()

# 1. Declarations: per-point blend arrays and the two relaxation times.
cpg = replace_once(
    cpg,
    "LOGICAL         , SAVE :: LLSLAB_INIT = .FALSE.\n"
    "LOGICAL                :: LLSLAB\n"
    "REAL(KIND=JPRB)        :: ZSLAM, ZSBETA",
    "LOGICAL         , SAVE :: LLSLAB_INIT = .FALSE.\n"
    "LOGICAL                :: LLSLAB\n"
    "REAL(KIND=JPRB)        :: ZSLAM, ZSBETA\n"
    "REAL(KIND=JPRB)        :: ZSWP(NPOI), ZSBW(NPOI), ZSBB(NPOI)\n"
    "LOGICAL         , SAVE :: LLSLAB_ASYM = .FALSE.\n"
    "REAL(KIND=JPRB) , SAVE :: ZTAUD = 0.0_JPRB , ZTAUN = 0.0_JPRB",
    "cpg1s declarations",
)

# 2. Initialization: read the two timescales from the environment.
cpg = replace_once(
    cpg,
    "  LLSLAB = ZSLAM > 0.0_JPRB\n"
    "  ZSBETA = 0.0_JPRB\n"
    "  IF (LLSLAB) ZSBETA = EXP(-TSTEP/(ZSLAM*1206.84_JPRB*600.0_JPRB))\n",
    "  LLSLAB = ZSLAM > 0.0_JPRB\n"
    "  ZSBETA = 0.0_JPRB\n"
    "  IF (LLSLAB) ZSBETA = EXP(-TSTEP/(ZSLAM*1206.84_JPRB*600.0_JPRB))\n"
    "  CALL GET_ENVIRONMENT_VARIABLE('SURF_SLAB_TAU_DAY_H', CSLABENV, STATUS=ISLABST)\n"
    "  IF (ISLABST == 0 .AND. LEN_TRIM(CSLABENV) > 0) READ(CSLABENV,*) ZTAUD\n"
    "  CALL GET_ENVIRONMENT_VARIABLE('SURF_SLAB_TAU_NIGHT_H', CSLABENV, STATUS=ISLABST)\n"
    "  IF (ISLABST == 0 .AND. LEN_TRIM(CSLABENV) > 0) READ(CSLABENV,*) ZTAUN\n"
    "  LLSLAB_ASYM = ZTAUD > 0.0_JPRB .AND. ZTAUN > 0.0_JPRB\n"
    "  IF (LLSLAB_ASYM) THEN\n"
    "    ZTAUD = ZTAUD*3600.0_JPRB\n"
    "    ZTAUN = ZTAUN*3600.0_JPRB\n"
    "  ENDIF\n",
    "cpg1s init block",
)

# 3. Log line: include the timescales.
cpg = replace_once(
    cpg,
    "  WRITE(NULOUT,*) 'CPG1S slab closure: lambda=',ZSLAM,' beta=',ZSBETA\n",
    "  WRITE(NULOUT,*) 'CPG1S slab closure: lambda=',ZSLAM,' beta=',ZSBETA, &\n"
    "   & ' tau_day_s=',ZTAUD,' tau_night_s=',ZTAUN,' asym=',LLSLAB_ASYM\n",
    "cpg1s log line",
)

# 4. Recurrence: shortwave-blended per-point beta when asymmetric.
cpg = replace_once(
    cpg,
    "    ZSLABDT(IST:IEND) = ZSBETA*ZSLABDT(IST:IEND) &\n"
    "         & - (1.0_JPRB-ZSBETA)*ZSLAM*ZAHFSG(IST:IEND)\n",
    "    IF (LLSLAB_ASYM) THEN\n"
    "      ZSWP(IST:IEND) = MAX(FSSRD(IST:IEND),0.0_JPRB)\n"
    "      ZSBW(IST:IEND) = ZSWP(IST:IEND)/(ZSWP(IST:IEND)+50.0_JPRB)\n"
    "      ZSBB(IST:IEND) = EXP(-TSTEP*(ZSBW(IST:IEND)/ZTAUD &\n"
    "           & + (1.0_JPRB-ZSBW(IST:IEND))/ZTAUN))\n"
    "      ZSLABDT(IST:IEND) = ZSBB(IST:IEND)*ZSLABDT(IST:IEND) &\n"
    "           & - (1.0_JPRB-ZSBB(IST:IEND))*ZSLAM*ZAHFSG(IST:IEND)\n"
    "    ELSE\n"
    "      ZSLABDT(IST:IEND) = ZSBETA*ZSLABDT(IST:IEND) &\n"
    "           & - (1.0_JPRB-ZSBETA)*ZSLAM*ZAHFSG(IST:IEND)\n"
    "    ENDIF\n",
    "cpg1s slab recurrence",
)

CPG1S.write_text(cpg)
print("patched OK")
