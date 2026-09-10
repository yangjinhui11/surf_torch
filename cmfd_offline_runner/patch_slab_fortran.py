#!/usr/bin/env python3
"""Fix the driver-level slab closure: export the tile-fraction-weighted
grid-mean surface sensible heat flux from CALLPAR1S (PAHFSTI is local to it;
the previous patch used ZFTLHSB, the snow-covered-fraction latent heat flux,
which is identically zero in this offline configuration, so the anomaly never
left zero) and use it in the cpg1s recurrence."""

from pathlib import Path

CALLPAR1S = Path("/home/qixiang/yangjinhui/openifs/openifs-48r1/ifs-source"
                 "/surf/offline/driver/callpar1s.F90")
CPG1S = Path("/home/qixiang/yangjinhui/openifs/openifs-48r1/ifs-source"
             "/surf/offline/driver/cpg1s.F90")


def replace_once(text: str, old: str, new: str, where: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{where}: expected 1 occurrence, found {count}: {old!r}")
    return text.replace(old, new, 1)


callpar = CALLPAR1S.read_text()

# 1. New output dummy at the end of the dummy-argument list.
callpar = replace_once(
    callpar,
    "     & , PDHTLS,PDHTSS,PDHTTS,PDHTIS,PDHIIS,PDHSSS,PDHWLS,PDHRESS,PDHCO2S,PDHBIOS,PDHVEGS)",
    "     & , PDHTLS,PDHTSS,PDHTTS,PDHTIS,PDHIIS,PDHSSS,PDHWLS,PDHRESS,PDHCO2S,PDHBIOS,PDHVEGS &\n"
    "     & , PAHFSG )",
    "callpar1s dummy list",
)

# 2. Declaration next to the other (KLON) flux outputs.
callpar = replace_once(
    callpar,
    "REAL(KIND=JPRB) :: PFTG12(KLON),PFTLHEV(KLON),PFTLHSB(KLON),",
    "REAL(KIND=JPRB) :: PAHFSG(KLON)\nREAL(KIND=JPRB) :: PFTG12(KLON),PFTLHEV(KLON),PFTLHSB(KLON),",
    "callpar1s declaration",
)

# 3. Tile-weighted grid mean right after the VDF call returns PAHFSTI.
callpar = replace_once(
    callpar,
    " & PDHTLS,PDHTSS,PDHTTS,PDHTIS,PDHCO2S,PDHVEGS)\n",
    " & PDHTLS,PDHTSS,PDHTTS,PDHTIS,PDHCO2S,PDHVEGS)\n"
    "\n"
    "! Grid-mean surface sensible heat flux (tile-fraction weighted) for the\n"
    "! driver-level slab closure.\n"
    "DO JL=KIDIA,KFDIA\n"
    "  PAHFSG(JL)=0.0_JPRB\n"
    "  DO JTILE=1,KTILES\n"
    "    PAHFSG(JL)=PAHFSG(JL)+ZFRTI(JL,JTILE)*PAHFSTI(JL,JTILE)\n"
    "  ENDDO\n"
    "ENDDO\n",
    "callpar1s post-vdf insertion",
)
CALLPAR1S.write_text(callpar)

cpg = CPG1S.read_text()

# 4. Local array for the new output.
cpg = replace_once(
    cpg,
    "REAL(KIND=JPRB) , SAVE, ALLOCATABLE :: ZSLABDT(:)",
    "REAL(KIND=JPRB) , SAVE, ALLOCATABLE :: ZSLABDT(:)\n"
    "REAL(KIND=JPRB) :: ZAHFSG(NPOI)",
    "cpg1s declaration",
)

# 5. Pass it as the trailing actual argument of CALLPAR1S.
cpg = replace_once(
    cpg,
    "     & ZDHCO2S(IST:IEND,:,:),ZDHBIOS(IST:IEND,:,:),ZDHVEGS(IST:IEND,:,:))",
    "     & ZDHCO2S(IST:IEND,:,:),ZDHBIOS(IST:IEND,:,:),ZDHVEGS(IST:IEND,:,:) &\n"
    "     & ,ZAHFSG(IST:IEND))",
    "cpg1s CALLPAR1S call tail",
)

# 6. Advance the anomaly with the sensible heat flux, not ZFTLHSB.
cpg = replace_once(
    cpg,
    "         & - (1.0_JPRB-ZSBETA)*ZSLAM*ZFTLHSB(IST:IEND)",
    "         & - (1.0_JPRB-ZSBETA)*ZSLAM*ZAHFSG(IST:IEND)",
    "cpg1s slab recurrence",
)
CPG1S.write_text(cpg)
print("patched OK")
