#!/usr/bin/env python3
"""Remaining cpg1s.F90 edits after patch_slab_fortran.py applied the
callpar1s.F90 side (callpar1s is already patched on disk)."""

from pathlib import Path

CPG1S = Path("/home/qixiang/yangjinhui/openifs/openifs-48r1/ifs-source"
             "/surf/offline/driver/cpg1s.F90")


def replace_once(text: str, old: str, new: str, where: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{where}: expected 1 occurrence, found {count}: {old!r}")
    return text.replace(old, new, 1)


cpg = CPG1S.read_text()

cpg = replace_once(
    cpg,
    "REAL(KIND=JPRB) , SAVE, ALLOCATABLE :: ZSLABDT(:)",
    "REAL(KIND=JPRB) , SAVE, ALLOCATABLE :: ZSLABDT(:)\n"
    "REAL(KIND=JPRB) :: ZAHFSG(NPOI)",
    "cpg1s declaration",
)

cpg = replace_once(
    cpg,
    "     & , ZDHCO2S(IST:IEND,:,:),ZDHBIOS(IST:IEND,:,:),ZDHVEGS(IST:IEND,:,:))",
    "     & , ZDHCO2S(IST:IEND,:,:),ZDHBIOS(IST:IEND,:,:),ZDHVEGS(IST:IEND,:,:) &\n"
    "     & ,ZAHFSG(IST:IEND))",
    "cpg1s CALLPAR1S call tail",
)

cpg = replace_once(
    cpg,
    "         & - (1.0_JPRB-ZSBETA)*ZSLAM*ZFTLHSB(IST:IEND)",
    "         & - (1.0_JPRB-ZSBETA)*ZSLAM*ZAHFSG(IST:IEND)",
    "cpg1s slab recurrence",
)
CPG1S.write_text(cpg)
print("cpg1s patched OK")
