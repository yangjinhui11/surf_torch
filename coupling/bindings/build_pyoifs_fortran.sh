#!/usr/bin/env bash
# build_pyoifs_fortran.sh -- compile the Fortran thunk module (PYOIFS_F).
#
# Produces pyoifs_f.o and pyoifs_f.mod next to this script.  The object is
# linked into the coupled offline executable by build_offline_coupled.sh
# (P1.4) together with libpyoifs.so and libsurf_ref.so.  Standalone here
# means: depends only on ISO_C_BINDING and libgfortran -- no PARKIND1, no
# YOMHOOK, no MPI -- so it compiles without the OpenIFS source tree.
#
# Usage
# -----
#   ./build_pyoifs_fortran.sh            # release
#   ./build_pyoifs_fortran.sh debug      # -O0 -g -fcheck=all
#   ./build_pyoifs_fortran.sh clean
#   FC=gfortran-13 ./build_pyoifs_fortran.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/fortran/pyoifs_f.F90"
OBJ="${SCRIPT_DIR}/fortran/pyoifs_f.o"

: "${FC:=gfortran}"
: "${FFLAGS_RELEASE:=-O2 -fPIC -std=f2008 -fno-range-check}"
: "${FFLAGS_DEBUG:=-O0 -g -fPIC -std=f2008 -fcheck=all -fbacktrace -Wall -Wextra -fno-range-check}"

mode="${1:-release}"
case "${mode}" in
  release) FFLAGS="${FFLAGS_RELEASE}" ;;
  debug)   FFLAGS="${FFLAGS_DEBUG}"   ;;
  clean)
    echo "[build_pyoifs_fortran] removing artefacts"
    rm -f "${SCRIPT_DIR}/fortran"/pyoifs_f.o "${SCRIPT_DIR}/fortran"/pyoifs_f.mod
    exit 0 ;;
  *) echo "usage: $0 [release|debug|clean]" >&2; exit 2 ;;
esac

[[ -f "${SRC}" ]] || { echo "[build_pyoifs_fortran] error: no ${SRC}" >&2; exit 1; }

echo "[build_pyoifs_fortran] ${FC} -c ${FFLAGS} -> ${OBJ}"
"${FC}" -c ${FFLAGS} -J "${SCRIPT_DIR}/fortran" "${SRC}" -o "${OBJ}"

[[ -f "${OBJ}" ]] || { echo "[build_pyoifs_fortran] error: no object" >&2; exit 1; }

# Confirm the three PUBLIC entry points are present (lowercased symbol, as
# gfortran emits on Linux).  These are the symbols callpar1s will link.
if command -v nm >/dev/null 2>&1; then
  missing=0
  for sym in __pyoifs_f_MOD_pyoifs_init __pyoifs_f_MOD_pyoifs_surftstp __pyoifs_f_MOD_pyoifs_finalize __pyoifs_f_MOD_pyoifs_last_error; do
    if nm "${OBJ}" 2>/dev/null | grep -qi "${sym}"; then
      echo "[build_pyoifs_fortran] OK -- symbol '${sym}' present"
    else
      echo "[build_pyoifs_fortran] error: symbol '${sym}' missing" >&2
      missing=1
    fi
  done
  [[ ${missing} -eq 0 ]] || exit 1
fi
echo "[build_pyoifs_fortran] done."
