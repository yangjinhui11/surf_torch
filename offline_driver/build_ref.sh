#!/usr/bin/env bash
# build_ref.sh -- compile the standalone OpenIFS surface reference kernel.
#
# Produces ``libsurf_ref.so`` next to this script.  The library is a
# *standalone* Fortran reimplementation of the three surface kernels the
# PyTorch port currently implements:
#
#   * VLAMSK        -- per-tile skin conductivity
#   * SURFSEB_CTL   -- Penman-Monteith surface energy balance
#   * VEXCS         -- Beljaars Monin-Obukhov exchange coefficients
#   * DQSDT         -- skin-T saturation Jacobian (vsurf 278-285)
#   * STOMATA       -- Jarvis stomatal resistance (vsurf 294-422)
#   * VEVAP         -- equivalent evapotranspiration efficiency (vevap_mod)
#   * VDFDIFH       -- offline single-column VDFDIFH1S SURFSEB wrapper
#   * FARQUHAR      -- Farquhar photosynthesis + Ball-Berry stomatal
#                      conductance (farquhar_mod.F90, Yin et al. 2009)
#
# It depends on nothing but libgfortran: no PARKIND1, no YOMHOOK, no
# MPI, no NetCDF.  Every physical constant is passed in at call time
# from the Python ctypes wrapper (offline_driver/surf_ref.py), which
# builds them from the same surf_pytorch.constants bundle used by the
# PyTorch port -- so the two implementations share an identical
# parameterisation and any mismatch is purely arithmetic.
#
# Usage
# -----
#   ./build_ref.sh                 # release build, ./libsurf_ref.so
#   ./build_ref.sh debug           # -O0 -g -fcheck=all -fbacktrace
#   ./build_ref.sh clean           # remove build artefacts
#   FC=gfortran-13 ./build_ref.sh  # pick a different compiler
#
# The script is intentionally a thin wrapper so it stays auditable.

set -euo pipefail

# Resolve the directory of *this* script so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORTRAN_DIR="${SCRIPT_DIR}/fortran"
SRC="${FORTRAN_DIR}/surf_ref.F90"
OUT="${SCRIPT_DIR}/libsurf_ref.so"

# Default compiler + flags; allow caller override.
: "${FC:=gfortran}"
: "${FFLAGS_RELEASE:=-O2 -fPIC -shared -std=f2008}"
: "${FFLAGS_DEBUG:=-O0 -g -fPIC -shared -std=f2008 -fcheck=all -fbacktrace -Wall -Wextra}"

mode="${1:-release}"
case "${mode}" in
  release)
    FFLAGS="${FFLAGS_RELEASE}"
    ;;
  debug)
    FFLAGS="${FFLAGS_DEBUG}"
    ;;
  clean)
    echo "[build_ref] removing build artefacts"
    rm -f "${OUT}" "${FORTRAN_DIR}"/surf_ref.mod "${FORTRAN_DIR}"/*.o
    exit 0
    ;;
  *)
    echo "usage: $0 [release|debug|clean]" >&2
    exit 2
    ;;
esac

if [[ ! -f "${SRC}" ]]; then
  echo "[build_ref] error: source not found at ${SRC}" >&2
  exit 1
fi

# gfortran emits surf_ref.mod next to the .o; -J keeps it in the
# fortran/ dir instead of polluting the caller's CWD.  -shared + -o
# produces the loadable library directly in one step.
echo "[build_ref] ${FC} ${FFLAGS} -> ${OUT}"
"${FC}" ${FFLAGS} -J "${FORTRAN_DIR}" "${SRC}" -o "${OUT}"

if [[ ! -f "${OUT}" ]]; then
  echo "[build_ref] error: build did not produce ${OUT}" >&2
  exit 1
fi

# Sanity-check the symbols the Python wrapper will look up.
if command -v nm >/dev/null 2>&1; then
  missing=0
  for sym in surf_ref_vlamsk surf_ref_seb surf_ref_vexcs surf_ref_dqsdt surf_ref_stomata surf_ref_vevap surf_ref_vdfdifh surf_ref_soilt surf_ref_snow surf_ref_soilw surf_ref_farquhar; do
    if nm "${OUT}" 2>/dev/null | grep -q "${sym}"; then
      echo "[build_ref] OK -- symbol '${sym}' present"
    else
      echo "[build_ref] error: symbol '${sym}' missing from ${OUT}" >&2
      missing=1
    fi
  done
  if [[ ${missing} -ne 0 ]]; then
    exit 1
  fi
fi

echo "[build_ref] done.  Use it via:"
echo "    from offline_driver.surf_ref import SurfRef"
