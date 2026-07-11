#!/usr/bin/env bash
# build_pyoifs.sh -- compile the C glue (pyoifs_c.c) into libpyoifs.so.
#
# Produces ``libpyoifs.so`` next to this script.  The library embeds a
# CPython interpreter and forwards OpenIFS surface calls into the
# PyTorch port via coupling/bindings/pyoifs_shim.py.
#
# Dependencies
# ------------
# * libpython3.10 -- resolved at link time via the interpreter's own
#   config (we ask sysconfig for LIBDIR + LDLIBRARY).  The resulting .so
#   carries the libpython dependency in its NEEDED list, so it only
#   loads inside an environment that has the same libpython.
# * Python.h      -- from sysconfig's INCLUDEPY.
#
# Usage
# -----
#   ./build_pyoifs.sh                 # release build
#   ./build_pyoifs.sh debug           # -O0 -g
#   ./build_pyoifs.sh clean           # remove artefacts
#
# The script is a thin wrapper around gcc, mirroring build_ref.sh so the
# two .so files are auditable side-by-side.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/pyoifs_c.c"
OUT="${SCRIPT_DIR}/libpyoifs.so"

# Resolve the Python include + lib dirs from the running interpreter so
# we always link against the same libpython that loads this .so.
PY_INCLUDE="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PY_LIBDIR="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"

: "${CC:=gcc}"
: "${CFLAGS_RELEASE:=-O2 -fPIC -fvisibility=default -Wall -Wextra}"
: "${CFLAGS_DEBUG:=-O0 -g -fPIC -fvisibility=default -Wall -Wextra}"

mode="${1:-release}"
case "${mode}" in
  release) CFLAGS="${CFLAGS_RELEASE}";;
  debug)   CFLAGS="${CFLAGS_DEBUG}";;
  clean)
    echo "[build_pyoifs] removing ${OUT}"
    rm -f "${OUT}"
    exit 0
    ;;
  *)
    echo "usage: $0 [release|debug|clean]" >&2
    exit 2
    ;;
esac

if [[ ! -f "${SRC}" ]]; then
  echo "[build_pyoifs] error: source not found at ${SRC}" >&2
  exit 1
fi

# -shared produces the loadable library; -lpython3.10 resolves the CPython
# symbols (Py_Initialize, PyGILState_*, etc.) at load time.  -Wl,-rpath
# bakes the libpython path into the .so so it loads without LD_LIBRARY_PATH.
echo "[build_pyoifs] ${CC} ${CFLAGS} -I${PY_INCLUDE} -> ${OUT}"
"${CC}" ${CFLAGS} \
  -I"${PY_INCLUDE}" \
  -shared -o "${OUT}" \
  "${SRC}" \
  -L"${PY_LIBDIR}" -lpython3.10 \
  -Wl,-rpath,"${PY_LIBDIR}"

if [[ ! -f "${OUT}" ]]; then
  echo "[build_pyoifs] error: build did not produce ${OUT}" >&2
  exit 1
fi

# Sanity-check the exported symbols the Fortran thunk will BIND(C) against.
if command -v nm >/dev/null 2>&1; then
  missing=0
  for sym in pyoifs_init pyoifs_step pyoifs_finalize pyoifs_last_error; do
    if nm "${OUT}" 2>/dev/null | grep -q " T ${sym}"; then
      echo "[build_pyoifs] OK -- symbol '${sym}' present"
    else
      echo "[build_pyoifs] error: symbol '${sym}' missing from ${OUT}" >&2
      missing=1
    fi
  done
  if [[ ${missing} -ne 0 ]]; then
    exit 1
  fi
fi

echo "[build_pyoifs] done."
