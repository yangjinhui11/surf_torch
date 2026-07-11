#!/usr/bin/env bash
# run_offline_coupled.sh -- launch the PyTorch-coupled offline surface exe.
#
# This wrapper sets up the three runtime concerns that the linked
# ``osmMASTER_coupled`` binary cannot bake into its own DT_RPATH without
# breaking the OpenIFS shared-library transitive resolution:
#
#   1. libpython3.10 must come from the ANACONDA env (3.10.9) that owns the
#      surf_pytorch + torch installation, NOT the system 3.10.12 -- otherwise
#      Py_Initialize would load the wrong stdlib.  We force this with
#      LD_PRELOAD of anaconda's libpython3.10.so.1.0.  LD_PRELOAD is surgical:
#      it overrides only libpython, leaving libstdc++/libcurl to resolve from
#      the system path (the OpenIFS libeckit/libmultio/libnetcdf need the
#      system versions with the @GLIBCXX_3.4.30 / @CURL_OPENSSL_4 symbols).
#
#      (Putting /opt/anaconda3/lib in LD_LIBRARY_PATH or DT_RPATH instead
#      would leak anaconda's libcurl/libstdc++ into transitive resolution and
#      break libeckit at startup with GLIBCXX_3.4.30-not-found errors.)
#
#   2. OIFS_PYTHON_HOME tells the Fortran bridge where the conda env lives so
#      PYOIFS_INIT -> Py_SetPythonHome points CPython at the right stdlib.
#      Default: /opt/anaconda3 (the env that owns torch + surf_pytorch).
#
#   3. PYTHONPATH must include the surf_pytorch package root so the embedded
#      interpreter can ``import coupling.bindings.pyoifs_shim`` and the port.
#      Default: the repo root (parent of surf_pytorch/).
#
# Usage
# -----
#   ./run_offline_coupled.sh                      # run in current dir
#   ./run_offline_coupled.sh -- input             # pass args to the exe
#   RUN_DIR=/path/to/run ./run_offline_coupled.sh # cd to a run dir first
#   OIFS_PYTHON_HOME=/other/conda ./run_offline_coupled.sh
#
# Prerequisites
# -------------
#   * build_offline/bin/osmMASTER_coupled built (build_offline_coupled.sh).
#   * surf_pytorch/coupling/bindings/libpyoifs.so built (build_pyoifs.sh).
#   * An ``input`` namelist + forcing data in the run directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"                       # repo root
BUILD_DIR="${BUILD_DIR:-${ROOT}/build_offline}"
EXE="${BUILD_DIR}/bin/osmMASTER_coupled"

# Defaults -- overridable from the environment.
: "${OIFS_PYTHON_HOME:=/opt/anaconda3}"
: "${PY_LIBDIR:=${OIFS_PYTHON_HOME}/lib}"
: "${PY_LIBNAME:=python3.10}"
PY_LIB="${PY_LIBDIR}/lib${PY_LIBNAME}.so.1.0"

# OMP_NUM_THREADS: force single-threaded unless PYOIFS_ALLOW_OMP is set.
# cpg1s fans the surface call out over an OpenMP team; PyTorch also spawns an
# intra-op thread pool.  When both run concurrently the glibc malloc arenas
# get cross-contaminated and __libc_free in SURFRAD_CTL aborts with
# "munmap_chunk(): invalid pointer" on a worker thread.  For the NLON=NLAT=1
# offline driver there is no parallelism to exploit anyway, so a single
# thread sidesteps the arena conflict at zero cost.  We *unconditionally*
# override (not :=) because login profiles on this box export
# OMP_NUM_THREADS=128, which would otherwise resurrect the arena abort.
# A caller that has validated multi-thread safety exports PYOIFS_ALLOW_OMP=1.
if [[ -z "${PYOIFS_ALLOW_OMP:-}" ]]; then
  OMP_NUM_THREADS=1
fi
export OMP_NUM_THREADS

[[ -x "${EXE}" ]] || { echo "[run-coupled] exe not found: ${EXE}" >&2; exit 1; }
[[ -f "${PY_LIB}" ]] || { echo "[run-coupled] libpython not found: ${PY_LIB}" >&2; exit 1; }

# Optional run directory: cd there so the 'input' namelist + fort.* land in it.
if [[ -n "${RUN_DIR:-}" ]]; then
  mkdir -p "${RUN_DIR}"
  cd "${RUN_DIR}"
fi

# Export the environment the Fortran bridge reads.
export OIFS_PYTHON_HOME

# PYTHONNOUSERSITE=1: the embedded CPython must NOT consult the user site
# directory (~/.local/lib/python3.10/site-packages).  That dir can hold
# numpy/torch C-extensions built against a *different* Python than the
# anaconda one we embed (here: numpy 1.26.4 / torch 2.11.0 vs the conda
# env's 1.23.5 / 1.12.1).  Loading a mismatched numpy makes its GC-tracked
# objects disagree with the interpreter's pymalloc arena -> segfault inside
# subtype_traverse/visit_decref during a __mul__ that triggers a collection.
# Forcing it off means the interpreter uses its OWN site-packages only.
export PYTHONNOUSERSITE=1

# PYTHONPATH: the surf_pytorch *package* lives at ${ROOT}/surf_pytorch/surf_pytorch
# (the project dir ${ROOT}/surf_pytorch/ has no __init__.py, so the repo root
# alone would make it a namespace package and 'from surf_pytorch import
# LandSurface' would fail).  We add BOTH the repo root (so
# 'import coupling.bindings.pyoifs_shim' resolves) and the project dir (so
# 'import surf_pytorch' resolves to the real package).  Prepend so a
# caller-set PYTHONPATH still wins for everything else.
export PYTHONPATH="${ROOT}/surf_pytorch:${ROOT}:${PYTHONPATH:-}"

echo "[run-coupled] exe      : ${EXE}"
echo "[run-coupled] py-home  : ${OIFS_PYTHON_HOME}"
echo "[run-coupled] libpython: ${PY_LIB} (LD_PRELOAD)"
echo "[run-coupled] no-usrsite: ${PYTHONNOUSERSITE}"
echo "[run-coupled] run dir  : $(pwd)"
echo "[run-coupled] args     : $*"

# LD_PRELOAD forces ONLY libpython to the anaconda build; it does not expose
# the rest of /opt/anaconda3/lib to the loader, so libeckit/libmultio still
# get the system libstdc++/libcurl they were built against.
exec env "LD_PRELOAD=${PY_LIB}" "${EXE}" "$@"
