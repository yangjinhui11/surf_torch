#!/usr/bin/env bash
# build_offline_coupled.sh -- build the PyTorch-coupled offline surface exe.
#
# Produces ``${BUILD_DIR}/bin/osmMASTER_coupled`` -- a copy of the OpenIFS
# offline surface driver (``osmMASTER``) whose ``callpar1s`` has been patched
# with an ``LEPYOIFS`` branch that redirects the surface step to the PyTorch
# port via the libpyoifs.so bridge:
#
#     osmMASTER_coupled
#        |  callpar1s (LEPYOIFS branch) -> PYOIFS_SURFTSTP
#        v
#     pyoifs_f.F90  (MODULE PYOIFS_F, this tree)
#        |  BIND(C) -> pyoifs_init/step/finalize
#        v
#     libpyoifs.so  (surfac_pytorch/coupling/bindings/)
#        |  PyGILState_Ensure / PyObject_CallObject
#        v
#     surf_pytorch.LandSurface.forward
#
# Strategy
# --------
# The offline driver is built by an ecbuild/CMake tree (BUILD_DIR).  Rather
# than hand-roll a parallel build (which would have to reproduce the ~40
# dependency .so link line), this script:
#   1. Drives the existing CMake target so callpar1s.F90 + yoephy.F90 are
#      recompiled by the real IFS toolchain (/usr/bin/f95, gfortran-11) with
#      the full IFS module tree.  This is the authoritative syntax/type check
#      for the P1.3 Fortran edits.
#   2. Compiles pyoifs_f.F90 standalone with the same gfortran, emitting
#      pyoifs_f.mod into the build's module include path so callpar1s's
#      ``USE PYOIFS_F`` resolves.
#   3. Re-links: takes the CMake-generated osmMASTER link command verbatim
#      (link.txt), and appends the pyoifs_f object + -lpyoifs + -lpython3.10,
#      changing only the output name.  This inherits every dependency the
#      original exe already links.
#
# Usage
# -----
#   ./build_offline_coupled.sh            # build bin/osmMASTER_coupled
#   ./build_offline_coupled.sh relink     # skip step 1, just relink
#   BUILD_DIR=/path ./build_offline_coupled.sh
#
# Prerequisites
# -------------
#   * build_offline/ configured and the baseline osmMASTER built once.
#   * surf_pytorch/coupling/bindings/libpyoifs.so built (build_pyoifs.sh).
#   * surf_pytorch/coupling/bindings/fortran/pyoifs_f.F90 present.
#   * libpython3.10 at PY_LIBDIR (default /opt/anaconda3/lib).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                 # surf_pytorch/
BINDINGS_DIR="${ROOT}/coupling/bindings"
THUNK_SRC="${BINDINGS_DIR}/fortran/pyoifs_f.F90"

# Defaults -- overridable from the environment.
: "${BUILD_DIR:=/home/qixiang/yangjinhui/openifs/build_offline}"
: "${PY_LIBDIR:=/opt/anaconda3/lib}"
: "${PY_LIBNAME:=python3.10}"

mode="${1:-build}"

[[ -d "${BUILD_DIR}" ]] || { echo "[coupled] BUILD_DIR not found: ${BUILD_DIR}" >&2; exit 1; }
[[ -f "${THUNK_SRC}"  ]] || { echo "[coupled] thunk source missing: ${THUNK_SRC}" >&2; exit 1; }
[[ -f "${BINDINGS_DIR}/libpyoifs.so" ]] || { echo "[coupled] libpyoifs.so missing -- run build_pyoifs.sh first" >&2; exit 1; }

MODULE_DIR="${BUILD_DIR}/module"
OSM_DIR="${BUILD_DIR}/CMakeFiles/osmMASTER.dir"
LINK_TXT="${OSM_DIR}/link.txt"
OUT_EXE="${BUILD_DIR}/bin/osmMASTER_coupled"

# --------------------------------------------------------------------------- #
# Step 1 -- compile the thunk FIRST so pyoifs_f.mod is on the module path
# before CMake tries to recompile callpar1s (whose ``USE PYOIFS_F`` needs it).
# The thunk has no IFS module deps, so a minimal gfortran invocation suffices;
# emit the .mod into the build's module include dir and the .o next to the
# other osmMASTER objects.
# --------------------------------------------------------------------------- #
echo "[coupled] (1/3) compiling pyoifs_f.F90 -> module path..."
PYOIFS_OBJ="${OSM_DIR}/pyoifs_f.o"
/usr/bin/f95 -c -O2 -fPIC -fno-range-check -ffree-line-length-none \
  -J "${MODULE_DIR}" -I "${MODULE_DIR}" \
  "${THUNK_SRC}" -o "${PYOIFS_OBJ}"
[[ -f "${MODULE_DIR}/pyoifs_f.mod" ]] || { echo "[coupled] pyoifs_f.mod not emitted" >&2; exit 1; }
echo "[coupled]      pyoifs_f.mod + pyoifs_f.o ready"

# --------------------------------------------------------------------------- #
# Step 2 -- recompile the modified Fortran via the CMake target.
# Now that pyoifs_f.mod is present, ``USE PYOIFS_F`` resolves and callpar1s
# compiles end to end.  The build will still fail at *link* time (libpyoifs
# is not on the CMake link line) -- that's expected and fine; step 3 does
# the real link.  A *compilation* failure on callpar1s/yophy is the signal.
# --------------------------------------------------------------------------- #
if [[ "${mode}" != "relink" ]]; then
  echo "[coupled] (2/3) rebuilding osmMASTER sources via CMake..."
  ( cd "${BUILD_DIR}" && cmake --build . --target osmMASTER -- -j4 2>&1 \
      | grep -E 'Building Fortran|Error|Fatal|error:|undefined reference' || true ) || true
  if ! ls "${OSM_DIR}/surf/offline/driver/callpar1s.F90.o" >/dev/null 2>&1; then
    echo "[coupled] callpar1s.F90 did not compile -- see CMake output above" >&2
    exit 1
  fi
  echo "[coupled]      callpar1s.F90.o rebuilt OK"
fi

# --------------------------------------------------------------------------- #
# Step 3 -- relink as osmMASTER_coupled, appending the pyoifs bridge.
# We reuse the CMake link command verbatim and only swap the output name
# and tack on the extra object + libraries.
# --------------------------------------------------------------------------- #
echo "[coupled] (3/3) linking ${OUT_EXE}..."

# Reuse the CMake link command exactly the way CMake invokes it
# (``cmake -E cmake_link_script link.txt``).  We copy link.txt to a sibling
# file, edit the two osmMASTER-named tokens in place (output exe + Map file),
# and append the bridge objects/libs.  Running it through cmake_link_script
# rather than ``eval`` preserves CMake's exact link semantics (it has been
# observed that a raw ``eval`` of the link line breaks the final
# -Wl,-rpath-link transitive-resolution token, producing spurious curl/C++
# undefined references that the original exe does not have).
LINK_COUPL="${OSM_DIR}/link_coupled.txt"

# Copy link.txt WITHOUT its trailing newline.  CMake's link.txt ends with
# '\n', and we must append the bridge args onto the SAME line as the link
# command -- cmake_link_script runs each line as a separate command, so a
# trailing newline would split the bridge off into its own (bogus) command
# and the real link would run without pyoifs_f.o / -lpyoifs on it.
printf '%s' "$(cat "${LINK_TXT}")" > "${LINK_COUPL}"

# Pre-existing build-tree quirk: this build_offline tree was configured
# against ``eccodes`` but the working sibling (verInt) uses ``eccodes_back``.
# Linking against ``eccodes`` fails because libeckit's transitive curl/C++
# symbols cannot be resolved (eccodes_back/lib carries the eccodes build
# whose own DT_NEEDED chain resolves cleanly).  Swap eccodes -> eccodes_back
# everywhere in the link line so the coupled exe links the same way the
# known-good verInt osmMASTER does.  This is a property of the build tree,
# not of the coupling.
sed -i 's#app/eccodes/lib#app/eccodes_back/lib#g' "${LINK_COUPL}"

# Edit the output name and the linker Map file.  These are the only two
# tokens in link.txt that mention the osmMASTER target name.
sed -i \
  -e 's#-Wl,-Map=load.map.osmMASTER#-Wl,-Map=load.map.osmMASTER_coupled#g' \
  -e 's# -o bin/osmMASTER # -o '"${OUT_EXE}"' #g' \
  "${LINK_COUPL}"

# Append the bridge onto the same line: thunk object, libpyoifs (+dir+rpath),
# libpython by full path, then the system transitive deps.
#
# Why these exact choices:
#  * libpython3.10 is referenced by its FULL .so path (not -L/-l) so we avoid
#    adding ``-L/opt/anaconda3/lib`` to the link line; that directory holds
#    anaconda's own libcurl/libstdc++ which lack the ``@CURL_OPENSSL_4`` /
#    ``@GLIBCXX_3.4.30`` symbol versions libeckit/libmultio/libnetcdf were
#    built against -- exposing them at link time would trigger spurious curl/C++
#    undefined references.  Explicit ``-lcurl -lstdc++`` resolve the transitive
#    needs against the SYSTEM /usr/lib/x86_64-linux-gnu versions.
#  * We do NOT add ``-Wl,-rpath,/opt/anaconda3/lib`` either: doing so would
#    make the RUNTIME loader find anaconda's libstdc++.so.6 / libcurl.so.4
#    first (it lacks GLIBCXX_3.4.30 / CURL_OPENSSL_4), breaking libmultio /
#    libatlas / libeckit at startup.  Instead libpython is resolved at run
#    time via LD_LIBRARY_PATH (see run wrapper / test harness), which can be
#    scoped to the Python interpreter only and does not pollute the C++/curl
#    resolution that the OpenIFS shared libs depend on.
PY_LIB="${PY_LIBDIR}/lib${PY_LIBNAME}.so"
{
  printf ' %s' "${PYOIFS_OBJ}"
  printf ' -L%s -Wl,-rpath,%s -lpyoifs' "${BINDINGS_DIR}" "${BINDINGS_DIR}"
  printf ' %s' "${PY_LIB}"
  printf ' -lcurl -lstdc++'
  printf '\n'
} >> "${LINK_COUPL}"

# Execute exactly as CMake would.  cmake_link_script runs the file with /bin/sh
# from the current directory, so cwd must be BUILD_DIR (object paths in
# link.txt are build-dir-relative).
( cd "${BUILD_DIR}" && cmake -E cmake_link_script "${LINK_COUPL}" --verbose=ON )

if [[ ! -x "${OUT_EXE}" ]]; then
  echo "[coupled] link did not produce ${OUT_EXE}" >&2
  exit 1
fi

echo "[coupled] done: ${OUT_EXE}"
echo "[coupled] verify symbols:"
nm -D "${BINDINGS_DIR}/libpyoifs.so" 2>/dev/null | grep -qE 'pyoifs_(init|step|finalize)' \
  && echo "    libpyoifs exports pyoifs_*  OK" \
  || echo "    WARN: pyoifs_* not found in libpyoifs.so"
ldd "${OUT_EXE}" 2>/dev/null | grep -E 'libpyoifs|libpython' \
  && echo "    runtime linkage to pyoifs/python resolved" \
  || echo "    WARN: libpyoifs/libpython not in ldd output (check rpath)"
