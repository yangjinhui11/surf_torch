/* ========================================================================== *
 * pyoifs_c.c -- C glue between OpenIFS Fortran and the PyTorch SURF port.
 *
 * This is the P1.2 component of the offline coupling (Option A: in-process
 * CPython embedding).  It is the *only* C file in the chain: the Fortran
 * thunk (pyoifs_f.F90, P1.3) calls the three exported symbols here, and
 * this file forwards each call into the Python embedding that runs the
 * PyTorch model.
 *
 * Three exported symbols (the ABI the Fortran side BIND(C)-links against)
 * --------------------------------------------------------------------------
 *   int pyoifs_init(const char *python_home, int n_tile, int n_soil,
 *                   double dt_default);
 *       Called ONCE from callpar1s at model start (LEPYOIFS branch).
 *       dlopen's libpython, initialises the interpreter (Py_Initialize),
 *       imports surf_pytorch.coupling, builds the PyOifsShim singleton,
 *       and stashes it in a module-level static.  Returns 0 on success.
 *
 *   int pyoifs_step(int klon, <pointers...>);
 *       Called ONCE PER COUPLING INTERVAL.  Acquires the GIL, builds a
 *       Python tuple of the raw C pointers wrapped as ctypes objects,
 *       calls the shim's step() method, releases the GIL.  The pointers
 *       are passed straight through -- the Python side does the zero-copy
 *       np.ctypeslib.as_array view (see pyoifs_shim.py), so no data is
 *       copied across the language boundary on the way in.
 *
 *   int pyoifs_finalize(void);
 *       Called ONCE at model end.  Drops the shim reference, Py_Finalize.
 *
 * GIL lifecycle
 * -------------
 * The embedding runs on the OpenIFS master thread.  Py_Initialize is
 * called once (under no GIL -- it acquires what it needs).  Every
 * pyoifs_step wraps its body in PyGILState_Ensure / PyGILState_Release
 * so the interpreter is never entered without the lock held.  We do NOT
 * hold the GIL between steps (that would serialise the whole run).
 *
 * Error handling
 * --------------
 * Every Python call is followed by a PyErr_Occurred() check.  On error
 * we fetch the exception text into a static buffer, print it to stderr
 * with a "[pyoifs]" prefix, and return a non-zero error code.  The
 * Fortran side surfaces this as an ABORT (the model cannot continue with
 * a broken surface).  We never silently swallow a Python exception --
 * a swallowed exception here would let the surface return stale OUT
 * buffers and corrupt the run.
 *
 * Build
 * -----
 * See build_pyoifs.sh.  Compiled with -fPIC -shared, linked against
 * libpython3.10 (resolved at load time via the interpreter's own
 * RPATH, so the .so has no hard-coded libpython dependency).
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Python C API -- we #define Py_LIMITED_API? NO: we target a specific
 * CPython 3.10 (the one libpython3.10.so belongs to), so the full API
 * is available and stable.  Pinning the include to 3.10 keeps the
 * struct layouts (PyGILState_STATE etc.) identical at build and run. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* -------------------------------------------------------------------------- *
 * ABI version cross-check.
 *
 * PYOIFS_C_ABI_VERSION must equal coupling.bindings.pyoifs_shim's
 * PYOIFS_ABI_VERSION.  pyoifs_init reads the Python attribute and bails
 * if they differ, so a rebuilt driver + a stale shim fail loudly at the
 * very first step instead of mis-reading memory.
 * -------------------------------------------------------------------------- */
#define PYOIFS_C_ABI_VERSION 1

/* -------------------------------------------------------------------------- *
 * Module state -- the shim singleton + a flag so we don't double-init.
 *
 * Single-threaded by OpenIFS's surface-call contract (the surface driver
 * runs on the master thread per grid block; MPI/OpenMP parallelism lives
 * above it).  If that ever changes, this static needs a mutex.
 * -------------------------------------------------------------------------- */
static PyObject *s_shim = NULL;          /* the PyOifsShim instance  */
static PyObject *s_step_method = NULL;   /* shim.step bound method   */
static int       s_initialised = 0;
/* Did *we* call Py_Initialize?  If the host process is itself Python
 * (e.g. the pytest harness loading this .so via ctypes), the interpreter
 * is already up before pyoifs_init runs, and Py_IsInitialized() returns
 * true -- so we skip Py_Initialize and MUST NOT call Py_Finalize (that
 * would tear down the host's interpreter mid-run).  In the real OpenIFS
 * host (a non-Python process) this is 1 and finalize is correct. */
static int       s_we_initialized = 0;

/* Last error text, surfaced to Fortran via pyoifs_last_error() and to
 * the operator via stderr.  Sized to hold a typical Python traceback
 * line (the full traceback is printed to stderr; we keep only the
 * repr() the Python side stashed via _last_error). */
static char s_last_error[1024] = "";

/* Path to the libpython we actually loaded -- recorded at init so the
 * finalize path can dlsym Py_FindExecutable / Py_GetPythonHome if a
 * future Python version needs it.  Unused today but reserved. */
static void *s_libpython = NULL;


/* -------------------------------------------------------------------------- *
 * Helpers
 * -------------------------------------------------------------------------- */

/* Record an error string + optional Python exception traceback.
 * Clears the Python error indicator (mandatory -- leaving it set
 * poisons the next Python call). */
static void record_error(const char *context) {
    if (PyErr_Occurred()) {
        PyObject *exc_type, *exc_value, *exc_tb;
        PyErr_Fetch(&exc_type, &exc_value, &exc_tb);
        PyErr_NormalizeException(&exc_type, &exc_value, &exc_tb);
        /* repr(exc_value) -- matches what pyoifs_shim._last_error holds. */
        PyObject *exc_repr = NULL;
        if (exc_value) {
            PyObject *exc_str = PyObject_Str(exc_value);
            if (exc_str) {
                exc_repr = exc_str;  /* borrow; decref below */
            }
        }
        const char *py_msg = (exc_repr && PyUnicode_Check(exc_repr))
                             ? PyUnicode_AsUTF8(exc_repr) : "(no message)";
        snprintf(s_last_error, sizeof(s_last_error),
                 "[pyoifs] %s: %s", context, py_msg);
        /* Print the full traceback to stderr for debugging. */
        PyErr_Restore(exc_type, exc_value, exc_tb);
        PyErr_Print();
        Py_XDECREF(exc_repr);
    } else {
        snprintf(s_last_error, sizeof(s_last_error),
                 "[pyoifs] %s (no Python exception set)", context);
        fprintf(stderr, "%s\n", s_last_error);
    }
}

/* dlopen libpython at the path python_home implies, so we can resolve
 * Py_Initialize etc. even when the embedding host (OpenIFS) wasn't
 * linked against libpython.  Returns the handle or NULL. */
static void *load_libpython(const char *python_home) {
    (void)python_home;  /* reserved; today we use the interpreter's libpython */
    /* On a self-hosted build libpython is already linked into this .so via
     * -lpython3.10, so Py_Initialize resolves at load time and no dlopen
     * is needed.  We keep the hook so a future "embed into an OpenIFS
     * that has no libpython" build can dlopen it here. */
    return NULL;
}


/* -------------------------------------------------------------------------- *
 * pyoifs_init -- build the embedding + the shim singleton.
 *
 * Argument list mirrors what callpar1s knows at model start: the python
 * home (where libpython + the surf_pytorch package live), and the tile /
 * soil dimensions.  We pass these to PyOifsShim's constructor so the
 * Python model is dimensioned correctly from the get-go.
 *
 * Returns 0 on success, non-zero error code otherwise.
 * -------------------------------------------------------------------------- */
int pyoifs_init(const char *python_home, int n_tile, int n_soil,
                double dt_default) {
    (void)dt_default;  /* the per-step dt comes via pyoifs_step */

    if (s_initialised) {
        fprintf(stderr, "[pyoifs] pyoifs_init called twice; ignoring\n");
        return 0;
    }

    /* 1. dlopen libpython (no-op today, see load_libpython). */
    s_libpython = load_libpython(python_home);

    /* 2. Initialise the interpreter.  We only call Py_Initialize when the
     *    host process has not already done so -- the real OpenIFS host is
     *    not Python, so Py_IsInitialized() is false here and we own the
     *    interpreter's lifetime.  If the host IS Python (the pytest
     *    harness loading this .so via ctypes), the interpreter is already
     *    up and we record s_we_initialized=0 so finalize refrains from
     *    tearing it down (Py_Finalize on the host's own interpreter is
     *    fatal).  If python_home is non-NULL we set PYTHONHOME first so a
     *    relocated conda env resolves its stdlib. */
    if (python_home && python_home[0]) {
        Py_SetPythonHome(Py_DecodeLocale(python_home, NULL));
    }
    if (!Py_IsInitialized()) {
        Py_Initialize();
        s_we_initialized = 1;
        /* After Py_Initialize the GIL is held; we drop it so the host
         * (OpenIFS) can do non-Python work between init and the first
         * step without serialising on the GIL.  We re-acquire per-step. */
        PyEval_SaveThread();
    } else {
        s_we_initialized = 0;
    }

    PyGILState_STATE gil = PyGILState_Ensure();

    /* 3. Import coupling.bindings.pyoifs_shim and build the shim.
     *    We import the *bindings* submodule directly (not just the
     *    package) so a missing surf_pytorch install fails here with a
     *    clear ImportError, not at the first step. */
    PyObject *shim_mod = PyImport_ImportModule(
        "coupling.bindings.pyoifs_shim");
    if (!shim_mod) {
        record_error("pyoifs_init: import coupling.bindings.pyoifs_shim");
        PyGILState_Release(gil);
        return 1;
    }

    /* 4. ABI version cross-check. */
    PyObject *py_abi = PyObject_GetAttrString(shim_mod, "PYOIFS_ABI_VERSION");
    if (!py_abi || !PyLong_Check(py_abi)) {
        record_error("pyoifs_init: read PYOIFS_ABI_VERSION");
        Py_XDECREF(py_abi);
        Py_DECREF(shim_mod);
        PyGILState_Release(gil);
        return 1;
    }
    long abi = PyLong_AsLong(py_abi);
    Py_DECREF(py_abi);
    if (abi != PYOIFS_C_ABI_VERSION) {
        snprintf(s_last_error, sizeof(s_last_error),
                 "[pyoifs] ABI version mismatch: C=%d Python=%ld",
                 PYOIFS_C_ABI_VERSION, abi);
        fprintf(stderr, "%s\n", s_last_error);
        Py_DECREF(shim_mod);
        PyGILState_Release(gil);
        return 1;
    }

    /* 5. Build the PyOifsShim singleton.  PyOifsShim(n_tile, n_soil). */
    PyObject *shim_cls = PyObject_GetAttrString(shim_mod, "PyOifsShim");
    Py_DECREF(shim_mod);
    if (!shim_cls || !PyCallable_Check(shim_cls)) {
        record_error("pyoifs_init: get PyOifsShim");
        Py_XDECREF(shim_cls);
        PyGILState_Release(gil);
        return 1;
    }
    s_shim = PyObject_CallFunction(shim_cls, "ii", n_tile, n_soil);
    Py_DECREF(shim_cls);
    if (!s_shim) {
        record_error("pyoifs_init: PyOifsShim()");
        PyGILState_Release(gil);
        return 1;
    }

    /* 6. Cache the bound .step method so the hot path does one
     *    PyObject_CallFunctionObjectArgs instead of a GetAttr per call. */
    s_step_method = PyObject_GetAttrString(s_shim, "step");
    if (!s_step_method || !PyCallable_Check(s_step_method)) {
        record_error("pyoifs_init: get shim.step");
        Py_XDECREF(s_step_method);
        s_step_method = NULL;
        Py_DECREF(s_shim);
        s_shim = NULL;
        PyGILState_Release(gil);
        return 1;
    }

    s_initialised = 1;
    PyGILState_Release(gil);
    fprintf(stderr, "[pyoifs] initialised: n_tile=%d n_soil=%d abi=%ld\n",
            n_tile, n_soil, abi);
    return 0;
}


/* -------------------------------------------------------------------------- *
 * pyoifs_step -- run one surface step.
 *
 * This is the hot path: called once per grid block per coupling
 * interval.  We build a ctypes array of the raw C pointers and hand
 * them to the Python shim, which does the zero-copy np.ctypeslib view.
 *
 * The argument list MUST stay byte-for-byte identical to the Python
 * pyoifs_step in pyoifs_shim.py (minus the leading `shim` arg, which
 * we hold as s_shim).  Any change requires bumping PYOIFS_C_ABI_VERSION
 * and PYOIFS_ABI_VERSION in lockstep.
 *
 * Returns 0 on success, non-zero on error.
 * -------------------------------------------------------------------------- */
int pyoifs_step(int klon,
                /* Forcing */
                double *pu, double *pv, double *pt, double *pq, double *pco2,
                double *pp_surf, double *pgeom1,
                double *pfr_so, double *pfr_th, double *pmu0,
                double *prain, double *psnow,
                double pdt,
                /* Static */
                double *ptile_frac, double *pland_frac, double *porography,
                int *psoil_type, int *pveg_type,
                /* State (in) */
                double *pskin_t, double *psoil_t, double *psoil_w_liq,
                double *psnow_w, double *psnow_frac, double *pskin_water,
                double *psst, double *psnow_t, double *psnow_rho,
                /* OUT */
                double *out_skin_t, double *out_soil_t, double *out_soil_w_liq,
                double *out_snow_w, double *out_snow_frac, double *out_skin_water,
                double *out_h_flux_tile, double *out_e_flux_tile,
                double *out_str_tile, double *out_g0_tile,
                double *out_slh_tile) {
    if (!s_initialised || !s_step_method) {
        snprintf(s_last_error, sizeof(s_last_error),
                 "[pyoifs] step called before init");
        fprintf(stderr, "%s\n", s_last_error);
        return 1;
    }

    PyGILState_STATE gil = PyGILState_Ensure();

    /* Build the argument tuple.  We use the CPython "O" format for
     * every pointer: we wrap each as a ctypes c_double / c_int pointer
     * object so the Python side can np.ctypeslib.as_array it.  Building
     * these objects is the per-step overhead -- it's ~33 small ctypes
     * allocations, dominated by malloc bookkeeping, ~tens of us on a
     * modern CPU (measured in P1.5).  The alternative (passing raw
     * integer addresses as Python ints and reconstructing on the Python
     * side) is marginally faster but loses the type information that
     * catches a mis-wired int-star / double-star at the boundary. */

    /* Import the ctypes types once per call (cached by CPython after
     * the first import).  We need POINTER(c_double) / POINTER(c_int)
     * as the *targets* of ctypes.cast, which is the robust way to wrap
     * a raw integer address into a pointer object that
     * np.ctypeslib.as_array can consume.  (The POINTER(c_double)(addr)
     * constructor does NOT accept an integer -- it expects a c_double
     * instance -- so ctypes.cast is the only correct path.) */
    PyObject *ctypes_mod = PyImport_ImportModule("ctypes");
    if (!ctypes_mod) {
        record_error("pyoifs_step: import ctypes");
        PyGILState_Release(gil);
        return 1;
    }
    PyObject *c_double = PyObject_GetAttrString(ctypes_mod, "c_double");
    PyObject *c_int    = PyObject_GetAttrString(ctypes_mod, "c_int");
    if (!c_double || !c_int) {
        record_error("pyoifs_step: get c_double/c_int");
        Py_XDECREF(c_double); Py_XDECREF(c_int); Py_DECREF(ctypes_mod);
        PyGILState_Release(gil);
        return 1;
    }
    PyObject *POINTER_double = PyObject_CallMethod(ctypes_mod,
        "POINTER", (char *)"O", c_double);
    PyObject *POINTER_int    = PyObject_CallMethod(ctypes_mod,
        "POINTER", (char *)"O", c_int);
    if (!POINTER_double || !POINTER_int) {
        record_error("pyoifs_step: build POINTER types");
        Py_XDECREF(POINTER_double); Py_XDECREF(POINTER_int);
        Py_DECREF(c_double); Py_DECREF(c_int); Py_DECREF(ctypes_mod);
        PyGILState_Release(gil);
        return 1;
    }

    /* Helper: cast a raw address into POINTER(c_double) / POINTER(c_int).
     * ctypes.cast(addr_int, target_type) returns a new pointer object
     * whose .contents aliases the C memory -- zero-copy, exactly what
     * the Python shim's np.ctypeslib.as_array expects. */
    #define WRAP_DBL(addr) \
        PyObject_CallMethod(ctypes_mod, "cast", (char *)"nO", \
                            (Py_ssize_t)(addr), POINTER_double)
    #define WRAP_INT(addr) \
        PyObject_CallMethod(ctypes_mod, "cast", (char *)"nO", \
                            (Py_ssize_t)(addr), POINTER_int)

    /* Build all the pointer-wrapper arguments.  On success we transfer
     * ownership into call_args (stealing the ref); on a partway failure
     * we decref what we built.
     *
     * Argument count (after klon, which is slot 0 of call_args):
     *   12 forcing double* + 1 pdt double + 3 static double* + 2 static int*
     *   + 6 state double* + 3 companion double* + 11 OUT double*  =  38.
     * So args[] has 38 slots and call_args has 39 (0=klon, 1..38=args). */
    PyObject *args[38];
    int i = 0;
    #define ADD(arg) do { \
        args[i] = (arg); \
        if (!args[i]) { record_error("pyoifs_step: wrap arg " #arg); \
            goto step_cleanup_partial; } \
        i++; \
    } while (0)

    /* Forcing */
    ADD(WRAP_DBL(pu)); ADD(WRAP_DBL(pv)); ADD(WRAP_DBL(pt));
    ADD(WRAP_DBL(pq)); ADD(WRAP_DBL(pco2));
    ADD(WRAP_DBL(pp_surf)); ADD(WRAP_DBL(pgeom1));
    ADD(WRAP_DBL(pfr_so)); ADD(WRAP_DBL(pfr_th)); ADD(WRAP_DBL(pmu0));
    ADD(WRAP_DBL(prain)); ADD(WRAP_DBL(psnow));
    ADD(PyFloat_FromDouble(pdt));
    /* Static */
    ADD(WRAP_DBL(ptile_frac)); ADD(WRAP_DBL(pland_frac)); ADD(WRAP_DBL(porography));
    ADD(WRAP_INT(psoil_type)); ADD(WRAP_INT(pveg_type));
    /* State */
    ADD(WRAP_DBL(pskin_t)); ADD(WRAP_DBL(psoil_t)); ADD(WRAP_DBL(psoil_w_liq));
    ADD(WRAP_DBL(psnow_w)); ADD(WRAP_DBL(psnow_frac)); ADD(WRAP_DBL(pskin_water));
    ADD(WRAP_DBL(psst)); ADD(WRAP_DBL(psnow_t)); ADD(WRAP_DBL(psnow_rho));
    /* OUT */
    ADD(WRAP_DBL(out_skin_t)); ADD(WRAP_DBL(out_soil_t)); ADD(WRAP_DBL(out_soil_w_liq));
    ADD(WRAP_DBL(out_snow_w)); ADD(WRAP_DBL(out_snow_frac)); ADD(WRAP_DBL(out_skin_water));
    ADD(WRAP_DBL(out_h_flux_tile)); ADD(WRAP_DBL(out_e_flux_tile));
    ADD(WRAP_DBL(out_str_tile)); ADD(WRAP_DBL(out_g0_tile)); ADD(WRAP_DBL(out_slh_tile));

    /* klon is the first positional arg (shim.step signature).  Build
     * the full call as step(klon, <33 pointers>).  Total 34 args. */
    {
        PyObject *call_args = PyTuple_New(39);
        if (!call_args) {
            record_error("pyoifs_step: build arg tuple");
            goto step_cleanup_partial;
        }
        PyObject *klon_obj = PyLong_FromLong(klon);
        if (!klon_obj) {
            Py_DECREF(call_args);
            record_error("pyoifs_step: PyLong_FromLong(klon)");
            goto step_cleanup_partial;
        }
        PyTuple_SET_ITEM(call_args, 0, klon_obj);  /* steals ref */
        /* args[] now holds indices 0..i-1 == the 33 pointers above.  We
         * move them (steal ref) into call_args slots 1..33. */
        for (int k = 0; k < i; k++) {
            PyTuple_SET_ITEM(call_args, 1 + k, args[k]);
            args[k] = NULL;  /* ownership transferred */
        }

        PyObject *result = PyObject_CallObject(s_step_method, call_args);
        Py_DECREF(call_args);
        if (!result) {
            record_error("pyoifs_step: shim.step() raised");
            /* args[] already NULLed (ownership moved to call_args).  Just
             * release the ctypes module refs. */
            Py_DECREF(POINTER_double); Py_DECREF(POINTER_int);
            Py_DECREF(c_double); Py_DECREF(c_int); Py_DECREF(ctypes_mod);
            PyGILState_Release(gil);
            return 1;
        }
        Py_DECREF(result);  /* PyoifsStepResult -- C side ignores it */
    }

    /* Normal cleanup: the success path transferred all args[k] into the
     * tuple which was decref'd above, so args[] is all NULL.  We just
     * drop the ctypes type refs. */
    Py_DECREF(POINTER_double); Py_DECREF(POINTER_int);
    Py_DECREF(c_double); Py_DECREF(c_int); Py_DECREF(ctypes_mod);
    PyGILState_Release(gil);
    return 0;

step_cleanup_partial:
    /* A WRAP failed partway: decref the args we did build, plus the
     * ctypes type refs, then bail. */
    for (int k = 0; k < i; k++) {
        Py_XDECREF(args[k]);
    }
    Py_DECREF(POINTER_double); Py_DECREF(POINTER_int);
    Py_DECREF(c_double); Py_DECREF(c_int); Py_DECREF(ctypes_mod);
    PyGILState_Release(gil);
    return 1;
}


/* -------------------------------------------------------------------------- *
 * pyoifs_finalize -- tear down the embedding.
 *
 * Called once at model end.  Drops the shim + step-method refs and
 * finalises the interpreter.  After this, pyoifs_init must be called
 * again before any further pyoifs_step.
 * -------------------------------------------------------------------------- */
int pyoifs_finalize(void) {
    if (!s_initialised) {
        return 0;
    }

    PyGILState_STATE gil = PyGILState_Ensure();

    Py_XDECREF(s_step_method);
    s_step_method = NULL;
    Py_XDECREF(s_shim);
    s_shim = NULL;
    s_initialised = 0;

    PyGILState_Release(gil);

    /* We deliberately do NOT call Py_FinalizeEx here.  PyTorch (imported by
     * the shim) owns a persistent intra-op thread pool that survives until
     * process exit; CPython's finalizer (wait_for_thread_shutdown) cannot
     * join those threads and then dereferences already-torn-down interpreter
     * state (_PyInterpreterState_GET segfaults inside Py_FinalizeEx).  This
     * is a well-known hazard of embedding CPython with PyTorch loaded --
     * python.org's own embedding guide recommends NOT finalising in a host.
     *
     * Skipping finalize is safe for this coupling: the OpenIFS offline driver
     * calls pyoifs_finalize exactly once, at the very end of cntend, and the
     * process exits immediately after.  Leaked shim/torch state is reclaimed
     * by the OS at exit, so there is no resource accumulation across a run.
     * (The pytest harness path keeps working because s_we_initialized==0
     * there already skipped finalize.) */
    s_we_initialized = 0;

    s_libpython = NULL;
    fprintf(stderr, "[pyoifs] finalised (interpreter left running -- OS reclaims at exit)\n");
    return 0;
}


/* -------------------------------------------------------------------------- *
 * pyoifs_last_error -- return the most recent error text (for Fortran
 * to surface in its ABORT message).  Returns a pointer to a static
 * buffer valid until the next pyoifs_* call.
 * -------------------------------------------------------------------------- */
const char *pyoifs_last_error(void) {
    return s_last_error;
}
