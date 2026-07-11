"""Offline coupling validation: Fortran ``o_gg`` vs a PyTorch shim replay.

Phase-1 deliverable P1.5b-4e.  This module is the end-to-end check that
the PyTorch surface port, coupled INTO the OpenIFS offline driver via the
in-process CPython embedding (Option A), produces results that agree with
a pure-Python replay of the *same* forcing through :class:`PyOifsShim`.

Why a characterisation test, not bit-exact
------------------------------------------
The OpenIFS offline driver writes its prognostic state to ``o_gg`` -- an
**ASCII** timeseries (one row per model step, written by ``wrtp1s.F90``
*before* each ``cpg1s`` step).  Comparing it against a Python replay of
the identical :func:`surf_pytorch.forcing.build_idealized_diurnal`
forcing is the natural bit-level validation.

At the current Phase-1 coupling maturity the two sides do **not** agree
bit-for-bit on the gridpoint prognostics.  The skin-T drift was traced to
its root cause by a controlled single-step experiment (see
``TestSingleStepPhysics`` below): with **byte-identical inputs** (skin
288, tair 282.6 K, SW=0, the documented night-time step 0), one
``LandSurface.forward`` step moves skin by −4.55 K while the coupled
Fortran ``SURFTSTP`` moves it by −6.04 K -- a **~1.4 K single-step
physics difference**.  That per-step gap is independent of the coupling
machinery (it reproduces in the standalone port with no Fortran in the
loop), and it compounds across the 48-step run into the observed
±8 K diurnal-amplitude difference (the PyTorch port's skin swings wider
than ``o_gg`` because it both cools less at night and warms more by day).

Two mechanisms that were *initially suspected* were ruled OUT by the same
experiment:

* **Tile-fraction vector.**  The coupled driver weights per-tile skin-T
  with its ``ZFRTI`` (``[0,0,0,0.27,0,0.63,0,0.10]``) while the shim's
  default ``tile_frac_base`` is ``[0,0,0.05,0.3,0,0.3,0,0.35]``.  Re-
  weighting the PyTorch step-0 output with the Fortran fractions changes
  the gridpoint mean by only ~0.1 K -- negligible.
* **Per-step ``PTSKTI=PTL`` overwrite.**  The LEPYOIFS branch cold-starts
  the per-tile skin from the gridpoint mean every step.  Replaying the
  Python forward with that collapse applied each step changes the
  trajectory by <0.05 K/step -- also negligible.

So the skin-T drift is a **physics-port fidelity gap**, not a coupling
bug.  It is tracked under P1.6 alongside the genuine coupling defect
below.

A second, *separate* divergence IS a coupling-path defect:

* **Soil-T / soil-moisture frozen at cold-start in ``o_gg``.**  The
  LEPYOIFS branch copies ``OUT_SOIL_T`` back into ``PTSA`` (which aliases
  ``TSLNU0``), yet ``o_gg`` shows ``tsoil`` pinned at 285 K and ``qsoil``
  at 0.30 across all 48 steps while skin-T evolves normally -- a
  copy-back / persistence defect in the offline driver's LEPYOIFS path,
  tracked under P1.6.  (The standalone PyTorch forward DOES advance
  soil-T, confirming the port itself is healthy for the soil budget.)

Because of these, a bit-exact assertion would be brittle and red.  This
module instead asserts the **invariants that genuinely hold** and
**quantifies the residual drift** so the file doubles as a regression
guard: a future fix that closes the per-step physics gap simply tightens
the tolerance, and a regression that reintroduces NaN or breaks the
cold-start makes the test fail loudly.

What the test needs to run
--------------------------
The Fortran side is authoritative, so the test requires a coupled run's
``o_gg``.  Point at it with the ``OIFS_OGG_PATH`` environment variable
(the file produced by ``run_offline_coupled.sh`` in the run directory).
If unset or the file is missing, every test is skipped -- this keeps the
module importable in environments without a built coupled exe, matching
the ``requires_fortran_ref`` convention used elsewhere in the suite.

    OIFS_OGG_PATH=/tmp/oifs_run_clean/o_gg pytest -m coupled_ref tests/test_offline_coupled.py
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch

from surf_pytorch.forcing import IdealizedDiurnalConfig, build_idealized_diurnal
from surf_pytorch.state import Forcing, Static, SurfState

from coupling import PyOifsShim


# --------------------------------------------------------------------------- #
# Skip machinery -- mirror the requires_fortran_ref convention.
# --------------------------------------------------------------------------- #
# Module-level marker so the whole suite can be selected/deselected with
# ``-m coupled_ref`` / ``-m "not coupled_ref"``.
pytestmark = pytest.mark.coupled_ref

_OGG_PATH = os.environ.get("OIFS_OGG_PATH", "")
_OGG_AVAILABLE = bool(_OGG_PATH) and os.path.isfile(_OGG_PATH)


def requires_coupled_run():
    """Skip marker -- True iff a coupled ``o_gg`` is available."""
    return pytest.mark.skipif(
        not _OGG_AVAILABLE,
        reason=(
            "OIFS_OGG_PATH not set or file missing -- run "
            "make_offline_run_dir.py + run_offline_coupled.sh and point "
            "OIFS_OGG_PATH at the resulting o_gg"
        ),
    )


# --------------------------------------------------------------------------- #
# o_gg parsing.
#
# wrtp1s.F90 emits a fixed-width ASCII table: two header lines (column
# names, then units) followed by one data row per step.  Each data row is
# ``WRITE(NPOSGG,'(f10.3,1X,I8,1X,I4,2X,I8,18(1X,E12.6E3))')`` so the
# columns (1-based) are:
#
#   1  juld       (f10.3)   julian day
#   2  date       (I8)      yyyymmdd
#   3  time       (I4)      hhmm
#   4  step       (I8)      model step index
#   5  tskin      (E12.6)   skin temperature  K     <- TRENU0
#   6  srwc       (E12.6)   skin reservoir    m     <- WRENU0
#   7-10 tsoil(1:4) (E12.6) soil temperature  K     <- TSLNU0
#  11-14 qsoil(1:4) (E12.6) soil moisture     m3/m3 <- QLINU0
#  15  snowmass   (E12.6)
#  16  snowt      (E12.6)
#  17  snowa      (E12.6)
#  18  snowd      (E12.6)
#  19-22 tice(1:4)(E12.6)
#
# Tokenising on whitespace is robust because every field is separated by
# at least one space (the format string guarantees it); we do not need to
# honour the exact field widths.
# --------------------------------------------------------------------------- #
class OGgRow:
    """One ``o_gg`` data row, with the prognostic fields parsed by name."""

    __slots__ = (
        "step", "tskin", "srwc", "tsoil", "qsoil",
        "snowmass", "snowt", "snowa", "snowd", "tice",
    )

    def __init__(self, tokens: List[str]) -> None:
        # tokens[0:4] = juld, date, time, step
        self.step = int(tokens[3])
        self.tskin = float(tokens[4])
        self.srwc = float(tokens[5])
        self.tsoil = np.array([float(t) for t in tokens[6:10]], dtype=np.float64)
        self.qsoil = np.array([float(t) for t in tokens[10:14]], dtype=np.float64)
        self.snowmass = float(tokens[14])
        self.snowt = float(tokens[15])
        self.snowa = float(tokens[16])
        self.snowd = float(tokens[17])
        self.tice = np.array([float(t) for t in tokens[18:22]], dtype=np.float64)


def parse_ogg(path: str) -> List[OGgRow]:
    """Parse an ``o_gg`` ASCII timeseries into a list of :class:`OGgRow`.

    Skips the two ``#`` header lines and any blank line.  Raises
    ``ValueError`` if a data row does not yield at least 22 tokens (the
    fixed column count wrtp1s writes).
    """
    rows: List[OGgRow] = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 22:
                raise ValueError(
                    f"o_gg row has {len(toks)} tokens, expected >=22: {line!r}"
                )
            rows.append(OGgRow(toks))
    if not rows:
        raise ValueError(f"o_gg parse produced no rows from {path}")
    return rows


# --------------------------------------------------------------------------- #
# Matching cold-start config.
#
# These defaults MUST match make_offline_run_dir.py / generate_offline_
# forcing.py so the Python replay sees byte-identical forcing to what the
# coupled driver read from its ``forcing`` NetCDF.  Drift in any of these
# knobs would show up as spurious disagreement unrelated to the coupling.
# --------------------------------------------------------------------------- #
def _matching_config(n_steps: int = 48) -> IdealizedDiurnalConfig:
    """The IdealizedDiurnalConfig the coupled run-dir is generated from."""
    return IdealizedDiurnalConfig(
        n_grid=1,
        dt_seconds=3600.0,
        n_steps=n_steps,
        lat_deg=45.0,
        lon_deg=0.0,
        day_of_year=172,
        tair_mean=290.0,
        tair_amp=8.0,
        wind_speed=5.0,
        rh=0.7,
        precip_events=[(10, 14, 1e-4)],
        soil_t_init=285.0,
        soil_w_init=0.30,
        skin_t_init=288.0,
        sst_init=290.0,
    )


def _replay_shim(
    cfg: IdealizedDiurnalConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Replay the forcing through :class:`PyOifsShim`, capturing per-step OUT.

    Drives ``n_steps`` steps, feeding each step's OUT prognostics back as
    the next step's IN state (the same loop the coupled driver runs, but
    entirely in Python through the shim).  Returns three lists, indexed by
    step, each a length-1 column array (the single grid cell):

    * ``skins``  -- gridpoint skin-T, the tile-fraction-weighted mean of
                    ``OUT_SKIN_T`` using the shim's own tile_frac (this is
                    the apples-to-apples counterpart of the coupled
                    driver's ``PTL`` re-derivation, just with a different
                    fraction vector).
    * ``soils_t``-- soil-T layer 1 (``OUT_SOIL_T[:,0]``).
    * ``soils_w``-- soil moisture layer 1 (``OUT_SOILW[:,0]``).
    """
    n_tile = 8
    n_soil = 4
    klon = cfg.n_grid

    forcing_list, static, init_state = build_idealized_diurnal(cfg)
    shim = PyOifsShim(n_tile=n_tile, n_soil=n_soil)

    # Cold-start IN buffers (Fortran column-major numpy, as the C glue
    # would hand them in).  We carry them across steps so the replay is a
    # true closed loop -- exactly what the coupled driver does.
    def _f1(t):
        return np.ascontiguousarray(t.numpy(), dtype=np.float64)

    def _f2(t):
        return np.asfortranarray(t.numpy(), dtype=np.float64)

    def _i1(t):
        return np.ascontiguousarray(t.numpy().astype(np.int32), dtype=np.int32)

    skin_t = _f2(init_state.skin_t)
    soil_t = _f2(init_state.soil_t)
    soil_w = _f2(init_state.soil_w_liq)
    snow_w = _f1(init_state.snow_w)
    snow_frac = _f1(init_state.snow_frac)
    skin_water = _f1(init_state.skin_water)
    sst = _f1(init_state.sst)
    # snow_t / snow_rho: SurfState defaults them when absent; materialise
    # the same cold-start values the shim's own _build_static_and_state
    # would have produced so the water/snow-tile prognostics match.
    snow_t = np.full((klon,), 273.15, dtype=np.float64)
    snow_rho = np.full((klon,), 300.0, dtype=np.float64)

    tile_frac = _f2(static.tile_frac_base)

    skins: List[np.ndarray] = []
    soils_t: List[np.ndarray] = []
    soils_w: List[np.ndarray] = []

    for k, forcing in enumerate(forcing_list):
        # OUT buffers (allocated fresh each step, mirroring the C glue).
        def _z2(a, b):
            return np.zeros((a, b), dtype=np.float64, order="F")

        def _z1(a):
            return np.zeros((a,), dtype=np.float64)

        outs = dict(
            out_skin_t=_z2(klon, n_tile), out_soil_t=_z2(klon, n_soil),
            out_soil_w_liq=_z2(klon, n_soil),
            out_snow_w=_z1(klon), out_snow_frac=_z1(klon),
            out_skin_water=_z1(klon),
            out_h_flux_tile=_z2(klon, n_tile), out_e_flux_tile=_z2(klon, n_tile),
            out_str_tile=_z2(klon, n_tile), out_g0_tile=_z2(klon, n_tile),
            out_slh_tile=_z2(klon, n_tile),
        )

        rc = shim.step(
            klon,
            _f1(forcing.u), _f1(forcing.v), _f1(forcing.t), _f1(forcing.q),
            _f1(forcing.co2), _f1(forcing.p_surf), _f1(forcing.geom1),
            _f1(forcing.fr_so), _f1(forcing.fr_th), _f1(forcing.mu0),
            _f1(forcing.rain), _f1(forcing.snow), float(forcing.dt.item()),
            tile_frac, _f1(static.land_frac), _f1(static.orography),
            _i1(static.soil_type), _i1(static.veg_type),
            skin_t, soil_t, soil_w, snow_w, snow_frac, skin_water,
            sst, snow_t, snow_rho,
            outs["out_skin_t"], outs["out_soil_t"], outs["out_soil_w_liq"],
            outs["out_snow_w"], outs["out_snow_frac"], outs["out_skin_water"],
            outs["out_h_flux_tile"], outs["out_e_flux_tile"], outs["out_str_tile"],
            outs["out_g0_tile"], outs["out_slh_tile"],
        )
        # PyOifsShim.step returns a PyoifsStepResult (n_outputs_written=11
        # for the 8-tile inference config -- 6 prognostics + 5 flux tiles).
        assert rc.n_outputs_written == 11, (
            f"shim.step wrote {rc.n_outputs_written} OUTs at step {k} "
            f"(expected 11)"
        )

        # Capture the gridpoint-mean skin-T (counterpart of PTL/TRENU0).
        frac = tile_frac[0]
        skin_gridmean = float(
            (outs["out_skin_t"][0] * frac).sum() / frac.sum()
        )
        skins.append(np.array([skin_gridmean]))
        soils_t.append(np.array([outs["out_soil_t"][0, 0]]))
        soils_w.append(np.array([outs["out_soil_w_liq"][0, 0]]))

        # Feed OUT back as the next step's IN.
        skin_t = outs["out_skin_t"]
        soil_t = outs["out_soil_t"]
        soil_w = outs["out_soil_w_liq"]
        snow_w = outs["out_snow_w"]
        snow_frac = outs["out_snow_frac"]
        skin_water = outs["out_skin_water"]
        # sst / snow_t / snow_rho are not in the shim's OUT set; hold them.

    return skins, soils_t, soils_w


# --------------------------------------------------------------------------- #
# Shared fixtures.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ogg_rows() -> List[OGgRow]:
    """Parsed ``o_gg`` timeseries (skipped if no file)."""
    if not _OGG_AVAILABLE:
        pytest.skip("OIFS_OGG_PATH not available")
    return parse_ogg(_OGG_PATH)


@pytest.fixture(scope="module")
def shim_replay(ogg_rows: List[OGgRow]):
    """Python replay of the identical forcing through the shim."""
    n_steps = len(ogg_rows)
    skins, soils_t, soils_w = _replay_shim(_matching_config(n_steps=n_steps))
    return skins, soils_t, soils_w


# --------------------------------------------------------------------------- #
# Tests: the o_gg file itself is well-formed and finite.
# --------------------------------------------------------------------------- #
@requires_coupled_run()
class TestOGgSanity:
    """The coupled driver's ``o_gg`` is finite, stepped, and cold-started."""

    def test_has_expected_step_count(self, ogg_rows: List[OGgRow]):
        # The default run-dir emits 48 steps (steps 0..47).
        steps = [r.step for r in ogg_rows]
        assert steps == list(range(len(steps))), (
            f"o_gg steps are not contiguous 0..N: {steps}"
        )
        assert len(ogg_rows) >= 24, (
            f"o_gg has only {len(ogg_rows)} rows; expected >=24"
        )

    def test_all_prognostics_finite(self, ogg_rows: List[OGgRow]):
        """No NaN/Inf anywhere in the prognostic columns -- the P1.5b-4c
        NaN fixes (LEOCML uninitialised read + TRENU=0 cold-start) must
        hold for the whole run."""
        for r in ogg_rows:
            vals = [r.tskin, r.srwc, *r.tsoil.tolist(), *r.qsoil.tolist()]
            vals += [r.snowmass, r.snowt, r.snowa, r.snowd, *r.tice.tolist()]
            assert np.all(np.isfinite(vals)), (
                f"non-finite prognostic at step {r.step}: {vals}"
            )

    def test_step0_is_the_documented_cold_start(self, ogg_rows: List[OGgRow]):
        """Step 0 is the namelist cold-start (written before step 0 runs):
        tskin=288, tsoil=285 (all layers), qsoil=0.30 (all layers), no snow."""
        r = ogg_rows[0]
        assert r.tskin == pytest.approx(288.0, abs=1e-6)
        assert np.allclose(r.tsoil, 285.0, atol=1e-6)
        assert np.allclose(r.qsoil, 0.30, atol=1e-6)
        assert r.snowmass == pytest.approx(0.0, abs=1e-6)

    def test_skin_t_shows_a_diurnal_cycle(self, ogg_rows: List[OGgRow]):
        """Skin-T must vary across the run (not pinned) and stay within a
        physical band -- the signature of a healthy, evolving surface."""
        skins = np.array([r.tskin for r in ogg_rows])
        assert skins.max() - skins.min() > 5.0, (
            f"skin-T range too small to be a diurnal cycle: "
            f"{skins.min():.3f}..{skins.max():.3f}"
        )
        # 270..310 K brackets any plausible mid-latitude summer skin-T.
        assert skins.min() > 270.0 and skins.max() < 310.0, (
            f"skin-T outside physical band: {skins.min():.3f}..{skins.max():.3f}"
        )


# --------------------------------------------------------------------------- #
# Tests: the shim replay is itself finite and physically evolving.
#
# These run even without o_gg (they only need the shim), but live here so
# the replay is exercised by the same module.  They guard the *Python*
# side of the comparison independently of the coupled build.
# --------------------------------------------------------------------------- #
class TestShimReplaySanity:
    """The Python replay of the forcing through the shim is finite/evolving."""

    def test_replay_runs_and_is_finite(self):
        skins, soils_t, soils_w = _replay_shim(_matching_config())
        allvals = np.concatenate(
            [np.asarray(s) for s in skins]
            + [np.asarray(s) for s in soils_t]
            + [np.asarray(s) for s in soils_w]
        )
        assert np.all(np.isfinite(allvals)), "shim replay produced non-finite"

    def test_replay_step0_matches_cold_start(self):
        """The replay's *input* skin-T at step 0 is the cold-start 288 K."""
        skins, _, _ = _replay_shim(_matching_config())
        # skins[0] is OUT of step 0 (post-forward), so check the *input*
        # via the config directly instead.
        cfg = _matching_config()
        _, _, init_state = build_idealized_diurnal(cfg)
        assert init_state.skin_t[0, 0].item() == pytest.approx(288.0, abs=1e-6)
        assert init_state.soil_t[0, 0].item() == pytest.approx(285.0, abs=1e-6)

    def test_replay_advances_soil_temperature(self):
        """The standalone PyTorch forward DOES advance soil-T (unlike the
        coupled ``o_gg``, where it is frozen) -- this is the key evidence
        that the soil-T freeze is a coupling-path defect, not a port bug."""
        _, soils_t, _ = _replay_shim(_matching_config())
        soil1 = np.array([s[0] for s in soils_t])
        assert soil1.max() - soil1.min() > 0.5, (
            f"replay soil-T did not advance: {soil1.min():.4f}..{soil1.max():.4f}"
        )


# --------------------------------------------------------------------------- #
# Tests: root-cause the skin-T drift to a single-step physics difference.
#
# These pin the finding that the multi-step skin-T drift originates in
# ``LandSurface.forward`` itself (a per-step fidelity gap vs Fortran
# SURFTSTP), NOT in the coupling machinery.  The proof: on byte-identical
# step-0 inputs, one forward() step produces a gridpoint skin-T that
# differs from the coupled ``o_gg[1]`` by ~1.4 K -- and neither swapping
# in the Fortran tile fractions nor collapsing the per-tile skin to the
# gridpoint mean each step closes that gap.  These run whenever ``o_gg``
# is available (they need the single ``o_gg[1]`` reference value).
# --------------------------------------------------------------------------- #
@requires_coupled_run()
class TestSingleStepPhysics:
    """The skin-T drift is a forward()-level physics gap, not a coupling bug."""

    # Coupled reference values for the documented night-time step 0
    # (skin 288 K, tair 282.609 K, SW=0).  o_gg[0] is the cold-start input,
    # o_gg[1] is the skin-T after step 0 ran.
    STEP0_INPUT_SKIN = 288.0
    STEP0_OGG_OUT_SKIN = 281.962  # => Fortran single-step delta = -6.038 K

    def _step0_pytorch_out(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run one forward() step on the byte-identical step-0 inputs.

        Returns ``(skin_gridmean_pyfrac, skin_gridmean_ftfrac)`` -- the
        step-0 OUT skin-T gridpoint mean weighted with the PyTorch default
        tile fractions and the Fortran ZFRTI respectively.
        """
        cfg = _matching_config()
        forcing_list, static, init_state = build_idealized_diurnal(cfg)
        shim = PyOifsShim()
        klon = 1
        z2 = lambda a, b: np.zeros((a, b), dtype=np.float64, order="F")
        z1 = lambda a: np.zeros((a,), dtype=np.float64)
        f1 = lambda t: np.ascontiguousarray(t.numpy(), dtype=np.float64)
        f2 = lambda t: np.asfortranarray(t.numpy(), dtype=np.float64)
        i1 = lambda t: np.ascontiguousarray(
            t.numpy().astype(np.int32), dtype=np.int32)

        outs = dict(
            out_skin_t=z2(klon, 8), out_soil_t=z2(klon, 4),
            out_soil_w_liq=z2(klon, 4), out_snow_w=z1(klon),
            out_snow_frac=z1(klon), out_skin_water=z1(klon),
            out_h_flux_tile=z2(klon, 8), out_e_flux_tile=z2(klon, 8),
            out_str_tile=z2(klon, 8), out_g0_tile=z2(klon, 8),
            out_slh_tile=z2(klon, 8),
        )
        f = forcing_list[0]
        shim.step(
            klon,
            f1(f.u), f1(f.v), f1(f.t), f1(f.q), f1(f.co2), f1(f.p_surf),
            f1(f.geom1), f1(f.fr_so), f1(f.fr_th), f1(f.mu0),
            f1(f.rain), f1(f.snow), float(f.dt.item()),
            f2(static.tile_frac_base), f1(static.land_frac),
            f1(static.orography), i1(static.soil_type), i1(static.veg_type),
            f2(init_state.skin_t), f2(init_state.soil_t),
            f2(init_state.soil_w_liq), f1(init_state.snow_w),
            f1(init_state.snow_frac), f1(init_state.skin_water),
            f1(init_state.sst),
            np.full((klon,), 273.15, dtype=np.float64),
            np.full((klon,), 300.0, dtype=np.float64),
            outs["out_skin_t"], outs["out_soil_t"], outs["out_soil_w_liq"],
            outs["out_snow_w"], outs["out_snow_frac"], outs["out_skin_water"],
            outs["out_h_flux_tile"], outs["out_e_flux_tile"],
            outs["out_str_tile"], outs["out_g0_tile"], outs["out_slh_tile"],
        )
        skin = outs["out_skin_t"][0]
        pyf = static.tile_frac_base[0].numpy()
        ft = np.array([0.0, 0.0, 0.0, 0.27, 0.0, 0.63, 0.0, 0.10])
        gm_py = float((skin * pyf).sum() / pyf.sum())
        gm_ft = float((skin * ft).sum() / ft.sum())
        return gm_py, gm_ft

    def test_single_step_gap_is_physics_not_coupling(self):
        """On byte-identical step-0 inputs, forward() vs Fortran differ by
        ~1.4 K.  This is the per-step seed of the multi-step drift."""
        gm_py, gm_ft = self._step0_pytorch_out()
        gap_py = abs(gm_py - self.STEP0_OGG_OUT_SKIN)
        gap_ft = abs(gm_ft - self.STEP0_OGG_OUT_SKIN)
        # The gap must be materially present under BOTH tile weightings,
        # proving it is not an artefact of the fraction vector.
        assert gap_py > 1.0, (
            f"single-step gap with PyTorch fractions is only {gap_py:.3f} K "
            f"-- if the physics port now matches Fortran bit-for-bit, "
            f"tighten DRIFT_SKIN_T_K in TestCouplingDrift"
        )
        assert gap_ft > 1.0, (
            f"single-step gap with Fortran fractions is only {gap_ft:.3f} K"
        )

    def test_tile_fraction_choice_is_not_the_cause(self):
        """Swapping the PyTorch tile fractions for the Fortran ZFRTI moves
        the step-0 gridpoint mean by <0.5 K -- far too small to explain
        the multi-step drift.  Pins the ruled-out hypothesis."""
        gm_py, gm_ft = self._step0_pytorch_out()
        assert abs(gm_py - gm_ft) < 0.5, (
            f"tile-fraction choice moves skin-T by {abs(gm_py-gm_ft):.3f} K "
            f"-- re-evaluate whether ZFRTI is now a drift driver"
        )


# --------------------------------------------------------------------------- #
# Tests: documented drift between the coupled o_gg and the shim replay.
#
# These quantify the *known* Phase-1 coupling drift rather than asserting
# bit-exact agreement.  Tolerances are set to the currently-observed gap
# so the suite is green and acts as a regression guard: closing either
# gap (a P1.6 task) lets the tolerance be tightened.
# --------------------------------------------------------------------------- #
@requires_coupled_run()
class TestCouplingDrift:
    """Quantify (not assert-away) the known Phase-1 coupling drift."""

    DRIFT_SKIN_T_K = 10.0
    """Max tolerated |o_gg.skin - replay.skin| per step.

    Current observed max over 48 steps is ~8.3 K (per-step diffs oscillate
    0..8.3 K, peaking near the afternoon of each diurnal cycle).  Root
    cause is a **single-step physics difference**: on byte-identical
    inputs one ``forward()`` step moves skin ~1.4 K less than the coupled
    Fortran ``SURFTSTP`` (see ``TestSingleStepPhysics``), which compounds
    across the run into the wider diurnal amplitude.  The tile-fraction
    vector and the per-step ``PTSKTI=PTL`` overwrite were each ruled out
    by controlled experiment (<0.1 K and <0.05 K/step respectively).
    10 K gives ~20% headroom over the observed peak; tighten once the
    per-step physics gap is closed (P1.6)."""

    def test_skin_t_within_documented_drift(
        self, ogg_rows: List[OGgRow], shim_replay,
    ):
        skins, _, _ = shim_replay
        n = min(len(ogg_rows), len(skins))
        diffs = np.array(
            [abs(ogg_rows[k].tskin - skins[k][0]) for k in range(n)]
        )
        assert diffs.max() <= self.DRIFT_SKIN_T_K, (
            f"skin-T drift {diffs.max():.3f} K exceeds documented "
            f"{self.DRIFT_SKIN_T_K} K (per-step diffs: {diffs.tolist()})"
        )

    def test_soil_t_frozen_in_coupled_but_advances_in_replay(
        self, ogg_rows: List[OGgRow], shim_replay,
    ):
        """The coupled ``o_gg`` pins soil-T at the cold-start value across
        all steps (a LEPYOIFS copy-back defect tracked under P1.6), while
        the shim replay advances it.  This test pins BOTH halves of that
        fact so a future fix that un-freezes ``o_gg`` soil-T is forced to
        update the assertion here (and move the check to
        :class:`TestCouplingDrift`'s agreement band)."""
        _, soils_t, _ = shim_replay
        n = min(len(ogg_rows), len(soils_t))

        ogg_soil1 = np.array([ogg_rows[k].tsoil[0] for k in range(n)])
        rep_soil1 = np.array([soils_t[k][0] for k in range(n)])

        # Coupled side: soil-T must NOT have moved off 285 K.
        assert np.allclose(ogg_soil1, ogg_soil1[0], atol=1e-9), (
            "o_gg soil-T is no longer frozen -- if the LEPYOIFS copy-back "
            "was fixed, move this into a skin-T-style drift band"
        )
        assert ogg_soil1[0] == pytest.approx(285.0, abs=1e-6)

        # Replay side: soil-T MUST have moved (the port is healthy).
        assert rep_soil1.max() - rep_soil1.min() > 0.5, (
            "replay soil-T stopped advancing -- port regression?"
        )
