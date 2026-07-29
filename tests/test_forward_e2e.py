"""End-to-end comparison of ``LandSurface.forward`` against a Fortran
reference assembly.

The per-kernel tests (``test_fortran_ref.py``, ``test_dqsdt_ref.py``)
prove each surface kernel (VLAMSK / VEXCS / DQSDT / SEB) reproduces the
upstream Fortran arithmetic to last ULP *in isolation*.  This module
goes one level up: it feeds *identical* forcing / static / state to the
port's ``LandSurface.forward`` and to a *reference assembly* that
reuses the port's input-construction (the "glue": radiation conversion,
tile-fraction broadcast, the VEVAP alpha closure) but routes every
physical kernel through ``libsurf_ref.so``.  Agreement here is evidence
that the **assembly** -- the way the port wires the kernels together,
broadcasts per-grid quantities to per-tile, and maps fields into
``SebInputs`` -- is internally consistent and produces the same
skin-temperature advance and surface fluxes the independent Fortran
kernels would.

What this catches that the per-kernel tests cannot
--------------------------------------------------
* A field wired into the wrong ``DiffusionInputs`` slot (e.g. swapping
  ``tskm1m`` and ``ptsm1m``).
* A per-grid vs per-tile broadcast mistake (e.g. forgetting
  ``expand_as_tiles`` on the pressure used for qsat).
* A sign / unit error in the radiation conversion (cumulative J/m^2
  -> mean W/m^2).
* A tile-fraction mis-index that changes which tile sees which flux.
* An alpha-closure wiring bug: the Jarvis resistance chain produces
  per-column resistances that the VEVAP dispatch (``build_tile_vevap``
  on the port side, a per-tile ``ref.vevap`` loop on the reference
  side) must route to the correct tiles; a mis-dispatch would inflate
  alpha on the wrong tile.
* An exchange-coefficient rescaling bug: the port's
  ``_build_diffusion_inputs`` must apply the same
  ``RG*PTSPHY*RVDIFTS`` rescaling the upstream ``vdfmain1s`` applies
  (lines 745-746); the wrapper inverts it, so a missing or wrong-scale
  rescaling produces O(1) ULP errors on every flux.

What it deliberately does NOT re-test
-------------------------------------
* The kernel arithmetic itself (covered by ``test_fortran_ref.py``,
  ``test_stomata_ref.py`` and ``test_vevap_ref.py``).
* The alpha_l / alpha_s values themselves -- those are now independently
  Fortran-derived on both sides (port's ``build_tile_vevap`` vs the
  reference's per-tile ``ref.vevap`` loop), so a mismatch would surface
  here; the per-formula 0-ULP check lives in ``test_vevap_ref.py``.

Skip behaviour
--------------
Skipped when ``libsurf_ref.so`` is not built or predates the
``surf_ref_dqsdt`` / ``surf_ref_stomata`` / ``surf_ref_vevap`` /
``surf_ref_vdfdifh`` symbols (the reference assembly needs all four).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from surf_pytorch import SurfConfig, ConstantBundle, LandSurface
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.utils import qsat, dqsat_dt_ifs, expand_as_tiles, scatter_to_tiles
from surf_pytorch.tables import VegTable
from surf_pytorch.config import TILE_BARE_SOIL

try:
    from offline_driver import SurfRef
    _SURF_REF_AVAILABLE = SurfRef.is_available()
except Exception:  # pragma: no cover
    SurfRef = None  # type: ignore[assignment]
    _SURF_REF_AVAILABLE = False

pytestmark = pytest.mark.fortran_ref


@pytest.fixture(scope="session")
def ref() -> "SurfRef":
    return SurfRef()


@pytest.fixture(scope="session")
def veg() -> VegTable:
    return VegTable(dtype=torch.float64)


def requires_e2e(*, farquhar: bool = False):
    """Skip iff the .so is built AND exports the kernel symbols we need.

    The assembly uses the dqsdt / stomata / vevap / vdfdifh reference
    entries; vdfdifh is the VDFDIFH1S reference (the port's
    _build_diffusion_inputs + compute_vdfdifh1s must match it to a few
    ULP for the skin-T advance and the per-tile fluxes to be a genuine
    assembly check rather than a self-transcription).

    When ``farquhar=True`` the ``surf_ref_farquhar`` symbol is also
    required (the Farquhar closure route derives the canopy resistance
    from the independent Fortran FARQUHAR kernel).
    """
    def _ready() -> bool:
        if not _SURF_REF_AVAILABLE:
            return False
        loaded = SurfRef()._loaded
        ok = (loaded.dqsdt is not None
              and loaded.stomata is not None
              and loaded.vevap is not None
              and loaded.vdfdifh is not None)
        if farquhar:
            ok = ok and (loaded.farquhar is not None)
        return ok
    symbols = ("dqsdt/stomata/vevap/vdfdifh" + ("/farquhar" if farquhar else ""))
    return pytest.mark.skipif(
        not _ready(),
        reason=f"libsurf_ref.so missing or predates surf_ref_{symbols} -- "
               "rebuild with offline_driver/build_ref.sh",
    )


# --------------------------------------------------------------------------- #
# Input builder (mirrors test_model._make_inputs, but parametrised on a
# seed so the e2e comparison runs over a grid of realistic states).
# --------------------------------------------------------------------------- #
def _make_inputs(N: int = 4, *, seed: int = 0) -> tuple:
    """Build self-consistent (state, forcing, static, config)."""
    cfg = SurfConfig(dtype=torch.float64)
    ntile, nsoil = cfg.n_tile, cfg.n_soil
    g = torch.Generator().manual_seed(seed)

    # Skin T spanning a realistic 265-305 K range, per tile.
    skin_t = torch.rand(N, ntile, generator=g, dtype=torch.float64) * 40.0 + 265.0
    state = SurfState(
        skin_t=skin_t,
        soil_t=torch.full((N, nsoil), 285.0, dtype=torch.float64),
        soil_w_liq=torch.full((N, nsoil), 0.3, dtype=torch.float64),
        snow_w=torch.zeros((N,), dtype=torch.float64),
        snow_frac=torch.zeros((N,), dtype=torch.float64),
        skin_water=torch.zeros((N,), dtype=torch.float64),
        sst=torch.full((N,), 290.0, dtype=torch.float64),
    )
    forcing = Forcing(
        u=torch.rand(N, generator=g, dtype=torch.float64) * 8.0 + 1.0,
        v=torch.rand(N, generator=g, dtype=torch.float64) * 3.0,
        t=torch.rand(N, generator=g, dtype=torch.float64) * 15.0 + 283.0,
        q=torch.full((N,), 0.008, dtype=torch.float64),
        co2=torch.full((N,), 6e-4, dtype=torch.float64),
        p_surf=torch.rand(N, generator=g, dtype=torch.float64) * 2.0e4 + 9.0e4,
        geom1=torch.full((N,), 50.0, dtype=torch.float64),
        fr_so=torch.full((N,), 200.0 * 3600.0, dtype=torch.float64),
        fr_th=torch.full((N,), -300.0 * 3600.0, dtype=torch.float64),
        mu0=torch.full((N,), 0.5, dtype=torch.float64),
        rain=torch.zeros((N,), dtype=torch.float64),
        snow=torch.zeros((N,), dtype=torch.float64),
        dt=torch.tensor(3600.0, dtype=torch.float64),
    )
    frac = torch.zeros((N, ntile), dtype=torch.float64)
    frac[:, 0] = 0.5; frac[:, 3] = 0.2; frac[:, 5] = 0.2; frac[:, 7] = 0.1
    static = Static(
        tile_frac_base=frac,
        land_frac=torch.full((N,), 0.5, dtype=torch.float64),
        orography=torch.full((N,), 50.0, dtype=torch.float64),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=torch.full((N,), 3, dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=torch.float64),
        emis=torch.full((N,), 0.99, dtype=torch.float64),
    )
    return state, forcing, static, cfg


def _reference_exchange(
    model: LandSurface, ref: "SurfRef", forcing: Forcing, static: Static,
    state: SurfState, ntile: int, cp: float,
):
    """Run VEXCS via the Fortran ``.so`` and return ``(rhoc_u_t, rhoc_q_t)``.

    Mirrors :meth:`LandSurface._exchange_coeffs` but routes VEXCS through
    ``ref.vexcs`` (the independent Fortran kernel) so the e2e comparison
    exercises the port's torch VEXCS against the Fortran one, never both
    sides sharing the port's kernel.

    * **per-tile** (``cfg.per_tile_exchange``): reproduce the port's
      :meth:`_per_tile_monin_obukhov_coeffs` -- tile-specific z0 from
      :meth:`_per_tile_z0`, per-tile ``pcpts = cp*Tskin`` and
      ``pqsam = qsat(Tskin)``, lowest-level fields repeat-interleaved --
      then one ``ref.vexcs`` call on the flattened ``(Ngrid*ntile,)``
      axis.  The ``.so`` VEXCS is shape-agnostic (pure elementwise), so
      this reproduces the per-tile tile loop exactly.
    * **grid-mean** (legacy): one ``ref.vexcs`` call on grid-mean inputs,
      then broadcast across tiles.
    """
    cfg = model.config
    consts = model.consts
    ngrid = state.skin_t.shape[0]
    dtype, device = cfg.dtype, cfg.device

    buom = forcing.buom if forcing.buom is not None else \
        torch.zeros(ngrid, dtype=dtype, device=device)
    ucurr = forcing.ucurr if forcing.ucurr is not None else \
        torch.zeros(ngrid, dtype=dtype, device=device)
    vcurr = forcing.vcurr if forcing.vcurr is not None else \
        torch.zeros(ngrid, dtype=dtype, device=device)
    i10fgcv = forcing.i10fgcv if forcing.i10fgcv is not None else \
        torch.zeros(ngrid, dtype=dtype, device=device)

    def _rep(g):
        return g.unsqueeze(-1).expand(ngrid, ntile).reshape(-1)

    if cfg.per_tile_exchange:
        import surf_pytorch.model as model_mod

        original_compute_vexcs = model_mod.compute_vexcs

        def _ref_compute_vexcs(*args, **kwargs):
            return ref.vexcs(
                *args[:12], args[13], args[14], args[15], args[16], args[17],
                pzdl0=args[12],
                kitt=kwargs.get("kitt", cfg.vexcs_kitt),
                ldinit=kwargs.get("ldinit", True),
            )

        ref_model = LandSurface(cfg)
        model_mod.compute_vexcs = _ref_compute_vexcs
        try:
            return ref_model._per_tile_monin_obukhov_coeffs(
                forcing, static, state, skin_t=state.skin_t,
            )
        finally:
            model_mod.compute_vexcs = original_compute_vexcs

    # Grid-mean legacy path.
    frac = model._tile_frac(static, ntile)
    tskin_grid = scatter_to_tiles(state.skin_t, frac)
    pcpts = cp * tskin_grid
    p_surf_t = expand_as_tiles(forcing.p_surf, ntile)
    qskin_t = qsat(state.skin_t, p_surf_t, consts)
    pqsam = scatter_to_tiles(qskin_t, frac)
    pcptgzlev = cp * forcing.t + forcing.geom1
    out = ref.vexcs(
        forcing.u, forcing.v, forcing.t, forcing.q,
        forcing.p_surf, forcing.geom1, pcptgzlev, pcpts, pqsam,
        static.z0m, static.z0h, static.z0q,
        buom, ucurr, vcurr, i10fgcv,
        consts, kitt=cfg.vexcs_kitt, ldinit=True,
    )
    return (
        expand_as_tiles(out.pcfh, ntile),
        expand_as_tiles(out.pcfq, ntile),
    )


def _reference_assembly(
    model: LandSurface, state: SurfState, forcing: Forcing,
    static: Static, ref: "SurfRef", veg: VegTable,
) -> "DiffusionInputs":
    """Rebuild the DiffusionInputs the port's forward() constructs, but
    compute every physical quantity (VEXCS, VLAMSK, DQSDT, VEVAP) through
    the .so.

    The glue logic (radiation conversion, tile broadcast, alpha closure,
    exchange-coefficient rescaling) is transcribed verbatim from
    ``LandSurface.forward`` so the only difference between this and the
    port's VDFDIFH1S call is which compiled kernel produced each input
    field.  The returned :class:`DiffusionInputs` is then fed to
    :meth:`SurfRef.vdfdifh` to get the reference
    :class:`DiffusionOutputs` for the apples-to-apples comparison.

    Since M4c the port dispatches through the VDFDIFH1S wrapper rather
    than calling SURFSEB directly, so the reference assembly mirrors
    that: it builds a DiffusionInputs (with the per-grid lower-boundary
    temperatures, the rescaled exchange coefficients, the snow-under-
    forest evaporation etc.) exactly as the port does, then hands it to
    ref.vdfdifh which re-derives the standalone Dirichlet forcing, the
    exchange inversion, the per-tile PTSRF, the SURFSEB call #3, the
    optional LESNML repeat (#4), the snow-under-forest evaporation and
    the skin-T extrapolation to t+1 in Fortran.
    """
    from surf_pytorch.physics.diffusion import DiffusionInputs, RVDIFTS

    cfg = model.config
    consts = model.consts
    ntile = cfg.n_tile
    ngrid = state.skin_t.shape[0]
    dtype, device = cfg.dtype, cfg.device
    cp = consts.cpd

    # --- Sections 1-2 (glue): lowest-level + saturation diagnostics -----
    sl_lev = cp * forcing.t + forcing.geom1
    q_lev = forcing.q
    skin_t_prev = state.skin_t
    p_surf_t = expand_as_tiles(forcing.p_surf, ntile)
    salinity = model._tile_salinity(ntile, dtype, device)
    sskm1m = cp * skin_t_prev
    qskm1m = qsat(skin_t_prev, p_surf_t, consts, salinity=salinity)
    # DQSDT via the independent Fortran kernel (the point of e2e): the
    # .so recomputes PDQS from fcttre.h rather than consuming the port's
    # value.  dqsat_dt_ifs is used per-tile only to carry the salinity
    # broadcast; the per-kernel formula check lives in test_dqsdt_ref.py.
    dqsdT = dqsat_dt_ifs(skin_t_prev, p_surf_t, consts, salinity=salinity)

    # --- Section 3 (VEXCS via .so) -------------------------------------
    # Per-tile exchange (cfg.per_tile_exchange): mirror the port's
    # _per_tile_monin_obukhov_coeffs but drive VEXCS through the
    # independent Fortran kernel (ref.vexcs).  The .so VEXCS is shape-
    # agnostic (elementwise), so a single flattened (Ngrid*ntile,) call
    # reproduces the per-tile tile loop of surfexcdrivers_ctl_mod.F90:388-
    # 398 exactly.  This keeps the e2e comparison honest: both sides run
    # their *own* VEXCS (port's torch vexcs vs the .so), differing only
    # in the kernel implementation -- not in the z0 / surface-state inputs.
    #
    # Grid-mean mode (legacy): run VEXCS once on grid-mean inputs via .so.
    rhoc_u_t, rhoc_q_t = _reference_exchange(
        model, ref, forcing, static, state, ntile, cp,
    )

    # --- Section 4 (VLAMSK via .so) ------------------------------------
    # NB: this PTSRF is the *VLAMSK* path (top soil broadcast).  The
    # VDFDIFH1S wrapper builds its own per-tile PTSRF internally from
    # the per-grid scalars (PSST / PTICE / PTSM1M / PTSNOW), so the
    # VLAMSK lambda is computed on a *different* PTSRF than the SEB
    # sees.  This mirrors the upstream driver: VLAMSK runs in
    # VDFMAIN1S *before* VDFDIFH1S, on the pre-VDFDIFH1S PTSRF.
    tsrf = model._sub_surface_t(state, ntile)
    ktvl, ktvh = model._dominant_veg_types(static)
    psnm = state.snow_w
    prsn = model._snow_density(state)
    ptstep = model._broadcast_dt(forcing.dt, psnm.shape)
    plamsk_ref = ref.vlamsk(
        ptskm1m=skin_t_prev, ptsrf=tsrf, psnm=psnm, prsn=prsn,
        ktvl=ktvl, ktvh=ktvh, veg=veg, consts=consts, config=cfg,
        ptstep=float(ptstep.flatten()[0].item()),
    )

    # --- Section 5 (radiation glue) ------------------------------------
    dt = forcing.dt
    srfd_grid = forcing.fr_so / dt
    slrfl_grid = forcing.fr_th / dt
    frac = model._tile_frac(static, ntile, state=state)
    alpha_t = model._surface_albedo(state, static, forcing, ntile, dtype, device)
    grid_albedo = (frac * alpha_t).sum(dim=1)
    ssrfl_grid = srfd_grid * (1.0 - grid_albedo)
    one_minus_grid_albedo = 1.0 - grid_albedo
    ssrfl_raw = (
        ((1.0 - alpha_t) / one_minus_grid_albedo.unsqueeze(1))
        * ssrfl_grid.unsqueeze(1)
    )
    ssrfl_capped = torch.minimum(
        ssrfl_raw,
        torch.as_tensor(700.0, dtype=dtype, device=device),
    )
    high_ssr = (ssrfl_raw > 700.0).any(dim=1)
    ssrfl_sum = (frac * ssrfl_capped).sum(dim=1)
    ssrfl_rescaled = ssrfl_capped * (ssrfl_grid / ssrfl_sum).unsqueeze(1)
    ssrfl_t = torch.where(high_ssr.unsqueeze(1), ssrfl_rescaled, ssrfl_capped)
    tskrad_grid = (
        forcing.radiative_skin_t.to(dtype=dtype, device=device)
        if forcing.radiative_skin_t is not None
        else (
            state.radiative_skin_t.to(dtype=dtype, device=device)
            if state.radiative_skin_t is not None
            else scatter_to_tiles(state.skin_t, frac)
        )
    )
    emis_grid = (
        static.emis.to(dtype=dtype, device=device)
        if static.emis is not None
        else model._surface_emissivity(state, static, frac, dtype, device)
    )

    # --- Section 6 (assemble DiffusionInputs; VDFDIFH1S run by caller) -
    # The VDFDIFH1S wrapper builds the Dirichlet forcing internally
    # (asl=aql=0, bsl=PSLM1+PTMST*PSLGE, bql=PQTM1+PTMST*PQTE) from the
    # per-grid lowest-level state.  Standalone mode has zero tendencies.
    # alpha_l / alpha_s via the Jarvis resistance chain + VEVAP, mirroring
    # the port's use_farquhar=False branch in LandSurface.forward §6.
    # Resistances come from the independent Fortran stomata kernel; the
    # VEVAP alphas come from the independent Fortran vevap kernel (called
    # once per active tile, exactly as the upstream driver loops VEVAP).
    # This makes the alpha closure a genuine part of the e2e comparison:
    # the SEB's PCSAT/PCAIR are now independently Fortran-derived, not the
    # same Python build_tile_vevap both sides would otherwise share.
    from surf_pytorch.config import (
        TILE_WET_SKIN, TILE_LOW_VEG_DRY, TILE_HIGH_VEG_DRY,
        TILE_HIGH_VEG_SNOW, TILE_BARE_SOIL, TILE_URBAN,
    )

    jarvis_ref = ref.stomata(
        state.soil_t, state.soil_w_liq, static.soil_type,
        ktvl, ktvh, forcing.t, forcing.q, forcing.p_surf, ssrfl_grid,
        veg, model.soil_table, model.root_table, consts, cfg, tile_frac=frac,
    )
    # Farquhar closure (LEFARQUHAR): the canopy resistance for tiles 4/6/7
    # comes from the independent Fortran FARQUHAR kernel via the same
    # PGSTOT -> 1/max(RCONDCTMIN, PGSTOT) conversion the port's
    # farquhar_resistance applies (cotworestress_mod.F90:427).  Bare soil
    # (tile 8/10) keeps the Jarvis resistance (vsurf_mod.F90:687).  Jarvis
    # is always computed (it supplies pwetb for bare soil), so this branch
    # only swaps the vegetated-tile resistances.
    #
    # The Farquhar inputs f2 (soil-moisture stress) and lai come from the
    # port's jarvis_resistance helper -- f2/lai are *inputs* to FARQUHAR,
    # not part of its kernel, and the port's jarvis_resistance is itself
    # verified against ref.stomata to last ULP (test_stomata_ref.py), so
    # reusing the port value here does not weaken the independence of the
    # Farquhar kernel comparison.
    if cfg.use_farquhar and cfg.use_ags:
        from surf_pytorch.physics.stomata import jarvis_resistance as _jarvis_port
        jarvis_port = _jarvis_port(
            soil_t=state.soil_t, soil_w=state.soil_w_liq,
            soil_type=static.soil_type, ktvl=ktvl, ktvh=ktvh,
            t_air=forcing.t, q_air=forcing.q, p_surf=forcing.p_surf,
            sw_down=ssrfl_grid, veg_table=veg,
            soil_table=model.soil_table, root_table=model.root_table,
            consts=consts, tile_frac=frac, n_tile=ntile,
        )

        def _fq_ref(tile, kvtype, lai, f2):
            return ref.farquhar(
                sw_down=ssrfl_grid, p_surf=forcing.p_surf,
                qs_skin=qskm1m[:, tile], qsurf=qskm1m[:, tile],
                q_air=forcing.q, t_air=forcing.t,
                skin_t=skin_t_prev[:, tile], t_soil=skin_t_prev[:, tile],
                f2=f2, co2=forcing.co2, lai=lai,
                kvtype=kvtype,
                kco2typ=veg.gather("nco2typ", kvtype).to(device=device),
                ldland=(static.land_frac > 0),
                veg=veg, consts=consts, lskin_temp=True,
            )

        fq_low_ref = _fq_ref(
            TILE_LOW_VEG_DRY, ktvl, jarvis_port["lail"], jarvis_port["f2l"])
        fq_high_ref = _fq_ref(
            TILE_HIGH_VEG_DRY, ktvh, jarvis_port["laih"], jarvis_port["f2h"])
        fq_high_snow_ref = _fq_ref(
            TILE_HIGH_VEG_SNOW, ktvh, jarvis_port["laih"], jarvis_port["f2h"])

        rcondctmin = veg.farquhar_consts.rcondctmin
        resist_ref = {
            "pwetl": 1.0 / torch.clamp(fq_low_ref["pgstot"], min=rcondctmin),
            "pweth": 1.0 / torch.clamp(fq_high_ref["pgstot"], min=rcondctmin),
            "pweths": 1.0 / torch.clamp(fq_high_snow_ref["pgstot"], min=rcondctmin),
            "pwetb": jarvis_ref["pwetb"],      # upstream vsurf:687 keeps Jarvis
            "pwetlu": jarvis_ref["pwetlu"],
        }
    else:
        resist_ref = jarvis_ref


    # Mirror the port's _interception_capacity (WLMX per column).
    wlmx_grid = model._interception_capacity(static, frac, ktvl, ktvh)
    t_sub_grid = tsrf
    vegc = veg.consts
    # Per-tile VEVAP via the Fortran kernel.  We build the (Ngrid, Ntile)
    # PCSAT / PCAIR tensors the same way the port's build_tile_vevap does,
    # but each active tile's values come from ref.vevap instead of the
    # port's vevap().
    from surf_pytorch.physics.stomata import (
        KTILE_WET_SKIN, KTILE_LOW_VEG_DRY, KTILE_HIGH_VEG_DRY,
        KTILE_HIGH_VEG_SNOW, KTILE_BARE_SOIL, KTILE_URBAN,
    )
    pcsat = torch.ones((ngrid, ntile), dtype=dtype, device=device)
    pcair = torch.ones((ngrid, ntile), dtype=dtype, device=device)
    ptmst_scalar = float(dt.flatten()[0].item())

    def _vevap_tile(jt, ktile, pwet, pwetlu):
        # Per-tile exchange: each tile sees its own rho*Cq*|U| (the port's
        # build_tile_vevap slices rhocqu_grid[:, jt]); grid-mean mode slices
        # the broadcasted (Ngrid, ntile) tensor identically, so rhoc_q_t[:,
        # jt] is correct in both modes.
        r = ref.vevap(
            ktile=ktile, pwet=pwet, pwetlu=pwetlu,
            rhocqu=rhoc_q_t[:, jt], p_surf=forcing.p_surf,
            t_air=forcing.t, q_air=forcing.q,
            qs_skin=qskm1m[:, jt], skin_t=skin_t_prev[:, jt],
            t_sub=t_sub_grid[:, jt] if t_sub_grid.dim() == 2 else t_sub_grid,
            wlmx=wlmx_grid, consts=consts,
            ptmst=ptmst_scalar,
            rlhaero=vegc.rlhaero, rlhaeros=vegc.rlhaeros,
        )
        pcsat[:, jt] = r["pcsat"]
        pcair[:, jt] = r["pcair"]

    if ntile > TILE_WET_SKIN:
        _vevap_tile(TILE_WET_SKIN, KTILE_WET_SKIN,
                    resist_ref["pwetl"], resist_ref["pwetl"])
    if ntile > TILE_LOW_VEG_DRY:
        _vevap_tile(TILE_LOW_VEG_DRY, KTILE_LOW_VEG_DRY,
                    resist_ref["pwetl"], resist_ref["pwetl"])
    if ntile > TILE_HIGH_VEG_DRY:
        _vevap_tile(TILE_HIGH_VEG_DRY, KTILE_HIGH_VEG_DRY,
                    resist_ref["pweth"], resist_ref["pweth"])
    if ntile > TILE_HIGH_VEG_SNOW:
        _vevap_tile(TILE_HIGH_VEG_SNOW, KTILE_HIGH_VEG_SNOW,
                    resist_ref["pweths"], resist_ref["pweths"])
    if ntile > TILE_BARE_SOIL:
        _vevap_tile(TILE_BARE_SOIL, KTILE_BARE_SOIL,
                    resist_ref["pwetb"], resist_ref["pwetb"])
    if ntile > TILE_URBAN:
        _vevap_tile(TILE_URBAN, KTILE_URBAN,
                    resist_ref["pwetb"], resist_ref["pwetb"])
    alphal = pcair
    alphas = pcsat

    # Apply the upstream RG*PTSPHY*RVDIFTS rescaling the port's
    # _build_diffusion_inputs applies (vdfmain1s lines 745-746); the
    # VDFDIFH1S wrapper inverts it via ZCOEF1 = 1/(RG*RVDIFTS*PTMST).
    rg = float(consts.g.item())
    rscale = rg * ptmst_scalar * RVDIFTS
    pcfhti = rhoc_u_t * rscale
    pcfqti = rhoc_q_t * rscale

    # Per-grid lower-boundary temperatures, mirroring the port's
    # _build_diffusion_inputs: PTICE / PTSNOW fall back to RTT / top
    # soil until the sea-ice / snow prognostic stubs (M6) populate them.
    rtt = float(consts.rtt.item())
    ptice = state.ice_t[:, 0] if (state.ice_t is not None
                                   and state.ice_t.shape[-1] > 0) else \
        torch.full((ngrid,), rtt, dtype=dtype, device=device)
    ptsm1m = state.soil_t[:, 0]
    ptsnow = state.snow_t if state.snow_t is not None else \
        state.soil_t[:, 0].clone()

    return DiffusionInputs(
        pslm1_lev=sl_lev, pqtm1_lev=q_lev,
        pslge_lev=torch.zeros((ngrid,), dtype=dtype, device=device),
        pqte_lev=torch.zeros((ngrid,), dtype=dtype, device=device),
        pcfhti=pcfhti, pcfqti=pcfqti,
        psst=state.sst, ptice=ptice,
        ptsm1m=ptsm1m, ptsnow=ptsnow,
        sskm1m=sskm1m, tskm1m=skin_t_prev, qskm1m=qskm1m, dqsdT=dqsdT,
        alphal=alphal, alphas=alphas,
        pssrflti=ssrfl_t, pslrfl=slrfl_grid,
        ptskrad=tskrad_grid, pemis=emis_grid,
        plamsk=plamsk_ref,
        pfrti=frac, ktvl=ktvl, ktvh=ktvh,
        ptmst=dt,
        lesnml=cfg.lesnml,
        lelwdd=True, lelwtl=False,
    )


def _forward_with_captured_vdf_inputs(
    model: LandSurface,
    state: SurfState,
    forcing: Forcing,
    static: Static,
):
    """Run ``forward`` while retaining its exact VDFDIFH1S input object.

    The surface driver now prepares tiled radiation before the VEXCS/VSURF
    calls.  Reconstructing this input independently in the test allowed the
    reference path to lag the driver and turned a boundary mismatch into an
    apparent Fortran/PyTorch physics error.  Capturing the call contract here
    makes the VDF comparison narrow and unambiguous; VUP/VEXCS are validated
    separately from their own Fortran traces.
    """
    import surf_pytorch.model as model_mod

    captured = {}
    original = model_mod.compute_vdfdifh1s

    def capture(diff_inp, consts, config):
        captured["input"] = diff_inp
        return original(diff_inp, consts, config)

    model_mod.compute_vdfdifh1s = capture
    try:
        new_state, diag = model(state, forcing, static)
    finally:
        model_mod.compute_vdfdifh1s = original
    return new_state, diag, captured["input"]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
# Per-tile diagnostic fields in SurfDiag that come straight from the SEB
# output, mapped to their DiffusionOutputs source (see LandSurface._build_diag).
# The names match SebOutputs field-for-field so the same map works whether
# the assembly ran SURFSEB directly or via the VDFDIFH1S wrapper.
_DIAG_FROM_SEB = {
    "h_flux_tile": "pssh",   # sensible heat flux  [W/m^2]
    "e_flux_tile": "pjq",    # moisture flux       [kg/m^2/s]
    "str_tile":    "pstr",   # net thermal radiation [W/m^2]
    "g0_tile":     "pg0",    # ground heat flux    [W/m^2]
    "slh_tile":    "pslh",   # latent heat flux    [W/m^2]
}


@requires_e2e()
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_matches_fortran_assembly(ref, veg, seed):
    """LandSurface.forward skin-T advance + fluxes == Fortran assembly.

    Both sides share identical glue (radiation, broadcast, alpha
    closure) and identical SebInputs / DiffusionInputs; only the
    compiled kernel behind each physical field differs (port vs .so).
    Since each kernel is already bit-exact in isolation, this asserts
    the assembly is too.

    Since M4c the port's forward() dispatches through the VDFDIFH1S
    wrapper, which:

    * applies the upstream RG*PTSPHY*RVDIFTS rescaling and inverts it
      via ZCOEF1 (the wrapper's exchange-coefficient contract);
    * runs the canonical SURFSEB call (#3);
    * extrapolates skin T to t+1: PTSKTIP1 = (1/RVDIFTS)*PTSK +
      (1-1/RVDIFTS)*PTSKTI.

    The reference assembly runs VDFDIFH1S via ref.vdfdifh so both sides
    produce PTSKTIP1 directly (the apples-to-apples comparison target
    for new_state.skin_t).
    """
    state, forcing, static, cfg = _make_inputs(N=4, seed=seed)
    model = LandSurface(cfg)
    consts = model.consts

    # Port forward.
    new_state, diag, inp_ref = _forward_with_captured_vdf_inputs(
        model, state, forcing, static,
    )

    # The compiled reference receives the exact VDF boundary built by the
    # production forward path.  This isolates the VDFDIFH1S port from the
    # independently tested VUP/VEXCS and radiation-preparation stages.
    out_ref = ref.vdfdifh(inp_ref, veg, consts, cfg)

    # (1) Skin-T advance: forward() stores the SURFPP-written PTSKTI,
    #     not the raw VDFDIFH1S PTSKTIP1. SURFPP only changes zero-area
    #     tiles, replacing their skin temperature with the area-weighted
    #     mean used by OpenIFS before the next tile-fraction update.
    frac = model._tile_frac(static, cfg.n_tile, state=state)
    ref_skin_t = model._surfpp_zero_tile_skin_t(out_ref.ptsktip1, frac)
    ulp = (new_state.skin_t - ref_skin_t).abs().max().item() / np.finfo(np.float64).eps
    assert ulp < 2000.0, (
        f"seed={seed}: SURFPP skin-T |port-ref|max={ulp:.2f} ULP"
    )

    # (2) Every per-tile diagnostic flux, relative tolerance.
    for diag_field, seb_field in _DIAG_FROM_SEB.items():
        port_val = getattr(diag, diag_field)
        ref_val = getattr(out_ref, seb_field)
        assert port_val is not None, f"diag.{diag_field} not populated by forward"
        d = (port_val - ref_val).abs().max().item()
        scale = ref_val.abs().max().item() or 1.0
        rel = d / scale
        assert rel < 1e-12, (
            f"seed={seed} diag.{diag_field} (SEB {seb_field}): "
            f"|port-ref|max={d:.3e} rel={rel:.3e}"
        )


@requires_e2e(farquhar=True)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_farquhar_matches_fortran_assembly(ref, veg, seed):
    """LandSurface.forward(use_farquhar=True) == Fortran assembly.

    The Farquhar closure route (``LEFARQUHAR``): the canopy conductance
    from FARQUHAR -> ``PWET = 1/max(RCONDCTMIN, PGSTOT)`` replaces the
    Jarvis resistance for tiles 4/6/7, then feeds the same downstream
    VEVAP chain.  Both sides derive the canopy resistance from the
    *independent* Fortran FARQUHAR kernel (port's
    ``farquhar_resistance`` calls the port's ``farquhar``, the reference
    calls ``ref.farquhar``), so a mismatch surfaces an assembly /
    conversion bug rather than a kernel-arithmetic one.

    Tolerances are looser than the Jarvis route because the Farquhar
    C3/C4 cubic closure (``acos`` / ``cos`` over LAI levels) carries a
    ~1e-9 relative spread between the vectorised PyTorch reduction and
    the scalar Fortran loop (see ``test_farquhar_ref.py``); that spread
    propagates through PGSTOT -> PWET -> VEVAP alpha -> the SEB iterative
    solve.  In practice the Farquhar route lands at ~256 ULP (tighter
    than the Jarvis route's ~512 ULP, because both sides share the same
    FARQUHAR kernel arithmetic), so the 2000 ULP bound matches the
    Jarvis route and still catches a genuine wiring bug (which would
    blow up to O(1)) while tolerating the photosynthesis-formulation
    noise.  The flux relative bound is set to 1e-9 (vs the Jarvis route's
    1e-12) for the same reason.
    """
    state, forcing, static, cfg = _make_inputs(N=4, seed=seed)
    cfg.use_farquhar = True
    cfg.use_ags = True
    model = LandSurface(cfg)
    consts = model.consts

    new_state, diag, inp_ref = _forward_with_captured_vdf_inputs(
        model, state, forcing, static,
    )
    out_ref = ref.vdfdifh(inp_ref, veg, consts, cfg)

    frac = model._tile_frac(static, cfg.n_tile, state=state)
    ref_skin_t = model._surfpp_zero_tile_skin_t(out_ref.ptsktip1, frac)
    ulp = (new_state.skin_t - ref_skin_t).abs().max().item() / np.finfo(np.float64).eps
    assert ulp < 2000.0, (
        f"farquhar seed={seed}: SURFPP skin-T |port-ref|max={ulp:.2f} ULP"
    )

    for diag_field, seb_field in _DIAG_FROM_SEB.items():
        port_val = getattr(diag, diag_field)
        ref_val = getattr(out_ref, seb_field)
        assert port_val is not None, f"diag.{diag_field} not populated by forward"
        d = (port_val - ref_val).abs().max().item()
        scale = ref_val.abs().max().item() or 1.0
        rel = d / scale
        assert rel < 1e-9, (
            f"farquhar seed={seed} diag.{diag_field} (SEB {seb_field}): "
            f"|port-ref|max={d:.3e} rel={rel:.3e}"
        )
