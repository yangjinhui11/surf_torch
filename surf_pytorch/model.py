"""Top-level ``nn.Module`` for the OpenIFS land-surface package.

This is the public entry point: ``surf_pytorch.LandSurface`` is an
``nn.Module`` whose ``forward`` advances the surface state by one
coupling interval, given atmospheric forcing.  It owns the four
lookup tables (soil / veg / roots / ritbl) and the constant bundle,
and dispatches to the physics sub-modules:

* :mod:`surf_pytorch.physics.seb`     - surface energy balance (PM)   [M3]
* :mod:`surf_pytorch.physics.diffusion`- implicit vertical diffusion  [M4.2]
* :mod:`surf_pytorch.physics.exchange`- Monin-Obukhov exchange coeffs [M4.1]
* :mod:`surf_pytorch.physics.soil`    - HTESSEL soil temp/moisture    [M5]
* :mod:`surf_pytorch.physics.canopy`  - skin-reservoir interception   [M6]
* :mod:`surf_pytorch.physics.snow`    - single-layer snow             [M6]

Currently implemented:

* **M3** -- the Penman-Monteith surface energy balance
  (:func:`surf_pytorch.physics.seb.surf_seb`) plus its
  skin-conductivity pre-compute
  (:func:`surf_pytorch.physics.vlamsk.vlamsk`).  ``forward`` runs the
  SEB step and returns the updated skin temperature, the turbulent fluxes
  and grid-mean lowest-level (Sl, Ql).
* **M4.1** -- the Beljaars Monin-Obukhov surface-layer exchange
  coefficients (surf_pytorch.physics.exchange.vexcs) with the
  tiled VUPDZ0 roughness and stability closure.
* **M4.2** -- the offline-driver VDFDIFH1S SURFSEB wrapper
  (:func:`surf_pytorch.physics.diffusion.vdfdifh1s`).  ``forward``
  dispatches through this wrapper so the snow-under-forest evaporation,
  the optional LESNML snow-melt repeat (call #4) and the skin-T
  extrapolation to ``t+1`` all flow through automatically.  The
  exchange coefficients are rescaled by ``RG*PTSPHY*RVDIFTS`` before
  being passed in, mirroring the upstream ``vdfmain1s`` contract; the
  wrapper inverts that rescaling.
* **M5d** -- the HTESSEL soil-temperature and soil-moisture prognostics
  (:mod:`surf_pytorch.physics.soil`) advance one step each, fed by the
  per-tile turbulent fluxes, the snow-bottom heat flux and the canopy
  interception mismatch.
* **M6 (canopy)** -- the skin-reservoir / canopy interception update
  (:func:`surf_pytorch.physics.canopy.update_canopy`, ``SRFWL``)
  advances the skin-reservoir water content and produces the rainfall
  throughfall plus the per-tile interception-layer evaporation mismatch
  that the soil-moisture solve consumes.
* **M6c (snow)** -- the single-layer snow scheme
  (:func:`surf_pytorch.physics.snow.compute_snow`, ``SRFSN_LWIMP``)
  advances the snow prognostic and produces the snow-bottom heat flux.

Still stubbed (drop-in placeholders, gated behind SurfConfig flags):

* snow prognostics -- the single-layer snow scheme
  (:mod:`surf_pytorch.physics.snow`) is fully ported (M6c); the richer
  multi-layer ``SRFSN_WEBALS`` remains deferred behind ``LESLML``.
* FLake lake model (:mod:`surf_pytorch.physics.flake`) -- identity
  update matching the upstream ``LEFLAKE = .FALSE.`` branch; the real
  ``FLAKE_DRIVER`` body is a future drop-in via
  :meth:`_update_lake`.
* KPP ocean mixed-layer (:mod:`surf_pytorch.physics.ocean_ml`) --
  identity update matching the upstream ``LEOCSA = .FALSE.`` default;
  the real ``OCEAN_ML_DRIVER`` body is a future drop-in via
  :meth:`_update_ocean_ml`.
* Farquhar photosynthesis (:mod:`surf_pytorch.physics.farquhar`) --
  full Farquhar + Ball-Berry conductance port.  ``use_farquhar=True``
  computes the carbon diagnostics (LEFARQUHAR); ``use_ags=True`` also
  feeds its canopy conductance into VEVAP (LEAGS).  The two switches are
  independent in the upstream configuration.
"""

from __future__ import annotations

from typing import Optional, Tuple
from dataclasses import replace

import torch
from torch import nn

from .config import (
    SurfConfig, N_BASE_TILES,
    TILE_WATER, TILE_ICE, TILE_WET_SKIN,
    TILE_LOW_VEG_DRY, TILE_LOW_VEG_SNOW,
    TILE_HIGH_VEG_DRY, TILE_HIGH_VEG_SNOW, TILE_BARE_SOIL,
    TILE_URBAN,
)
from .constants import ConstantBundle, PhysicalConstants
from .state import Forcing, Static, SurfState, SurfDiag
from .tables import SoilTable, VegTable, RootTable, RitblTable
from .utils import qsat, dqsat_dt_ifs, expand_as_tiles, scatter_to_tiles
from .physics.seb import SebInputs, SebOutputs, surf_seb  # noqa: F401  (re-exported for tests / downstream consumers)
from .physics.vlamsk import vlamsk as compute_vlamsk
from .physics.exchange import (
    vexcs as compute_vexcs, vexcss as compute_vexcss, ExchangeOutputs, RKAP, RPARZI, REPDU2, psihu, psihs,
)
from .physics.diffusion import (
    DiffusionInputs, DiffusionOutputs, vdfdifh1s as compute_vdfdifh1s,
    RVDIFTS,
)
from .physics.offline_vdf import vdfdifm1s_surface_stress
from .physics.stomata import (
    jarvis_resistance as compute_jarvis_resistance,
    build_tile_alpha as compute_tile_alpha,
    build_tile_vevap as compute_tile_vevap,
    farquhar_resistance as compute_farquhar_resistance,
)
from .physics.soil import (
    SoilTempInputs, SoilTempOutputs,
    compute_soil_temperature as port_soilt,
    SoilMoistInputs, SoilMoistOutputs,
    compute_soil_moisture as port_soilw,
)
from .physics.snow import (
    SnowInputs, SnowOutputs,
    compute_snow as port_snow,
)
from .physics.snow_density import (
    SnowDensityInputs, SnowDensityOutputs,
    compute_snow_density as port_snow_density,
)
from .physics.snow_albedo import (
    SnowAlbedoInputs, SnowAlbedoOutputs,
    compute_snow_albedo as port_snow_albedo,
)
from .physics.canalb import (
    CanalbInputs, compute_canalb as port_canalb,
)
from .physics.canopy import (
    CanopyInputs, CanopyOutputs,
    update_canopy as port_canopy,
)
# M6 stubs -- identity-update placeholders for the FLake lake model, the
# KPP ocean mixed-layer model and the Farquhar photosynthesis / stomatal-
# conductance model.  Each is gated behind a SurfConfig flag (use_flake /
# use_kpp / use_farquhar) and reproduces the offline-driver default until
# the real transcription lands; see surf_pytorch/physics/{flake,ocean_ml,
# farquhar}.py for the drop-in contract.
from .physics.flake import (
    LakeInputs, LakeOutputs, flake_driver as port_flake,
)
from .physics.ocean_ml import (
    OceanMlInputs, OceanMlOutputs, ocean_ml_driver as port_ocean_ml,
)
from .physics.farquhar import (
    FarquharInputs, FarquharOutputs, farquhar as port_farquhar,
)


class LandSurface(nn.Module):
    """OpenIFS land-surface package as a batched ``nn.Module``.

    The module is *pure* (no global state): every call consumes a
    :class:`SurfState` and a :class:`Forcing`, and returns the updated
    state plus a :class:`SurfDiag`.  All tensors are batched along the
    leading ``Ngrid`` axis so the model is drop-in compatible with ML
    training loops.

    Parameters
    ----------
    config:
        :class:`SurfConfig` -- dimensions, feature flags, dtype.
    consts:
        Optional :class:`PhysicalConstants` override (default: IFS
        canonical values with the offline-driver ``YOETHF`` overrides).

    Notes
    -----
    The buffers (LUTs) are non-learnable by default.  Pass
    ``learnable_soil=("rvclay", ...)`` etc. via the config to register
    selected LUT columns as ``nn.Parameter`` instead.
    """

    def __init__(
        self,
        config: SurfConfig | None = None,
        *,
        consts: PhysicalConstants | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SurfConfig()
        self.consts = ConstantBundle(
            consts or PhysicalConstants(),
            dtype=self.config.dtype,
            device=self.config.device,
        )

        # Tables.  Each one is itself an nn.Module so its buffers /
        # parameters register with self automatically.
        self.soil_table = SoilTable(
            dtype=self.config.dtype,
            device=self.config.device,
            levgen=self.config.levgen,
            learnable=dict.fromkeys(self.config.learnable_soil, True),
        )
        self.veg_table = VegTable(
            dtype=self.config.dtype,
            device=self.config.device,
            learnable=dict.fromkeys(self.config.learnable_veg, True),
        )
        self.root_table = RootTable(
            dtype=self.config.dtype,
            device=self.config.device,
        )
        self.ritbl_table = RitblTable(
            dtype=self.config.dtype,
            device=self.config.device,
        )

        # Stateful SURFEXCDRIVER warm-start fields. OpenIFS threads the
        # previous tiled stress and turbulent fluxes through VUPDZ0 before
        # each VEXCS call. They are initialised lazily so one-off kernel
        # tests that call helper methods directly keep their cold-start path.
        self._surf_step_count = 0
        self._last_pustrti = None
        self._last_pvstrti = None
        self._last_pahfsti = None
        self._last_pevapti = None
        self._last_pzdl = None
        self._last_pcfm_tile = None
        self._last_pcfh_tile = None
        self._last_pcfq_tile = None
        self._last_vdf_pstr_tile = None
        self._last_vdf_ptsk_tile = None
        self._last_vdf_ptsktip1_tile = None
        self._last_vdf_tskm1m_tile = None
        self._last_vdf_pslrfl = None
        self._last_vdf_pemis = None
        self._last_vdf_ptskrad = None
        # Exact VEXCS arguments after VSURF and roughness blending.  These
        # diagnostic caches do not participate in the time integration.
        self._last_vexcs_pcpts_tile = None
        self._last_vexcs_pqsam_tile = None
        self._last_vexcs_z0m_tile = None
        self._last_vexcs_z0h_tile = None
        self._last_vexcs_z0q_tile = None
        self._last_vexcs_pzdl_input_tile = None
        self._last_vexcs_pbuom_input_tile = None
        self._last_vup_pzdl0 = None
        self._last_vup_pbuom0 = None
        # The final SRFSN_LWIMP call is retained solely for controlled
        # Fortran/PyTorch trace comparisons.  These references are never
        # read by the physical time integration.
        self._last_snow_inputs = None
        self._last_snow_outputs = None
        self._last_vevap_pcsnw = None
        self._last_vevap_pcfq = None
        self._last_jarvis = None
        self._last_vup_praqti = None

        # Physics sub-modules are added in M3-M6.  The SEB step is a
        # pure function (no learnable parameters), so we don't wrap it
        # in an nn.Module -- it's called directly from forward().

    # ----------------------------------------------------------------- #
    # Forward: advance one coupling interval
    # ----------------------------------------------------------------- #
    def _apply(self, fn):
        """Hook into nn.Module device/dtype propagation.

        ConstantBundle is a plain dataclass (not an nn.Module), so its
        tensors are invisible to the parent _apply().  We explicitly
        propagate the conversion function ``fn`` across every constant
        tensor so that model.to(device) / model.float() / model.double()
        moves consts alongside the registered parameters and buffers.
        """
        ret = super()._apply(fn)
        for _cname in PhysicalConstants.__dataclass_fields__:
            t = getattr(self.consts, _cname)
            if isinstance(t, torch.Tensor):
                setattr(self.consts, _cname, fn(t))
        return ret

    def forward(
        self,
        state: SurfState,
        forcing: Forcing,
        static: Static,
    ) -> Tuple[SurfState, SurfDiag]:
        """Advance the surface state by one coupling interval.

        Runs the offline-driver VDFDIFH1S SURFSEB wrapper (M4.2), which
        closes the Penman-Monteith surface energy balance and returns
        the updated skin temperature together with the per-tile
        turbulent fluxes and the grid-mean lowest-level dry static
        energy / specific humidity.  The HTESSEL prognostics then
        advance one step each, in upstream call order
        (``surftstp_ctl_mod.F90``): the skin-reservoir / canopy
        interception update (M6, SRFWL) computes the rainfall
        throughfall and the per-tile interception-layer evaporation
        mismatch; the single-layer snow scheme (M6c, SRFSN_LWIMP)
        advances the snow prognostic and produces the snow-bottom heat
        flux; then the soil-temperature (M5d-soilt, SRFTS) and
        soil-moisture (M5d-soilw, SRFWEXC_VG) solves advance the soil
        state, fed by the VDFDIFH1S per-tile fluxes, the snow-bottom
        heat flux and the canopy throughfall / interception mismatch.

        Exchange-coefficient contract
        -----------------------------
        :meth:`_exchange_coeffs` returns the *physical* ``rho*Ch*|U|``
        pair.  ``forward`` rescales these by ``RG*PTSPHY*RVDIFTS``
        before populating ``DiffusionInputs.pcfhti`` / ``pcfqti``,
        mirroring the upstream ``vdfmain1s`` (lines 745-746) contract;
        the VDFDIFH1S wrapper inverts that rescaling via
        ``ZCOEF1 = 1/(RG*RVDIFTS*PTMST)`` so the SEB receives the
        physical exchange coefficients.

        Boundary modes
        --------------
        vup_boundary_mode=offline executes the ported tiled VUPDZ0/VDF
        closure. vup_boundary_mode=external requires the complete tiled
        atmospheric boundary captured for controlled replay. No neutral or grid-mean
        exchange fallback is available through this interface.

        Remaining optional physics
        --------------------------
        * _sub_surface_t broadcasts the top soil-layer T to every tile;
          proper sea-ice / lake skin stubs come with the snow (M6) / lake
          modules.
        * :meth:`_snow_density` -- t-1 reader (PRSNM1M): prefers the
          prognostic ``state.snow_rho``, else the 300 kg/m^3 fresh-snow
          cold-start.  The *evolved* density is produced by
        :meth:`_update_snow_density` (``SRFSN_LWIMP`` section 3.1),
          now wired into forward() and written back to
          ``new_state.snow_rho``.
        * :meth:`_dominant_veg_types` -- returns the low / high veg-type
          indices per column.  Uses ``static.veg_type_low`` /
          ``veg_type_high`` when the explicit split is provided (the
          faithful SURFBC path), else falls back to ``static.veg_type``
          for both.
        * :meth:`_compute_tile_frac` / :meth:`_tile_frac` -- when
          ``Static`` carries the vegetation-cover inputs (``cvl_frac`` /
          ``cvh_frac``), the per-step tile fractions PFRTI are computed
          from the ``surfbc_ctl_mod.F90`` land-branch split (cover +
          snow-cover ZCVS + wet-skin ZCVW); otherwise the static
          ``tile_frac_base`` approximation is used (cold start).
        * Lowest-level coupling ``asl = aql = 0`` (Dirichlet boundary,
          no surface-flux feedback to the lowest model level within
          the step) -- matches the offline-driver setup
          (``vdfdifh1s.F90`` section 1.0-1.5).
        * Moisture-flux closure via the faithful VEVAP transcription
          (PCSAT -> alphas, PCAIR -> alphal).  Two canopy-resistance
          sources are supported (both feed the same downstream VEVAP
          chain): Jarvis by default, or Farquhar / Ball-Berry when both
          ``use_farquhar`` (LEFARQUHAR) and ``use_ags`` (LEAGS) are true.
          This preserves the OpenIFS case LEFARQUHAR=.TRUE., LEAGS=.FALSE.,
          where carbon diagnostics are computed but Jarvis still controls
          PWET*. Bare soil / urban resistance always uses the Jarvis formula
          (``vsurf_mod.F90:687``).

        Returns
        -------
        (next_state, diagnostics)
            ``next_state`` is a fresh :class:`SurfState` (the input is
            untouched) suitable for chaining into the next step.
            ``diagnostics`` holds the fluxes and 2 m / 10 m fields.
        """
        # VUPDZ0 carries preceding SURFPP fluxes along one integration
        # trajectory.  Load them from the input state so a module instance
        # can safely serve independent columns and state.like() feedback.
        self._surf_step_count = state.vupdz0_step
        self._last_pustrti = state.vupdz0_pustrti
        self._last_pvstrti = state.vupdz0_pvstrti
        self._last_pahfsti = state.vupdz0_pahfsti
        self._last_pevapti = state.vupdz0_pevapti
        self._last_pzdl = state.vupdz0_pzdl
        # In strict offline validation, the atmospheric diffusion driver can
        # provide the complete VUPDZ0 tile boundary for this surface step.
        # These values are authoritative over the trajectory cache.
        for forcing_name, cache_name in (
            ("vupdz0_pustrti", "_last_pustrti"),
            ("vupdz0_pvstrti", "_last_pvstrti"),
            ("vupdz0_pahfsti", "_last_pahfsti"),
            ("vupdz0_pevapti", "_last_pevapti"),
        ):
            value = getattr(forcing, forcing_name)
            if value is not None:
                setattr(self, cache_name, value)
        ntile = self.config.n_tile
        ngrid = state.skin_t.shape[0]
        dtype = self.config.dtype
        device = state.skin_t.device

        # ------------------------------------------------------------------ #
        # 1. Lowest-model-level quantities.
        # ------------------------------------------------------------------ #
        t_lev = forcing.t
        q_lev = forcing.q
        cp = self.consts.cpd

        # Dry static energy at the lowest model level: Sl = cp*T + g*z.
        # geom1 is the lowest-level geopotential (m^2/s^2).
        sl_lev = cp * t_lev + forcing.geom1

        # ------------------------------------------------------------------ #
        # 2. Skin-reservoir saturation diagnostics at t-1 (per tile).
        # ------------------------------------------------------------------ #
        # Saturation qsat / dqsatdT mirror vsurf_mod.F90 (278-284): the
        # ocean tile (TILE_WATER) carries the salinity factor ZSALIN=0.98
        # when LEOCSA is active, every other tile ZSALIN=1.0.  The dqsatdT
        # is the upstream PDQS form (see utils.dqsat_dt_ifs) -- NOT the
        # exact derivative -- so the SEB skin-T Jacobian matches the
        # Fortran driver to last-ULP.
        skin_t_prev = state.skin_t
        p_surf_t = expand_as_tiles(forcing.p_surf, ntile)
        salinity = self._tile_salinity(ntile, dtype, device)
        qskm1m = qsat(skin_t_prev, p_surf_t, self.consts, salinity=salinity)
        dqsdT = dqsat_dt_ifs(skin_t_prev, p_surf_t, self.consts, salinity=salinity)

        # SURFEXCDRIVER_CTL prepares tile radiation before VSURF/VEXCS.
        # VEXCS must therefore see the same tile-wise downward SW as the
        # subsequent surface-energy calculation, not the grid-mean forcing.
        frac = self._step_tile_frac(forcing, static, state, ntile)
        ktvl, ktvh = self._dominant_veg_types(static)
        alpha_t, ssrfl_t = self._tile_shortwave(
            forcing, static, state, frac, ntile, dtype, device,
        )

        # ------------------------------------------------------------------ #
        # 3. Turbulent exchange coefficients.  VEXCS is always evaluated
        #    per tile, matching SURFEXCDRIVER_CTL's tile loop.
        # ------------------------------------------------------------------ #
        rhoc_u, rhoc_q = self._exchange_coeffs(
            forcing, static, state, skin_t=skin_t_prev, frac=frac,
            ktvl=ktvl, ktvh=ktvh, tile_shortwave=ssrfl_t,
        )
        rhoc_u_t, rhoc_q_t = rhoc_u, rhoc_q

        # ------------------------------------------------------------------ #
        # 4. Skin-layer conductivity (VLAMSK) -- per-tile lambda_skin.
        # ------------------------------------------------------------------ #
        tsrf = self._sub_surface_t(state, ntile)   # (Ngrid, Ntile)
        ktvl, ktvh = self._dominant_veg_types(static)
        psnm = state.snow_w
        prsn = self._snow_density(state)
        ptstep = self._broadcast_dt(forcing.dt, psnm.shape)

        plamsk = compute_vlamsk(
            ptskm1m=skin_t_prev, ptsrf=tsrf,
            psnm=psnm, prsn=prsn, psntice=None,
            ktvl=ktvl, ktvh=ktvh,
            veg_table=self.veg_table, consts=self.consts, config=self.config,
            ptstep=ptstep,
            lesn09=self.config.lesn09,
        )

        # ------------------------------------------------------------------ #
        # 5. Radiation -- convert cumulative fluxes (J/m^2) to mean W/m^2.
        # ------------------------------------------------------------------ #
        dt = forcing.dt
        srfd_grid = forcing.fr_so / dt
        slrfl_grid = forcing.fr_th / dt
        # PSSRFL remains the tile-area-weighted net SW. Farquhar consumes
        # this grid diagnostic while VSURF consumes the tile boundary above.
        ssrfl_grid = (frac * ssrfl_t).sum(dim=1)
        # Surface albedo -- per-tile broadband alpha (dimensionless).
        # Returns all-zeros when neither the snow-albedo prognostic nor the
        # background albedo fields are set, so the absorbed SW is then
        # ``ssrfl_t`` unchanged (bit-identical to the build before the
        # albedo work). Uses the t-1 PASN (step-start field), consistent
        # with the t-1 skin T used for radiation above; the freshly-evolved
        # PASN is written to new_state.snow_alb in section 7 for next step.
        # Gridpoint skin T at radiation time (PTSKRAD).  It is an explicit
        # carried state: after snow/skin-water prognostics change PFRTI, a
        # re-average of PTSKTI with the *new* fractions does not reproduce
        # the PTL used by the previous radiation call.  External forcing
        # remains authoritative for coupled callers.
        tskrad_grid = (
            forcing.radiative_skin_t.to(dtype=dtype, device=device)
            if forcing.radiative_skin_t is not None
            else (
                state.radiative_skin_t.to(dtype=dtype, device=device)
                if state.radiative_skin_t is not None
                else scatter_to_tiles(state.skin_t, frac)
            )
        )

        # Use the provided static emissivity when available; the strict
        # reference-assembly tests supply the Fortran value directly.
        # Fall back to the diagnostic surface-emissivity reconstruction only
        # when no explicit emissivity was provided.
        emis_grid = (
            static.emis.to(dtype=dtype, device=device)
            if static.emis is not None
            else self._surface_emissivity(state, static, frac, dtype, device)
        )

        # ------------------------------------------------------------------ #
        # 6. Build DiffusionInputs and run the VDFDIFH1S wrapper (M4.2).
        # ------------------------------------------------------------------ #
        # Lowest-level forcing at KLEV -- the offline-driver setup
        # (vdfdifh1s.F90 section 1.0-1.5) builds the Dirichlet boundary
        # condition internally: asl = aql = 0, bsl = PSLM1+PTMST*PSLGE,
        # bql = PQTM1+PTMST*PQTE.  Standalone mode has zero tendencies
        # (the surface step does not update the lowest model level
        # within the coupling interval), so PSLGE/PQTE are zero here.

        # Moisture-flux closure: alpha_l (PCAIR -> potential-from-skin)
        # and alpha_s (PCSAT -> saturation-at-skin-T) per tile.  These are
        # the VEVAP coefficients (vevap_mod.F90); they are equal for the
        # vegetated dry tiles but differ for the wet-skin (interception
        # clamp) and snow-under-high-veg (ZRAS) branches.
        #
        # Jarvis controls VEVAP unless both LEFARQUHAR and LEAGS are active.
        # LEFARQUHAR alone still evaluates carbon diagnostics but, exactly as
        # in vsurf_mod.F90:694-717, does not overwrite PWETL/PWETH/PWETHS.
        #
        # Both moisture-flux closures share the downstream VEVAP chain. The
        # only difference is the canopy-resistance source for tiles 4/6/7.
        # Bare soil (PWETB) always uses the Jarvis formula even under
        # LEFARQUHAR (vsurf_mod.F90:687 -- COTWORESTRESS only computes the
        # vegetation canopy conductance), so the Jarvis chain runs in both
        # modes.  Wet-skin (KTILE=3) is driven by the interception clamp,
        # not by the canopy resistance, so it is unaffected.
        lai_low, lai_high = self._effective_lai(static, ktvl, ktvh)
        jarvis = compute_jarvis_resistance(
            soil_t=state.soil_t,
            soil_w=state.soil_w_liq,
            soil_type=static.soil_type,
            ktvl=ktvl, ktvh=ktvh,
            t_air=forcing.t, q_air=forcing.q, p_surf=forcing.p_surf,
            sw_down=self._tile_downward_shortwave(ssrfl_t, alpha_t),
            veg_table=self.veg_table,
            soil_table=self.soil_table,
            root_table=self.root_table,
            consts=self.consts,
            tile_frac=frac, n_tile=ntile,
            lai_low=lai_low,
            lai_high=lai_high,
        )
        self._last_jarvis = {name: value.detach() for name, value in jarvis.items()}
        if self.config.use_farquhar:
            # LEFARQUHAR branch (cotworestress_mod.F90:412-431): the
            # Farquhar canopy conductance replaces Jarvis PWETL/PWETH/
            # PWETHS for the vegetated tiles; PWETB is preserved.
            farquhar_resist = compute_farquhar_resistance(
                forcing=forcing, skin_t_prev=skin_t_prev, qskm1m=qskm1m,
                ssrfl_grid=ssrfl_grid, ktvl=ktvl, ktvh=ktvh,
                jarvis=jarvis, veg_table=self.veg_table, consts=self.consts,
                land_mask=static.land_frac > 0,
            )
            # Carbon-flux diagnostics from the Farquhar call (PGSTOT is the
            # raw canopy conductance before the RCONDCTMIN resistance floor;
            # PGPP/PRD/PAN are the kgC/m^2/s fluxes).  Surfaced in SurfDiag
            # so callers can read GPP / respiration without re-running the
            # photosynthesis kernel.  Photosynthesis is diagnostic here --
            # it does not feed back on the prognostic state in the current
            # single-step forward.
            _farquhar_diag = {
                "farquhar_pgstot": farquhar_resist["farquhar_pgstot"],
                "farquhar_pgpp":   farquhar_resist["farquhar_pgpp"],
                "farquhar_prd":    farquhar_resist["farquhar_prd"],
                "farquhar_pan":    farquhar_resist["farquhar_pan"],
            }
            # LEAGS is the independent OpenIFS switch that transfers ZWET
            # from COTWORESTRESS into the VEVAP PWET* arrays.
            resist = farquhar_resist if self.config.use_ags else jarvis
        else:
            resist = jarvis
            _farquhar_diag = None
        # PWLMX per column (surfbc_ctl_mod.F90 258-260):
        #   RWLMAX*(1 - CVL - CVH + CVL*LAIL + CVH*LAIH)
        # When Static carries the cover inputs (cvl_frac / cvh_frac /
        # lai_low / lai_high) the canonical formula is used directly;
        # otherwise the coverage is reconstructed from the tile fractions
        # and the dominant veg type's LAI stands in for both low / high
        # veg (cold-start approximation).  The KTILE=3 interception clamp
        # only fires when ZEP<0, so a coarse WLMX has negligible effect
        # on the SEB alphas outside that regime.
        wlmx_grid = self._interception_capacity(static, frac, ktvl, ktvh)
        # t_sub for VEVAP: SURFEXCDRIVER passes the per-tile ZTSA
        # (snow tiles get PTSN, snow-free land gets top-soil T).
        t_sub_grid = tsrf
        vegc = self.veg_table.consts
        vevap_out = compute_tile_vevap(
            pwetl=resist["pwetl"], pweth=resist["pweth"],
            pwetb=resist["pwetb"],
            rhocqu_grid=rhoc_q,
            p_surf=forcing.p_surf, t_air=forcing.t, q_air=forcing.q,
            skin_t=skin_t_prev, t_sub_grid=t_sub_grid,
            qs_skin=qskm1m, wlmx_grid=wlmx_grid,
            consts=self.consts, config=self.config,
            pwetlu=resist["pwetlu"], pweths=resist["pweths"],
            ptmst=dt,
            rlhaero=vegc.rlhaero, rlhaeros=vegc.rlhaeros,
        )
        # The final, prognostic VDFDIFH1S SURFSEB call uses VEVAP's
        # stressed PCPTSTI/PCAIRTI/PCSATTI fields.  The Fortran wrapper
        # makes earlier unstressed calls solely for potential-evaporation
        # diagnostics; those must not replace the forecast boundary.
        sskm1m = vevap_out["pcpts"]
        alphas = vevap_out["pcsat"]
        alphal = vevap_out["pcair"]
        pevapsnw = self._snow_under_forest_evap(
            forcing=forcing,
            state=state,
            rhoc_q_t=rhoc_q_t,
            pcsnw=vevap_out["pcsnw"],
            frac=frac,
            ntile=ntile,
            dtype=dtype,
            device=device,
        )
        self._last_vevap_pcsnw = vevap_out["pcsnw"].detach()
        self._last_vevap_pcfq = rhoc_q_t.detach()

        # Pack into DiffusionInputs and dispatch through the VDFDIFH1S
        # wrapper.  This routes the per-tile PTSRF build, the exchange-
        # coefficient inversion, the canonical SURFSEB call (#3), the
        # optional LESNML snow-melt repeat (#4), the snow-under-forest
        # evaporation and the skin-T extrapolation to t+1 through the
        # same code path the offline driver uses.
        diff_inp = self._build_diffusion_inputs(
            state=state, forcing=forcing, static=static,
            sl_lev=sl_lev, q_lev=q_lev,
            rhoc_u_t=rhoc_u_t, rhoc_q_t=rhoc_q_t,
            sskm1m=sskm1m, skin_t_prev=skin_t_prev,
            qskm1m=qskm1m, dqsdT=dqsdT,
            alphal=alphal, alphas=alphas,
            ssrfl_t=ssrfl_t, slrfl_grid=slrfl_grid,
            tskrad_grid=tskrad_grid, emis_grid=emis_grid,
            plamsk=plamsk, tsrf=tsrf, frac=frac, ktvl=ktvl, ktvh=ktvh,
            pevapsnw=pevapsnw,
            dt=dt, ntile=ntile, ngrid=ngrid, dtype=dtype, device=device,
        )
        out = compute_vdfdifh1s(diff_inp, self.consts, self.config)
        self._last_vdf_tskm1m_tile = diff_inp.tskm1m.detach()
        self._last_vdf_pslrfl = diff_inp.pslrfl.detach()
        self._last_vdf_pemis = diff_inp.pemis.detach()
        self._last_vdf_ptskrad = diff_inp.ptskrad.detach()

        vdf_boundary = (
            forcing.vdf_pfrti, forcing.vdf_pahfsti, forcing.vdf_pevapti,
            forcing.vdf_pslrflti, forcing.vdf_ptsktip1,
        )
        has_authoritative_vdf = all(value is not None for value in vdf_boundary)
        if any(value is not None for value in vdf_boundary):
            if not has_authoritative_vdf:
                raise ValueError(
                    "Authoritative VDF forcing requires pfrti, pahfsti, "
                    "pevapti, pslrflti, and ptsktip1 together"
                )
            frac = forcing.vdf_pfrti.to(dtype=dtype, device=device)
            out = replace(
                out,
                pssh=forcing.vdf_pahfsti.to(dtype=dtype, device=device),
                pjq=forcing.vdf_pevapti.to(dtype=dtype, device=device),
                pstr=forcing.vdf_pslrflti.to(dtype=dtype, device=device),
                ptsktip1=forcing.vdf_ptsktip1.to(dtype=dtype, device=device),
            )
        self._last_vdf_pstr_tile = out.pstr.detach()
        self._last_vdf_ptsk_tile = out.ptsk.detach()
        self._last_vdf_ptsktip1_tile = out.ptsktip1.detach()

        # SURFPP_CTL post-processes exact-zero-area tiles before the land
        # routines consume their fields.  This remains necessary when the
        # VDF boundary is authoritative: the capture contains the raw
        # VDFDIFH1S PTSKTIP1, whereas SURFPPS writes the grid-mean skin
        # temperature into an inactive tile's carried PTSKTI state.
        # Nonzero (including tiny) snow tiles remain untouched.
        if has_authoritative_vdf:
            surfpp_pssh, surfpp_pjq = self._surfpp_zero_tile_fluxes(
                out.pssh, out.pjq, frac,
            )
            surfpp_skin_t = self._surfpp_zero_tile_skin_t(out.ptsktip1, frac)
        else:
            surfpp_pssh, surfpp_pjq = self._surfpp_zero_tile_fluxes(out.pssh, out.pjq, frac)
            surfpp_skin_t = self._surfpp_zero_tile_skin_t(out.ptsktip1, frac)
        land_flux_out = replace(out, pssh=surfpp_pssh, pjq=surfpp_pjq)
        # SURFPP persists these post-processed fluxes for the next VUPDZ0
        # call. Its zero-area handling uses this VDF step's partition.
        self._last_pahfsti = land_flux_out.pssh.detach().clone()
        self._last_pevapti = land_flux_out.pjq.detach().clone()
        # SURFTSTP builds ZSLRFLTI from the coupled radiation field.  With
        # LELWTL disabled, it copies the grid net longwave flux to every
        # tile; the VDF/SEB tile pstr diagnostic is not that boundary.
        slrfl_t = (
            out.pstr if self.config.lelwtl
            else slrfl_grid.unsqueeze(1).expand_as(out.pstr)
        )

        # ------------------------------------------------------------------ #
        # 6. Skin-reservoir / canopy interception update (M6, SRFWL).
        # ------------------------------------------------------------------ #
        # Advances the skin-reservoir water content (PWL) by one coupling
        # interval and produces the rainfall throughfall (PTSFC/PTSFL) plus
        # the per-tile interception-layer evaporation mismatch (PEINTTI)
        # that the soil-moisture solve consumes.  Upstream call order
        # (surftstp_ctl_mod.F90:268 SRFWLS / 612 SRFWL): the canopy update
        # precedes the snow update, so this slot sits between VDFDIFH1S
        # and _update_snow.  Rainfall throughfall PRSFL <- forcing.rain/dt
        # (M2d); convective rain PRSFC stays zero (the port's
        # :class:`Forcing` does not split convective vs large-scale, so
        # all rain is treated as large-scale, matching the convention used
        # for the snowfall throughfall in _update_snow).  PEVAPSNW
        # (snow-under-forest evaporation) is zero on the cold start -- the
        # single-layer snow scheme does not thread it back yet.
        canopy_out = self._update_canopy(
            state=state, static=static, diff_out=land_flux_out, frac=frac,
            wlmx_grid=wlmx_grid, ktvl=ktvl, ktvh=ktvh,
            forcing=forcing, dt=dt,
            pevapsnw=pevapsnw,
            ngrid=ngrid, dtype=dtype, device=device,
        )

        # ------------------------------------------------------------------ #
        # 6a. HTESSEL single-layer snow update (M6c, SRFSN_LWIMP).
        # ------------------------------------------------------------------ #
        # Advances the snow prognostic (PTSNM1M, PSNM1M; PRSNM1M is carried
        # through unchanged into the snow solve) and produces the snow-
        # bottom heat flux PGSN that the soil-T solve consumes below.  The
        # snow density PRSNM1M is then advanced separately by the
        # SRFSN_LWIMP section-3.1 density-evolution port
        # (_update_snow_density) -- the ``-s`` single-layer variant the
        # port mirrors does not evolve density itself, so the density
        # block runs as a post-step.  Upstream call order
        # (surftstp_ctl_mod.F90): SRFSN_LWIMP precedes SRFTS, so this slot sits between
        # VDFDIFH1S and _update_soil_temperature.  Snowfall throughfall
        # PSSFL <- forcing.snow/dt (M2d); convective snowfall PSSFC stays
        # zero (the port's :class:`Forcing` does not split convective vs
        # large-scale, so all snow is treated as large-scale, matching the
        # same convention used for rainfall in _update_soil_moisture).
        snow_out = self._update_snow(
            state=state, static=static, diff_out=land_flux_out,
            pssrflti=ssrfl_t, pslrfl=slrfl_grid, pslrflti=slrfl_t, pfrti=frac,
            forcing=forcing,
            pevapsnw=pevapsnw,
            ptsfc=canopy_out.ptsfc, ptsfl=canopy_out.ptsfl,
            dt=dt, ngrid=ngrid, dtype=dtype, device=device,
        )
        # Advance the snow density (PRSN <- SRFSN_LWIMP section 3.1).
        new_snow_rho = self._update_snow_density(
            state=state, snow_out=snow_out, forcing=forcing, frac=frac,
            dt=dt, dtype=dtype, device=device,
        )
        # Advance the snow albedo (PASN <- SRFSN lines 340-361).  Returns
        # None when the snow-albedo prognostic is unset, leaving the
        # subsystem dormant (the radiation block above already saw alpha=0
        # in that case).
        new_snow_alb = self._update_snow_albedo(
            state=state, static=static, snow_out=snow_out, forcing=forcing,
            dt=dt, dtype=dtype, device=device,
        )
        if self.config.lesn09:
            # Match SURFBC_CTL exactly: 100 * SUM(SWE / max(rho, 1e-12))
            # * RQSNCR.  The grouping matters at vanishing snow mass.
            snow_rho_safe = torch.clamp(new_snow_rho, min=1.0e-12)
            snow_frac_raw = (
                100.0 * (snow_out.psstar / snow_rho_safe) * 0.1
            )
        else:
            snow_frac_raw = snow_out.psstar * self.soil_table.consts.rqsmcr
        # Preserve every positive residual cover fraction.  The Fortran
        # SURFBC/SURFPPS path tests exact zero; dropping sub-threshold cover
        # while retaining its snow mass resets that tile's skin temperature.
        new_snow_frac = torch.clamp(snow_frac_raw, 0.0, 1.0)
        # CPG1S writes the SURFTSTP snow result through the common GP
        # tendency update: ZSNSE1=(ZSN-old)/dt followed by
        # GP1=GP0+dt*ZGPE.  Preserve that operation order.  Its final-bit
        # residual is physically relevant because SURFBC/SURFPP branch on
        # an exact zero tile fraction at the next coupling step.
        snow_tendency = (snow_out.psstar - state.snow_w) * torch.reciprocal(dt)
        new_snow_w = state.snow_w + dt * snow_tendency

        # ------------------------------------------------------------------ #
        # 6b. HTESSEL soil-temperature update (M5d-soilt).
        # ------------------------------------------------------------------ #
        # Advances the soil-temperature prognostic by one coupling
        # interval via SRFRCGS (apparent heat capacity) + SRFTS (Thomas
        # coefficients) + SRFWDIFS (tridiag solve).  Inputs come from the
        # t-1 soil state (soil_t, soil_w_liq) and the per-tile turbulent
        # fluxes VDFDIFH1S just produced (PAHFSTI -> pssh, PEVAPTI ->
        # pjq, net SW/LW from the radiation terms).  The snow-bottom heat
        # flux PGSN flows from the SRFSN_LWIMP solve above.
        new_soil_t = self._update_soil_temperature(
            state=state, static=static, diff_out=land_flux_out,
            pssrflti=ssrfl_t, pslrfl=slrfl_grid, pslrflti=slrfl_t, pfrti=frac,
            pgsn=snow_out.pgsn,
            dt=dt, ngrid=ngrid, dtype=dtype, device=device,
        )

        # ------------------------------------------------------------------ #
        # 6c. HTESSEL soil-moisture update (M5d-soilw).
        # ------------------------------------------------------------------ #
        # Advances the soil-moisture prognostic by one coupling interval
        # via SRFWEXC_VG (Van Genuchten K/D + VIC runoff + root
        # extraction) + SRFWDIF (Thomas tridiag) + SRFWINC + SRFWNG
        # (clip / redistribute / runoff split).  Inputs come from the
        # t-1 soil state (soil_w_liq, soil_t) and the per-tile turbulent
        # fluxes VDFDIFH1S just produced (PEVAPTI -> pjq), plus the
        # precipitation throughfall.  The canopy update (_update_canopy)
        # supplies the large-scale throughfall PTSFL (rainfall after
        # interception) and the per-tile interception-layer evaporation
        # mismatch PEINTTI (the difference between the realised and the
        # raw PEVAPTI the soil must close against); convective throughfall
        # PTSFC stays zero because the offline forcing has no convective
        # split. Snowmelt and the snow-mass clipping mismatch are passed
        # through to SRFWEXC_VG, as in the Fortran single-layer scheme. PSDOR
        # (orography std-dev for the VIC runoff) defaults to flat terrain
        # (0 m) -- cold start.
        snow_rain_intercept = snow_out.ptsflin
        if snow_rain_intercept is None:
            snow_rain_intercept = torch.zeros_like(canopy_out.ptsfl)
        soil_ptsfl = torch.clamp(canopy_out.ptsfl - snow_rain_intercept, min=0.0)
        soilw_out = self._update_soil_moisture(
            state=state, static=static, diff_out=land_flux_out,
            forcing=forcing, pfrti=frac,
            ptsfl=soil_ptsfl, peintti=canopy_out.peintti,
            pmsn=snow_out.pmsn, pemssn=snow_out.pemssn, pevapsnw=pevapsnw,
            dt=dt, ngrid=ngrid, dtype=dtype, device=device,
        )
        new_soil_w = soilw_out.pwsa

        # ------------------------------------------------------------------ #
        # 7. Pack the new state and diagnostics.
        # ------------------------------------------------------------------ #
        # The wrapper extrapolates skin T to t+1 (ZTPFAC2*PTSK +
        # ZTPFAC3*PTSKTI); use that as the new prognostic skin state so
        # the next coupling interval starts from the t+1 value the
        # offline driver would write back to PTSKTI.  Soil temperature
        # and soil moisture advance through the HTESSEL solves above.
        # Snow state advances through SRFSN_LWIMP: PTSN <- ZTSN,
        # PSNM1M <- ZSSTAR (snow mass after P-E); PRSNM1M advances
        # through the SRFSN_LWIMP section-3.1 density-evolution port
        # (_update_snow_density).  The skin-reservoir water advances
        # through SRFWL: PWL <- canopy_out.pwl.
        new_snow_t = snow_out.ptsn
        skin_water_tendency = (canopy_out.pwl - state.skin_water) / dt
        skin_water_out = state.skin_water + dt * skin_water_tendency
        provisional_state = state.like(
            skin_t=surfpp_skin_t,
            soil_t=new_soil_t,
            snow_t=new_snow_t,
            snow_w=new_snow_w,
            snow_frac=new_snow_frac,
            skin_water=skin_water_out,
        )
        next_frac = self._tile_frac(static, ntile, state=provisional_state)
        # SURFPPS_CTL already replaces zero-fraction tile temperatures by
        # the grid-mean skin temperature.  A snow tile that becomes active
        # at the next SURFBC call must retain that value; replacing it with
        # TSRF here incorrectly resets the tile to RTT.
        skin_t_out = surfpp_skin_t
        new_state = state.like(
            skin_t=skin_t_out.clone(),
            # This is PTL at t+1.  It must use the fractions from the VDF
            # step, before snow/skin-water updates change the tile split.
            radiative_skin_t=scatter_to_tiles(skin_t_out, frac).clone(),
            soil_t=new_soil_t,
            soil_w_liq=new_soil_w,
            snow_t=new_snow_t,
            snow_w=new_snow_w,
            snow_rho=new_snow_rho,
            snow_frac=new_snow_frac,
            snow_alb=new_snow_alb,
            skin_water=skin_water_out,
            vupdz0_step=self._surf_step_count + 1,
            vupdz0_pustrti=self._last_pustrti,
            vupdz0_pvstrti=self._last_pvstrti,
            vupdz0_pahfsti=self._last_pahfsti,
            vupdz0_pevapti=self._last_pevapti,
            vupdz0_pzdl=self._last_pzdl,
        )
        diag = (
            self._build_diag(out, static, ntile, state=state)
            if self.config.compute_diagnostics
            else SurfDiag()
        )

        # Surface the Farquhar carbon-flux diagnostics when active.  These
        # are per-grid (Ngrid,) tensors already computed inside the
        # use_farquhar branch; attaching them here keeps the prognostic
        # forward path untouched.
        if _farquhar_diag is not None and self.config.compute_diagnostics:
            diag.farquhar_pgstot = _farquhar_diag["farquhar_pgstot"]
            diag.farquhar_pgpp   = _farquhar_diag["farquhar_pgpp"]
            diag.farquhar_prd    = _farquhar_diag["farquhar_prd"]
            diag.farquhar_pan    = _farquhar_diag["farquhar_pan"]

        # Optional invariant checks (SurfConfig.validate).  Cheap; catches
        # porting regressions early.  Disabled in differentiable / training
        # hot loops by setting validate=False.
        if self.config.validate:
            # Validate the tile fractions the step actually used (the
            # t-1 state SURFBC splits on -- surfbc_ctl_mod.F90 runs
            # before SRFWL / SRFSN).
            self._validate_state(state, forcing, static, new_state, ntile)

        return new_state, diag

    # ----------------------------------------------------------------- #
    # Helpers used by forward()
    # ----------------------------------------------------------------- #
    def _surfpp_zero_tile_fluxes(
        self,
        pssh: torch.Tensor,
        pjq: torch.Tensor,
        frac: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mirror SURFPP_CTL zero-fraction PAHFSTI/PEVAPTI replacement."""
        ntile = pssh.shape[1]
        max_land_tile = min(ntile, TILE_BARE_SOIL + 1)
        # SURFPP_CTL accumulates tiles in ascending tile order.  Do not use
        # a GPU reduction here: its tree order changes the last bit of the
        # mean, which then feeds the exact-zero tile replacement below.
        mean_pssh = torch.zeros_like(pssh[:, :1])
        mean_pjq = torch.zeros_like(pjq[:, :1])
        for tile in range(ntile):
            mean_pssh = mean_pssh + frac[:, tile:tile + 1] * pssh[:, tile:tile + 1]
            mean_pjq = mean_pjq + frac[:, tile:tile + 1] * pjq[:, tile:tile + 1]
        # SURFPP_CTL only substitutes the tile mean when PFRTI is exactly
        # zero. Tiny, non-zero snow and wet-skin fractions retain their
        # own fluxes and skin temperatures for the next VUPDZ0 call.
        zero_frac = frac[:, :max_land_tile] == 0.0
        pssh_land = torch.where(
            zero_frac, mean_pssh.expand(-1, max_land_tile), pssh[:, :max_land_tile]
        )
        pjq_land = torch.where(
            zero_frac, mean_pjq.expand(-1, max_land_tile), pjq[:, :max_land_tile]
        )
        if max_land_tile < ntile:
            pssh_land = torch.cat((pssh_land, pssh[:, max_land_tile:]), dim=1)
            pjq_land = torch.cat((pjq_land, pjq[:, max_land_tile:]), dim=1)
        return pssh_land, pjq_land

    def _surfpp_zero_tile_skin_t(
        self,
        ptsktip1: torch.Tensor,
        frac: torch.Tensor,
    ) -> torch.Tensor:
        """Mirror SURFPPS_CTL zero-fraction PTSKTI replacement."""
        # Keep the same ordered accumulation as SURFPP_CTL's ZTSK loop.
        mean_tsk = torch.zeros_like(ptsktip1[:, :1])
        for tile in range(ptsktip1.shape[1]):
            mean_tsk = mean_tsk + (
                frac[:, tile:tile + 1] * ptsktip1[:, tile:tile + 1]
            )
        # Match SURFPP_CTL: only exact zero-area tiles receive ZTSK.
        zero_frac = frac == 0.0
        return torch.where(zero_frac, mean_tsk.expand_as(ptsktip1), ptsktip1)

    def _snow_under_forest_evap(
        self,
        *,
        forcing: Forcing,
        state: SurfState,
        rhoc_q_t: torch.Tensor,
        pcsnw: torch.Tensor,
        frac: torch.Tensor,
        ntile: int,
        dtype: torch.dtype,
        device,
    ) -> torch.Tensor:
        """Compute SURFEXCDRIVER_CTL PEVAPSNW for tile 7."""
        ngrid = forcing.q.shape[0]
        if forcing.snow_under_forest_evap is not None:
            external = forcing.snow_under_forest_evap
            if tuple(external.shape) != (ngrid,):
                raise ValueError(
                    "Forcing.snow_under_forest_evap must have shape "
                    f"({ngrid},), got {tuple(external.shape)}"
                )
            return external.to(dtype=dtype, device=device)
        zero = torch.zeros((ngrid,), dtype=dtype, device=device)
        if ntile <= TILE_HIGH_VEG_SNOW:
            return zero

        pcsnw_high = pcsnw[:, TILE_HIGH_VEG_SNOW]
        # SURFEXCDRIVER computes the snow saturation humidity only when
        # both tile 7 and its VEVAP snow-through-canopy factor are active.
        # Do not impose an artificial temperature floor: cold but physical
        # snow temperatures are part of the Fortran calculation.
        # SURFEXCDRIVER_CTL tests PFRTI(:,7) > 0 exactly.  A small but
        # positive sheltered-snow tile still exchanges moisture and must
        # not be suppressed by SURFBC's separate fraction threshold.
        active = (frac[:, TILE_HIGH_VEG_SNOW] > 0.0) & (pcsnw_high > 0.0)
        snow_t = state.snow_t if state.snow_t is not None else state.soil_t[:, 0]
        snow_t_safe = torch.where(
            active,
            snow_t.to(dtype=dtype, device=device),
            torch.full((ngrid,), self.consts.rtt, dtype=dtype, device=device),
        )
        qssn = qsat(
            snow_t_safe,
            forcing.p_surf.to(dtype=dtype, device=device),
            self.consts,
        )
        pevapsnw = (
            rhoc_q_t[:, TILE_HIGH_VEG_SNOW]
            * pcsnw_high
            * (forcing.q.to(dtype=dtype, device=device) - qssn)
        )
        return torch.where(active, pevapsnw, zero)

    def _exchange_coeffs(
        self,
        forcing: Forcing,
        static: Static,
        state: SurfState,
        skin_t: Optional[torch.Tensor] = None,
        frac: Optional[torch.Tensor] = None,
        ktvl: Optional[torch.Tensor] = None,
        ktvh: Optional[torch.Tensor] = None,
        tile_shortwave: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-tile ``(rho*Ch*|U|, rho*Cq*|U|``).

        The calculation mirrors the offline driver's VEXCS tile loop with
        tile-specific roughness and surface state.  No grid-mean or neutral
        exchange approximation is available in the scientific interface.
        """
        return self._per_tile_monin_obukhov_coeffs(
            forcing, static, state, skin_t=skin_t, frac=frac,
            ktvl=ktvl, ktvh=ktvh, tile_shortwave=tile_shortwave,
        )

    def _vexcs_step_controls(self) -> Tuple[bool, int]:
        """SURFEXCDRIVER VEXCS controls: LLINIT and IITT from KSTEP."""
        kstep = int(getattr(self, "_surf_step_count", 0))
        return kstep == 0, 3 if kstep <= 3 else 1

    def _vupdz0_warm_start(
        self,
        forcing: Forcing,
        state: SurfState,
        ntile: int,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Rebuild VUPDZ0 PZDLTI/PBUOMTI from previous tiled fluxes."""
        ngrid = state.skin_t.shape[0]
        dtype = state.skin_t.dtype
        device = state.skin_t.device
        zero = torch.zeros((ngrid, ntile), dtype=dtype, device=device)
        needed = (self._last_pustrti, self._last_pvstrti,
                  self._last_pahfsti, self._last_pevapti)
        if any(x is None for x in needed):
            return None, zero.reshape(-1)
        if any(tuple(x.shape) != (ngrid, ntile) for x in needed):
            return None, zero.reshape(-1)

        pustr = self._last_pustrti.to(dtype=dtype, device=device)
        pvstr = self._last_pvstrti.to(dtype=dtype, device=device)
        pahfs = self._last_pahfsti.to(dtype=dtype, device=device)
        pevap = self._last_pevapti.to(dtype=dtype, device=device)

        rho = forcing.p_surf / (
            self.consts.rd * forcing.t * (1.0 + self.consts.retv * forcing.q)
        )
        zrowq = pevap
        zrowt = pahfs / self.consts.cpd
        pbuom = self.consts.g * (
            -self.consts.retv * zrowq - zrowt / state.skin_t
        ) / rho.unsqueeze(-1)

        zust2 = torch.sqrt(pustr.square() + pvstr.square()) / rho.unsqueeze(-1)
        pos_buom = torch.clamp(pbuom, min=0.0)
        wstar_base = pos_buom * RPARZI
        if torch.is_grad_enabled():
            wstar_base = torch.clamp(
                wstar_base, min=torch.finfo(wstar_base.dtype).eps,
            )
        zwst2 = torch.where(
            pbuom > 0.0,
            wstar_base ** (2.0 / 3.0),
            torch.zeros_like(pbuom),
        )
        zust2 = torch.where(pbuom > 0.0, zust2 + 2.0e-3 * zwst2, zust2)
        zust = torch.clamp(torch.sqrt(torch.clamp(zust2, min=0.0)), min=1.0e-4)
        pzdl = -forcing.geom1.unsqueeze(-1) * RKAP * pbuom / (
            self.consts.g * zust ** 3
        )
        return pzdl.reshape(-1), pbuom.reshape(-1)

    def _vupdz0_cold_start_fluxes(
        self,
        forcing: Forcing,
        z0m: torch.Tensor,
    ) -> None:
        """Initialise VUPDZ0's step-zero neutral tile flux estimate.

        ``vupdz0_mod.F90`` performs this calculation internally when
        ``KSTEP == 0``.  Offline Torch calls must therefore not begin with
        zero momentum stress simply because no previous VDF call exists.
        Coupled calls bypass this helper and receive their VUP fields from
        the atmosphere.
        """
        dtype, device = z0m.dtype, z0m.device
        ngrid, ntile = z0m.shape
        ucurr = forcing.ucurr if forcing.ucurr is not None else torch.zeros(
            ngrid, dtype=dtype, device=device
        )
        vcurr = forcing.vcurr if forcing.vcurr is not None else torch.zeros(
            ngrid, dtype=dtype, device=device
        )
        zdu2 = torch.clamp(
            (forcing.u - ucurr).square() + (forcing.v - vcurr).square(),
            min=REPDU2,
        )
        zdua = torch.sqrt(zdu2).unsqueeze(-1)
        rho = forcing.p_surf / (
            self.consts.rd * forcing.t * (1.0 + self.consts.retv * forcing.q)
        )
        cdn = (RKAP / torch.log(
            1.0 + forcing.geom1.unsqueeze(-1) / (self.consts.g * z0m)
        )) ** 2
        self._last_pustrti = (
            rho.unsqueeze(-1) * forcing.u.unsqueeze(-1) * zdua * cdn
        ).detach().clone()
        self._last_pvstrti = (
            rho.unsqueeze(-1) * forcing.v.unsqueeze(-1) * zdua * cdn
        ).detach().clone()
        self._last_pahfsti = torch.zeros((ngrid, ntile), dtype=dtype, device=device)
        self._last_pevapti = torch.zeros((ngrid, ntile), dtype=dtype, device=device)

    def _vupdz0_praqti(
        self,
        forcing: Forcing,
        state: SurfState,
        z0m: torch.Tensor,
        z0q: torch.Tensor,
        pzdl0: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Preliminary moisture aerodynamic resistance from VUPDZ0."""
        ngrid, ntile = z0m.shape
        dtype = state.skin_t.dtype
        device = state.skin_t.device
        pzdl = torch.zeros((ngrid, ntile), dtype=dtype, device=device) if pzdl0 is None else pzdl0.reshape(ngrid, ntile)

        rho = forcing.p_surf / (
            self.consts.rd * forcing.t * (1.0 + self.consts.retv * forcing.q)
        )
        if self._last_pustrti is not None and self._last_pvstrti is not None                 and tuple(self._last_pustrti.shape) == (ngrid, ntile)                 and tuple(self._last_pvstrti.shape) == (ngrid, ntile):
            pustr = self._last_pustrti.to(dtype=dtype, device=device)
            pvstr = self._last_pvstrti.to(dtype=dtype, device=device)
            zust2 = torch.sqrt(pustr.square() + pvstr.square()) / rho.unsqueeze(-1)
            # Match VUPDZ0's free-convection correction when previous fluxes exist.
            if self._last_pahfsti is not None and self._last_pevapti is not None                     and tuple(self._last_pahfsti.shape) == (ngrid, ntile)                     and tuple(self._last_pevapti.shape) == (ngrid, ntile):
                pahfs = self._last_pahfsti.to(dtype=dtype, device=device)
                pevap = self._last_pevapti.to(dtype=dtype, device=device)
                pbuom = self.consts.g * (
                    -self.consts.retv * pevap - (pahfs / self.consts.cpd) / state.skin_t
                ) / rho.unsqueeze(-1)
                wstar_base = torch.clamp(pbuom, min=0.0) * RPARZI
                if torch.is_grad_enabled():
                    wstar_base = torch.clamp(
                        wstar_base, min=torch.finfo(wstar_base.dtype).eps,
                    )
                zwst2 = torch.where(
                    pbuom > 0.0,
                    wstar_base ** (2.0 / 3.0),
                    torch.zeros_like(pbuom),
                )
                zust2 = torch.where(pbuom > 0.0, zust2 + 2.0e-3 * zwst2, zust2)
            zust = torch.clamp(torch.sqrt(torch.clamp(zust2, min=0.0)), min=1.0e-4)
        else:
            # VUPDZ0 only constructs neutral stresses at KSTEP=0.  A restart
            # at KSTEP>0 has no preceding SURFPP flux in its restart file,
            # so Fortran retains zero stress and applies REPUST directly.
            zust = torch.full((ngrid, ntile), 1.0e-4, dtype=dtype, device=device)

        znlev = forcing.geom1.unsqueeze(-1) / self.consts.g + z0m
        z1dz0q = znlev / z0q
        zxlnq = torch.log(z1dz0q)
        stable = pzdl > 0.0
        zprh1 = torch.where(stable, psihs(pzdl), psihu(pzdl))
        eta0 = pzdl / z1dz0q
        zprq0 = torch.where(stable, psihs(eta0), psihu(eta0))
        return (zxlnq - zprh1 + zprq0) / (zust * RKAP)

    def _vsurf_apparent_humidity(
        self,
        forcing: Forcing,
        static: Static,
        state: SurfState,
        frac: torch.Tensor,
        ktvl: torch.Tensor,
        ktvh: torch.Tensor,
        qs_skin: torch.Tensor,
        praqti: torch.Tensor,
        ntile: int,
        tile_shortwave: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """VSURF apparent surface humidity ZQSATI before VEXCS."""
        if tile_shortwave is None:
            # Compatibility path for direct helper callers. The forward path
            # provides the Fortran-style tiled radiation boundary.
            sw_down = forcing.fr_so / forcing.dt
        else:
            alpha_t = self._surface_albedo(
                state, static, forcing, ntile, state.skin_t.dtype,
                state.skin_t.device,
            )
            sw_down = self._tile_downward_shortwave(tile_shortwave, alpha_t)
        lai_low, lai_high = self._effective_lai(static, ktvl, ktvh)
        jarvis = compute_jarvis_resistance(
            soil_t=state.soil_t,
            soil_w=state.soil_w_liq,
            soil_type=static.soil_type,
            ktvl=ktvl, ktvh=ktvh,
            t_air=forcing.t, q_air=forcing.q, p_surf=forcing.p_surf,
            sw_down=sw_down,
            veg_table=self.veg_table,
            soil_table=self.soil_table,
            root_table=self.root_table,
            consts=self.consts,
            tile_frac=frac, n_tile=ntile,
            lai_low=lai_low,
            lai_high=lai_high,
        )
        pqsam = qs_skin.clone()
        qdef = torch.clamp(forcing.q.unsqueeze(-1) - qs_skin, max=0.0)

        def set_tile(tile: int, resistance: torch.Tensor) -> None:
            if ntile > tile:
                r = resistance.unsqueeze(-1)
                pqsam[:, tile:tile + 1] = qs_skin[:, tile:tile + 1] + qdef[:, tile:tile + 1] * r / (r + praqti[:, tile:tile + 1])

        set_tile(TILE_LOW_VEG_DRY, jarvis["pwetl"])
        set_tile(TILE_HIGH_VEG_DRY, jarvis["pweth"])
        set_tile(TILE_HIGH_VEG_SNOW, jarvis.get("pweths", jarvis["pweth"]))
        set_tile(TILE_BARE_SOIL, jarvis["pwetb"])
        if ntile > TILE_URBAN:
            set_tile(TILE_URBAN, jarvis["pwetb"])
        return torch.clamp(pqsam, min=1.0e-12)

    def _cache_vexcs_stress(
        self,
        out: ExchangeOutputs,
        forcing: Forcing,
        frac: torch.Tensor,
        ngrid: int,
        ntile: int,
    ) -> None:
        """Cache the offline VDFMAIN1S stress for the next VUPDZ0 call.

        SURFPP copies the VDFINCR lowest-level grid stress to every tile.
        In OSM offline mode VDFDIFM1S has no vertical solve; VDFINCR reduces
        exactly to the area-weighted VEXCS coefficient times relative wind.
        """
        dtype = out.pcfm.dtype
        device = out.pcfm.device
        ucurr = forcing.ucurr if forcing.ucurr is not None else torch.zeros(
            ngrid, dtype=dtype, device=device)
        vcurr = forcing.vcurr if forcing.vcurr is not None else torch.zeros(
            ngrid, dtype=dtype, device=device)
        momentum = vdfdifm1s_surface_stress(
            out.pcfm.reshape(ngrid, ntile), frac,
            # VDFDIFM1S/VDFINCR form PUDIF/PVDIF from the lowest-level
            # state plus the atmospheric momentum tendency.  The resulting
            # updated velocity is what SURFPP returns to the next VUPDZ0
            # call as PUSTRTI/PVSTRTI.
            forcing.u.to(dtype=dtype, device=device) + (
                forcing.dt * forcing.u_tendency.to(dtype=dtype, device=device)
                if forcing.u_tendency is not None else 0.0
            ),
            forcing.v.to(dtype=dtype, device=device) + (
                forcing.dt * forcing.v_tendency.to(dtype=dtype, device=device)
                if forcing.v_tendency is not None else 0.0
            ),
            ucurr,
            vcurr,
        )
        stress_u = (forcing.u_stress.to(dtype=dtype, device=device)
                    if forcing.u_stress is not None else momentum.stress_u)
        stress_v = (forcing.v_stress.to(dtype=dtype, device=device)
                    if forcing.v_stress is not None else momentum.stress_v)
        pustr_grid = stress_u.unsqueeze(-1)
        pvstr_grid = stress_v.unsqueeze(-1)

        self._last_pustrti = pustr_grid.expand(ngrid, ntile).detach().clone()
        self._last_pvstrti = pvstr_grid.expand(ngrid, ntile).detach().clone()

    def _per_tile_z0(
        self,
        static: Static,
        frac: torch.Tensor,
        ktvl: torch.Tensor,
        ktvh: torch.Tensor,
        ntile: int,
        dtype: torch.dtype,
        device,
        state: Optional[SurfState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-tile roughness lengths ``(z0m, z0h, z0q)`` each (Ngrid, ntile).

        Transcribes ``vupdz0s_mod.F90:296-378`` (the cold-start static-z0
        branch; the dynamic roughness-feedback from prior fluxes is not
        ported).  For every land tile z0q = z0h.  Low / high vegetation
        tiles key off the dominant veg type (``ktvl`` / ``ktvh``);
        bare-soil uses the VegTable index-0 desert sentinel; the wet-skin
        tile blends low / high / bare by the cover fractions CVL/CVH.

        Water / ice tiles use a Charnock placeholder (``1e-4`` m): these
        tiles have zero fraction in the pure-land default and their
        sea-state-dependent z0 (the ``LWCOU2W`` branch) is out of scope
        for the offline single-column driver.
        """
        ngrid = frac.shape[0]
        # Per-type lookups indexed by the dominant veg type.  VegTable
        # index 0 is the desert sentinel (bare-soil), 1..20 are veg types.
        z0m_low = self.veg_table.gather("rvz0m", ktvl)
        z0h_low = self.veg_table.gather("rvz0h", ktvl)
        z0m_high = self.veg_table.gather("rvz0m", ktvh)
        z0h_high = self.veg_table.gather("rvz0h", ktvh)
        # Snow type 12 (exposed-snow, vupdz0_mod.F90).
        snow_idx = torch.full_like(ktvl, 12)
        z0m_snow = self.veg_table.gather("rvz0m", snow_idx)
        z0h_snow = self.veg_table.gather("rvz0h", snow_idx)
        # Bare-soil / desert (index 0 sentinel).
        bare_idx = torch.zeros_like(ktvl)
        z0m_bare = self.veg_table.gather("rvz0m", bare_idx)
        z0h_bare = self.veg_table.gather("rvz0h", bare_idx)

        # VUPDZ0 uses the RVCOV-corrected climatological covers (PCVL/PCVH),
        # not the transient snow-split tile fractions. Retain the latter as a
        # fallback for callers that do not provide vegetation-cover metadata.
        covers = self._cover_fractions(static)
        if covers is None:
            pcvl = frac[..., TILE_LOW_VEG_DRY] + frac[..., TILE_LOW_VEG_SNOW]
            pcvh = frac[..., TILE_HIGH_VEG_DRY] + frac[..., TILE_HIGH_VEG_SNOW]
        else:
            pcvl, pcvh = covers[0], covers[1]
        one_minus = 1.0 - pcvl - pcvh
        # Wet-skin blend: cvl*RVZ0M(ktvl)+cvh*RVZ0M(ktvh)+(1-cvl-cvh)*RVZ0M(0)
        z0m_wet = pcvl * z0m_low + pcvh * z0m_high + one_minus * z0m_bare
        z0h_wet = pcvl * z0h_low + pcvh * z0h_high + one_minus * z0h_bare
        # Exposed-snow blend (VUPDZ0 tile 5): low vegetation + bare soil,
        # normalized by the non-high-vegetation cover, then blended toward
        # snow type 12 by snow depth. This is distinct from the grid log
        # blend used only for momentum roughness.
        denom_low = torch.clamp(1.0 - pcvh, min=torch.finfo(dtype).tiny)
        z0m_low_bare = pcvl / denom_low * z0m_low + one_minus / denom_low * z0m_bare
        z0h_low_bare = pcvl / denom_low * z0h_low + one_minus / denom_low * z0h_bare

        # Placeholder for water / ice (frac=0 in pure-land default).
        z0_placeholder = torch.full(
            (ngrid,), 1.0e-4, dtype=dtype, device=device,
        )

        # Initialise every tile to a safe default (bare-soil z0).  This
        # guards the n_tile=10 topology: the lake / urban tiles (8 / 9)
        # have no vupdz0s branch (their sea-state / canopy roughness lives
        # in the FLake / TEB modules that are not ported), so without an
        # explicit fill the torch.empty below would leave those slots with
        # uninitialised memory that propagates NaN through VEXCS.  Any tile
        # that gets overwritten below is unaffected; tiles 8 / 9 keep the
        # bare-soil placeholder, which is benign when their fraction is
        # zero (the pure-land default) and merely a stub when it is not.
        z0m = torch.full(
            (ngrid, ntile), 0.0, dtype=dtype, device=device,
        )
        z0m[:, :] = z0m_bare.unsqueeze(-1)
        z0h = torch.full(
            (ngrid, ntile), 0.0, dtype=dtype, device=device,
        )
        z0h[:, :] = z0h_bare.unsqueeze(-1)
        z0m[:, TILE_WATER] = z0_placeholder
        z0h[:, TILE_WATER] = z0_placeholder
        if ntile > TILE_ICE:
            z0m[:, TILE_ICE] = z0_placeholder
            z0h[:, TILE_ICE] = z0_placeholder
        z0m[:, TILE_WET_SKIN] = z0m_wet
        z0h[:, TILE_WET_SKIN] = z0h_wet
        z0m[:, TILE_LOW_VEG_DRY] = z0m_low
        z0h[:, TILE_LOW_VEG_DRY] = z0h_low
        # VUPDZ0 blends the exposed-snow roughness with the low/bare value
        # using total physical snow depth, PDSN = SWE / snow density.
        if state is None or state.snow_w is None or state.snow_rho is None:
            snow_weight = torch.zeros(ngrid, dtype=dtype, device=device)
        else:
            snow_depth = state.snow_w.to(dtype=dtype, device=device) / torch.clamp(
                state.snow_rho.to(dtype=dtype, device=device),
                min=torch.finfo(dtype).tiny,
            )
            snow_weight = torch.clamp(snow_depth / 0.25, min=0.0, max=1.0)
        z0m[:, TILE_LOW_VEG_SNOW] = (
            snow_weight * z0m_snow + (1.0 - snow_weight) * z0m_low_bare
        )
        z0h[:, TILE_LOW_VEG_SNOW] = (
            snow_weight * z0h_snow + (1.0 - snow_weight) * z0h_low_bare
        )
        z0m[:, TILE_HIGH_VEG_DRY] = z0m_high
        z0h[:, TILE_HIGH_VEG_DRY] = z0h_high
        z0m[:, TILE_HIGH_VEG_SNOW] = z0m_high      # sheltered snow uses ktvh
        z0h[:, TILE_HIGH_VEG_SNOW] = z0h_high
        if ntile > TILE_BARE_SOIL:
            z0m[:, TILE_BARE_SOIL] = z0m_bare
            z0h[:, TILE_BARE_SOIL] = z0h_bare
        # z0q = z0h for every land tile (vupdz0_mod.F90).
        z0q = z0h.clone()
        return z0m, z0h, z0q

    def _blend_momentum_z0(
        self,
        z0m: torch.Tensor,
        frac: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """SURFEXCDRIVER_CTL log-blend for momentum roughness.

        VUPDZ0 first computes PRAQTI with tile-specific PZ0MTI.  Only after
        VUPDZ0 returns does SURFEXCDRIVER_CTL compute the grid log-blend and
        write it back to every tile before VEXCS.  Keeping this as a separate
        helper preserves that ordering.
        """
        dtype = z0m.dtype
        device = z0m.device
        zblend = torch.as_tensor(10.0, dtype=dtype, device=device)
        tiny = torch.as_tensor(1.0e-12, dtype=dtype, device=device)
        z0_safe = torch.clamp(z0m, min=tiny, max=zblend * (1.0 - 1.0e-12))
        cblend = (frac / torch.log(zblend / z0_safe).square()).sum(dim=-1)
        cblend_safe = torch.where(cblend > 0.0, cblend, torch.ones_like(cblend))
        blended = zblend * torch.exp(-1.0 / torch.sqrt(cblend_safe))
        return torch.where(cblend > 0.0, blended, fallback)

    def _per_tile_monin_obukhov_coeffs(
        self,
        forcing: Forcing,
        static: Static,
        state: SurfState,
        skin_t: Optional[torch.Tensor] = None,
        frac: Optional[torch.Tensor] = None,
        ktvl: Optional[torch.Tensor] = None,
        ktvh: Optional[torch.Tensor] = None,
        tile_shortwave: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-tile Beljaars Monin-Obukhov exchange coefficients (VEXCS).

        Mirrors the offline driver's tile loop
        (``surfexcdrivers_ctl_mod.F90:388-398``): one VEXCS call over the
        flattened ``(Ngrid, Ntile)`` axis, with tile-specific roughness
        lengths (:meth:`_per_tile_z0`) and tile-specific surface state
        (per-tile ``pcpts = cp*Tskin``, per-tile ``pqsam = qsat(Tskin)``).
        Returns per-tile ``(rhoc_u, rhoc_q)`` each of shape
        ``(Ngrid, Ntile)`` -- the SEB consumes these directly (no
        broadcast needed).

        The :func:`vexcs` kernel is pure elementwise torch math, so a
        single vectorised call on the flattened axis reproduces the
        per-tile loop exactly (including the per-tile stability
        iteration, which is column-local).
        """
        ntile = self.config.n_tile
        cp = self.consts.cpd
        skin_t = state.skin_t if skin_t is None else skin_t
        ngrid = skin_t.shape[0]
        dtype = self.config.dtype
        device = skin_t.device

        if frac is None:
            frac = self._step_tile_frac(forcing, static, state, ntile)
        if ktvl is None or ktvh is None:
            ktvl, ktvh = self._dominant_veg_types(static)

        # Per-tile saturation humidity and roughness lengths.
        p_surf_t = expand_as_tiles(forcing.p_surf, ntile)
        qs_skin = qsat(skin_t, p_surf_t, self.consts)
        z0m_tile, z0h, z0q = self._per_tile_z0(
            static, frac, ktvl, ktvh, ntile, dtype, device, state=state,
        )
        boundary_mode = self.config.vup_boundary_mode
        if boundary_mode == "external":
            roughness = {
                "vupdz0_z0mti": forcing.vupdz0_z0mti,
                "vupdz0_z0hti": forcing.vupdz0_z0hti,
                "vupdz0_z0qti": forcing.vupdz0_z0qti,
            }
            missing = [name for name, value in roughness.items() if value is None]
            if missing:
                raise ValueError(
                    "External-boundary mode requires authoritative VUPDZ0 roughness fields: "
                    + ", ".join(missing)
                )
            for name, value in roughness.items():
                if tuple(value.shape) != (ngrid, ntile):
                    raise ValueError(
                        f"Forcing.{name} must have shape ({ngrid}, {ntile}), "
                        f"got {tuple(value.shape)}"
                    )
            z0m_tile = roughness["vupdz0_z0mti"].to(dtype=dtype, device=device)
            z0h = roughness["vupdz0_z0hti"].to(dtype=dtype, device=device)
            z0q = roughness["vupdz0_z0qti"].to(dtype=dtype, device=device)

        # Optional buoyancy flux / ocean current -- zero when absent.
        buom_g = forcing.buom if forcing.buom is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)
        ucurr_g = forcing.ucurr if forcing.ucurr is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)
        vcurr_g = forcing.vcurr if forcing.vcurr is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)
        i10fgcv_g = forcing.i10fgcv if forcing.i10fgcv is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)

        # Flatten (Ngrid, ntile) -> (Ngrid*ntile,) for a single vectorised
        # VEXCS call.  Lowest-level fields (u/v/t/q/p_surf/geom1/pcptgzlev)
        # are grid-mean, so repeat-interleave them across the tile axis.
        def _rep(g):
            return g.unsqueeze(-1).expand(ngrid, ntile).reshape(-1)
        pcptgzlev_g = cp * forcing.t + forcing.geom1
        ldinit, kitt = self._vexcs_step_controls()
        if boundary_mode == "offline":
            if self._surf_step_count == 0:
                self._vupdz0_cold_start_fluxes(forcing, z0m_tile)
            pzdl0, pbuom0 = self._vupdz0_warm_start(forcing, state, ntile)
            praqti = self._vupdz0_praqti(forcing, state, z0m_tile, z0q, pzdl0)
            self._last_vup_pzdl0 = (
                None if pzdl0 is None else pzdl0.reshape(ngrid, ntile).detach()
            )
            self._last_vup_pbuom0 = pbuom0.reshape(ngrid, ntile).detach()
            self._last_vup_praqti = praqti.detach()
        elif boundary_mode == "external":
            external = {
                "vupdz0_pustrti": forcing.vupdz0_pustrti,
                "vupdz0_pvstrti": forcing.vupdz0_pvstrti,
                "vupdz0_pahfsti": forcing.vupdz0_pahfsti,
                "vupdz0_pevapti": forcing.vupdz0_pevapti,
                "vupdz0_pbuomti": forcing.vupdz0_pbuomti,
                "vupdz0_pzdlti": forcing.vupdz0_pzdlti,
                "vupdz0_praqti": forcing.vupdz0_praqti,
            }
            missing = [name for name, value in external.items() if value is None]
            if missing:
                raise ValueError(
                    "External-boundary mode requires "
                    "authoritative VUPDZ0 forcing fields: " + ", ".join(missing)
                )
            for name, value in external.items():
                if tuple(value.shape) != (ngrid, ntile):
                    raise ValueError(
                        f"Forcing.{name} must have shape ({ngrid}, {ntile}), "
                        f"got {tuple(value.shape)}"
                    )
            self._last_pustrti = external["vupdz0_pustrti"].to(dtype=dtype, device=device).clone()
            self._last_pvstrti = external["vupdz0_pvstrti"].to(dtype=dtype, device=device).clone()
            self._last_pahfsti = external["vupdz0_pahfsti"].to(dtype=dtype, device=device).clone()
            self._last_pevapti = external["vupdz0_pevapti"].to(dtype=dtype, device=device).clone()
            pzdl0 = external["vupdz0_pzdlti"].to(dtype=dtype, device=device).reshape(-1)
            pbuom0 = external["vupdz0_pbuomti"].to(dtype=dtype, device=device).reshape(-1)
            praqti = external["vupdz0_praqti"].to(dtype=dtype, device=device)
            self._last_vup_praqti = praqti.detach()
        else:
            raise ValueError(
                "SurfConfig.vup_boundary_mode must be 'offline' or 'external', "
                f"got {boundary_mode!r}"
            )
        # SURFEXCDRIVER_CTL calls VSURF before VEXCS and passes the
        # resulting apparent surface humidity ZQSATI(:,JTILE) into VEXCS.
        # PRAQTI is computed above solely to close this VSURF relation.
        pqsam = self._vsurf_apparent_humidity(
            forcing, static, state, frac, ktvl, ktvh, qs_skin, praqti, ntile,
            tile_shortwave=tile_shortwave,
        )
        pcpts = skin_t * cp
        # After VUPDZ0 computes tile roughness, SURFEXCDRIVER_CTL forms the
        # grid log blend and writes that PZ0M back to every tile before the
        # VEXCS calls. Heat and moisture roughness remain tile-specific.
        z0m_fallback = (
            static.z0m.to(dtype=dtype, device=device)
            if static.z0m is not None
            else z0m_tile[:, TILE_BARE_SOIL]
        )
        z0m_blend = self._blend_momentum_z0(z0m_tile, frac, z0m_fallback)
        z0m = z0m_blend.unsqueeze(-1).expand_as(z0m_tile)
        self._last_vexcs_pcpts_tile = pcpts.detach()
        self._last_vexcs_pqsam_tile = pqsam.detach()
        self._last_vexcs_z0m_tile = z0m.detach()
        self._last_vexcs_z0h_tile = z0h.detach()
        self._last_vexcs_z0q_tile = z0q.detach()
        self._last_vexcs_pzdl_input_tile = pzdl0.reshape(ngrid, ntile).detach()
        self._last_vexcs_pbuom_input_tile = pbuom0.reshape(ngrid, ntile).detach()
        if boundary_mode == "offline":
            out: ExchangeOutputs = compute_vexcs(
                _rep(forcing.u), _rep(forcing.v), _rep(forcing.t), _rep(forcing.q),
                _rep(forcing.p_surf), _rep(forcing.geom1), _rep(pcptgzlev_g),
                pcpts.reshape(-1), pqsam.reshape(-1),
                z0m.reshape(-1), z0h.reshape(-1), z0q.reshape(-1),
                pzdl0, pbuom0, _rep(ucurr_g), _rep(vcurr_g), _rep(i10fgcv_g),
                self.consts, self.ritbl_table,
                kitt=kitt,
                ldinit=ldinit,
            )
            self._last_pcfm_tile = out.pcfm.reshape(ngrid, ntile).detach()
            self._last_pcfh_tile = out.pcfh.reshape(ngrid, ntile).detach()
            self._last_pcfq_tile = out.pcfq.reshape(ngrid, ntile).detach()
            self._last_vexcs_pzdl_tile = out.pzdl.reshape(ngrid, ntile).detach()
            self._cache_vexcs_stress(out, forcing, frac, ngrid, ntile)
            self._last_pzdl = scatter_to_tiles(
                out.pzdl.reshape(ngrid, ntile), frac,
            ).detach()
            return (
                out.pcfh.reshape(ngrid, ntile),
                out.pcfq.reshape(ngrid, ntile),
            )
        out: ExchangeOutputs = compute_vexcs(
            _rep(forcing.u), _rep(forcing.v), _rep(forcing.t), _rep(forcing.q),
            _rep(forcing.p_surf), _rep(forcing.geom1), _rep(pcptgzlev_g),
            pcpts.reshape(-1), pqsam.reshape(-1),
            z0m.reshape(-1), z0h.reshape(-1), z0q.reshape(-1),
            pzdl0, pbuom0, _rep(ucurr_g), _rep(vcurr_g), _rep(i10fgcv_g),
            self.consts, self.ritbl_table,
            kitt=kitt,
            ldinit=ldinit,
        )
        self._last_pcfm_tile = out.pcfm.reshape(ngrid, ntile).detach()
        self._last_pcfh_tile = out.pcfh.reshape(ngrid, ntile).detach()
        self._last_pcfq_tile = out.pcfq.reshape(ngrid, ntile).detach()
        self._last_vexcs_pzdl_tile = out.pzdl.reshape(ngrid, ntile).detach()
        self._cache_vexcs_stress(out, forcing, frac, ngrid, ntile)
        # Cache the per-tile z/L (grid-mean for the warm-start thread).
        self._last_pzdl = scatter_to_tiles(
            out.pzdl.reshape(ngrid, ntile), frac,
        ).detach()
        return (
            out.pcfh.reshape(ngrid, ntile),
            out.pcfq.reshape(ngrid, ntile),
        )

    def _sub_surface_t(self, state: SurfState, ntile: int) -> torch.Tensor:
        """Per-tile sub-surface temperature PTSRF (K).

        Land snow-free tiles use the top soil layer temperature; snow
        tiles use the prognostic snow temperature, matching
        ``SURFEXCDRIVER`` where ``PTSRF(:,5)`` and ``PTSRF(:,7)`` are
        set from ``PTSN(:,1)`` before ``VLAMSK``.  The water tile uses
        SST (skin pins to it via lambda_skin = LARGE).

        Note: this is the *VLAMSK* path (which needs the per-tile PTSRF
        to compute lambda_skin).  The VDFDIFH1S wrapper builds its own
        per-tile PTSRF internally from the per-grid scalar fields on
        :class:`DiffusionInputs` -- see :meth:`_build_diffusion_inputs`.
        """
        tsrf = expand_as_tiles(state.soil_t[:, 0], ntile).clone()
        if state.sst is not None:
            tsrf[:, TILE_WATER] = state.sst
        snow_t = state.snow_t if state.snow_t is not None else \
            self.consts.rtt.to(dtype=tsrf.dtype, device=tsrf.device).expand(tsrf.shape[0])
        if ntile > TILE_LOW_VEG_SNOW:
            tsrf[:, TILE_LOW_VEG_SNOW] = snow_t
        if ntile > TILE_HIGH_VEG_SNOW:
            tsrf[:, TILE_HIGH_VEG_SNOW] = snow_t
        return tsrf

    def _surface_emissivity(
        self,
        state: SurfState,
        static: Static,
        frac: torch.Tensor,
        dtype: torch.dtype,
        device,
    ) -> torch.Tensor:
        """SURFRAD broadband longwave emissivity PEMIT for NEMISSSCHEME=1."""
        weights = torch.tensor([0.149, 0.111, 0.126, 0.190, 0.290, 0.134], dtype=dtype, device=device)
        remiss_land = torch.tensor([0.98836, 0.98179, 0.98257, 0.98685, 0.98746, 0.98746], dtype=dtype, device=device)
        remiss_desert = torch.tensor([0.97395, 0.89598, 0.93831, 0.96009, 0.89529, 0.91139], dtype=dtype, device=device)
        remiss_snow = torch.tensor([0.97851, 0.98896, 0.98112, 0.96727, 0.98421, 0.99240], dtype=dtype, device=device)
        remiss_sea = torch.tensor([0.93780, 0.94782, 0.95285, 0.91103, 0.88590, 0.87016], dtype=dtype, device=device)
        e_land = (remiss_land * weights).sum()
        e_desert = (remiss_desert * weights).sum()
        e_snow = (remiss_snow * weights).sum()
        e_sea = (remiss_sea * weights).sum()

        soil_type = static.soil_type.to(device=device)
        wcap = self.soil_table.gather("wcap", soil_type).to(dtype=dtype, device=device)
        wpwp = self.soil_table.gather("wpwp", soil_type).to(dtype=dtype, device=device)
        zprop = torch.clamp(
            (state.soil_w_liq[:, 0] - wpwp) / torch.clamp(wcap - wpwp, min=torch.finfo(dtype).tiny),
            0.0, 1.0,
        )
        e_soilveg = e_land * zprop + e_desert * (1.0 - zprop)

        ntile = frac.shape[1]
        emis_t = torch.zeros_like(frac)
        if ntile > TILE_WATER:
            emis_t[:, TILE_WATER] = e_sea
        if ntile > TILE_ICE:
            emis_t[:, TILE_ICE] = e_snow
        if ntile > TILE_WET_SKIN:
            emis_t[:, TILE_WET_SKIN] = e_land
        if ntile > TILE_LOW_VEG_DRY:
            emis_t[:, TILE_LOW_VEG_DRY] = e_soilveg
        if ntile > TILE_LOW_VEG_SNOW:
            emis_t[:, TILE_LOW_VEG_SNOW] = e_snow
        if ntile > TILE_HIGH_VEG_DRY:
            emis_t[:, TILE_HIGH_VEG_DRY] = e_soilveg
        if ntile > TILE_HIGH_VEG_SNOW:
            emis_t[:, TILE_HIGH_VEG_SNOW] = e_soilveg
        if ntile > TILE_BARE_SOIL:
            albd = static.albd.to(dtype=dtype, device=device) if static.albd is not None else torch.zeros_like(zprop)
            emis_t[:, TILE_BARE_SOIL] = torch.where(albd > 0.30, e_desert, e_soilveg)
        return (emis_t * frac).sum(dim=1)

    def _build_diffusion_inputs(
        self,
        *,
        state: SurfState,
        forcing: Forcing,
        static: Static,
        sl_lev: torch.Tensor,
        q_lev: torch.Tensor,
        rhoc_u_t: torch.Tensor,
        rhoc_q_t: torch.Tensor,
        sskm1m: torch.Tensor,
        skin_t_prev: torch.Tensor,
        qskm1m: torch.Tensor,
        dqsdT: torch.Tensor,
        alphal: torch.Tensor,
        alphas: torch.Tensor,
        ssrfl_t: torch.Tensor,
        slrfl_grid: torch.Tensor,
        tskrad_grid: torch.Tensor,
        emis_grid: torch.Tensor,
        plamsk: torch.Tensor,
        tsrf: torch.Tensor,
        frac: torch.Tensor,
        ktvl: torch.Tensor,
        ktvh: torch.Tensor,
        pevapsnw: torch.Tensor,
        dt: torch.Tensor,
        ntile: int,
        ngrid: int,
        dtype: torch.dtype,
        device,
    ) -> DiffusionInputs:
        """Pack the forward() state into a :class:`DiffusionInputs`.

        Applies the upstream ``RG*PTSPHY*RVDIFTS`` rescaling to the raw
        ``rho*Ch*|U|`` exchange coefficients (mirroring ``vdfmain1s``
        lines 745-746) so the VDFDIFH1S wrapper's inversion
        ``ZCOEF1 = 1/(RG*RVDIFTS*PTMST)`` recovers the physical value.

        The per-grid lower-boundary temperatures (PSST / PTICE /
        PTSM1M / PTSNOW) are surfaced from the :class:`SurfState`; the
        VDFDIFH1S wrapper assembles the per-tile PTSRF internally.

        Lowest-level tendencies PSLGE / PQTE default to zero for legacy
        standalone calls.  Offline OSM forcing supplies both current and
        next atmospheric records; callers can provide the equivalent
        tendencies on ``Forcing`` so the Dirichlet boundary condition is
        ``bsl = PSLM1 + PTMST*PSLGE`` and ``bql = PQTM1 + PTMST*PQTE``.
        """
        cp = self.consts.cpd          # 0-d tensor, broadcasts
        rg = self.consts.g
        ptmst = dt if torch.is_tensor(dt) else torch.tensor(dt, dtype=dtype, device=device)
        if ptmst.ndim == 1:
            ptmst = ptmst.unsqueeze(-1)
        # vdfmain1s (lines 745-746) rescaling, inverted by the wrapper.
        # Preserve VDFMAIN1S's sequential multiplication order (lines
        # 755-759).  Forming RG*PTSPHY*RVDIFTS first changes the rounded
        # PCFHTI/PCFQTI boundary passed to VDFDIFH1S.
        pcfhti = ((rhoc_u_t * rg) * ptmst) * RVDIFTS
        pcfqti = ((rhoc_q_t * rg) * ptmst) * RVDIFTS

        # Per-grid lower-boundary temperatures.
        psst = state.sst if state.sst is not None else \
            torch.full((ngrid,), state.soil_t[:, 0].mean(),
                       dtype=dtype, device=device)
        # PTICE / PTSNOW: until the sea-ice / snow prognostic stubs
        # land (M6) fall back to RTT for ice and to the top soil-layer
        # temperature for snow -- this matches the offline-driver
        # default when those fields are absent from the cold-start.
        rtt = self.consts.rtt
        ptice = state.ice_t[:, 0] if (state.ice_t is not None
                                      and state.ice_t.shape[-1] > 0) else \
            rtt.expand(ngrid)
        ptsm1m = state.soil_t[:, 0]
        ptsnow = state.snow_t if state.snow_t is not None else \
            state.soil_t[:, 0].clone()

        # Optional lake tile fields -- surfaced from SurfState.lake_t
        # when present, otherwise left as None (the wrapper only
        # consumes them when n_tile > TILE_LAKE).
        phlice = getattr(state, "_phlice", None)
        ptlice = getattr(state, "_ptlice", None)
        ptlwml = state.lake_t

        return DiffusionInputs(
            pslm1_lev=sl_lev,
            pqtm1_lev=q_lev,
            pslge_lev=forcing.sl_tendency.to(dtype=dtype, device=device)
            if forcing.sl_tendency is not None else torch.zeros((ngrid,), dtype=dtype, device=device),
            pqte_lev=forcing.q_tendency.to(dtype=dtype, device=device)
            if forcing.q_tendency is not None else torch.zeros((ngrid,), dtype=dtype, device=device),
            pcfhti=pcfhti, pcfqti=pcfqti,
            psst=psst, ptice=ptice,
            ptsm1m=ptsm1m, ptsnow=ptsnow,
            sskm1m=sskm1m, tskm1m=skin_t_prev,
            qskm1m=qskm1m, dqsdT=dqsdT,
            alphal=alphal, alphas=alphas,
            pssrflti=ssrfl_t, pslrfl=slrfl_grid,
            ptskrad=tskrad_grid, pemis=emis_grid,
            plamsk=plamsk,
            pfrti=frac, ktvl=ktvl, ktvh=ktvh,
            ptmst=dt,
            pevapsnw=pevapsnw,
            phlice=phlice, ptlice=ptlice, ptlwml=ptlwml,
            lesnml=self.config.lesnml,
            lelwdd=self.config.lelwdd, lelwtl=self.config.lelwtl,
        )

    def _update_soil_temperature(
        self,
        *,
        state: SurfState,
        static: Static,
        diff_out,                       # DiffusionOutputs
        pssrflti: torch.Tensor,
        pslrfl: torch.Tensor,
        pslrflti: torch.Tensor,
        pfrti: torch.Tensor,
        pgsn: torch.Tensor,
        dt: torch.Tensor,
        ngrid: int,
        dtype: torch.dtype,
        device,
    ) -> torch.Tensor:
        """Advance the soil-temperature prognostic by one step (M5d-soilt).

        Builds a :class:`SoilTempInputs` from the t-1 soil state and the
        per-tile turbulent fluxes VDFDIFH1S just produced, runs the full
        HTESSEL soil-temperature solve (SRFRCGS + SRFTS + SRFWDIFS) and
        returns the new ``(Ngrid, Nsoil)`` soil-temperature profile.

        Mapping from VDFDIFH1S outputs to the SRFTS dummy list:

        * ``PAHFSTI`` (tile sensible heat, W/m^2)  <- ``diff_out.pssh``
        * ``PEVAPTI`` (tile moisture flux, kg/m^2/s) <- ``diff_out.pjq``
        * ``PSSRFLTI`` (tile net SW, W/m^2)        <- ``pssrflti`` arg
        * ``PSLRFL``   (net LW, W/m^2)             <- ``pslrfl`` arg
        * ``PFRTI``    (tile fractions)            <- ``pfrti`` arg
        * ``PGSN`` (snow-bottom heat flux)         <- ``pgsn`` arg (M6c)

        Sign conventions agree: both VDFDIFH1S and SRFTS use
        positive-downward fluxes (into the soil column), so the
        ``pssh`` / ``pjq`` columns carry directly through.  The latent
        contribution ``RLVTT*PEVAP`` is added inside
        :func:`compute_soil_temperature`'s ``_surface_net_heat_flux``.

        ``PCVL`` / ``PCVH`` (vegetation cover for the SRFRCGS apparent
        heat capacity) are reconstructed from the tile fractions: low-veg
        cover = dry-low-veg + snow-low-veg tile fractions, high-veg
        cover = dry-high-veg + snow-high-veg tile fractions.  This
        mirrors :meth:`_interception_capacity`'s reconstruction.
        """
        from .config import (
            TILE_LOW_VEG_DRY, TILE_LOW_VEG_SNOW,
            TILE_HIGH_VEG_DRY, TILE_HIGH_VEG_SNOW,
        )

        ntile = pfrti.shape[-1]

        # SRFRCG receives SURFBC's corrected static vegetation covers.
        # Snow-low (tile 5) also contains bare ground, so its fraction
        # cannot reconstruct PCVL after the dynamic snow split.
        covers = self._cover_fractions(static)
        if covers is not None:
            pcvl, pcvh, _, _, _ = covers
        else:
            pcvl = pfrti.new_zeros(pfrti.shape[:-1])
            pcvh = pfrti.new_zeros(pfrti.shape[:-1])
            if ntile > TILE_LOW_VEG_DRY:
                pcvl = pcvl + pfrti[..., TILE_LOW_VEG_DRY]
            if ntile > TILE_LOW_VEG_SNOW:
                pcvl = pcvl + pfrti[..., TILE_LOW_VEG_SNOW]
            if ntile > TILE_HIGH_VEG_DRY:
                pcvh = pcvh + pfrti[..., TILE_HIGH_VEG_DRY]
            if ntile > TILE_HIGH_VEG_SNOW:
                pcvh = pcvh + pfrti[..., TILE_HIGH_VEG_SNOW]

        # Snow-bottom heat flux PGSN comes from the SRFSN_LWIMP solve
        # run by _update_snow() just before this helper; see M6c.
        # (Forwarded unchanged -- sign convention already matches:
        # PGSN positive downward into the soil column.)

        # Land mask from the static descriptor (LDLAND).
        ldland = static.land_frac > 0.5

        inp = SoilTempInputs(
            ptsam1m=state.soil_t,
            pwsam1m=state.soil_w_liq,
            pfrti=pfrti,
            pahfsti=diff_out.pssh,
            pevapti=diff_out.pjq,
            pssrflti=pssrflti,
            pslrfl=pslrfl,
            # SURFTSTP_CTL either copies the grid radiation to all tiles or
            # uses tiled radiation, according to LELWTL.
            pslrflti=pslrflti,
            pgsn=pgsn,
            ksoty=static.soil_type,
            pcvl=pcvl, pcvh=pcvh,
            ldland=ldland,
            ptmst=dt,
        )
        out: SoilTempOutputs = port_soilt(
            inp, self.soil_table, self.consts, self.config,
        )
        return out.ptsa

    def _update_snow(
        self,
        *,
        state: SurfState,
        static: Static,
        diff_out,                       # DiffusionOutputs
        pssrflti: torch.Tensor,
        pslrfl: torch.Tensor,
        pslrflti: torch.Tensor | None,
        pfrti: torch.Tensor,
        forcing: Forcing,
        pevapsnw: torch.Tensor,
        ptsfc: torch.Tensor,
        ptsfl: torch.Tensor,
        dt: torch.Tensor,
        ngrid: int,
        dtype: torch.dtype,
        device,
    ) -> SnowOutputs:
        """Advance the single-layer snow prognostic by one step (M6c).

        Mirrors ``SRFSN_LWIMP``.  Builds a :class:`SnowInputs` from the t-1 snow
        state triple ``(PSSNM1M, PTSNM1M, PRSNM1M)`` + the soil top
        temperature and the per-tile turbulent / radiative fluxes
        VDFDIFH1S just produced, runs the snow solve and returns the
        :class:`SnowOutputs`.

        Upstream call order (``surftstp_ctl_mod.F90``): SRFSN_LWIMP
        outputs ``ZGSN`` / ``ZTSN`` before ``SRFTS`` consumes ``ZGSN`` as the snow-bottom heat
        flux).  The caller therefore threads ``snow_out.pgsn`` into
        :meth:`_update_soil_temperature`.

        Snowfall / rainfall throughfall (M2d): upstream ``SRFSN_LWIMP``
        consumes the snowfall fluxes ``PSSFC`` / ``PSSFL`` directly --
        these come from the cloud microphysics (``PFPLCN`` / ``PFPLSN``)
        in the fully coupled model, and ``SRFWLS`` (canopy interception)
        does **not** process snowfall (only rainfall), so the snow
        throughfall is the raw snowfall flux.  The offline driver
        therefore feeds ``forcing.snow`` straight in as ``PSSFL``
        (large-scale; convective ``PSSFC`` stays zero because the
        port's :class:`Forcing` does not split convective vs large-scale,
        matching the convention used for rainfall in
        :meth:`_update_soil_moisture`).  The rainfall throughfall
        ``PTSFC`` / ``PTSFL`` / snow-evap mismatch ``PEVAPSNW`` remain
        at the cold-start zero -- they are consumed by the soil-
        moisture solve, not the snow solve, and the single-layer snow
        path does not couple meltwater back to soil moisture.

        Mapping to the ``SnowInputs`` field list::

            PSSNM1M  (snow mass)            <- state.snow_w
            PTSNM1M  (snow T)               <- state.snow_t (or RTT cold start)
            PRSNM1M  (snow density)         <- self._snow_density(state)
            PTSAM1M_TOP (soil top T)        <- state.soil_t[:, 0]
            PFRTI    (tile fractions)       <- pfrti arg
            PAHFSTI  (tile sensible heat)   <- diff_out.pssh
            PEVAPTI  (tile moisture flux)   <- diff_out.pjq
            PSSRFLTI (tile net SW)          <- pssrflti arg
            PSLRFL   (grid net LW)          <- pslrfl arg
            PSSFL    (large-scale snowfall) <- forcing.snow / dt   (M2d)
            PSSFC                           <- 0 (no convective split)
            PTSFC/PTSFL                     <- canopy throughfall
            PEVAPSNW                        <- driver evaporation
            PTMST    (coupling interval)    <- dt arg
        """
        # Cold-start fallback for PTSNM1M: if no snow temperature is
        # carried on the state (e.g. a pure-land / no-snow initial
        # condition), pin it to RTT -- the upstream cold-start value.
        # Both branches are differentiable.
        rtt = self.consts.rtt.to(dtype=dtype, device=device)
        if state.snow_t is not None:
            ptsnm1m = state.snow_t
        else:
            ptsnm1m = torch.ones_like(state.snow_w) * rtt

        zsa_top = state.soil_t[:, 0]
        zero1d = torch.zeros((ngrid,), dtype=dtype, device=device)

        # Snowfall throughfall (M2d): forcing.snow is the *cumulative*
        # snowfall over the coupling interval (kg/m^2), so dividing by
        # dt recovers the large-scale snowfall rate (kg/m^2/s).
        # compute_snow rescales this by the interception grid fraction
        # (tiles 3..8 sum), which is ~1 over pure land, before it enters
        # the snow-mass budget ZSSTAR.
        ptmst = dt if torch.is_tensor(dt) else torch.tensor(dt, dtype=dtype, device=device)
        pssfl = forcing.snow / ptmst

        inp = SnowInputs(
            pssnm1m=state.snow_w,
            ptsnm1m=ptsnm1m,
            prsnm1m=self._snow_density(state),
            ptsam1m_top=zsa_top,
            pfrti=pfrti,
            pahfsti=diff_out.pssh,
            pevapti=diff_out.pjq,
            pssrflti=pssrflti,
            pslrfl=pslrfl,
            pslrflti=pslrflti,
            pssfc=zero1d, pssfl=pssfl,
            ptsfc=ptsfc.to(dtype=dtype, device=device),
            ptsfl=ptsfl.to(dtype=dtype, device=device),
            pevapsnw=pevapsnw.to(dtype=dtype, device=device),
            ptmst=dt,
        )
        snow_out = port_snow(inp, self.soil_table, self.consts, self.config)
        self._last_snow_inputs = inp
        self._last_snow_outputs = snow_out
        return snow_out

    def _update_canopy(
        self,
        *,
        state: SurfState,
        static: Static,
        diff_out,                       # DiffusionOutputs
        frac: torch.Tensor,
        wlmx_grid: torch.Tensor,
        ktvl: torch.Tensor,
        ktvh: torch.Tensor,
        forcing: Forcing,
        dt: torch.Tensor,
        pevapsnw: torch.Tensor,
        ngrid: int,
        dtype: torch.dtype,
        device,
    ) -> CanopyOutputs:
        """Advance the skin-reservoir / canopy prognostic by one step (M6).

        Mirrors ``SRFWL`` (``srfwl_mod.F90``).  Builds a
        :class:`CanopyInputs` from the t-1 skin-reservoir water
        (``PWLM1M`` <- ``state.skin_water``), the vegetation cover /
        interception capacity (``PCVL`` / ``PCVH`` / ``PWLMX``), the
        per-tile moisture flux VDFDIFH1S just produced
        (``PEVAPTI`` <- ``diff_out.pjq``) and the rainfall throughfall,
        runs the skin-reservoir update and returns the
        :class:`CanopyOutputs`.

        Upstream call order (``surftstp_ctl_mod.F90``): SRFWLS (:268)
        / SRFWL (:612) precedes SRFSN (:320 / :669), so the caller
        threads the throughfall ``PTSFL`` into :meth:`_update_snow`
        (the snow-rainfall interception budget) and the interception-
        layer evaporation mismatch ``PEINTTI`` into
        :meth:`_update_soil_moisture`.

        Rainfall throughfall (M2d): the port's :class:`Forcing` does
        not split convective vs large-scale precip, so all rain is
        treated as large-scale (``PRSFL`` <- ``forcing.rain / dt``;
        convective ``PRSFC`` stays zero), matching the convention used
        for the snowfall throughfall in :meth:`_update_snow`.

        Mapping to the :class:`CanopyInputs` field list::

            PWLM1M   (skin-reservoir water) <- state.skin_water
            PCVL/PCVH (veg cover)           <- corrected static SURFBC cover
            PWLMX    (interception capacity) <- wlmx_grid arg
            PFRTI    (tile fractions)       <- frac arg
            PEVAPTI  (tile moisture flux)   <- diff_out.pjq
            PRSFL    (large-scale rain)     <- forcing.rain / dt   (M2d)
            PRSFC/PEVAPSNW                   <- 0 (cold start)
            PTMST    (coupling interval)    <- dt arg
        """
        ptmst = dt if torch.is_tensor(dt) else torch.tensor(dt, dtype=dtype, device=device)
        ntile = frac.shape[-1]

        # SRFWL receives SURFBC's corrected static vegetation covers.
        # Wet-skin is a mixture of bare and vegetated surfaces, so PCVL /
        # PCVH cannot be reconstructed from the evolving tile fractions.
        from .config import (
            TILE_LOW_VEG_DRY, TILE_LOW_VEG_SNOW,
            TILE_HIGH_VEG_DRY, TILE_HIGH_VEG_SNOW,
        )
        covers = self._cover_fractions(static)
        if covers is not None:
            pcvl, pcvh, _, _, _ = covers
        else:
            pcvl = frac.new_zeros(frac.shape[:-1])
            pcvh = frac.new_zeros(frac.shape[:-1])
            if ntile > TILE_LOW_VEG_DRY:
                pcvl = pcvl + frac[..., TILE_LOW_VEG_DRY]
            if ntile > TILE_LOW_VEG_SNOW:
                pcvl = pcvl + frac[..., TILE_LOW_VEG_SNOW]
            if ntile > TILE_HIGH_VEG_DRY:
                pcvh = pcvh + frac[..., TILE_HIGH_VEG_DRY]
            if ntile > TILE_HIGH_VEG_SNOW:
                pcvh = pcvh + frac[..., TILE_HIGH_VEG_SNOW]

        # Rainfall fluxes: large-scale from forcing.rain; convective
        # zero (cold start).  PEVAPSNW is computed by SURFEXCDRIVER before
        # VDF and must be threaded through SRFWL.
        prsfl = forcing.rain / ptmst
        prsfc = torch.zeros((ngrid,), dtype=dtype, device=device)
        pevapsnw = pevapsnw.to(dtype=dtype, device=device)

        # Land mask from the static descriptor (LDLAND).
        ldland = static.land_frac > 0.5

        # Seed a cold-start skin-reservoir water content if the state
        # carries None (a fresh / pure-soil initial condition): zero
        # intercepted water, matching the upstream cold start.
        if state.skin_water is not None:
            pwlm1m = state.skin_water
        else:
            pwlm1m = torch.zeros((ngrid,), dtype=dtype, device=device)

        inp = CanopyInputs(
            ptmst=dt,
            pwlm1m=pwlm1m,
            pcvl=pcvl, pcvh=pcvh, pwlmx=wlmx_grid,
            pfrti=frac,
            pevapti=diff_out.pjq,
            prsfc=prsfc, prsfl=prsfl,
            pevapsnw=pevapsnw,
            ldland=ldland,
            lewbsoilfix=getattr(self.soil_table.consts, "lewbsoilfix", True),
        )
        return port_canopy(inp, self.veg_table)

    def _update_soil_moisture(
        self,
        *,
        state: SurfState,
        static: Static,
        diff_out,                       # DiffusionOutputs
        forcing: Forcing,
        pfrti: torch.Tensor,
        ptsfl: torch.Tensor,
        peintti: torch.Tensor,
        pmsn: torch.Tensor | None,
        pemssn: torch.Tensor | None,
        pevapsnw: torch.Tensor,
        dt: torch.Tensor,
        ngrid: int,
        dtype: torch.dtype,
        device,
    ) -> SoilMoistOutputs:
        """Advance the soil-moisture prognostic by one step (M5d-soilw).

        Builds a :class:`SoilMoistInputs` from the t-1 soil state, the
        per-tile turbulent fluxes VDFDIFH1S just produced, and the
        precipitation throughfall, runs the full HTESSEL soil-moisture
        solve (SRFWEXC_VG + SRFWDIF + SRFWINC + SRFWNG) and returns the
        full :class:`SoilMoistOutputs`.

        Mapping from forward() context to the SRFWEXC_VG dummy list:

        * ``PEVAPTI`` (tile moisture flux, kg/m^2/s) <- ``diff_out.pjq``
        * ``PFRTI``   (tile fractions)               <- ``pfrti`` arg
        * ``PTSFL``   (large-scale throughfall rate) <- ``ptsfl`` arg
          (rainfall after canopy interception, from :meth:`_update_canopy`)
        * ``PTSFC``   (convective throughfall rate)  <- zero (cold start;
          the port's :class:`Forcing` does not yet split convective vs
          large-scale precip, so all rain is treated as large-scale)
        * ``PMSN`` / ``PEMSSN``                       <- ``snow_out``
        * ``PEVAPSNW``                                 <- driver evaporation
        * ``PEINTTI`` (interception-layer evap mismatch) <- ``peintti``
          arg (from :meth:`_update_canopy`)

        Sign conventions agree: both VDFDIFH1S and SRFWEXC_VG use the
        positive-downward convention (positive PEVAPTI = dew into the
        soil, negative = evaporation removing water).  The HTESSEL
        solve clamps the result to ``[WRES, WSAT]`` per layer.

 `PSDOR` (sub-grid orography standard deviation) is read from
 :class:Static when supplied; legacy callers fall back to 0 m.
        """
        # Convective throughfall is unavailable in the offline forcing.
        ptsfc = torch.zeros((ngrid,), dtype=dtype, device=device)
        pmsn = pmsn.to(dtype=dtype, device=device) if pmsn is not None else torch.zeros((ngrid,), dtype=dtype, device=device)
        pemssn = pemssn.to(dtype=dtype, device=device) if pemssn is not None else torch.zeros((ngrid,), dtype=dtype, device=device)
        pevapsnw = pevapsnw.to(dtype=dtype, device=device)

        # PSDOR is the restart/climatology sub-grid orography standard deviation.
        psdor = static.orography_stddev.to(dtype=dtype, device=device) if static.orography_stddev is not None else torch.zeros((ngrid,), dtype=dtype, device=device)

        # Land mask from the static descriptor (LDLAND).
        ldland = static.land_frac > 0.5
        ktvl, ktvh = self._dominant_veg_types(static)

        inp = SoilMoistInputs(
            pwsam1m=state.soil_w_liq,
            ptsam1m=state.soil_t,
            pfrti=pfrti,
            pevapti=diff_out.pjq,
            ptsfc=ptsfc, ptsfl=ptsfl,
            pmsn=pmsn, pemssn=pemssn, pevapsnw=pevapsnw, peintti=peintti,
            ksoty=static.soil_type,
            ktvl=ktvl, ktvh=ktvh,
            psdor=psdor, ldland=ldland,
            ptmst=dt,
        )
        out: SoilMoistOutputs = port_soilw(
            inp, self.soil_table, self.veg_table, self.consts, self.config,
        )
        return out

    def _update_lake(self, state: SurfState, diff_out, ngrid: int,
                     dtype: torch.dtype, device) -> LakeOutputs:
        """Advance the FLake lake prognostics by one step (M6 stub).

        Builds a :class:`LakeInputs` from the lake-tile surface fluxes
        (zeros on the cold start -- the offline driver has no lake tile
        when ``use_flake`` is ``False``) and the t-1 lake state, then
        invokes the FLake driver.  The stub returns the t-1 state
        unchanged, matching the upstream ``LEFLAKE = .FALSE.`` branch
        (surftstp_ctl_mod.F90:808-818); the real FLake body is a future
        drop-in.

        The forward path does not yet carry the seven FLake prognostics
        on :class:`SurfState`, so this method seeds them from the
        available skin / SST fields as a cold-start placeholder.  When
        the prognostics land on :class:`SurfState` the seeding block is
        deleted and the inputs read straight off ``state``.
        """
        # The stub ignores the timestep; carry it for the real driver.
        ptstep = 3600.0

        # Cold-start lake prognostics -- identity update means the
        # returned values equal these regardless, so the placeholder
        # choice (SST) is harmless.  Real prognostics land on SurfState
        # when the full FLake is ported.
        sst = state.sst if state.sst is not None else \
            torch.full((ngrid,), 283.0, dtype=dtype, device=device)
        zeros = torch.zeros((ngrid,), dtype=dtype, device=device)

        inp = LakeInputs(
            ptlice_m1m=sst.clone(), ptlmnw_m1m=sst.clone(),
            ptlwml_m1m=sst.clone(), ptlbot_m1m=sst.clone(),
            ptlsf_m1m=sst.clone(),
            phlice_m1m=zeros.clone(), phlml_m1m=zeros.clone(),
            pssrflti=zeros.clone(), pslrfl=zeros.clone(),
            pahfsti=zeros.clone(), pevapti=zeros.clone(),
            pustrti=zeros.clone(), pvstrti=zeros.clone(),
            pldepth=torch.full((ngrid,), 10.0, dtype=dtype, device=device),
            pgemu=torch.full((ngrid,), 0.5, dtype=dtype, device=device),
            ld_lake=torch.zeros((ngrid,), dtype=torch.bool, device=device),
            ptstep=ptstep,
        )
        return port_flake(inp)

    def _update_ocean_ml(self, state: SurfState, diff_out, ngrid: int,
                         dtype: torch.dtype, device) -> OceanMlOutputs:
        """Advance the ocean mixed-layer prognostics by one step (M6 stub).

        Builds an :class:`OceanMlInputs` from the water-tile surface
        fluxes (zeros on the cold start) and the t-1 ocean temperature
        (seeded from SST), then invokes the ocean-ML driver.  The stub
        returns the t-1 state unchanged, matching the upstream
        ``LEOCSA = .FALSE.`` default (prescribed SST); the real KPP body
        is a future drop-in.

        As with :meth:`_update_lake`, the 3-D ocean prognostics are not
        yet on :class:`SurfState`; SST seeds the surface layer and the
        deeper layers are held at the same value (irrelevant for the
        identity stub).
        """
        sst = state.sst if state.sst is not None else \
            torch.full((ngrid,), 290.0, dtype=dtype, device=device)
        nlev_ocean = 8
        pto0 = sst.unsqueeze(-1).expand(ngrid, nlev_ocean).clone()
        pso0 = torch.full((ngrid, nlev_ocean), 35.0, dtype=dtype, device=device)
        puo0 = torch.zeros((ngrid, nlev_ocean), dtype=dtype, device=device)
        pvo0 = torch.zeros((ngrid, nlev_ocean), dtype=dtype, device=device)
        pzo = torch.linspace(0.0, 50.0, nlev_ocean + 1,
                            dtype=dtype, device=device).unsqueeze(0).expand(
                                ngrid, nlev_ocean + 1).clone()
        pdo = torch.full((ngrid, nlev_ocean),
                         50.0 / nlev_ocean, dtype=dtype, device=device)
        zeros = torch.zeros((ngrid,), dtype=dtype, device=device)

        inp = OceanMlInputs(
            pto0=pto0, pso0=pso0, puo0=puo0, pvo0=pvo0,
            pzo=pzo, pdo=pdo,
            pssrflti=zeros.clone(), pslrfl=zeros.clone(),
            pahfsti=zeros.clone(), pevapti=zeros.clone(),
            pustrti=zeros.clone(), pvstrti=zeros.clone(),
            ld_ocn_kpp=torch.zeros((ngrid,), dtype=torch.bool, device=device),
            ptstep=3600.0,
        )
        return port_ocean_ml(inp)

    def _interception_capacity(
        self,
        static: Static,
        frac: torch.Tensor,
        ktvl: torch.Tensor,
        ktvh: torch.Tensor,
    ) -> torch.Tensor:
        """Per-column maximum interception reservoir PWLMX (kg/m^2).

        Mirrors ``surfbc_ctl_mod.F90`` 258-260:

            PWLMX = RWLMAX * (1 - CVL - CVH + CVL*LAIL + CVH*LAIH)

        where CVL / CVH are the low / high vegetation cover fractions and
        LAIL / LAIH the cover-weighted leaf-area indices.  Two paths:

        * When ``Static`` carries the canonical cover inputs (``cvl_frac``
          / ``cvh_frac``), PWLMX is computed directly from them and their
          PLAIL / PLAIH -- the faithful SURFBC formula.  This is the
          same PWLMX :meth:`_compute_tile_frac` uses for ZCVW, so the
          canopy clamp and the tile split stay mutually consistent.

        * Otherwise CVL / CVH are reconstructed from the active tile
          fractions (low-veg dry + low-veg snow, high-veg dry + high-veg
          snow) and the dominant veg type's LAI stands in for both --
          the cold-start approximation.  The result only feeds the
          KTILE=3 interception clamp, which fires when
          ``ZEP = PCFQ*(q_air-qs) < 0`` (dew / frost), so a coarse WLMX
          has negligible effect on the SEB alphas outside that regime.
        """
        rwlmax = self.soil_table.consts.rwlmax
        # Canonical cover path: use the SURFBC inputs directly so PWLMX
        # matches the ZCVW the tile split computes.
        covers = self._cover_fractions(static)
        if covers is not None:
            cvl, cvh, _pcil, plail, plaih = covers
            return rwlmax * (1.0 - cvl - cvh + cvl * plail + cvh * plaih)
        # Static-path reconstruction from the tile fractions.
        from .config import (
            TILE_LOW_VEG_DRY, TILE_LOW_VEG_SNOW,
            TILE_HIGH_VEG_DRY, TILE_HIGH_VEG_SNOW,
        )
        ntile = frac.shape[-1]
        # Cover fractions: clamp the indices to the active tile count so
        # this works for both 8 and 10-tile configs.
        cvl = frac.new_zeros(frac.shape[:-1])
        cvh = frac.new_zeros(frac.shape[:-1])
        if ntile > TILE_LOW_VEG_DRY:
            cvl = cvl + frac[..., TILE_LOW_VEG_DRY]
        if ntile > TILE_LOW_VEG_SNOW:
            cvl = cvl + frac[..., TILE_LOW_VEG_SNOW]
        if ntile > TILE_HIGH_VEG_DRY:
            cvh = cvh + frac[..., TILE_HIGH_VEG_DRY]
        if ntile > TILE_HIGH_VEG_SNOW:
            cvh = cvh + frac[..., TILE_HIGH_VEG_SNOW]
        # Dominant-type LAI for low and high veg (RVLAI, 0..20 indexed).
        lai_low = self.veg_table.gather("rvlai", ktvl)
        lai_high = self.veg_table.gather("rvlai", ktvh)
        return rwlmax * (1.0 - cvl - cvh + cvl * lai_low + cvh * lai_high)

    def _dominant_veg_types(self, static: Static) -> Tuple[torch.Tensor, torch.Tensor]:
        """Low / high vegetation type per column (long).

        Returns ``(veg_type_low, veg_type_high)``.  When ``Static``
        carries the explicit ``veg_type_low`` / ``veg_type_high`` split
        (the faithful SURFBC path -- SURFBC keys separate low / high veg
        off PCVL / PCVH cover), both are returned directly.  Otherwise
        the single ``veg_type`` field is returned for both -- the
        cold-start dominant-type approximation the VegTable lookups
        (RVTRSR / RVLAMSK / RVZ0M, which differ between indices 1..20)
        tolerate.
        """
        vl = static.veg_type_low if static.veg_type_low is not None \
            else static.veg_type
        vh = static.veg_type_high if static.veg_type_high is not None \
            else static.veg_type
        return vl, vh

    def _snow_density(self, state: SurfState) -> torch.Tensor:
        """Snow density PRSN (kg/m^3).

        Prefers the prognostic ``state.snow_rho`` (PRSNM1M) when present;
        otherwise falls back to the offline driver's 300 kg/m^3 fresh-
        snow cold-start value.  This is the *t-1* reader (PRSNM1M) used
        wherever the snow solve consumes the old density (VLAMSK,
        SRFSN_LWIMP).  The *evolved* density at t is produced by
        :meth:`_update_snow_density` (the ``SRFSN_LWIMP`` section-3.1
        port), which is called after the snow solve and written back to
        ``new_state.snow_rho``.
        """
        if state.snow_rho is not None:
            return state.snow_rho
        return torch.full_like(state.snow_w, 300.0)

    def _update_snow_density(
        self,
        *,
        state: SurfState,
        snow_out: SnowOutputs,
        forcing: Forcing,
        frac: torch.Tensor,
        dt: torch.Tensor,
        dtype: torch.dtype,
        device,
    ) -> torch.Tensor:
        """Evolve the snow density by one step (``SRFSN_LWIMP`` section 3.1).

        Runs the single-layer density-evolution port
        (:func:`surf_pytorch.physics.snow_density.compute_snow_density`)
        on the just-completed snow solve outputs.  The density block is
        the density evolution following ``SRFSN_LWIMP`` advances ``PRSNM1M`` using
        the fresh-snow density, depth-weighted mixing, overburden /
        metamorphism compaction and the melt-compaction term, and resets
        the density to ``RHOMINSN`` at no-snow points.

        Inputs come from the t-1 snow state (``PSSNM1M``, ``PRSNM1M``,
        ``PTSNM1M``), the new snow temperature ``PTSN`` and tile
        fraction / mask from the snow solve, and the surface forcing
        (lowest-level air temperature ``PTSRF``, surface wind
        ``PUSRF`` / ``PVSRF`` from ``forcing.u`` / ``forcing.v``, and the
        large-scale + convective snowfall ``PSSFL`` / ``PSSFC``).

        Mapping to the :class:`SnowDensityInputs` field list::

            PSSNM1M  (snow mass)         <- state.snow_w
            PRSNM1M  (snow density)      <- self._snow_density(state)
            PTSNM1M  (snow T)            <- state.snow_t (or RTT cold start)
            PTSN     (new snow T)        <- snow_out.ptsn
            PTSRF    (lowest-level air T)<- forcing.t
            PUSRF/PVSRF (surface wind)   <- forcing.u / forcing.v
            PSSFL    (large-scale snow)  <- forcing.snow / dt
            PSSFC/PTSFC/PTSFL/PEVAPSNW   <- 0 (cold start)
            ZFRSN    (snow tile frac)    <- snow_out.zfrsn
            LLNOSNOW (no-snow mask)      <- snow_out.llnosnow
            PTMST    (coupling interval) <- dt
        """
        rtt = self.consts.rtt.to(dtype=dtype, device=device)
        if state.snow_t is not None:
            ptsnm1m = state.snow_t
        else:
            ptsnm1m = torch.ones_like(state.snow_w) * rtt

        # SRFSN_LWIMP receives PTSRF as the lowest-level air temperature
        # from SURFTSTP (PT(:,KLEV)), not a tile-weighted skin temperature.
        # This controls the density assigned to fresh snowfall.
        ptsrf = forcing.t
        # No-snow mask: fall back to the snow-fraction test if the snow
        # solve did not populate LLNOSNOW (defensive; the solve always
        # sets it).
        if snow_out.llnosnow is not None:
            llnosnow = snow_out.llnosnow
        else:
            llnosnow = snow_out.zfrsn < float(self.soil_table.consts.rfrsmall)

        ptmst = dt if torch.is_tensor(dt) else torch.tensor(dt, dtype=dtype, device=device)
        zero1d = torch.zeros_like(state.snow_w)
        pssfl = forcing.snow / ptmst

        dinp = SnowDensityInputs(
            pssnm1m=state.snow_w,
            prsnm1m=self._snow_density(state),
            ptsnm1m=ptsnm1m,
            ptsn=snow_out.ptsn,
            ptsrf=ptsrf,
            pusrf=forcing.u,
            pvsrf=forcing.v,
            pssfl=pssfl, pssfc=zero1d,
            zfrsn=snow_out.zfrsn,
            llnosnow=llnosnow,
            ptmst=dt,
        )
        return port_snow_density(
            dinp, self.soil_table, self.consts,
            differentiable=self.config.differentiable,
        ).prsn

    def _update_snow_albedo(
        self,
        *,
        state: SurfState,
        static: Static,
        snow_out: SnowOutputs,
        forcing: Forcing,
        dt: torch.Tensor,
        dtype: torch.dtype,
        device,
    ) -> Optional[torch.Tensor]:
        """Evolve the snow albedo by one step.

        Runs the snow-albedo aging port
        (:func:`surf_pytorch.physics.snow_albedo.compute_snow_albedo`)
        on the just-completed snow solve outputs.  The aging block is the
        ``SRFSN_LWIMP`` albedo update advances ``PASNM1M`` using the
        melt exponential decay (RTAUF), the cold linear decay (RTAUA),
        and (for the legacy path) the fresh-snow reset (RSFRESH) or (for
        the SRFSN_ASN path) the linear snowfall brightening, plus the
        no-snow reset to ``RALFMAXSN``.

        **Activation gate**: returns ``None`` when ``state.snow_alb`` is
        unset, leaving the albedo subsystem dormant (forward() is then
        bit-identical to the build before the albedo work).  Once a caller
        cold-starts ``snow_alb`` (typically ``RALFMAXSN = 0.85``) the
        subsystem stays active for every subsequent step.

        **Glacier branch**: when ``static.ice_land_frac`` (PCIL) is
        supplied, the full SRFSN_ASN scheme is activated (land + glacier
        parts blended by PCIL).  When PCIL is absent the legacy pure-
        land-snow path runs bit-for-bit.

        Mapping to the :class:`SnowAlbedoInputs` field list::

            PASNM1M     (snow albedo)      <- state.snow_alb
            PMSN        (melt rate)        <- snow_out.pmsn (None -> zeros)
            ZSSFL+ZSSFC (total snowfall)   <- forcing.snow / dt (ZSSFC = 0)
            LLNOSNOW    (no-snow mask)     <- snow_out.llnosnow
            PTMST       (coupling interval)<- dt
            PCIL        (land-ice frac)    <- static.ice_land_frac (None->legacy)
            PTSNM1M(:,1)(top snow T)       <- state.snow_t (None -> RTT)
            PSNOWF      (snowfall rate)    <- forcing.snow / dt
            LDNH        (NH mask)          <- static.lat > 0 (None -> SH)
        """
        if state.snow_alb is None:
            return None

        # No-snow mask: fall back to the snow-fraction test if the snow
        # solve did not populate LLNOSNOW (defensive; the solve always
        # sets it).
        if snow_out.llnosnow is not None:
            llnosnow = snow_out.llnosnow
        else:
            llnosnow = snow_out.zfrsn < float(self.soil_table.consts.rfrsmall)

        ptmst = dt if torch.is_tensor(dt) else torch.tensor(dt, dtype=dtype, device=device)
        pssflpssfc = forcing.snow / ptmst
        # Melt rate PMSN: zero where the melt branch is inactive (the
        # snow solve exports it already-masked; fall back to zeros if not).
        pmsn = snow_out.pmsn if snow_out.pmsn is not None else \
            torch.zeros_like(state.snow_alb)
        pmsnint = snow_out.pmsnint if snow_out.pmsnint is not None else \
            torch.zeros_like(state.snow_alb)

        # Glacier-branch fields (only wired when PCIL is supplied -- the
        # sentinel that activates the full SRFSN_ASN scheme).
        pcil = static.ice_land_frac.to(dtype=dtype, device=device) \
            if static.ice_land_frac is not None else None
        if pcil is not None:
            ptsnm1m_top = state.snow_t.to(dtype=dtype, device=device) \
                if state.snow_t is not None else None
            psnowf = pssflpssfc.to(dtype=dtype, device=device)
            ldnh = (static.lat.to(dtype=dtype, device=device) > 0.0) \
                if static.lat is not None else None
        else:
            ptsnm1m_top = None
            psnowf = None
            ldnh = None

        ainp = SnowAlbedoInputs(
            pasnm1m=state.snow_alb,
            pmsn=pmsn,
            pmsnint=pmsnint,
            pssflpssfc=pssflpssfc,
            llnosnow=llnosnow,
            ptmst=dt,
            pcil=pcil,
            ptsnm1m_top=ptsnm1m_top,
            psnowf=psnowf,
            ldnh=ldnh,
        )
        return port_snow_albedo(
            ainp, self.soil_table, self.consts,
            differentiable=self.config.differentiable,
        ).pasn

    def _surface_albedo(
        self,
        state: SurfState,
        static: Static,
        forcing: Forcing,
        ntile: int,
        dtype: torch.dtype,
        device,
    ) -> torch.Tensor:
        """Per-tile broadband surface albedo alpha ``(Ngrid, ntile)``.

        Two-tier mapping (a simplification of the SURFRAD tile albedo
        scheme, ``surfrad_ctl_mod.F90`` 527-660):

        * **Low-vegetation snow tile** (TILE_LOW_VEG_SNOW) takes the snow
          albedo ``PASN`` (t-1 ``state.snow_alb``).
        * **High-vegetation snow tile** (TILE_HIGH_VEG_SNOW) takes the
          SURFRAD snow-under-forest albedo table, keyed by high vegetation
          type, rather than the open-snow ``PASN``.
        * **Wet-skin tile** (TILE_WET_SKIN) takes the wet background
          albedo ``albw``, falling back to the dry ``albd`` then 0.
        * **Dry vegetation / bare-soil tiles** take the dry background
          albedo ``albd``, falling back to 0.
        * **Urban tile** (TILE_URBAN), when the urban cover ``PCUR`` is
          supplied, takes the canyon albedo from :meth:`_urban_canyon_albedo`
          (Masson/Harman/Porson, ``canalb_mod.F90``), blended with the
          roof albedo via the road ratio ``RWRR`` (``surfrad_ctl_mod.F90``
          597-602).
        * Water / ice tiles use explicit surface albedo fields when
          available, otherwise fall back to the supplied background
          albedo and only then to 0.

        **Activation gate**: when ``state.snow_alb`` is ``None`` AND
        ``static.albd`` / ``static.albw`` are both ``None`` AND no urban
        tile is active, returns an all-zeros tensor so the absorbed SW is
        unchanged -- the bit-identical fallback that keeps existing tests
        / golden snapshots valid.  Otherwise the supplied fields drive the
        alpha.

        The snow albedo used here is the **t-1** value (``state.snow_alb``,
        the step-start field), consistent with the t-1 skin temperature
        used for radiation; the freshly-evolved PASN is written to
        ``new_state.snow_alb`` in section 7 and takes effect next step.

        Note: the urban-snow adjustment (Jarvi 2014, ``surfrad_ctl_mod``
        667-672) is NOT included -- it requires mixing the snow-tile
        albedo into the urban tile, which depends on the snow tile
        fractions.  It is a future refinement.
        """
        ngrid = state.skin_t.shape[0]
        has_snow = state.snow_alb is not None
        has_bg = static.albd is not None or static.albw is not None
        has_urban = (ntile > TILE_URBAN) and (static.urban_frac is not None)
        if not has_snow and not has_bg and not has_urban:
            return torch.zeros(
                (ngrid, ntile), dtype=dtype, device=device,
            )

        alpha = torch.zeros((ngrid, ntile), dtype=dtype, device=device)

        # Background / explicit surface albedo fields (broadcast
        # (Ngrid,) -> per-tile when present).
        albd = static.albd.to(dtype=dtype, device=device) \
            if static.albd is not None else None
        albw = static.albw.to(dtype=dtype, device=device) \
            if static.albw is not None else None
        albwater = getattr(static, "water_alb", None)
        albice = getattr(static, "ice_alb", None)
        if albwater is not None:
            albwater = albwater.to(dtype=dtype, device=device)
        if albice is not None:
            albice = albice.to(dtype=dtype, device=device)

        # Water and ice tiles are physical surfaces in the Fortran
        # driver, so do not leave them at zero when a dedicated field or
        # a reasonable background fallback is available.
        if ntile > TILE_WATER:
            if albwater is not None:
                alpha[:, TILE_WATER] = albwater
            elif albw is not None:
                alpha[:, TILE_WATER] = albw
            elif albd is not None:
                alpha[:, TILE_WATER] = albd
        if ntile > TILE_ICE:
            if albice is not None:
                alpha[:, TILE_ICE] = albice
            elif albw is not None:
                alpha[:, TILE_ICE] = albw
            elif albd is not None:
                alpha[:, TILE_ICE] = albd

        # Low-vegetation snow: PASN (t-1).  Snow under high vegetation is
        # a forest-canopy snow albedo from SURFRAD, not the open-snow PASN.
        if has_snow:
            pasn = state.snow_alb.to(dtype=dtype, device=device)
            if ntile > TILE_LOW_VEG_SNOW:
                alpha[:, TILE_LOW_VEG_SNOW] = pasn
            if ntile > TILE_HIGH_VEG_SNOW:
                alpha[:, TILE_HIGH_VEG_SNOW] = self._snow_forest_albedo(
                    static, dtype, device
                )

        # Background albedo for the non-snow tiles.  The wet-skin tile
        # prefers albw (falls back to albd then 0); the dry vegetation
        # and bare-soil tiles use albd (falls back to 0).
        if has_bg:
            if ntile > TILE_WET_SKIN:
                wet = albw if albw is not None else albd
                if wet is not None:
                    alpha[:, TILE_WET_SKIN] = wet
            if albd is not None:
                if ntile > TILE_LOW_VEG_DRY:
                    alpha[:, TILE_LOW_VEG_DRY] = albd
                if ntile > TILE_HIGH_VEG_DRY:
                    alpha[:, TILE_HIGH_VEG_DRY] = albd
                if ntile > TILE_BARE_SOIL:
                    alpha[:, TILE_BARE_SOIL] = albd

        # Urban tile (TILE_URBAN): canyon albedo from CANALB, blended
        # with the roof albedo via the road ratio (surfrad 597-602).
        if has_urban:
            alpha[:, TILE_URBAN] = self._urban_canyon_albedo(
                forcing, static, dtype, device,
            )
        return alpha

    def _tile_shortwave(
        self,
        forcing: Forcing,
        static: Static,
        state: SurfState,
        frac: torch.Tensor,
        ntile: int,
        dtype: torch.dtype,
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return SURFRAD tile albedo and net shortwave flux.

        SURFEXCDRIVER_CTL constructs ``PSSRFLTI`` before the tile VEXCS
        calls. Keeping this calculation shared prevents VSURF from seeing
        a grid-mean radiation boundary while the energy balance sees tiles.
        """
        alpha_t = self._surface_albedo(
            state, static, forcing, ntile, dtype, device,
        )
        srfd_grid = forcing.fr_so.to(dtype=dtype, device=device) / forcing.dt
        grid_albedo = (frac * alpha_t).sum(dim=1)
        ssrfl_grid = srfd_grid * (1.0 - grid_albedo)
        ssrfl_raw = (
            (1.0 - alpha_t)
            / (1.0 - grid_albedo).unsqueeze(1)
            * ssrfl_grid.unsqueeze(1)
        )
        ssrfl_capped = torch.minimum(
            ssrfl_raw,
            torch.as_tensor(700.0, dtype=dtype, device=device),
        )
        high_ssr = (ssrfl_raw > 700.0).any(dim=1)
        ssrfl_sum = (frac * ssrfl_capped).sum(dim=1)
        ssrfl_rescaled = ssrfl_capped * (
            ssrfl_grid / ssrfl_sum
        ).unsqueeze(1)
        ssrfl_t = torch.where(
            high_ssr.unsqueeze(1), ssrfl_rescaled, ssrfl_capped,
        )
        if forcing.tiled_shortwave is not None:
            external = forcing.tiled_shortwave
            expected = (state.skin_t.shape[0], ntile)
            if tuple(external.shape) != expected:
                raise ValueError(
                    f"Forcing.tiled_shortwave must have shape {expected}, "
                    f"got {tuple(external.shape)}"
                )
            ssrfl_t = external.to(dtype=dtype, device=device)
        return alpha_t, ssrfl_t

    @staticmethod
    def _tile_downward_shortwave(
        ssrfl_t: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """Recover tile downward SW passed to VSURF from net tile SW."""
        denom = 1.0 - alpha_t
        return torch.where(
            denom > torch.finfo(ssrfl_t.dtype).eps,
            ssrfl_t / denom,
            torch.zeros_like(ssrfl_t),
        )

    def _snow_forest_albedo(
        self,
        static: Static,
        dtype: torch.dtype,
        device,
    ) -> torch.Tensor:
        """SURFRAD broadband albedo for snow under high vegetation."""
        ngrid = static.tile_frac_base.shape[0]
        # susrad_mod.F90 initialises RALB_SNOW_FOREST(type, uv/nir).
        # The offline two-band setup uses an effective UV/Vis solar weight
        # of 0.45976, inferred from the Fortran SURFRAD PALBTI output.
        uv_weight = 0.45976
        default_uv = 0.15 * 1.10598
        default_nir = 0.15 * 0.90981
        uv = torch.full((21,), default_uv, dtype=dtype, device=device)
        nir = torch.full((21,), default_nir, dtype=dtype, device=device)
        for veg_type, uv_val, nir_val in (
            (3, 0.31, 0.24),
            (4, 0.39, 0.27),
            (5, 0.35, 0.27),
            (6, 0.44, 0.33),
            (18, 0.32, 0.25),
            (19, 0.32, 0.25),
        ):
            uv[veg_type] = uv_val
            nir[veg_type] = nir_val
        if static.veg_type_high is None:
            idx = torch.zeros((ngrid,), dtype=torch.long, device=device)
        else:
            idx = static.veg_type_high.to(device=device, dtype=torch.long)
        idx = torch.clamp(idx, 0, 20)
        return uv_weight * uv[idx] + (1.0 - uv_weight) * nir[idx]

    def _urban_canyon_albedo(
        self,
        forcing: Forcing,
        static: Static,
        dtype: torch.dtype,
        device,
    ) -> torch.Tensor:
        """Urban tile albedo (canyon + roof blend, ``surfrad_ctl_mod`` 597-602).

        Computes the canyon albedo ``PCANALB`` via the CANALB port
        (:mod:`surf_pytorch.physics.canalb`) from the solar zenith
        (``forcing.mu0``) and the YOS_URB canyon material/geometry, then
        blends it with the roof albedo ``RROOALB`` via the road-to-total
        ratio ``RWRR``::

            alpha_urban = RROOALB/(1+RWRR) + PCANALB * RWRR/(1+RWRR)

        The canyon fields fall back to the YOS_URB defaults
        (RHWR=1.0, RWRR=1.0, roof/road/wall albedo 0.16/0.05/0.6) when
        not supplied on ``Static``.  Returns the per-grid urban-tile
        albedo ``(Ngrid,)``.
        """
        ngrid = forcing.mu0.shape[0]

        def _f(field, default):
            if field is None:
                return torch.full(
                    (ngrid,), default, dtype=dtype, device=device,
                )
            return field.to(dtype=dtype, device=device)

        rwrr = _f(static.urban_road_ratio, 1.0)
        rhwr = _f(static.urban_height_ratio, 1.0)
        rroalb = _f(static.roof_alb, 0.16)
        rroaalb = _f(static.road_alb, 0.05)
        rwalalb = _f(static.wall_alb, 0.6)
        mu0 = forcing.mu0.to(dtype=dtype, device=device)

        # Canyon albedo (canalb_mod.F90).
        cinp = CanalbInputs(mu0=mu0, rwalalb=rwalalb, rroaalb=rroaalb, rhwr=rhwr)
        pcanalb = port_canalb(
            cinp, differentiable=self.config.differentiable,
        ).pcanalb

        # Roof + canyon blend (surfrad_ctl_mod.F90:597-599).
        denom = 1.0 + rwrr
        return rroalb / denom + pcanalb * (rwrr / denom)



    def _effective_lai(
        self,
        static: Static,
        ktvl: torch.Tensor,
        ktvh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the LAI fields selected by SURFBC's LELAIV switch."""
        if self.config.lelaiv:
            lai_low = static.lai_low
            lai_high = static.lai_high
            if lai_low is None:
                lai_low = self.veg_table.gather("rvlai", ktvl)
            if lai_high is None:
                lai_high = self.veg_table.gather("rvlai", ktvh)
            return lai_low, lai_high
        return (
            self.veg_table.gather("rvlai", ktvl),
            self.veg_table.gather("rvlai", ktvh),
        )

    def _cover_fractions(self, static: Static):
        """Return corrected SURFBC cover inputs (PCVL, PCVH, PCIL, PLAIL, PLAIH).

        ``Static.cvl_frac`` / ``cvh_frac`` carry the raw climatological
        covers PCVLC / PCVHC.  SURFBC first applies the PFT-dependent
        RVCOV correction before it builds PFRTI; doing that here avoids
        requiring callers to pre-correct cover fractions. Returns a
        5-tuple of ``(Ngrid,)`` tensors (PCIL / PLAIL / PLAIH fall back
        to zero / dominant-LAI when unset, so a caller can set just
        ``cvl_frac`` / ``cvh_frac`` for a minimal split).  Returns
        ``None`` when the cover fields are absent, signalling the caller
        to keep the ``tile_frac_base`` static approximation.
        """
        if static.cvl_frac is None or static.cvh_frac is None:
            return None
        ngrid = static.cvl_frac.shape[0]
        dtype = static.cvl_frac.dtype
        device = static.cvl_frac.device
        pcil = static.ice_land_frac if static.ice_land_frac is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)
        ktvl, ktvh = self._dominant_veg_types(static)
        # SURFBC overwrites climatological LAI with RVLAI when LELAIV is
        # false (the offline OSM default).  Reading Mlail/Mlaih unconditionally
        # here made PWLMX, wet-skin partitioning, and canopy interception
        # inconsistent with the Fortran path.
        plail, plaih = self._effective_lai(static, ktvl, ktvh)
        pcvl = static.cvl_frac * self.veg_table.gather("rvcov", ktvl)
        pcvh = static.cvh_frac * self.veg_table.gather("rvcov", ktvh)
        return pcvl, pcvh, pcil, plail, plaih

    def _compute_tile_frac(
        self,
        state: SurfState,
        static: Static,
        ntile: int,
    ) -> torch.Tensor:
        """Faithful SURFBC land-branch tile split (``surfbc_ctl_mod.F90`` 328-399).

        Builds PFRTI(1..ntile) from the canonical cover inputs (PCVL /
        PCVH / PCIL), the snow-cover fraction ZCVS (``state.snow_frac``)
        and the wet-skin ratio ZCVW (``state.skin_water / PWLMX``).  This
        is the actual tile-fraction computation SURFBC runs every step;
        the ``tile_frac_base`` static array used elsewhere is only a
        cold-start approximation of it.

        Ported land-branch equations (``surfbc_ctl_mod.F90`` 368-377,
        plus the urban branch at 385-396 when ``static.urban_frac``
        (PCUR) is supplied)::

            ZFRTIT   = clamp(PCIL, 0, 1)                 # land-ice tile
            ZPCVL    = PCVL * (1 - ZFRTIT)               # low-veg cover
            ZPCVH    = PCVH * (1 - ZFRTIT)               # high-veg cover
            ZPCVB    = 1 - ZPCVL - ZPCVH - ZFRTIT        # bare-soil cover
            PWLMX    = RWLMAX*(1-PCVL-PCVH+PCVL*PLAIL+PCVH*PLAIH)
            ZCVW     = clamp(PWLM1M / PWLMX, 0, 1)
            PFRTI(2) = ZFRTIT * (1 - ZCVS)               # ice, snow-free
            PFRTI(3) = ZCVW * (1 - ZCVS) * (ZPCVL+ZPCVH+ZPCVB)   # wet-skin
            PFRTI(4) = ZPCVL * (1 - ZCVS) * (1 - ZCVW)   # low-veg dry
            PFRTI(5) = ZCVS * (ZPCVB + ZPCVL + ZFRTIT)   # snow on low/bare/ice
            PFRTI(6) = ZPCVH * (1 - ZCVS) * (1 - ZCVW)   # high-veg dry
            PFRTI(7) = ZCVS * ZPCVH                       # snow under high veg
            PFRTI(8) = ZPCVB * (1 - ZCVS) * (1 - ZCVW)   # bare soil

        **Urban branch** (activated when ``static.urban_frac`` is set and
        ``ntile > TILE_URBAN``).  The land covers are re-rescaled to
        leave room for the urban fraction, and PCUR enters the wet-skin
        and snow-on-low pools; a dedicated tile 10 carries the
        snow-free, dry urban canopy (``surfbc_ctl_mod.F90`` 385-396)::

            PCUR supplied (ZPCVLK = 0, no lake):
            ZPCVL = PCVL * (1 - PCUR - ZFRTIT)
            ZPCVH = PCVH * (1 - PCUR - ZFRTIT)
            ZPCVB = 1 - ZPCVL - ZPCVH - PCUR - ZFRTIT
            PFRTI(3)  = ZCVW*(1-ZCVS)*(ZPCVL+ZPCVH+ZPCVB+PCUR)  # +PCUR
            PFRTI(5)  = ZCVS*(ZPCVB+ZPCVL+ZFRTIT)               # low/bare/ice snow
            PFRTI(7)  = ZCVS*(ZPCVH+PCUR)                       # urban snow joins high-snow pool
            PFRTI(10) = PCUR*(1-ZCVS)*(1-ZCVW)                  # urban

        The sum-to-one invariant is preserved by construction (the PCUR
        cover removed from the dry low/high/bare pools reappears in the
        wet-skin pool, the snow-on-low pool, and tile 10).

        Tile 1 (water) is left at zero (pure-land offline driver); the
        optional lake column is zero (FLake not ported).

        Args:
            state:  the prognostic state at the *start* of the step
                    (snow_frac -> ZCVS, skin_water -> PWLM1M).
            static: the static descriptor carrying the cover inputs.
            ntile:  active tile count (8 or 10).

        Returns:
            ``(Ngrid, ntile)`` tile fractions summing to 1 over the land
            tiles (tile 1 / lake / urban columns zero in the pure-land
            default).
        """
        covers = self._cover_fractions(static)
        assert covers is not None, (
            "_compute_tile_frac requires cover inputs on Static "
            "(set static.cvl_frac / cvh_frac)"
        )
        pcvl, pcvh, pcil, plail, plaih = covers
        ngrid = pcvl.shape[0]
        dtype = pcvl.dtype
        device = pcvl.device

        # Snow cover fraction ZCVS <- state.snow_frac (already clamped
        # 0..1 by the snow solve; fall back to zero on a fresh state).
        # Match SURFBC's machine-precision no-snow branch: a round-off
        # residue from the CPG tendency update is not physical snow cover
        # and must not select a separate snow-tile thermal trajectory.
        zcvs = state.snow_frac if state.snow_frac is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)
        zcvs = torch.clamp(zcvs, 0.0, 1.0)
        if state.snow_w is not None:
            snow_w = state.snow_w.to(dtype=dtype, device=device)
            snow_zero = snow_w.abs() <= torch.finfo(dtype).eps * torch.maximum(
                torch.ones_like(snow_w), snow_w.abs()
            )
            zcvs = torch.where(snow_zero, torch.zeros_like(zcvs), zcvs)

        # Interception capacity PWLMX (surfbc_ctl_mod.F90:308-310) and
        # wet-skin fraction ZCVW <- PWLM1M / PWLMX (:311).
        rwlmax = self.soil_table.consts.rwlmax
        pwlmx = rwlmax * (1.0 - pcvl - pcvh + pcvl * plail + pcvh * plaih)
        pwlmx = torch.clamp(pwlmx, min=torch.finfo(dtype).tiny)
        pwlm1m = state.skin_water if state.skin_water is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)
        zcvw = torch.clamp(pwlm1m / pwlmx, 0.0, 1.0)
        # Apply the same CPG round-off treatment as ZCVS above. A residual
        # wet-skin store at machine precision has no physical area, and a
        # CPU/GPU disagreement on its exact sign must not retain a separate
        # wet-skin exchange trajectory on only one platform.
        skin_zero = pwlm1m.abs() <= torch.finfo(dtype).eps * torch.maximum(
            torch.ones_like(pwlm1m), pwlm1m.abs()
        )
        zcvw = torch.where(skin_zero, torch.zeros_like(zcvw), zcvw)

        # Land-ice tile fraction ZFRTIT and the residual cover split
        # (surfbc_ctl_mod.F90:349,369-371).
        zfrtit = torch.clamp(pcil, 0.0, 1.0)

        # Urban branch (surfbc_ctl_mod.F90:385-396).  When PCUR is
        # supplied and the topology carries an urban tile, the land
        # covers are re-rescaled to leave room for the urban fraction
        # (ZPCVLK = 0 -- the port has no lake).  Otherwise the pure-land
        # split (369-371) is used.
        has_urban = (ntile > TILE_URBAN) and (static.urban_frac is not None)
        if has_urban:
            pcur = static.urban_frac.to(dtype=dtype, device=device)
            zpcvl = pcvl * (1.0 - pcur - zfrtit)
            zpcvh = pcvh * (1.0 - pcur - zfrtit)
            zpcvb = 1.0 - zpcvl - zpcvh - pcur - zfrtit
        else:
            pcur = None
            zpcvl = pcvl * (1.0 - zfrtit)        # low-veg cover
            zpcvh = pcvh * (1.0 - zfrtit)        # high-veg cover
            zpcvb = 1.0 - zpcvl - zpcvh - zfrtit  # bare-soil cover

        one_m_cvs = 1.0 - zcvs
        one_m_cvw = 1.0 - zcvw

        frac = torch.zeros((ngrid, ntile), dtype=dtype, device=device)
        # PFRTI(2) -- ice tile (port index 1), snow-free land ice.
        frac[:, TILE_ICE] = zfrtit * one_m_cvs
        # PFRTI(3) -- wet-skin tile (port index 2).  The urban branch
        # adds PCUR to the canopy pool (surfbc_ctl_mod.F90:389).
        wet_pool = (zpcvl + zpcvh + zpcvb) if pcur is None else \
            (zpcvl + zpcvh + zpcvb + pcur)
        frac[:, TILE_WET_SKIN] = zcvw * one_m_cvs * wet_pool
        # PFRTI(4) -- low-veg dry (port index 3).
        frac[:, TILE_LOW_VEG_DRY] = zpcvl * one_m_cvs * one_m_cvw
        # PFRTI(5) -- snow on low-veg / bare / ice (port index 4).  The
        # urban branch adds PCUR (surfbc_ctl_mod.F90:391).
        low_snow_pool = zpcvb + zpcvl + zfrtit
        frac[:, TILE_LOW_VEG_SNOW] = zcvs * low_snow_pool
        # PFRTI(6) -- high-veg dry (port index 5).
        frac[:, TILE_HIGH_VEG_DRY] = zpcvh * one_m_cvs * one_m_cvw
        # PFRTI(7) -- snow under high veg (port index 6).
        high_snow_pool = zpcvh if pcur is None else (zpcvh + pcur)
        frac[:, TILE_HIGH_VEG_SNOW] = zcvs * high_snow_pool
        # PFRTI(8) -- bare soil (port index 7).
        if ntile > TILE_BARE_SOIL:
            frac[:, TILE_BARE_SOIL] = zpcvb * one_m_cvs * one_m_cvw
        # PFRTI(10) -- urban tile (port index 9).  Dedicated snow-free,
        # dry canopy tile carrying PCUR (surfbc_ctl_mod.F90:395).
        if has_urban:
            frac[:, TILE_URBAN] = pcur * one_m_cvs * one_m_cvw
        return frac

    def _step_tile_frac(
        self,
        forcing: Forcing,
        static: Static,
        state: SurfState,
        ntile: int,
    ) -> torch.Tensor:
        """Return tile areas for the active surface step.

        An external validation caller may supply the complete VDF tile-area
        boundary.  That boundary must govern radiation, VEXCS and the land
        updates consistently; replacing it only after the radiation call
        permits a different snow-cover branch on the following time step.
        Standalone offline runs retain the prognostic SURFBC split.
        """
        external = forcing.vdf_pfrti
        if external is None:
            return self._tile_frac(static, ntile, state=state)
        expected = (state.skin_t.shape[0], ntile)
        if tuple(external.shape) != expected:
            raise ValueError(
                f"Forcing.vdf_pfrti must have shape {expected}, "
                f"got {tuple(external.shape)}"
            )
        return external.to(dtype=self.config.dtype, device=state.skin_t.device)

    def _tile_frac(
        self, static: Static, ntile: int, state: Optional[SurfState] = None,
    ) -> torch.Tensor:
        """Tile fractions PFRTI ``(Ngrid, ntile)``.

        Two paths:

        * **Dynamic** (faithful SURFBC split): when ``state`` is given
          and ``static`` carries the cover inputs (``cvl_frac`` /
          ``cvh_frac``), the fractions are recomputed every step from
          the covers + snow cover + skin reservoir via
          :meth:`_compute_tile_frac`.  This is the
          ``surfbc_ctl_mod.F90`` land-branch algorithm.

        * **Static** (cold-start approximation, default): the fractions
          come straight from ``static.tile_frac_base`` -- either the
          canonical 8-tile array (padded with zeros for lake / urban
          when ``ntile > 8``) or an already full-width ``ntile`` array
          whose lake / urban columns the caller populated directly.
          This path keeps every existing test's tile layout unchanged.

        Args:
            static: the static descriptor.
            ntile:  active tile count.
            state:  optional prognostic state; required for the dynamic
                    path (it supplies ZCVS / ZCVW).  When ``None`` the
                    static path is used unconditionally.
        """
        if state is not None and static.cvl_frac is not None \
                and static.cvh_frac is not None:
            return self._compute_tile_frac(state, static, ntile)
        ngrid = static.tile_frac_base.shape[0]
        dtype = static.tile_frac_base.dtype
        device = static.tile_frac_base.device
        base = static.tile_frac_base
        if base.shape[1] == ntile:
            return base
        frac = torch.zeros((ngrid, ntile), dtype=dtype, device=device)
        # Copy as many leading columns as the base array and the target
        # both hold.  ``min`` keeps this safe if a future caller hands in
        # an unexpectedly narrow base; ``N_BASE_TILES`` is the documented
        # 8-tile width.
        ncopy = min(base.shape[1], N_BASE_TILES, ntile)
        frac[:, :ncopy] = base[:, :ncopy]
        return frac

    def _broadcast_dt(self, dt: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Broadcast the scalar / (Ngrid,) timestep to ``shape``."""
        if dt.dim() == 0:
            return dt.expand(shape)
        return dt.broadcast_to(shape)

    def _build_diag(
        self, out, static: Static, ntile: int,
        state: Optional[SurfState] = None,
    ) -> SurfDiag:
        """Pack SEB / VDFDIFH1S outputs into a :class:`SurfDiag`.

        Accepts either :class:`SebOutputs` or :class:`DiffusionOutputs`
        -- both expose the per-tile ``pssh`` / ``pjq`` / ``pstr`` /
        ``pg0`` / ``pslh`` fields the diagnostics consume.  Typed loose
        (no annotation) so a runtime isinstance check is unnecessary.
        """
        frac = self._tile_frac(static, ntile, state=state)
        return SurfDiag(
            h_flux_tile=out.pssh,
            e_flux_tile=out.pjq,
            str_tile=out.pstr,
            g0_tile=out.pg0,
            slh_tile=out.pslh,
            h_flux=scatter_to_tiles(out.pssh, frac),
            e_flux=scatter_to_tiles(out.pjq, frac),
            net_rad=scatter_to_tiles(out.pstr, frac),
            ground_heat=scatter_to_tiles(out.pg0, frac),
        )

    def _tile_salinity(
        self, ntile: int, dtype: torch.dtype, device,
    ) -> torch.Tensor:
        """Per-tile ``ZSALIN`` factor (vsurf_mod.F90 262-265).

        ``0.98`` on the ocean tile when the ``LEOCSA`` namelist switch is
        active, ``1.0`` on every other tile (the offline-driver default,
        which is what :attr:`SurfConfig.use_kpp`/prescribed-SST runs use).
        This port currently leaves ``LEOCSA`` at its offline default
        (``False``) -- so every entry is ``1.0`` -- but the plumbing is
        here so a future namelist-driven config can flip just the ocean
        column without touching the saturation functions.
        """
        sal = torch.ones((1, ntile), dtype=dtype, device=device)
        # When LEOCSA is exposed as a config flag this becomes
        #   if leocsa: sal[:, TILE_WATER] = 0.98
        # Left as 1.0 for now to match the offline-driver default.
        return sal

    def _validate_state(
        self,
        state: SurfState,
        forcing: Forcing,
        static: Static,
        new_state: SurfState,
        ntile: int,
    ) -> None:
        """Cheap invariant checks (SurfConfig.validate).

        Asserts the physically-mandatory conditions that a correct
        surface step must satisfy; raises ``AssertionError`` on failure.
        Intentionally loose (no per-ULP checks) so legitimate warm/cold
        regimes do not trip it.
        """
        # Tile fractions must sum to ~1 over the active tiles.
        frac = self._tile_frac(static, ntile, state=state)
        fsum = frac.sum(dim=-1)
        assert torch.allclose(fsum, torch.ones_like(fsum), atol=1e-6), (
            f"tile fractions do not sum to 1: max|sum-1| = "
            f"{(fsum - 1.0).abs().max().item():.3e}"
        )
        # Outputs must be finite (catches NaN/inf from a bad port).
        assert torch.isfinite(new_state.skin_t).all(), (
            "non-finite skin temperature after surface step"
        )
        # Physically-sane skin temperature band (broad; covers polar night
        # through hot desert).  150 K / 350 K are generous guard rails.
        assert new_state.skin_t.min().item() > 150.0, (
            f"skin T below 150 K: min = {new_state.skin_t.min().item():.1f}"
        )
        assert new_state.skin_t.max().item() < 350.0, (
            f"skin T above 350 K: max = {new_state.skin_t.max().item():.1f}"
        )


__all__ = ["LandSurface"]
