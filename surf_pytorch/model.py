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
  coefficients (:func:`surf_pytorch.physics.exchange.vexcs`).  When the
  :class:`Static` descriptor carries roughness lengths (``z0m`` /
  ``z0h`` / ``z0q``) the model uses the full stability-dependent
  scheme; otherwise it falls back to a neutral ``Cd = 1e-3`` placeholder.
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
  (:func:`surf_pytorch.physics.snow.compute_snow`, ``SRFSN_LWIMPS``)
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
  full Farquhar + Ball-Berry conductance port.  When
  ``use_farquhar=True`` the moisture-flux closure is driven by the
  Farquhar canopy conductance (``PGSTOT -> 1/max(RCONDCTMIN, PGSTOT)``)
  instead of the Jarvis multiplicative stress model, mirroring the
  upstream ``LEFARQUHAR`` branch (``cotworestress_mod.F90`` 412-431).
"""

from __future__ import annotations

from typing import Optional, Tuple

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
from .physics.exchange import vexcs as compute_vexcs, ExchangeOutputs
from .physics.diffusion import (
    DiffusionInputs, DiffusionOutputs, vdfdifh1s as compute_vdfdifh1s,
    RVDIFTS,
)
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
        mismatch; the single-layer snow scheme (M6c, SRFSN_LWIMPS)
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

        Placeholders active in this step (replaced in later milestones)
        ----------------------------------------------------------------
        * :meth:`_exchange_coeffs` -- when ``Static`` lacks roughness
          lengths the model uses a neutral ``Cd = 1e-3`` placeholder
          with ``Ch = Cq = Cd`` instead of the full Louis/Beljaars
          Monin-Obukhov scheme.  When z0 is populated the full VEXCS
          path runs (M4.1).
        * :meth:`_sub_surface_t` -- broadcasts the top soil-layer T to
          every tile; proper sea-ice / lake skin stubs come with the
          snow (M6) / lake modules.
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
          chain): the Jarvis multiplicative stress model (default,
          ``use_farquhar=False``) or the Farquhar photosynthesis /
          Ball-Berry conductance model (``use_farquhar=True``, mirror of
          the upstream ``LEFARQUHAR`` branch in
          ``cotworestress_mod.F90`` 412-431).  Bare soil / urban
          resistance always uses the Jarvis formula even under Farquhar
          (``vsurf_mod.F90:687``).

        Returns
        -------
        (next_state, diagnostics)
            ``next_state`` is a fresh :class:`SurfState` (the input is
            untouched) suitable for chaining into the next step.
            ``diagnostics`` holds the fluxes and 2 m / 10 m fields.
        """
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
        skin_t_prev = state.skin_t                 # (Ngrid, Ntile)
        p_surf_t = expand_as_tiles(forcing.p_surf, ntile)
        salinity = self._tile_salinity(ntile, dtype, device)
        qskm1m = qsat(skin_t_prev, p_surf_t, self.consts, salinity=salinity)
        dqsdT = dqsat_dt_ifs(skin_t_prev, p_surf_t, self.consts, salinity=salinity)
        sskm1m = cp * skin_t_prev

        # ------------------------------------------------------------------ #
        # 3. Turbulent exchange coefficients.  When the static descriptor
        #    carries roughness lengths (PZ0MM/HM/QM) we run the full
        #    Beljaars Monin-Obukhov scheme (VEXCS); otherwise we fall back
        #    to the neutral-Cd placeholder so existing tests that do not
        #    populate z0 keep working.
        # ------------------------------------------------------------------ #
        rhoc_u, rhoc_q = self._exchange_coeffs(forcing, static, state)
        if self.config.per_tile_exchange:
            # _exchange_coeffs already returns per-tile (Ngrid, ntile).
            rhoc_u_t, rhoc_q_t = rhoc_u, rhoc_q
        else:
            rhoc_u_t = expand_as_tiles(rhoc_u, ntile)
            rhoc_q_t = expand_as_tiles(rhoc_q, ntile)

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
        )

        # ------------------------------------------------------------------ #
        # 5. Radiation -- convert cumulative fluxes (J/m^2) to mean W/m^2.
        # ------------------------------------------------------------------ #
        dt = forcing.dt
        ssrfl_grid = forcing.fr_so / dt
        slrfl_grid = forcing.fr_th / dt
        ssrfl_t = expand_as_tiles(ssrfl_grid, ntile)
        # Surface albedo -- per-tile broadband alpha (dimensionless).
        # Returns all-zeros when neither the snow-albedo prognostic nor the
        # background albedo fields are set, so the absorbed SW is then
        # ``ssrfl_t`` unchanged (bit-identical to the build before the
        # albedo work).  Uses the t-1 PASN (step-start field), consistent
        # with the t-1 skin T used for radiation above; the freshly-evolved
        # PASN is written to new_state.snow_alb in section 7 for next step.
        alpha_t = self._surface_albedo(state, static, forcing, ntile, dtype, device)
        ssrfl_t = ssrfl_t * (1.0 - alpha_t)

        # Tiled skin T at radiation time -- cold-start uses the t-1 tile
        # fractions-weighted mean.
        frac = self._tile_frac(static, ntile, state=state)
        tskrad_grid = scatter_to_tiles(state.skin_t, frac)

        # Emissivity -- fall back to 0.99 (offline-driver default).
        emis_grid = static.emis if static.emis is not None else \
            torch.full((ngrid,), 0.99, dtype=dtype, device=device)

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
        # When use_farquhar is True we keep the unit-alpha placeholder
        # (Farquhar photosynthesis lands in a later milestone); otherwise
        # the Jarvis stomatal-resistance chain (vsurf §2.3) feeds the
        # faithful VEVAP transcription (vsurf/vevap), which fixes the
        # rho-convention bug that the old build_tile_alpha carried
        # (an extra 1/rho in the alpha denominator).
        #
        # Both moisture-flux closures (Jarvis and Farquhar) share the
        # downstream VEVAP chain: the only difference is the *source* of
        # the canopy resistance PWET* for tiles 4/6/7.  Jarvis computes
        # PWETL/PWETH/PWETHS from the multiplicative stress model; Farquhar
        # derives them from the Farquhar photosynthesis conductance
        # (PGSTOT -> 1/max(RCONDCTMIN, PGSTOT); cotworestress_mod.F90:427).
        # Bare soil (PWETB) always uses the Jarvis formula even under
        # LEFARQUHAR (vsurf_mod.F90:687 -- COTWORESTRESS only computes the
        # vegetation canopy conductance), so the Jarvis chain runs in both
        # modes.  Wet-skin (KTILE=3) is driven by the interception clamp,
        # not by the canopy resistance, so it is unaffected.
        jarvis = compute_jarvis_resistance(
            soil_t=state.soil_t,
            soil_w=state.soil_w_liq,
            soil_type=static.soil_type,
            ktvl=ktvl, ktvh=ktvh,
            t_air=forcing.t, q_air=forcing.q, p_surf=forcing.p_surf,
            sw_down=ssrfl_grid,
            veg_table=self.veg_table,
            soil_table=self.soil_table,
            root_table=self.root_table,
            consts=self.consts,
            tile_frac=frac, n_tile=ntile,
        )
        if self.config.use_farquhar:
            # LEFARQUHAR branch (cotworestress_mod.F90:412-431): the
            # Farquhar canopy conductance replaces Jarvis PWETL/PWETH/
            # PWETHS for the vegetated tiles; PWETB is preserved.
            resist = compute_farquhar_resistance(
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
                "farquhar_pgstot": resist["farquhar_pgstot"],
                "farquhar_pgpp":   resist["farquhar_pgpp"],
                "farquhar_prd":    resist["farquhar_prd"],
                "farquhar_pan":    resist["farquhar_pan"],
            }
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
        # t_sub for the KTILE=7 snow-under-canopy stability selector:
        # the top soil-layer temperature, broadcast per grid.
        t_sub_grid = tsrf[:, 0]
        vegc = self.veg_table.consts
        vevap_out = compute_tile_vevap(
            pwetl=resist["pwetl"], pweth=resist["pweth"],
            pwetb=resist["pwetb"],
            rhocqu_grid=rhoc_q,
            p_surf=forcing.p_surf, t_air=forcing.t, q_air=forcing.q,
            skin_t=skin_t_prev, t_sub_grid=t_sub_grid,
            qs_skin=qskm1m, wlmx_grid=wlmx_grid,
            consts=self.consts, config=self.config,
            pwetlu=resist["pwetl"], pweths=resist["pweths"],
            ptmst=dt,
            rlhaero=vegc.rlhaero, rlhaeros=vegc.rlhaeros,
        )
        # SEB consumes PCSAT (-> alpha_s, saturation at skin T) and
        # PCAIR (-> alpha_l, potential from skin reservoir).  The
        # other VEVAP outputs (pcsatu/pcairu/pcptsu/pcsnw/pdhvegs)
        # feed downstream canopy / M5 updates, not the SEB closure.
        alphas = vevap_out["pcsat"]
        alphal = vevap_out["pcair"]

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
            dt=dt, ntile=ntile, ngrid=ngrid, dtype=dtype, device=device,
        )
        out = compute_vdfdifh1s(diff_inp, self.consts, self.config)

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
            state=state, static=static, diff_out=out, frac=frac,
            wlmx_grid=wlmx_grid, ktvl=ktvl, ktvh=ktvh,
            forcing=forcing, dt=dt,
            ngrid=ngrid, dtype=dtype, device=device,
        )

        # ------------------------------------------------------------------ #
        # 6a. HTESSEL single-layer snow update (M6c, SRFSN_LWIMPS).
        # ------------------------------------------------------------------ #
        # Advances the snow prognostic (PTSNM1M, PSNM1M; PRSNM1M is carried
        # through unchanged into the snow solve) and produces the snow-
        # bottom heat flux PGSN that the soil-T solve consumes below.  The
        # snow density PRSNM1M is then advanced separately by the
        # SRFSN_LWIMP section-3.1 density-evolution port
        # (_update_snow_density) -- the ``-s`` single-layer variant the
        # port mirrors does not evolve density itself, so the density
        # block runs as a post-step.  Upstream call order
        # (surftstps_ctl_mod.F90): SRFSN_LWIMPS (:320, OUT ZGSN/ZTSN)
        # precedes SRFTS (:345, IN ZGSN), so this slot sits between
        # VDFDIFH1S and _update_soil_temperature.  Snowfall throughfall
        # PSSFL <- forcing.snow/dt (M2d); convective snowfall PSSFC stays
        # zero (the port's :class:`Forcing` does not split convective vs
        # large-scale, so all snow is treated as large-scale, matching the
        # same convention used for rainfall in _update_soil_moisture).
        snow_out = self._update_snow(
            state=state, static=static, diff_out=out,
            pssrflti=ssrfl_t, pslrfl=slrfl_grid, pfrti=frac,
            forcing=forcing,
            dt=dt, ngrid=ngrid, dtype=dtype, device=device,
        )
        # Advance the snow density (PRSN <- SRFSN_LWIMP section 3.1).
        new_snow_rho = self._update_snow_density(
            state=state, snow_out=snow_out, forcing=forcing,
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

        # ------------------------------------------------------------------ #
        # 6b. HTESSEL soil-temperature update (M5d-soilt).
        # ------------------------------------------------------------------ #
        # Advances the soil-temperature prognostic by one coupling
        # interval via SRFRCGS (apparent heat capacity) + SRFTS (Thomas
        # coefficients) + SRFWDIFS (tridiag solve).  Inputs come from the
        # t-1 soil state (soil_t, soil_w_liq) and the per-tile turbulent
        # fluxes VDFDIFH1S just produced (PAHFSTI -> pssh, PEVAPTI ->
        # pjq, net SW/LW from the radiation terms).  The snow-bottom heat
        # flux PGSN flows from the SRFSN_LWIMPS solve above.
        new_soil_t = self._update_soil_temperature(
            state=state, static=static, diff_out=out,
            pssrflti=ssrfl_t, pslrfl=slrfl_grid, pfrti=frac,
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
        # PTSFC and snowmelt / snow-evap mismatch PMSN / PEMSSN /
        # PEVAPSNW stay zero -- the ``-s`` single-layer snow variant
        # upstream has no SRFWEXC/SRFWDIF soil-W snow coupling.  PSDOR
        # (orography std-dev for the VIC runoff) defaults to flat terrain
        # (0 m) -- cold start.
        soilw_out = self._update_soil_moisture(
            state=state, static=static, diff_out=out,
            forcing=forcing, pfrti=frac,
            ptsfl=canopy_out.ptsfl, peintti=canopy_out.peintti,
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
        # Snow state advances through SRFSN_LWIMPS: PTSN <- ZTSN,
        # PSNM1M <- ZSSTAR (snow mass after P-E); PRSNM1M advances
        # through the SRFSN_LWIMP section-3.1 density-evolution port
        # (_update_snow_density).  The skin-reservoir water advances
        # through SRFWL: PWL <- canopy_out.pwl.
        new_state = state.like(
            skin_t=out.ptsktip1,
            soil_t=new_soil_t,
            soil_w_liq=new_soil_w,
            snow_t=snow_out.ptsn,
            snow_w=snow_out.psstar,
            snow_rho=new_snow_rho,
            snow_alb=new_snow_alb,
            skin_water=canopy_out.pwl,
        )
        diag = self._build_diag(out, static, ntile, state=state)

        # Surface the Farquhar carbon-flux diagnostics when active.  These
        # are per-grid (Ngrid,) tensors already computed inside the
        # use_farquhar branch; attaching them here keeps the prognostic
        # forward path untouched.
        if _farquhar_diag is not None:
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
    def _exchange_coeffs(
        self,
        forcing: Forcing,
        static: Static,
        state: SurfState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rho*Ch*|U|, rho*Cq*|U|)`` at the lowest model level.

        Two modes, selected by :attr:`SurfConfig.per_tile_exchange`:

        * **per-tile** (default, ``per_tile_exchange=True``): mirrors the
          offline driver's tile loop (``surfexcdrivers_ctl_mod.F90:388-
          398``) -- a single vectorised VEXCS call over the flattened
          ``(Ngrid, Ntile)`` axis, with tile-specific roughness lengths
          and tile-specific surface state.  Returns per-tile tensors of
          shape ``(Ngrid, Ntile)``.  This is the dominant single-step
          skin-T term.
        * **grid-mean** (legacy, ``per_tile_exchange=False``): runs VEXCS
          once on the grid-mean z0 + grid-mean skin, returns
          ``(Ngrid,)``; :meth:`forward` then broadcasts the value across
          tiles via :func:`expand_as_tiles`.

        In both modes, when the static descriptor lacks roughness lengths
        (``z0m`` / ``z0h`` / ``z0q`` all ``None``), the per-tile path
        derives them from the :class:`VegTable`; if that also fails it
        falls back to the neutral ``Cd = 1e-3`` placeholder.
        """
        if self.config.per_tile_exchange:
            return self._per_tile_monin_obukhov_coeffs(forcing, static, state)
        if static.z0m is None or static.z0h is None or static.z0q is None:
            return self._neutral_exchange_coeffs(forcing)
        return self._gridmean_monin_obukhov_coeffs(forcing, static, state)

    def _neutral_exchange_coeffs(
        self, forcing: Forcing
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Neutral ``Cd = 1e-3`` placeholder used when z0 is unavailable.

        Kept as a separate helper so the test-suite can exercise the
        fallback path explicitly.  Returns grid-mean ``(Ngrid,)`` tensors.
        """
        wind = torch.sqrt(forcing.u ** 2 + forcing.v ** 2)
        rho = forcing.p_surf / (self.consts.rd * forcing.t)
        cd_n = torch.full_like(wind, 1.0e-3)
        rhoc = rho * cd_n * wind
        return rhoc, rhoc

    def _per_tile_z0(
        self,
        static: Static,
        frac: torch.Tensor,
        ktvl: torch.Tensor,
        ktvh: torch.Tensor,
        ntile: int,
        dtype: torch.dtype,
        device,
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
        # Snow type 12 (exposed-snow, vupdz0s_mod.F90:351).
        snow_idx = torch.full_like(ktvl, 12)
        z0m_snow = self.veg_table.gather("rvz0m", snow_idx)
        z0h_snow = self.veg_table.gather("rvz0h", snow_idx)
        # Bare-soil / desert (index 0 sentinel).
        bare_idx = torch.zeros_like(ktvl)
        z0m_bare = self.veg_table.gather("rvz0m", bare_idx)
        z0h_bare = self.veg_table.gather("rvz0h", bare_idx)

        # Cover fractions CVL / CVH (low / high veg tile fractions).
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
        one_minus = 1.0 - cvl - cvh
        # Wet-skin blend: cvl*RVZ0M(ktvl)+cvh*RVZ0M(ktvh)+(1-cvl-cvh)*RVZ0M(0)
        z0m_wet = cvl * z0m_low + cvh * z0m_high + one_minus * z0m_bare
        z0h_wet = cvl * z0h_low + cvh * z0h_high + one_minus * z0h_bare

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
        z0m[:, TILE_LOW_VEG_SNOW] = z0m_snow
        z0h[:, TILE_LOW_VEG_SNOW] = z0h_snow
        z0m[:, TILE_HIGH_VEG_DRY] = z0m_high
        z0h[:, TILE_HIGH_VEG_DRY] = z0h_high
        z0m[:, TILE_HIGH_VEG_SNOW] = z0m_high      # sheltered snow uses ktvh
        z0h[:, TILE_HIGH_VEG_SNOW] = z0h_high
        if ntile > TILE_BARE_SOIL:
            z0m[:, TILE_BARE_SOIL] = z0m_bare
            z0h[:, TILE_BARE_SOIL] = z0h_bare
        # z0q = z0h for every land tile (vupdz0s_mod.F90:337,345,...).
        z0q = z0h.clone()
        return z0m, z0h, z0q

    def _per_tile_monin_obukhov_coeffs(
        self,
        forcing: Forcing,
        static: Static,
        state: SurfState,
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
        ngrid = state.skin_t.shape[0]
        dtype = self.config.dtype
        device = state.skin_t.device

        frac = self._tile_frac(static, ntile, state=state)
        ktvl, ktvh = self._dominant_veg_types(static)

        # Per-tile surface state (kept per-tile -- NOT scattered to grid).
        pcpts = cp * state.skin_t                       # (Ngrid, ntile)
        p_surf_t = expand_as_tiles(forcing.p_surf, ntile)
        pqsam = qsat(state.skin_t, p_surf_t, self.consts)

        # Per-tile roughness lengths.
        z0m, z0h, z0q = self._per_tile_z0(
            static, frac, ktvl, ktvh, ntile, dtype, device,
        )

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
        out: ExchangeOutputs = compute_vexcs(
            _rep(forcing.u), _rep(forcing.v), _rep(forcing.t), _rep(forcing.q),
            _rep(forcing.p_surf), _rep(forcing.geom1), _rep(pcptgzlev_g),
            pcpts.reshape(-1), pqsam.reshape(-1),
            z0m.reshape(-1), z0h.reshape(-1), z0q.reshape(-1),
            None, _rep(buom_g), _rep(ucurr_g), _rep(vcurr_g), _rep(i10fgcv_g),
            self.consts, self.ritbl_table,
            kitt=self.config.vexcs_kitt,
            ldinit=True,
        )
        # Cache the per-tile z/L (grid-mean for the warm-start thread).
        self._last_pzdl = scatter_to_tiles(
            out.pzdl.reshape(ngrid, ntile), frac,
        ).detach()
        return (
            out.pcfh.reshape(ngrid, ntile),
            out.pcfq.reshape(ngrid, ntile),
        )

    def _gridmean_monin_obukhov_coeffs(
        self,
        forcing: Forcing,
        static: Static,
        state: SurfState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full Beljaars Monin-Obukhov exchange coefficients (VEXCS).

        Builds the grid-mean inputs that VEXCS expects (surface dry
        static energy, surface specific humidity, ocean current, W*
        buoyancy flux) and dispatches to
        :func:`surf_pytorch.physics.exchange.vexcs`.  Returns only the
        SEB-relevant ``(rhoc_u = pcfh, rhoc_q = pcfq)`` pair; the full
        :class:`ExchangeOutputs` is cached on the SurfDiag builder.

        Legacy path (``per_tile_exchange=False``): the returned grid-mean
        ``(Ngrid,)`` tensors are broadcast across tiles by the caller.
        """
        ntile = self.config.n_tile
        cp = self.consts.cpd

        # Grid-mean surface dry static energy = cp * Tskin (the geom
        # term cancels because the skin is at z=0).  Tile-fraction
        # weighted mean over the per-tile skin temperature.
        frac = self._tile_frac(static, ntile, state=state)
        tskin_grid = scatter_to_tiles(state.skin_t, frac)
        pcpts = cp * tskin_grid

        # Surface specific humidity: tile-fraction-weighted qsat at
        # the t-1 skin temperature (computed already in section 2 of
        # forward, but here we recompute for self-containment so the
        # helper is callable from tests).
        p_surf_t = expand_as_tiles(forcing.p_surf, ntile)
        qskin_t = qsat(state.skin_t, p_surf_t, self.consts)
        pqsam = scatter_to_tiles(qskin_t, frac)

        # Lowest-level dry static energy: cp*T + geom1.
        pcptgzlev = cp * forcing.t + forcing.geom1

        # Optional buoyancy flux and ocean current -- zero when absent
        # (matches the upstream default PBUOM=0, PUCURR=PVCURR=0).
        ngrid = forcing.u.shape[0]
        dtype = self.config.dtype
        device = forcing.u.device
        buom = forcing.buom if forcing.buom is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)
        ucurr = forcing.ucurr if forcing.ucurr is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)
        vcurr = forcing.vcurr if forcing.vcurr is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)
        i10fgcv = forcing.i10fgcv if forcing.i10fgcv is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)

        out: ExchangeOutputs = compute_vexcs(
            forcing.u, forcing.v, forcing.t, forcing.q,
            forcing.p_surf, forcing.geom1, pcptgzlev, pcpts, pqsam,
            static.z0m, static.z0h, static.z0q,
            None, buom, ucurr, vcurr, i10fgcv,
            self.consts, self.ritbl_table,
            kitt=self.config.vexcs_kitt,
            ldinit=True,
        )
        # Cache the latest z/L on the module for warm-start on the next
        # step (the upstream driver threads PZDL between calls).  Tests
        # can inspect self._last_pzdl.
        self._last_pzdl = out.pzdl.detach()
        return out.pcfh, out.pcfq

    def _sub_surface_t(self, state: SurfState, ntile: int) -> torch.Tensor:
        """Per-tile sub-surface temperature PTSRF (K).

        Land tiles use the top soil layer temperature; the water tile
        uses SST (skin pins to it via lambda_skin = LARGE); ice uses
        the sea-ice skin.  Until the sea-ice / lake stubs land this is
        simply a broadcast of the top soil layer to every tile.

        Note: this is the *VLAMSK* path (which needs the per-tile PTSRF
        to compute lambda_skin).  The VDFDIFH1S wrapper builds its own
        per-tile PTSRF internally from the per-grid scalar fields on
        :class:`DiffusionInputs` -- see :meth:`_build_diffusion_inputs`.
        """
        tsrf = expand_as_tiles(state.soil_t[:, 0], ntile).clone()
        if state.sst is not None:
            tsrf[:, TILE_WATER] = state.sst
        return tsrf

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

        Lowest-level tendencies PSLGE / PQTE default to zero (standalone
        mode -- the surface step does not update the lowest model level
        within the coupling interval), so the Dirichlet boundary
        condition reduces to ``bsl = PSLM1`` and ``bql = PQTM1``.
        """
        cp = self.consts.cpd          # 0-d tensor, broadcasts
        rg = self.consts.g
        ptmst = dt if torch.is_tensor(dt) else torch.tensor(dt, dtype=dtype, device=device)
        # vdfmain1s (lines 745-746) rescaling, inverted by the wrapper.
        rscale = rg * ptmst * RVDIFTS
        pcfhti = rhoc_u_t * rscale
        pcfqti = rhoc_q_t * rscale

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

        # Optional snow-under-forest evaporation -- None until the
        # canopy / snow prognostic update (M6) feeds a non-zero value.
        pevapsnw = getattr(state, "_pevapsnw", None)

        # Optional lake tile fields -- surfaced from SurfState.lake_t
        # when present, otherwise left as None (the wrapper only
        # consumes them when n_tile > TILE_LAKE).
        phlice = getattr(state, "_phlice", None)
        ptlice = getattr(state, "_ptlice", None)
        ptlwml = state.lake_t

        return DiffusionInputs(
            pslm1_lev=sl_lev,
            pqtm1_lev=q_lev,
            pslge_lev=torch.zeros((ngrid,), dtype=dtype, device=device),
            pqte_lev=torch.zeros((ngrid,), dtype=dtype, device=device),
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
            lelwdd=True, lelwtl=False,
        )

    def _update_soil_temperature(
        self,
        *,
        state: SurfState,
        static: Static,
        diff_out,                       # DiffusionOutputs
        pssrflti: torch.Tensor,
        pslrfl: torch.Tensor,
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

        # Vegetation cover fractions from the tile fractions.
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

        # Snow-bottom heat flux PGSN comes from the SRFSN_LWIMPS solve
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
        pfrti: torch.Tensor,
        forcing: Forcing,
        dt: torch.Tensor,
        ngrid: int,
        dtype: torch.dtype,
        device,
    ) -> SnowOutputs:
        """Advance the single-layer snow prognostic by one step (M6c).

        Mirrors ``SRFSN_LWIMPS`` (NCSNEC=1, the ``-s`` offline single-
        layer variant).  Builds a :class:`SnowInputs` from the t-1 snow
        state triple ``(PSSNM1M, PTSNM1M, PRSNM1M)`` + the soil top
        temperature and the per-tile turbulent / radiative fluxes
        VDFDIFH1S just produced, runs the snow solve and returns the
        :class:`SnowOutputs`.

        Upstream call order (``surftstps_ctl_mod.F90``): SRFSN_LWIMPS
        (line 320, outputs ``ZGSN`` / ``ZTSN``) is called BEFORE
        ``SRFTS`` (line 345, consumes ``ZGSN`` as the snow-bottom heat
        flux).  The caller therefore threads ``snow_out.pgsn`` into
        :meth:`_update_soil_temperature`.

        Snowfall / rainfall throughfall (M2d): upstream ``SRFSN_LWIMPS``
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
            PSSFC/PTSFC/PTSFL/PEVAPSNW      <- 0 (cold start)
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
            pssfc=zero1d, pssfl=pssfl,
            ptsfc=zero1d, ptsfl=zero1d,
            pevapsnw=zero1d,
            ptmst=dt,
        )
        return port_snow(inp, self.soil_table, self.consts, self.config)

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
            PCVL/PCVH (veg cover)           <- reconstructed from frac
            PWLMX    (interception capacity) <- wlmx_grid arg
            PFRTI    (tile fractions)       <- frac arg
            PEVAPTI  (tile moisture flux)   <- diff_out.pjq
            PRSFL    (large-scale rain)     <- forcing.rain / dt   (M2d)
            PRSFC/PEVAPSNW                   <- 0 (cold start)
            PTMST    (coupling interval)    <- dt arg
        """
        ptmst = dt if torch.is_tensor(dt) else torch.tensor(dt, dtype=dtype, device=device)
        ntile = frac.shape[-1]

        # Vegetation cover fractions PCVL / PCVH from the active tile
        # fractions (mirrors _interception_capacity / _update_soil_-
        # temperature).  Low-veg = dry-low + snow-low; high-veg =
        # dry-high + snow-high.
        from .config import (
            TILE_LOW_VEG_DRY, TILE_LOW_VEG_SNOW,
            TILE_HIGH_VEG_DRY, TILE_HIGH_VEG_SNOW,
        )
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
        # zero (cold start).  PEVAPSNW zero (snow-under-forest evap;
        # the single-layer snow scheme does not thread it back yet).
        prsfl = forcing.rain / ptmst
        prsfc = torch.zeros((ngrid,), dtype=dtype, device=device)
        pevapsnw = torch.zeros((ngrid,), dtype=dtype, device=device)

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
        * ``PMSN`` / ``PEMSSN`` / ``PEVAPSNW``       <- zero (the ``-s``
          single-layer snow variant has no soil-W snow coupling)
        * ``PEINTTI`` (interception-layer evap mismatch) <- ``peintti``
          arg (from :meth:`_update_canopy`)

        Sign conventions agree: both VDFDIFH1S and SRFWEXC_VG use the
        positive-downward convention (positive PEVAPTI = dew into the
        soil, negative = evaporation removing water).  The HTESSEL
        solve clamps the result to ``[WRES, WSAT]`` per layer.

        ``PSDOR`` (orography std-dev for the VIC sub-grid runoff) is
        not carried by :class:`Static` -- the port has geopotential
        height only.  We pass zero (flat terrain); the VIC runoff is
        still activated through its moisture-dependence, just without
        the orographic enhancement.
        """
        # Convective throughfall + snow / snow-evap mismatch terms stay
        # zero (cold start: no convective precip split, no soil-W snow
        # coupling in the ``-s`` single-layer variant).
        ptsfc = torch.zeros((ngrid,), dtype=dtype, device=device)
        pmsn = torch.zeros((ngrid,), dtype=dtype, device=device)
        pemssn = torch.zeros((ngrid,), dtype=dtype, device=device)
        pevapsnw = torch.zeros((ngrid,), dtype=dtype, device=device)

        # PSDOR (orography std-dev) -- flat-terrain cold start.
        psdor = torch.zeros((ngrid,), dtype=dtype, device=device)

        # Land mask from the static descriptor (LDLAND).
        ldland = static.land_frac > 0.5

        inp = SoilMoistInputs(
            pwsam1m=state.soil_w_liq,
            ptsam1m=state.soil_t,
            pfrti=pfrti,
            pevapti=diff_out.pjq,
            ptsfc=ptsfc, ptsfl=ptsfl,
            pmsn=pmsn, pemssn=pemssn, pevapsnw=pevapsnw, peintti=peintti,
            ksoty=static.soil_type,
            ktvl=static.veg_type, ktvh=static.veg_type,
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
        SRFSN_LWIMPS).  The *evolved* density at t is produced by
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
        dt: torch.Tensor,
        dtype: torch.dtype,
        device,
    ) -> torch.Tensor:
        """Evolve the snow density by one step (``SRFSN_LWIMP`` section 3.1).

        Runs the single-layer density-evolution port
        (:func:`surf_pytorch.physics.snow_density.compute_snow_density`)
        on the just-completed snow solve outputs.  The density block is
        the section ``SRFSN_LWIMPS`` omits: it advances ``PRSNM1M`` using
        the fresh-snow density, depth-weighted mixing, overburden /
        metamorphism compaction and the melt-compaction term, and resets
        the density to ``RHOMINSN`` at no-snow points.

        Inputs come from the t-1 snow state (``PSSNM1M``, ``PRSNM1M``,
        ``PTSNM1M``), the new snow temperature ``PTSN`` and tile
        fraction / mask from the snow solve, and the surface forcing
        (skin temperature ``PTSRF`` from the t-1 skin, surface wind
        ``PUSRF`` / ``PVSRF`` from ``forcing.u`` / ``forcing.v``, and the
        large-scale + convective snowfall ``PSSFL`` / ``PSSFC``).

        Mapping to the :class:`SnowDensityInputs` field list::

            PSSNM1M  (snow mass)         <- state.snow_w
            PRSNM1M  (snow density)      <- self._snow_density(state)
            PTSNM1M  (snow T)            <- state.snow_t (or RTT cold start)
            PTSN     (new snow T)        <- snow_out.ptsn
            PTSRF    (surface T)         <- state.skin_t grid-mean
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

        # Surface skin temperature (PTSRF): grid-mean of the per-tile
        # skin T at t-1, matching the SRFSN_LWIMP surface-temperature
        # input.  Equal-weight grid-mean is the cold-start convention
        # (the port does not carry a separate radiative-skin field).
        ptsrf = state.skin_t.mean(dim=-1)
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
        section ``SRFSN_LWIMPS`` omits: it advances ``PASNM1M`` using the
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

        * **Snow tiles** (TILE_LOW_VEG_SNOW, TILE_HIGH_VEG_SNOW) take the
          snow albedo ``PASN`` (t-1 ``state.snow_alb``).
        * **Wet-skin tile** (TILE_WET_SKIN) takes the wet background
          albedo ``albw``, falling back to the dry ``albd`` then 0.
        * **Dry vegetation / bare-soil tiles** take the dry background
          albedo ``albd``, falling back to 0.
        * **Urban tile** (TILE_URBAN), when the urban cover ``PCUR`` is
          supplied, takes the canyon albedo from :meth:`_urban_canyon_albedo`
          (Masson/Harman/Porson, ``canalb_mod.F90``), blended with the
          roof albedo via the road ratio ``RWRR`` (``surfrad_ctl_mod.F90``
          597-602).
        * Water / ice tiles (and any unsupplied tile) stay 0.

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

        # Background albedo fields (broadcast (Ngrid,) -> per-tile).
        albd = static.albd.to(dtype=dtype, device=device) \
            if static.albd is not None else None
        albw = static.albw.to(dtype=dtype, device=device) \
            if static.albw is not None else None

        # Snow tiles: PASN (t-1).  When snow_alb is unset the snow tiles
        # fall through to the background (effectively 0 / albd).
        if has_snow:
            pasn = state.snow_alb.to(dtype=dtype, device=device)
            if ntile > TILE_LOW_VEG_SNOW:
                alpha[:, TILE_LOW_VEG_SNOW] = pasn
            if ntile > TILE_HIGH_VEG_SNOW:
                alpha[:, TILE_HIGH_VEG_SNOW] = pasn

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



    def _cover_fractions(self, static: Static):
        """Return canonical SURFBC cover inputs (PCVL, PCVH, PCIL, PLAIL, PLAIH).

        Activates only when ``static.cvl_frac`` is provided -- the
        sentinel field for the faithful tile-split path.  Returns a
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
        plail = static.lai_low if static.lai_low is not None else \
            self.veg_table.gather("rvlai", ktvl)
        plaih = static.lai_high if static.lai_high is not None else \
            self.veg_table.gather("rvlai", ktvh)
        return static.cvl_frac, static.cvh_frac, pcil, plail, plaih

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
            PFRTI(5)  = ZCVS*(ZPCVB+ZPCVL+PCUR+ZFRTIT)          # +PCUR
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
        zcvs = state.snow_frac if state.snow_frac is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)
        zcvs = torch.clamp(zcvs, 0.0, 1.0)

        # Interception capacity PWLMX (surfbc_ctl_mod.F90:308-310) and
        # wet-skin fraction ZCVW <- PWLM1M / PWLMX (:311).
        rwlmax = self.soil_table.consts.rwlmax
        pwlmx = rwlmax * (1.0 - pcvl - pcvh + pcvl * plail + pcvh * plaih)
        pwlmx = torch.clamp(pwlmx, min=torch.finfo(dtype).tiny)
        pwlm1m = state.skin_water if state.skin_water is not None else \
            torch.zeros(ngrid, dtype=dtype, device=device)
        zcvw = torch.clamp(pwlm1m / pwlmx, 0.0, 1.0)

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
        low_snow_pool = (zpcvb + zpcvl + zfrtit) if pcur is None else \
            (zpcvb + zpcvl + pcur + zfrtit)
        frac[:, TILE_LOW_VEG_SNOW] = zcvs * low_snow_pool
        # PFRTI(6) -- high-veg dry (port index 5).
        frac[:, TILE_HIGH_VEG_DRY] = zpcvh * one_m_cvs * one_m_cvw
        # PFRTI(7) -- snow under high veg (port index 6).
        frac[:, TILE_HIGH_VEG_SNOW] = zcvs * zpcvh
        # PFRTI(8) -- bare soil (port index 7).
        if ntile > TILE_BARE_SOIL:
            frac[:, TILE_BARE_SOIL] = zpcvb * one_m_cvs * one_m_cvw
        # PFRTI(10) -- urban tile (port index 9).  Dedicated snow-free,
        # dry canopy tile carrying PCUR (surfbc_ctl_mod.F90:395).
        if has_urban:
            frac[:, TILE_URBAN] = pcur * one_m_cvs * one_m_cvw
        return frac

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
