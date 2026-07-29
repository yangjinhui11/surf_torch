"""State, forcing and static-field containers for the land-surface model.

These dataclasses mirror the argument lists of the offline driver's
``VDFMAIN1S`` entry point (see ``surf/offline/driver/callpar1s.F90``
lines 906-955 and ``vdfmain1s.F90`` lines 1-130).  They are plain
Python containers (no autograd, no buffer registration) so the model
can pull tensors in/out for both inference and training.

Tensor layout convention
------------------------
The PyTorch port standardises on **grid-major batching**:

* the leading dim of every grid field is the *batch / column* dim
  (``Ngrid``), which can be 1 for a single column test and an arbitrary
  power-of-two for ML training batches;
* soil/snow vertical fields carry their level axis as the last dim
  (``Ngrid, Nlev``) so that ``tridiag_solve`` can stride over ``-1``;
* tile fields carry the tile axis as the last dim
  (``Ngrid, Ntile``);
* atmospheric levels are arranged surface-first (level ``KLEV`` is the
  lowest model level in IFS; here we keep IFS order so the surface is
  the *last* index along ``Nlev_atm``).  Helpers in
  :mod:`surf_pytorch.utils` flip on demand.

Fields are intentionally listed in the same order as the Fortran
declarations so a side-by-side audit of the offline-driver output dump
is easy.  Optional fields (lake, urban, multi-layer snow) are present
as ``None`` placeholders -- the physics modules assert at construction
time if they are required and missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


# --------------------------------------------------------------------------- #
# Atmospheric forcing at the lowest model level (VDFMAIN1S inputs)
# --------------------------------------------------------------------------- #
@dataclass
class Forcing:
    """Time-varying atmospheric forcing at the lowest model level.

    All fields have shape ``(Ngrid,)`` unless noted.  Each entry below
    is annotated with the corresponding Fortran argument from
    ``VDFMAIN1S`` (see ``vdfmain1s.F90`` lines 1-130 and the call in
    ``callpar1s.F90`` 906-955).

    Radiation fluxes are *cumulative over the coupling interval* in IFS
    (i.e. already multiplied by the timestep -- the suffix ``1`` in
    ``PFRSO(1,KLEV)`` indicates the surface-level entry of an
    accumulated field).  We follow the same convention so a future
    merge with the radiation stub needs no rescaling.
    """

    # Lowest-level state (VDFMAIN1S args PUMLEV/PVMLEV/PTMLEV/PQMLEV).
    u: torch.Tensor                # (Ngrid,)     zonal wind          [m/s]
    v: torch.Tensor                # (Ngrid,)     meridional wind     [m/s]
    t: torch.Tensor                # (Ngrid,)     abs. temperature    [K]
    q: torch.Tensor                # (Ngrid,)     specific humidity   [kg/kg]
    co2: torch.Tensor              # (Ngrid,)     CO2 mixing ratio    [kg/kg]

    # Surface pressure and lowest-level geopotential (PAPHMS/PGEOMLEV).
    p_surf: torch.Tensor           # (Ngrid,)     surface pressure    [Pa]
    geom1: torch.Tensor            # (Ngrid,)     lowest-level geopotential height [m2/s2]

    # Radiation -- already accumulated over the coupling interval.
    fr_so: torch.Tensor            # (Ngrid,)     solar radiation at sfc  [J/m^2]
    fr_th: torch.Tensor            # (Ngrid,)     thermal radiation at sfc [J/m^2]
    mu0: torch.Tensor              # (Ngrid,)     solar zenith angle cosine

    # Precipitation (rain + snow) accumulated over the coupling interval.
    rain: torch.Tensor             # (Ngrid,)     liquid precip      [kg/m^2]
    snow: torch.Tensor             # (Ngrid,)     solid precip       [kg/m^2]

    # Forcing "above" tile (wind etc at lowest level already given).
    # LW/LW-derived net at the surface that the radiative scheme
    # produces -- kept separate from the cumulative fluxes for the
    # Penman-Monteith closure (seb module consumes both).

    # Timestep -- the coupling interval in seconds (PTSPHY).
    dt: torch.Tensor               # (Ngrid,) scalar broadcast, [s]

    # Optional Monin-Obukhov inputs (M4, VEXCS).  When absent the
    # exchange-coefficient module falls back to neutral scaling.
    buom: Optional[torch.Tensor] = None    # (Ngrid,) surface buoyancy flux
    ucurr: Optional[torch.Tensor] = None   # (Ngrid,) ocean current u [m/s]
    vcurr: Optional[torch.Tensor] = None   # (Ngrid,) ocean current v [m/s]
    i10fgcv: Optional[torch.Tensor] = None  # (Ngrid,) convective gust [m/s]

    # Optional lowest-level momentum fluxes from the atmospheric diffusion
    # solve (SURFPP_CTL PSTRTULEV/PSTRTVLEV). When supplied, these are the
    # authoritative PUSTRTI/PVSTRTI warm-start values for the next step.
    u_stress: Optional[torch.Tensor] = None  # (Ngrid,) E-W surface stress [N/m2]
    v_stress: Optional[torch.Tensor] = None  # (Ngrid,) N-S surface stress [N/m2]

    # Optional lowest-level tendencies used by the offline driver when it
    # receives both current and next atmospheric forcing records.
    u_tendency: Optional[torch.Tensor] = None  # (Ngrid,) zonal wind tendency [m/s2]
    v_tendency: Optional[torch.Tensor] = None  # (Ngrid,) meridional wind tendency [m/s2]
    sl_tendency: Optional[torch.Tensor] = None  # (Ngrid,) dry static energy tendency [J/kg/s]
    q_tendency: Optional[torch.Tensor] = None   # (Ngrid,) specific humidity tendency [kg/kg/s]

    # Optional radiative-time skin temperature PTSKRAD. The offline driver
    # carries this diagnostic across radiation and surface substeps.
    radiative_skin_t: Optional[torch.Tensor] = None  # (Ngrid,) [K]

    # Optional authoritative tiled VUPDZ0 boundary from an atmospheric
    # diffusion driver.  These fields support strict offline comparison;
    # absent values preserve the self-contained surface-layer path.
    vupdz0_pustrti: Optional[torch.Tensor] = None  # (Ngrid, Ntile) [N/m2]
    vupdz0_pvstrti: Optional[torch.Tensor] = None  # (Ngrid, Ntile) [N/m2]
    vupdz0_pahfsti: Optional[torch.Tensor] = None  # (Ngrid, Ntile) [W/m2]
    vupdz0_pevapti: Optional[torch.Tensor] = None  # (Ngrid, Ntile) [kg/m2/s]
    vupdz0_z0mti: Optional[torch.Tensor] = None    # (Ngrid, Ntile) momentum roughness [m]
    vupdz0_z0hti: Optional[torch.Tensor] = None    # (Ngrid, Ntile) heat roughness [m]
    vupdz0_z0qti: Optional[torch.Tensor] = None    # (Ngrid, Ntile) moisture roughness [m]
    vupdz0_pbuomti: Optional[torch.Tensor] = None  # (Ngrid, Ntile) [m2/s3]
    vupdz0_pzdlti: Optional[torch.Tensor] = None   # (Ngrid, Ntile) [-]
    vupdz0_praqti: Optional[torch.Tensor] = None   # (Ngrid, Ntile) [s/m]

    # Optional authoritative VDFDIFH1S output at the land-atmosphere
    # coupling point.  A coupled atmospheric model owns these fields;
    # supplying the complete set lets the land prognostics be verified
    # independently of a VDF port while retaining its diagnostics.
    vdf_pfrti: Optional[torch.Tensor] = None       # (Ngrid, Ntile) [-]
    vdf_pahfsti: Optional[torch.Tensor] = None     # (Ngrid, Ntile) [W/m2]
    vdf_pevapti: Optional[torch.Tensor] = None     # (Ngrid, Ntile) [kg/m2/s]
    vdf_pslrflti: Optional[torch.Tensor] = None    # (Ngrid, Ntile) [W/m2]
    vdf_ptsktip1: Optional[torch.Tensor] = None    # (Ngrid, Ntile) [K]

    # Snow-under-forest evaporation is formed by SURFEXCDRIVER outside the
    # VDFDIFH1S output set.  An atmospheric caller may supply it as an
    # authoritative boundary for coupled trajectory validation.
    snow_under_forest_evap: Optional[torch.Tensor] = None  # (Ngrid,) [kg/m2/s]
    tiled_shortwave: Optional[torch.Tensor] = None  # (Ngrid, Ntile) [W/m2]


    # Optional: atmospheric profiles for explicit diffusion above the
    # surface layer (VDFMAIN1S args PUM1/.../PAPHM1, full KLEV axis).
    # None by default -- the surface-only mode ignores these.
    u_profile: Optional[torch.Tensor] = None   # (Ngrid, Nlev_atm)
    v_profile: Optional[torch.Tensor] = None   # (Ngrid, Nlev_atm)
    t_profile: Optional[torch.Tensor] = None   # (Ngrid, Nlev_atm)
    q_profile: Optional[torch.Tensor] = None   # (Ngrid, Nlev_atm)
    p_profile: Optional[torch.Tensor] = None   # (Ngrid, Nlev_atm+1) half-levels

    def to(self, *, dtype=None, device=None) -> "Forcing":
        """Move every tensor to dtype/device (None keeps current)."""
        def _mv(x):
            if x is None:
                return None
            return x.to(dtype=dtype or x.dtype, device=device or x.device)
        return Forcing(
            u=_mv(self.u), v=_mv(self.v), t=_mv(self.t), q=_mv(self.q),
            co2=_mv(self.co2), p_surf=_mv(self.p_surf), geom1=_mv(self.geom1),
            fr_so=_mv(self.fr_so), fr_th=_mv(self.fr_th), mu0=_mv(self.mu0),
            rain=_mv(self.rain), snow=_mv(self.snow), dt=_mv(self.dt),
            buom=_mv(self.buom), ucurr=_mv(self.ucurr), vcurr=_mv(self.vcurr),
            i10fgcv=_mv(self.i10fgcv),
            u_stress=_mv(self.u_stress), v_stress=_mv(self.v_stress),
            sl_tendency=_mv(self.sl_tendency), q_tendency=_mv(self.q_tendency),
            radiative_skin_t=_mv(self.radiative_skin_t),
            vupdz0_pustrti=_mv(self.vupdz0_pustrti),
            vupdz0_pvstrti=_mv(self.vupdz0_pvstrti),
            vupdz0_pahfsti=_mv(self.vupdz0_pahfsti),
            vupdz0_pevapti=_mv(self.vupdz0_pevapti),
            vupdz0_z0mti=_mv(self.vupdz0_z0mti),
            vupdz0_z0hti=_mv(self.vupdz0_z0hti),
            vupdz0_z0qti=_mv(self.vupdz0_z0qti),
            vupdz0_pbuomti=_mv(self.vupdz0_pbuomti),
            vupdz0_pzdlti=_mv(self.vupdz0_pzdlti),
            vupdz0_praqti=_mv(self.vupdz0_praqti),
            vdf_pfrti=_mv(self.vdf_pfrti),
            vdf_pahfsti=_mv(self.vdf_pahfsti),
            vdf_pevapti=_mv(self.vdf_pevapti),
            vdf_pslrflti=_mv(self.vdf_pslrflti),
            vdf_ptsktip1=_mv(self.vdf_ptsktip1),
            snow_under_forest_evap=_mv(self.snow_under_forest_evap),
            tiled_shortwave=_mv(self.tiled_shortwave),
            u_profile=_mv(self.u_profile), v_profile=_mv(self.v_profile),
            t_profile=_mv(self.t_profile), q_profile=_mv(self.q_profile),
            p_profile=_mv(self.p_profile),
        )


# --------------------------------------------------------------------------- #
# Static (per-column, time-invariant) descriptors
# --------------------------------------------------------------------------- #
@dataclass
class Static:
    """Per-column static descriptors.

    These come from climatology / ancillary files and are *constant*
    during a model run -- but they may still be optimisable in the ML
    setting, so they live on the model as Parameters/buffers depending
    on the learnable flag in :class:`SurfConfig`.

    All tensor fields are ``(Ngrid,)`` unless noted.
    """

    # Tile coverage fractions summing to 1 over the 8 base tiles
    # (water / ice / wet-skin / dry-low-veg / snow-low-veg / dry-high-veg
    # / snow-high-veg / bare-soil).  These are the *base* fractions;
    # the actual per-step tile fractions PFRTI are recomputed inside
    # the surface driver from snow cover and skin water.
    tile_frac_base: torch.Tensor          # (Ngrid, 8)

    # Land-sea mask and orography.
    land_frac: torch.Tensor               # (Ngrid,)  land fraction (0..1)
    orography: torch.Tensor               # (Ngrid,)  geopotential [m2/s2]

    # Soil texture & vegetation type indices (used to look up
    # tables/soil.py and tables/veg.py).
    soil_type: torch.Tensor               # (Ngrid,)  long, 0..7
    veg_type: torch.Tensor                # (Ngrid,)  long, 0..20

    # Soil layer thicknesses (m), constant across columns but kept as
    # a tensor for convenience (broadcast to (Ngrid, Nsoil)).
    soil_thickness: torch.Tensor          # (Nsoil,) or (Ngrid, Nsoil)
    orography_stddev: Optional[torch.Tensor] = None  # (Ngrid,) SDOR [m]

    # Albedo background (visible / near-IR, dry / wet skin) from
    # SURFBC.  Optional here -- surfaced by SURFRAD.
    albd: Optional[torch.Tensor] = None   # (Ngrid,)
    albw: Optional[torch.Tensor] = None   # (Ngrid,)
    emis: Optional[torch.Tensor] = None   # (Ngrid,)  surface emissivity

    # Vegetation cover fractions (PCVL / PCVH from SURFBC).  When
    # provided, the model recomputes the per-step tile fractions PFRTI
    # from these covers + the prognostic snow cover / skin reservoir
    # (the faithful ``surfbc_ctl_mod.F90`` land-branch split).  When
    # absent the model falls back to ``tile_frac_base`` as a static
    # approximation -- this preserves the cold-start behaviour of every
    # existing test.  All five cover/LAI fields must be set together
    # (the SURFBC split keys off ``cvl_frac`` alone for activation).
    cvl_frac: Optional[torch.Tensor] = None   # (Ngrid,)  PCVL low-veg cover 0..1
    cvh_frac: Optional[torch.Tensor] = None   # (Ngrid,)  PCVH high-veg cover 0..1
    ice_land_frac: Optional[torch.Tensor] = None  # (Ngrid,) PCIL land-ice 0..1
    lai_low: Optional[torch.Tensor] = None    # (Ngrid,)  PLAIL low-veg LAI
    lai_high: Optional[torch.Tensor] = None   # (Ngrid,)  PLAIH high-veg LAI

    # Per-column low / high vegetation type indices (1..20).  When
    # absent, the model uses ``veg_type`` for both (the cold-start
    # dominant-type approximation).  Surfbc uses separate PCVL/PCVH
    # cover but a single RVLAI lookup per class, so the type split is
    # only consumed by VegTable gathers that key off the class.
    veg_type_low: Optional[torch.Tensor] = None   # (Ngrid,) long
    veg_type_high: Optional[torch.Tensor] = None  # (Ngrid,) long

    # ---- Urban canyon fields (YOS_URB) ---- #
    # ``urban_frac`` (PCUR) is both the canyon-albedo input AND the
    # activation sentinel for the urban tile (tile 9 / F90 tile 10):
    # when None, the SURFBC urban tile branch stays inert (tile fraction
    # stays 0) and the surface-albedo subsystem sees no urban tile -- so
    # forward() is bit-identical to the build before the urban work.  The
    # geometry / material fields each fall back to the YOS_URB defaults
    # (RHWR=1.0, RWRR=1.0, roof/road/wall albedo 0.16/0.05/0.6).
    urban_frac: Optional[torch.Tensor] = None        # PCUR    urban cover fraction [-] (Ngrid,)
    urban_height_ratio: Optional[torch.Tensor] = None  # RHWR  height/roadwidth [-]
    urban_road_ratio: Optional[torch.Tensor] = None  # RWRR  road/(road+building) [-]
    roof_alb: Optional[torch.Tensor] = None          # RROOALB roof albedo  [-]
    road_alb: Optional[torch.Tensor] = None          # RROAALB road albedo  [-]
    wall_alb: Optional[torch.Tensor] = None          # RWALALB wall albedo  [-]

    # Roughness lengths for the Monin-Obukhov scheme (PZ0MM / PZ0HM /
    # PZ0QM).  These are grid-mean values; the upstream VDFEXCS driver
    # tiles them only inside the per-tile loop.  Required for M4
    # (``physics.exchange``); fall back to neutral defaults otherwise.
    z0m: Optional[torch.Tensor] = None    # (Ngrid,)  momentum roughness [m]
    z0h: Optional[torch.Tensor] = None    # (Ngrid,)  heat roughness     [m]
    z0q: Optional[torch.Tensor] = None    # (Ngrid,)  moisture roughness [m]

    # Latitude / longitude for astronomical / radiation computations.
    lat: Optional[torch.Tensor] = None    # (Ngrid,)  radians
    lon: Optional[torch.Tensor] = None    # (Ngrid,)  radians

    def to(self, *, dtype=None, device=None) -> "Static":
        def _mv(x):
            if x is None:
                return None
            if torch.is_floating_point(x):
                return x.to(dtype=dtype or x.dtype, device=device or x.device)
            return x.to(device=device or x.device)
        return Static(
            tile_frac_base=_mv(self.tile_frac_base),
            land_frac=_mv(self.land_frac),
            orography=_mv(self.orography),
            orography_stddev=_mv(self.orography_stddev),
            soil_type=_mv(self.soil_type),
            veg_type=_mv(self.veg_type),
            soil_thickness=_mv(self.soil_thickness),
            albd=_mv(self.albd), albw=_mv(self.albw), emis=_mv(self.emis),
            z0m=_mv(self.z0m), z0h=_mv(self.z0h), z0q=_mv(self.z0q),
            lat=_mv(self.lat), lon=_mv(self.lon),
            cvl_frac=_mv(self.cvl_frac), cvh_frac=_mv(self.cvh_frac),
            ice_land_frac=_mv(self.ice_land_frac),
            lai_low=_mv(self.lai_low), lai_high=_mv(self.lai_high),
            veg_type_low=_mv(self.veg_type_low),
            veg_type_high=_mv(self.veg_type_high),
            urban_frac=_mv(self.urban_frac),
            urban_height_ratio=_mv(self.urban_height_ratio),
            urban_road_ratio=_mv(self.urban_road_ratio),
            roof_alb=_mv(self.roof_alb),
            road_alb=_mv(self.road_alb),
            wall_alb=_mv(self.wall_alb),
        )


# --------------------------------------------------------------------------- #
# Prognostic surface state (VDFMAIN1S UPDATED/OUTPUT fields)
# --------------------------------------------------------------------------- #
@dataclass
class SurfState:
    """Prognostic surface state -- updated each timestep.

    Mirrors the IFS surf-package prognostic set (``SURF`` "TEND" output
    group from ``VDFMAIN1S``).  All quantities are at the start of the
    coupling interval; the model advances them in place.

    Tensor shapes:
        skin temperature, soil temperature:  ``(Ngrid, Ntile)`` for skin,
        ``(Ngrid, Nsoil)`` for soil; tile-axis is explicit so the
        Penman-Monteith elimination can be vectorised.

    Initialisation is performed by the model from ``Static`` and a
    climatology (cold-start).  For site validation the offline driver
    reads the initial state from a forcing file -- see
    ``tests/reference`` for the gold initial conditions.
    """

    # Skin temperature per tile (PTSKTI).
    skin_t: torch.Tensor                 # (Ngrid, Ntile)

    # Soil temperature profile (PTSAM1M).
    soil_t: torch.Tensor                 # (Ngrid, Nsoil)

    # Soil moisture (liquid + ice) per layer (PWSAM1M).
    soil_w_liq: torch.Tensor             # (Ngrid, Nsoil)

    # Snow water equivalent and snow cover fraction (PSNS, PNEIJ).
    snow_w: torch.Tensor                 # (Ngrid,)
    snow_frac: torch.Tensor              # (Ngrid,)

    # Canopy / skin reservoir water (PWL, intercepted water, [m of water]).
    skin_water: torch.Tensor             # (Ngrid,)

    # Optional companion fields (None by default).
    soil_w_ice: Optional[torch.Tensor] = None   # (Ngrid, Nsoil)

    # Sea-surface / sea-ice skin (PSST, PTSN/PTIA) -- only used by the
    # water / ice tiles; the land-surface core can ignore them.
    sst: Optional[torch.Tensor] = None           # (Ngrid,)
    snow_t: Optional[torch.Tensor] = None        # (Ngrid,) PTSNM1M
    snow_rho: Optional[torch.Tensor] = None      # (Ngrid,) PRSNM1M
    # Snow albedo (PASNM1M).  Doubles as the activation sentinel for the
    # surface-albedo subsystem: when ``None`` the subsystem stays dormant
    # (alpha = 0 everywhere) and forward() is bit-identical to the build
    # before the albedo work was wired in.  Same pattern as Static.cvl_frac.
    snow_alb: Optional[torch.Tensor] = None      # (Ngrid,) PASNM1M [-]
    ice_t: Optional[torch.Tensor] = None         # (Ngrid, Nice_layers)

    # Gridpoint skin temperature at the latest radiation timestep
    # (PTSKRAD/PTL).  This cannot in general be reconstructed from skin_t
    # after snow cover changes, because the radiation step used the tile
    # fractions from before that prognostic update.
    radiative_skin_t: Optional[torch.Tensor] = None  # (Ngrid,) [K]

    # FLake lake state -- delayed to a stub.
    lake_t: Optional[torch.Tensor] = None        # (Ngrid,)

    # VUPDZ0 requires the previous SURFPP fluxes.  They are integration
    # metadata, not physical prognostics, but belong to the trajectory
    # state so independent calls cannot inherit another rollout's history.
    vupdz0_step: int = 0
    vupdz0_pustrti: Optional[torch.Tensor] = None
    vupdz0_pvstrti: Optional[torch.Tensor] = None
    vupdz0_pahfsti: Optional[torch.Tensor] = None
    vupdz0_pevapti: Optional[torch.Tensor] = None
    vupdz0_pzdl: Optional[torch.Tensor] = None

    def to(self, *, dtype=None, device=None) -> "SurfState":
        def _mv(x):
            if x is None:
                return None
            return x.to(dtype=dtype or x.dtype, device=device or x.device)
        return SurfState(
            skin_t=_mv(self.skin_t), soil_t=_mv(self.soil_t),
            soil_w_liq=_mv(self.soil_w_liq),
            snow_w=_mv(self.snow_w), snow_frac=_mv(self.snow_frac),
            skin_water=_mv(self.skin_water),
            soil_w_ice=_mv(self.soil_w_ice),
            sst=_mv(self.sst), snow_t=_mv(self.snow_t),
            snow_rho=_mv(self.snow_rho), snow_alb=_mv(self.snow_alb),
            ice_t=_mv(self.ice_t),
            radiative_skin_t=_mv(self.radiative_skin_t),
            lake_t=_mv(self.lake_t),
            vupdz0_step=self.vupdz0_step,
            vupdz0_pustrti=_mv(self.vupdz0_pustrti),
            vupdz0_pvstrti=_mv(self.vupdz0_pvstrti),
            vupdz0_pahfsti=_mv(self.vupdz0_pahfsti),
            vupdz0_pevapti=_mv(self.vupdz0_pevapti),
            vupdz0_pzdl=_mv(self.vupdz0_pzdl),
        )

    def like(self, **kwargs) -> "SurfState":
        """Return a copy with selected fields replaced (chainable)."""
        out = SurfState(**{**self.__dict__, **kwargs})
        return out


# --------------------------------------------------------------------------- #
# Diagnostic / output state (fluxes, etc.)
# --------------------------------------------------------------------------- #
@dataclass
class SurfDiag:
    """Diagnostic output (non-prognostic) computed by the surface step.

    These are the VDFMAIN1S *OUTPUT* arrays: turbulent fluxes, roughness
    lengths, 2 m / 10 m diagnostics.  Held in a separate container so
    the prognostic state stays compact.
    """

    # Roughness lengths (PZ0M / PZ0H).
    z0m: Optional[torch.Tensor] = None   # (Ngrid,)
    z0h: Optional[torch.Tensor] = None   # (Ngrid,)

    # Sensible + latent heat flux per tile (PAHFSTI / PEVAPTI).
    h_flux_tile: Optional[torch.Tensor] = None   # (Ngrid, Ntile) [W/m^2]
    e_flux_tile: Optional[torch.Tensor] = None   # (Ngrid, Ntile) [kg/m^2/s]

    # Per-tile radiation / ground / latent-heat fluxes (used for SEB
    # closure checks).  All positive downward (see
    # :mod:`surf_pytorch.physics.seb`).
    str_tile: Optional[torch.Tensor] = None      # (Ngrid, Ntile) net LW [W/m^2]
    g0_tile: Optional[torch.Tensor] = None       # (Ngrid, Ntile) ground H [W/m^2]
    slh_tile: Optional[torch.Tensor] = None      # (Ngrid, Ntile) latent H [W/m^2]

    # Grid-mean fluxes (PFTLHEV / PFTLHSB / PFWSB).
    h_flux: Optional[torch.Tensor] = None        # (Ngrid,) [W/m^2]
    e_flux: Optional[torch.Tensor] = None        # (Ngrid,) [kg/m^2/s]
    sub_flux: Optional[torch.Tensor] = None      # (Ngrid,) runoff [kg/m^2/s]

    # 2 m / 10 m diagnostics.
    t2m: Optional[torch.Tensor] = None           # (Ngrid,) [K]
    q2m: Optional[torch.Tensor] = None           # (Ngrid,) [kg/kg]
    u10m: Optional[torch.Tensor] = None          # (Ngrid,) [m/s]
    v10m: Optional[torch.Tensor] = None          # (Ngrid,) [m/s]

    # Surface energy balance closure diagnostics.
    net_rad: Optional[torch.Tensor] = None       # (Ngrid,) [W/m^2]
    ground_heat: Optional[torch.Tensor] = None   # (Ngrid,) [W/m^2]

    # Carbon flux diagnostics (populated only when the Farquhar
    # photosynthesis port is active -- ``SurfConfig.use_farquhar``).  All
    # per-grid (Ngrid,), evaluated at the lowest model level following the
    # upstream FARQUHAR convention (farquhar_mod.F90).  Held in ``SurfDiag``
    # because photosynthesis is diagnostic here: it does not feed back on
    # the prognostic skin-T / soil state in the current single-step
    # forward (a future milestone will close the stomatal-conductance loop
    # against the VEVAP alpha chain).
    farquhar_pgstot: Optional[torch.Tensor] = None  # canopy cond. [m/s]
    farquhar_pgpp:   Optional[torch.Tensor] = None  # GPP [kgC/m^2/s]
    farquhar_prd:    Optional[torch.Tensor] = None  # dark respiration [kgC/m^2/s]
    farquhar_pan:    Optional[torch.Tensor] = None  # net assimilation [kgC/m^2/s]


__all__ = ["Forcing", "Static", "SurfState", "SurfDiag"]
