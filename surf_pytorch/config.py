"""Top-level configuration for the land-surface model.

``SurfConfig`` is the single object a user passes to
:class:`surf_pytorch.LandSurface`.  It bundles:

* dimensioning integers (n_soil, n_tile, etc.) -- kept in sync with
  the offline driver's ``sudim1s.F90`` defaults;
* feature flags (which tiles are active, is differentiation enabled);
* the default dtype / device;
* the per-table learnable overrides (which LUTs may be optimised).

All defaults match the offline single-column driver so that ``import
surf_pytorch; m = LandSurface()`` is ready to plug into the
single-column comparison harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


# --------------------------------------------------------------------------- #
# Tile indices -- matches yoeswmt / surf_driver_defs.h
# The 8 standard IFS tiles; lake and urban are optional extensions.
#
# IMPORTANT: this port is 0-based, upstream Fortran surf* is 1-based.
# `F90_TILE_OF[port_idx]` gives the matching Fortran JT (== port_idx + 1).
# This map is the single authoritative source -- seb.py / vlamsk.py /
# exchange.py reference it instead of restating the offset in comments,
# which is where a previous version had a stale "tile 3" / index 2
# mismatch in _tile_solar_transmission.
# --------------------------------------------------------------------------- #
TILE_WATER        = 0
TILE_ICE          = 1
TILE_WET_SKIN     = 2
TILE_LOW_VEG_DRY  = 3
TILE_LOW_VEG_SNOW = 4
TILE_HIGH_VEG_DRY = 5
TILE_HIGH_VEG_SNOW= 6
TILE_BARE_SOIL    = 7
TILE_LAKE         = 8
TILE_URBAN        = 9
N_BASE_TILES      = 8

# port tile index -> upstream Fortran JT (1-based).  Single source of truth
# for the 0/1-based offset that used to be scattered across CASE comments.
F90_TILE_OF = {
    TILE_WATER: 1,         # ocean
    TILE_ICE: 2,           # sea ice
    TILE_WET_SKIN: 3,      # wet skin / interception
    TILE_LOW_VEG_DRY: 4,   # low vegetation, snow-free
    TILE_LOW_VEG_SNOW: 5,  # snow on low vegetation
    TILE_HIGH_VEG_DRY: 6,  # high vegetation, snow-free
    TILE_HIGH_VEG_SNOW: 7, # snow under high vegetation
    TILE_BARE_SOIL: 8,     # bare soil
    TILE_LAKE: 9,          # lake (optional)
    TILE_URBAN: 10,        # urban (optional)
}
# Human-readable names indexed by port tile id (mirrors F90_TILE_OF order).
TILE_NAMES = {
    TILE_WATER: "water",
    TILE_ICE: "ice",
    TILE_WET_SKIN: "wet_skin",
    TILE_LOW_VEG_DRY: "low_veg_dry",
    TILE_LOW_VEG_SNOW: "low_veg_snow",
    TILE_HIGH_VEG_DRY: "high_veg_dry",
    TILE_HIGH_VEG_SNOW: "high_veg_snow",
    TILE_BARE_SOIL: "bare_soil",
    TILE_LAKE: "lake",
    TILE_URBAN: "urban",
}


@dataclass
class SurfConfig:
    """Configuration for :class:`surf_pytorch.LandSurface`.

    Fields are grouped by purpose; default values reproduce the offline
    driver configuration.  Changing ``n_soil`` or the tile count
    invalidates any cached state and the model rebuilds its tables.
    """

    # ----- Dimensions -----
    n_soil: int = 4                # HTESSEL default soil layers
    n_snow: int = 1                # single-layer snow (multi-layer stub)
    n_tile: int = N_BASE_TILES     # 8 base tiles; set 10 for lake+urban
    n_soil_types: int = 7          # NSOTY from sussoil
    n_veg_types: int = 20          # NVTYPES from susveg

    # ----- Soil layer thicknesses (m), IFS cy48r1 default -----
    soil_thickness: tuple = (0.07, 0.21, 0.72, 1.89)

    # ----- Physics feature flags -----
    # When True, lake / urban tiles are added to the tile dimension.
    use_lake: bool = False
    use_urban: bool = False

    # When True, build the FLake lake model; otherwise leave a stub
    # that returns SST as the lake skin temperature.
    use_flake: bool = False

    # When True, use the KPP ocean mixed-layer; otherwise the water
    # tile uses a prescribed SST.
    use_kpp: bool = False

    # When True, run the Farquhar photosynthesis model and expose its
    # carbon-flux diagnostics.  This mirrors LEFARQUHAR.  Its canopy
    # conductance affects moisture fluxes only when use_ags is also True.
    use_farquhar: bool = False

    # When True, feed the canopy resistance returned by COTWORESTRESS
    # into VEVAP (LEAGS).  This is intentionally independent of
    # use_farquhar: OpenIFS permits LEFARQUHAR=.TRUE., LEAGS=.FALSE.,
    # which computes carbon diagnostics while retaining Jarvis PWET*.
    use_ags: bool = False

    # LESN09 selects the cy40+ snow constants in several SURF helper
    # routines.  OpenIFS 48r1 offline SURF prints LESN09=T by default;
    # VLAMSK uses it to choose RQSNCR=1/10 m for the forest-snow
    # stability cap.
    lesn09: bool = True

    # LEVGEN selects the Van Genuchten field-capacity suction used while
    # constructing the soil lookup table.  This must match the calling
    # OpenIFS/OSM namelist because it affects heat capacity, conductivity,
    # and the soil-water prognostic coefficients.
    levgen: bool = True

    # When True, the VDFDIFH1S snow-melt repeat (call #4, LESNML) is
    # invoked whenever a low-veg-snow tile is at melting point; this
    # pins TSRF to RTT and LAMSK to 100 W/m^2/K and re-runs SURFSEB.
    # Default False -- the offline-driver default is no snow tile so
    # the repeat is a no-op; flip to True once snow prognostics (M6)
    # populate the low-veg-snow fraction.
    lesnml: bool = False

    # When True (default), the soil-moisture path uses the VIC saturation-
    # excess runoff formulation (LESSRO=.TRUE. in srfwexc_vg); when False
    # the infiltration-limit formulation is used.  The offline-driver
    # default is VIC.
    lessro: bool = True

    # When True, use the OpenIFS offline default LESOILCOND path in SRFT:
    # temperature-dependent freeze fraction plus the FSOILTCOND statement
    # function.  Set False only for the older SURF_REF_SOILT regression
    # helper, which transcribes the legacy ZFF=0 conductivity block.
    lesoilcond: bool = True

    # LELAIV controls the LAI source used by SURFBC. The supported offline
    # forcing cases set LELAIV=.TRUE., so the monthly climatology carried by
    # Static is used by default. Set False only for a Fortran namelist that
    # explicitly selects the vegetation lookup-table LAI.
    lelaiv: bool = True

    # ----- Differentiability -----
    # When False (default), every physics routine runs under no_grad
    # for speed; the model can still be advanced step-by-step.
    # When True, the same code runs under enable_grad so a downstream
    # loss can backprop through the surface step.  Lookup indices and
    # branches are detached regardless (see utils.detach_index).
    differentiable: bool = False

    # When True, the soil/veg LUTs are registered as parameters and may
    # be optimised during ML training; the table dictionaries in this
    # config select which columns get the learnable treatment.
    learnable_soil: tuple = ()     # names from SoilTable buffer set
    learnable_veg: tuple = ()

    # ----- Numeric -----
    dtype: torch.dtype = torch.float64   # bit-match Fortran double
    device: Optional[str] = None         # None -> default device

    # VEXCS secant-iteration count after the back-substitution.  The
    # upstream SURFEXCDRIVER call uses 3 by default (cold start); the
    # port exposes it here so warm-start tests can drop it to 0..1.
    vexcs_kitt: int = 3

    # ``offline`` owns the ported OSM VUPDZ0/VDFMAIN1S closure.
    # ``coupled`` requires authoritative tiled atmospheric-VDF fields.
    # No approximate VUP boundary mode is retained.
    vup_boundary_mode: str = "offline"
    # Longwave-radiation switches mirrored from SURFSEB/SURFTSTP.
    # LELWDD controls dLW/dT linearisation around the radiation skin
    # temperature; LELWTL recomputes tiled LW and must also feed tiled
    # PSLRFLTI into the full SURFTSTP snow path.
    lelwdd: bool = True
    # The value is selected by the calling OpenIFS/OSM configuration.
    lelwtl: bool = True

    # ----- Validation -----
    # When True, the model asserts invariants (tile fractions sum to 1,
    # moisture within [0, wsat], etc.) at the end of each step.  Cheap
    # and catches porting bugs early; disable for production speed.
    validate: bool = True

    # Pack tile and grid-mean flux diagnostics on every forward step.
    # Forecast and throughput workflows that consume only the prognostic
    # state can disable this to avoid duplicate tile-fraction reductions.
    # The default preserves the public API and diagnostic behaviour.
    compute_diagnostics: bool = True

    def n_active_tiles(self) -> int:
        """Number of active tiles including optional lake/urban."""
        n = N_BASE_TILES
        if self.use_lake:
            n += 1
        if self.use_urban:
            n += 1
        if n != self.n_tile:
            raise ValueError(
                f"n_tile={self.n_tile} but active tiles count {n} "
                f"(use_lake={self.use_lake}, use_urban={self.use_urban})"
            )
        return n


__all__ = ["SurfConfig", "TILE_WATER", "TILE_ICE", "TILE_WET_SKIN",
           "TILE_LOW_VEG_DRY", "TILE_LOW_VEG_SNOW", "TILE_HIGH_VEG_DRY",
           "TILE_HIGH_VEG_SNOW", "TILE_BARE_SOIL", "TILE_LAKE", "TILE_URBAN",
           "N_BASE_TILES"]
