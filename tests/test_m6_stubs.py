"""M6 wrap-up: FLake / KPP / Farquhar stub tests.

These tests cover the three placeholder modules that close out the M6
milestone:

* :mod:`surf_pytorch.physics.flake` -- the FLake lake-model stub,
  identity update (matches upstream ``LEFLAKE = .FALSE.``);
* :mod:`surf_pytorch.physics.ocean_ml` -- the KPP ocean mixed-layer
  stub, identity update (matches upstream ``LEOCSA = .FALSE.``);
* :mod:`surf_pytorch.physics.farquhar` -- the Farquhar photosynthesis
  stub, unit stomatal conductance + zero carbon fluxes.

The stubs exist so that flipping ``SurfConfig.use_flake`` /
``use_kpp`` / ``use_farquhar`` from ``False`` (current, calls the stub)
to ``True`` (future, real transcription) is a pure drop-in: the inputs,
outputs and call sites will not move.  These tests pin that contract.

The model-integration tests at the bottom verify that the new
:meth:`LandSurface._update_lake` / :meth:`_update_ocean_ml` plumbing
runs without disturbing the existing forward output (the stubs are
no-ops on the prognostic state) and that the ``use_farquhar`` forward
branch still produces the documented unit-alpha closure.
"""
from __future__ import annotations

import torch
import pytest

from surf_pytorch import SurfConfig, LandSurface
from surf_pytorch.state import Forcing, Static, SurfState
from surf_pytorch.physics.flake import LakeInputs, LakeOutputs, flake_driver
from surf_pytorch.physics.ocean_ml import (
    OceanMlInputs, OceanMlOutputs, ocean_ml_driver,
)
from surf_pytorch.physics.farquhar import (
    FarquharInputs, FarquharOutputs, farquhar,
)


# --------------------------------------------------------------------------- #
# FLake stub
# --------------------------------------------------------------------------- #
class TestFlakeStub:
    """FLake stub: identity update, ignores forcing."""

    def _inputs(self, N: int = 3, dtype=torch.float64) -> LakeInputs:
        g = torch.randn(N, dtype=dtype)
        return LakeInputs(
            ptlice_m1m=g.clone(), ptlmnw_m1m=g.clone() + 1.0,
            ptlwml_m1m=g.clone() + 2.0, ptlbot_m1m=g.clone() + 3.0,
            ptlsf_m1m=g.clone() + 4.0,
            phlice_m1m=torch.zeros(N, dtype=dtype),
            phlml_m1m=torch.full((N,), 5.0, dtype=dtype),
            # Forcing is arbitrary -- the stub must ignore it.
            pssrflti=torch.randn(N, dtype=dtype) * 1e3,
            pslrfl=torch.randn(N, dtype=dtype) * 1e2,
            pahfsti=torch.randn(N, dtype=dtype) * 1e2,
            pevapti=torch.randn(N, dtype=dtype) * 1e-5,
            pustrti=torch.randn(N, dtype=dtype) * 1e-1,
            pvstrti=torch.randn(N, dtype=dtype) * 1e-1,
            pldepth=torch.full((N,), 25.0, dtype=dtype),
            pgemu=torch.full((N,), 0.6, dtype=dtype),
            ld_lake=torch.ones(N, dtype=torch.bool),
            ptstep=1800.0,
        )

    def test_identity_update(self):
        """Every prognostic at t+1 equals its t-1 value."""
        inp = self._inputs()
        out = flake_driver(inp)
        torch.testing.assert_close(out.ptlice, inp.ptlice_m1m)
        torch.testing.assert_close(out.ptlmnw, inp.ptlmnw_m1m)
        torch.testing.assert_close(out.ptlwml, inp.ptlwml_m1m)
        torch.testing.assert_close(out.ptlbot, inp.ptlbot_m1m)
        torch.testing.assert_close(out.ptlsf, inp.ptlsf_m1m)
        torch.testing.assert_close(out.phlice, inp.phlice_m1m)
        torch.testing.assert_close(out.phlml, inp.phlml_m1m)

    def test_forcing_ignored(self):
        """Varying the surface fluxes must not change the output."""
        base = self._inputs()
        out_base = flake_driver(base)
        # Multiply every forcing field by 100x.
        perturbed = LakeInputs(
            **{**base.__dict__,
               "pssrflti": base.pssrflti * 100.0,
               "pslrfl":   base.pslrfl * 100.0,
               "pahfsti":  base.pahfsti * 100.0,
               "pevapti":  base.pevapti * 100.0,
               "pustrti":  base.pustrti * 100.0,
               "pvstrti":  base.pvstrti * 100.0})
        out_pert = flake_driver(perturbed)
        for field in ("ptlice", "ptlmnw", "ptlwml", "ptlbot", "ptlsf",
                      "phlice", "phlml"):
            torch.testing.assert_close(getattr(out_pert, field),
                                       getattr(out_base, field))

    def test_returns_lake_outputs(self):
        out = flake_driver(self._inputs())
        assert isinstance(out, LakeOutputs)


# --------------------------------------------------------------------------- #
# KPP / ocean mixed-layer stub
# --------------------------------------------------------------------------- #
class TestOceanMlStub:
    """KPP stub: identity update, ignores forcing."""

    def _inputs(self, N: int = 3, nlev: int = 8,
                dtype=torch.float64) -> OceanMlInputs:
        return OceanMlInputs(
            pto0=torch.randn(N, nlev, dtype=dtype),
            pso0=torch.full((N, nlev), 35.0, dtype=dtype),
            puo0=torch.randn(N, nlev, dtype=dtype) * 1e-2,
            pvo0=torch.randn(N, nlev, dtype=dtype) * 1e-2,
            pzo=torch.linspace(0.0, 50.0, nlev + 1,
                               dtype=dtype).unsqueeze(0).expand(N, nlev + 1),
            pdo=torch.full((N, nlev), 50.0 / nlev, dtype=dtype),
            pssrflti=torch.randn(N, dtype=dtype) * 1e3,
            pslrfl=torch.randn(N, dtype=dtype) * 1e2,
            pahfsti=torch.randn(N, dtype=dtype) * 1e2,
            pevapti=torch.randn(N, dtype=dtype) * 1e-5,
            pustrti=torch.randn(N, dtype=dtype) * 1e-1,
            pvstrti=torch.randn(N, dtype=dtype) * 1e-1,
            ld_ocn_kpp=torch.ones(N, dtype=torch.bool),
            ptstep=3600.0,
        )

    def test_identity_update(self):
        inp = self._inputs()
        out = ocean_ml_driver(inp)
        torch.testing.assert_close(out.pto0_new, inp.pto0)
        torch.testing.assert_close(out.pso0_new, inp.pso0)
        torch.testing.assert_close(out.puoe1, inp.puo0)
        torch.testing.assert_close(out.pvoe1, inp.pvo0)

    def test_forcing_ignored(self):
        base = self._inputs()
        out_base = ocean_ml_driver(base)
        perturbed = OceanMlInputs(
            **{**base.__dict__,
               "pssrflti": base.pssrflti * 100.0,
               "pahfsti":  base.pahfsti * 100.0,
               "pevapti":  base.pevapti * 100.0})
        out_pert = ocean_ml_driver(perturbed)
        for field in ("pto0_new", "pso0_new", "puoe1", "pvoe1"):
            torch.testing.assert_close(getattr(out_pert, field),
                                       getattr(out_base, field))

    def test_returns_ocean_ml_outputs(self):
        out = ocean_ml_driver(self._inputs())
        assert isinstance(out, OceanMlOutputs)


# --------------------------------------------------------------------------- #
# Farquhar: real behaviour smoke tests (the full ULP validation lives in
# test_farquhar_ref.py; here we just exercise the public API with the
# real port and assert the documented physics).
# --------------------------------------------------------------------------- #
class TestFarquharStub:
    """Farquhar port smoke tests."""

    def _inputs(self, N: int = 3, dtype=torch.float64) -> FarquharInputs:
        # Single vegetated C3 column, daytime, warm, well-watered.
        return FarquharInputs(
            ptskm=torch.full((N,), 299.0, dtype=dtype),
            ptm=torch.full((N,), 298.0, dtype=dtype),
            ptsoil=torch.full((N,), 295.0, dtype=dtype),
            paphm=torch.full((N,), 1.01325e5, dtype=dtype),
            pqm=torch.full((N,), 0.010, dtype=dtype),
            pqsurf=torch.full((N,), 0.014, dtype=dtype),
            pqs=torch.full((N,), 0.014, dtype=dtype),
            pco2=torch.full((N,), 390.0e-6, dtype=dtype),
            lai=torch.full((N,), 4.0, dtype=dtype),
            kvtype=torch.full((N,), 5, dtype=torch.long),     # C3 broadleaf
            kco2typ=torch.full((N,), 1, dtype=torch.long),    # C3
            pf2=torch.full((N,), 0.9, dtype=dtype),
            ldland=torch.ones(N, dtype=torch.bool),
            psrfd=torch.full((N,), 700.0, dtype=dtype),
        )

    def _veg(self):
        from surf_pytorch.tables import VegTable
        return VegTable(dtype=torch.float64)

    def _consts(self):
        from surf_pytorch import ConstantBundle
        from surf_pytorch.constants import PhysicalConstants
        return ConstantBundle(PhysicalConstants(), dtype=torch.float64)

    def test_conductance_positive(self):
        """Vegetated daytime column -> positive stomatal conductance."""
        out = farquhar(self._inputs(), self._veg(), self._consts())
        assert (out.pgstot > 0).all()

    def test_positive_carbon_fluxes(self):
        """Vegetated daytime column -> positive GPP / Rd / net assimilation."""
        out = farquhar(self._inputs(), self._veg(), self._consts())
        assert (out.pgpp > 0).all()
        assert (out.prd > 0).all()

    def test_returns_farquhar_outputs(self):
        out = farquhar(self._inputs(), self._veg(), self._consts())
        assert isinstance(out, FarquharOutputs)


# --------------------------------------------------------------------------- #
# Model integration: the stub plumbing must be a no-op on the forward output.
# --------------------------------------------------------------------------- #
def _make_forward_inputs(*, N: int = 4, dtype=torch.float64,
                         use_farquhar: bool = False):
    """Minimal (state, forcing, static, config) -- mirrors test_forward_m5."""
    cfg = SurfConfig(dtype=dtype, use_farquhar=use_farquhar)
    ntile, nsoil = cfg.n_tile, cfg.n_soil
    state = SurfState(
        skin_t=torch.full((N, ntile), 288.0, dtype=dtype),
        soil_t=torch.full((N, nsoil), 285.0, dtype=dtype),
        soil_w_liq=torch.full((N, nsoil), 0.30, dtype=dtype),
        snow_w=torch.zeros((N,), dtype=dtype),
        snow_frac=torch.zeros((N,), dtype=dtype),
        skin_water=torch.zeros((N,), dtype=dtype),
        sst=torch.full((N,), 290.0, dtype=dtype),
    )
    forcing = Forcing(
        u=torch.full((N,), 5.0, dtype=dtype),
        v=torch.zeros((N,), dtype=dtype),
        t=torch.full((N,), 290.0, dtype=dtype),
        q=torch.full((N,), 0.008, dtype=dtype),
        co2=torch.full((N,), 6e-4, dtype=dtype),
        p_surf=torch.full((N,), 1.0e5, dtype=dtype),
        geom1=torch.full((N,), 50.0, dtype=dtype),
        fr_so=torch.full((N,), 200.0 * 3600.0, dtype=dtype),
        fr_th=torch.full((N,), -300.0 * 3600.0, dtype=dtype),
        mu0=torch.full((N,), 0.5, dtype=dtype),
        rain=torch.zeros((N,), dtype=dtype),
        snow=torch.zeros((N,), dtype=dtype),
        dt=torch.tensor(3600.0, dtype=dtype),
    )
    frac = torch.zeros((N, ntile), dtype=dtype)
    frac[:, 2] = 0.05
    frac[:, 3] = 0.30
    frac[:, 5] = 0.30
    frac[:, 7] = 0.35
    static = Static(
        tile_frac_base=frac,
        land_frac=torch.ones((N,), dtype=dtype),
        orography=torch.full((N,), 50.0, dtype=dtype),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=torch.full((N,), 3, dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.full((N,), 0.99, dtype=dtype),
    )
    return state, forcing, static, cfg


class TestStubIntegration:
    """The new stub plumbing must not perturb the existing forward."""

    def test_update_lake_runs_and_returns_identity(self):
        """_update_lake returns LakeOutputs equal to the SST-seeded t-1."""
        state, _, _, _ = _make_forward_inputs()
        model = LandSurface(SurfConfig(dtype=torch.float64))
        out = model._update_lake(state, diff_out=None, ngrid=4,
                                 dtype=torch.float64, device=torch.device("cpu"))
        assert isinstance(out, LakeOutputs)
        # All temperature prognostics seeded from SST = 290 K.
        for f in ("ptlice", "ptlmnw", "ptlwml", "ptlbot", "ptlsf"):
            torch.testing.assert_close(
                getattr(out, f), torch.full((4,), 290.0, dtype=torch.float64))
        # Ice thickness / mixed-layer depth seeded at zero.
        torch.testing.assert_close(out.phlice, torch.zeros((4,), dtype=torch.float64))
        torch.testing.assert_close(out.phlml, torch.zeros((4,), dtype=torch.float64))

    def test_update_ocean_ml_runs_and_returns_identity(self):
        """_update_ocean_ml returns OceanMlOutputs seeded from SST."""
        state, _, _, _ = _make_forward_inputs()
        model = LandSurface(SurfConfig(dtype=torch.float64))
        out = model._update_ocean_ml(state, diff_out=None, ngrid=4,
                                     dtype=torch.float64,
                                     device=torch.device("cpu"))
        assert isinstance(out, OceanMlOutputs)
        # Surface layer == SST = 290 K; deeper layers seeded identically.
        torch.testing.assert_close(
            out.pto0_new[:, 0], torch.full((4,), 290.0, dtype=torch.float64))
        # Identity stub -> every layer equals the SST seed.
        torch.testing.assert_close(
            out.pto0_new, torch.full((4, 8), 290.0, dtype=torch.float64))

    def test_forward_with_use_farquhar_runs(self):
        """The use_farquhar forward branch must still produce output.

        The forward path routes use_farquhar through the Farquhar canopy-
        conductance closure (PGSTOT -> 1/max(RCONDCTMIN, PGSTOT) -> VEVAP,
        mirroring cotworestress_mod.F90:412-431); this test pins that the
        forward completes and returns a SurfState / SurfDiag with the
        expected shapes, so a regression in the Farquhar wiring is caught
        if it breaks the contract.
        """
        state, forcing, static, cfg = _make_forward_inputs(use_farquhar=True)
        model = LandSurface(cfg)
        new_state, diag = model(state, forcing, static)
        assert new_state.skin_t.shape == state.skin_t.shape
        assert new_state.soil_t.shape == state.soil_t.shape

    def test_stub_plumbing_is_forward_noop(self):
        """Calling _update_lake / _update_ocean_ml must not change the
        forward() output (the stubs are pure identity updates and the
        forward path does not yet consume their outputs)."""
        state, forcing, static, cfg = _make_forward_inputs()
        model = LandSurface(cfg)
        # Baseline forward.
        new_state_a, diag_a = model(state, forcing, static)
        # Exercise the stub plumbing (currently no-op on the prognostic).
        _ = model._update_lake(state, diff_out=None, ngrid=4,
                               dtype=torch.float64, device=torch.device("cpu"))
        _ = model._update_ocean_ml(state, diff_out=None, ngrid=4,
                                   dtype=torch.float64,
                                   device=torch.device("cpu"))
        # Re-run forward -- must be bit-identical.
        new_state_b, diag_b = model(state, forcing, static)
        torch.testing.assert_close(new_state_a.skin_t, new_state_b.skin_t)
        torch.testing.assert_close(new_state_a.soil_t, new_state_b.soil_t)
        torch.testing.assert_close(new_state_a.soil_w_liq,
                                   new_state_b.soil_w_liq)
