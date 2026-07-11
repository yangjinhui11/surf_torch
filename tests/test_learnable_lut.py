"""Tests for the learnable-LUT ML-training API.

``SurfConfig.learnable_soil`` / ``learnable_veg`` promote selected LUT
columns from frozen ``nn.Module`` buffers to optimisable
``nn.Parameter``s so a downstream loss can backprop through the surface
step into the table -- the headline "differentiable nn.Module" use case
advertised in the README.

Two layers are covered:

1. **Registration** (``SoilTable`` / ``VegTable`` via the model ctor):
   named columns flip buffer -> Parameter, non-listed columns stay
   frozen, and the default config carries zero parameters.  This is the
   path that previously crashed (``model.py:150`` passed a ``set`` where
   ``SoilTable.__init__`` expects a ``dict``-with-``.get``); the
   ``dict.fromkeys(...)`` fix is pinned here.

2. **Gradient flow** through ``forward``: a veg-LUT column that drives
   the SEB / stomatal path (``rvlai``) receives a finite, non-zero
   gradient on skin temperature -- the API is genuinely trainable end
   to end.  Soil-LUT columns are documented as *not* training through
   the forward outputs: HTESSEL consumes them inside clamp bounds
   (``SRFWNG`` clips soil moisture to ``[wrest, wsat]``) and indexed
   lookups, so their forward-output gradient is structurally zero --
   this is the upstream physics, not a port bug, and is asserted as
   expected behaviour so a future "make wsat trainable" change is a
   deliberate decision rather than a silent drift.
"""

from __future__ import annotations

import torch

from surf_pytorch import LandSurface, SurfConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cold_start(N: int = 2, *, differentiable: bool = True):
    """A minimal (state, forcing, static) triple with a grad-capable input.

    Reuses the canonical slab shape from ``test_model._make_inputs`` but
    keeps the construction local so this module stays self-contained.
    ``skin_t`` is the differentiability entry point (matches
    ``test_model.test_grad_runs_end_to_end``): forward only builds a
    graph when at least one input leaf requires grad.
    """
    dtype = torch.float64
    cfg = SurfConfig(dtype=dtype, differentiable=differentiable)
    ntile, nsoil = cfg.n_tile, cfg.n_soil
    from surf_pytorch.state import Forcing, Static, SurfState

    state = SurfState(
        skin_t=torch.full((N, ntile), 288.0, dtype=dtype, requires_grad=True),
        soil_t=torch.full((N, nsoil), 285.0, dtype=dtype),
        soil_w_liq=torch.full((N, nsoil), 0.3, dtype=dtype),
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
    frac[:, 0] = 0.5
    frac[:, 3] = 0.2
    frac[:, 5] = 0.2
    frac[:, 7] = 0.1
    static = Static(
        tile_frac_base=frac,
        land_frac=torch.full((N,), 0.5, dtype=dtype),
        orography=torch.full((N,), 50.0, dtype=dtype),
        soil_type=torch.full((N,), 3, dtype=torch.long),
        veg_type=torch.full((N,), 3, dtype=torch.long),
        soil_thickness=torch.tensor([0.07, 0.21, 0.72, 1.89], dtype=dtype),
        emis=torch.full((N,), 0.99, dtype=dtype),
    )
    return state, forcing, static


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
class TestLearnableRegistration:
    """Buffer <-> Parameter promotion through ``SurfConfig.learnable_*``."""

    def test_default_config_has_no_parameters(self):
        """Frozen default: zero optimisable parameters, all LUTs buffers."""
        model = LandSurface(SurfConfig(dtype=torch.float64))
        assert sum(p.numel() for p in model.parameters()) == 0
        # wsat is a buffer, not a Parameter.
        assert "wsat" in dict(model.soil_table.named_buffers())
        assert "wsat" not in dict(model.soil_table.named_parameters())

    def test_learnable_soil_promotes_named_columns(self):
        """``learnable_soil=('wsat',)`` flips wsat to a Parameter.

        Regression for the ``model.py:150`` bug: the soil path previously
        passed ``set(...)`` to ``SoilTable`` (which expects a dict with
        ``.get``), crashing at construction.  The fix mirrors VegTable's
        ``dict.fromkeys(..., True)`` call.
        """
        model = LandSurface(
            SurfConfig(dtype=torch.float64, learnable_soil=("wsat",)),
        )
        soil_params = dict(model.soil_table.named_parameters())
        assert "wsat" in soil_params
        assert soil_params["wsat"].requires_grad
        # Non-listed columns stay frozen.
        assert "rcsoil" not in soil_params
        assert "rcsoil" in dict(model.soil_table.named_buffers())

    def test_learnable_veg_promotes_named_columns(self):
        """``learnable_veg=('rvlai',)`` flips rvlai to a Parameter."""
        model = LandSurface(
            SurfConfig(dtype=torch.float64, learnable_veg=("rvlai",)),
        )
        veg_params = dict(model.veg_table.named_parameters())
        assert "rvlai" in veg_params
        assert veg_params["rvlai"].requires_grad
        assert "rvrsmin" not in veg_params

    def test_unknown_column_name_is_silently_ignored(self):
        """A typo'd column name is ignored (no matching PER_CLASS field).

        ``dict.fromkeys(('wres',), True)`` creates the key, but
        ``SoilTable`` only iterates its ``PER_CLASS_FIELDS`` -- there is
        no ``wres`` field (it is ``wrest``), so nothing is promoted.
        This documents the silent-ignore behaviour so a future strict-
        validation change is intentional.
        """
        model = LandSurface(
            SurfConfig(dtype=torch.float64, learnable_soil=("wres",)),
        )
        assert sum(p.numel() for p in model.parameters()) == 0

    def test_learnable_columns_register_in_top_level_parameters(self):
        """Parameters propagate to ``model.parameters()`` (optimizer-visible)."""
        model = LandSurface(
            SurfConfig(
                dtype=torch.float64,
                learnable_soil=("wsat",),
                learnable_veg=("rvlai",),
            ),
        )
        names = {n for n, _ in model.named_parameters()}
        assert {"soil_table.wsat", "veg_table.rvlai"} <= names


# --------------------------------------------------------------------------- #
# Gradient flow through forward()
# --------------------------------------------------------------------------- #
class TestLearnableGradient:
    """End-to-end differentiability of the learnable LUT columns."""

    def test_veg_lut_is_trainable_through_skin_temperature(self):
        """rvlai (leaf-area index) flows into the SEB / stomatal path and
        receives a finite, non-zero gradient on skin temperature.

        This is the headline "differentiable nn.Module" guarantee: a
        documented learnable column actually trains end to end.
        """
        state, forcing, static = _cold_start(N=2)
        model = LandSurface(
            SurfConfig(
                dtype=torch.float64, differentiable=True,
                learnable_veg=("rvlai",),
            ),
        )
        rvlai = dict(model.veg_table.named_parameters())["rvlai"]
        assert rvlai.grad is None            # before backward

        new_state, _ = model(state, forcing, static)
        assert new_state.skin_t.requires_grad
        new_state.skin_t.sum().backward()

        assert rvlai.grad is not None
        assert torch.isfinite(rvlai.grad).all()
        assert rvlai.grad.abs().sum() > 0.0, (
            "rvlai gradient is zero -- veg LUT not actually trainable"
        )

    def test_veg_lut_gradient_changes_under_optimizer_step(self):
        """An SGD step on the rvlai parameter actually moves it.

        The optimiser sees the parameter via ``model.parameters()`` and
        the gradient above is non-zero, so ``step()`` must change the
        stored value (the true training loop works).
        """
        state, forcing, static = _cold_start(N=2)
        model = LandSurface(
            SurfConfig(
                dtype=torch.float64, differentiable=True,
                learnable_veg=("rvlai",),
            ),
        )
        rvlai = dict(model.veg_table.named_parameters())["rvlai"]
        before = rvlai.detach().clone()

        opt = torch.optim.SGD(model.parameters(), lr=1.0e6)   # huge lr
        opt.zero_grad()
        new_state, _ = model(state, forcing, static)
        new_state.skin_t.sum().backward()
        opt.step()

        assert (rvlai.detach() - before).abs().sum() > 0.0, (
            "rvlai unchanged after an SGD step -- parameter not wired to optimiser"
        )

    def test_soil_lut_grad_is_structurally_zero(self):
        """Soil-LUT columns do NOT train through forward outputs.

        HTESSEL consumes wsat / wrest inside the SRFWNG clamp bounds
        (soil moisture clipped to ``[wrest, wsat]``) and via indexed
        gathers, so the forward-output gradient w.r.t. those table
        entries is structurally zero.  This is the upstream physics,
        not a port regression: documenting it as expected behaviour
        means a future change that *does* make wsat trainable is a
        deliberate decision rather than a silent drift.

        Two complementary facts are pinned:

        * backprop from ``soil_w_liq`` -- the output where wsat enters
          the graph, via the soil-moisture solve -- yields a gradient
          that *is present* (the Parameter is wired in) but all-zero
          (the clamp bounds kill it);
        * backprop from ``skin_t`` yields ``None`` -- the skin-temperature
          path has no connection to the soil-moisture wsat usage at all.
        """
        state, forcing, static = _cold_start(N=2)
        model = LandSurface(
            SurfConfig(
                dtype=torch.float64, differentiable=True,
                learnable_soil=("wsat",),
            ),
        )
        wsat = dict(model.soil_table.named_parameters())["wsat"]
        new_state, _ = model(state, forcing, static)
        # soil_w_liq is the output where wsat enters the graph (the
        # soil-moisture solve); the SRFWNG clamp bounds zero the gradient.
        new_state.soil_w_liq.sum().backward()
        assert wsat.grad is not None, (
            "wsat should be in the soil_w_liq graph -- clamp bounds make it "
            "zero, not absent"
        )
        assert wsat.grad.abs().sum() == 0.0, (
            "expected structurally-zero wsat gradient (SRFWNG clamp); "
            "non-zero would mean wsat became trainable, a deliberate "
            "change worth its own test"
        )

        # And the disconnected path: skin_t never reaches wsat, so its
        # backward leaves wsat.grad at None (no accumulation at all).
        model2 = LandSurface(
            SurfConfig(
                dtype=torch.float64, differentiable=True,
                learnable_soil=("wsat",),
            ),
        )
        wsat2 = dict(model2.soil_table.named_parameters())["wsat"]
        # Re-run forward on a fresh state so the graph is independent.
        new_state2, _ = model2(state, forcing, static)
        new_state2.skin_t.sum().backward()
        assert wsat2.grad is None, (
            "skin_t loss has no path to wsat -- its grad should stay None, "
            "distinct from the soil_w_liq path which yields a zero tensor"
        )
