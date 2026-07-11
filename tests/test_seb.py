"""Unit tests for the surface energy balance (``surf_pytorch.physics.seb``)
and the skin-conductivity pre-compute (``surf_pytorch.physics.vlamsk``).

Tests are grouped:

* ``TestVlamskConstants`` -- pin the hard-wired scalars to the Fortran
  values (SUSVEG/SUSSOIL/SUSURB literal assignments).
* ``TestVlamsk`` -- per-tile ``lambda_skin`` output for a controlled
  input set, cross-checked against the Fortran ``VLAMSK`` SELECT CASE.
* ``TestSebOceanPin`` -- in the ``lambda_skin -> inf`` limit the skin
  temperature pins to the sub-surface temperature (ocean tile).
* ``TestSebPurePythonReference`` -- a standalone pure-Python port of
  ``SURFSEB_CTL`` for one tile/column, used as ground truth for the
  torch port.
* ``TestSebClosure`` -- verifies the IFS surface energy balance
  ``SSR + STR + SSH + SLH - G0 = 0`` (all four surface fluxes positive
  downward, ``G0`` positive downward into the ground) for
  self-consistent inputs.
* ``TestSebDifferentiable`` -- autograd runs end-to-end when the
  ``differentiable`` flag is on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from surf_pytorch import SurfConfig, ConstantBundle
from surf_pytorch.tables import VegTable
from surf_pytorch.physics import vlamsk as vlamsk_mod
from surf_pytorch.physics.vlamsk import vlamsk, LARGE, LARGE_SN, LARGE_WT, SNOW
from surf_pytorch.physics.seb import SebInputs, SebOutputs, surf_seb
from surf_pytorch.utils import qsat, dqsat_dt_ifs


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_seb_inputs(
    *,
    N: int = 2,
    T: int = 8,
    tsk1: float = 290.0,
    psurf: float = 1.0e5,
    tsrf: float = 285.0,
    rhochu: float = 0.01,
    rhocqu: float = 0.01,
    ssrfl: float = 200.0,
    slrfl: float = -300.0,
    tair: float = 290.0,
    qair: float = 0.008,
    lamsk: torch.Tensor | float = 10.0,
    alphal: float = 0.0,
    alphas: float = 1.0,
    ktvl: int = 1,
    ktvh: int = 3,
    lelwtl: bool = False,
    dtype: torch.dtype = torch.float64,
) -> tuple[SebInputs, VegTable, ConstantBundle]:
    """Build a self-consistent SebInputs for tests.

    qsat values are derived from (tsk1, psurf) so the linearisation
    in SURFSEB is exact at the t-1 skin temperature.
    """
    consts = ConstantBundle(dtype=dtype)
    veg = VegTable(dtype=dtype)
    cp = float(consts.cpd.item())

    tsk_t = torch.full((N, T), tsk1, dtype=dtype)
    p_t = torch.full((N, T), psurf, dtype=dtype)
    qsk_t = qsat(tsk_t, p_t, consts)
    dqsdT_t = dqsat_dt_ifs(tsk_t, p_t, consts)
    ssk_t = cp * tsk_t

    if isinstance(lamsk, (int, float)):
        lamsk_t = torch.full((N, T), float(lamsk), dtype=dtype)
    else:
        lamsk_t = lamsk

    inp = SebInputs(
        sskm1m=ssk_t, tskm1m=tsk_t, qskm1m=qsk_t, dqsdT=dqsdT_t,
        rhochu=torch.full((N, T), rhochu, dtype=dtype),
        rhocqu=torch.full((N, T), rhocqu, dtype=dtype),
        alphal=torch.full((N, T), alphal, dtype=dtype),
        alphas=torch.full((N, T), alphas, dtype=dtype),
        ssrfl=torch.full((N, T), ssrfl, dtype=dtype),
        slrfl_grid=torch.full((N,), slrfl, dtype=dtype),
        tskrad_grid=torch.full((N,), tsk1, dtype=dtype),
        emis_grid=torch.full((N,), 0.99, dtype=dtype),
        tsrf=torch.full((N, T), tsrf, dtype=dtype),
        lamsk=lamsk_t,
        frti=torch.full((N, T), 1.0 / T, dtype=dtype),
        ktvl_grid=torch.full((N,), ktvl, dtype=torch.long),
        ktvh_grid=torch.full((N,), ktvh, dtype=torch.long),
        asl_grid=torch.zeros((N,), dtype=dtype),
        bsl_grid=torch.full((N,), cp * tair, dtype=dtype),
        aql_grid=torch.zeros((N,), dtype=dtype),
        bql_grid=torch.full((N,), qair, dtype=dtype),
        ptmst=torch.tensor(3600.0, dtype=dtype),
        lelwdd=True, lelwtl=lelwtl,
    )
    return inp, veg, consts


# --------------------------------------------------------------------------- #
# Pure-Python single-column reference of SURFSEB_CTL (ground truth)
# --------------------------------------------------------------------------- #
def _seb_ref_single_tile(
    *,
    ptskm1m: float, psskm1m: float, pqskm1m: float, pdqsdT: float,
    prhochu: float, prhocqu: float, palphal: float, palphas: float,
    pssrfl: float, ptsrf: float, plamsk: float,
    pslrfl: float, ptskrad: float, pemis: float,
    pasl: float, pbsl: float, paql: float, pbql: float,
    zfrsr: float, lam: float, rcpd: float, rtmp2: float, rsigma: float,
) -> dict:
    """Pure-Python port of SURFSEB_CTL for one tile, one column.

    Used as the ground truth for the torch port -- both must agree to
    the last ULP for the port to be considered bit-identical.
    """
    zicptm1 = ptskm1m / psskm1m
    zcjs1 = prhochu
    zcjs3 = -prhochu
    zcjq2 = prhocqu * palphal
    zcjq3 = -prhocqu * palphas * pdqsdT * zicptm1
    zcjq4 = -prhocqu * palphas * (pqskm1m - pdqsdT * ptskm1m)

    zdlwdt = -4.0 * pemis * rsigma * ptskrad ** 3
    zcoef1 = lam - rcpd * ptskm1m * rtmp2
    zz = (zdlwdt - plamsk) * zicptm1 + zcjs3 + zcoef1 * zcjq3
    zizz = 1.0 / zz

    zdss1 = -zcjs1 * zizz
    zdss2 = -zcoef1 * zcjq2 * zizz
    zdss4 = (
        -pssrfl * zfrsr - pslrfl + zdlwdt * ptskrad
        - zcoef1 * zcjq4 - plamsk * ptsrf
    ) * zizz

    zdjs1 = zcjs1 + zcjs3 * zdss1
    zdjs2 = zcjs3 * zdss2
    zdjs4 = zcjs3 * zdss4
    zdjq1 = zcjq3 * zdss1
    zdjq2 = zcjq2 + zcjq3 * zdss2
    zdjq4 = zcjq3 * zdss4 + zcjq4

    # Single tile -> equal-weight average.
    zejs1, zejs2, zejs4 = zdjs1, zdjs2, zdjs4
    zejq1, zejq2, zejq4 = zdjq1, zdjq2, zdjq4

    zz1 = 1.0 - pasl * zejs1
    zz2 = paql * zejs2
    zz3 = pbsl * zejs1 + pbql * zejs2 + zejs4
    zy1 = 1.0 - paql * zejq2
    zy2 = pasl * zejq1
    zy3 = pbsl * zejq1 + pbql * zejq2 + zejq4

    zzdet = zz1 * zy1 - zz2 * zy2
    zjq = (zz3 * zy2 + zz1 * zy3) / zzdet
    zjs = (zz2 * zy3 + zz3 * zy1) / zzdet

    psl = pasl * zjs + pbsl
    pql = paql * zjq + pbql

    pssk = zdss1 * psl + zdss2 * pql + zdss4
    pjs = zdjs1 * psl + zdjs2 * pql + zdjs4
    pjq = zdjq1 * psl + zdjq2 * pql + zdjq4
    ptsk = pssk * zicptm1

    pssh = pjs - rcpd * ptskm1m * rtmp2 * pjq
    pslh = lam * pjq
    pstr = pslrfl + zdlwdt * (ptsk - ptskrad)
    pg0 = plamsk * (ptsk - ptsrf)

    return dict(
        ptsk=ptsk, pjs=pjs, pjq=pjq, pssk=pssk, psl=psl, pql=pql,
        pssh=pssh, pslh=pslh, pstr=pstr, pg0=pg0,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestVlamskConstants:
    """Pin the hard-wired scalars to the Fortran literal assignments."""

    def test_large_ocean_pin(self):
        # VLAMSK_MOD line 114, surfseb_ctl_mod line 210.
        assert vlamsk_mod.LARGE == 1.0e10

    def test_large_snow_melt_cap(self):
        # VLAMSK_MOD line 115, surfseb_ctl_mod line 211.
        assert vlamsk_mod.LARGE_SN == 50.0

    def test_large_wet_skin(self):
        # VLAMSK_MOD line 117, surfseb_ctl_mod line 213.
        assert vlamsk_mod.LARGE_WT == 20.0

    def test_snow_conductivity(self):
        # VLAMSK_MOD line 119, surfseb_ctl_mod line 215.
        assert vlamsk_mod.SNOW == 7.0
        assert vlamsk_mod.SNOW_GLACIER == 8.0
        assert vlamsk_mod.SNOW_HIGH_VEG == 20.0

    def test_rho_constants(self):
        # SUSSOIL line 486-488.
        assert vlamsk_mod.RHO_ICE == 920.0
        assert vlamsk_mod.RHO_CI == 2.05e6

    def test_rqncr_default(self):
        # SUSSOIL line 136: RQSNCR = 1/15 (default; 1/10 with LESN09).
        assert vlamsk_mod.RQSNCR_DEFAULT == pytest.approx(1.0 / 15.0)
        assert vlamsk_mod.RQSNCR_SN09 == pytest.approx(1.0 / 10.0)


class TestVlamsk:
    """Per-tile lambda_skin output for controlled inputs."""

    def _run(self, *, tsk=290.0, tsrf=285.0, ktvl=1, ktvh=3, psnm=100.0,
             prsn=300.0, ptstep=None, dtype=torch.float64):
        cfg = SurfConfig(dtype=dtype)
        consts = ConstantBundle(dtype=dtype)
        veg = VegTable(dtype=dtype)
        N = 2
        T = cfg.n_tile
        plamsk = vlamsk(
            ptskm1m=torch.full((N, T), tsk, dtype=dtype),
            ptsrf=torch.full((N, T), tsrf, dtype=dtype),
            psnm=torch.full((N,), psnm, dtype=dtype),
            prsn=torch.full((N,), prsn, dtype=dtype),
            psntice=None,
            ktvl=torch.full((N,), ktvl, dtype=torch.long),
            ktvh=torch.full((N,), ktvh, dtype=torch.long),
            veg_table=veg, consts=consts, config=cfg,
            ptstep=ptstep,
        )
        return plamsk[0]  # column 0 tile values

    def test_ocean_tile_large(self):
        # Tile 0 (water) -> LARGE.
        plamsk = self._run()
        assert plamsk[0].item() == LARGE

    def test_sea_ice_tile_lookup(self):
        # Tile 1 (sea-ice) -> RVLAMSK(12) = 58 (unstable here: tsk=290>tsrf=285
        # is actually stable so should be RVLAMSKS(12)=58 too).
        plamsk = self._run()
        assert plamsk[1].item() == 58.0  # RVLAMSK(12) = RVLAMSKS(12) = 58

    def test_wet_skin_harmonic(self):
        # Tile 2 (wet skin) -> LARGE_WT = 20.
        plamsk = self._run()
        assert plamsk[2].item() == LARGE_WT

    def test_low_veg_dry(self):
        # Tile 3 (low veg) with ktvl=1 -> RVLAMSKS(1) = 10 (stable).
        plamsk = self._run(ktvl=1)
        assert plamsk[3].item() == 10.0

    def test_low_veg_snow_melt_window(self):
        # Tile 4 (snow on low veg): Tsk>=Tsrf and Tsk>RTT-0.2 -> LARGE_SN=50.
        plamsk = self._run(tsk=290.0, tsrf=285.0)
        assert plamsk[4].item() == LARGE_SN

    def test_low_veg_snow_cold(self):
        # Tile 4 (snow on low veg): Tsk<Tsrf -> SNOW=7.
        plamsk = self._run(tsk=270.0, tsrf=275.0)
        assert plamsk[4].item() == SNOW

    def test_high_veg_dry(self):
        # Tile 5 (high veg) with ktvh=3 -> RVLAMSKS(3) = 10.
        plamsk = self._run(ktvh=3)
        assert plamsk[5].item() == 10.0

    def test_bare_soil_desert(self):
        # Tile 7 (bare soil) -> RVLAMSKS(8) = 15 (unstable stable branch).
        plamsk = self._run()
        assert plamsk[7].item() == 15.0

    def test_high_veg_snow_stability(self):
        # Tile 6 (snow under high veg): RVLAMSKS(3)=10 reduced by
        # stability factor 1/(1+stab*10) where stab>0 (tsk>tsrf).
        # The stability correction requires ptstep; with ptstep=None
        # stab=0 and plamsk = base = 10 (verified by the companion test).
        plamsk = self._run(
            tsk=290.0, tsrf=285.0, psnm=100.0, prsn=300.0,
            ptstep=torch.tensor(3600.0, dtype=torch.float64),
        )
        # base = 10, stab = 3600/(max(1/15, 100/300)*2.05e6*300/920)
        # depth = max(1/15, 0.333) = 0.333
        # rho_cp = 2.05e6 * 300 / 920 = 6.685e5
        # stab = 3600 / (0.333 * 6.685e5) = 0.01618
        # plamsk = 10/(1+0.01618*10) = 10/1.1618 = 8.608
        assert plamsk[6].item() == pytest.approx(8.608, rel=1e-3)

    def test_high_veg_snow_no_ptstep(self):
        # Without ptstep the stability factor is zero -> plamsk = base.
        plamsk = self._run(tsk=290.0, tsrf=285.0, psnm=100.0, prsn=300.0,
                           ptstep=None)
        # Stable (tsk>tsrf) -> RVLAMSKS(3) = 10; stab=0 -> unchanged.
        assert plamsk[6].item() == 10.0


class TestSebOceanPin:
    """Skin T pins to Tsrf when lambda_skin -> inf (ocean tile)."""

    def test_ocean_skin_equals_sst(self, atol_loose):
        # Single-tile (water), LARGE conductivity -> Tsk = Tsrf exactly.
        inp, _, consts = _build_seb_inputs(
            N=2, T=1, tsrf=288.0, lamsk=1.0e10,
        )
        cfg = SurfConfig(dtype=torch.float64)
        import dataclasses
        cfg1 = dataclasses.replace(cfg, n_tile=1)
        out = surf_seb(inp, consts, cfg1)
        assert torch.allclose(out.ptsk[:, 0], inp.tsrf[:, 0], atol=atol_loose)

    def test_finite_lambda_deviation(self):
        # Finite lambda_skin -> Tsk deviates from Tsrf.
        inp, _, consts = _build_seb_inputs(N=2, T=1, tsrf=288.0, lamsk=10.0)
        import dataclasses
        cfg1 = dataclasses.replace(SurfConfig(dtype=torch.float64), n_tile=1)
        out = surf_seb(inp, consts, cfg1)
        # Tsk should be different from Tsrf (the SEB pushes it away).
        assert not torch.allclose(out.ptsk[:, 0], inp.tsrf[:, 0], atol=1e-3)


class TestSebPurePythonReference:
    """Torch port matches a pure-Python reference of SURFSEB_CTL.

    The pure-Python reference is the ground truth for the translation;
    the torch port must reproduce it to the last ULP.
    """

    def test_single_tile_matches_reference(self, atol_strict):
        # Build torch inputs for a single tile.
        inp, _, consts = _build_seb_inputs(
            N=1, T=1, tsk1=290.0, tsrf=285.0, lamsk=10.0,
            rhochu=0.01, rhocqu=0.01, ssrfl=200.0, slrfl=-300.0,
            tair=290.0, qair=0.008,
        )
        import dataclasses
        cfg1 = dataclasses.replace(SurfConfig(dtype=torch.float64), n_tile=1)
        out = surf_seb(inp, consts, cfg1)

        # Run pure-Python reference with the same scalar inputs.
        # The torch port uses _tile_solar_transmission which gives 0.95 for
        # tile index 3 -- but with T=1 we only have tile 0 (water, frsr=1).
        # Use frsr=1 to match the water-tile behaviour.
        rcpd = float(consts.cpd.item())
        rsigma = float(consts.rsigma.item())
        rtmp2 = float(consts.rvtmp2.item())
        rlvtt = float(consts.rlvtt.item())
        ref = _seb_ref_single_tile(
            ptskm1m=290.0, psskm1m=rcpd * 290.0,
            pqskm1m=float(inp.qskm1m[0, 0]),
            pdqsdT=float(inp.dqsdT[0, 0]),
            prhochu=0.01, prhocqu=0.01, palphal=0.0, palphas=1.0,
            pssrfl=200.0, ptsrf=285.0, plamsk=10.0,
            pslrfl=-300.0, ptskrad=290.0, pemis=0.99,
            pasl=0.0, pbsl=rcpd * 290.0, paql=0.0, pbql=0.008,
            zfrsr=1.0, lam=rlvtt,
            rcpd=rcpd, rtmp2=rtmp2, rsigma=rsigma,
        )

        # Compare every output.  Note: psl / pql are grid-mean (Ngrid,)
        # tensors, while the per-tile fields are (Ngrid, Ntile).
        for key_py, key_torch in [
            ("ptsk", "ptsk"), ("pjs", "pjs"), ("pjq", "pjq"),
            ("pssh", "pssh"), ("pslh", "pslh"), ("pstr", "pstr"),
            ("pg0", "pg0"),
        ]:
            actual = float(getattr(out, key_torch)[0, 0])
            expected = ref[key_py]
            assert actual == pytest.approx(expected, abs=atol_strict), (
                f"{key_py}: torch={actual}, ref={expected}"
            )
        # Grid-mean fields.
        for key_py, key_torch in [("psl", "psl"), ("pql", "pql")]:
            actual = float(getattr(out, key_torch)[0])
            expected = ref[key_py]
            assert actual == pytest.approx(expected, abs=atol_strict), (
                f"{key_py}: torch={actual}, ref={expected}"
            )


class TestSebClosure:
    """IFS surface energy balance ``SSR + STR + SSH + SLH - G0 = 0``.

    All four surface fluxes are **positive downward** (into the surface)
    and ``G0`` is **positive downward** into the ground; this is the
    SURFSEB_CTL output convention documented in :mod:`surf_pytorch.physics.seb`.
    The linearised SEB is exactly satisfied (to floating-point) once Tsk
    is solved, so the closure residual is at the rounding-error level.
    """
    @staticmethod
    def _resid(out: SebOutputs, ssr: torch.Tensor, frsr: torch.Tensor) -> torch.Tensor:
        """SEB residual ``SSR_eff + STR + SSH + SLH - G0``.

        ``SSR_eff = SSR * FRSR`` is the SW that the SEB actually sees --
        tiles 4 (low veg), 6/7 (high veg) transmit a fraction
        ``1 - RVTRSR`` of the SW through the canopy, so only ``SSR*FRSR``
        participates in the surface energy balance (see
        :func:`surf_pytorch.physics.seb._tile_solar_transmission`).
        """
        return ssr * frsr + out.pstr + out.pssh + out.pslh - out.pg0

    def test_closure_single_tile(self):
        # Single tile (water, frsr=1), self-consistent qsat -> closure
        # residual at rounding-error level (SEB is exactly solved for Tsk).
        inp, _, consts = _build_seb_inputs(
            N=1, T=1, tsk1=290.0, tsrf=288.0, lamsk=15.0,
            rhochu=0.005, rhocqu=0.005,
        )
        import dataclasses
        from surf_pytorch.physics.seb import _tile_solar_transmission
        cfg1 = dataclasses.replace(SurfConfig(dtype=torch.float64), n_tile=1)
        out = surf_seb(inp, consts, cfg1)
        frsr = _tile_solar_transmission(1, 1, inp.ssrfl.dtype, inp.ssrfl.device)
        resid = float(self._resid(out, inp.ssrfl, frsr)[0, 0])
        assert abs(resid) < 1e-6, (
            f"SEB residual {resid:.3e} W/m^2 too large; expected ~0"
        )

    def test_closure_improves_with_small_dt(self):
        # As Tsk stays close to Tsk_t1 (weak forcing), the qsat
        # linearisation remains well-conditioned and the residual
        # stays at rounding level.
        inp, _, consts = _build_seb_inputs(
            N=1, T=1, tsk1=290.0, tsrf=290.0, lamsk=10.0,
            rhochu=0.001, rhocqu=0.001, ssrfl=10.0, slrfl=-15.0,
            tair=290.0, qair=0.005,
        )
        import dataclasses
        from surf_pytorch.physics.seb import _tile_solar_transmission
        cfg1 = dataclasses.replace(SurfConfig(dtype=torch.float64), n_tile=1)
        out = surf_seb(inp, consts, cfg1)
        frsr = _tile_solar_transmission(1, 1, inp.ssrfl.dtype, inp.ssrfl.device)
        resid = float(self._resid(out, inp.ssrfl, frsr)[0, 0])
        assert abs(resid) < 1e-6, (
            f"SEB residual {resid:.3e} W/m^2 too large for near-equilibrium"
        )

    def test_closure_multi_tile(self):
        # Multi-tile case: closure holds per tile (with the per-tile SW
        # transmission factor FRSR applied to SSR), not just in the mean.
        inp, _, consts = _build_seb_inputs(N=3, T=4, tsk1=290.0, tsrf=288.0)
        import dataclasses
        from surf_pytorch.physics.seb import _tile_solar_transmission
        cfg4 = dataclasses.replace(SurfConfig(dtype=torch.float64), n_tile=4)
        out = surf_seb(inp, consts, cfg4)
        frsr = _tile_solar_transmission(3, 4, inp.ssrfl.dtype, inp.ssrfl.device)
        resid = self._resid(out, inp.ssrfl, frsr)
        assert resid.abs().max().item() < 1e-6, (
            f"per-tile SEB residual max {resid.abs().max().item():.3e}"
        )

    def test_closure_no_turbulence(self):
        # rhoc=0 -> SSH=SLH=0, leaving radiation vs ground only.
        # Cleanest check of the sign convention.
        inp, _, consts = _build_seb_inputs(
            N=1, T=1, tsk1=290.0, tsrf=285.0, lamsk=15.0,
            rhochu=0.0, rhocqu=0.0,
        )
        import dataclasses
        from surf_pytorch.physics.seb import _tile_solar_transmission
        cfg1 = dataclasses.replace(SurfConfig(dtype=torch.float64), n_tile=1)
        out = surf_seb(inp, consts, cfg1)
        assert float(out.pssh[0, 0]) == pytest.approx(0.0, abs=1e-12)
        assert float(out.pslh[0, 0]) == pytest.approx(0.0, abs=1e-12)
        frsr = _tile_solar_transmission(1, 1, inp.ssrfl.dtype, inp.ssrfl.device)
        resid = float(self._resid(out, inp.ssrfl, frsr)[0, 0])
        assert abs(resid) < 1e-9, (
            f"SEB residual {resid:.3e} W/m^2 (rhoc=0 case)"
        )


class TestSebDifferentiable:
    """The SEB runs end-to-end under autograd when differentiable=True."""

    def test_grad_flows_to_tskm1m(self):
        # Use 2 tiles so the 2x2 mean solve is exercised.
        cfg = SurfConfig(dtype=torch.float64, differentiable=True)
        consts = ConstantBundle(dtype=torch.float64)
        cp = float(consts.cpd.item())
        N, T = 2, cfg.n_tile

        # tskm1m requires grad -- it is the t-1 skin T used by the
        # linearisation.
        tskm1m = torch.full((N, T), 290.0, dtype=torch.float64,
                            requires_grad=True)
        psurf = torch.full((N, T), 1.0e5, dtype=torch.float64)
        qskm1m = qsat(tskm1m, psurf, consts)
        dqsdT = dqsat_dt_ifs(tskm1m, psurf, consts)
        sskm1m = cp * tskm1m

        inp = SebInputs(
            sskm1m=sskm1m, tskm1m=tskm1m, qskm1m=qskm1m, dqsdT=dqsdT,
            rhochu=torch.full((N, T), 0.01, dtype=torch.float64),
            rhocqu=torch.full((N, T), 0.01, dtype=torch.float64),
            alphal=torch.zeros((N, T), dtype=torch.float64),
            alphas=torch.ones((N, T), dtype=torch.float64),
            ssrfl=torch.full((N, T), 200.0, dtype=torch.float64),
            slrfl_grid=torch.full((N,), -300.0, dtype=torch.float64),
            tskrad_grid=torch.full((N,), 290.0, dtype=torch.float64),
            emis_grid=torch.full((N,), 0.99, dtype=torch.float64),
            tsrf=torch.full((N, T), 285.0, dtype=torch.float64),
            lamsk=torch.full((N, T), 10.0, dtype=torch.float64),
            frti=torch.full((N, T), 1.0 / T, dtype=torch.float64),
            ktvl_grid=torch.full((N,), 1, dtype=torch.long),
            ktvh_grid=torch.full((N,), 3, dtype=torch.long),
            asl_grid=torch.zeros((N,), dtype=torch.float64),
            bsl_grid=torch.full((N,), cp * 290.0, dtype=torch.float64),
            aql_grid=torch.zeros((N,), dtype=torch.float64),
            bql_grid=torch.full((N,), 0.008, dtype=torch.float64),
            ptmst=torch.tensor(3600.0, dtype=torch.float64),
            lelwdd=True, lelwtl=False,
        )
        out = surf_seb(inp, consts, cfg)
        loss = out.ptsk.sum()
        loss.backward()
        assert tskm1m.grad is not None
        assert torch.isfinite(tskm1m.grad).all()
        assert tskm1m.grad.abs().sum() > 0  # non-trivial gradient

    def test_no_grad_default(self):
        # differentiable=False (default) -> no graph built.
        cfg = SurfConfig(dtype=torch.float64, differentiable=False)
        inp, _, consts = _build_seb_inputs(N=2, T=2)
        import dataclasses
        cfg2 = dataclasses.replace(cfg, n_tile=2)
        out = surf_seb(inp, consts, cfg2)
        assert not out.ptsk.requires_grad


class TestSebSolarTransmission:
    """Pin the per-tile short-wave transmission factor ``FRSR``.

    Regression guard for the wet-skin tile: ``surfseb_ctl_mod.F90``
    CASE(3) sets ``ZFRSR = 1 - RVTRSR(1)`` for the wet-skin tile exactly
    like CASE(4) does for low-veg dry.  A previous port dropped the
    wet-skin assignment, leaving FRSR=1 there and biasing the SW seen by
    that tile by RVTRSR(1) = 5%.  These tests pin every non-unit FRSR so
    the regression cannot silently return.

    Reference: ``surfseb_ctl_mod.F90`` lines 320, 329, 346, 355.
    """

    def test_full_8tile_table(self):
        # 0-based tile indices with the SURFSEB_CTL convention:
        #   2 wet skin       -> 1 - RVTRSR(1) = 0.95
        #   3 low-veg dry    -> 1 - RVTRSR(1) = 0.95
        #   5 high-veg dry   -> 1 - RVTRSR(3) = 0.97
        #   6 high-veg snow  -> 1 - RVTRSR(3) = 0.97
        #   all others       -> 1.0
        from surf_pytorch.physics.seb import _tile_solar_transmission
        frsr = _tile_solar_transmission(1, 8, torch.float64, "cpu")[0]
        expected = [1.0, 1.0, 0.95, 0.95, 1.0, 0.97, 0.97, 1.0]
        assert frsr.tolist() == pytest.approx(expected, abs=1e-15)

    def test_wet_skin_transmits_sw(self):
        # The bug regressed this exact assertion -- guard it on its own
        # so the failure message names the wet-skin tile directly.
        from surf_pytorch.physics.seb import _tile_solar_transmission
        frsr = _tile_solar_transmission(1, 8, torch.float64, "cpu")[0]
        assert frsr[2].item() == pytest.approx(0.95, abs=1e-15), (
            "wet-skin tile (0-based 2) must use 1 - RVTRSR(1) = 0.95, "
            "matching surfseb_ctl_mod.F90 CASE(3); got "
            f"{frsr[2].item()}"
        )

    def test_truncated_tile_axis(self):
        # With fewer tiles than 8 the function must not raise and must
        # still set every available non-unit entry (e.g. with 4 tiles,
        # tiles 2 and 3 are both 0.95).
        from surf_pytorch.physics.seb import _tile_solar_transmission
        frsr = _tile_solar_transmission(1, 4, torch.float64, "cpu")[0]
        assert frsr.tolist() == pytest.approx(
            [1.0, 1.0, 0.95, 0.95], abs=1e-15
        )
