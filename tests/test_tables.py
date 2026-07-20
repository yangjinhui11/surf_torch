"""Lookup-table tests.

Each test pins one row of the LUT to the exact literal from the
corresponding Fortran initialiser routine.  A regression here means the
port has drifted from the reference driver -- catch it before the
physics core consumes the bad value.

References
----------
* soil: ``surf/module/sussoil_mod.F90`` lines 289-296 (base),
  340-433 (derived).
* veg : ``surf/module/susveg_mod.F90`` lines 261-360 (z0m/z0h/rsmin).
* root: ``surf/module/srfrootfr_mod.F90`` lines 123-130 (ZAA/ZBB).
* ritbl: ``surf/module/suvexcs_mod.F90`` lines 78-133 (Newton table).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from surf_pytorch import tables as T


pytestmark = pytest.mark.fortran_ref
RIMAX_TEST = 10.0   # mirrors tables.RIMAX


# --------------------------------------------------------------------------- #
# SoilTable
# --------------------------------------------------------------------------- #
class TestSoilTable:
    @pytest.fixture(scope="class")
    def soil(self, dtype, device):
        return T.SoilTable(dtype=dtype, device=device)

    def test_dim(self, soil):
        assert soil.mvgalpha.shape[0] == 8   # index 0 sentinel + 7 types
        assert soil.wsat.shape[0] == 8

    def test_index0_is_water(self, soil):
        """Index 0 (water) has all-zero hydrology per sussoil."""
        assert soil.mvgalpha[0].item() == 0.0
        assert soil.wsat[0].item() == 0.0

    def test_base_table_values(self, soil):
        """ZMVGALPHA, ZWCONSM, ZNFAC, ZLAMBDA, ZWSATM from sussoil 289-296."""
        # ZMVGALPHA(1:7)
        expected = [3.83, 3.14, 0.83, 3.67, 2.65, 1.30, 3.14]
        got = soil.mvgalpha[1:8].tolist()
        assert got == pytest.approx(expected, abs=1e-12)
        # ZWCONSM(1:7)  [m/s]
        expected = [6.94e-6, 1.16e-6, 2.63e-7, 2.87e-6, 1.74e-6, 0.93e-6, 1.16e-6]
        got = soil.wconsm[1:8].tolist()
        assert got == pytest.approx(expected, rel=1e-6)
        # ZWSATM(1:7)
        expected = [0.403, 0.439, 0.430, 0.520, 0.614, 0.766, 0.439]
        got = soil.wsat[1:8].tolist()
        assert got == pytest.approx(expected, abs=1e-12)
        # ZWRES(1:7)
        expected = [0.025, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010]
        got = soil.wrest[1:8].tolist()
        assert got == pytest.approx(expected, abs=1e-12)

    def test_gather(self, soil):
        """gather() indexes by soil-type (detached)."""
        soiltype = torch.tensor([0, 3, 7], dtype=torch.long)  # water, loam, type 7
        wsat = soil.gather("wsat", soiltype)
        assert wsat.tolist() == pytest.approx([0.0, 0.430, 0.439], abs=1e-12)

    def test_freezing_constants(self, soil, dtype):
        """RTF1..4 from sussoil freezing section."""
        # These are constants on the SoilConstants dataclass, not per-class.
        # Spot-check RLAMBDAICE / RLAMBDAWAT are present.
        assert T.RLAMBDAICE > 0
        assert T.RLAMBDAWAT > 0

    def test_saturated_van_genuchten_diffusivity_uses_signed_water_range(self, soil):
        """RDMAXM must use Fortran SIGN(magnitude, WSAT-WREST)."""
        js = 3
        wsat = soil.wsat[js].item()
        wrest = soil.wrest[js].item()
        mfac = soil.mfac[js].item()
        lam = soil.gather("lambda", torch.tensor(js)).item()
        alpha = soil.mvgalpha[js].item()
        wconsm = soil.wconsm[js].item()
        wfac = np.copysign(max(abs(wsat - wrest), np.finfo(float).eps), wsat - wrest)
        zse = (0.999 * wsat - wrest) / wfac
        term = 1.0 - zse ** (1.0 / mfac)
        expected = ((1.0 - mfac) * wconsm / (alpha * mfac * wfac)) * (
            zse ** (lam - 1.0 / mfac)
            * (term ** mfac + term ** (-mfac) - 2.0)
        )
        assert soil.dmax[js].item() == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# VegTable
# --------------------------------------------------------------------------- #
class TestVegTable:
    @pytest.fixture(scope="class")
    def veg(self, dtype, device):
        return T.VegTable(dtype=dtype, device=device)

    def test_dim(self, veg):
        # Index 0 sentinel + 20 types = 21 columns.
        assert veg.rvz0m.shape[0] == 21

    def test_index0_mirrors_desert(self, veg):
        """VegTable index 0 mirrors type 8 (desert) per susveg."""
        assert veg.rvz0m[0].item() == veg.rvz0m[8].item()
        assert veg.rvlai[0].item() == veg.rvlai[8].item()

    def test_z0m_table(self, veg):
        """RVZ0M(1:11) from susveg 315-325."""
        expected = [0.25, 0.1, 2.00, 2.00, 2.00, 2.00, 0.47, 0.013, 0.034, 0.5, 0.17]
        got = veg.rvz0m[1:12].tolist()
        assert got == pytest.approx(expected, abs=1e-12)

    def test_z0h_smaller_than_z0m(self, veg):
        """z0h <= z0m for all veg types (roughness sublayer physics)."""
        assert torch.all(veg.rvz0h[1:] <= veg.rvz0m[1:] + 1e-9)

    def test_gather(self, veg):
        vegtype = torch.tensor([3, 8], dtype=torch.long)  # eng needleleaf, desert
        z0m = veg.gather("rvz0m", vegtype)
        assert z0m.tolist() == pytest.approx([2.00, 0.013], abs=1e-12)


# --------------------------------------------------------------------------- #
# RootTable
# --------------------------------------------------------------------------- #
class TestRootTable:
    @pytest.fixture(scope="class")
    def roots(self, dtype, device):
        return T.RootTable(dtype=dtype, device=device)

    def test_dim(self, roots):
        # (Nsoil=4, NVTYPES+1=21)
        assert roots.rvrootsa.shape == (4, 21)

    def test_root_fractions_sum_to_one(self, roots):
        """Root fractions sum to 1 along the soil axis (Zeng constraint)."""
        s = roots.rvrootsa.sum(dim=0)
        # Classes with ZAA=0 (12,14,15,20) put everything in the top layer;
        # the sum is still 1 for every class.
        assert torch.allclose(s, torch.ones_like(s), atol=1e-9)

    def test_funny_classes_top_heavy(self, roots):
        """Types 12,14,15,20 (ZAA=0) get all roots in the top layer."""
        for vtype in (12, 14, 15, 20):
            top = roots.rvrootsa[0, vtype].item()
            assert abs(top - 1.0) < 1e-9, f"vtype {vtype} top={top}"

    def test_gather(self, roots):
        # Scalar (0-d) veg-type -> (nsoil,) per the (*batch, nsoil) contract.
        vegtype = torch.tensor(3, dtype=torch.long)
        fr = roots.gather(vegtype)
        assert fr.shape == (4,)
        assert abs(fr.sum().item() - 1.0) < 1e-9
        # Batched veg-type -> (N, nsoil).
        vegtype2 = torch.tensor([3, 8, 12], dtype=torch.long)
        fr2 = roots.gather(vegtype2)
        assert fr2.shape == (3, 4)
        assert torch.allclose(fr2.sum(-1), torch.ones(3, dtype=fr2.dtype), atol=1e-9)


# --------------------------------------------------------------------------- #
# RitblTable
# --------------------------------------------------------------------------- #
class TestRitblTable:
    @pytest.fixture(scope="class")
    def ritbl(self, dtype, device):
        return T.RitblTable(dtype=dtype, device=device)

    def test_dim(self, ritbl):
        assert ritbl.ritbl.shape[0] == T.JPRITBL == 101
        assert T.RIMAX == 10.0
        # DRITBL = RIMAX/(JPRITBL-1) = 0.1
        assert ritbl.dritbl.item() == pytest.approx(0.1, abs=1e-12)

    def test_neutral_ri_zero(self, ritbl):
        """At Ri=0 the table's eta must be 0 (neutral -> eta=0)."""
        zri = torch.tensor([0.0], dtype=ritbl.ritbl.dtype)
        eta = ritbl.initial_guess(zri)
        assert abs(eta.item() - 0.0) < 1e-9

    def test_eta_increases_with_ri(self, ritbl):
        """Stable branch: eta(Ri) is monotone increasing."""
        zri = torch.linspace(0.0, RIMAX_TEST, 20, dtype=ritbl.ritbl.dtype)
        eta = ritbl.initial_guess(zri)
        diffs = eta[1:] - eta[:-1]
        assert torch.all(diffs >= -1e-9)

    def test_strongly_stable_saturates(self, ritbl):
        """Ri > RIMAX clamps to the last entry (numerical floor)."""
        zri = torch.tensor([15.0], dtype=ritbl.ritbl.dtype)
        eta = ritbl.initial_guess(zri)
        assert eta.item() == pytest.approx(ritbl.ritbl[-1].item(), abs=1e-12)

    def test_monotone_increasing_stable(self, ritbl):
        """eta(Ri) is monotone increasing for the stable branch."""
        etas = ritbl.ritbl[1:]   # entries at Ri = 0, 0.1, 0.2, ...
        diffs = etas[1:] - etas[:-1]
        assert torch.all(diffs >= -1e-9)
