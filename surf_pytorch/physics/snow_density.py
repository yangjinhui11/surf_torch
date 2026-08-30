"""雪密度演化（``SRFSN_LWIMP`` 第 3.1 节）。

本模块是位于 ``SRFSN_LWIMP``（``srfsn_lwimp_mod.F90`` 第 357-377 行
与第 494-527 行）中的雪密度块的 Python 移植——带预报密度的单层雪
方案。移植的 :mod:`surf_pytorch.physics.snow` 所镜像的简化单层路径
不做密度演化，密度演化在姊妹例程 ``SRFSN_LWIMP``（无 ``s``）和多层
``SRFSN_RSN``（由 ``SRFSN_DRIVER`` 调用）中完成。本模块提供
``SRFSN_LWIMP`` 使用的单层密度演化，使移植的雪预报量随 ``PSNM1M``
和 ``PTSNM1M`` 一起推进 ``PRSNM1M``。

方案概述
--------------------
新的雪密度 ``PRSN``（kg/m^3）由五项贡献构建，与 ``SRFSN_LWIMP``
第 494-527 行完全一致：

1. **新雪密度** ``ZRHOMINSN``——地表温度与地表风速四分之一幂的
   线性函数，下界为 ``RHOMINSND``（50 kg/m^3）::

       ZRHOMINSN = RHOMINSNA + RHOMINSNB*(PTSRF - RTT)
                                 + RHOMINSNC*(U^2 + V^2)^0.25
       ZRHOMINSN = max(ZRHOMINSN, RHOMINSND)

2. 旧积雪密度与新雪密度按雪*深度*加权的**混合**（CY43 之后的公式，
   Fortran 第 506-508 行）。``ZSSFC + ZSSFL`` 是耦合间隔内的降雪::

       ZRSTAR = (PSSNM1M + PTMST*(ZSSFC+ZSSFL))
              / (PTMST*(ZSSFC+ZSSFL)/ZRHOMINSN + PSSNM1M/PRSNM1M)

3. **覆压压实**（雪重下的黏性变形，Fortran 第 511-512 行）::

       ZRSNDTOVER = 0.5*PSSNM1M*RG
                  / (RSNDTOVERA*EXP(RSNDTOVERB*(RTT-PTSN)
                                    + RSNDTOVERC*PRSNM1M))

4. **破坏性变质作用**（雪粒圆化，Fortran 第 514-515 行）；当密度超过
   ``RSNDTDESTROI`` 后，``RSNDTDESTC`` 项抑制变质作用::

       ZRSNDTDEST = RSNDTDESTA
                  * EXP(-RSNDTDESTB*(RTT-PTSN)
                        - RSNDTDESTC*max(0, PRSNM1M - RSNDTDESTROI))

5. **消融压实**（Fortran 第 517-521 行），由内部消融/再冻结驱动的
   密度增加。``ZPMSNINT`` 是雪穿过消融窗口升温时释放的内部消融能量；
   该项仅在 ``ZPMSNINT > 0`` 且积雪足够深（``PSSNM1M >= 15``
   kg/m^2）时起作用::

       ZRSNDTMELT = ZTMST*(ZPMSNINT*PTMST/RLMLT)/(PSSNM1M - ZLW)

   ``ZPMSNINT`` 与 ``ZLW`` 来自雪液态水解析函数（第 357-377 行）：
   液态水容量 ``ZLWC``、消融窗口形状 ``ZFUNC`` 及其导数 ``ZDFUNC``、
   实际液态水 ``ZLW = ZFUNC*ZLWC``，以及内部消融通量
   ``ZPMSNINT = ZCONSB*ZFRSN*(PTSN - PTSNM1M)/PTMST``，其中
   ``ZCONSB = RLMLT*ZLWC*ZDFUNC``。

新密度为::

    PRSN = ZRSTAR + ZRSTAR*(ZRSNDTOVER + ZRSNDTDEST + ZRSNDTMELT)*PTMST
    PRSN = min(450, PRSN)
    PRSN = max(RHOMINSN, PRSN)

在无雪点（``LLNOSNOW``），密度重置为 ``RHOMINSN``（100 kg/m^3），
与 Fortran 的 ``PRSN(JL) = RHOMINSN`` 分支一致。

符号 / 单位约定
----------------------
所有密度单位为 kg/m^3。``PTSRF`` 是地表（表皮）温度，单位 K；
``PUSRF``/``PVSRF`` 是地表风分量，单位 m/s。风只通过
``(U^2+V^2)^0.25`` 进入，因此重要的是大小。``PTSN`` 是雪求解得到的
*新*雪温（Fortran 在压实/变质指数中使用刚算出的 ``PTSN(JL)``，而非
``PTSNM1M``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..constants import ConstantBundle
from ..tables.soil import SoilTable


@dataclass
class SnowDensityInputs:
    """单层雪密度求解的输入。

    除注明外，每个张量均为 ``(Ngrid,)``。命名镜像 ``SRFSN_LWIMP``
    的局部变量 / 哑元变量，便于逐行对照审计。
    """

    # ---- Snow state at t-1 ----
    pssnm1m: torch.Tensor       # PSSNM1M  snow mass (SWE)     [kg/m^2]   (Ngrid,)
    prsnm1m: torch.Tensor       # PRSNM1M  snow density        [kg/m^3]   (Ngrid,)
    ptsnm1m: torch.Tensor       # PTSNM1M  snow temperature    [K]        (Ngrid,)

    # ---- New snow state at t (from SRFSN_LWIMP solve) ----
    ptsn: torch.Tensor          # PTSN  new snow temperature   [K]        (Ngrid,)

    # ---- Surface forcing ----
    ptsrf: torch.Tensor         # PTSRF  surface (skin) T      [K]        (Ngrid,)
    pusrf: torch.Tensor         # PUSRF  surface zonal wind    [m/s]      (Ngrid,)
    pvsrf: torch.Tensor         # PVSRF  surface merid wind    [m/s]      (Ngrid,)

    # ---- Snowfall throughfall (per grid, large-scale + convective) ----
    pssfl: torch.Tensor         # PSSFL  large-scale snowfall  [kg/m^2/s] (Ngrid,)
    pssfc: torch.Tensor         # PSSFC  convective snowfall   [kg/m^2/s] (Ngrid,)

    # ---- Snow geometry / mask ----
    zfrsn: torch.Tensor         # ZFRSN  snow tile fraction    [-]        (Ngrid,)
    llnosnow: torch.Tensor      # LLNOSNOW  no-snow mask (bool) (Ngrid,)

    # ---- Timestep ----
    ptmst: torch.Tensor         # PTMST  coupling interval     [s]        scalar


@dataclass
class SnowDensityOutputs:
    """单层雪密度求解的输出。"""

    prsn: torch.Tensor          # PRSN  new snow density       [kg/m^3]   (Ngrid,)


def compute_snow_density(
    inp: SnowDensityInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
    *,
    differentiable: bool = True,
) -> SnowDensityOutputs:
    """运行单层雪密度演化（``SRFSN_LWIMP`` 3.1）。

    Args
    ----
    inp:
        :class:`SnowDensityInputs`——打包的雪状态、地表强迫以及来自
        雪求解的雪几何 / 掩码。
    soil_table:
        :class:`surf_pytorch.tables.soil.SoilTable`——提供密度常数
        （rhominsn*, rsndtover*, rsndtdest*, rhomaxsn_new）与液态水
        容量标量（rlwcswe*, rtempamp, rhoci, rhoice, rlamice, rdsmax,
        rfrsmall, rfrtiny）。
    consts:
        :class:`ConstantBundle`——提供 ``rtt``、``rlmlt``、``rg``
        和 ``pi``。
    differentiable:
        若为 ``False`` 则在 ``torch.no_grad()`` 下运行以提速。

    Returns
    -------
    :class:`SnowDensityOutputs`——新的雪密度 ``PRSN``。
    """

    def _core() -> SnowDensityOutputs:
        return _density_core(inp, soil_table, consts)

    if not differentiable:
        with torch.no_grad():
            return _core()
    return _core()


def _density_core(
    inp: SnowDensityInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
) -> SnowDensityOutputs:
    """内部：假定 no_grad / enable_grad 的决定已经做出。"""
    dtype = inp.pssnm1m.dtype
    device = inp.pssnm1m.device

    # ------------------------------------------------------------------ #
    # Scalar constants (same read order as the Fortran ASSOCIATE).
    # ------------------------------------------------------------------ #
    rtt = consts.rtt
    rlmlt = consts.rlmlt
    rg = consts.g
    rpi = consts.pi

    sc = soil_table.consts
    # snow-density constants are fixed; float() of a device tensor is a
    # host sync (forbidden during CUDA-graph capture) -- extract once.
    _scc = _density_core.__dict__.get('_sc_floats')
    if _scc is None or _scc[0] is not sc:
        _scc = (sc, tuple(float(getattr(sc, n)) for n in (
            'rhominsna', 'rhominsnb', 'rhominsnc', 'rhominsnd', 'rsndtovera', 'rsndtoverb', 'rsndtoverc', 'rsndtdesta', 'rsndtdestb', 'rsndtdestc', 'rsndtdestroi', 'rhomaxsn_new', 'rhominsn', 'rdsmax', 'rhoice', 'rhoci', 'rlwcswea', 'rlwcsweb', 'rlwcswec', 'rtempamp', 'rfrtiny')))
        _density_core.__dict__['_sc_floats'] = _scc
    (rhominsna, rhominsnb, rhominsnc, rhominsnd, rsndtovera, rsndtoverb, rsndtoverc, rsndtdesta, rsndtdestb, rsndtdestc, rsndtdestroi, rhomaxsn_new, rhominsn, rdsmax, rhoice, rhoci, rlwcswea, rlwcsweb, rlwcswec, rtempamp, rfrtiny) = _scc[1]

    ptmst = inp.ptmst
    ztmst = 1.0 / ptmst
    zt0 = rtt
    zhoice = 1.0 / rhoice

    # ------------------------------------------------------------------ #
    # Aliases for readability.
    # ------------------------------------------------------------------ #
    pssnm1m = inp.pssnm1m
    prsnm1m = inp.prsnm1m
    ptsnm1m = inp.ptsnm1m
    ptsn = inp.ptsn
    ptsrf = inp.ptsrf
    pusrf = inp.pusrf
    pvsrf = inp.pvsrf
    pssfl = inp.pssfl
    pssfc = inp.pssfc
    zfrsn = inp.zfrsn
    llnosnow = inp.llnosnow

    # ------------------------------------------------------------------ #
    # Snow liquid-water analytical functions (SRFSN_LWIMP lines 357-377).
    # These feed the melt-compaction term ZRSNDTMELT below.  They are the
    # same melt-window functions used by SRFSN_LWIMP, recomputed here so
    # the density block is self-contained.
    # ------------------------------------------------------------------ #
    zeps = 1.0e-12
    # real snow depth at t-1 (capped at RDSNMAX) and the liquid-water
    # capacity fraction ZCONSLWC (collapses to RLWCSWEC at high density).
    zfrsn_safe = torch.clamp(zfrsn, min=rfrtiny)
    prsnm1m_safe = torch.clamp(prsnm1m, min=zeps)
    # Preserve SRFSN_LWIMP's denominator grouping exactly.  Replacing this
    # division with products of reciprocals changes the final binary64 bits.
    zdsnr = pssnm1m / (prsnm1m_safe * zfrsn_safe)
    zdsn_safe = torch.clamp(zdsnr, min=zeps)
    zconslwc = torch.clamp((rlwcswea - prsnm1m) / rlwcswea, min=0.0)
    zlwc = pssnm1m * (rlwcsweb + (rlwcswec - rlwcsweb) * zconslwc)
    # rescale LWC by the effective-depth cap RDSNMAX
    zlwc = zlwc * torch.clamp(zdsnr, max=rdsmax) / zdsn_safe

    # ZFUNC / ZDFUNC: melt-window sine ramp over [RTT-RTEMPAMP, RTT].
    # Below the window ZFUNC=0 (frozen, no liquid water); inside the
    # window ZFUNC ramps 0->1->0 and ZDFUNC is its derivative; above RTT
    # ZFUNC=0 (all liquid) and ZDFUNC = pi/RTEMPAMP.
    t_below = ptsnm1m < (zt0 - 0.5 * rtempamp)
    t_inside = (ptsnm1m >= (zt0 - 0.5 * rtempamp)) & (ptsnm1m < zt0)
    phase = rpi * (ptsnm1m - zt0) / rtempamp
    zfunc_inside = 1.0 + torch.sin(phase)
    zdfunc_inside = torch.cos(phase) * rpi / rtempamp
    zdfunc_above = (rpi / rtempamp).expand(ptsnm1m.shape)
    zfunc = torch.where(t_below, torch.zeros_like(ptsnm1m),
                torch.where(t_inside, zfunc_inside, torch.zeros_like(ptsnm1m)))
    zdfunc = torch.where(t_below, torch.zeros_like(ptsnm1m),
                torch.where(t_inside, zdfunc_inside, zdfunc_above))
    zlw = zfunc * zlwc

    # ZCONSB: apparent heat-capacity contribution from the liquid-water
    # melt window (latent heat released per K of warming).
    zconsb = rlmlt * zlwc * zdfunc
    # ZPMSNINT: internal melt/refreeze energy flux (Fortran line 447).
    zpmsnint = zconsb * zfrsn * (ptsn - ptsnm1m) * ztmst

    # ------------------------------------------------------------------ #
    # 3.1 NEW SNOW DENSITY (SRFSN_LWIMP lines 494-527).
    # ------------------------------------------------------------------ #
    # Fresh-snow density (lines 498-500).
    wind_sq = pusrf ** 2 + pvsrf ** 2
    zrhominsn = rhominsna + rhominsnb * (ptsrf - zt0) \
        + rhominsnc * wind_sq ** 0.25
    zrhominsn = torch.clamp(zrhominsn, min=rhominsnd)

    # Depth-weighted mixing of old pack + fresh snow (lines 506-508).
    zssf = (pssfl + pssfc) * ptmst            # snowfall over the interval
    denom = zssf / torch.clamp(zrhominsn, min=zeps) + pssnm1m / prsnm1m_safe
    zrstar = (pssnm1m + zssf) / torch.clamp(denom, min=zeps)

    # Overburden compaction (lines 511-512).
    zrsndtover = 0.5 * pssnm1m * rg \
        / (rsndtovera * torch.exp(rsndtoverb * (zt0 - ptsn)
                                  + rsndtoverc * prsnm1m))

    # Destructive metamorphism (lines 514-515).
    zrsndtdest = rsndtdesta \
        * torch.exp(-rsndtdestb * (zt0 - ptsn)
                    - rsndtdestc * torch.clamp(prsnm1m - rsndtdestroi, min=0.0))

    # Melt compaction (lines 517-521): only where internal melt is
    # positive AND the pack is deep enough (PSSNM1M >= 15 kg/m^2).
    melt_active = (zpmsnint > 0.0) & (pssnm1m >= 15.0)
    pack_safe = torch.clamp(pssnm1m - zlw, min=zeps)
    zrsndtmelt = torch.where(
        melt_active,
        ztmst * (zpmsnint * ptmst / rlmlt) / pack_safe,
        torch.zeros_like(pssnm1m),
    )

    # Total compaction -> new density (lines 523-527).  The single-layer
    # path caps the evolved density at 450 (Fortran line 526); the multi-
    # layer path and the namelist RHOMAXSN_NEW use 500.  We follow the
    # single-layer SRFSN_LWIMP cap of 450.
    zrsndt = zrsndtover + zrsndtdest + zrsndtmelt
    prsn_snow = zrstar + zrstar * zrsndt * ptmst
    prsn_snow = torch.clamp(prsn_snow, max=450.0)
    prsn_snow = torch.clamp(prsn_snow, min=rhominsn)

    # No-snow branch (Fortran line 491): density resets to RHOMINSN.
    prsn = torch.where(llnosnow, torch.full((), rhominsn, dtype=prsn_snow.dtype, device=prsn_snow.device).expand(prsn_snow.shape), prsn_snow)

    return SnowDensityOutputs(prsn=prsn)


__all__ = [
    "SnowDensityInputs",
    "SnowDensityOutputs",
    "compute_snow_density",
]
