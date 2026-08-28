"""地表水库／冠层截留更新（``SRFWL``）。

本模块是 ``SRFWL`` 的 Python 移植——即 ``SURFTSTP_CTL``
（``surftstp_ctl_mod.F90:612``）所使用的地表水库（冠层截留）预报量更新。
它将地表水库含水量 ``PWL``（截留水量，kg/m^2）推进一个耦合时间间隔，
并产出到达土壤的穿透雨通量 ``PTSFC`` / ``PTSFL``（对流性／大尺度降水），
以及土壤水分求解所消耗的逐斑块截留层蒸发不匹配量 ``PEINTTI``。

方案内容
--------------------
地表水库表示两次降雨事件之间滞留在叶片与裸面上的水分。在一个耦合时间
间隔 ``PTMST`` 内，预报量通过五个可向量化阶段推进
（``srfwl_mod.F90`` 164-300）：

1. **截断至容量上限**（label 2）：``PWL = clip(PWLM1M, 0, PWLMX)``。
   任何截断都以虚拟蒸发的形式记到 ``PEINTTI(:,3)``（湿表皮斑块）上，
   土壤水分求解必须看到这部分。

2. **水库蒸发**（label 3）：当湿表皮斑块蒸发时（``PEVAPTI(:,3) < 0``），
   水库按 ``PTMST*PFRTI(:,3)*PEVAPTI(:,3)`` 减少，下限取零。实际
   （取零下限之后）的通量记到 ``PEINTTI(:,3)`` 上。

3. **露水收集**（label 4）：水汽通量为正的斑块 4/6/7/8/10（端口
   3/5/6/7/9）向水库凝结露水，直至 ``PWLMX``。高植被下积雪斑块
   （F90 7／端口 6）排除冠层下积雪升华 ``PEVAPSNW``（仅液态部分被
   截留）。每个实际露水增量记到相应的 ``PEINTTI`` 列上。

4. **收支累积**（label 5）：``PFWEL1 = (PWL - PWLM1M)/PTMST``
   （净地表水库通量倾向，kg/m^2/s）。

5. **降水截留**（label 6）：大尺度与对流降雨被植被覆盖度
   ``(CVL+CVH)*RVINTER`` 截留，直至剩余容量 ``PWLMX-PWL``。
   穿透雨 ``PTSFL`` / ``PTSFC`` 即到达土壤的部分。

符号约定
---------------
``PEVAPTI`` 以*向下*为正（露水／沉积进入地表），向上（蒸发）为负。
``PEINTTI`` 遵循相同约定：正值表示截留水库*吸收*了通量，土壤因此
*看不到*这部分（故土壤水分求解从其地表水汽输入中减去它）；负值表示
水库*释放*了通量，土壤必须将其视为额外输入。土壤水分求解按斑块消耗
``PEINTTI = PEVAPTI(: realisable) - PEVAPTI(: intercepted)``。

冷启动简化
--------------------------
* 冷启动时对流降雨 ``PRSFC`` 为零（本移植的 :class:`Forcing` 不区分
  对流与大尺度降水），故 ``PTSFC`` 保持为零；仅大尺度分支 ``PRSFL``
  起作用。这与 :meth:`LandSurface._update_soil_moisture` 中土壤水分
  穿透雨所用的约定一致。
* 在单层雪方案将其接回之前，``PEVAPSNW``（林下积雪蒸发）为零；因此
  露水收集分支对高植被积雪斑块的处理与其他斑块相同。

索引约定
----------------
所有逐斑块张量均为 0 索引。Fortran 斑块 3（湿表皮）映射为
:data:`surf_pytorch.config.TILE_WET_SKIN`（2）；露水收集斑块
F90 4/6/7/8/10 映射为端口 3/5/6/7/9（低植被干／高植被干／
高植被雪／裸土／城市）。斑块索引由 ``ntile`` 防护，较窄的斑块切片
不会引发越界。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import (
    TILE_WET_SKIN,
    TILE_LOW_VEG_DRY,
    TILE_HIGH_VEG_DRY,
    TILE_HIGH_VEG_SNOW,
    TILE_BARE_SOIL,
    TILE_URBAN,
)
from ..tables.veg import VegTable


# --------------------------------------------------------------------------- #
# Fortran tile indices used by SRFWL, mapped to the port tile enum.
# Fortran PFRTI(JL,3) == port TILE_WET_SKIN (2); dew-collection tiles are
# Fortran 3/4/6/7/8/10 == port 2/3/5/6/7/9.  Indices are guarded by ntile below.
# --------------------------------------------------------------------------- #
# Dew-collection tile set (Fortran tiles 3, 4, 6, 7, 8, 10).  Each entry is the
# (port_index, is_snow_under_high_veg) pair; only the snow-under-high-veg
# tile excludes PEVAPSNW from its intercepted flux.
_DEW_TILES = (
    (TILE_WET_SKIN, False),        # F90 3 -> port 2
    (TILE_LOW_VEG_DRY, False),     # F90 4 -> port 3
    (TILE_HIGH_VEG_DRY, False),    # F90 6 -> port 5
    (TILE_HIGH_VEG_SNOW, True),    # F90 7 -> port 6 (snow-under-high-veg)
    (TILE_BARE_SOIL, False),       # F90 8 -> port 7
    (TILE_URBAN, False),           # F90 10 -> port 9
)


@dataclass
class CanopyInputs:
    """地表水库更新的全部输入。

    除注明外，每个张量形状为 ``(Ngrid,)``。逐斑块字段为
    ``(Ngrid, Ntile)``。命名对应 ``SRFWL`` 的形参列表。
    """

    # Coupling interval [s].
    ptmst: torch.Tensor             # PTMST  (scalar or (Ngrid,))

    # t-1 skin-reservoir water content [kg/m^2].
    pwlm1m: torch.Tensor            # PWLM1M

    # Vegetation cover fractions (0-1) and interception capacity [kg/m^2].
    pcvl: torch.Tensor              # PCVL   low-veg cover
    pcvh: torch.Tensor              # PCVH   high-veg cover
    pwlmx: torch.Tensor             # PWLMX  max skin-reservoir capacity

    # Per-tile fractions / moisture flux [(Ngrid, Ntile)].
    pfrti: torch.Tensor             # PFRTI  tile fractions (0-1)
    pevapti: torch.Tensor           # PEVAPTI  surface moisture flux [kg/m^2/s]

    # Rainfall fluxes at the surface [kg/m^2/s].
    prsfc: torch.Tensor             # PRSFC  convective rain
    prsfl: torch.Tensor             # PRSFL  large-scale rain

    # Snow-under-forest evaporation [kg/m^2/s] (excluded from the
    # snow-high-veg dew collection); zero on the cold start.
    pevapsnw: torch.Tensor          # PEVAPSNW

    # Land mask (bool).
    ldland: torch.Tensor            # LDLAND
    lewbsoilfix: bool = True         # YDSOIL%LEWBSOILFIX


@dataclass
class CanopyOutputs:
    """地表水库更新的输出。

    除 ``peintti`` 为 ``(Ngrid, Ntile)`` 外，其余均为 ``(Ngrid,)``。
    """

    pwl: torch.Tensor               # PWL  t+1 skin-reservoir water [kg/m^2]
    pfwel1: torch.Tensor            # PFWEL1  net skin-reservoir flux [kg/m^2/s]
    ptsfc: torch.Tensor             # PTSFC  convective throughfall [kg/m^2/s]
    ptsfl: torch.Tensor             # PTSFL  large-scale throughfall [kg/m^2/s]
    peintti: torch.Tensor           # PEINTTI  interception-layer evap mismatch


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def update_canopy(
    inp: CanopyInputs,
    veg_table: VegTable,
) -> CanopyOutputs:
    """将地表水库预报量推进一个耦合步长。

    ``SRFWL``（``srfwl_mod.F90`` 164-300）的向量化转录。每个阶段都是
    在 ``(Ngrid,)`` / ``(Ngrid, Ntile)`` 轴上的逐元素 torch 运算，
    因此单次调用即可精确复现上游的 JL 循环（包括逐斑块的露水收集
    循环，该循环是列局部的）。

    Parameters
    ----------
    inp:
        :class:`CanopyInputs` —— t-1 时刻状态 + 通量 + 掩码。
    veg_table:
        :class:`VegTable` —— 提供 ``RVINTER``（截留效率，
        默认 0.5）。

    Returns
    -------
    :class:`CanopyOutputs`
        t+1 时刻的地表水库水量 ``PWL``、净通量倾向 ``PFWEL1``、
        对流／大尺度穿透雨 ``PTSFC`` / ``PTSFL``，以及逐斑块截留层
        蒸发不匹配量 ``PEINTTI``。
    """
    device = inp.pwlm1m.device
    dtype = inp.pwlm1m.dtype
    ngrid = inp.pwlm1m.shape[0]
    ntile = inp.pevapti.shape[-1]
    ptmst = inp.ptmst
    # Broadcast a scalar timestep to (Ngrid,) for uniform arithmetic.
    if ptmst.dim() == 0:
        ptmst = ptmst.expand(ngrid).to(dtype=dtype, device=device)
    ztmst = 1.0 / ptmst                          # 1/PTMST

    # Interception efficiency RVINTER (YDVEG%RVINTER, default 0.5).
    rvinter = veg_table.consts.rvinter
    # Convective-precip efficiency factor RPSFR (YDSOIL%RPSFR, default 2.0).
    # Upstream folds PRSFC through RPSFR: ZTPRCP = PRSFC*RPSFR and the
    # intercepted mass is scaled back by 1/RPSFR (ZPSFR = 1/RPSFR).  The
    # veg table does not yet carry RPSFR, so we hard-code the offline-
    # driver default (rdnml_params.F90:299 RPSFR = 2.0).  This mirrors the
    # single value the comparison harness runs with.
    rpsfr = 2.0
    zpsfr = 1.0 / rpsfr

    # Security floor on the reservoir capacity and precip flux
    # (ZEPTINY = 10*TINY(RHOH2O); ZEPPRCP = ZEPTINY).  TINY(float64) ~ 2e-16.
    zeptiny = torch.full((ngrid,), 10.0 * torch.finfo(dtype).tiny,
                         dtype=dtype, device=device)
    zepprcp = zeptiny

    ldland = inp.ldland                       # (Ngrid,) bool
    land = ldland.to(dtype=dtype)             # 1.0 on land, 0.0 on sea

    # ------------------------------------------------------------------ #
    # 1. Initialise PEINTTI and apply the LEWBSOILFIX reservoir clamp.
    # ------------------------------------------------------------------ #
    peintti = torch.zeros((ngrid, ntile), dtype=dtype, device=device)
    pwl_clamped = torch.minimum(inp.pwlmx - zeptiny, inp.pwlm1m)
    pwl_clamped = torch.clamp(pwl_clamped, min=0.0) * land
    if inp.lewbsoilfix:
        pwl = pwl_clamped
    else:
        pwl = inp.pwlm1m * land
    if ntile > TILE_WET_SKIN and inp.lewbsoilfix:
        clamp_flux = (pwl_clamped - inp.pwlm1m) * ztmst
        peintti[:, TILE_WET_SKIN] = clamp_flux * land

    # ------------------------------------------------------------------ #
    # 2. Evaporation of the skin reservoir (label 3, PEVAPTI(:,3) < 0).
    # ------------------------------------------------------------------ #
    # Wet-skin tile is port TILE_WET_SKIN (2) == Fortran tile 3.  Guarded
    # by ntile so a narrower slab does not index out of range.
    evap_wet = torch.zeros((ngrid,), dtype=dtype, device=device)
    if ntile > TILE_WET_SKIN:
        evap_wet = inp.pevapti[:, TILE_WET_SKIN]
    is_evap = evap_wet < 0.0
    # ZQHFLW = PTMST*PFRTI(:,3)*PEVAPTI(:,3)  (negative: draining reservoir).
    zqhf_lw = ptmst * _tile_frac(inp.pfrti, TILE_WET_SKIN, ntile) * evap_wet
    # PWL = max(0, PWL + ZQHFLW); only where evaporating.
    zwl_evap = pwl
    pwl_new = torch.clamp(zwl_evap + zqhf_lw, min=0.0)
    pwl = torch.where(is_evap, pwl_new, pwl)
    # Book the realised (post-floor) flux onto PEINTTI(:,3):
    #   PEINTTI(:,3) += (PWL - ZWL)/PTMST.
    realised_evap = (pwl - zwl_evap) * ztmst
    if ntile > TILE_WET_SKIN:
        peintti[:, TILE_WET_SKIN] = peintti[:, TILE_WET_SKIN] + \
            torch.where(is_evap, realised_evap, torch.zeros_like(realised_evap))

    dew_touched_reservoir = torch.zeros((ngrid,), dtype=torch.bool, device=device)

    # ------------------------------------------------------------------ #
    # 3. Dew collection (label 4): tiles 4/6/7/8/10 with ZELIQUID > 0
    #    deposit dew onto the reservoir up to PWLMX.
    # ------------------------------------------------------------------ #
    for port_idx, is_snow_high in _DEW_TILES:
        if port_idx >= ntile:
            continue
        pevap_tile = inp.pevapti[:, port_idx]
        if is_snow_high:
            # F90 tile 7: ZELIQUID = PFRTI*(PEVAPTI - PEVAPSNW) -- only the
            # liquid fraction is intercepted; the snow-under-canopy
            # sublimation PEVAPSNW bypasses the reservoir.
            zliquid = inp.pfrti[:, port_idx] * (pevap_tile - inp.pevapsnw)
        else:
            zliquid = inp.pfrti[:, port_idx] * pevap_tile
        is_dew = zliquid > 0.0
        # ZQHFLW = PTMST*ZELIQUID; PWL += min(PWLMX - ZEPTINY - PWL, ZQHFLW).
        zqhf = ptmst * zliquid
        zwl_dew = pwl
        headroom = inp.pwlmx - zeptiny - zwl_dew
        inc = torch.minimum(headroom, zqhf)
        # SRFWL: PWL = ZWL + MIN(PWLMX - ZEPTINY - ZWL, ZQHFLW).
        # Do not clamp the subsequent addition to PWLMX: doing so changes
        # its IEEE-754 rounding and can erase the tiny non-zero tile fraction
        # retained by the reference implementation.
        dew_candidate = zwl_dew + inc
        pwl = torch.where(is_dew, dew_candidate, pwl)
        # Book the realised dew increment onto PEINTTI(:, tile).
        realised_dew = (pwl - zwl_dew) * ztmst
        dew_touched_reservoir = dew_touched_reservoir | (is_dew & (pwl != zwl_dew))
        peintti[:, port_idx] = peintti[:, port_idx] + \
            torch.where(is_dew, realised_dew, torch.zeros_like(realised_dew))

    # ------------------------------------------------------------------ #
    # 4. Budget accumulation (label 5): PFWEL1 = (PWL - PWLM1M)/PTMST.
    #    Sea points carry PFWEL1 = 0 (reservoir is forced to zero there).
    # ------------------------------------------------------------------ #
    pfwel1 = (pwl - inp.pwlm1m) * ztmst * land

    # ------------------------------------------------------------------ #
    # 5. Interception of precipitation (label 6).
    #    Large-scale: ZMPRCP = min(PWLMX - ZEPTINY - PWL, PTMST*ZIPRCP);
    #                 PTSFL = PRSFL - ZMPRCP/PTMST.
    #    Convective:  scaled by RPSFR (ZTPRCP = PRSFC*RPSFR, mass back by
    #                 1/RPSFR); PTSFC = PRSFC - ZMPRCP/PTMST.
    #    Sea points: throughfall = raw rainfall.
    # ------------------------------------------------------------------ #
    headroom_l = inp.pwlmx - zeptiny - pwl

    # Large-scale rain.
    zziprcp_l = inp.prsfl * (inp.pcvl + inp.pcvh) * rvinter
    zmprcp_l = torch.minimum(headroom_l, ptmst * zziprcp_l)
    has_ls = inp.prsfl > zepprcp
    # ZMPRCP is added back to the reservoir (PWL += ZMPRCP) and removed
    # from the throughfall (PTSFL = PRSFL - ZMPRCP/PTMST).  Both updates
    # only fire where there is large-scale rain (has_ls); elsewhere the
    # reservoir is untouched and the throughfall is zero.
    zmprcp_l = torch.where(has_ls, zmprcp_l, torch.zeros_like(zmprcp_l))
    pwl_before_ls = pwl
    pwl = pwl + zmprcp_l
    ptsfl_land = torch.where(has_ls, inp.prsfl - zmprcp_l * ztmst,
                             torch.zeros_like(inp.prsfl))

    # Convective rain (RPSFR-folded).  Recompute the headroom after the
    # large-scale interception has filled part of the reservoir.
    headroom_c = inp.pwlmx - zeptiny - pwl
    ztprcp_c = inp.prsfc * rpsfr
    zziprcp_c = ztprcp_c * (inp.pcvl + inp.pcvh) * rvinter
    zmprcp_c = torch.minimum(headroom_c, ptmst * zziprcp_c) * zpsfr
    has_cv = inp.prsfc > zepprcp
    zmprcp_c = torch.where(has_cv, zmprcp_c, torch.zeros_like(zmprcp_c))
    pwl_before_cv = pwl
    pwl = pwl + zmprcp_c
    ptsfc_land = torch.where(has_cv, inp.prsfc - zmprcp_c * ztmst,
                             torch.zeros_like(inp.prsfc))

    # Sea points: throughfall = raw rainfall (no interception).
    ptsfl = torch.where(ldland, ptsfl_land, inp.prsfl)
    ptsfc = torch.where(ldland, ptsfc_land, inp.prsfc)

    # Sea points: force the reservoir to zero (label 2 / 3 sea branch).
    pwl = pwl * land

    return CanopyOutputs(
        pwl=pwl,
        pfwel1=pfwel1,
        ptsfc=ptsfc,
        ptsfl=ptsfl,
        peintti=peintti,
    )


def _tile_frac(pfrti: torch.Tensor, port_idx: int, ntile: int) -> torch.Tensor:
    """返回 ``pfrti[:, port_idx]``；若该斑块不存在则返回零。

    对应本移植中贯穿使用的 ``ntile`` 防护模式，使较窄的斑块切片
    （例如 4 斑块测试）不会索引越界。
    """
    ngrid = pfrti.shape[0]
    if port_idx >= ntile:
        return torch.zeros((ngrid,), dtype=pfrti.dtype, device=pfrti.device)
    return pfrti[:, port_idx]
