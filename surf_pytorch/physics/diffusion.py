"""隐式垂直扩散——离线单柱 SURFSEB 包装器。

本模块是 ``surf/offline/driver/vdfdifh1s.F90`` 的 Python 移植版，
即闭合地表能量平衡的隐式垂直扩散例程的*离线驱动*版本。

在完整的 IFS 中，``VDFDIFH`` 在模式层上运行 Thomas 算法的三对角求解，
并将向上回扫系数 ``(asl, bsl, aql, bql)`` 传入 ``SURFSEB``。在离线
单柱驱动（``vdfdifh1s.F90``）中，整个三对角块被 ``!1s `` 前缀注释掉，
并以 Dirichlet 边界条件（第 1.0 - 1.5 节，离线设置）替代::

    asl = 0, aql = 0
    bsl = PSLM1(:,KLEV) + PTMST*PSLGE(:,KLEV)
    bql = PQTM1(:,KLEV) + PTMST*PQTE(:,KLEV)

因此本移植版仅由最低模式层强迫组装 SURFSEB 输入，并运行 ``SURFSEB``
（第 3 次调用，即规范调用）。一个可选的、有条件的融雪重复调用
（第 4 次调用，由 ``LESNML`` 开关控制）处理低植被下雪面显式融化的分支。
可选的非胁迫蒸发诊断调用（第 1 次水体、第 2 次完全非胁迫）未在此转录——
它们仅用于诊断，且需要来自上游 VSURF 的额外 ``PCPTSTIU/PCAIRTIU/PCSATTIU``
输入；当参考内核调用这些分支时，M4-rem-c 将添加一个 ``pjqu`` 输出。

所复现的步骤（vdfdifh1s.F90 的节号）
------------------------------------------------
1.0-1.5  独立强迫（离线设置）::

     asl = 0,  aql = 0
     bsl = PSLM1(:,KLEV) + PTMST*PSLGE(:,KLEV)
     bql = PQTM1(:,KLEV) + PTMST*PQTE(:,KLEV)

1.6  逐瓦片的次表层温度 ``PTSRF``（第 372-388 行）：

       水体=PSST，海冰=PTICE，湿皮肤/低植被干/高植被干/
       裸土=PTSM1M，低植被雪/高植被雪=PTSNOW，湖泊=PTLICE
       （若冰厚超过 1e-9 m），否则为 PTLWML。

1.6b 交换系数反演（第 409-415 行）::

       rhoChU = PCFHTI / (RG * RVDIFTS * PTMST)
       rhoCqU = PCFQTI / (RG * RVDIFTS * PTMST)

     其中 PCFHTI / PCFQTI 携带 ``vdfmain1s`` 在调用 VDFDIFH1S 之前
     施加的上游 ``RG*PTSPHY*RVDIFTS`` 重标定（第 745-746 行）。
     在离线单柱驱动中 ``PTMST == PTSPHY``，因此该反演精确恢复
     ``rho*Ch*|U|``——即 :mod:`surf_pytorch.physics.seb` 所消耗的量。

1.7  SURFSEB 规范调用（第 3 次）-> 逐瓦片的 (PJS, PJQ, PSSH, PSLH,
     PSTR, PG0, PTSK) 以及格点平均的 (PSL, PQL)。

1.7a 林下积雪蒸发（第 557-558 行），施加于 Fortran 瓦片 7，
     即 :data:`surf_pytorch.config.TILE_HIGH_VEG_SNOW`::

       PJQ[:,TILE_HIGH_VEG_SNOW] += PEVAPSNW
       PSLH[:,TILE_HIGH_VEG_SNOW] += PEVAPSNW*RLSTT

1.7c 皮肤温度向 t+1 的外推（第 585-590 行）::

       PTSKTIP1 = ZTPFAC2*PTSK + ZTPFAC3*PTSKTI
                = (1/RVDIFTS)*PTSK + (1-1/RVDIFTS)*PTSKTI

LESNML 融雪重复（第 4 次调用，第 518-547 行）——可选，由
``DiffusionInputs.lesnml`` 控制。当任一柱满足 ``PFRTI(:,5) >= 1e-6``
*并且*第 3 次调用在低植被雪瓦片上的皮肤温度处于冰点时，例程将该瓦片的
``TSRF`` 固定为 ``RTT``、``LAMSK`` 固定为 100 W/m^2/K（模拟融化的雪面），
并重新运行 SURFSEB。当 ``lesnml`` 为 False 时该过程为空操作。

符号约定
---------------
输出继承 :mod:`surf_pytorch.physics.seb` 中记录的 SURFSEB_CTL 约定：
全部四个地表通量以及地面热通量均为**向下为正**。完整的闭合说明
参见该模块的文档字符串。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import (
    TILE_WATER,
    TILE_ICE,
    TILE_WET_SKIN,
    TILE_LOW_VEG_DRY,
    TILE_LOW_VEG_SNOW,
    TILE_HIGH_VEG_DRY,
    TILE_HIGH_VEG_SNOW,
    TILE_BARE_SOIL,
    TILE_LAKE,
    TILE_URBAN,
    SurfConfig,
)
from ..constants import ConstantBundle
from .seb import SebInputs, SebOutputs, surf_seb


# --------------------------------------------------------------------------- #
# VDFDIFH1S 硬编码常量（yoevdf / SUEVDF）
# --------------------------------------------------------------------------- #
# 超隐式时间权重 RVDIFTS。同时用于交换系数反演
# （ZCOEF1 = 1/(RG*RVDIFTS*PTMST)）和皮肤温度外推
# （ZTPFAC2 = 1/RVDIFTS，ZTPFAC3 = 1 - ZTPFAC2）。
#
# 注意：完整 IFS 使用默认值 RVDIFTS=1.5（在 arpifs/phys_ec/
# suvdf.F90 中设置），但*离线单柱驱动*在
# surf/offline/driver/suvdf.F90:57 中将其覆盖为 1.0。由于本移植版
# 以离线驱动为（验证）目标，此处固定 RVDIFTS=1.0。取此值时外推退化为
# PTSKTIP1 = PTSK（ZTPFAC2=1，ZTPFAC3=0），与驱动程序第 0 步的皮肤温度
# 转储完全一致。暴露为模块常量以便测试固定它。
RVDIFTS = 1.0

# 融雪重复（LESNML 分支）阈值，对应 vdfdifh1s 第 523-528 行。
_LESNML_FRAC_MIN = 1.0e-6     # 判定瓦片活跃的 PFRTI 阈值
_LESNML_TSK_EPS = 1.0e-6      # TSK - RTT 阈值（雪处于融点）
_LESNML_LAMSK_MELT = 100.0    # 融雪时固定的 lambda_skin [W/m^2/K]

# 湖冰厚度阈值（vdfdifh1s 第 382 行，SUSFLAKE 的 H_ICE_MIN）。
_LAKE_HICE_MIN = 1.0e-9


@dataclass
class DiffusionInputs:
    """VDFDIFH1S（离线单柱驱动）的全部输入。

    除非另有说明，每个张量的形状为 ``(Ngrid, Ntile)``（逐格点字段
    为 ``(Ngrid,)``）。命名与 ``vdfdifh1s.F90`` 的 Fortran 哑元列表
    保持一致，便于逐行对照审计。

    最低模式层强迫
    --------------------------
    ``pslm1_lev`` / ``pqtm1_lev`` 是 t-1 时刻最低层的干静力能与
    比湿；``pslge_lev`` / ``pqte_lev`` 是它们的倾向。它们构成离线
    驱动的 Dirichlet 边界条件（``bsl``、``bql``）。

    重标定的交换系数
    ------------------------------
    ``pcfhti`` / ``pcfqti`` 携带上游 ``vdfmain1s``（第 745-746 行）
    施加的 ``RG*PTSPHY*RVDIFTS`` 重标定；包装器通过
    ``ZCOEF1 = 1/(RG*RVDIFTS*PTMST)`` 将其反演。当由
    :class:`surf_pytorch.LandSurface` 调用时，模型在填充这些字段之前，
    会将原始的 ``rho*Ch*|U|``（来自
    :func:`surf_pytorch.physics.exchange.vexcs`）乘以
    ``RG*PTSPHY*RVDIFTS``，与上游约定一致。
    """

    # ---- KLEV 处最低模式层强迫（逐格点） ----
    pslm1_lev: torch.Tensor       # PSLM1(:,KLEV)  干静力能  [J/kg]
    pqtm1_lev: torch.Tensor       # PQTM1(:,KLEV)  比湿  [kg/kg]
    pslge_lev: torch.Tensor       # PSLGE(:,KLEV)  干静力能倾向  [J/kg/s]
    pqte_lev: torch.Tensor        # PQTE(:,KLEV)   湿度倾向  [kg/kg/s]

    # ---- 重标定的湍流交换系数（逐瓦片） ----
    pcfhti: torch.Tensor          # PCFHTI  热量交换系数（重标定后）
    pcfqti: torch.Tensor          # PCFQTI  水汽交换系数（重标定后）

    # ---- 下边界温度（逐格点），用于构建 PTSRF ----
    psst: torch.Tensor            # PSST  开阔海面温度  [K]
    ptice: torch.Tensor           # PTICE 海冰皮肤温度  [K]
    ptsm1m: torch.Tensor          # PTSM1M 顶层土壤温度  [K]
    ptsnow: torch.Tensor          # PTSNOW 雪面皮肤温度  [K]

    # ---- t-1 时刻的皮肤储库（逐瓦片，来自 VSURF） ----
    sskm1m: torch.Tensor          # PCPTSTI  cp*Tsk              [J/kg]
    tskm1m: torch.Tensor          # PTSKTI   t-1 时刻皮肤温度    [K]
    qskm1m: torch.Tensor          # PQSTI    qsat(Tsk)           [kg/kg]
    dqsdT: torch.Tensor           # PDQSTI   Tsk 处的 dqsat/dT   [kg/kg/K]

    # ---- 水汽通量闭合系数（逐瓦片，来自 VEVAP） ----
    alphal: torch.Tensor          # PCAIR  ql 的乘子      [-]
    alphas: torch.Tensor          # PCSAT  qs 的乘子      [-]

    # ---- 辐射 / 发射率 ----
    pssrflti: torch.Tensor        # PSSRFLTI (Ngrid, Ntile) 逐瓦片净短波  [W/m^2]
    pslrfl: torch.Tensor          # PSLRFL   (Ngrid,) 净长波              [W/m^2]
    ptskrad: torch.Tensor         # PTSKRAD  (Ngrid,) 辐射时刻的皮肤温度  [K]
    pemis: torch.Tensor           # PEMIS    (Ngrid,) 地表发射率          [-]

    # ---- 来自 VLAMSK 的皮肤热导率（逐瓦片） ----
    plamsk: torch.Tensor          # PLAMSK (Ngrid, Ntile)  [W/m^2/K]

    # ---- 瓦片覆盖比例与主导植被类型 ----
    pfrti: torch.Tensor           # PFRTI (Ngrid, Ntile)
    ktvl: torch.Tensor            # KTVL (Ngrid,) 长整型
    ktvh: torch.Tensor            # KTVH (Ngrid,) 长整型

    # ---- 时间步长 ----
    ptmst: torch.Tensor           # PTMST 标量  [s]

    # ---- 林下积雪蒸发（逐格点） ----
    pevapsnw: Optional[torch.Tensor] = None    # PEVAPSNW  [kg/m^2/s]

    # ---- 可选的湖泊瓦片皮肤量（仅当 n_tile > TILE_LAKE 时使用） ----
    phlice: Optional[torch.Tensor] = None      # PHLICE (Ngrid,) 湖冰厚度 [m]
    ptlice: Optional[torch.Tensor] = None      # PTLICE (Ngrid,) 湖冰皮肤温度 [K]
    ptlwml: Optional[torch.Tensor] = None      # PTLWML (Ngrid,) 湖泊混合层温度 [K]

    # ---- LESNML 融雪重复开关（第 4 次调用） ----
    lesnml: bool = False

    # ---- 继承自 SebInputs 的特性开关 ----
    lelwdd: bool = True
    lelwtl: bool = False


@dataclass
class DiffusionOutputs:
    """VDFDIFH1S（离线单柱驱动）的全部输出。

    继承 SURFSEB_CTL 的符号约定——全部四个地表通量以及地面热通量
    均为向下为正。闭合说明参见 :mod:`surf_pytorch.physics.seb`。
    """

    # 规范的 SURFSEB 输出（第 3 次调用，经林下积雪蒸发修正后）。
    pjq: torch.Tensor             # PJQ (Ngrid, Ntile) 水汽通量  [kg/m^2/s]
    pssh: torch.Tensor            # PSSH (Ngrid, Ntile) 感热通量 [W/m^2]
    pslh: torch.Tensor            # PSLH (Ngrid, Ntile) 潜热通量 [W/m^2]
    pstr: torch.Tensor            # PSTR (Ngrid, Ntile) 净热辐射 [W/m^2]
    pg0: torch.Tensor             # PG0  (Ngrid, Ntile) 地面热通量 [W/m^2]
    ptsk: torch.Tensor            # PTSK (Ngrid, Ntile) t+1/2 时刻皮肤温度 [K]

    # 外推到 t+1 的皮肤温度（ZTPFAC2*PTSK + ZTPFAC3*PTSKTI）。
    ptsktip1: torch.Tensor        # PTSKTIP1 (Ngrid, Ntile) t+1 时刻皮肤温度 [K]

    # 来自 SURFSEB 的格点平均最低层更新。
    psl: torch.Tensor             # PSL (Ngrid,) t+1/2 时刻干静力能  [J/kg]
    pql: torch.Tensor             # PQL (Ngrid,) t+1/2 时刻比湿  [kg/kg]


def vdfdifh1s(
    inp: DiffusionInputs,
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    differentiable: Optional[bool] = None,
) -> DiffusionOutputs:
    """运行围绕 SURFSEB 的离线单柱 VDFDIFH1S 包装器。

    参数
    ----
    inp:
        :class:`DiffusionInputs`——打包的逐（格点，瓦片）与逐格点
        输入。重标定交换系数的约定参见该数据类。
    consts:
        :class:`ConstantBundle`——提供 ``RG``、``RLSTT``、``RTT``
        以及 :func:`surf_seb` 所消耗的常量。
    config:
        :class:`SurfConfig`——``n_tile`` 决定哪些瓦片列处于活跃状态。
    differentiable:
        为本次调用覆盖 ``config.differentiable``。

    返回
    -------
    :class:`DiffusionOutputs`——逐瓦片通量、外推的皮肤温度，
    以及格点平均的最低层状态。
    """
    diff = config.differentiable if differentiable is None else differentiable

    def _core() -> DiffusionOutputs:
        return _vdfdifh1s_core(inp, consts, config)

    if not diff:
        with torch.no_grad():
            return _core()
    return _core()


def _vdfdifh1s_core(
    inp: DiffusionInputs,
    consts: ConstantBundle,
    config: SurfConfig,
) -> DiffusionOutputs:
    """内部函数：假定 no_grad / enable_grad 的决定已经做出。"""
    ngrid = inp.pslm1_lev.shape[0]
    ntile = config.n_tile
    dtype = inp.pslm1_lev.dtype
    device = inp.pslm1_lev.device

    rg = consts.g
    rlstt = consts.rlstt
    rtt = consts.rtt
    ptmst = inp.ptmst  # 0 维张量

    # ------------------------------------------------------------------ #
    # 步骤 1.0-1.5：独立 Dirichlet 强迫（离线设置）。
    # ------------------------------------------------------------------ #
    asl_grid = torch.zeros((ngrid,), dtype=dtype, device=device)
    bsl_grid = inp.pslm1_lev + ptmst * inp.pslge_lev
    aql_grid = torch.zeros((ngrid,), dtype=dtype, device=device)
    bql_grid = inp.pqtm1_lev + ptmst * inp.pqte_lev

    # ------------------------------------------------------------------ #
    # 步骤 1.6：逐瓦片的次表层温度 PTSRF。
    # ------------------------------------------------------------------ #
    tsrf = _build_tsrf(inp, ngrid, ntile, dtype, device)

    # ------------------------------------------------------------------ #
    # 步骤 1.6b：交换系数反演（ZCOEF1）。
    # ------------------------------------------------------------------ #
    # PCFHTI / PCFQTI 携带上游的 RG*PTSPHY*RVDIFTS 重标定；
    # SEB 消耗的是反演后的 "rho*Ch*|U|" 形式。
    ptmst_tile = ptmst.unsqueeze(-1) if ptmst.ndim == 1 else ptmst
    zcoef1 = 1.0 / (rg * RVDIFTS * ptmst_tile)
    rhochu = inp.pcfhti * zcoef1
    rhocqu = inp.pcfqti * zcoef1

    # ------------------------------------------------------------------ #
    # 步骤 1.7：SURFSEB 规范调用（第 3 次）。
    # ------------------------------------------------------------------ #
    seb_inp = SebInputs(
        sskm1m=inp.sskm1m, tskm1m=inp.tskm1m,
        qskm1m=inp.qskm1m, dqsdT=inp.dqsdT,
        rhochu=rhochu, rhocqu=rhocqu,
        alphal=inp.alphal, alphas=inp.alphas,
        ssrfl=inp.pssrflti,
        slrfl_grid=inp.pslrfl, tskrad_grid=inp.ptskrad, emis_grid=inp.pemis,
        tsrf=tsrf, lamsk=inp.plamsk,
        frti=inp.pfrti, ktvl_grid=inp.ktvl, ktvh_grid=inp.ktvh,
        asl_grid=asl_grid, bsl_grid=bsl_grid,
        aql_grid=aql_grid, bql_grid=bql_grid,
        ptmst=inp.ptmst,
        lelwdd=inp.lelwdd, lelwtl=inp.lelwtl,
    )
    out = surf_seb(seb_inp, consts, config)

    ptsk = out.ptsk           # 第 3 次调用的皮肤温度（可能被第 4 次覆盖）
    pstr = out.pstr
    pg0 = out.pg0
    pssh = out.pssh
    pjq = out.pjq.clone() if inp.pevapsnw is not None else out.pjq
    pslh = out.pslh.clone() if inp.pevapsnw is not None else out.pslh

    # ------------------------------------------------------------------ #
    # LESNML 融雪重复（第 4 次调用）：若任一柱存在融化的
    # 低植被雪瓦片（Fortran JT=5 = 移植版 TILE_LOW_VEG_SNOW=4），
    # 则以该瓦片固定为 RTS=RTT、LAMSK=100 重新运行 SURFSEB。
    # ------------------------------------------------------------------ #
    if inp.lesnml and ntile > TILE_LOW_VEG_SNOW:
        ptsk, pstr, pg0, pssh, pjq, pslh = _lesnml_repeat(
            inp, seb_inp, out, ptsk, pstr, pg0, pssh, pjq, pslh,
            consts, config, rtt, dtype, device,
        )

    # ------------------------------------------------------------------ #
    # 步骤 1.7a：林下积雪蒸发。
    # Fortran 瓦片 7 == 移植版 TILE_HIGH_VEG_SNOW（6）。
    # ------------------------------------------------------------------ #
    if inp.pevapsnw is not None and ntile > TILE_HIGH_VEG_SNOW:
        pjq[:, TILE_HIGH_VEG_SNOW] = pjq[:, TILE_HIGH_VEG_SNOW] + inp.pevapsnw
        pslh[:, TILE_HIGH_VEG_SNOW] = pslh[:, TILE_HIGH_VEG_SNOW] + inp.pevapsnw * rlstt

    # ------------------------------------------------------------------ #
    # 步骤 1.7c：皮肤温度向 t+1 的外推。
    # ------------------------------------------------------------------ #
    ztpfac2 = 1.0 / RVDIFTS
    ztpfac3 = 1.0 - ztpfac2
    ptsktip1 = ztpfac2 * ptsk + ztpfac3 * inp.tskm1m

    return DiffusionOutputs(
        pjq=pjq, pssh=pssh, pslh=pslh, pstr=pstr, pg0=pg0,
        ptsk=ptsk, ptsktip1=ptsktip1,
        psl=out.psl, pql=out.pql,
    )


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #
def _build_tsrf(
    inp: DiffusionInputs,
    ngrid: int,
    ntile: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """逐瓦片的次表层温度 PTSRF（vdfdifh1s 第 372-388 行）。

    瓦片映射（Fortran JT -> 移植版索引）：
        1 水体        -> PSST
        2 海冰        -> PTICE
        3 湿皮肤      -> PTSM1M
        4 低植被干    -> PTSM1M
        5 低植被雪    -> PTSNOW
        6 高植被干    -> PTSM1M
        7 高植被雪    -> PTSNOW
        8 裸土        -> PTSM1M
        9 湖泊        -> 冰厚超过 1e-9 时取 PTLICE，否则取 PTLWML
       10 城镇        -> PTSM1M（尚未接线——需要 LEURBAN 开关）
    """
    tsrf = torch.empty((ngrid, ntile), dtype=dtype, device=device)
    tsrf[:, TILE_WATER] = inp.psst
    if ntile > TILE_ICE:
        tsrf[:, TILE_ICE] = inp.ptice
    if ntile > TILE_WET_SKIN:
        tsrf[:, TILE_WET_SKIN] = inp.ptsm1m
    if ntile > TILE_LOW_VEG_DRY:
        tsrf[:, TILE_LOW_VEG_DRY] = inp.ptsm1m
    if ntile > TILE_LOW_VEG_SNOW:
        tsrf[:, TILE_LOW_VEG_SNOW] = inp.ptsnow
    if ntile > TILE_HIGH_VEG_DRY:
        tsrf[:, TILE_HIGH_VEG_DRY] = inp.ptsm1m
    if ntile > TILE_HIGH_VEG_SNOW:
        tsrf[:, TILE_HIGH_VEG_SNOW] = inp.ptsnow
    if ntile > TILE_BARE_SOIL:
        tsrf[:, TILE_BARE_SOIL] = inp.ptsm1m
    if ntile > TILE_LAKE and inp.phlice is not None \
            and inp.ptlice is not None and inp.ptlwml is not None:
        # vdfdifh1s 第 382 行：PHLICE > 1e-9 时选取冰面皮肤温度，
        # 否则选取混合层温度。
        is_ice = (inp.phlice > _LAKE_HICE_MIN).to(dtype)
        tsrf[:, TILE_LAKE] = is_ice * inp.ptlice + (1.0 - is_ice) * inp.ptlwml
    elif ntile > TILE_LAKE:
        # FLake 预报量未暴露（use_flake=False 的占位分支）。
        # 回退到 PTSM1M，以免湖泊瓦片停留在 ``torch.empty``
        # 分配的内存上（未初始化内存——与下方城镇瓦片同类的
        # 跨柱"泄漏"问题）。
        tsrf[:, TILE_LAKE] = inp.ptsm1m
    if ntile > TILE_URBAN:
        # vdfdifh1s 第 387 行：城镇瓦片的次表层温度为 PTSM1M
        # （与裸土相同）。若无此行，该列将停留在 ``torch.empty``
        # 分配的内存上——即*未初始化内存*，其取值依赖于批次调用，
        # 会产生跨柱"泄漏"，n_tile=10 的批次无关性测试可捕获该问题。
        tsrf[:, TILE_URBAN] = inp.ptsm1m
    return tsrf


def _lesnml_repeat(
    inp: DiffusionInputs,
    seb_inp: SebInputs,
    out: SebOutputs,
    ptsk: torch.Tensor,
    pstr: torch.Tensor,
    pg0: torch.Tensor,
    pssh: torch.Tensor,
    pjq: torch.Tensor,
    pslh: torch.Tensor,
    consts: ConstantBundle,
    config: SurfConfig,
    rtt: float,
    dtype: torch.dtype,
    device,
) -> tuple:
    """LESNML 融雪重复（第 4 次调用，vdfdifh1s 第 518-547 行）。

    对于满足 ``PFRTI(:,TILE_LOW_VEG_SNOW) >= 1e-6`` *并且*第 3 次调用
    在该瓦片上的皮肤温度处于融点（``ZTSK(:,5) >= RTT - 1e-6``）的柱，
    将该瓦片的 ``TSRF`` 固定为 ``RTT``、``LAMSK`` 固定为 100 W/m^2/K，
    并重新运行 SURFSEB。对于这些柱，新输出会覆盖*所有*瓦片上的第 3 次
    调用输出（与 Fortran 一致——Fortran 在整个 KLON/NTILES 切片上调用
    SURFSEB，而非逐柱调用）。
    """
    # 构建逐（格点，瓦片）的重复掩码。Fortran 中的 LREPEAT 是跨柱
    # 做 OR 归约的；随后 SURFSEB 调用在整个切片上运行。此处复现
    # 该行为：只要任一柱触发即进行重复调用。
    active = inp.pfrti[:, TILE_LOW_VEG_SNOW] >= _LESNML_FRAC_MIN
    melting = out.ptsk[:, TILE_LOW_VEG_SNOW] >= rtt - _LESNML_TSK_EPS
    any_repeat = (active & melting).any()
    if not any_repeat:
        return ptsk, pstr, pg0, pssh, pjq, pslh

    # 对融化的柱，将低植被雪瓦片上的 TSRF 与 LAMSK 固定。
    tsrf_r = seb_inp.tsrf.clone()
    lamsk_r = seb_inp.lamsk.clone()
    mask = (active & melting).to(dtype)
    # 掩码为 1 处固定为 RTT，其余位置保持原值。
    tsrf_r[:, TILE_LOW_VEG_SNOW] = (
        mask * rtt + (1.0 - mask) * tsrf_r[:, TILE_LOW_VEG_SNOW]
    )
    lamsk_r[:, TILE_LOW_VEG_SNOW] = (
        mask * _LESNML_LAMSK_MELT + (1.0 - mask) * lamsk_r[:, TILE_LOW_VEG_SNOW]
    )

    repeat_inp = SebInputs(
        sskm1m=seb_inp.sskm1m, tskm1m=seb_inp.tskm1m,
        qskm1m=seb_inp.qskm1m, dqsdT=seb_inp.dqsdT,
        rhochu=seb_inp.rhochu, rhocqu=seb_inp.rhocqu,
        alphal=seb_inp.alphal, alphas=seb_inp.alphas,
        ssrfl=seb_inp.ssrfl,
        slrfl_grid=seb_inp.slrfl_grid, tskrad_grid=seb_inp.tskrad_grid,
        emis_grid=seb_inp.emis_grid,
        tsrf=tsrf_r, lamsk=lamsk_r,
        frti=seb_inp.frti, ktvl_grid=seb_inp.ktvl_grid, ktvh_grid=seb_inp.ktvh_grid,
        asl_grid=seb_inp.asl_grid, bsl_grid=seb_inp.bsl_grid,
        aql_grid=seb_inp.aql_grid, bql_grid=seb_inp.bql_grid,
        ptmst=seb_inp.ptmst,
        lelwdd=seb_inp.lelwdd, lelwtl=seb_inp.lelwtl,
    )
    repeat_out = surf_seb(repeat_inp, consts, config)
    return repeat_out.ptsk, repeat_out.pstr, repeat_out.pg0, \
        repeat_out.pssh, repeat_out.pjq, repeat_out.pslh


__all__ = [
    "DiffusionInputs",
    "DiffusionOutputs",
    "vdfdifh1s",
    "RVDIFTS",
]
