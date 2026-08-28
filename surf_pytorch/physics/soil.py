"""HTESSEL 土壤温度更新（表观热容 + 隐式 Thomas 求解）。

本模块是 ``SURFTSTP_CTL`` 土壤温度分支的 Python 移植
（``surf/module/surftstp_ctl_mod.F90`` 第 3 节），
由三个上游例程组成：

* ``SRFRCGS`` (``surf/module/srfrcgs_mod.F90``) -- 表观体积热容。
  "表观"（apparent）一词指的是冻结/融化项：当土壤水分处于冻结窗口
  ``[RTF2, RTF1]`` 内时，热容通过
  ``PCTSA = ZRCSOIL - RLMLT*RHOH2O*ZWA*ZDFDT`` 吸收潜热的释放/吸收，
  其中 ``ZDFDT = -0.5*RTF4*COS(RTF4*(T-RTF3))``
  （即 SRFENE 中液态水比例 ``F(T) = 0.5*(1 - SIN(RTF4*(T-RTF3)))`` 的导数）。

* ``SRFTS``  (``surf/module/srfts_mod.F90``)     -- 组装土壤温度求解的
  三对角系数：地表净热通量 ``ZSURFL``、Peters-Liddard / Kersten
  热导率 ``ZDIF``、层热容 ``ZCDZ = PCTSA*RDAT``，以及右端项。

* ``SRFWDIFS`` (``surf/module/srfwdifs_mod.F90``) -- Thomas 三对角求解器。
  尽管带有 ``W``（水）前缀，这是一个通用求解器，同时用于土壤温度
  （来自 ``SRFTS``）和土壤水分（来自 M5b 路径）。它携带 ``LDALLAYS``
  标志，允许 ``SRFTS`` 选择不求解顶层以下的层次。

新温度由下式闭合：
``PTSA = PTSAM1M*ZCONS2 + ZTSA``，其中
``ZCONS2 = 1 - 1/RSIMP``（超隐式时间权重；离线驱动中 RSIMP=1，
因此 ZCONS2=0，新温度 T 就是 Thomas 解 ``ZTSA`` 本身——即全隐式
Euler 步）。

复现的步骤（SRFTS 节号）
----------------------------------------
1.  计算常数
    ``ZCONS1 = PTMST*RSIMP``，``ZCONS2 = 1 - 1/RSIMP``。

2.  地表净热通量 ``ZSURFL``（SRFTS 第 183-218 行）：
    对四个面向土壤的贴块（湿皮肤 3、低植被-干 4、高植被-干 6、
    裸土 8）上的净短波 + 净长波 +（感热 + 潜热）做面积加权平均，
    再加上雪底热通量 ``PGSN``。

3.  逐层热导率 ``ZDIF``（SRFTS 第 223-247 行，
    Peters-Liddard / Kersten 公式）：
    ``ZLAMBDASAT = RLAMSAT1M * RLAMBDAICE**(WSAT-W) * RLAMBDAWAT**W``，
    ``ZKERSTEN   = RKERST2*LOG10(MAX(RKERST1, W/WSAT)) + RKERST3``，
    ``ZDIF       = RLAMBDADRYM + ZKERSTEN*(ZLAMBDASAT - RLAMBDADRYM)``。

4.  Thomas 系数组装（SRFTS 第 252-281 行）：
    层间热耦合（半层 JK+1/2）
    ``ZLST(JK) = ZCONS1*(ZDIF(JK)+ZDIF(JK+1))/(RDAT(JK)+RDAT(JK+1))``，
    以及层热容 ``ZCDZ(JK) = PCTSA(JK)*RDAT(JK)``。
    地表右端项 ``ZRHS(:,1) = PTMST*ZSURFL/ZCDZ(:,1)`` 注入地表净通量；
    深层的右端项为零（无内部源项）。
    底层 ``ZLST(:,KLEVS) = 0`` 强制零通量下边界条件。

5.  Thomas 求解（SRFWDIFS）：见 :func:`_solve_tridiag`。

6.  新温度 ``PTSA = PTSAM1M*ZCONS2 + ZTSA``。

冷启动简化
--------------------------
当 RSIMP=1（离线驱动默认值）时 ZCONS2=0，因此求解器就是一个
直接的全隐式步，且逐贴块的 ZSURFL 退化为四贴块的面积加权平均。
此处雪底热通量 ``PGSN`` 为零（M6 雪方案尚未接入），雪蒸发不匹配量
``PEMSSN`` 为零，与 M5 设计文档中记录的冷启动假设一致。

符号约定
---------------
``ZSURFL`` 以*向下*（进入土壤）为正，与
:mod:`surf_pytorch.physics.seb` 中记录的 SURFSEB 地面热通量约定一致。
来自 ``VDFDIFH1S`` 的感热和潜热通量遵循相同的约定；潜热贡献为
``RLVTT*PEVAP``（蒸发为正，以汽化潜热的形式从土壤中带走能量）。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Optional

import torch

from ..config import (
    TILE_WET_SKIN,
    TILE_LOW_VEG_DRY,
    TILE_HIGH_VEG_DRY,
    TILE_BARE_SOIL,
    SurfConfig,
)
from ..constants import ConstantBundle
from ..tables.soil import SoilTable


# --------------------------------------------------------------------------- #
# 硬编码常数（yoevdf / yos_soil）
# --------------------------------------------------------------------------- #
# RRCSICE -- 海冰的体积热容（sussoil 第 467 行）。
# Surf 点（LDLAND=.F., LDSICE=.F.）的 PCTSA=0，因此它们永远不会进入
# 温度求解；这里通过将整个廓线置零来镜像该行为。
_RRCSICE = 1.884e6

# RGH2O -- 液态水的体积热容（sussoil 第 135 行），
# 由 SRFRCG 在海冰分支中使用（ZGICE = 0.5*RGH2O）。Surf 点
# 无论如何都会被置零，但为保持一致性此处保留该常数。
_RGH2O = 4.18e6

_SRFT_DEBUG_STEP = 0
_SRFW_DEBUG_STEP = 0


# --------------------------------------------------------------------------- #
# 土壤温度分支（SRFRCGS + SRFTS + SRFWDIFS）
# --------------------------------------------------------------------------- #
@dataclass
class SoilTempInputs:
    """HTESSEL 土壤温度求解的全部输入。

    逐（格点，层）场的张量为 ``(Ngrid, Nsoil)``，逐格点场为
    ``(Ngrid,)``，逐贴块通量场为 ``(Ngrid, Ntile)``。
    命名与 ``SRFTS`` 的 Fortran 哑元列表一致，便于并排审查。

    冷启动说明：``pgsn``（PGSN，从雪层到土壤的地面热通量）在 M6
    单层雪预报量接入之前为零；传入非零值是无害的，且向前兼容。
    """

    # ---- t-1 时刻的土壤状态 ----
    ptsam1m: torch.Tensor            # PTSAM1M  土壤温度          [K]      (Ngrid, Nsoil)
    pwsam1m: torch.Tensor            # PWSAM1M  土壤湿度          [m3/m3]  (Ngrid, Nsoil)

    # ---- 逐贴块通量（来自 VDFDIFH1S / SURFSEB） ----
    pfrti: torch.Tensor              # PFRTI    贴块面积比        [-]      (Ngrid, Ntile)
    pahfsti: torch.Tensor            # PAHFSTI  贴块感热通量      [W/m^2]  (Ngrid, Ntile)
    pevapti: torch.Tensor            # PEVAPTI  贴块水汽通量      [kg/m^2/s] (Ngrid, Ntile)
    pssrflti: torch.Tensor           # PSSRFLTI 贴块净短波        [W/m^2]  (Ngrid, Ntile)

    # ---- 逐格点辐射 ----
    pslrfl: torch.Tensor             # PSLRFL   净长波            [W/m^2]  (Ngrid,)

    # ---- 雪底热通量（M6 向前兼容；冷启动时为零） ----
    pgsn: torch.Tensor               # PGSN     来自雪的地面热通量 [W/m^2] (Ngrid,)

    # ---- 静态描述量 ----
    ksoty: torch.Tensor              # KSOTY    土壤质地类型      [long]   (Ngrid,)  0..7
    pcvl: torch.Tensor               # PCVL     低植被覆盖度      [-]      (Ngrid,)
    pcvh: torch.Tensor               # PCVH     高植被覆盖度      [-]      (Ngrid,)

    # ---- 掩膜 ----
    ldland: torch.Tensor             # LDLAND   陆地掩膜（bool）            (Ngrid,)

    # ---- 时间步长 ----
    ptmst: torch.Tensor              # PTMST    时间步长          [s]     标量

    # SRFT 使用逐贴块的净长波（PSLRFLTI）。保留上面的 pslrfl 以兼容
    # 旧式 SRFTS 风格的调用方和测试；当该字段存在时，它在地表热通量
    # 组装中优先使用。
    pslrflti: Optional[torch.Tensor] = None  # (Ngrid, Ntile) [W/m^2]


@dataclass
class SoilTempOutputs:
    """HTESSEL 土壤温度求解的全部输出。"""

    ptsa: torch.Tensor               # PTSA  新的土壤温度        [K]     (Ngrid, Nsoil)
    pctsa: torch.Tensor              # PCTSA 表观热容            [J/K/m^3] (Ngrid, Nsoil)

    # 进入土壤顶层的地表净热通量（向下为正）。
    # 作为诊断量输出，以便无需重新推导即可从 model.forward()
    # 进行能量平衡闭合检查。
    zsurfl: torch.Tensor             # ZSURFL 地表净热通量       [W/m^2] (Ngrid,)


def compute_soil_temperature(
    inp: SoilTempInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    differentiable: Optional[bool] = None,
) -> SoilTempOutputs:
    """运行 HTESSEL 土壤温度求解（SRFRCGS + SRFTS + SRFWDIFS）。

    Args
    ----
    inp:
        :class:`SoilTempInputs` -- 打包的土壤状态、通量与掩膜。
    soil_table:
        :class:`surf_pytorch.tables.soil.SoilTable` -- 逐类别查找表
        （wsat、lambdadry、lamsat1、rcsoil）以及与类别无关的热力
        标量（rsimp、rlambdace、rlambdawat、rkerst1/2/3）。
    consts:
        :class:`ConstantBundle` -- 提供 ``rtt``、``rlmlt``、``rlvtt``
        以及冻结过渡区端点 ``rtf1..rtf4``。
    config:
        :class:`SurfConfig` -- ``n_soil`` 选定土壤层数，
        ``soil_thickness`` 提供 ``RDAT``。
    differentiable:
        为本次调用覆盖 ``config.differentiable``。

    Returns
    -------
    :class:`SoilTempOutputs` -- 新温度、表观热容以及地表净热通量诊断量。
    """
    diff = config.differentiable if differentiable is None else differentiable

    def _core() -> SoilTempOutputs:
        return _soil_temperature_core(inp, soil_table, consts, config)

    if not diff:
        with torch.no_grad():
            return _core()
    return _core()


def _soil_temperature_core(
    inp: SoilTempInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
    config: SurfConfig,
) -> SoilTempOutputs:
    """内部：假定 no_grad / enable_grad 的决定已经做出。"""
    ngrid = inp.ptsam1m.shape[0]
    nlevs = config.n_soil
    dtype = inp.ptsam1m.dtype
    device = inp.ptsam1m.device

    # 标量常数（按与 Fortran 相同的顺序求值，使复合表达式
    # 到最后一个 ULP 都一致）。
    rlmlt = consts.rlmlt
    rlvtt = consts.rlvtt
    rtt = consts.rtt
    rtf1 = consts.rtf1
    rtf2 = consts.rtf2
    rtf3 = consts.rtf3
    rtf4 = consts.rtf4
    rho_h2o = float(soil_table.consts.rho_h2o)

    rsimp = soil_table.consts.rsimp
    ptmst = inp.ptmst

    # 按土壤类型索引的逐类别查找表列（0..7；索引 0 是水体哨兵值，
    # 全部为零——gather 在那里返回 0，这是正确的，因为 surf 点
    # 在下方会被掩膜掉）。
    ksoty = inp.ksoty.long().detach()
    wsat = soil_table.gather("wsat", ksoty)             # (Ngrid,)
    lambdadry = soil_table.gather("lambdadry", ksoty)   # (Ngrid,)
    lamsat1 = soil_table.gather("lamsat1", ksoty)       # (Ngrid,)
    rcsoil = soil_table.gather("rcsoil", ksoty)         # (Ngrid,)
    wcap = soil_table.gather("wcap", ksoty)             # (Ngrid,)

    # 土壤层厚度 RDAT（sussoil 第 178-181 行）。广播到
    # (Ngrid, Nsoil)，使逐层表达式可以向量化；温度求解
    # 将这些作为普通的逐层标量使用。
    rdat = _soil_thickness(config, dtype, device)       # (Nsoil,)
    rdat_b = rdat.expand(ngrid, nlevs)                  # (Ngrid, Nsoil)

    land = inp.ldland.to(dtype)                         # (Ngrid,)  1 / 0

    # ------------------------------------------------------------------ #
    # 步骤 1：SRFRCGS -- 表观体积热容。
    # ------------------------------------------------------------------ #
    pctsa = _apparent_heat_capacity(
        inp.ptsam1m, wcap, rcsoil, inp.pcvl, inp.pcvh,
        land, rlmlt, rho_h2o, rtf1, rtf2, rtf3, rtf4,
        nlevs, dtype, device,
    )

    # ------------------------------------------------------------------ #
    # 步骤 2：SRFTS 第 2 节 -- 地表净热通量 ZSURFL。
    # ------------------------------------------------------------------ #
    zsurfl = _surface_net_heat_flux(
        inp.pfrti, inp.pssrflti, inp.pslrfl,
        inp.pahfsti, inp.pevapti, inp.pgsn,
        land, rlvtt, dtype, device, inp.pslrflti,
    )

    # ------------------------------------------------------------------ #
    # 步骤 3：SRFT 第 3 节 -- 逐层热导率 ZDIF。
    # 离线 namelist 默认使用 LESOILCOND=.TRUE.；保留旧的 ZFF=0 路径，
    # 供独立的 SURF_REF_SOILT 测试辅助程序使用。
    # ------------------------------------------------------------------ #
    zdif = _thermal_conductivity(
        inp.pwsam1m, inp.ptsam1m, wsat, lambdadry, lamsat1,
        land, soil_table.consts,
        rtf1, rtf2, rtf3, rtf4,
        nlevs, dtype, device,
        lesoilcond=getattr(config, "lesoilcond", True),
    )

    # ------------------------------------------------------------------ #
    # 步骤 4：SRFTS 第 4 节 -- Thomas 系数组装。
    # ------------------------------------------------------------------ #
    zcons1 = ptmst * rsimp
    # ZLST 是半层热耦合；Fortran 将其索引为 1..KLEVS，
    # 但只使用 1..KLEVS-1（ZLST(:,KLEVS)=0 强制零下边界条件）。
    zlst = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)
    zcdz = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)
    zrhs = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)

    # 层 1..KLEVS-1 使用层间耦合
    #   ZLST(JK) = ZCONS1*(ZDIF(JK)+ZDIF(JK+1))/(RDAT(JK)+RDAT(JK+1))。
    # 底层 ZLST(:,KLEVS) 保持为零（零通量下边界条件）。
    if nlevs > 1:
        zdif_lo = zdif[:, :-1]                          # ZDIF(:,1..KLEVS-1)
        zdif_hi = zdif[:, 1:]                           # ZDIF(:,2..KLEVS)
        rdat_lo = rdat_b[:, :-1]
        rdat_hi = rdat_b[:, 1:]
        zlst[:, :-1] = zcons1 * (zdif_lo + zdif_hi) / (rdat_lo + rdat_hi)

    zcdz = pctsa * rdat_b                              # ZCDZ = PCTSA*RDAT

    # 地表右端项将地表净通量注入顶层。
    # ZRHS(:,1) = PTMST*ZSURFL/ZCDZ(:,1)。保护 ZCDZ(:,1) > 0；surf
    # 点的 ZCDZ=0 且会被掩膜掉，但安全除法可保持 autograd 图
    # 处处有限。
    zcdz1_safe = torch.where(
        zcdz[:, 0] != 0.0, zcdz[:, 0], torch.ones_like(zcdz[:, 0])
    )
    zrhs[:, 0] = ptmst * zsurfl / zcdz1_safe
    zrhs[:, 0] = torch.where(zcdz[:, 0] != 0.0, zrhs[:, 0], torch.zeros_like(zrhs[:, 0]))

    # ------------------------------------------------------------------ #
    # 步骤 5：SRFWDIFS -- Thomas 三对角求解。
    # ------------------------------------------------------------------ #
    ztsa = _solve_tridiag(inp.ptsam1m, zlst, zrhs, zcdz, land, rsimp)

    _write_srft_debug(
        inp.ptsam1m, inp.pwsam1m, zsurfl, inp.pgsn, zdif, zlst, zcdz, zrhs, ztsa,
        inp.ptsam1m * (1.0 - 1.0 / rsimp) + ztsa,
        inp.pfrti, inp.pssrflti, inp.pslrfl, inp.pslrflti,
        inp.pahfsti, inp.pevapti, rlvtt,
    )

    # ------------------------------------------------------------------ #
    # 步骤 6：SRFTS 第 7 节 -- 新温度。
    #   PTSA = PTSAM1M*ZCONS2 + ZTSA,  ZCONS2 = 1 - 1/RSIMP。
    # Surf 点保留其 t-1 时刻的温度（LDLAND=.F. 分支）。
    # ------------------------------------------------------------------ #
    zcons2 = 1.0 - 1.0 / rsimp
    ptsa_land = inp.ptsam1m * zcons2 + ztsa
    ptsa = torch.where(
        land[:, None].to(dtype) != 0.0, ptsa_land, inp.ptsam1m
    )

    return SoilTempOutputs(ptsa=ptsa, pctsa=pctsa, zsurfl=zsurfl)


def _write_srft_debug(
    ptsam1m: torch.Tensor,
    pwsam1m: torch.Tensor,
    zsurfl: torch.Tensor,
    pgsn: torch.Tensor,
    zdif: torch.Tensor,
    zlst: torch.Tensor,
    zcdz: torch.Tensor,
    zrhs: torch.Tensor,
    ztsa: torch.Tensor,
    ptsa_land: torch.Tensor,
    pfrti: torch.Tensor,
    pssrflti: torch.Tensor,
    pslrfl: torch.Tensor,
    pslrflti: Optional[torch.Tensor],
    pahfsti: torch.Tensor,
    pevapti: torch.Tensor,
    rlvtt: float,
) -> None:
    """可选的 CSV 诊断，与 Fortran SRFT 调试列对应。"""
    global _SRFT_DEBUG_STEP
    path = os.environ.get("SURF_TORCH_SRFT_DEBUG_CSV")
    if not path:
        return

    ngrid, nlevs = ptsam1m.shape
    if nlevs != 4:
        return

    column = os.environ.get("SURF_TORCH_SRFT_DEBUG_COLUMN")
    if column:
        try:
            selected = int(column) - 1
        except ValueError as exc:
            raise ValueError(
                "SURF_TORCH_SRFT_DEBUG_COLUMN must be a 1-based integer"
            ) from exc
        if not 0 <= selected < ngrid:
            raise ValueError(
                "SURF_TORCH_SRFT_DEBUG_COLUMN is outside the current domain"
            )
        indices = (selected,)
    else:
        indices = range(ngrid)

    write_header = not os.path.exists(path)
    rows = []
    f3 = pfrti[:, TILE_WET_SKIN]
    f4 = pfrti[:, TILE_LOW_VEG_DRY]
    f6 = pfrti[:, TILE_HIGH_VEG_DRY]
    f8 = pfrti[:, TILE_BARE_SOIL]
    zssrfl = (
        f3 * pssrflti[:, TILE_WET_SKIN]
        + f4 * pssrflti[:, TILE_LOW_VEG_DRY]
        + f6 * pssrflti[:, TILE_HIGH_VEG_DRY]
        + f8 * pssrflti[:, TILE_BARE_SOIL]
    )
    if pslrflti is None:
        zslrfl = (f3 + f4 + f6 + f8) * pslrfl
    else:
        zslrfl = (
            f3 * pslrflti[:, TILE_WET_SKIN]
            + f4 * pslrflti[:, TILE_LOW_VEG_DRY]
            + f6 * pslrflti[:, TILE_HIGH_VEG_DRY]
            + f8 * pslrflti[:, TILE_BARE_SOIL]
        )
    zthfl = (
        f3 * (pahfsti[:, TILE_WET_SKIN] + rlvtt * pevapti[:, TILE_WET_SKIN])
        + f4 * (pahfsti[:, TILE_LOW_VEG_DRY] + rlvtt * pevapti[:, TILE_LOW_VEG_DRY])
        + f6 * (pahfsti[:, TILE_HIGH_VEG_DRY] + rlvtt * pevapti[:, TILE_HIGH_VEG_DRY])
        + f8 * (pahfsti[:, TILE_BARE_SOIL] + rlvtt * pevapti[:, TILE_BARE_SOIL])
    )

    arrays = [
        ptsam1m.detach().cpu(),
        pwsam1m.detach().cpu(),
        zsurfl.detach().cpu(),
        pgsn.detach().cpu(),
        zssrfl.detach().cpu(),
        zslrfl.detach().cpu(),
        zthfl.detach().cpu(),
        zdif.detach().cpu(),
        zlst.detach().cpu(),
        zcdz.detach().cpu(),
        zrhs.detach().cpu(),
        ztsa.detach().cpu(),
        ptsa_land.detach().cpu(),
    ]
    (
        ptsam1m_c, pwsam1m_c, zsurfl_c, pgsn_c, zssrfl_c, zslrfl_c,
        zthfl_c, zdif_c, zlst_c, zcdz_c, zrhs_c, ztsa_c, ptsa_c,
    ) = arrays
    for jl in indices:
        rows.append([
            _SRFT_DEBUG_STEP, jl + 1,
            *[float(ptsam1m_c[jl, k]) for k in range(4)],
            *[float(pwsam1m_c[jl, k]) for k in range(4)],
            float(zsurfl_c[jl]), float(pgsn_c[jl]),
            float(zssrfl_c[jl]), float(zslrfl_c[jl]), float(zthfl_c[jl]),
            *[float(zdif_c[jl, k]) for k in range(4)],
            *[float(zlst_c[jl, k]) for k in range(4)],
            *[float(zcdz_c[jl, k]) for k in range(4)],
            *[float(zrhs_c[jl, k]) for k in range(4)],
            *[float(ztsa_c[jl, k]) for k in range(4)],
            *[float(ptsa_c[jl, k]) for k in range(4)],
        ])

    fieldnames = [
        "step", "jl",
        "ptsa_in1", "ptsa_in2", "ptsa_in3", "ptsa_in4",
        "pwsa_in1", "pwsa_in2", "pwsa_in3", "pwsa_in4",
        "zsurfl", "pgsn", "zssrfl", "zslrfl", "zthfl",
        "zdif1", "zdif2", "zdif3", "zdif4",
        "zlst1", "zlst2", "zlst3", "zlst4",
        "zcdz1", "zcdz2", "zcdz3", "zcdz4",
        "zrhs1", "zrhs2", "zrhs3", "zrhs4",
        "ztsa1", "ztsa2", "ztsa3", "ztsa4",
        "ptsa1", "ptsa2", "ptsa3", "ptsa4",
    ]
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(fieldnames)
        writer.writerows(rows)
    _SRFT_DEBUG_STEP += 1


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #
def _soil_thickness(
    config: SurfConfig, dtype: torch.dtype, device,
) -> torch.Tensor:
    """以 1 维张量返回 RDAT（土壤层厚度，m）。

    对应 sussoil 第 178-181 行：4 层默认为 (0.07, 0.21, 0.72, 1.89)。
    SurfConfig.soil_thickness 是唯一权威来源，因此采用非默认层数的
    配置仍可正常工作。
    """
    return torch.as_tensor(
        config.soil_thickness, dtype=dtype, device=device
    )[: config.n_soil]


def _apparent_heat_capacity(
    ptsam1m: torch.Tensor,
    wcap: torch.Tensor,        # (Ngrid,) 逐类别田间持水量
    rcsoil: torch.Tensor,      # (Ngrid,) 逐类别干土热容
    pcvl: torch.Tensor,        # (Ngrid,)
    pcvh: torch.Tensor,        # (Ngrid,)
    land: torch.Tensor,        # (Ngrid,) 1/0
    rlmlt: float,
    rho_h2o: float,
    rtf1: float, rtf2: float, rtf3: float, rtf4: float,
    nlevs: int,
    dtype: torch.dtype, device,
) -> torch.Tensor:
    """SRFRCGS -- 表观体积热容（J/K/m^3）。

    复现 srfrcgs_mod.F90 第 132-188 行。对于陆地点::

        ZDFDT = -0.5*RTF4*COS(RTF4*(T-RTF3))   if RTF2 < T < RTF1 else 0
        ZWA   = (PCVL + PCVH) * RWCAPM(JS)
        PCTSA = RRCSOILM(JS) - RLMLT*RHOH2O*ZWA*ZDFDT

    Surf 点（LDLAND=.F.）被置零；海冰点在顶层使用 RRCSICE、下层为零
    （此处不涉及——离线驱动通过 SRFI 运行海冰分支，本移植将海冰
    经由单独路径处理）。
    """
    t = ptsam1m
    in_freeze = (t > rtf2) & (t < rtf1)
    zdfdt = -0.5 * rtf4 * torch.cos(rtf4 * (t - rtf3))
    zdfdt = torch.where(in_freeze, zdfdt, torch.zeros_like(zdfdt))

    # ZWA = (PCVL + PCVH) * RWCAPM(JS) -- 逐类别田间持水量乘以
    # 植被覆盖度。广播 (Ngrid,) -> (Ngrid, Nsoil)。
    zwa = (pcvl + pcvh) * wcap                       # (Ngrid,)
    zwa_b = zwa[:, None].expand(-1, nlevs)

    pctsa = rcsoil[:, None] - rlmlt * rho_h2o * zwa_b * zdfdt   # (Ngrid, Nsoil)

    # Surf 点：整个廓线置零（LDLAND=.F., LDSICE=.F. 分支）。
    land_b = land[:, None].to(dtype)
    pctsa = torch.where(land_b != 0.0, pctsa, torch.zeros_like(pctsa))
    return pctsa


def _surface_net_heat_flux(
    pfrti: torch.Tensor,
    pssrflti: torch.Tensor,
    pslrfl: torch.Tensor,
    pahfsti: torch.Tensor,
    pevapti: torch.Tensor,
    pgsn: torch.Tensor,
    land: torch.Tensor,
    rlvtt: float,
    dtype: torch.dtype, device,
    pslrflti: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SRFTS 第 2 节 -- 地表净热通量 ZSURFL（向下为正）。

    复现 srfts_mod.F90 第 183-218 行。四个面向土壤的贴块是
    湿皮肤（3）、低植被-干（4）、高植被-干（6）、裸土（8）。
    它们的贴块面积比在陆地上之和约为 1（其余是水体/冰/雪贴块），
    因此面积加权平均就是贴块加权求和；雪底热通量 ``PGSN``
    作为格点平均项加入。

    湖泊特例（SRFTS 第 205-213 行，LEFLAKE 分支）此处未转写：
    离线驱动中 LEFLAKE=.FALSE.，且 M5 移植将湖泊贴块经由单独
    路径处理。为雪传入非零 PGSN 是 M6 向前兼容的。
    """
    f3 = pfrti[:, TILE_WET_SKIN]
    f4 = pfrti[:, TILE_LOW_VEG_DRY]
    f6 = pfrti[:, TILE_HIGH_VEG_DRY]
    f8 = pfrti[:, TILE_BARE_SOIL]

    zssrfl = (
        f3 * pssrflti[:, TILE_WET_SKIN]
        + f4 * pssrflti[:, TILE_LOW_VEG_DRY]
        + f6 * pssrflti[:, TILE_HIGH_VEG_DRY]
        + f8 * pssrflti[:, TILE_BARE_SOIL]
    )
    if pslrflti is None:
        zslrfl = (f3 + f4 + f6 + f8) * pslrfl
    else:
        zslrfl = (
            f3 * pslrflti[:, TILE_WET_SKIN]
            + f4 * pslrflti[:, TILE_LOW_VEG_DRY]
            + f6 * pslrflti[:, TILE_HIGH_VEG_DRY]
            + f8 * pslrflti[:, TILE_BARE_SOIL]
        )
    zthfl = (
        f3 * (pahfsti[:, TILE_WET_SKIN] + rlvtt * pevapti[:, TILE_WET_SKIN])
        + f4 * (pahfsti[:, TILE_LOW_VEG_DRY] + rlvtt * pevapti[:, TILE_LOW_VEG_DRY])
        + f6 * (pahfsti[:, TILE_HIGH_VEG_DRY] + rlvtt * pevapti[:, TILE_HIGH_VEG_DRY])
        + f8 * (pahfsti[:, TILE_BARE_SOIL] + rlvtt * pevapti[:, TILE_BARE_SOIL])
    )

    zsurfl = zssrfl + zslrfl + zthfl + pgsn

    # Surf 点：ZSURFL=0（SRFTS 第 216 行 ELSE 分支）。此处掩膜，
    # 使诊断量只携带陆地值；三对角右端项使用该掩膜后的场。
    land_mask = land.to(dtype)
    zsurfl = torch.where(land_mask != 0.0, zsurfl, torch.zeros_like(zsurfl))
    return zsurfl


def _thermal_conductivity(
    pwsam1m: torch.Tensor,
    ptsam1m: torch.Tensor,
    wsat: torch.Tensor,        # (Ngrid,)
    lambdadry: torch.Tensor,   # (Ngrid,)
    lamsat1: torch.Tensor,     # (Ngrid,)
    land: torch.Tensor,
    soil_consts,               # SoilConstants 数据类
    rtf1: float, rtf2: float, rtf3: float, rtf4: float,
    nlevs: int,
    dtype: torch.dtype, device,
    *,
    lesoilcond: bool = True,
) -> torch.Tensor:
    """SRFT 第 3 节 -- 逐层土壤热导率 ZDIF。

    默认路径对应 ``srft_mod.F90`` 在 ``LESOILCOND=.TRUE.`` 时的情形，
    并使用 ``fcsurf.h`` 中的 ``FSOILTCOND`` 语句函数::

        ZFF        = 依赖于温度的冻结比例
        ZWU        = WSAT * (1 - ZFF)
        ZLWT       = RLAMBDAWAT ** ZWU
        ZLIC       = RLAMBDAICE ** (WSAT * ZFF)
        ZLAMBDASAT = RLAMSAT1M * ZLIC * ZLWT
        ZCOND      = MAX(RKERST1, W / WSAT)
        ZKERSTEN   = RKERST2 * LOG10(ZCOND) + RKERST3
        ZDIF       = RLAMBDADRYM + ZKERSTEN * (ZLAMBDASAT - RLAMBDADRYM)

    ``lesoilcond=False`` 选择独立的 ``SURF_REF_SOILT`` 辅助分支：
    ``ZFF=0`` 且 ``ZWU=W``。

    在 (Ngrid, Nsoil) 上向量化；逐类别列沿土壤维广播。
    Surf 点保持为零（LDLAND=.F. 分支）。
    """
    w = pwsam1m
    t = ptsam1m
    wsat_b = wsat[:, None].expand(-1, nlevs)
    lamsat1_b = lamsat1[:, None].expand(-1, nlevs)
    lambdadry_b = lambdadry[:, None].expand(-1, nlevs)

    rlambdace = float(soil_consts.rlambdace)      # RLAMBDAICE
    rlambdawat = float(soil_consts.rlambdawat)
    rkerst1 = float(soil_consts.rkerst1)
    rkerst2 = float(soil_consts.rkerst2)
    rkerst3 = float(soil_consts.rkerst3)

    if lesoilcond:
        zff_ramp = 0.5 * (1.0 - torch.sin(rtf4 * (t - rtf3)))
        zff = torch.where(
            t <= rtf2,
            torch.ones_like(t),
            torch.where(t < rtf1, zff_ramp, torch.zeros_like(t)),
        )
        # SRFT 最终会用 fcsurf.h 中的 FSOILTCOND 语句函数
        # 覆盖其展开的热导率。逐字匹配该函数，
        # 包括除法和指数分组方式。
        zwu = wsat_b * (1.0 - zff)
        zlic_exp = wsat_b * zff
    else:
        zwu = w
        zlic_exp = wsat_b - zwu

    zlwt = rlambdawat ** zwu
    zlic = rlambdace ** zlic_exp
    zlambdasat = lamsat1_b * zlic * zlwt

    zcond = torch.maximum(torch.full_like(w, rkerst1), w / wsat_b)
    zkersten = rkerst2 * torch.log10(zcond) + rkerst3

    zdif = lambdadry_b + zkersten * (zlambdasat - lambdadry_b)

    land_b = land[:, None].to(dtype)
    zdif = torch.where(land_b != 0.0, zdif, torch.zeros_like(zdif))
    return zdif


def _solve_tridiag(
    pwsam1m: torch.Tensor,    # 此处实际是 PTSAM1M（通用求解器）
    pcfw: torch.Tensor,       # ZLST（lambda-star，半层耦合）
    prhsw: torch.Tensor,      # ZRHS（右端项）
    pcdz: torch.Tensor,       # ZCDZ = PCTSA*RDAT
    land: torch.Tensor,
    rsimp: float,
) -> torch.Tensor:
    """SRFWDIFS -- Thomas 三对角求解器，在 Ngrid 上批量处理。

    尽管带有 ``W`` 前缀，这是一个通用求解器，同时用于土壤温度
    （由 SRFTS 调用）和土壤水分（由 M5b 路径调用）。两者的语义
    完全一致；变量名与 Fortran 哑元列表一致，便于并排审查。

    算法（srfwdifs_mod.F90 第 117-176 行，LDALLAYS=.TRUE.）::

        ZTPFAC2       = 1/RSIMP
        PWSADIF(:,JK) = ZTPFAC2*PWSAM1M(:,JK) + PRHSW(:,JK)

        # 顶层消元（JK=1）
        ZTCOE       = PCFW(:,1)
        ZQDP        = 1/PCDZ(:,1)
        ZDISC       = 1/(1 + PCFW(:,1)*ZQDP)
        ZEBSW(:,1)  = ZDISC*(PCFW(:,1)*ZQDP)
        PWSADIF(:,1)= ZDISC*PWSADIF(:,1)

        # 中间层 JK=2..KLEVS-1
        ZFAC        = ZTCOE*ZQDP(JK)
        ZTCOE       = PCFW(:,JK)
        ZDISC       = 1/(1 + ZFAC*(1-ZEBSW(:,JK-1)) + PCFW(:,JK)*ZQDP(JK))
        ZEBSW(:,JK) = ZDISC*(PCFW(:,JK)*ZQDP(JK))
        PWSADIF(:,JK)= ZDISC*(PWSADIF(:,JK) + ZFAC*PWSADIF(:,JK-1))

        # 底层 JK=KLEVS
        ZFAC        = ZTCOE*ZQDP(KLEVS)
        ZDISC       = 1/(1 + ZFAC*(1-ZEBSW(:,KLEVS-1)) + PCFW(:,KLEVS)*ZQDP(KLEVS))
        PWSADIF(:,KLEVS) = ZDISC*(PWSADIF(:,KLEVS) + ZFAC*PWSADIF(:,KLEVS-1))

        # 回代 JK=KLEVS-1..1
        PWSADIF(:,JK) += ZEBSW(:,JK)*PWSADIF(:,JK+1)

    中间层/底层的扫描携带一个*列递推*（ZTCOE 馈入下一层的 ZFAC），
    因此无法沿层维完全向量化；我们将其展开为对 JK 的 Python 循环。
    当 nlevs=4 时只有三次迭代——代价微不足道，且精确镜像 Fortran
    的控制流。

    实现说明：扫描以*函数式*方式书写（逐层解张量累积在 Python
    列表中，最后通过 :func:`torch.stack` 合并），而不是使用原地
    索引赋值。原地写法更贴近 Fortran 的字面形式，但会破坏 autograd
    （对计算图叶节点的原地写入会使版本计数器失效）。函数式写法在
    前向传播中逐位一致，并且保持可微，可用于 ML 训练。

    陆地掩膜：surf 点永远不会被更新（Fortran 用
    ``IF (LDLAND(JL))`` 守护每一次赋值）。我们在扫描中以乘性门控
    携带陆地掩膜，使非陆地列保持其初始值（零），与 Fortran 的
    OUT 数组一致。
    """
    ngrid, nlevs = pwsam1m.shape
    dtype = pwsam1m.dtype
    device = pwsam1m.device

    ztpfac2 = 1.0 / rsimp

    # 1.1 右端项设置。
    pwsadif = ztpfac2 * pwsam1m + prhsw

    # 将非陆地列门控为零，使其对递推不产生任何贡献（Fortran 完全
    # 跳过这些列；我们将其置零，并在最后通过掩膜更新恢复原值）。
    land_b = land[:, None].to(dtype)
    pwsadif = pwsadif * land_b

    # 逐层消元系数和滚动解。两者均以函数式方式累积（无原地写入），
    # 以便 autograd 能沿扫描反向传播。
    ebsw: list[torch.Tensor] = [torch.empty(0)] * nlevs
    sol: list[torch.Tensor] = [torch.empty(0)] * nlevs

    # 1.2 顶层消元（JK=1）。
    ztcoe = pcfw[:, 0]                              # PCFW(:,1)；视图，从不被修改
    zqdp = 1.0 / pcdz[:, 0]
    zdisc = 1.0 / (1.0 + pcfw[:, 0] * zqdp)
    ebsw[0] = zdisc * (pcfw[:, 0] * zqdp)
    sol[0] = zdisc * pwsadif[:, 0]

    # 1.3 中间层 JK=2..KLEVS-1（Python 循环，严格的 Fortran 顺序）。
    for jk in range(1, nlevs - 1):
        zqdp = 1.0 / pcdz[:, jk]
        zfac = ztcoe * zqdp                         # ZTCOE = PCFW(:,JK-1)
        ztcoe = pcfw[:, jk]                         # 为下一次迭代推进
        zdisc = 1.0 / (
            1.0 + zfac * (1.0 - ebsw[jk - 1]) + pcfw[:, jk] * zqdp
        )
        ebsw[jk] = zdisc * (pcfw[:, jk] * zqdp)
        sol[jk] = zdisc * (pwsadif[:, jk] + zfac * sol[jk - 1])

    # 1.4 底层消元（JK=KLEVS）。保护 nlevs==1 的情形（单层
    # 配置跳过中间层循环，复用 JK=1 的 ZTCOE）。
    if nlevs >= 2:
        jk = nlevs - 1
        zqdp = 1.0 / pcdz[:, jk]
        zfac = ztcoe * zqdp
        zdisc = 1.0 / (
            1.0 + zfac * (1.0 - ebsw[jk - 1]) + pcfw[:, jk] * zqdp
        )
        sol[jk] = zdisc * (pwsadif[:, jk] + zfac * sol[jk - 1])

    # 1.5 回代 JK=KLEVS-1..1。每层构建一个全新的张量，
    # 使计算图保持干净（sol[jk+1] 仅以只读方式被消费）。
    for jk in range(nlevs - 2, -1, -1):
        sol[jk] = sol[jk] + ebsw[jk] * sol[jk + 1]

    return torch.stack(sol, dim=1)


# =========================================================================== #
# 土壤水分分支（SRFWEXC_VG + SRFWDIF + SRFWINC + SRFWNG）
# =========================================================================== #
"""采用 Van Genuchten 水力学的 HTESSEL 土壤水分更新。

这是 ``SURFTSTP_CTL`` 土壤水分分支的 Python 移植
（``surf/module/surftstp_ctl_mod.F90`` 第 4 节），由四个上游例程组成：

* ``SRFWEXC_VG`` (``surf/module/srfwexc_vg_mod.F90``) -- 组装水分求解的
  三对角系数：Van Genuchten 水力传导度 K(W) 和扩散率 D(W)、
  VIC 饱和超渗产流*或*入渗限制产流（由 LESSRO 门控）、
  来自四个面向土壤贴块的根系吸水，以及地表强迫
  （穿透雨 + 融雪 - 产流 + 蒸发）。

* ``SRFWDIF`` (``surf/module/srfwdif_mod.F90``) -- 与温度路径相同的
  Thomas 求解器（即上面的通用 :func:`_solve_tridiag`）。尽管带有
  ``W`` 前缀，两条路径共享同一算法。

* ``SRFWINC`` (``surf/module/srfwinc_mod.F90``) -- 将 Thomas 解
  ``PWSADIF`` 转换为未滤波的 t+1 时刻湿度
  ``PWSA = PWSADIF - ZTPFAC3*PWSAM1M``，并将重力通量
  重新施加到排水诊断量上。

* ``SRFWNG`` (``surf/module/srfwng_mod.F90``) -- 负湿度截断并向下
  再分配。层 JK 中多余的水被推入 JK+1（列递推）；饱和超出量
  成为地表径流（顶层）或地下径流（深层）。

Van Genuchten 水力学
------------------------
设有效饱和度 ``Se = (W - Wres) / (Wsat - Wres)``::

    K(Se) = WCONS * Se**lam * (1 - (1 - Se**(1/m))**m)**2
    D(Se) = ((1-m)*WCONS / (alpha*m*WFAC)) * Se**(lam-1/m)
              * ( (1 - Se**(1/m))**m + (1 - Se**(1/m))**(-m) - 2 )

其中 ``m = 1 - 1/n``，``WFAC = Wsat - Wres``。在气干下限
``W <= 1.001*Wres`` 处，函数钳制到 ``K=0, D=Dmin``；接近饱和
``W >= 0.999*Wsat`` 时钳制到 ``K=WCONS, D=Dmax``。冻结函数
``ZF = 0.5*(1-SIN(RTF4*(T-RTF3)))`` 进一步削减两者（K 被压到零，
D 向 Dmin 线性插值），因此冻结土壤既不明显排水也不明显扩散。

冷启动简化
--------------------------
* 雪输入（PMSN 融雪、PEMSSN 蒸发不匹配、PEVAPSNW 林下积雪蒸发）
  在冷启动时全部为零。
* 截留层蒸发不匹配量 PEINTTI 为零（在 M6 接入皮肤层水库之前，
  离线驱动不调用 SRFWL）。
* 跳过湖泊/城市分支（KTILES > 9）——离线驱动运行 8 贴块基础配置。

径流符号约定
-----------------------
``PROS``（地表径流）和 ``PROD``（地下径流）以*流出*土柱为正，
单位为 kg/m^2（在时间步内累积）。底层的排水由调用方加入
``PROD``（SURFTSTP 第 4.6 节）。
"""


@dataclass
class SoilMoistInputs:
    """HTESSEL 土壤水分求解的全部输入。

    逐（格点，层）场的张量为 ``(Ngrid, Nsoil)``，逐格点场为
    ``(Ngrid,)``，逐贴块通量场为 ``(Ngrid, Ntile)``。
    命名与 ``SRFWEXC_VG`` 的 Fortran 哑元列表一致。

    冷启动说明：雪/截留不匹配输入（``pmsn``、``pemssn``、
    ``pevapsnw``、``peintti``）全部默认为零，在 M6 雪预报量
    填充它们之前可以保持为零。
    """

    # ---- Soil state at t-1 ----
    pwsam1m: torch.Tensor            # PWSAM1M  soil moisture     [m3/m3]  (Ngrid, Nsoil)
    ptsam1m: torch.Tensor            # PTSAM1M  soil temperature  [K]      (Ngrid, Nsoil)

    # ---- Per-tile fluxes (from VDFDIFH1S / SURFSEB) ----
    pfrti: torch.Tensor              # PFRTI    tile fractions    [-]      (Ngrid, Ntile)
    pevapti: torch.Tensor            # PEVAPTI  tile moisture flux [kg/m^2/s] (Ngrid, Ntile)

    # ---- Precipitation throughfall (per grid) ----
    ptsfc: torch.Tensor              # PTSFC  convective throughfall [kg/m^2/s]
    ptsfl: torch.Tensor              # PTSFL  large-scale throughfall [kg/m^2/s]

    # ---- Snow inputs (M6 forward-compat; zero at cold start) ----
    pmsn: torch.Tensor               # PMSN     snowmelt              [kg/m^2/s]
    pemssn: torch.Tensor             # PEMSSN   snow evap mismatch    [kg/m^2/s]
    pevapsnw: torch.Tensor           # PEVAPSNW snow-under-forest evap [kg/m^2/s]
    peintti: torch.Tensor            # PEINTTI  interception-layer evap mismatch (Ngrid, Ntile)

    # ---- Static descriptors ----
    ksoty: torch.Tensor              # KSOTY  soil texture type [long]  (Ngrid,)  0..7
    ktvl: torch.Tensor               # KTVL   low-veg type       [long]  (Ngrid,)  0..20
    ktvh: torch.Tensor               # KTVH   high-veg type      [long]  (Ngrid,)  0..20
    psdor: torch.Tensor              # PSDOR  orography param    [m]     (Ngrid,)

    # ---- Masks ----
    ldland: torch.Tensor             # LDLAND  land mask (bool)          (Ngrid,)

    # ---- Timestep ----
    ptmst: torch.Tensor              # PTMST  timestep            [s]     scalar


@dataclass
class SoilMoistOutputs:
    """HTESSEL 土壤水分求解的全部输出。"""

    pwsa: torch.Tensor               # PWSA  new soil moisture  [m3/m3]   (Ngrid, Nsoil)
    pros: torch.Tensor               # PROS  surface runoff     [kg/m^2]  (Ngrid,)
    prod: torch.Tensor               # PROD  sub-surface runoff  [kg/m^2]  (Ngrid,)
    pwfsd: torch.Tensor              # PWFSD flux layer1->layer2 [kg/m^2/s] (Ngrid,)
    psawgfl: torch.Tensor            # PSAWGFL gravity water flux [kg/m^2/s] (Ngrid, Nsoil)


def compute_soil_moisture(
    inp: SoilMoistInputs,
    soil_table: SoilTable,
    veg_table,                       # surf_pytorch.tables.veg.VegTable
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    differentiable: Optional[bool] = None,
) -> SoilMoistOutputs:
    """运行 HTESSEL 土壤水分求解（SRFWEXC_VG + SRFWDIF + SRFWINC + SRFWNG）。

    Args
    ----
    inp:
        :class:`SoilMoistInputs` -- 打包的土壤状态、通量与掩膜。
    soil_table:
        :class:`surf_pytorch.tables.soil.SoilTable` -- 逐类别 VG 查找表
        （wsat、wrest、wcap、wpwp、wconsm、mvgalpha、mfac、lambda、
        dmax、dmin）以及与类别无关的标量（rsimp、rsrdep）。
    veg_table:
        :class:`surf_pytorch.tables.veg.VegTable` -- 提供逐植被类型的
        根系廓线 ``rvrootsa``。
    consts:
        :class:`ConstantBundle` -- 提供 ``rtt`` 以及冻结过渡区端点
        ``rtf1..rtf4``。
    config:
        :class:`SurfConfig` -- ``n_soil``、``soil_thickness``，以及
        ``lessro`` 标志（VIC 与入渗限制产流的切换）。
    differentiable:
        为本次调用覆盖 ``config.differentiable``。

    Returns
    -------
    :class:`SoilMoistOutputs` -- 新湿度、径流分解（地表 + 地下）、
    层间通量以及重力水通量廓线。
    """
    diff = config.differentiable if differentiable is None else differentiable

    def _core() -> SoilMoistOutputs:
        return _soil_moisture_core(inp, soil_table, veg_table, consts, config)

    if not diff:
        with torch.no_grad():
            return _core()
    return _core()


def _soil_moisture_core(
    inp: SoilMoistInputs,
    soil_table: SoilTable,
    veg_table,
    consts: ConstantBundle,
    config: SurfConfig,
) -> SoilMoistOutputs:
    """内部：假定 no_grad / enable_grad 的决定已经做出。"""
    ngrid = inp.pwsam1m.shape[0]
    nlevs = config.n_soil
    dtype = inp.pwsam1m.dtype
    device = inp.pwsam1m.device

    # Scalar constants.
    rtf1 = consts.rtf1
    rtf2 = consts.rtf2
    rtf3 = consts.rtf3
    rtf4 = consts.rtf4
    rho_h2o = float(soil_table.consts.rho_h2o)
    rsimp = soil_table.consts.rsimp
    rsrdep = soil_table.consts.rsrdep
    rsigormin = soil_table.consts.rsigormin
    rsigormax = soil_table.consts.rsigormax
    rpsfr = getattr(soil_table.consts, "rpsfr", 1.0)
    ptmst = inp.ptmst
    lessro = getattr(config, "lessro", True)

    # Per-class LUT columns indexed by soil type.
    ksoty = inp.ksoty.long().detach()
    wsat = soil_table.gather("wsat", ksoty)
    wrest = soil_table.gather("wrest", ksoty)
    wcap = soil_table.gather("wcap", ksoty)
    wpwp = soil_table.gather("wpwp", ksoty)
    wconsm = soil_table.gather("wconsm", ksoty)
    mvgalpha = soil_table.gather("mvgalpha", ksoty)
    mfac = soil_table.gather("mfac", ksoty)
    lam = soil_table.gather("lambda", ksoty)
    dmax = soil_table.gather("dmax", ksoty)
    dmin = soil_table.gather("dmin", ksoty)

    rdaw = _soil_thickness(config, dtype, device)         # (Nsoil,)
    rdaw_b = rdaw.expand(ngrid, nlevs)                    # (Ngrid, Nsoil)

    land = inp.ldland.to(dtype)                           # (Ngrid,)

    # ------------------------------------------------------------------ #
    # Step 2.1: liquid fraction ZF (T-dependent freeze function) and the
    # liquid-equivalent soil moisture ZLIQ used for root extraction.
    # ------------------------------------------------------------------ #
    zf = _liquid_fraction(inp.ptsam1m, rtf1, rtf2, rtf3, rtf4)    # (Ngrid, Nsoil)
    wpwp_b = wpwp[:, None].expand(-1, nlevs)
    wcap_b = wcap[:, None].expand(-1, nlevs)
    zliq = torch.clamp(inp.pwsam1m * (1.0 - zf), min=wpwp_b, max=wcap_b)

    # ------------------------------------------------------------------ #
    # Step 2.2: per-tile root extraction weights ZROOTW and normaliser ZEXT.
    # Only tiles 4 (low-veg-dry), 6 (high-veg-dry), 7 (high-veg-snow) have
    # root extraction; for positive evap (dew) all roots are in layer 1.
    # ------------------------------------------------------------------ #
    zrootw, zext = _root_extraction_weights(
        inp.pfrti, inp.pevapti, inp.pevapsnw, zliq, wpwp, zf,
        inp.ktvl, inp.ktvh, ksoty, veg_table, nlevs,
        lewbsoilfix=getattr(soil_table.consts, "lewbsoilfix", True),
    )

    # ------------------------------------------------------------------ #
    # Step 2.3: top-layer coefficients (Van Genuchten K/D + runoff +
    # tile contributions + ZWSFL surface forcing).
    # ------------------------------------------------------------------ #
    pcfw = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)
    prhsw = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)
    psawgfl = torch.zeros((ngrid, nlevs), dtype=dtype, device=device)
    pros = torch.zeros((ngrid,), dtype=dtype, device=device)
    # Top layer.
    pcfw, prhsw, psawgfl, pros, zsawext_l1, zconds = _top_layer_coeffs(
        inp, ksoty, wsat, wrest, wcap, wpwp, wconsm, mvgalpha, mfac, lam,
        dmax, dmin, zf, zrootw, zext,
        rdaw, land, ptmst, rsimp, rsrdep, rho_h2o, lessro,
        rsigormin, rsigormax, rpsfr,
        pcfw, prhsw, psawgfl, pros,
    )

    # ------------------------------------------------------------------ #
    # Step 2.4: deeper-layer coefficients (JK=2..KLEVS).
    # ------------------------------------------------------------------ #
    pcfw, prhsw, psawgfl = _deep_layer_coeffs(
        inp, ksoty, wsat, wrest, wcap, wpwp, wconsm, mvgalpha, mfac, lam,
        dmax, dmin, zf, zrootw, zext,
        rdaw, land, ptmst, rsimp, rho_h2o,
        pcfw, prhsw, psawgfl,
    )

    # ------------------------------------------------------------------ #
    # Step 4.3: SRFWDIF -- Thomas solve for PWSADIF.
    # ZCDZ here is RDAW (the layer thickness profile).
    # ------------------------------------------------------------------ #
    zcdz = rdaw_b.clone()
    pwsadif = _solve_tridiag(inp.pwsam1m, pcfw, prhsw, zcdz, inp.ldland, rsimp)

    # ------------------------------------------------------------------ #
    # Step 4.4: SRFWINC -- new unfiltered moisture + drainage diagnostic.
    # ------------------------------------------------------------------ #
    pwsa_pre, psawgfl = _soil_moisture_update(
        inp.pwsam1m, pwsadif, pcfw, psawgfl, inp.ldland, rsimp, ptmst, rho_h2o,
    )
    pwfsd = psawgfl[:, 0].clone()

    # ------------------------------------------------------------------ #
    # Step 4.5: SRFWNG -- clip negative moisture, redistribute downward,
    # and split saturation excess into surface / sub-surface runoff.
    # ------------------------------------------------------------------ #
    pwsa, pros, prod, pwfsd = _clip_and_redistribute(
        pwsa_pre, inp.pwsam1m, pros, pwfsd, wsat, wrest, rdaw,
        land, inp.ldland, ptmst, rho_h2o,
    )

    _write_srfwater_debug(
        inp.pwsam1m, pcfw, prhsw, pwsadif, pwsa_pre, pwsa, psawgfl,
        pros, prod, pwfsd,
    )

    # ------------------------------------------------------------------ #
    # Step 4.6: SURFTSTP_CTL section 4.6 -- bottom-layer gravity drainage
    # is added to the sub-surface runoff diagnostic (so PROD carries the
    # full water *out* of the column).  Without this the water budget is
    # not closed.
    # ------------------------------------------------------------------ #
    prod = prod + psawgfl[:, -1] * ptmst * land

    return SoilMoistOutputs(
        pwsa=pwsa, pros=pros, prod=prod, pwfsd=pwfsd, psawgfl=psawgfl,
    )


def _write_srfwater_debug(
    pwsam1m, pcfw, prhsw, pwsadif, pwsa_pre, pwsa, psawgfl,
    pros, prod, pwfsd,
) -> None:
    """为单个列写出可选的 SRFWEXC/SRFWDIF/SRFWINC 审计信息。"""
    global _SRFW_DEBUG_STEP
    path = os.environ.get("SURF_TORCH_SRF_WATER_DEBUG_CSV")
    column = os.environ.get("SURF_TORCH_SRF_WATER_DEBUG_COLUMN")
    if not path or not column:
        return
    try:
        jl = int(column) - 1
    except ValueError as exc:
        raise ValueError("SURF_TORCH_SRF_WATER_DEBUG_COLUMN must be 1-based") from exc
    if not 0 <= jl < pwsam1m.shape[0]:
        raise ValueError("SURF_TORCH_SRF_WATER_DEBUG_COLUMN is outside the domain")

    values = [
        pwsam1m, pcfw, prhsw, pwsadif, pwsa_pre, pwsa, psawgfl,
        pros[:, None], prod[:, None], pwfsd[:, None],
    ]
    cpu = [value.detach().cpu() for value in values]
    row = [_SRFW_DEBUG_STEP, jl + 1]
    for value in cpu[:7]:
        row.extend(float(x) for x in value[jl])
    for value in cpu[7:]:
        row.append(float(value[jl, 0]))
    header = ["step", "jl"]
    for prefix in ("pwsa_in", "pcfw", "prhsw", "pwsadif", "pwsa_pre", "pwsa", "psawgfl"):
        header.extend(f"{prefix}{k}" for k in range(1, 5))
    header.extend(("pros", "prod", "pwfsd"))
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(header)
        writer.writerow(row)
    _SRFW_DEBUG_STEP += 1


# --------------------------------------------------------------------------- #
# Helpers: soil-moisture branch
# --------------------------------------------------------------------------- #
def _liquid_fraction(
    ptsam1m: torch.Tensor,
    rtf1: float, rtf2: float, rtf3: float, rtf4: float,
) -> torch.Tensor:
    """液态水比例 ZF，作为土壤温度的函数。

    复现 srfwexc_vg_mod.F90 第 219-227 行::

        ZF = 0.5*(1 - SIN(RTF4*(T-RTF3)))   if RTF2 < T < RTF1
        ZF = 1.0                             if T <= RTF2  (all liquid)
        ZF = 0.0                             if T >= RTF1  (all frozen)

    在冻结窗口之外 ZF 为分段常数，与上游 SRFENE 的定义一致。
    在 (Ngrid, Nsoil) 上向量化。
    """
    t = ptsam1m
    zf = 0.5 * (1.0 - torch.sin(rtf4 * (t - rtf3)))
    zf = torch.where(t > rtf2, zf, torch.ones_like(zf))      # T <= RTF2 -> 1
    zf = torch.where(t < rtf1, zf, torch.zeros_like(zf))     # T >= RTF1 -> 0
    return zf


def _root_extraction_weights(
    pfrti: torch.Tensor,
    pevapti: torch.Tensor,
    pevapsnw: torch.Tensor,
    zliq: torch.Tensor,            # (Ngrid, Nsoil) liquid-equivalent moisture
    wpwp: torch.Tensor,            # (Ngrid,) wilting point
    zf: torch.Tensor,              # (Ngrid, Nsoil) liquid fraction
    ktvl: torch.Tensor, ktvh: torch.Tensor,
    ksoty: torch.Tensor,
    veg_table,
    nlevs: int,
    lewbsoilfix: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐贴块根系吸水权重 ZROOTW 和归一化因子 ZEXT。

    复现 srfwexc_vg_mod.F90 第 268-310 行（TILES + LAYERS 循环）。
    只有贴块 4（低植被-干，KTVL）、6（高植被-干，KTVH）和 7
    （高植被-雪，KTVH）携带根系吸水。对于正蒸发（露水凝结），
    所有根系都在第 1 层；对于蒸腾，权重为
    ``RVROOTSA(JK, ITYP) * (ZLIQ - RWPWPM + ZEPSILONADD)``。

    Returns
    -------
    zrootw: ``(Ngrid, Nsoil, 3)`` -- 逐（格点，层，贴块∈{4,6,7}）。
    zext:   ``(Ngrid, 3)`` -- 逐贴块归一化因子（钳制到 >= ZEPSILON）。
    """
    ngrid = pfrti.shape[0]
    dtype = pfrti.dtype
    device = pfrti.device
    tile_local = (3, 5, 6)          # 0-indexed for tiles 4, 6, 7

    # Per-veg-type root profiles: (Ngrid, Nsoil) for each of low / high veg.
    roots_lo = veg_table.gather_roots(ktvl.long().detach())
    roots_hi = veg_table.gather_roots(ktvh.long().detach())

    wpwp_b = wpwp[:, None].expand(-1, nlevs)
    zeps_add = 100.0 * torch.finfo(dtype).eps if lewbsoilfix else 0.0

    zrootw = torch.zeros((ngrid, nlevs, 3), dtype=dtype, device=device)
    zext = torch.zeros((ngrid, 3), dtype=dtype, device=device)

    # Tile 4 (low-veg-dry) -> uses KTVL roots.
    zevap4 = pfrti[:, 3] * pevapti[:, 3]
    dew4 = zevap4 > 0.0
    # Layer 1 = 1 for dew; else roots*(zliq - wpwp).
    w4 = roots_lo * (zliq - wpwp_b + zeps_add)
    zrootw[:, 0, 0] = torch.where(dew4, torch.ones_like(zrootw[:, 0, 0]), w4[:, 0])
    zrootw[:, 1:, 0] = torch.where(
        dew4[:, None].expand(-1, nlevs - 1),
        torch.zeros_like(zrootw[:, 1:, 0]),
        w4[:, 1:],
    )
    # srfwexc_mod.F90 222-265: ZEXT accumulates layers JK=1..KLEVS
    # sequentially inside the LAYERS loop.
    zext[:, 0] = torch.zeros_like(zrootw[:, 0, 0])
    for jk in range(zrootw.shape[1]):
        zext[:, 0] = zext[:, 0] + zrootw[:, jk, 0]

    # Tile 6 (high-veg-dry) -> uses KTVH roots.
    zevap6 = pfrti[:, 5] * pevapti[:, 5]
    dew6 = zevap6 > 0.0
    w6 = roots_hi * (zliq - wpwp_b + zeps_add)
    zrootw[:, 0, 1] = torch.where(dew6, torch.ones_like(zrootw[:, 0, 1]), w6[:, 0])
    zrootw[:, 1:, 1] = torch.where(
        dew6[:, None].expand(-1, nlevs - 1),
        torch.zeros_like(zrootw[:, 1:, 1]),
        w6[:, 1:],
    )
    # srfwexc_mod.F90 222-265: ZEXT accumulates layers JK=1..KLEVS
    # sequentially inside the LAYERS loop.
    zext[:, 1] = torch.zeros_like(zrootw[:, 0, 1])
    for jk in range(zrootw.shape[1]):
        zext[:, 1] = zext[:, 1] + zrootw[:, jk, 1]

    # Tile 7 (high-veg-snow) -> uses KTVH roots; evap excludes PEVAPSNW.
    zevap7 = pfrti[:, 6] * (pevapti[:, 6] - pevapsnw)
    dew7 = zevap7 > 0.0
    w7 = roots_hi * (zliq - wpwp_b + zeps_add)
    zrootw[:, 0, 2] = torch.where(dew7, torch.ones_like(zrootw[:, 0, 2]), w7[:, 0])
    zrootw[:, 1:, 2] = torch.where(
        dew7[:, None].expand(-1, nlevs - 1),
        torch.zeros_like(zrootw[:, 1:, 2]),
        w7[:, 1:],
    )
    # srfwexc_mod.F90 222-265: ZEXT accumulates layers JK=1..KLEVS
    # sequentially inside the LAYERS loop.
    zext[:, 2] = torch.zeros_like(zrootw[:, 0, 2])
    for jk in range(zrootw.shape[1]):
        zext[:, 2] = zext[:, 2] + zrootw[:, jk, 2]

    # Clamp ZEXT to >= ZEPSILON (Fortran MAX(ZEPSILON, ZEXT)).
    zeps = 100.0 * torch.finfo(dtype).eps
    zext = torch.clamp(zext, min=zeps)
    return zrootw, zext


def _vg_hydraulics(
    w: torch.Tensor,             # (Ngrid,) moisture at the half-level
    wsat: torch.Tensor,
    wrest: torch.Tensor,
    wconsm: torch.Tensor,
    mvgalpha: torch.Tensor,
    mfac: torch.Tensor,
    lam: torch.Tensor,
    dmax: torch.Tensor,
    dmin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Van Genuchten 水力传导度 K 和扩散率 D。

    复现 srfwexc_vg_mod.F90 中的半层 VG 块（顶层第 318-332 行，
    深层第 509-523 行）。三分段的分段形式在气干下限
    （W <= 1.001*Wres）和接近饱和（W >= 0.999*Wsat）处钳制 K/D；
    两者之间使用闭式 VG 表达式。

    Returns
    -------
    (zk, zd): ``(Ngrid,)`` 给定半层湿度处的水力传导度 [m/s]
    和扩散率 [m^2/s]。
    """
    wfac = wsat - wrest

    # Fortran evaluates the Van Genuchten closed form only in the middle
    # branch.  torch.where evaluates both branches eagerly, so protect the
    # fractional powers for floor/saturated columns with a valid placeholder.
    floor = w <= 1.001 * wrest
    sat = w >= 0.999 * wsat
    mid = ~(floor | sat)
    se = (w - wrest) / wfac
    se_safe = torch.where(mid, se, torch.full_like(se, 0.5))
    rmfac = 1.0 / mfac
    se_rmfac = se_safe ** rmfac
    # NOTE the bracketing matches the Fortran closed form exactly.
    # The K formula is
    #   ZKD = ZWCONS * ZSE**ZLAM * (1 - (1 - ZSE**ZRMFAC)**ZMFAC)**2
    # The D formula is
    #   ZDD = ((1-ZMFAC)*ZWCONS / (ZALPHA*ZMFAC*ZWFAC)) * ZSE**(ZLAM-ZRMFAC)
    #         * ((1-ZSE**ZRMFAC)**ZMFAC + (1-ZSE**ZRMFAC)**(-ZMFAC) - 2)
    # The two share the building block (1 - se^rm) but bracket it
    # differently -- do NOT reuse the K-bracket for D.  We precompute the
    # raw power and form each bracket separately.
    raw_pow = 1.0 - se_rmfac                       # (1 - se^rm)
    k_bracket = 1.0 - raw_pow ** mfac              # 1 - (1 - se^rm)^m
    d_summand = raw_pow ** mfac + raw_pow ** (-mfac) - 2.0

    # Mid-branch K and D (closed-form VG).
    zk_mid = wconsm * (se_safe ** lam) * (k_bracket ** 2)
    # D uses SIGN(MAX(ABS(x),eps),x) to protect against zero -- with the
    # default positive alpha/mfac/wfac this is just MAX(ABS(x),eps).
    denom = torch.clamp((mvgalpha * mfac * wfac).abs(), min=eps)
    zd_mid = ((1.0 - mfac) * wconsm / denom) * (se_safe ** (lam - rmfac)) * d_summand

    # Branch selection.

    zk = torch.where(mid, zk_mid, torch.zeros_like(zk_mid))
    zk = torch.where(sat, wconsm, zk)                  # near-sat -> K = WCONS
    # floor -> K = 0 (already zero).

    zd = torch.where(mid, zd_mid, dmin)               # floor -> D = Dmin
    zd = torch.where(sat, dmax, zd)                   # near-sat -> D = Dmax
    return zk, zd


def _top_layer_coeffs(
    inp: SoilMoistInputs,
    ksoty, wsat, wrest, wcap, wpwp, wconsm, mvgalpha, mfac, lam,
    dmax, dmin, zf, zrootw, zext,
    rdaw, land, ptmst, rsimp, rsrdep, rho_h2o, lessro,
    rsigormin, rsigormax, rpsfr,
    pcfw, prhsw, psawgfl, pros,
):
    """SRFWEXC_VG 第 2.3 节 -- 顶层系数。

    返回更新后的 (pcfw, prhsw, psawgfl, pros)，以及 SRFWNG 所需的
    诊断量 (zsawext_l1, zconds)。
    """
    ngrid = inp.pwsam1m.shape[0]
    nlevs = rdaw.shape[0]
    dtype = inp.pwsam1m.dtype
    device = inp.pwsam1m.device
    zeps = 100.0 * torch.finfo(dtype).eps
    zhoh2o = 1.0 / rho_h2o

    # ZW = MAX(MAX(W(:,1), W(:,2)), WREST) -- half-level-1 moisture.
    w_top = torch.maximum(
        torch.maximum(inp.pwsam1m[:, 0], inp.pwsam1m[:, 1]), wrest
    )
    zk, zd = _vg_hydraulics(
        w_top, wsat, wrest, wconsm, mvgalpha, mfac, lam, dmax, dmin, zeps,
    )

    # Freezing reduction: ZFF = MIN(ZF(:,1), ZF(:,2)).
    if nlevs >= 2:
        zff = torch.minimum(zf[:, 0], zf[:, 1])
    else:
        zff = zf[:, 0]
    zd = zff * dmin + (1.0 - zff) * zd
    zk = (1.0 - zff) * zk                   # ZK = 0 for frozen soil

    # Runoff: VIC saturation-excess OR infiltration-limit (LESSRO switch).
    zrot = _runoff(
        inp, ksoty, wsat, wrest, wpwp, wconsm, dmax, dmin, zf,
        rdaw, rsrdep, lessro, ptmst, rho_h2o,
        rsigormin, rsigormax, rpsfr,
    )

    # Surface forcing ZWSFL = throughfall + snowmelt + conv - runoff.
    zwsfl = inp.ptsfl + inp.pmsn + inp.ptsfc - zrot

    # Tile contributions (tiles 3, 4, 6, 7, 8).
    # Tile 3 (wet-skin): PFRTI*PEVAPTI - PEINTTI.
    zwsfl = zwsfl + inp.pfrti[:, 2] * inp.pevapti[:, 2] - inp.peintti[:, 2]
    # Tiles 4, 6, 7: root-extraction split into layer 1.
    for it, tile in enumerate((3, 5, 6)):       # tiles 4, 6, 7 (0-indexed 3,5,6)
        if tile == 6:
            evap_tile = inp.pfrti[:, tile] * (inp.pevapti[:, tile] - inp.pevapsnw)
        else:
            evap_tile = inp.pfrti[:, tile] * inp.pevapti[:, tile]
        zextk = (evap_tile - inp.peintti[:, tile]) * zrootw[:, 0, it] / zext[:, it]
        zwsfl = zwsfl + zextk
    # Tile 8 (bare-soil): PFRTI*PEVAPTI + PEMSSN - PEINTTI.
    zwsfl = (
        zwsfl
        + inp.pfrti[:, 7] * inp.pevapti[:, 7]
        + inp.pemssn - inp.peintti[:, 7]
    )

    # PCFW(:,1) and PRHSW(:,1).
    if nlevs >= 2:
        denom_cfw = 0.5 * (rdaw[0] + rdaw[1])
    else:
        denom_cfw = 0.5 * rdaw[0]
    pcfw[:, 0] = zd * ptmst * rsimp / denom_cfw
    prhsw[:, 0] = ptmst * (-zk + zwsfl * zhoh2o) / rdaw[0]

    # Diagnostics.
    psawgfl[:, 0] = rho_h2o * zk
    pros = ptmst * zrot

    # Land mask: zero out non-land columns.
    land_b = land
    pcfw[:, 0] = pcfw[:, 0] * land_b
    prhsw[:, 0] = prhsw[:, 0] * land_b
    psawgfl[:, 0] = psawgfl[:, 0] * land_b
    pros = pros * land_b
    return pcfw, prhsw, psawgfl, pros, None, None


def _runoff(
    inp: SoilMoistInputs,
    ksoty, wsat, wrest, wpwp, wconsm, dmax, dmin, zf,
    rdaw, rsrdep, lessro, ptmst, rho_h2o,
    rsigormin, rsigormax, rpsfr,
) -> torch.Tensor:
    """地表径流 ZROT（kg/m^2/s，以流出土壤为正）。

    由 ``lessro`` 门控的两个分支：
    * LESSRO = .TRUE.  -> VIC 饱和超渗产流（srfwexc_vg 第 340-391 行）。
    * LESSRO = .FALSE. -> 入渗限制产流（srfwexc_vg 第 393-416 行）。

    返回逐格点径流率（kg/m^2/s）。surf 点由调用方的陆地掩膜
    置零。
    """
    ngrid = inp.pwsam1m.shape[0]
    nlevs = rdaw.shape[0]
    dtype = inp.pwsam1m.dtype
    device = inp.pwsam1m.device
    invtmst = 1.0 / ptmst
    js = ksoty

    zrsfl = inp.ptsfl + inp.pmsn + inp.ptsfc
    raining = zrsfl > 0.0

    if lessro:
        # VIC: relative saturation in the upper RSRDEP layer.
        # ZFRK(JK) = fraction of layer JK inside the RSRDEP window.
        # rdaw is constant across the batch, so we evaluate ZFRK once as a
        # plain 1-D tensor and broadcast (no per-grid state involved).
        rdaw_py = rdaw.detach().to("cpu", dtype=torch.float64).tolist()
        zfrk_list = []
        for jk in range(nlevs):
            zsum_lo = sum(rdaw_py[:jk]) if jk > 0 else 0.0
            zsum_hi = sum(rdaw_py[: jk + 1])
            val = max(0.0, min(rsrdep, zsum_hi) - zsum_lo) / rdaw_py[jk]
            zfrk_list.append(max(0.0, val))
        zfrk_1d = torch.as_tensor(zfrk_list, dtype=dtype, device=device)
        zfrk = zfrk_1d.expand(ngrid, nlevs)

        # Orographic sub-grid coefficient ZBWS.  RSIGORMIN/MAX are the
        # upstream defaults (100/1000) -- see sussoil_mod.F90:480-481.
        zroeff = (inp.psdor - rsigormin) / (inp.psdor + rsigormax)
        zroeff = torch.clamp(zroeff, min=0.0, max=0.5)
        zroeff = torch.clamp(zroeff, min=0.01)
        zbws = zroeff
        zb1 = 1.0 + zbws
        zbm = 1.0 / zb1

        # ZW = liquid water in upper RSRDEP window.
        zwk = torch.clamp(inp.pwsam1m - wpwp[:, None], min=0.0) * (
            1.0 - zf
        ) + wpwp[:, None]
        rdaw_g = rdaw.expand(ngrid, nlevs)
        # srfwexc_vg_mod.F90 392/399: SUM(ZFRK*ZWK*RDAW) is a sequential
        # layer sum (gfortran SUM == DO loop, verified bitwise).
        zw = torch.zeros_like(zfrk[:, 0])
        for jk in range(zfrk.shape[1]):
            zw = zw + (zfrk[:, jk] * zwk[:, jk]) * rdaw_g[:, jk]
        zwmk = (wsat[:, None] - wpwp[:, None]) * (1.0 - zf) + wpwp[:, None]
        zwmax = torch.zeros_like(zfrk[:, 0])
        for jk in range(zfrk.shape[1]):
            zwmax = zwmax + (zfrk[:, jk] * zwmk[:, jk]) * rdaw_g[:, jk]

        zconw1 = zwmax * zb1
        zlyeps = torch.clamp(zw - zwmax, min=0.0)
        # ZLIMRS = -ZLYEPS unless ZLYEPS > 0.1*ZWMAX (then 0).
        zlimrs = -zlyeps
        zlimrs = torch.where(
            zlyeps > 0.1 * zwmax, torch.zeros_like(zlimrs), zlimrs
        )
        # ZLYSIC = saturation-area fraction.
        # ``where`` does not short-circuit its operands. Use a neutral
        # denominator before masking zero-capacity columns, otherwise the
        # inactive 0/0 division poisons an unrolled autograd graph.
        zwmax_safe = torch.where(zwmax != 0.0, zwmax, torch.ones_like(zwmax))
        zlysic = torch.clamp((zw - zlyeps) / zwmax_safe, min=0.0, max=1.0)
        zlysic = torch.where(zwmax != 0.0, zlysic, torch.zeros_like(zlysic))
        # ZVOL = saturation-excess volume.
        zpow_base = 1.0 - zlysic
        if torch.is_grad_enabled():
            # At saturation zpow_base=0 and zbm is fractional. The forward
            # value is finite but d(x**zbm)/dx is singular at x=0. The
            # epsilon floor is only for the differentiated graph.
            zpow_base = torch.clamp(zpow_base, min=torch.finfo(dtype).eps)
        zconw1_safe = torch.where(zconw1 != 0.0, zconw1, torch.ones_like(zconw1))
        zvol = torch.clamp(
            zpow_base ** zbm - zrsfl / (invtmst * rho_h2o * zconw1_safe),
            min=0.0,
        )
        # ZROS (m): saturation-excess runoff.
        zros = zrsfl / (invtmst * rho_h2o) - torch.maximum(
            zwmax - zw, zlimrs
        )
        zros = torch.where(zvol > 0.0, zros + zwmax * zvol ** zb1, zros)
        zros = torch.clamp(zros, min=0.0)
        zrot = zros * rho_h2o * invtmst
        zrot = torch.where(raining, zrot, torch.zeros_like(zrot))
        return zrot

    # Infiltration-limit runoff.
    zdsurf = dmax
    zksurf = wconsm
    zff1 = zf[:, 0]
    zdsurf = zff1 * dmin + (1.0 - zff1) * dmax
    zksurf = (1.0 - zff1) * wconsm
    zinfmax = (
        zdsurf * (wsat - inp.pwsam1m[:, 0]) / (0.5 * rdaw[0]) + zksurf
    ) * rho_h2o
    # Large-scale precip runoff.
    zrol = torch.clamp(inp.ptsfl + inp.pmsn - zinfmax, min=0.0)
    # Convective precip runoff. Keep the Fortran operation order:
    # ZROC = MAX(0, RPSFR * PTSFC - ZINFMAX) * ZPSFR,
    # with RPSFR=2 and ZPSFR=1/RPSFR=0.5 in SUSSOIL.
    rpsfr = torch.as_tensor(rpsfr, dtype=dtype, device=device)
    zpsfr = 1.0 / rpsfr
    zroc = torch.clamp(rpsfr * inp.ptsfc - zinfmax, min=0.0) * zpsfr
    zrot = zrol + zroc
    return zrot


def _deep_layer_coeffs(
    inp: SoilMoistInputs,
    ksoty, wsat, wrest, wcap, wpwp, wconsm, mvgalpha, mfac, lam,
    dmax, dmin, zf, zrootw, zext,
    rdaw, land, ptmst, rsimp, rho_h2o,
    pcfw, prhsw, psawgfl,
):
    """SRFWEXC_VG 第 2.4 节 -- 层 JK=2..KLEVS 的系数。

    每个半层 JK+1/2 使用 MAX(W(:,JK), W(:,JK+1)) 作为 K/D 的
    半层湿度；层 JK *顶部*的重力通量 K 使用
    MAX(W(:,JK-1), W(:,JK))（Fortran 中的 ZKM 变量）。冻结削减
    通过 MIN(ZF(:,JK), ...) 耦合相邻层。
    """
    ngrid = inp.pwsam1m.shape[0]
    nlevs = rdaw.shape[0]
    dtype = inp.pwsam1m.dtype
    device = inp.pwsam1m.device
    zeps = 100.0 * torch.finfo(dtype).eps
    zhoh2o = 1.0 / rho_h2o
    land_b = land

    for jk in range(1, nlevs):
        # Half-level JK+1/2 moisture (for D and the lower K).
        if jk < nlevs - 1:
            w_half = torch.maximum(
                torch.maximum(inp.pwsam1m[:, jk], inp.pwsam1m[:, jk + 1]),
                wrest,
            )
        else:
            w_half = torch.maximum(inp.pwsam1m[:, jk], wrest)
        zk, zd = _vg_hydraulics(
            w_half, wsat, wrest, wconsm, mvgalpha, mfac, lam, dmax, dmin, zeps,
        )

        # ZKM = K at the top of this layer (half-level JK-1/2).
        wm = torch.maximum(
            torch.maximum(inp.pwsam1m[:, jk - 1], inp.pwsam1m[:, jk]), wrest
        )
        zkm, _ = _vg_hydraulics_toponly(
            wm, wsat, wrest, wconsm, mvgalpha, mfac, lam, zeps,
        )

        # Bottom layer: ZD = 0 (no flux below).
        if jk == nlevs - 1:
            zd_layer = torch.zeros_like(zd)
        else:
            zd_layer = zd

        # Freezing reduction.
        if jk < nlevs - 1:
            zff = torch.minimum(zf[:, jk], zf[:, jk + 1])
        else:
            zff = zf[:, jk]
        zffm = torch.minimum(zf[:, jk - 1], zf[:, jk])
        zd_layer = zff * dmin + (1.0 - zff) * zd_layer
        zk = (1.0 - zff) * zk
        zkm = (1.0 - zffm) * zkm

        # Root-extraction contributions to ZWSFL (tiles 4, 6, 7).
        zwsfl = torch.zeros((ngrid,), dtype=dtype, device=device)
        for it, tile in enumerate((3, 5, 6)):
            if tile == 6:
                evap_tile = inp.pfrti[:, tile] * (inp.pevapti[:, tile] - inp.pevapsnw)
            else:
                evap_tile = inp.pfrti[:, tile] * inp.pevapti[:, tile]
            zextk = (evap_tile - inp.peintti[:, tile]) * zrootw[:, jk, it] / zext[:, it]
            zwsfl = zwsfl + zextk

        # PCFW and PRHSW.
        if jk < nlevs - 1:
            denom_cfw = 0.5 * (rdaw[jk] + rdaw[jk + 1])
            pcfw[:, jk] = zd_layer * ptmst * rsimp / denom_cfw
        else:
            pcfw[:, jk] = torch.zeros((ngrid,), dtype=dtype, device=device)
        prhsw[:, jk] = ptmst * (zkm - zk + zwsfl * zhoh2o) / rdaw[jk]

        psawgfl[:, jk] = rho_h2o * zk

        # Land mask.
        pcfw[:, jk] = pcfw[:, jk] * land_b
        prhsw[:, jk] = prhsw[:, jk] * land_b
        psawgfl[:, jk] = psawgfl[:, jk] * land_b
    return pcfw, prhsw, psawgfl


def _vg_hydraulics_toponly(
    w, wsat, wrest, wconsm, mvgalpha, mfac, lam, eps,
):
    """仅 Van Genuchten K（无 D），用于层顶 ZKM 的 K。

    复现 srfwexc_vg_mod.F90 第 529-540 行（ZKM 块）。与
    :func:`_vg_hydraulics` 相同的三分段分段形式，但只返回 K
    （ZKM 从不作为扩散率使用）。
    """
    wfac = wsat - wrest
    floor = w <= wrest
    sat = w >= 0.999 * wsat
    mid = ~(floor | sat)
    se = (w - wrest) / wfac
    se_safe = torch.where(mid, se, torch.full_like(se, 0.5))
    rmfac = 1.0 / mfac
    se_rmfac = se_safe ** rmfac
    raw_pow = 1.0 - se_rmfac
    # NOTE: same bracketing as :func:`_vg_hydraulics` -- the K closed form
    # is (1 - (1 - se^rm)^m)**2, NOT ((1 - se^rm)^m)**2.  See the lengthy
    # comment in _vg_hydraulics for the derivation.
    k_bracket = 1.0 - raw_pow ** mfac

    zkm_mid = wconsm * (se_safe ** lam) * (k_bracket ** 2)

    zkm = torch.where(mid, zkm_mid, torch.zeros_like(zkm_mid))
    zkm = torch.where(sat, wconsm, zkm)
    # Return (zkm, None) so the call site mirrors _vg_hydraulics.
    return zkm, None


def _soil_moisture_update(
    pwsam1m, pwsadif, pcfw, psawgfl, ldland, rsimp, ptmst, rho_h2o,
):
    """SRFWINC -- 将 Thomas 解转换为 t+1 湿度 + 排水诊断量。

    复现 srfwinc_mod.F90 第 120-144 行::

        ZTPFAC3 = 1/RSIMP - 1
        ZCONS   = RHOH2O/PTMST
        PWSA(:,JK) = PWSADIF(:,JK) - ZTPFAC3*PWSAM1M(:,JK)
        PSAWGFL(:,JK) += ZCONS*PCFW(:,JK)*(PWSADIF(:,JK) - PWSADIF(:,JK+1))

    当 RSIMP=1 时，ZTPFAC3=0 且 PWSA = PWSADIF（Thomas 解
    直接就是新湿度）。Surf 点保留其 t-1 时刻的湿度。
    """
    ztpfac3 = 1.0 / rsimp - 1.0
    zcons = rho_h2o / ptmst
    nlevs = pwsam1m.shape[1]
    land_b = ldland.to(pwsam1m.dtype)[:, None]

    pwsa_land = pwsadif - ztpfac3 * pwsam1m
    pwsa = torch.where(land_b != 0.0, pwsa_land, pwsam1m)

    # Update the gravity-flux diagnostic (layers 1..KLEVS-1).
    for jk in range(nlevs - 1):
        psawgfl[:, jk] = psawgfl[:, jk] + zcons * pcfw[:, jk] * (
            pwsadif[:, jk] - pwsadif[:, jk + 1]
        )
    # Surf points: zero the diagnostic.
    psawgfl = torch.where(land_b != 0.0, psawgfl, torch.zeros_like(psawgfl))
    return pwsa, psawgfl


def _clip_and_redistribute(
    pwsa, pwsam1m, pros, pwfsd, wsat, wrest, rdaw,
    land, ldland, ptmst, rho_h2o,
):
    """SRFWNG -- 截断负湿度，向下再分配，并将饱和超出量分解为
    地表 / 地下径流。

    复现 srfwng_mod.F90 第 130-182 行。截断是一个列递推：
    层 JK 中任何负湿度都被推入 JK+1（按层厚比缩放），任何
    饱和超出量成为地表径流（顶层）或地下径流（深层）。
    皮肤层水库截断（PWL > PWLMX）被省略——离线驱动将皮肤层
    水库留给单独路径（M6 / SRFWL）。

    由于再分配携带层轴递推，我们将其展开为对 JK 的 Python
    循环；nlevs=4 时只有三次迭代。循环体以函数式方式书写
    （逐层湿度张量累积在列表中），以保持 autograd 干净。
    """
    ngrid = pwsa.shape[0]
    nlevs = rdaw.shape[0]
    dtype = pwsa.dtype
    device = pwsa.device
    ztmst = 1.0 / ptmst
    zhoh2o = 1.0 / rho_h2o
    zidi = 1.0 / rdaw                                     # (Nsoil,)

    prod = torch.zeros((ngrid,), dtype=dtype, device=device)
    land_b = land

    # Functional accumulation of the clipped moisture profile.
    sol: list[torch.Tensor] = [pwsa[:, jk] for jk in range(nlevs)]

    for jk in range(nlevs):
        w = sol[jk]
        # Clip negative: ZDWD = MIN(W, 0); push into JK+1.  Note ZDI here
        # is the *layer thickness* (RDAW) -- upstream names it ZDI, with
        # ZIDI = 1/RDAW its reciprocal.  The inter-layer redistribution
        # uses the thickness ratio rdaw[jk]/rdaw[jk+1].
        zdwd = torch.clamp(w, max=0.0)
        w = w - zdwd
        if jk < nlevs - 1:
            sol[jk + 1] = sol[jk + 1] + zdwd * rdaw[jk] * zidi[jk + 1]
        # Saturation excess.  Runoff / drainage carry the excess through
        # the full layer depth, so they scale by rdaw[jk] (kg/m^2).
        zrrri = torch.clamp(w - wsat, min=0.0)
        if jk == 0:
            pros = pros + rdaw[jk] * rho_h2o * zrrri
            pwfsd = pwfsd + zdwd * rho_h2o * rdaw[jk] * ztmst
        else:
            prod = prod + rdaw[jk] * rho_h2o * zrrri
        sol[jk] = w - zrrri

    pwsa_out = torch.stack(sol, dim=1)
    # Land mask: zero runoff for surf points, keep t-1 moisture.
    pwsa_out = torch.where(
        ldland.to(dtype)[:, None] != 0.0, pwsa_out, pwsam1m
    )
    pros = pros * land_b
    prod = prod * land_b
    pwfsd = pwfsd * land_b
    return pwsa_out, pros, prod, pwfsd


__all__ = [
    "SoilTempInputs",
    "SoilTempOutputs",
    "compute_soil_temperature",
    "SoilMoistInputs",
    "SoilMoistOutputs",
    "compute_soil_moisture",
]
