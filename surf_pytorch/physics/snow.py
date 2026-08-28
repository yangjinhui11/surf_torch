"""HTESSEL 单层雪方案，对应 ``SURFTSTP_CTL`` 路径。

本模块移植了 ``SRFSN_LWIMP``，即当 ``LESNML=.FALSE.`` 且
``LESN09=.TRUE.`` 时，由 ``SURFTSTP_CTL`` 在 OpenIFS 耦合模式及其
离线驱动程序中调用的单层雪过程。该接口不提供简化雪求解器。

方案功能
--------
单层雪预报在一个耦合时间间隔 ``PTMST`` 内积分三个耦合的收支：

1. **雪量** ``PSN``（水当量，kg/m^2）：由大尺度 + 对流降雪
   （``PSSFL/PSSFC``）、雪瓦片的蒸发/升华、林下积雪蒸发
   （``PEVAPSNW``）以及穿透降水截获 / 融化推进。升华取自雪瓦片
   5（低植被积雪）和 7（高植被下积雪），按各自瓦片比例加权。

2. **雪温** ``PTSN``：对雪热容和雪-土壤导热耦合采用隐式 Euler。
   “雪热容”是通过解析融化过渡函数 ``ZFUNC`` 及其导数 ``ZDFUNC``
   （在 ``[RTT-RTEMPAMP, RTT]`` 区间上的正弦斜坡）吸收冻结/融化
   释放的表观热容。融化分支随后在加热到 ``RTT`` 与融化期间消耗
   潜热之间重新分配时间步。

3. **雪-土壤地表热通量** ``PGSN``：雪底（处于新雪温 ``PTSN``）与
   土壤顶层（处于 ``PTSAM1M(:,1)``）之间的导热通量，经过雪热阻
   ``ZSNRES = 0.5*ZDSN/(RLAMICE*(PRSNM1M*ZHOICE)**RALAMSN)`` 与
   半个土壤顶层热阻 ``ZSOILRES = 1/RVLAMSK(8)`` 串联。
   ``PGSN`` 按雪瓦片比例 ``ZFRSN`` 归一化，因此它是格点平均通量，
   可作为来自积雪的地表热通量直接送入 ``SRFTS``。

符号约定
--------
``PGSN`` 为*向下为正*（热量从积雪流入土壤），与
:mod:`surf_pytorch.physics.soil` 中的 ``ZSURFL`` 约定一致。来自
``VDFDIFH1S`` 的感热和潜热通量遵循相同的向下为正约定。潜热贡献
在低植被积雪瓦片上为 ``RLSTT*PEVAP``（升华），在高植被下积雪瓦片
上为 ``RLVTT*(PEVAPTI-PEVAPSNW) + RLSTT*PEVAPSNW``（冠层下部分
升华，其余部分蒸发冠层截获的液态水）。

冷启动简化
----------
离线驱动程序对雪方案做如下冷启动设置：

* ``PEVAPSNW``（林下积雪蒸发）= 0
* ``PSSFC`` / ``PSSFL``（对流 / 大尺度降雪）= 固态降水穿透量；
  若不存在，则两者均为零（无降雪输入）。
* ``PTSFC`` / ``PTSFL``（对流 / 大尺度降雨穿透量）
  = 液态降水穿透量；用于雪-降雨截获收支。
* ``PHLICEM1M``（湖冰厚度）= 0（FLake 占位）。

所有这些量都在 :class:`SnowInputs` 上以显式的零默认值暴露，因此
调用方可以只构建最小输入（状态 + 通量 + 瓦片），让方案将其视为
冷启动处理。

索引约定
--------
``PFRTI`` / ``PAHFSTI`` / ``PEVAPTI`` / ``PSSRFLTI`` 在本移植中
采用 0 基索引。Fortran 瓦片 5（低植被上的积雪）映射到
:data:`surf_pytorch.config.TILE_LOW_VEG_SNOW`（4），瓦片 7（高植被
下的积雪）映射到 :data:`surf_pytorch.config.TILE_HIGH_VEG_SNOW`
（6）。瓦片索引由 ``ntile`` 保护，因此较小的瓦片维度（例如 4 瓦片
的测试板）不会引发错误。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from ..config import (
    TILE_WET_SKIN,
    TILE_LOW_VEG_DRY,
    TILE_LOW_VEG_SNOW,
    TILE_HIGH_VEG_DRY,
    TILE_HIGH_VEG_SNOW,
    TILE_BARE_SOIL,
    TILE_LAKE,
    SurfConfig,
)
from ..constants import ConstantBundle
from ._ieee import true_div
from ..tables.soil import SoilTable


# --------------------------------------------------------------------------- #
# SRFSN_LWIMP 使用的 Fortran 瓦片索引，映射到本移植的瓦片枚举。
# Fortran PFRTI(JL,3..8) -> Python pfrti[:, 2..7]；湖泊为 PFRTI(JL,9) -> 8。
# --------------------------------------------------------------------------- #
# 雪瓦片比例索引（Fortran 中为 1 基）。
_F90_TILE_SNOW_LOW = 5      # -> 移植版 TILE_LOW_VEG_SNOW (4)
_F90_TILE_SNOW_HIGH = 7     # -> 移植版 TILE_HIGH_VEG_SNOW (6)
# 降雪截获缩放的格点比例（Fortran 瓦片 3..8 +
# 若存在湖瓦片 9）。Fortran 瓦片 3 == 移植版 TILE_WET_SKIN (2)，
# 瓦片 8 == 移植版 TILE_BARE_SOIL (7)，因此移植索引为 2..7（湖泊另加 8）。
# 早期代码使用了 TILE_LOW_VEG_DRY（移植版 3 == F90 4），
# 这会悄悄地在求和中漏掉湿表皮瓦片 —— 由针对 SURF_REF_SNOW 的
# M6b ULP 比较所捕获。
_INTERCEPT_FRAC_LO = TILE_WET_SKIN           # 移植版 2 == F90 3
_INTERCEPT_FRAC_HI = TILE_BARE_SOIL          # 移植版 7 == F90 8


@dataclass
class SnowInputs:
    """HTESSEL 单层雪求解的全部输入。

    除特别注明外，每个张量形状均为 ``(Ngrid,)``。逐瓦片字段为
    ``(Ngrid, Ntile)``。命名与 ``SRFSN_LWIMP`` 的形参列表对应，
    便于逐项对照审查。

    冷启动说明：降雪 / 穿透降水 / 截获不匹配输入
    （``pssfc``、``pssfl``、``ptsfc``、``ptsfl``、``pevapsnw``）
    均默认为零，冷启动时可以保持为零。
    """

    # ---- t-1 时刻的雪状态 ----
    pssnm1m: torch.Tensor             # PSSNM1M  雪量 (SWE)      [kg/m^2]    (Ngrid,)
    ptsnm1m: torch.Tensor             # PTSNM1M  雪温            [K]         (Ngrid,)
    prsnm1m: torch.Tensor             # PRSNM1M  雪密度          [kg/m^3]    (Ngrid,)

    # ---- 土壤顶层温度（用于雪-土壤耦合）----
    ptsam1m_top: torch.Tensor         # PTSAM1M(:,1)  顶层土壤温度 [K]         (Ngrid,)

    # ---- 逐瓦片通量（来自 VDFDIFH1S / SURFSEB）----
    pfrti: torch.Tensor               # PFRTI    瓦片比例        [-]         (Ngrid, Ntile)
    pahfsti: torch.Tensor             # PAHFSTI  瓦片感热通量    [W/m^2]     (Ngrid, Ntile)
    pevapti: torch.Tensor             # PEVAPTI  瓦片水汽通量    [kg/m^2/s]  (Ngrid, Ntile)
    pssrflti: torch.Tensor            # PSSRFLTI 瓦片净短波辐射  [W/m^2]     (Ngrid, Ntile)
    pslrflti: torch.Tensor            # PSLRFLTI 瓦片净长波辐射  [W/m^2]     (Ngrid, Ntile)

    # ---- 逐格点辐射 ----
    pslrfl: torch.Tensor              # PSLRFL   净长波辐射      [W/m^2]     (Ngrid,)

    # ---- 降雪穿透量（逐格点）----
    pssfc: torch.Tensor               # PSSFC  对流降雪          [kg/m^2/s]  (Ngrid,)
    pssfl: torch.Tensor               # PSSFL  大尺度降雪        [kg/m^2/s]  (Ngrid,)

    # ---- 降雨穿透量（逐格点；用于雪-降雨截获）----
    ptsfc: torch.Tensor               # PTSFC  对流降雨          [kg/m^2/s]  (Ngrid,)
    ptsfl: torch.Tensor               # PTSFL  大尺度降雨        [kg/m^2/s]  (Ngrid,)

    # ---- 林下积雪蒸发不匹配量（逐格点）----
    pevapsnw: torch.Tensor            # PEVAPSNW  林下积雪蒸发   [kg/m^2/s]  (Ngrid,)

    # ---- 时间步长 ----
    ptmst: torch.Tensor               # PTMST  时间步长          [s]         scalar

    # ---- 湖冰（FLake 占位；冷启动时为零）----
    phlicem1m: Optional[torch.Tensor] = None   # PHLICEM1M 湖冰厚度 [m] (Ngrid,)


@dataclass
class SnowOutputs:
    """HTESSEL 单层雪求解的全部输出。"""

    # ---- t 时刻的雪状态 ----
    ptsn: torch.Tensor                # PTSN  新雪温             [K]         (Ngrid,)
    pgsn: torch.Tensor                # PGSN  由雪传入土壤的地表热通量 [W/m^2] (Ngrid,)

    # ---- 诊断量（对应 SRFSN_LWIMP 的局部变量）----
    psstar: torch.Tensor              # ZSSTAR  更新后的雪量 (SWE) [kg/m^2] (Ngrid,)
    plwc: torch.Tensor                # ZLWC   积雪液态水容量 [kg/m^2] (Ngrid,)
    zfrsn: torch.Tensor               # ZFRSN  雪瓦片比例（归一化因子）[-] (Ngrid,)

    # ---- 无雪掩码，暴露出来以便密度演变块能在 LLNOSNOW 点 ---- #
    # 将密度重置为 RHOMINSN（SRFSN_LWIMP 第 491 行）。
    llnosnow: Optional[torch.Tensor] = None   # LLNOSNOW  无雪掩码 (bool) (Ngrid,)

    # ---- 融雪率，暴露出来以便雪反照率老化块能按 PMSN > 0 ---- #
    # 划分融化/冷分支（SRFSN 第 354 行）。无融化点为零；仅在融化分支
    # 激活的点上为正。
    pmsn: Optional[torch.Tensor] = None       # PMSN  融雪率 [kg/m^2/s] (Ngrid,)
    pmsnint: Optional[torch.Tensor] = None    # ZPMSNINT 内部相变 [W/m^2] (Ngrid,)
    pemssn: Optional[torch.Tensor] = None     # PEMSSN 雪量截断不匹配量 [kg/m^2/s]
    ptsfcin: Optional[torch.Tensor] = None    # PTSFCIN 被雪截获的对流降雨 [kg/m^2/s]
    ptsflin: Optional[torch.Tensor] = None    # PTSFLIN 被雪截获的大尺度降雨 [kg/m^2/s]


def compute_snow(
    inp: SnowInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    differentiable: Optional[bool] = None,
) -> SnowOutputs:
    """运行 HTESSEL 单层雪求解（``SRFSN_LWIMP``）。

    Args
    ----
    inp:
        :class:`SnowInputs` —— 打包的雪状态、通量与掩码。
    soil_table:
        :class:`surf_pytorch.tables.soil.SoilTable` —— 提供雪常数
        （rhoice、rhoci、rlamsn、rlwcswe*、rtempamp、rfrsmall、
        rfrtiny、rdsmax）。
    consts:
        :class:`ConstantBundle` —— 提供 ``rtt``、``rlmlt``、``rlvtt``、
        ``rlstt`` 和 ``pi``。
    config:
        :class:`SurfConfig` —— ``use_flake`` 选择湖泊分支。
    differentiable:
        为本次调用覆盖 ``config.differentiable``。

    Returns
    -------
    :class:`SnowOutputs` —— 新雪温、雪-土壤地表热通量、更新后的雪量、
    液态水容量以及雪瓦片比例归一化因子。
    """
    diff = config.differentiable if differentiable is None else differentiable

    def _core() -> SnowOutputs:
        return _snow_core(inp, soil_table, consts, config)

    if not diff:
        with torch.no_grad():
            return _core()
    return _core()


def _snow_core(
    inp: SnowInputs,
    soil_table: SoilTable,
    consts: ConstantBundle,
    config: SurfConfig,
) -> SnowOutputs:
    """内部函数：假定 no_grad / enable_grad 的决定已经做出。"""
    ngrid = inp.pssnm1m.shape[0]
    ntile = inp.pfrti.shape[-1]
    dtype = inp.pssnm1m.dtype
    device = inp.pssnm1m.device

    # ------------------------------------------------------------------ #
    # 标量常数（按与 Fortran ASSOCIATE 相同的顺序读取）。
    # ------------------------------------------------------------------ #
    rtt = consts.rtt
    rlmlt = consts.rlmlt
    rlvtt = consts.rlvtt
    rlstt = consts.rlstt
    rpi = consts.pi

    sc = soil_table.consts
    rhoice = float(sc.rhoice)
    rlamsn = float(sc.rlamdsn_pow)
    rdsmax = float(sc.rdsmax)
    rhoci = float(sc.rhoci)
    rlamice = float(sc.rlamice)
    rfrsmall = float(sc.rfrsmall)
    rfrtiny = float(sc.rfrtiny)
    rlwcswea = float(sc.rlwcswea)
    rlwcsweb = float(sc.rlwcsweb)
    rlwcswec = float(sc.rlwcswec)
    rtempamp = float(sc.rtempamp)

    ptmst = inp.ptmst

    # 半个土壤顶层中导热的土壤热阻。
    # RVLAMSK(8) 是裸土（沙漠）表皮层导热率
    #（susveg 第 175 行：15 W/m^2/K）；SRFSN_LWIMP 使用其倒数。
    # 该值在雪的语境中与类型无关（只用到索引 8），
    # 因此我们硬编码沙漠值以匹配 Fortran。
    rvlamsk_8 = 15.0
    zsoilres = 1.0 / rvlamsk_8

    # ------------------------------------------------------------------ #
    # 计算用常数（Fortran 第 164-174 行）。
    # ------------------------------------------------------------------ #
    zhoice = 1.0 / rhoice
    zt0 = rtt
    ztmst = 1.0 / ptmst
    zitempamp = 1.0 / rtempamp

    # ZRDSNMAX / ZRDSNRESMAX（Fortran 第 173-174 行）：均为 1.0 m。
    zrdsnmax = 1.0
    zrdsnresmax = 1.0

    # ------------------------------------------------------------------ #
    # 逐格点设置（Fortran 第 189-213 行）。
    # ------------------------------------------------------------------ #
    # 雪瓦片比例 ZFRSN = max(PFRTI(5) + PFRTI(7), RFRTINY)。
    # LLNOSNOW 在 RFRTINY 截断之前的原始瓦片比例上判定；
    # 否则一个恰好无雪的列可能被数值下掩蔽，
    # 从而错过 Fortran 的无雪重置路径。
    pf_low_snow = _tile_col(inp.pfrti, TILE_LOW_VEG_SNOW, dtype, device)
    pf_high_snow = _tile_col(inp.pfrti, TILE_HIGH_VEG_SNOW, dtype, device)
    zfrsn_raw = pf_low_snow + pf_high_snow
    zfrsn = torch.clamp(zfrsn_raw, min=rfrtiny)

    # LLNOSNOW：雪瓦片比例低于 RFRSMALL。
    llnosnow = zfrsn_raw < rfrsmall

    # 降雪截获缩放：F90 对陆地点直接取 GRIDFRAC=1.0（srfsn_lwimp_mod.F90
    # 258-266），仅用于缩放 PSSFC/PSSFL/PTSFC/PTSFL；ZFRLDGP（陆地分数和）
    # 在本 cycle 是死代码。分数和会舍入到 1±1ulp 并在 ZWFLUXRF_RAW 处
    # 播下 1-ulp 种子，因此这里精确取 1.0。
    zgridfrac = torch.ones(inp.pfrti.shape[:-1], dtype=dtype, device=device)

    # 缩放后的降雪 / 穿透降水（第 208-211 行）。
    zssfc = zgridfrac * inp.pssfc
    zssfl = zgridfrac * inp.pssfl
    ztsfc = zgridfrac * inp.ptsfc
    ztsfl = zgridfrac * inp.ptsfl

    # ------------------------------------------------------------------ #
    # 瓦片列提取（对 Ngrid 向量化）。
    # ------------------------------------------------------------------ #
    # 雪面净热通量使用瓦片 5 和 7（1 基）。
    pf5 = pf_low_snow                                       # PFRTI(:,5)
    pf7 = pf_high_snow                                      # PFRTI(:,7)
    pahfsti_5 = _tile_col(inp.pahfsti, TILE_LOW_VEG_SNOW, dtype, device)
    pahfsti_7 = _tile_col(inp.pahfsti, TILE_HIGH_VEG_SNOW, dtype, device)
    pevapti_5 = _tile_col(inp.pevapti, TILE_LOW_VEG_SNOW, dtype, device)
    pevapti_7 = _tile_col(inp.pevapti, TILE_HIGH_VEG_SNOW, dtype, device)
    pssrflti_5 = _tile_col(inp.pssrflti, TILE_LOW_VEG_SNOW, dtype, device)
    pssrflti_7 = _tile_col(inp.pssrflti, TILE_HIGH_VEG_SNOW, dtype, device)
    # SRFSN_LWIMP 接收由 SURFTSTP 构建的瓦片化长波辐射场。
    pslrflti_5 = _tile_col(inp.pslrflti, TILE_LOW_VEG_SNOW, dtype, device)
    pslrflti_7 = _tile_col(inp.pslrflti, TILE_HIGH_VEG_SNOW, dtype, device)

    ptsnm1m = inp.ptsnm1m
    pssnm1m = inp.pssnm1m
    prsnm1m = inp.prsnm1m
    ptsam1m_top = inp.ptsam1m_top
    pevapsnw = inp.pevapsnw

    # ------------------------------------------------------------------ #
    # 主循环体，向量化（Fortran 第 215-331 行）。
    # ------------------------------------------------------------------ #
    # 默认输出（无雪分支的取值；有雪处会被覆盖）。
    zt0_t = torch.as_tensor(zt0, dtype=dtype, device=device)

    # --- 雪面净热通量（ZHFLUX）---------------------------------------- #
    # ZHFLUXP = PFRTI(5)*(PAHFSTI(5)+RLSTT*PEVAPTI(5))
    #         + PFRTI(7)*(PAHFSTI(7)+RLVTT*(PEVAPTI(7)-PEVAPSNW)
    #                     +RLSTT*PEVAPSNW)
    #         + PFRTI(5)*PSSRFLTI(5)
    #         + PFRTI(7)*PSSRFLTI(7)
    #         + PFRTI(5)*PSLRFLTI(5)
    #         + PFRTI(7)*PSLRFLTI(7)
    # ZHFLUX  = ZHFLUXP / ZFRSN。保持 Fortran 的运算顺序：
    # 乘以预先计算的倒数会改变末位比特。
    zhfluxp = (
        pf5 * (pahfsti_5 + rlstt * pevapti_5)
        + pf7 * (pahfsti_7 + rlvtt * (pevapti_7 - pevapsnw) + rlstt * pevapsnw)
        + pf5 * pssrflti_5
        + pf7 * pssrflti_7
        + pf5 * pslrflti_5
        + pf7 * pslrflti_7
    )
    zhflux = zhfluxp / zfrsn

    # --- 初步雪量 ZSSTAR ---------------------------------------------- #
    # ZSSTAR = PSSNM1M + PTMST*(ZSSFC+ZSSFL+PFRTI(5)*PEVAPTI(5)
    #                          +PFRTI(7)*PEVAPSNW)
    zsstar_raw = pssnm1m + ptmst * (
        zssfc + zssfl + pf5 * pevapti_5 + pf7 * pevapsnw
    )
    zsstar = torch.clamp(zsstar_raw, min=0.0)

    # Active offline configuration: LESNML=.FALSE., LESN09=.TRUE.
    # SRFSN_LWIMP sends the negative pre-clipping snow mass as PEMSSN
    # to SRFWEXC_VG, then clips the snow store to zero.
    pemssn = torch.where(
        (~llnosnow) & (zsstar_raw < 0.0),
        zsstar_raw / ptmst,
        torch.zeros_like(zsstar_raw),
    )

    # --- Snow liquid water (ZLWC) ------------------------------------- #
    # ZDSNR  = PSSNM1M / (PRSNM1M * ZFRSN)  -- snow depth (real)
    # ZCONSLWC = max(0, (RLWCSWEA - PRSNM1M) / RLWCSWEA)
    # ZLWC   = PSSNM1M * (RLWCSWEB + (RLWCSWEC - RLWCSWEB)*ZCONSLWC)
    # then rescale by min(ZDSNR, ZRDSNMAX)/ZDSNR.
    zdsnr = pssnm1m / (prsnm1m * zfrsn)
    zconslwc = torch.clamp((rlwcswea - prsnm1m) / rlwcswea, min=0.0)
    zlwc = pssnm1m * (rlwcsweb + (rlwcswec - rlwcsweb) * zconslwc)
    zdsn_rp = torch.minimum(zdsnr, torch.as_tensor(zrdsnmax, dtype=dtype, device=device))
    # Guard divide-by-zero: where ZDSNR is zero (no snow) the snow
    # branch is not taken, but the expression must still be finite for
    # autograd.  Use a safe divide and zero out the no-snow points later.
    zdsnr_safe = torch.where(
        zdsnr != 0.0, zdsnr, torch.ones_like(zdsnr)
    )
    zlwc = zlwc * zdsn_rp / zdsnr_safe

    # --- Analytical melt-transition functions ZFUNC / ZDFUNC ---------- #
    # The branch on PTSNM1M selects three windows:
    #   * T < ZT0 - 0.5*RTEMPAMP  -> both 0 (cold snow, no liquid water)
    #   * ZT0 - 0.5*RTEMPAMP <= T < ZT0 -> sinusoidal ramp
    #   * T >= ZT0               -> ZFUNC=0, ZDFUNC=RPI*ZITEMPAMP (melting)
    cold = ptsnm1m < (zt0 - 0.5 * rtempamp)
    melting = ptsnm1m >= zt0
    ramp = ~(cold | melting)

    # Ramp branch:
    #   ZFUNC  = 1 + sin(RPI*(T - ZT0)*ZITEMPAMP)
    #   ZDFUNC = cos(RPI*(T - ZT0)*ZITEMPAMP) * RPI * ZITEMPAMP
    arg = rpi * (ptsnm1m - zt0) * zitempamp
    zfunc_ramp = 1.0 + torch.sin(arg)
    zdfunc_ramp = torch.cos(arg) * rpi * zitempamp

    zfunc = torch.zeros_like(ptsnm1m)
    zdfunc = torch.zeros_like(ptsnm1m)
    zfunc = torch.where(ramp, zfunc_ramp, zfunc)
    zdfunc = torch.where(
        ramp, zdfunc_ramp,
        torch.where(
            melting,
            torch.full_like(ptsnm1m, rpi * zitempamp),
            zdfunc,                                  # cold -> 0
        ),
    )

    # --- Terms for the implicit temperature equation ------------------ #
    # ZCONSB     = RLMLT*ZLWC*ZDFUNC
    # ZCONSAMAX  = (PRSNM1M*RHOCI*ZRDSNMAX)*ZHOICE
    # ZCOND1     = (PSSNM1M*RHOCI*ZHOICE)/ZFRSN
    # ZCONSA     = min(ZCOND1, ZCONSAMAX)
    # ZCONS2     = PTMST / (ZCONSA + ZCONSB)
    zconsb = rlmlt * zlwc * zdfunc
    zconsamax = (prsnm1m * rhoci * zrdsnmax) * zhoice
    zcond1 = (pssnm1m * rhoci * zhoice) / zfrsn
    zconsa = torch.minimum(zcond1, zconsamax)
    zcons_denom = zconsa + zconsb
    zcons_denom_safe = torch.where(
        zcons_denom != 0.0, zcons_denom, torch.ones_like(zcons_denom)
    )
    zcons2 = ptmst / zcons_denom_safe

    # --- Real snow depth ZDSN and snow-soil resistance ZRS ------------ #
    # ZDSN    = min(PSSNM1M/(PRSNM1M*ZFRSN), ZRDSNRESMAX)
    # ZSNRES  = 0.5*ZDSN / (RLAMICE*(PRSNM1M*ZHOICE)**RALAMSN)
    # ZRS     = 1 / (ZSNRES + ZSOILRES)
    zdsn = torch.minimum(
        pssnm1m / (prsnm1m * zfrsn),
        torch.as_tensor(zrdsnresmax, dtype=dtype, device=device),
    )
    zsnres = 0.5 * zdsn / (
        rlamice * (prsnm1m * zhoice) ** rlamsn
    )
    zrs = 1.0 / (zsnres + zsoilres)

    # --- Throughfall interception (lines 286-299) --------------------- #
    # ZWFLUXRF = (PFRTI(5)+PFRTI(7))*(ZTSFC+ZTSFL)     -- mass advected
    # ZCOND3   = ZWFLUXRF*PTMST
    # ZCOND4   = ZLWC*(1 - ZFUNC)
    # ZWFLUXRF = min(ZCOND3, ZCOND4)
    # ZCOND5   = ZCONSA*(ZT0 - PTSNM1M)*ZTMST
    # This is the SRFSN_LWIMP formulation.
    # ZCOND6   = RLMLT*ZWFLUXRF*ZTMST
    # ZHFLUXRF1= min(ZCOND5, ZCOND6); ZHFLUXRF1 = max(ZHFLUXRF1, 0)
    # ZWFLUXRF = ZHFLUXRF1 / RLMLT                     -- intercepted mass
    # ZSSTAR   = ZSSTAR + ZWFLUXRF*PTMST
    zwfluxrf_raw = (pf5 + pf7) * (ztsfc + ztsfl)
    zcond3 = zwfluxrf_raw * ptmst
    zcond4 = zlwc * (1.0 - zfunc)
    zwfluxrf = torch.minimum(zcond3, zcond4)
    zcond5 = zconsa * (zt0 - ptsnm1m) * ztmst
    zcond6 = rlmlt * zwfluxrf * ztmst
    zhfluxrf1 = torch.minimum(zcond5, zcond6)
    zhfluxrf1 = torch.clamp(zhfluxrf1, min=0.0)
    zwfluxrf = zhfluxrf1 / rlmlt
    zsstar = zsstar + zwfluxrf * ptmst

    # SRFSN_LWIMP returns the liquid throughfall captured by the snowpack.
    # SURFTSTP subtracts these rates before calling SRFWEXC_VG, preventing
    # rain that froze in the snow from entering the soil-water budget.
    zptsf = inp.ptsfc + inp.ptsfl
    valid_rain = (~llnosnow) & (zwfluxrf > 0.0) & (zptsf > 0.0)
    # ``torch.where`` evaluates both branches.  A snow-free column has
    # zptsf=0, so the masked direct division is harmless in forward mode
    # but produces a NaN DivBackward gradient. Keep the exact Fortran
    # denominator on the active path and use a neutral denominator elsewhere.
    zptsf_safe = torch.where(valid_rain, zptsf, torch.ones_like(zptsf))
    ptsfcin = torch.where(
        valid_rain,
        torch.clamp(zwfluxrf * inp.ptsfc / zptsf_safe, min=0.0),
        torch.zeros_like(zwfluxrf),
    )
    ptsflin = torch.where(
        valid_rain,
        torch.clamp(zwfluxrf * inp.ptsfl / zptsf_safe, min=0.0),
        torch.zeros_like(zwfluxrf),
    )

    # --- Solve implicit temperature equation -------------------------- #
    # Keep the Fortran division order.  Computing a reciprocal first and
    # multiplying is algebraically equivalent but drifts by ULPs in PGSN,
    # which can later alter a melt/no-melt threshold.
    ztstar = (
        ptsnm1m + zcons2 * (zhflux + zhfluxrf1 + ptsam1m_top * zrs)
    ) / (1.0 + zcons2 * zrs)

    # --- Branches (no-melt vs melt) ----------------------------------- #
    no_melt = ztstar <= zt0
    melt = ~no_melt

    # No-melt branch (lines 304-307):
    #   PTSN = ZTSTAR
    #   PGSN = (ZTSTAR - PTSAM1M(1))*ZRS
    ptsn_no_melt = ztstar
    pgsn_no_melt = (ztstar - ptsam1m_top) * zrs

    # Melt branch (lines 308-327).
    # ZDTHI = 1/(ZHFLUX + ZHFLUXRF1 - (ZT0 - PTSAM1M(1))*ZRS)
    # ZDTH  = ((ZT0 - PTSNM1M)*(ZCONSA + ZCONSB)) * ZDTHI
    # ZDTM  = PTMST - ZDTH
    # ZGSN  = (ZT0 - PTSAM1M(1))*ZRS
    # ZMSN  = (ZHFLUX + ZHFLUXRF1 - ZGSN)/RLMLT
    # ZSSTAR1 = ZSSTAR - ZDTM*ZMSN*ZFRSN
    zdthi_denom = zhflux + zhfluxrf1 - (zt0 - ptsam1m_top) * zrs
    zepsilon = 100.0 * torch.finfo(dtype).tiny
    zdthi_safe = torch.where(
        zdthi_denom < 0.0,
        -torch.clamp(torch.abs(zdthi_denom), min=zepsilon),
        torch.clamp(torch.abs(zdthi_denom), min=zepsilon),
    )
    # srfsn_lwimp 418: ZDTH is a true division by SIGN(MAX(eps,|w|),w);
    # a reciprocal-multiply here differs in the last bit at melt onset.
    zdth = true_div((zt0 - ptsnm1m) * (zconsa + zconsb), zdthi_safe)
    zdtm = ptmst - zdth
    zgsn_melt = (zt0 - ptsam1m_top) * zrs
    zmsn = (zhflux + zhfluxrf1 - zgsn_melt) / rlmlt
    zsstar1 = zsstar - zdtm * zmsn * zfrsn

    # Partial-melt branch (ZSSTAR1 >= 0): snow remains.
    #   PTSN = ZT0; PGSN = (ZT0 - PTSAM1M(1))*ZRS
    ptsn_partial = zt0_t
    pgsn_partial = zgsn_melt

    # Full-melt branch (ZSSTAR1 < 0): all snow melted.
    #   PTSN = ZT0
    #   ZDTM = ZSSTAR/(ZMSN*ZFRSN)
    #   PGSN = (((ZT0-PTSAM1M(1))*ZRS)*(ZDTM+ZDTH)
    #           + (ZHFLUX+ZHFLUXRF1)*(PTMST-ZDTM-ZDTH)) * ZTMST
    zdtm_full = torch.where(
        zmsn * zfrsn != 0.0,
        zsstar / (zmsn * zfrsn),
        torch.zeros_like(zsstar),
    )
    pgsn_full = (
        ((zt0 - ptsam1m_top) * zrs) * (zdtm_full + zdth)
        + (zhflux + zhfluxrf1) * (ptmst - zdtm_full - zdth)
    ) * ztmst

    full_melt = zsstar1 < 0.0
    zsstar_melt = torch.where(full_melt, torch.zeros_like(zsstar1), zsstar1)
    pgsn_melt = torch.where(full_melt, pgsn_full, pgsn_partial)
    ptsn_melt = torch.full_like(ptsnm1m, zt0)

    # Combine the melt / no-melt branches.
    ptsn_with_snow = torch.where(no_melt, ptsn_no_melt, ptsn_melt)
    pgsn_with_snow = torch.where(no_melt, pgsn_no_melt, pgsn_melt)
    zsstar_with_snow = torch.where(no_melt, zsstar, zsstar_melt)

    # ------------------------------------------------------------------ #
    # Apply the no-snow branch (Fortran SRFSN_LWIMP lines 276-281).
    # LEWBSOILFIX retains the residual snow-tile water exchange while the
    # cover fraction is below RFRSMALL; omitting it creates a mass jump at
    # the no-snow/snow transition.
    #   PTSN   = ZT0
    #   PGSN   = 0
    #   ZLWC   = 0
    # ------------------------------------------------------------------ #
    zsstar_nosnow = pssnm1m + ptmst * (zssfc + zssfl)
    if getattr(soil_table.consts, "lewbsoilfix", True):
        no_snow_exchange = torch.zeros_like(zsstar_nosnow)
        if ntile > TILE_LOW_VEG_SNOW:
            no_snow_exchange = no_snow_exchange + (
                inp.pfrti[:, TILE_LOW_VEG_SNOW]
                * inp.pevapti[:, TILE_LOW_VEG_SNOW]
            )
        if ntile > TILE_HIGH_VEG_SNOW:
            no_snow_exchange = no_snow_exchange + (
                inp.pfrti[:, TILE_HIGH_VEG_SNOW] * inp.pevapsnw
            )
        zsstar_nosnow = zsstar_nosnow + ptmst * no_snow_exchange
    ptsn = torch.where(llnosnow, zt0_t, ptsn_with_snow)
    pgsn = torch.where(
        llnosnow,
        torch.zeros_like(pgsn_with_snow),
        pgsn_with_snow,
    )
    zsstar_out = torch.where(llnosnow, zsstar_nosnow, zsstar_with_snow)
    zlwc_out = torch.where(
        llnosnow, torch.zeros_like(zlwc), zlwc
    )

    # The Fortran debug writer (srfsn_lwimp_mod.F90:533) snapshots PGSN
    # and PMSN *before* section 5 normalises them in place (lines
    # 643-644: PGSN=ZFRSN*PGSN, PMSN=ZFRSN*PMSN).  Keep these
    # pre-normalisation values for the CSV; the grid-square-scaled
    # fluxes are applied after the debug call below because SRFT and
    # SRFWEXC_VG consume them in grid-square units.
    pmsn_partial = zmsn * zdtm * ztmst
    pmsn_full = zmsn * zdtm_full * ztmst
    pmsn_melt = torch.where(full_melt, pmsn_full, pmsn_partial)
    pmsn_out = torch.where(melt, pmsn_melt, torch.zeros_like(zmsn))
    # SRFSN_LWIMP's LLNOSNOW branch explicitly resets both PGSN and PMSN.
    # PMSN is consumed by the soil-water budget in the same surface step;
    # retaining a provisional melt flux here injects spurious water when
    # the snow tile drops below RFRSMALL.
    pmsn_out = torch.where(llnosnow, torch.zeros_like(pmsn_out), pmsn_out)
    pmsnint = zconsb * zfrsn * (ptsn - ptsnm1m) / ptmst
    pmsnint = torch.where(llnosnow, torch.zeros_like(pmsnint), pmsnint)

    _append_snow_debug(
        pssnm1m=pssnm1m,
        ptsnm1m=ptsnm1m,
        prsnm1m=prsnm1m,
        ptsam1m_top=ptsam1m_top,
        pf5=pf5,
        pf7=pf7,
        pahfsti_5=pahfsti_5,
        pahfsti_7=pahfsti_7,
        pevapti_5=pevapti_5,
        pevapti_7=pevapti_7,
        pssrflti_5=pssrflti_5,
        pssrflti_7=pssrflti_7,
        pslrflti_5=pslrflti_5,
        pslrflti_7=pslrflti_7,
        pssfc=inp.pssfc,
        pssfl=inp.pssfl,
        ptsfc=inp.ptsfc,
        ptsfl=inp.ptsfl,
        pevapsnw=pevapsnw,
        zfrsn=zfrsn,
        llnosnow=llnosnow,
        zhflux=zhflux,
        zsstar_raw=zsstar_raw,
        zsstar=zsstar,
        zdsnr=zdsnr,
        zlwc=zlwc,
        zfunc=zfunc,
        zdfunc=zdfunc,
        zconsa=zconsa,
        zconsb=zconsb,
        zcons2=zcons2,
        zdsn=zdsn,
        zrs=zrs,
        zwfluxrf_raw=zwfluxrf_raw,
        zcond3=zcond3,
        zcond4=zcond4,
        zcond5=zcond5,
        zcond6=zcond6,
        zwfluxrf=zwfluxrf,
        zhfluxrf1=zhfluxrf1,
        ztstar=ztstar,
        no_melt=no_melt,
        zdth=zdth,
        zdtm=zdtm,
        zgsn_melt=zgsn_melt,
        zmsn=zmsn,
        zsstar1=zsstar1,
        full_melt=full_melt,
        ptsn=ptsn,
        pssn=zsstar_out,
        pgsn=pgsn,
        pmsn=pmsn_out,
        pmsnint=pmsnint,
    )

    # Section 5 (srfsn_lwimp_mod.F90:643-644): normalise the returned
    # fluxes to the grid square after the debug snapshot.
    pgsn = zfrsn * pgsn
    pmsn_out = zfrsn * pmsn_out

    return SnowOutputs(
        ptsn=ptsn,
        pgsn=pgsn,
        psstar=zsstar_out,
        plwc=zlwc_out,
        zfrsn=zfrsn,
        llnosnow=llnosnow,
        pmsn=pmsn_out,
        pmsnint=pmsnint,
        pemssn=pemssn,
        ptsfcin=ptsfcin,
        ptsflin=ptsflin,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _append_snow_debug(**cols: torch.Tensor) -> None:
    path = os.environ.get("SURF_TORCH_SNOW_DEBUG_CSV")
    if not path:
        return

    step = getattr(_append_snow_debug, "_step", 0)
    setattr(_append_snow_debug, "_step", step + 1)

    out = Path(path)
    names = list(cols)
    header = ["step", "jl"] + names
    write_header = not out.exists()

    def item(t: torch.Tensor, i: int) -> float:
        v = t.detach().cpu().reshape(-1)[i]
        if v.dtype == torch.bool:
            return int(v.item())
        return float(v.item())

    ngrid = next(iter(cols.values())).detach().reshape(-1).shape[0]
    column_text = os.environ.get("SURF_TORCH_SNOW_DEBUG_COLUMN", "")
    indices = range(ngrid)
    if column_text:
        column = int(column_text) - 1
        if not 0 <= column < ngrid:
            raise ValueError(f"SURF_TORCH_SNOW_DEBUG_COLUMN out of range: {column + 1}")
        indices = (column,)
    with out.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for i in indices:
            writer.writerow([step, i + 1] + [item(cols[name], i) for name in names])


def _tile_col(
    pfrti: torch.Tensor, idx: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """若瓦片轴足够宽，则返回 ``pfrti[:, idx]``。

    若瓦片维度窄于 ``idx + 1``（例如省略了雪瓦片的 4 瓦片测试板），
    则回退为零。这对应于
    :meth:`LandSurface._update_soil_temperature`
    中使用的 ``ntile > TILE_*`` 保护。
    """
    ntile = pfrti.shape[-1]
    if ntile > idx:
        return pfrti[..., idx]
    return torch.zeros(pfrti.shape[:-1], dtype=dtype, device=device)


def _interception_gridfrac(
    pfrti: torch.Tensor, use_flake: bool, dtype: torch.dtype, device
) -> torch.Tensor:
    """降雪截获缩放的格点比例（srfsn_lwimp_mod.F90 258-277）。

    F90 对陆地点直接赋 GRIDFRAC=1.0（海洋点 0.0），仅用于缩放
    PSSFC/PSSFL/PTSFC/PTSFL。陆地分数和 ZFRLDGP 在本 cycle 未被
    使用；按分数和计算会引入 1-ulp 舍入差。
    """
    return torch.ones(pfrti.shape[:-1], dtype=dtype, device=device)

__all__ = [
    "SnowInputs",
    "SnowOutputs",
    "compute_snow",
]
