"""地表能量平衡（Penman-Monteith 表皮温度消元）。

直接翻译自 ``surf/module/surfseb_ctl_mod.F90``（J. Polcher / P. Best
/ A. Beljaars 闭合方案）。这是陆面方案的核心：它利用围绕 t-1 时刻
表皮温度对 qsat 的线性化，从地表能量平衡中消去表皮温度，从而给出
新表皮温度和 tile 平均湍流通量的闭式解。

闭合方案概要
------------
最低模式层的干静力能 ``Sl`` 与比湿 ``Ql`` 通过垂直扩散三对角消元
（:mod:`surf_pytorch.physics.diffusion` 中的*上行*扫描，M4）与地表
耦合。该扫描产生线性关系::

    Sl = ASL*Js + BSL
    Ql = AQL*Jq + BQL

其中 ``Js`` 和 ``Jq`` 分别为干静力能通量和水汽通量。每个 tile 有其
自身的 SEB::

    -PSSRFL -PSLRFL + PSTR + PG0 + PSSH + PSLH = 0
    PSSH = Js - cp*Tsk*delta*Jq                （感热）
    PSLH = lambda * Jq                          （潜热，lambda=Lv/Ls）
    PG0  = lambda_skin * (Tsk - Tsrf)           （地面热通量）
    PSTR = PSLRFL + dLW/dT * (Tsk - Tsk_rad)    （线性化长波）

表皮温度通过 ``qsk(Tsk) ~ qsk_t1 + dqsdT_t1*(Tsk - Tsk_t1)`` 在 qsat
中线性化，因此整个 SEB 对 Tsk、Sl 和 Ql 均为线性。逐 tile 消去 Tsk
得到逐 tile 系数 ``ZD{JS,JQ,DSS}*``（Fortran 第 4 节）；tile 平均
并对 ``Sl,Ql`` 作 2x2 求解使系统闭合（第 5-6 节）；随后得到逐 tile
通量和表皮温度（第 7 节）。

符号约定（IFS 地表软件包，而非微气象学）
-------------------------------------------
本例程强制执行的 SEB 闭合为::

    PSSRFL + PSTR + PSSH + PSLH - PG0 = 0   （逐 tile）

即全部四个地表通量均采用 **正 = 向下**（进入地表）的符号，且
``PG0``（地面热通量）为 **正 = 向下** 进入地面：

* ``PSSRFL``、``PSTR``（辐射）：**正 = 向下**。``PSTR`` 为净热辐射
  = ``PSLRFL + dLW/dT*(Tsk - Tskrad)``，``PSLRFL`` 同样以向下为正，
  因此冷却地表（发射多于吸收）有 ``PSLRFL < 0``。
* ``PSSH``（感热）：**正 = 向下**（大气加热地表）。注意这与微气象学
  符号约定*相反*；``PSSH = PJS - cp*Tsk*delta*PJQ``，其中 ``PJS``
  为指向地表的干静力能通量，本身已携带向下符号。
* ``PSLH``（潜热）：**正 = 向下**（凝结加热地表）。同样与微气象学
  符号相反。
* ``PG0``（地面热通量）：**正 = 向下**，进入地面（自上方受热的地表
  有 ``PG0 > 0``）。

这是 SURFSEB_CTL 的*输出*约定（见 ``surfseb_ctl_mod.F90`` 第 467-470
行）；*消元*使用相反的内部符号（ZDSS4 中的 ``-PSSRFL``、``-PSLRFL``、
``+ZLAMSK*PTSRF``），经整理后与上述约定一致。本移植逐字复现 Fortran
消元并继承相同的输出约定——所有量均以向下为正。

内存布局
------
张量为 ``(Ngrid, Ntile)``，tile 轴位于最后一维。2x2 平均求解对 tile
轴做归约；输出 ``sl`` / ``ql`` 为 ``(Ngrid,)``。

tile 索引约定
----------------------
本移植保留 IFS 的 1 基 tile 语义，但张量寻址采用 0 基（见
:mod:`surf_pytorch.config`）。Fortran JT=2（海冰）对应本移植的
``TILE_ICE=1``；Fortran JT=5（低植被上的雪）对应本移植的
``TILE_LOW_VEG_SNOW=4``。潜热选择器 ``RLSTT``（升华）用于海冰和
低植被-雪 tile——偏移在此处施加，而非在调用方。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import (
    TILE_ICE,
    TILE_LOW_VEG_SNOW,
    TILE_LAKE,
    SurfConfig,
)
from ..constants import ConstantBundle
from .vlamsk import vlamsk, LARGE


# --------------------------------------------------------------------------- #
# SEB hard-wired constants (surfseb_ctl_mod.F90 lines 209-216)
# --------------------------------------------------------------------------- #
# ZDELTA == RVTMP2 (moist cp correction) -- the offline driver sets it to
# 0.0 via SUPHEC, so PSSH = Js by construction.  We keep the symbolic
# form so a future cp-moist option is a one-line change.
# (consumed from consts.rvtmp2)

# Large number that pins skin T to the sub-surface T (ocean, etc.).
_LARGE_SEB = 1.0e10


@dataclass
class SebInputs:
    """SURFSEB_CTL 的全部逐（网格，tile）输入。

    为清晰起见分组；:func:`surf_seb` 调用会解包这些输入。除名称以
    ``_grid`` 结尾（``(Ngrid,)``）或为标量场外，每个张量均为
    ``(Ngrid, Ntile)``。
    """

    # Skin reservoir at t-1 (linearised in qsat).
    sskm1m: torch.Tensor          # dry static energy of skin    [J/kg]
    tskm1m: torch.Tensor          # skin temperature at t-1      [K]
    qskm1m: torch.Tensor          # saturation q at tskm1m       [kg/kg]
    dqsdT: torch.Tensor           # dqsat/dT at tskm1m           [kg/kg/K]

    # Turbulent exchange: rho * Cd * |U| for heat and moisture.
    rhochu: torch.Tensor          # rho*Ch*|U|                   [kg/m^2/s]
    rhocqu: torch.Tensor          # rho*Cq*|U|                   [kg/m^2/s]

    # Moisture-flux closure coefficients (alpha_l * ql + alpha_s * qs).
    alphal: torch.Tensor          # multiplier of ql             [-]
    alphas: torch.Tensor          # multiplier of qs             [-]

    # Radiation and ground coupling.
    ssrfl: torch.Tensor           # net SW at the surface        [W/m^2]
    slrfl_grid: torch.Tensor      # net LW at the surface        [W/m^2]
    tskrad_grid: torch.Tensor     # mean skin T at rad time      [K]
    emis_grid: torch.Tensor       # surface emissivity           [-]

    # Lower boundary: sub-surface temperature and skin conductivity.
    tsrf: torch.Tensor            # surface T below skin         [K]
    lamsk: torch.Tensor           # skin layer conductivity      [W/m^2/K]

    # Tile fractions and dominant veg types (for lambda Lv/Ls selector).
    frti: torch.Tensor            # tile fractions               [-]
    ktvl_grid: torch.Tensor       # dominant low veg type (long)
    ktvh_grid: torch.Tensor       # dominant high veg type (long)

    # Snow / sea-ice / lake descriptors (used by lambda selector and
    # by the lake Lv/Ls split).
    snowm_grid: Optional[torch.Tensor] = None   # snow water eq. [kg/m^2]
    rsn_grid: Optional[torch.Tensor] = None     # snow density   [kg/m^3]
    thkice_grid: Optional[torch.Tensor] = None  # sea-ice thickness [m]
    sntice_grid: Optional[torch.Tensor] = None  # snow depth on sea ice
    hlice_grid: Optional[torch.Tensor] = None   # lake ice thickness [m]

    # Coupling coefficients from the upward diffusion sweep.
    asl_grid: torch.Tensor = None     # Sl = asl*Js + bsl
    bsl_grid: torch.Tensor = None
    aql_grid: torch.Tensor = None     # Ql = aql*Jq + bql
    bql_grid: torch.Tensor = None

    # Time step.
    ptmst: torch.Tensor = None        # scalar / (Ngrid,)

    # Feature flags.
    lelwdd: bool = True               # LW derivative form
    lelwtl: bool = False              # recompute tiled LW around mean Tsk
    lnemolimthk: bool = False         # NEMO sea-ice thickness source


@dataclass
class SebOutputs:
    """SURFSEB_CTL 的全部逐（网格，tile）输出。"""

    pjs: torch.Tensor           # flux of dry static energy   [W/m^2]
    pjq: torch.Tensor           # moisture flux               [kg/m^2/s]
    pssk: torch.Tensor          # new dry static energy       [J/kg]
    ptsk: torch.Tensor          # new skin temperature        [K]
    pssh: torch.Tensor          # sensible heat flux          [W/m^2]
    pslh: torch.Tensor          # latent heat flux            [W/m^2]
    pstr: torch.Tensor          # net thermal radiation       [W/m^2]
    pg0: torch.Tensor           # ground heat flux            [W/m^2]

    psl: torch.Tensor           # new lowest-level dry static energy [J/kg]
    pql: torch.Tensor           # new lowest-level specific humidity [kg/kg]


def surf_seb(
    inp: SebInputs,
    consts: ConstantBundle,
    config: SurfConfig,
    *,
    differentiable: Optional[bool] = None,
) -> SebOutputs:
    """运行 Penman-Monteith 地表能量平衡。

    Args
    ----
    inp:
        :class:`SebInputs` —— 打包的逐（网格，tile）与逐网格输入。
    consts:
        :class:`ConstantBundle` —— RCPD、RLSTT、RLVTT、RSIGMA、RTT、RVTMP2。
    config:
        :class:`SurfConfig` —— ``n_tile`` 选择哪些 tile 列被激活
        （潜热选择器将 JT=海冰/低植被-雪切换为升华，将 JT=9/湖泊切换
        为 Lv/Ls 混合）。
    differentiable:
        为本次调用覆盖 ``config.differentiable``。默认遵循 config 标志。

    Returns
    -------
    :class:`SebOutputs` —— 逐 tile 通量与表皮温度，以及网格平均的
    最低层干静力能和比湿。
    """
    diff = config.differentiable if differentiable is None else differentiable

    def _core() -> SebOutputs:
        return _seb_core(inp, consts, config)

    if not diff:
        with torch.no_grad():
            return _core()
    return _core()


def _seb_core(inp: SebInputs, consts: ConstantBundle, config: SurfConfig) -> SebOutputs:
    """内部：假定 no_grad / enable_grad 的决策已完成。"""
    ngrid, ntile = inp.sskm1m.shape

    # Promote per-grid inputs to per-(grid,tile) for vectorised maths.
    def _expand_grid(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(-1).expand(ngrid, ntile)

    slrfl = _expand_grid(inp.slrfl_grid)
    tskrad = _expand_grid(inp.tskrad_grid)
    emis = _expand_grid(inp.emis_grid)
    asl = _expand_grid(inp.asl_grid)
    bsl = _expand_grid(inp.bsl_grid)
    aql = _expand_grid(inp.aql_grid)
    bql = _expand_grid(inp.bql_grid)

    # ------------------------------------------------------------------ #
    # Section 1 -- lake ice mask (only matters if lake tile is active).
    # ------------------------------------------------------------------ #
    if ntile > TILE_LAKE and inp.hlice_grid is not None:
        rh_ice_min = 1.0e-9          # SUSFLAKE line 128
        lake_ice = (inp.hlice_grid > rh_ice_min).to(inp.sskm1m.dtype)
        lake_water = 1.0 - lake_ice
    else:
        # Sentinel: no lake tile -> all-water (the lake Lv/Ls branch is
        # never reached because ntile <= TILE_LAKE).
        lake_water = torch.ones(ngrid, dtype=inp.sskm1m.dtype,
                                device=inp.sskm1m.device)
        lake_ice = torch.zeros_like(lake_water)

    # ------------------------------------------------------------------ #
    # Section 2 -- dLW/dT (tile-independent).
    # ------------------------------------------------------------------ #
    if inp.lelwdd:
        # Linearise LW around the radiation time-level skin T.
        dlwdt_grid = (-4.0 * inp.emis_grid * consts.rsigma) * (
            (inp.tskrad_grid * inp.tskrad_grid) * inp.tskrad_grid)
    else:
        # Alternative: derive dLW/dT from the supplied net LW.
        dlwdt_grid = 4.0 * (inp.slrfl_grid / inp.tskrad_grid)
    dlwdt = _expand_grid(dlwdt_grid)

    # ------------------------------------------------------------------ #
    # Section 3 -- Js / Jq coefficients in (Sl, Ql, Ssk) (Fortran CJ*).
    # ------------------------------------------------------------------ #
    # ZICPTM1 = TSKM1M / SSKM1M  (inverse of cp*T mapping).  Avoided
    # dividing by SSKM1M where it is zero -- the ocean tile sets
    # SSKM1M = cp*Tsrf so this is well defined there.
    icptm1 = inp.tskm1m / inp.sskm1m
    cjs1 = inp.rhochu
    cjs3 = -inp.rhochu
    cjq2 = inp.rhocqu * inp.alphal
    cjq3 = -inp.rhocqu * inp.alphas * inp.dqsdT * icptm1
    cjq4 = -inp.rhocqu * inp.alphas * (
        inp.qskm1m - inp.dqsdT * inp.tskm1m
    )

    # ------------------------------------------------------------------ #
    # Section 4 -- eliminate Ssk via SEB -> coefficients in (Sl, Ql).
    # ------------------------------------------------------------------ #
    # Latent heat per tile: sublimation (RLSTT) for sea ice / low-veg
    # snow, vaporisation (RLVTT) elsewhere; lake tile is a Lv/Ls mix
    # weighted by lake water/ice mask.  Build per-tile lambda.
    lam = _tile_latent_heat(ngrid, ntile, inp.sskm1m.dtype,
                            inp.sskm1m.device, consts,
                            lake_water, lake_ice)

    # Solar transmission through vegetation per tile (ZFRSR).  Tiles
    # 2/3 (wet skin, low-veg dry) reduce SW by RVTRSR(1)=0.05 and tiles
    # 5/6 (high-veg dry, high-veg snow) by RVTRSR(3)=0.03; the rest see
    # the full net SW.  See :func:`_tile_solar_transmission` for the
    # tile-index mapping (0-based, mirroring surfseb_ctl_mod.F90 CASEs).
    frsr = _tile_solar_transmission(ngrid, ntile, inp.sskm1m.dtype,
                                    inp.sskm1m.device)

    delta = consts.rvtmp2
    coef1 = lam - consts.cpd * inp.tskm1m * delta

    # ZIZZ = 1/ZZ where ZZ = (dlwdt - lamsk)*icptm1 + cjs3 + coef1*cjq3
    zz = (dlwdt - inp.lamsk) * icptm1 + cjs3 + coef1 * cjq3
    zizz = 1.0 / zz

    dss1 = -cjs1 * zizz
    dss2 = -coef1 * cjq2 * zizz
    dss4 = (
        -inp.ssrfl * frsr - slrfl + dlwdt * tskrad
        - coef1 * cjq4 - inp.lamsk * inp.tsrf
    ) * zizz

    djs1 = cjs1 + cjs3 * dss1
    djs2 = cjs3 * dss2
    djs4 = cjs3 * dss4

    djq1 = cjq3 * dss1
    djq2 = cjq2 + cjq3 * dss2
    djq4 = cjq3 * dss4 + cjq4

    # ------------------------------------------------------------------ #
    # Section 5 -- tile-mean coefficients.
    # ------------------------------------------------------------------ #
    frti = inp.frti
    # surfseb_ctl_mod.F90 412-427: ZEJS*/ZEJQ* are zero-initialised and
    # accumulated tile-by-tile in a sequential DO loop; a pairwise tree
    # reduction rounds differently and seeds ulp drift in the 2x2 solve.
    ejs1 = torch.zeros(frti.shape[:-1], dtype=frti.dtype, device=frti.device)
    ejs2 = torch.zeros_like(ejs1)
    ejs4 = torch.zeros_like(ejs1)
    ejq1 = torch.zeros_like(ejs1)
    ejq2 = torch.zeros_like(ejs1)
    ejq4 = torch.zeros_like(ejs1)
    for jt in range(frti.shape[-1]):
        ejs1 = ejs1 + frti[..., jt] * djs1[..., jt]
        ejs2 = ejs2 + frti[..., jt] * djs2[..., jt]
        ejs4 = ejs4 + frti[..., jt] * djs4[..., jt]
        ejq1 = ejq1 + frti[..., jt] * djq1[..., jt]
        ejq2 = ejq2 + frti[..., jt] * djq2[..., jt]
        ejq4 = ejq4 + frti[..., jt] * djq4[..., jt]

    # ------------------------------------------------------------------ #
    # Section 6 -- 2x2 solve for the grid-mean (Sl, Ql).
    # ------------------------------------------------------------------ #
    zz1 = 1.0 - asl[..., 0] * ejs1
    zz2 = aql[..., 0] * ejs2
    zz3 = bsl[..., 0] * ejs1 + bql[..., 0] * ejs2 + ejs4
    zy1 = 1.0 - aql[..., 0] * ejq2
    zy2 = asl[..., 0] * ejq1
    zy3 = bsl[..., 0] * ejq1 + bql[..., 0] * ejq2 + ejq4

    zz_det = zz1 * zy1 - zz2 * zy2
    zizz_det = 1.0 / zz_det
    jq = (zz3 * zy2 + zz1 * zy3) * zizz_det
    js = (zz2 * zy3 + zz3 * zy1) * zizz_det

    sl = asl[..., 0] * js + bsl[..., 0]
    ql = aql[..., 0] * jq + bql[..., 0]

    # ------------------------------------------------------------------ #
    # Section 7 -- per-tile fluxes and skin temperature.
    # ------------------------------------------------------------------ #
    sl_t = sl.unsqueeze(-1)
    ql_t = ql.unsqueeze(-1)

    pssk = dss1 * sl_t + dss2 * ql_t + dss4
    pjs = djs1 * sl_t + djs2 * ql_t + djs4
    pjq = djq1 * sl_t + djq2 * ql_t + djq4
    ptsk = pssk * icptm1

    pssh = pjs - consts.cpd * inp.tskm1m * delta * pjq
    pslh = lam * pjq
    pstr = slrfl + dlwdt * (ptsk - tskrad)
    pg0 = inp.lamsk * (ptsk - inp.tsrf)

    # ------------------------------------------------------------------ #
    # Section 8 -- optional recompute of tiled LW around the gridbox
    # mean skin temperature (LELWTL, energy-conserving correction).
    # ------------------------------------------------------------------ #
    if inp.lelwtl:
        # surfseb_ctl_mod.F90 507-521: ZTSKMEAN is a sequential tile sum.
        tsk_mean = torch.zeros(frti.shape[:-1], dtype=frti.dtype, device=frti.device)
        for jt in range(frti.shape[-1]):
            tsk_mean = tsk_mean + frti[..., jt] * ptsk[..., jt]
        dlwdt_star = (-4.0 * emis[..., 0] * consts.rsigma) * (
            (tsk_mean * tsk_mean) * tsk_mean)
        pstr = inp.slrfl_grid.unsqueeze(-1) + dlwdt_star.unsqueeze(-1) * (
            ptsk - tsk_mean.unsqueeze(-1)
        )

    return SebOutputs(
        pjs=pjs, pjq=pjq, pssk=pssk, ptsk=ptsk,
        pssh=pssh, pslh=pslh, pstr=pstr, pg0=pg0,
        psl=sl, pql=ql,
    )


# --------------------------------------------------------------------------- #
# Per-tile latent heat selector
# --------------------------------------------------------------------------- #
def _tile_latent_heat(
    ngrid: int,
    ntile: int,
    dtype: torch.dtype,
    device,
    consts: ConstantBundle,
    lake_water: torch.Tensor,
    lake_ice: torch.Tensor,
) -> torch.Tensor:
    """逐 tile 潜热 ``lambda``（J/kg）。

    对应 surfseb_ctl_mod.F90 第 271-275 行和第 385-387 行：
        JT 属于 {海冰, 低植被-雪}    -> RLSTT（升华）
        JT == 湖泊                   -> RLVTT*water + RLSTT*ice
        其他                         -> RLVTT（蒸发）

    返回 ``(Ngrid, Ntile)`` 张量。tile 索引为 0 基
    （``TILE_ICE=1``、``TILE_LOW_VEG_SNOW=4``、``TILE_LAKE=8``）。
    """
    lam = torch.full((ngrid, ntile), consts.rlvtt,
                     dtype=dtype, device=device)
    if ntile > TILE_ICE:
        lam[:, TILE_ICE] = consts.rlstt
    if ntile > TILE_LOW_VEG_SNOW:
        lam[:, TILE_LOW_VEG_SNOW] = consts.rlstt
    if ntile > TILE_LAKE:
        lam[:, TILE_LAKE] = (
            consts.rlvtt * lake_water
            + consts.rlstt * lake_ice
        )
    return lam


def _tile_solar_transmission(
    ngrid: int,
    ntile: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """SEB 中乘以净短波的逐 tile 因子 ``1 - RVTRSR``。

    镜像 ``surfseb_ctl_mod.F90`` SELECT CASE 块中的 ``ZFRSR`` 赋值。
    Fortran 使用 1 基 ``JT``；本移植为 0 基，因此映射经由
    :data:`surf_pytorch.config.F90_TILE_OF`（port_idx -> Fortran JT
    = port_idx + 1）。``FRSR != 1`` 的 tile 有：

        port TILE_WET_SKIN     (=2, F90 JT=3) -> 1 - RVTRSR(1) = 0.95
        port TILE_LOW_VEG_DRY  (=3, F90 JT=4) -> 1 - RVTRSR(1) = 0.95
        port TILE_HIGH_VEG_DRY (=5, F90 JT=6) -> 1 - RVTRSR(3) = 0.97
        port TILE_HIGH_VEG_SNOW(=6, F90 JT=7) -> 1 - RVTRSR(3) = 0.97
        其余所有 tile                          -> 1.0

    这些值是*与植被类型无关*的常数，因为主导植被类型的查找已在上游
    完成，且 SURFSEB_CTL 统一使用固定索引 1 或 3（RVTRSR(1)=0.05、
    RVTRSR(3)=0.03，来自 SUSVEG）。

    （本 docstring 的早期版本在赋值 ``frsr[:, 2]`` 时写作“tile 3
    （湿表皮）”——代码是对的，但文字混淆了移植索引与 Fortran 索引。
    现在两者均已明确写出。）
    """
    from ..config import (
        TILE_WET_SKIN, TILE_LOW_VEG_DRY, TILE_HIGH_VEG_DRY, TILE_HIGH_VEG_SNOW,
    )
    frsr = torch.ones((ngrid, ntile), dtype=dtype, device=device)
    if ntile > TILE_WET_SKIN:           # F90 JT=3, RVTRSR(1)=0.05 -> 0.95
        frsr[:, TILE_WET_SKIN] = 0.95
    if ntile > TILE_LOW_VEG_DRY:        # F90 JT=4, RVTRSR(1)=0.05 -> 0.95
        frsr[:, TILE_LOW_VEG_DRY] = 0.95
    if ntile > TILE_HIGH_VEG_DRY:       # F90 JT=6, RVTRSR(3)=0.03 -> 0.97
        frsr[:, TILE_HIGH_VEG_DRY] = 0.97
    if ntile > TILE_HIGH_VEG_SNOW:      # F90 JT=7, RVTRSR(3)=0.03 -> 0.97
        frsr[:, TILE_HIGH_VEG_SNOW] = 0.97
    return frsr


__all__ = [
    "SebInputs",
    "SebOutputs",
    "surf_seb",
]
