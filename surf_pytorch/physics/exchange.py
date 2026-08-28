"""Monin-Obukhov 近地层交换系数（Fortran ``VEXCS``）。

``surf/module/vexcs_mod.F90``（48r1）的直接翻译。迭代求解总体
Richardson 数与稳定度参数 ``eta = z/L`` 之间的隐式关系，然后计算
缩放后的输送系数::

    PCFM = rho * Cm * |U|     [kg/m^2/s]
    PCFH = rho * Ch * |U|     [kg/m^2/s]
    PCFQ = rho * Cq * |U|     [kg/m^2/s]
    PKH  = Ch * |U|           [m/s]
    PKC  = Cc * |U|           [m/s]   (tracer, between lev and z0m)

该算法（Beljaars 1990）分为四节，按顺序转录如下：

1. *基本量*（vexcs_mod.F90 第 2 节）：总体 Richardson 数 ``ZRICLS``、
   有效风切变 ``ZDU2``（含 W* 对流速度尺度）、中性对数廓线比值，
   以及在回代中出现的 ``Z1DZ1D`` 因子。
2. *初猜*（第 3.1 节）：若 ``ldinit`` 为 True，例程自行构造两点
   初猜——稳定 Ri 查 RITBL 表，不稳定时用简单的对数比值。否则使用
   上一步的 ``PZDL``。
3. *割线迭代*（第 3.2-3.3 节）：一次回代产生第二近似，随后进行
   ``kitt`` 次线性插值（以相继迭代近似导数的 Newton 法）迭代。
   符号翻转保护使迭代值保持在零的正确一侧。
4. *输送系数*（第 4 节）：在 ``RCHBHDL * L`` 处的强稳定截断将廓线
   截断到有限的 NBL 高度，然后由积分后的 PSI 函数得到缩放后的
   输送系数。

可微性
-----------------
每个分支（稳定与不稳定、截断区域）均用 ``torch.where`` 求值，因此
该例程在 ``enable_grad`` 下端到端可微。未选中分支的贡献为零——与
分段 Fortran 语义一致。RITBL 初猜已 detach（它是常数表，绝不是输入
的函数）。

布局
------
所有输入/输出均为 ``(Ngrid,)`` —— ``VEXCS`` 是格点平均量。上游驱动
（``VDFEXCS``）中的逐斑块平铺位于此内核之上，此处未实现；本移植的
SEB 阶段消费广播到各斑块的格点平均系数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..constants import ConstantBundle
from ..tables import RitblTable
from ..tables.ritbl import (
    JPRITBL,
    RIMAX,
    RCHETA,
    RCHETB,
    RCHBA,
    RCHBB,
    RCHBC,
    RCHBD,
    RCHB23A,
    RCHBBCD,
    RCHBCD,
    RCDHALF,
    RCDHPI2,
)


# --------------------------------------------------------------------------- #
# Constants from YOS_EXC / YOS_EXCS that are not in PhysicalConstants.
# --------------------------------------------------------------------------- #
# Wind-speed floor for ZDU2 (REPDU2 in YOS_EXC). OpenIFS SURF
# initialises this as (0.1 m/s)^2 in suvexc_mod.F90:
#   REPDU2 = (0.1_JPRB)**2  ==  fl(0.1)*fl(0.1)  ==  0.010000000000000002
# 1.0e-2 is one ulp lower (0.01) and diverges from the Fortran on every
# calm-wind step where the clamp binds.
REPDU2 = 0.1 * 0.1

# von Karman constant (RKAP in YOS_EXC).  48r1 default.
RKAP = 0.4

# Boundary-layer height used to scale the W* contribution (RPARZI).
RPARZI = 1000.0

# Capping ratio for very-stable stratification: the column is truncated
# to RCHBHDL * L (RCHBHDL = 5).  From YOS_EXCS.
RCHBHDL = 5.0

# Iteration count for the secant iteration (KITT).  The upstream
# SURFEXCDRIVER call uses 3 by default when LDINIT=.T.
DEFAULT_KITT = 3


# --------------------------------------------------------------------------- #
# PSI stability functions (statement functions in fcsvdfs.h).
#
# These are vectorised torch translations of the JPRD (double) statement
# functions.  Each takes an ``eta`` tensor that may contain both stable
# (positive) and unstable (negative) values; the calling site selects
# the branch via ``torch.where``.
# --------------------------------------------------------------------------- #
def psimu(eta: torch.Tensor) -> torch.Tensor:
    """不稳定动量 PSI（Dyer-Hicks）。

    ``PSIMU(eta) = log((1 + sqrt(sqrt(1 - 16*eta)))^2
                       * (1 + sqrt(1 - 16*eta)) / 8)
                   - 2 * atan(sqrt(sqrt(1 - 16*eta))) + RCDHPI2``
    """
    z = torch.clamp(1.0 - RCDHALF * eta, min=0.0)   # protect sqrt arg
    if torch.is_grad_enabled():
        z = torch.clamp(z, min=torch.finfo(z.dtype).eps)
    sqrt_z = torch.sqrt(z)
    fourth_z = torch.sqrt(sqrt_z)
    # NB: the upstream statement function writes ``sqrt(1-16*eta)`` here,
    # which mathematically equals ``sqrt_z``.  We must reuse the *clamped*
    # ``sqrt_z`` (not a fresh ``sqrt(1-16*eta)``) so that on the off-branch
    # -- i.e. when ``eta > 1/16`` and this routine is masked out by
    # ``_psi_m``'s ``torch.where`` -- the unselected branch stays finite.
    # An unclamped ``sqrt`` of a negative argument yields NaN in the forward
    # pass, and although ``where`` masks the output, the NaN propagates
    # through the backward pass (``0 * nan = nan``), corrupting gradients of
    # callers that feed stable (``eta > 0``) columns into the kernel (e.g.
    # the per-tile VEXCS path, where small-z0 water/ice tiles push ``z/L``
    # past ``1/16``).
    inner = sqrt_z
    return (
        torch.log((1.0 + fourth_z) ** 2 * (1.0 + inner) * 0.125)
        - 2.0 * torch.atan(fourth_z)
        + RCDHPI2
    )


def psihu(eta: torch.Tensor) -> torch.Tensor:
    """不稳定热量／水汽 PSI（Dyer-Hicks）。

    ``PSIHU(eta) = 2 * log((1 + sqrt(1 - 16*eta)) / 2)``
    """
    arg = torch.clamp(1.0 - RCDHALF * eta, min=0.0)
    # The selected unstable branch has ``arg > 0``. On the unselected
    # stable branch, autograd still differentiates through this expression
    # and sqrt(0) has an infinite derivative. The epsilon is used only
    # while constructing a gradient graph, preserving operational forward
    # values and the Fortran calculation.
    if torch.is_grad_enabled():
        arg = torch.clamp(arg, min=torch.finfo(arg.dtype).eps)
    inner = torch.sqrt(arg)
    return 2.0 * torch.log((1.0 + inner) * 0.5)


def psims(eta: torch.Tensor) -> torch.Tensor:
    """稳定动量 PSI（Holtslag-de Bruin）。

    ``PSIMS(eta) = -RCHBB*(eta - RCHBCD)*exp(-RCHBD*eta)
                   - RCHBA*eta - RCHBBCD``
    """
    eta_stable = eta
    if torch.is_grad_enabled():
        # This function is selected only for eta>0. Prevent the unselected
        # unstable branch from overflowing exp() inside torch.where.
        eta_stable = torch.clamp(eta, min=0.0)
    return (
        -RCHBB * (eta_stable - RCHBCD) * torch.exp(-RCHBD * eta_stable)
        - RCHBA * eta_stable
        - RCHBBCD
    )


def psihs(eta: torch.Tensor) -> torch.Tensor:
    """稳定热量／水汽 PSI（带临界 Ri 上限的 Holtslag-de Bruin）。

    ``PSIHS(eta) = -RCHBB*(eta - RCHBCD)*exp(-RCHBD*eta)
                   - (1 + RCHB23A*eta)*sqrt(1 + RCHB23A*eta)
                   - RCHBBCD + 1``
    """
    eta_stable = eta
    if torch.is_grad_enabled():
        # PSIHS is selected only for eta>0; evaluating its exponential for
        # an unselected, very unstable eta can overflow before torch.where
        # masks the forward value and corrupt its backward product.
        eta_stable = torch.clamp(eta, min=0.0)
    inner = 1.0 + RCHB23A * eta_stable
    # ``_psi_h`` selects this stable function with ``torch.where``. On the
    # unselected unstable branch, inner can be negative and clamp(..., 0)
    # gives sqrt(0) an infinite backward derivative. Raise only that inactive
    # graph construction away from zero; the physical stable branch has
    # inner > 1 and retains the original Fortran expression exactly.
    sqrt_inner = torch.clamp(inner, min=0.0)
    if torch.is_grad_enabled():
        sqrt_inner = torch.clamp(sqrt_inner, min=torch.finfo(inner.dtype).eps)
    return (
        -RCHBB * (eta_stable - RCHBCD) * torch.exp(-RCHBD * eta_stable)
        - inner * torch.sqrt(sqrt_inner)
        - RCHBBCD + 1.0
    )


def _psi_m(eta: torch.Tensor) -> torch.Tensor:
    """混合符号 eta 上的动量 PSI（按符号选择分支）。"""
    return torch.where(eta > 0.0, psims(eta), psimu(eta))


def _psi_h(eta: torch.Tensor) -> torch.Tensor:
    """混合符号 eta 上的热量／水汽 PSI（按符号选择分支）。"""
    return torch.where(eta > 0.0, psihs(eta), psihu(eta))


# --------------------------------------------------------------------------- #
# Output container
# --------------------------------------------------------------------------- #
@dataclass
class ExchangeOutputs:
    """缩放后的 Monin-Obukhov 输送系数（VEXCS 输出）。

    所有字段形状为 ``(Ngrid,)``。``pcfm``/``pcfh``/``pcfq`` 是经 rho
    缩放、馈入 SEB 的系数；``pkh``/``pkc`` 是未缩放（不含 rho）的
    热量／示踪物交换速度。
    """

    pcfm: torch.Tensor     # rho*Cm*|U|  [kg/m^2/s]
    pcfh: torch.Tensor     # rho*Ch*|U|  [kg/m^2/s]
    pcfq: torch.Tensor     # rho*Cq*|U|  [kg/m^2/s]
    pkh: torch.Tensor      # Ch*|U|      [m/s]
    pkc: torch.Tensor      # Cc*|U|      [m/s]
    pzdl: torch.Tensor     # z/L at lowest model level


# --------------------------------------------------------------------------- #
# VEXCS
# --------------------------------------------------------------------------- #

# ------------------------------------------------------------------ #
# Debug: per-call VEXCS internals (SURF_VEXCS_CSV), mirroring the F90
# SURF_VEXCS_BIN records so the two can be diffed tile by tile.
# ------------------------------------------------------------------ #
import csv as _csv
import os as _os

_VEXCS_CSV = _os.environ.get("SURF_VEXCS_CSV", "").strip()
_VEXCS_CALLS = 0


def _vexcs_debug(zprh1_bs, zprh0_bs, zprm1_bs, zprm0_bs, zprh_bs, zprm_bs,
                 zeta3, pcfh, pcfq, pcfm, zaux2, zdu2, zricls, zhu,
                 zf2, zf3, zeta2, zeta3_c, zprm, zprh):
    global _VEXCS_CALLS
    if not _VEXCS_CSV:
        return
    groups = [t.reshape(-1) for t in (
        zprh1_bs, zprh0_bs, zprm1_bs, zprm0_bs, zprh_bs, zprm_bs,
        zeta3, pcfh, pcfq, pcfm, zaux2, zdu2, zricls, zhu,
        zf2, zf3, zeta2, zeta3_c, zprm, zprh)]
    ntile = groups[0].shape[0]
    new = not _os.path.exists(_VEXCS_CSV)
    with open(_VEXCS_CSV, "a", newline="") as fh:
        w = _csv.writer(fh)
        if new:
            w.writerow(["call"] + ["%s_t%d" % (n, t)
                                   for n in ("zprh1_bs", "zprh0_bs", "zprm1_bs", "zprm0_bs", "zprh_bs", "zprm_bs", "zeta3", "pcfh", "pcfq", "pcfm", "zaux2", "zdu2", "zricls", "zhu", "zf2", "zf3", "zeta2", "zeta3_c", "zprm", "zprh")
                                   for t in range(ntile)])
        row = [_VEXCS_CALLS]
        for g in groups:
            row += ["%r" % v for v in g.detach().cpu().tolist()]
        w.writerow(row)
    _VEXCS_CALLS += 1

def vexcs(
    # lowest-level state
    pu: torch.Tensor,            # (Ngrid,)  u at lowest level
    pv: torch.Tensor,            # (Ngrid,)  v at lowest level
    pt: torch.Tensor,            # (Ngrid,)  T at lowest level [K]
    pq: torch.Tensor,            # (Ngrid,)  q at lowest level [kg/kg]
    paphms: torch.Tensor,        # (Ngrid,)  surface pressure [Pa]
    pgeomlev: torch.Tensor,      # (Ngrid,)  geopotential at lowest level [m2/s2]
    pcptgzlev: torch.Tensor,     # (Ngrid,)  dry-static-energy at lowest level [m2/s2]
    pcpts: torch.Tensor,         # (Ngrid,)  dry-static-energy at surface   [m2/s2]
    pqsam: torch.Tensor,         # (Ngrid,)  surface specific humidity     [kg/kg]
    pz0mm: torch.Tensor,         # (Ngrid,)  momentum roughness length [m]
    pz0hm: torch.Tensor,         # (Ngrid,)  heat roughness length      [m]
    pz0qm: torch.Tensor,         # (Ngrid,)  moisture roughness length [m]
    pzdl0: Optional[torch.Tensor],  # (Ngrid,) initial guess z/L (None => ldinit)
    pbuom: torch.Tensor,         # (Ngrid,)  surface buoyancy flux
    pucurr: torch.Tensor,        # (Ngrid,)  ocean current u [m/s]
    pvcurr: torch.Tensor,        # (Ngrid,)  ocean current v [m/s]
    pi10fgcv: torch.Tensor,      # (Ngrid,)  convective gust [m/s]
    consts: ConstantBundle,
    ritbl: RitblTable,
    *,
    kitt: int = DEFAULT_KITT,
    ldinit: bool = True,
    repdu2: float = REPDU2,
    rkap: float = RKAP,
    rparzi: float = RPARZI,
    rchbhdl: float = RCHBHDL,
) -> ExchangeOutputs:
    """计算 Monin-Obukhov 交换系数（VEXCS 的移植）。

    仅关键字签名与上游 ``VEXCS`` 形参列表一一对应；上游取自
    ``YDCST`` / ``YDEXC`` 的常数，此处取自 ``consts`` 及本模块顶部
    的默认值。

    Args
    ----
    pzdl0:
        上一步的 ``z/L`` 初猜。若为 ``None`` 或 ``ldinit`` 为 True，
        例程通过 RITBL 查表／对数比值自行构造初猜（第 3.1 节，
        ``LDINIT=.T.`` 分支）。
    kitt:
        回代之后的割线迭代次数。上游默认值为 3（冷启动）／ 0..1
        （暖启动）。
    ldinit:
        即使提供了 ``pzdl0`` 也强制走内部初猜路径；对应上游的
        ``LDINIT`` 标志。
    """
    # ------------------------------------------------------------------ #
    # 1. Constants and bookkeeping
    # ------------------------------------------------------------------ #
    # Use 0-d tensors directly (not .item()) so torch.compile can trace.
    rd = consts.rd
    retv = consts.retv
    rg = consts.g
    rvtmp2 = consts.rvtmp2

    # Upstream section-1 scalars (vexcs_mod.F90 183-189).
    zcon1 = rvtmp2 - retv       # contribution of humidity to buoyancy
    zcon2 = 2.0 / 3.0           # exponent for (W*)^2
    zcon3 = rkap * rkap         # kappa^2 (gfortran expands RKAP**2)
    zipbl = rparzi              # PBL height for W* scaling

    # ------------------------------------------------------------------ #
    # 2. Basic quantities: W*, ZDU2, ZRICLS, log-profile ratios.
    # ------------------------------------------------------------------ #
    # Convective velocity scale (W*)^2 -- zero when buoyancy flux <= 0
    # (stable/neutral).  Buoyancy flux is *positive* upward in IFS, so a
    # negative PBUOM means stable stratification.
    pos_buom = torch.clamp(pbuom, min=0.0)
    wstar_base = pos_buom * zipbl
    if torch.is_grad_enabled():
        # ``x**(2/3)`` has an undefined derivative at x=0. The Fortran
        # branch remains exactly zero; this only makes the inactive branch
        # well-defined for PyTorch autograd.
        wstar_base = torch.clamp(wstar_base, min=torch.finfo(wstar_base.dtype).eps)
    zwst2 = torch.where(
        pbuom > 0.0,
        wstar_base ** zcon2,
        torch.zeros_like(pbuom),
    )

    # Effective wind-shear squared at the surface (vexcs 207-208).
    zdu2 = torch.clamp(
        (pu - pucurr) ** 2 + (pv - pvcurr) ** 2 + zwst2,
        min=repdu2,
    )

    # Bulk Richardson number ingredients (vexcs 209-212).
    zdroro = (
        2.0 * (pcptgzlev - pcpts)
        / (pcptgzlev + pcpts - pgeomlev)
        - zcon1 * (pq - torch.clamp(pqsam, max=0.1))
    )
    zricls = pgeomlev * zdroro / zdu2

    # Neutral log-profile ratios (vexcs 217-224).  ZNLEV is the height
    # of the lowest model level above the displacement (z0m).
    znlev = pgeomlev / rg + pz0mm
    z1dz0m = znlev / pz0mm
    z1dz0h = znlev / pz0hm
    z1dz0q = znlev / pz0qm
    z1dz1d = z1dz0m / (z1dz0m - 1.0)
    zxlnm = torch.log(z1dz0m)
    zxlnh = torch.log(z1dz0h)
    zxlnq = torch.log(z1dz0q)

    # Reciprocals used by the back-substitution (vexcs 294-297).
    z1dz0md = 1.0 / z1dz0m
    z1dz0hd = 1.0 / z1dz0h

    # ------------------------------------------------------------------ #
    # 3.1 First / second guess
    # ------------------------------------------------------------------ #
    if ldinit or pzdl0 is None:
        # Stable branch: interpolate the RITBL.
        zrib1 = zricls * zxlnm / zxlnh
        zeta1_ritbl = ritbl.initial_guess(zrib1)
        # Unstable branch: simple log-ratio (vexcs 321).
        zeta1_unstable = zrib1
        zeta1 = torch.where(zricls > 0.0, zeta1_ritbl, zeta1_unstable)

        # Second guess: multiply by the log-ratio factor (vexcs 327).
        zeta2 = zeta1 * zxlnm * z1dz1d
    else:
        zeta2 = pzdl0.clone()

    # ------------------------------------------------------------------ #
    # 3.2 Back-substitution: produces the second iterate (zeta3) and the
    #     first residual (zf2).
    # ------------------------------------------------------------------ #
    # PSI at the lowest level (zeta2) and at the roughness length
    # (zeta2 * z0 / znlev = zeta2 / z1dz0*).
    zprh1 = _psi_h(zeta2)
    zprh0 = _psi_h(zeta2 * z1dz0hd)
    zprm1 = _psi_m(zeta2)
    zprm0 = _psi_m(zeta2 * z1dz0md)

    zprh = zxlnh - zprh1 + zprh0
    zprm = zxlnm - zprm1 + zprm0
    zhu = zprm ** 2 * z1dz1d / zprh
    zeta3 = zricls * zhu
    zf2 = zricls - zeta2 / zhu
    zprh1_bs = zprh1
    zprh0_bs = zprh0
    zprm1_bs = zprm1
    zprm0_bs = zprm0
    zprh_bs = zprh
    zprm_bs = zprm

    # ------------------------------------------------------------------ #
    # 3.3 KITT secant iterations (vexcs 369-406).
    # ------------------------------------------------------------------ #
    for _ in range(kitt):
        zprh1 = _psi_h(zeta3)
        zprh0 = _psi_h(zeta3 * z1dz0hd)
        zprm1 = _psi_m(zeta3)
        zprm0 = _psi_m(zeta3 * z1dz0md)

        zprh = zxlnh - zprh1 + zprh0
        zprm = zxlnm - zprm1 + zprm0
        zf3 = zricls - zeta3 * zprh / (z1dz1d * zprm ** 2)

        # Secant step (vexcs 394).  When the residual difference is
        # negligible, hold the previous iterate (vexcs 396).
        denom = zf3 - zf2
        tiny = torch.full_like(denom, 1.0e-25)
        step = torch.where(
            torch.abs(denom) > 1.0e-25,
            zf2 * (zeta3 - zeta2) / torch.where(
                torch.abs(denom) > 1.0e-25, denom, tiny
            ),
            torch.zeros_like(denom),
        )
        zeta_new = torch.where(
            torch.abs(denom) > 1.0e-25,
            zeta2 - step,
            zeta3,
        )

        # Sign-flip safety: if the secant overshoots past zero, halve
        # the new iterate toward zeta3 (vexcs 401).
        zeta_new = torch.where(
            zeta_new * zeta3 < 0.0,
            zeta3 * 0.5,
            zeta_new,
        )

        # Roll the iterates forward (vexcs 402-404).
        zeta2 = zeta3
        zf2 = zf3
        zeta3 = zeta_new

    # ------------------------------------------------------------------ #
    # 4. Transfer coefficients.
    # ------------------------------------------------------------------ #
    # Very-stable cap: assume maximum NBL height = RCHBHDL * L and
    # uniform profiles from z0 to znlev (vexcs 424-434).
    over_cap = zeta3 > rchbhdl
    zl_safe = znlev / torch.where(zeta3 != 0.0, zeta3, torch.ones_like(zeta3))
    znlev_cap = rchbhdl * zl_safe + torch.maximum(pz0mm, pz0qm)
    zeta3_cap = znlev_cap / zl_safe

    z1dz0m_c = torch.where(over_cap, znlev_cap / pz0mm, z1dz0m)
    z1dz0h_c = torch.where(over_cap, znlev_cap / pz0hm, z1dz0h)
    z1dz0q_c = torch.where(over_cap, znlev_cap / pz0qm, z1dz0q)
    zxlnm_c = torch.where(over_cap, torch.log(z1dz0m_c), zxlnm)
    zxlnh_c = torch.where(over_cap, torch.log(z1dz0h_c), zxlnh)
    zxlnq_c = torch.where(over_cap, torch.log(z1dz0q_c), zxlnq)
    zeta3_c = torch.where(over_cap, zeta3_cap, zeta3)

    z1dz0md_c = 1.0 / z1dz0m_c
    z1dz0hd_c = 1.0 / z1dz0h_c
    z1dz0qd_c = 1.0 / z1dz0q_c

    # Final PSI evaluation at the (possibly capped) iterate.
    zprm1 = _psi_m(zeta3_c)
    zprm0 = _psi_m(zeta3_c * z1dz0md_c)
    zprh1 = _psi_h(zeta3_c)
    zprh0 = _psi_h(zeta3_c * z1dz0hd_c)
    zprq0 = _psi_h(zeta3_c * z1dz0qd_c)
    zprc0 = _psi_h(zeta3_c * z1dz0md_c)

    zprm = zxlnm_c - zprm1 + zprm0
    zprh = zxlnh_c - zprh1 + zprh0
    zprq = zxlnq_c - zprh1 + zprq0
    zprc = zxlnm_c - zprh1 + zprc0

    rho = paphms / (rd * pt * (1.0 + retv * pq))
    zuabs = torch.sqrt(zdu2)
    zaux2 = zuabs * rho * zcon3

    pcfm = zaux2 / (zprm ** 2)
    pcfh = zaux2 / (zprm * zprh)
    pcfq = zaux2 / (zprm * zprq)
    pkh = pcfh / rho
    pkc = zuabs * zcon3 / (zprm * zprc)

    _vexcs_debug(zprh1_bs, zprh0_bs, zprm1_bs, zprm0_bs, zprh_bs, zprm_bs,
                 zeta3, pcfh, pcfq, pcfm, zaux2, zdu2, zricls, zhu,
                 zf2, zf3, zeta2, zeta3_c, zprm, zprh)
    return ExchangeOutputs(
        pcfm=pcfm, pcfh=pcfh, pcfq=pcfq,
        pkh=pkh, pkc=pkc, pzdl=zeta3,
    )


__all__ = [
    "ExchangeOutputs",
    "vexcs",
    "psimu", "psihu", "psims", "psihs",
    "REPDU2", "RKAP", "RPARZI", "RCHBHDL", "DEFAULT_KITT",
]

def vexcss(
    pu: torch.Tensor, pv: torch.Tensor, pt: torch.Tensor, pq: torch.Tensor,
    paphms: torch.Tensor, pgeomlev: torch.Tensor, pcptgzlev: torch.Tensor,
    pcpts: torch.Tensor, pqsam: torch.Tensor, pz0mm: torch.Tensor,
    pz0hm: torch.Tensor, pz0qm: torch.Tensor, pbuom: torch.Tensor,
    consts: ConstantBundle, ptmst: torch.Tensor, rvdifts: float = 1.0,
) -> ExchangeOutputs:
    """离线 OSM ``VEXCSS`` 交换系数。

    Fortran ``VEXCSS`` 生成已为 VDF 驱动程序预先按
    ``RG * PTMST * PRVDIFTS`` 缩放的 C-star 系数。而公开的 Python
    交换接口约定采用物理量 ``rho*C*|U|`` 系数，与 :func:`vexcs`
    共享；VDF 缩放在之后构建 ``DiffusionInputs`` 时统一施加一次。
    """
    rlpbb = 5.0
    rlpcc = 5.0
    rlpdd = 5.0
    g = consts.g
    rd = consts.rd
    retv = consts.retv
    rvtmp2 = consts.rvtmp2
    # Fortran branches before evaluating the fractional power.  Use a
    # positive placeholder only in the inactive/autograd graph so the
    # derivative of x**(2/3) at x=0 is never formed; the selected output
    # remains exactly zero for PBUOM <= 0.
    wstar_arg = (pbuom * RPARZI).clamp_min(0.0)
    if torch.is_grad_enabled():
        wstar_arg = torch.where(
            pbuom > 0.0,
            wstar_arg,
            torch.full_like(wstar_arg, torch.finfo(wstar_arg.dtype).eps),
        )
    zwst2 = wstar_arg.pow(2.0 / 3.0)
    zwst2 = torch.where(pbuom > 0.0, zwst2, torch.zeros_like(zwst2))
    zdu2 = torch.maximum(
        pu.square() + pv.square() + zwst2,
        torch.full_like(pu, REPDU2),
    )
    qsam = torch.minimum(pqsam, torch.full_like(pqsam, 0.1))
    zdiv1 = 1.0 / (pcptgzlev + pcpts - pgeomlev)
    zdro = 2.0 * (pcptgzlev - pcpts) * zdiv1 - (rvtmp2 - retv) * (pq - qsam)
    zri = pgeomlev * zdro / zdu2
    znlev = pgeomlev / g + pz0mm
    lnm = torch.log(znlev / pz0mm)
    lnh = torch.log(znlev / pz0hm)
    lnq = torch.log(znlev / pz0qm)
    cdnm = 1.0 / lnm.square()
    cdnh = 1.0 / (lnm * lnh)
    cdnq = 1.0 / (lnm * lnq)
    stable = zri > 0.0
    # Fortran evaluates only the selected stability branch.  ``where``
    # evaluates both tensors in PyTorch, so keep the inactive stable sqrt
    # finite for unstable columns to avoid NaN gradients.
    z2s_st_arg = 1.0 + rlpdd * zri
    if torch.is_grad_enabled():
        z2s_st_arg = torch.clamp(z2s_st_arg, min=torch.finfo(z2s_st_arg.dtype).eps)
    else:
        z2s_st_arg = torch.clamp(z2s_st_arg, min=0.0)
    z2s_st = torch.sqrt(z2s_st_arg)
    fm_st = cdnm * z2s_st / (z2s_st + 2.0 * rlpbb * zri)
    fh_st = cdnh / (1.0 + 3.0 * rlpbb * zri * z2s_st)
    fq_st = cdnq / (1.0 + 3.0 * rlpbb * zri * z2s_st)
    # The unstable branch uses max(-Ri, ZRIMIN).  Keep its inactive
    # derivative finite when Ri is stable, where the max argument is zero.
    z2s_un_arg = torch.maximum(-zri, torch.full_like(zri, 1.0e-3))
    if torch.is_grad_enabled():
        z2s_un_arg = torch.clamp(z2s_un_arg, min=torch.finfo(z2s_un_arg.dtype).eps)
    z2s_un = torch.sqrt(z2s_un_arg)
    zcm = rlpcc * torch.sqrt(znlev / pz0mm)
    zch = rlpcc * torch.sqrt(znlev / pz0hm)
    zcq = rlpcc * torch.sqrt(znlev / pz0qm)
    zrk3b = RKAP * RKAP * 3.0 * rlpbb
    fm_un = cdnm * (1.0 - 2.0 * rlpbb * zri / (1.0 + zrk3b * cdnm * zcm * z2s_un))
    fh_un = cdnh * (1.0 - 3.0 * rlpbb * zri / (1.0 + zrk3b * cdnh * zch * z2s_un))
    fq_un = cdnq * (1.0 - 3.0 * rlpbb * zri / (1.0 + zrk3b * cdnq * zcq * z2s_un))
    # Fortran VEXCSS computes
    #   ZAUX2 = (PRVDIFTS*PTMST*RG) * rho * |U| * RKAP**2
    # and multiplies PCFM/PCFH/PCFQ by ZAUX2.  The rest of this port
    # expects physical rho*C*|U| and applies the VDF scaling exactly once
    # in model.py, so undo the C-star time/gravity scaling here.
    aux = (rvdifts * ptmst * g / rd * paphms) / (pt * (1.0 + retv * pq))
    cstar_scale = aux * torch.sqrt(zdu2) * (RKAP * RKAP)
    scale = cstar_scale / (rvdifts * ptmst * g)
    pcfm = scale * torch.where(stable, fm_st, fm_un)
    pcfh = scale * torch.where(stable, fh_st, fh_un)
    pcfq = scale * torch.where(stable, fq_st, fq_un)
    zero = torch.zeros_like(pcfm)
    return ExchangeOutputs(pcfm=pcfm, pcfh=pcfh, pcfq=pcfq, pkh=zero, pkc=zero, pzdl=zero)
