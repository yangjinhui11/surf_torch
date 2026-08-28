"""城市峡谷反照率（``CANALB``，``canalb_mod.F90`` 第 82-181 行）。

本模块移植 ECMWF 城市峡谷宽波段反照率——Masson (2000) 遮蔽方案与
Harman (2004) 矩阵交换公式以及 Porson (2010) 形状因子的组合。它是
``SURFRAD`` 在 ``surfrad_ctl_mod.F90:520`` 处调用以求取城市瓦片
（瓦片 10）峡谷反照率的算法。

方案概述
--------------------
对每个网格柱，例程构建三个峡谷"表面"——天空 (1)、路面 (2)、墙面
(3)——并求解一个 3x3 反射问题，得到峡谷整体的反照率。步骤如下：

1. **太阳分配**（Masson 2000 式 13-14，第 88-99 行）。由太阳天顶角
   余弦 ``PMU0`` 与峡谷高宽比 ``RHWR``（建筑高度与路宽之比），计算
   入射短波散射到路面（``x_r``）与墙面（``x_w``）上的比例::

       var1   = tan(acos(mu0))                 # limited away from 0
       omega0 = asin(min(1, 1/(var1*RHWR)))
       x_r    = (2/pi)*(omega0 - RHWR*var1*(1 - cos(omega0)))
       x_w    = (1 - x_r)/(2*RHWR)

   ``x_f = 0.3`` 是天空散射比例（第 85 行，硬编码）。

2. **形状因子**（Harman 2004 式 2-5 + Porson 2010 式 12-14，第
   101-116 行）。由 ``RHWR`` 构建视角因子矩阵 ``shf``，在三个表面
   之间分配出射通量。

3. **反射矩阵**（第 118-127 行）。耦合入射与反射通量的 3x3 系统，
   使用路面反照率 ``RROAALB`` 与墙面反照率 ``RWALALB``。

4. **求逆**（第 130-144 行）。反射矩阵用余子式解析求逆（此处逐字
   复现，而不是通过 ``torch.linalg.inv``，以便输入一致时结果与
   Fortran 逐位一致）。

5. **参考通量 + 组装**（Porson 2010 式 10，第 146-170 行）::

       ref = [RROAALB*(1-x_f)*x_r, RWALALB*(1-x_f)*x_w, x_f]
       OUT = rinv @ ref
       INC = shf @ OUT
       INC(1) += (1-x_f)*x_r - OUT(1)
       INC(2) += (1-x_f)*x_w - OUT(2)
       PCANALB = clamp(1 - (INC(1) + 2*RHWR*INC(2)), 0, 1)

   ``INC(3)`` 被计算但从未使用——与 Fortran 一致（墙面自吸收已折叠
   进墙面通量的 ``2*RHWR`` 权重中）。

输入 / 输出
----------------
``mu0`` 是太阳天顶角余弦，形状 ``(Ngrid,)``。三个材质/几何场
（``rwalalb``、``rroaalb``、``rhwr``）是 ``(Ngrid,)`` 张量（逐柱）
或跨网格广播的标量；上游默认值为 RWALALB=0.6、RROAALB=0.05、
RHWR=1.0。输出 ``pcanalb`` 是宽波段峡谷反照率 ``(Ngrid,)``，取值
在 [0,1] 内。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class CanalbInputs:
    """城市峡谷反照率求解的输入。

    每个字段均为 ``(Ngrid,)``，彼此自然广播。命名镜像 ``CANALB`` 的
    哑元变量 / ``YOS_URB`` 成员，便于逐行对照审计。
    """

    # ---- Solar geometry ----
    mu0: torch.Tensor              # PMU0  cosine solar zenith [-]  (Ngrid,)

    # ---- Canyon material albedos (YOS_URB) ----
    rwalalb: torch.Tensor          # RWALALB  wall albedo  [-]       (Ngrid,)
    rroaalb: torch.Tensor          # RROAALB  road albedo  [-]       (Ngrid,)

    # ---- Canyon geometry (YOS_URB) ----
    rhwr: torch.Tensor             # RHWR  height-to-roadwidth ratio [-] (Ngrid,)


@dataclass
class CanalbOutputs:
    """城市峡谷反照率求解的输出。"""

    pcanalb: torch.Tensor          # PCANALB  canyon albedo [-]      (Ngrid,)


def compute_canalb(
    inp: CanalbInputs,
    *,
    differentiable: bool = True,
) -> CanalbOutputs:
    """运行城市峡谷反照率（``canalb_mod.F90`` 82-181）。

    Args
    ----
    inp:
        :class:`CanalbInputs`——打包的太阳几何与峡谷材质 / 几何场。
    differentiable:
        若为 ``False`` 则在 ``torch.no_grad()`` 下运行以提速。

    Returns
    -------
    :class:`CanalbOutputs`——峡谷反照率 ``PCANALB``，取值在 [0, 1] 内。
    """

    def _core() -> CanalbOutputs:
        return _canalb_core(inp)

    if not differentiable:
        with torch.no_grad():
            return _core()
    return _core()


def _canalb_core(inp: CanalbInputs) -> CanalbOutputs:
    """内部：假定 no_grad / enable_grad 的决定已经做出。"""
    # ------------------------------------------------------------------ #
    # Scalar constants (Fortran lines 85-86).
    # ------------------------------------------------------------------ #
    x_f = 0.3
    pi = 4.0 * math.atan(1.0)

    mu0 = inp.mu0
    rwalalb = inp.rwalalb
    rroaalb = inp.rroaalb
    rhwr = inp.rhwr

    # ------------------------------------------------------------------ #
    # Solar partition (Masson 2000 eq 13-14, lines 88-99).
    # tan(acos(mu0)); limit away from 0 (line 92-94).
    # ------------------------------------------------------------------ #
    # acos in [0, pi/2] for mu0 in [0,1]; tan blows up at mu0=0.
    var1 = torch.tan(torch.acos(torch.clamp(mu0, min=1e-12, max=1.0)))
    var1 = torch.where(var1 == 0.0, torch.full_like(var1, 1e-4), var1)
    omega0 = 1.0 / (var1 * rhwr)
    omega0 = torch.asin(torch.clamp(omega0, max=1.0))      # line 96-97
    x_r = (2.0 / pi) * (omega0 - rhwr * torch.sin(omega0) * (1.0 - torch.cos(omega0)))
    x_w = (1.0 - x_r) / (2.0 * rhwr)

    # ------------------------------------------------------------------ #
    # Shape factors (Harman 2004 eq 2-5 + Porson 2010 eq 12-14, 101-116).
    # Build the 3x3 view-factor matrix `shf` as (Ngrid, 3, 3).
    # ------------------------------------------------------------------ #
    aa = torch.sqrt(1.0 + rhwr ** 2) - rhwr
    bb = torch.sqrt(1.0 + (1.0 / rhwr) ** 2) - (1.0 / rhwr)
    cc = 1.0 - aa
    dd = (1.0 - bb) / 2.0
    zero = torch.zeros_like(rhwr)
    # Row-major (i,j): shf[i, j]
    shf = torch.stack([
        torch.stack([zero, cc, aa], dim=-1),   # row 1: sky
        torch.stack([dd, bb, dd], dim=-1),     # row 2: road
        torch.stack([aa, cc, zero], dim=-1),   # row 3: wall
    ], dim=-2)                                 # (Ngrid, 3, 3)

    # ------------------------------------------------------------------ #
    # Reflection matrix (lines 118-127).
    # ------------------------------------------------------------------ #
    one = torch.ones_like(rhwr)
    rfl = torch.stack([
        torch.stack([one, -rroaalb * cc, -rroaalb * aa], dim=-1),  # row 1
        torch.stack([-rwalalb * dd, one - rwalalb * bb, -rwalalb * dd], dim=-1),
        torch.stack([zero, zero, one], dim=-1),                    # row 3
    ], dim=-2)                                                     # (Ngrid, 3, 3)

    # ------------------------------------------------------------------ #
    # Inverse by cofactors (Fortran lines 127-135).
    #
    # The Fortran assigns rinv in three column-groups (rinv(1,j)..rinv(3,j)
    # for j=1,2,3), which is the **adjugate** (transposed cofactor matrix)
    # -- the standard closed-form inverse.  We build rinv row-by-row here
    # (each outer stack entry is a row; inner stack entries are columns),
    # matching rinv(i,j) = (-1)^(i+j) * minor(j,i) / det.  The port
    # originally read the Fortran groups as rows, producing the transpose
    # of the inverse; this is the corrected, audited form.  See the
    # TestMatrixInverse / test_reference_value_recomputed_independently
    # tests, which cross-check rinv against torch.linalg.solve.
    # ------------------------------------------------------------------ #
    r11 = rfl[..., 0, 0]; r12 = rfl[..., 0, 1]; r13 = rfl[..., 0, 2]
    r21 = rfl[..., 1, 0]; r22 = rfl[..., 1, 1]; r23 = rfl[..., 1, 2]
    r31 = rfl[..., 2, 0]; r32 = rfl[..., 2, 1]; r33 = rfl[..., 2, 2]

    detinv = 1.0 / (
        r11 * r22 * r33 - r11 * r23 * r32
        - r12 * r21 * r33 + r12 * r23 * r31
        + r13 * r21 * r32 - r13 * r22 * r31
    )

    # Row 1: rinv(1,1)=+C11, rinv(1,2)=-C12, rinv(1,3)=+C13
    # Row 2: rinv(2,1)=-C21, rinv(2,2)=+C22, rinv(2,3)=-C23
    # Row 3: rinv(3,1)=+C31, rinv(3,2)=-C32, rinv(3,3)=+C33
    rinv = torch.stack([
        torch.stack([
             detinv * (r22 * r33 - r23 * r32),    # rinv(1,1) = +minor(1,1)
            -detinv * (r12 * r33 - r13 * r32),    # rinv(1,2) = -minor(1,2)
             detinv * (r12 * r23 - r13 * r22),    # rinv(1,3) = +minor(1,3)
        ], dim=-1),
        torch.stack([
            -detinv * (r21 * r33 - r23 * r31),    # rinv(2,1) = -minor(2,1)
             detinv * (r11 * r33 - r13 * r31),    # rinv(2,2) = +minor(2,2)
            -detinv * (r11 * r23 - r13 * r21),    # rinv(2,3) = -minor(2,3)
        ], dim=-1),
        torch.stack([
             detinv * (r21 * r32 - r22 * r31),    # rinv(3,1) = +minor(3,1)
            -detinv * (r11 * r32 - r12 * r31),    # rinv(3,2) = -minor(3,2)
             detinv * (r11 * r22 - r12 * r21),    # rinv(3,3) = +minor(3,3)
        ], dim=-1),
    ], dim=-2)   # (Ngrid, 3, 3)

    # ------------------------------------------------------------------ #
    # Reference flux (Porson 2010 eq 10, lines 146-149).
    # ------------------------------------------------------------------ #
    ref = torch.stack([
        rroaalb * (1.0 - x_f) * x_r,       # road
        rwalalb * (1.0 - x_f) * x_w,       # wall
        torch.full_like(x_f if torch.is_tensor(x_f) else rhwr, x_f),
    ], dim=-1)                              # (Ngrid, 3)

    # ------------------------------------------------------------------ #
    # Assembly (lines 151-170).
    #   OUT = rinv @ ref          (Ngrid,3)
    #   INC = shf @ OUT           (Ngrid,3)
    # ------------------------------------------------------------------ #
    out = torch.einsum("nij,nj->ni", rinv, ref)
    inc = torch.einsum("nij,nj->ni", shf, out)
    inc = inc.clone()   # avoid in-place op on autograd graph
    inc = torch.stack([
        inc[..., 0] + (1.0 - x_f) * x_r - out[..., 0],
        inc[..., 1] + (1.0 - x_f) * x_w - out[..., 1],
        inc[..., 2],
    ], dim=-1)

    # ------------------------------------------------------------------ #
    # Canyon albedo (line 168-170) + bounds clamp (173-179).
    # ------------------------------------------------------------------ #
    pcanalb = 1.0 - (inc[..., 0] + 2.0 * rhwr * inc[..., 1])
    pcanalb = torch.clamp(pcanalb, min=0.0, max=1.0)

    return CanalbOutputs(pcanalb=pcanalb)


__all__ = [
    "CanalbInputs",
    "CanalbOutputs",
    "compute_canalb",
]
