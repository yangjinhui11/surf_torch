"""离线 OSM 下边界 VDF 闭合。

本模块汇集独立陆面驱动所需的 ``VDFMAIN1S`` 部分。它有意*不*替换
耦合 OpenIFS 积分中的大气 VDF：耦合调用方从大气模型接收下边界通量。

离线 ``VDFDIFM1S`` 源码禁用了其垂直三对角求解；连同 ``VDFINCR``，
最低层动量通量按上游求值链逐位复现：

    PCFM   = sum_tiles frac*cfm_tile           (surfexcdriver_ctl_mod 943-949)
    ZCFM   = PCFM*RG*PTSPHY*RVDIFTS            (vdfmain1s 766)
    ZUDIF  = (1/RVDIFTS)*PUM1 + PTMST*PVOM     (vdfdifm1s)
    PSTRTU = ZCONS2*ZCFM*ZUDIF                 (vdfincr 184; ZCONS2=1/(RG*PTMST))

代数上 ``RG*PTSPHY*RVDIFTS`` 与 ``ZCONS2`` 相消，但 F90 保留全部中间
舍入：任何化简（如 ``PCFM*U``）都会引入间歇性 ulp 噪声，并在 VUPDZ0
的 ZDL 反馈环中放大。``VDFDIFH1S`` 仍负责瓦片化的感热与水汽通量。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .diffusion import RVDIFTS


@dataclass(frozen=True)
class OfflineVDFMomentum:
    """最低层 VDFMAIN1S 动量通量输出。"""

    stress_u: torch.Tensor
    stress_v: torch.Tensor


def vdfdifm1s_surface_stress(
    pcfm_tile: torch.Tensor,
    tile_fraction: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    rg: float,
    rvdifts: float = RVDIFTS,
    u_tendency: torch.Tensor | None = None,
    v_tendency: torch.Tensor | None = None,
) -> OfflineVDFMomentum:
    """按上游舍入顺序计算离线 ``VDFDIFM1S``/``VDFINCR`` 表面应力。

    ``u``/``v`` 为最低层绝对风速（PUM1/PVM1），``u_tendency``/
    ``v_tendency`` 为大气动量倾向（PVOM/PVOL）。F90 的 PUDIF 使用
    绝对风速，不含地表洋流修正。
    """
    if pcfm_tile.ndim != 2 or tile_fraction.shape != pcfm_tile.shape:
        raise ValueError("pcfm_tile and tile_fraction must both be (Ngrid, Ntile)")
    ngrid = pcfm_tile.shape[0]
    if u.shape != (ngrid,) or v.shape != (ngrid,):
        raise ValueError("u and v must have shape (Ngrid,)")
    if u_tendency is not None and u_tendency.shape != (ngrid,):
        raise ValueError("u_tendency must have shape (Ngrid,)")
    if v_tendency is not None and v_tendency.shape != (ngrid,):
        raise ValueError("v_tendency must have shape (Ngrid,)")

    # surfexcdriver_ctl_mod.F90 943-949: PCFMLEV accumulates tiles
    # sequentially from zero.
    pcfm = torch.zeros(ngrid, dtype=pcfm_tile.dtype, device=pcfm_tile.device)
    for jt in range(pcfm_tile.shape[1]):
        pcfm = pcfm + pcfm_tile[:, jt] * tile_fraction[:, jt]

    # vdfmain1s.F90 766: ZCFM = ZCFM*RG*PTSPHY*RVDIFTS, left associative.
    zcfm = ((pcfm * rg) * dt) * rvdifts
    # vdfdifm1s.F90: ZTPFAC2 = 1/RVDIFTS; PUDIF = ZTPFAC2*PUM1 + PTMST*PVOM.
    ztpfac2 = 1.0 / rvdifts
    udif = ztpfac2 * u if u_tendency is None else ztpfac2 * u + dt * u_tendency
    vdif = ztpfac2 * v if v_tendency is None else ztpfac2 * v + dt * v_tendency
    # vdfincr.F90 184: PSTRTU = ZCONS2*PCFM*PUDIF, left associative,
    # with ZCONS2 = 1/(RG*PTMST) formed as a scalar division.
    zcons2 = 1.0 / (rg * dt)
    return OfflineVDFMomentum(
        stress_u=(zcons2 * zcfm) * udif,
        stress_v=(zcons2 * zcfm) * vdif,
    )


__all__ = ["OfflineVDFMomentum", "vdfdifm1s_surface_stress"]
