"""海洋混合层模型占位桩（``OCEAN_ML_DRIVER`` / KPP）。

本模块是 ``OCEAN_ML_DRIVER`` 的占位移植——K 廓线参数化海洋混合层
模型（Large, McWilliams & Doney 1994），由 ``SURFTSTP_CTL`` 在逐点
掩码 ``LDOCN_KPP`` 为 ``.TRUE.`` 时调用。离线驱动冷启动时
``LEOCSA = .FALSE.``（*su0phy1s.F90:107*），这迫使
``LDOCN_KPP = .FALSE.`` 处处成立（surfbc_ctl_mod.F90:262），因此
KPP 驱动从不运行，SST 改为被规定。

真实 KPP 驱动的功能
-----------------------------
``OCEAN_ML_DRIVER`` 在固定深度网格（``KLEVO`` 层，默认 8——
*sudim1s.F90*）上积分一维柱状海洋。它在表面通量（净短波 / 长波、
感热 / 潜热、动量应力、对流 + 大尺度雨和雪）作用下推进预报的温度、
盐度以及（可选的）水平流，表面边界层涡扩散率使用 KPP 混合方案
（Large et al. 1994），内部混合基于 Richardson 数。

预报状态存放在 IFS 状态上：海洋层处的 ``PTO0``（温度）、``PSO0``
（盐度）、``PUO0``/``PVO0``（流），加上深度网格 ``PZO`` 与层厚
``PDO``。驱动返回 t+1 状态以及短波穿透诊断。

本桩的功能
-------------------
函数 :func:`ocean_ml_driver` 复现离线驱动的默认行为：将 t-1 SST 原样
返回（恒等更新），忽略地表强迫。这正是离线驱动在
``LEOCSA = .FALSE.`` 时的行为（规定 SST 路径），因此一旦完整的 KPP
转写落地，将 ``SurfConfig.use_kpp`` 从 ``False``（当前状态，调用本
桩）切换为 ``True`` 将是*纯*替换。

未来的里程碑将把真实的 ``OCEAN_ML_DRIVER`` 主体
（ocean_ml_driver_mod.F90 及 kpp_* 伴随例程）移植进本文件。在此之
前，恒等更新与上游 ``LEOCSA = .FALSE.`` 分支逐位一致，并由单元测试
:mod:`tests.test_forward_m6` 检验。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
# Input / output containers
# --------------------------------------------------------------------------- #
@dataclass
class OceanMlInputs:
    """t-1 海洋状态 + 地表强迫（OCEAN_ML_DRIVER 哑元列表）。

    预报量张量为 ``(Ngrid, Nlev_ocean)``；强迫与掩码张量为
    ``(Ngrid,)``。完整的 3-D 状态（温度、盐度、流）被接受但被桩
    忽略——它的存在是为了让签名与最终的真实驱动一致。只有
    ``pto0[:, 0]``（表皮层温度，供给水瓦片表皮）被传播到
    :class:`OceanMlOutputs`。
    """

    # t-1 prognostic state.
    pto0: torch.Tensor         # ocean temperature [K], (Ngrid, Nlev)
    pso0: torch.Tensor         # ocean salinity [psu], (Ngrid, Nlev)
    puo0: torch.Tensor         # ocean u-current [m/s], (Ngrid, Nlev)
    pvo0: torch.Tensor         # ocean v-current [m/s], (Ngrid, Nlev)

    # Static grid.
    pzo: torch.Tensor          # ocean level depths [m], (Ngrid, Nlev+1)
    pdo: torch.Tensor          # ocean layer thicknesses [m], (Ngrid, Nlev)

    # Forcing (accepted, ignored by the stub).
    pssrflti: torch.Tensor     # net SW radiation at surface [W/m^2]
    pslrfl:   torch.Tensor     # net LW radiation at surface [W/m^2]
    pahfsti:  torch.Tensor     # sensible heat flux downwards [W/m^2]
    pevapti:  torch.Tensor     # latent heat flux downwards [W/m^2]
    pustrti:  torch.Tensor     # x-momentum flux [N/m^2]
    pvstrti:  torch.Tensor     # y-momentum flux [N/m^2]

    # Masks / scalars.
    ld_ocn_kpp: torch.Tensor   # ocean-ML active mask (bool)
    ptstep:    float           # coupling interval [s]


@dataclass
class OceanMlOutputs:
    """t+1 海洋状态（上游的 ``PTOE1`` / ``PSOE1`` 局部变量）。

    前向路径只消费表面温度 ``pto0_new[:, 0]``（它固定水瓦片表皮）。
    更深的层次以及盐度 / 流为完整性而返回，以便真实驱动的转写可以
    在不改变输出契约的情况下填充它们。
    """

    pto0_new: torch.Tensor     # t+1 ocean temperature [K], (Ngrid, Nlev)
    pso0_new: torch.Tensor     # t+1 ocean salinity [psu], (Ngrid, Nlev)
    puoe1:    torch.Tensor     # t+1 u-current [m/s], (Ngrid, Nlev)
    pvoe1:    torch.Tensor     # t+1 v-current [m/s], (Ngrid, Nlev)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def ocean_ml_driver(inp: OceanMlInputs) -> OceanMlOutputs:
    """将海洋混合层状态推进一个耦合步长。

    **占位桩实现**——复现离线驱动默认（``LEOCSA = .FALSE.``）：每个
    预报量保持其 t-1 值。地表强迫被接受但不使用。

    一旦真实的 KPP 主体转写完成，本函数是唯一改变的部分——签名、
    数据类以及 :meth:`LandSurface._update_ocean_ml` 中的调用点保持
    不变。
    """
    return OceanMlOutputs(
        pto0_new=inp.pto0.clone(),
        pso0_new=inp.pso0.clone(),
        puoe1=inp.puo0.clone(),
        pvoe1=inp.pvo0.clone(),
    )
