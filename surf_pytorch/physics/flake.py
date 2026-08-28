"""FLake 湖泊模型占位桩（``FLAKE_DRIVER``）。

本模块是 ``FLAKE_DRIVER`` 的占位移植——``SURFTSTP_CTL``（第 4.7 节，
surftstp_ctl_mod.F90:795-818）在 namelist 开关 ``LEFLAKE`` 为
``.TRUE.`` 时使用的淡水湖泊模型。离线驱动冷启动时
``LEFLAKE = .FALSE.``（*sudim1s.F90*），因此改为运行上游第 808-818
行的 ``ELSE`` 分支：每个预报量保持其 t-1 值（恒等更新）。

真实 FLake 的功能
------------------------
FLake（Mironov 2008）是湖泊混合层与温跃层的双层参数化。它在每个湖
点积分七个预报量：

* ``PTLICE``  -- 空气-冰 / 雪-冰界面温度 [K]
* ``PTLMNW``  -- 水柱平均温度 [K]
* ``PTLWML``  -- 混合层温度 [K]
* ``PTLBOT``  -- 湖底温度 [K]
* ``PTLSF``   -- 冰下表面表皮温度 [K]
* ``PHLICE``  -- 湖冰厚度 [m]
* ``PHLML``   -- 混合层深度 [m]

驱动接受地表强迫（净短波 / 长波、感热 / 潜热、动量应力）加上湖深与
纬度，返回 t+1 预报量。它由湖点掩码（``LDLAKE``）门控，非湖点被
完全跳过。

本桩的功能
-------------------
函数 :func:`flake_driver` 只复现上游的 ``ELSE`` 分支：将每个 t-1 预
报量复制到 t+1 输出并返回，忽略地表强迫。这正是离线驱动默认的行为，
因此一旦完整的 FLake 转写落地，将 ``SurfConfig.use_flake`` 从
``False``（当前状态，调用本桩）切换为 ``True`` 将是*纯*替换——
输入 / 输出 / 调用点都不会变动。

未来的里程碑将把真实的 ``FLAKE_DRIVER`` 主体（flake_driver_mod.F90
及 flakeene / flakerad 伴随例程）移植进本文件。在此之前，恒等更新与
上游 ``LEFLAKE = .FALSE.`` 分支逐位一致，并由单元测试
:mod:`tests.test_forward_m6` 检验。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
# Input / output containers
# --------------------------------------------------------------------------- #
@dataclass
class LakeInputs:
    """t-1 湖泊预报量 + 地表强迫（FLAKE_DRIVER 哑元列表）。

    所有张量均为 ``(Ngrid,)``。预报量字段镜像 ``FLAKE_DRIVER`` 的
    ``PT* M1M`` / ``PH* M1M`` 参数；强迫字段镜像 ``PSSRFLTI`` 等。
    强迫被接受但被桩忽略——它的存在是为了让签名与最终的真实驱动
    一致。
    """

    # t-1 prognostics (the upstream P* M1M arguments).
    ptlice_m1m: torch.Tensor   # air-ice interface T [K]
    ptlmnw_m1m: torch.Tensor   # mean water-column T [K]
    ptlwml_m1m: torch.Tensor   # mixed-layer T [K]
    ptlbot_m1m: torch.Tensor   # lake-bottom T [K]
    ptlsf_m1m:  torch.Tensor   # lower-ice-surface skin T [K]
    phlice_m1m: torch.Tensor   # lake-ice thickness [m]
    phlml_m1m:  torch.Tensor   # mixed-layer depth [m]

    # Forcing (accepted, ignored by the stub).
    pssrflti: torch.Tensor     # net SW radiation at surface [W/m^2]
    pslrfl:   torch.Tensor     # net LW radiation at surface [W/m^2]
    pahfsti:  torch.Tensor     # sensible heat flux downwards [W/m^2]
    pevapti:  torch.Tensor     # latent heat flux downwards [W/m^2]
    pustrti:  torch.Tensor     # x-momentum flux [N/m^2]
    pvstrti:  torch.Tensor     # y-momentum flux [N/m^2]

    # Static descriptors.
    pldepth:  torch.Tensor     # lake depth [m]
    pgemu:    torch.Tensor     # sin(latitude)
    ld_lake:  torch.Tensor     # lake-point mask (bool)
    ptstep:   float            # coupling interval [s]


@dataclass
class LakeOutputs:
    """t+1 湖泊预报量（上游的 ``ZT*`` / ``ZH*`` 局部变量）。"""

    ptlice: torch.Tensor       # air-ice interface T [K]
    ptlmnw: torch.Tensor       # mean water-column T [K]
    ptlwml: torch.Tensor       # mixed-layer T [K]
    ptlbot: torch.Tensor       # lake-bottom T [K]
    ptlsf:  torch.Tensor       # lower-ice-surface skin T [K]
    phlice: torch.Tensor       # lake-ice thickness [m]
    phlml:  torch.Tensor       # mixed-layer depth [m]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def flake_driver(inp: LakeInputs) -> LakeOutputs:
    """将 FLake 预报量推进一个耦合步长。

    **占位桩实现**——复现 ``SURFTSTP_CTL`` 上游的 ``ELSE`` 分支
    （surftstp_ctl_mod.F90:808-818）：每个预报量保持其 t-1 值。与
    离线驱动默认（``LEFLAKE = .FALSE.``）逐位一致；地表强迫被接受
    但不使用。

    一旦真实的 FLake 主体转写完成，本函数是唯一改变的部分——签名、
    数据类以及 :meth:`LandSurface._update_lake` 中的调用点保持不变。
    """
    return LakeOutputs(
        ptlice=inp.ptlice_m1m.clone(),
        ptlmnw=inp.ptlmnw_m1m.clone(),
        ptlwml=inp.ptlwml_m1m.clone(),
        ptlbot=inp.ptlbot_m1m.clone(),
        ptlsf=inp.ptlsf_m1m.clone(),
        phlice=inp.phlice_m1m.clone(),
        phlml=inp.phlml_m1m.clone(),
    )
