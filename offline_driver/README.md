# `offline_driver/` — Fortran 参考工具（M2）

本包是 OpenIFS 48r1 陆面方案 PyTorch 移植的 **M2 验证工具**。它封装了移植代码
所实现的核心地表物理核的*独立* gfortran 构建，并通过 Python ctypes 桥接暴露出来，
使移植代码可对照一个与它**不共享任何代码**的 Fortran 参考来校验。当前 `surf_ref.F90`
共导出 **11 个核**：`VLAMSK`（皮肤导热）、`SEB`（能量平衡）、`VEXCS`（Monin-Obukhov
交换）、`DQSDT`（饱和 Jacobian）、`STOMATA`（Jarvis 气孔）、`VEVAP`（α 闭合）、
`VDFDIFH`（隐式垂直扩散）、`SOILT`（土壤温度）、`SOILW`（土壤湿度）、`SNOW`（单层雪）、
`FARQUHAR`（Farquhar 光合 + Ball-Berry）。

## 包含内容

```
offline_driver/
├── README.md            ← 本文件
├── __init__.py          ← 导出 SurfRef
├── build_ref.sh         # gfortran 一行命令构建 libsurf_ref.so
├── surf_ref.py          # ctypes 封装（SurfRef.vlamsk / seb / vexcs / dqsdt / stomata /
│                        #   vevap / vdfdifh / soilt / soilw / snow / farquhar）
├── libsurf_ref.so       # 构建产物（由 build_ref.sh 重新生成）
└── fortran/
    └── surf_ref.F90     # 独立 Fortran 源码（无 MPI/NetCDF 等）
```

## 为什么要单独一份参考？

移植代码自带的单元测试（`tests/test_seb.py`）使用纯 Python 的单瓦片参考与解析
闭合校验。这些测试有价值，但共享*移植代码*的算术顺序，因此一次抄录错误（如符号取反）
可能在两处都通过。本处的 Fortran 参考是上游各核源（`vlamsk_mod.F90`、
`surfseb_ctl_mod.F90`、`vexcs_mod.F90`、`vsurf_mod.F90`、`vevap_mod.F90`、
`vdfdifh1s.F90`、`surftstp_ctl_mod.F90`、`srfsn_lwimps_mod.F90`、`farquhar_mod.F90`
等）的一份全新抄录，用裸 gfortran 编译——两者一致即为移植代码复现了上游 IFS 算术的
有力证据。

选择独立设计而非封装既有 `osmMASTER` 可执行文件，原因是：

* **无外部依赖** —— 仅需 gfortran，无 MPI / NetCDF / ecbuild / namelist 文件；
* **精确的误差定位** —— `.so` 每个核暴露一个子例程（`surf_ref_vlamsk`、`surf_ref_seb`、
  `surf_ref_vexcs` … `surf_ref_farquhar`，共 11 个），失败的测试直接指向具体例程；
* **极易审计** —— `surf_ref.F90` 约 3100 行直接抄录，不涉及任何 IFS 构建系统。

## 构建

```bash
./build_ref.sh                 # release 构建，libsurf_ref.so
./build_ref.sh debug           # -O0 -g -fcheck=all -fbacktrace
./build_ref.sh clean           # 移除构建产物
FC=gfortran-13 ./build_ref.sh  # 选择不同的编译器
```

构建仅需 `gfortran`，在本脚本旁产出一个 `libsurf_ref.so`。构建还会运行 `nm`
验证 11 个预期公开符号已导出：`surf_ref_vlamsk`、`surf_ref_seb`、`surf_ref_vexcs`、
`surf_ref_dqsdt`、`surf_ref_stomata`、`surf_ref_vevap`、`surf_ref_vdfdifh`、
`surf_ref_soilt`、`surf_ref_soilw`、`surf_ref_snow`、`surf_ref_farquhar`
（对应 `surf_ref.F90:91-94` 的 `PUBLIC` 声明）。

## 从 Python 调用

```python
from offline_driver import SurfRef
from surf_pytorch import ConstantBundle, SurfConfig
from surf_pytorch.tables import VegTable
from surf_pytorch.physics.seb import SebInputs, surf_seb
from surf_pytorch.physics.vlamsk import vlamsk

ref = SurfRef()
veg = VegTable(dtype=torch.float64)
consts = ConstantBundle(dtype=torch.float64)
cfg = SurfConfig(dtype=torch.float64)

# VLAMSK
lamsk_ref = ref.vlamsk(tskm1m, tsrf, snm, rsn, ktvl, ktvh,
                       veg, consts, cfg, ptstep=3600.0)

# SEB（必须提供 lamsk；让两套实现在同一取值上同步）
out_ref = ref.seb(seb_inputs, veg, consts, cfg, lamsk=lamsk_ref)
out_port = surf_seb(seb_inputs, consts, cfg)
```

封装从移植代码消费的同一 `ConstantBundle` 取所有物理常数，因此两套实现共享完全
相同的参数化——任何不一致都纯粹源于算术差异。

## 测试

`tests/test_fortran_ref.py` 运行对比，并注册在 `fortran_ref` pytest 标记下。
当 `.so` 缺失时（如在无 gfortran 的 CI 上），测试干净跳过。

```bash
python -m pytest tests/test_fortran_ref.py -v
python -m pytest -m fortran_ref             # 标记筛选
```

在以下情形下断言 ULP 级一致（最大 `|diff| < 1e-9`，通常精确为 `0.0`）：

* 默认与随机化输入；
* 海洋 / 海冰 / 湿肤 / 植被 / 雪 / 裸土瓦片；
* 弱 vs 强湍流交换、昼 vs 夜辐射；
* 多种优势植被类型；
* LELWTL 能量守恒 LW 重算分支。

## 面向未来里程碑的设计说明

* 新增一个核（如 M4 的 Monin-Obukhov 迭代）意味着：
  1. 在 `surf_ref.F90` 中写一个 `SURF_REF_<NAME>` 子例程；
  2. 在 `SurfRef` 上加一个加载 + 配置 + 调用它的方法；
  3. 在 `tests/test_fortran_ref.py` 加一个对比测试类。
* Fortran/Python 签名契约仅存在于 ctypes 的 argtypes——更改任一侧时，请让逐参数的
  `# KLON, NTILES` 注释与 `.F90` 哑元列表保持同步。
* 每个 2-D 数组上的 `np.asfortranarray` 是*必需的*——Fortran 以首维 `KLON`
  索引 `X(JL, JT)`，因此缓冲区必须是列主序。

## 环境变量覆盖

* `SURF_REF_SO=/path/to/libsurf_ref.so` —— 树外 `.so` 位置。
* `FC=gfortran-13` —— `build_ref.sh` 的编译器覆盖。
