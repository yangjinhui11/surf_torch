# surf-pytorch

**OpenIFS 48r1 陆面包**（`openifs-48r1/ifs-source/surf/`）的 PyTorch 移植。
设计目标是既可作为可微分的 `nn.Module` 接入 ML 模型，又能复现 Fortran 单柱驱动器
的 bit-level 精度。

> **状态：** 里程碑 **M1–M6** 已完成。完整物理核心已移植，并与独立 Fortran 参考实现
> 完成 bit-level 验证：查找表 / 常数 / 饱和函数（M1）、Fortran 参考工具
> `libsurf_ref.so` 及其 ULP 级核函数对比（M2）、Penman-Monteith 地表能量平衡 +
> VLAMSK 肤层导热（M3）、Louis/Beljaars Monin-Obukhov 交换系数 + 隐式垂直扩散 /
> VDFDIFH1S 封装（M4）、HTESSEL 土壤温度与土壤湿度预报（M5）以及单层雪预报
> （SRFSN_LWIMPS）+ FLake/KPP/Farquhar 冷启动桩（M6）。`forward()` 运行完整的
> 离线驱动器地表步；尚存的冷启动占位（雪密度、优势植被分裂、海冰/湖面肤温）见
> `LandSurface.forward` 文档字符串。在此之上叠加了理想化日变化强迫生成器 + 多步单柱
> 工具（M2c）以及降雪穿透降水的接线（M2d/M2e），用于端到端物理闭合测试。
>
> **地表反照率增强（哨兵门控，休眠时 bit 级一致）：** 城市冠层宽带反照率
> （`CANALB`，Masson/Harman/Porson）与冰盖感知的雪反照率老化
> （`SRFSN_ASN` 陆地+冰川混合，含线性降雪增亮）已移植并接入 `_surface_albedo` /
> `_update_snow_albedo`。仅当提供 `static.urban_frac` / `static.ice_land_frac` 时
> 激活；当这两个哨兵为 `None` 时，传统路径运行方式不变，所有既有测试（含 golden
> 快照）保持 bit 级一致。详见下文 *地表反照率增强*。
>
> **两条真实数据 / 耦合验证轴** 补充了理想化 ULP 测试套件：(1) ecLand ERA5 站点
> 强迫驱动物理健康回归（`test_ecland_driven.py`）；(2) PyTorch 移植通过进程内
> CPython 桥接*内嵌进* OpenIFS 离线驱动器，并与相同强迫的纯 Python 垫片重放
> 进行对比（`test_offline_coupled.py`，由 `OIFS_OGG_PATH` 门控）。详见下文
> *真实数据与耦合验证*。

## 为什么做这个

OpenIFS 陆面方案（HTESSEL + Monin-Obukhov 扩散 + Penman-Monteith 地表能量平衡 +
单层雪）是一个经过充分验证的参考实现，但它以 Fortran 实现，对 ML 训练不透明。
本移植：

* 将地表步暴露为批量化的 `nn.Module`，使物理–ML 混合模型能对其反向传播；
* 把查找表（土壤质地、植被、根系廓度、Monin-Obukhov 稳定度）保留为可冻结 buffer
  或可学习参数；
* 每一步都按逐柱、逐时步对照自编译的 Fortran 离线驱动器验证。

## 目录结构

```
surf_pytorch/
├── pyproject.toml
├── README.md
├── docs/                      # PROGRESS.md（开发日志）+ INTELLIGENT_FORECASTING.md
│                              #   + DATA_ASSIMILATION_4DVAR.md（ML/资料同化研究方向）
├── surf_pytorch/
│   ├── __init__.py            # 公开 API
│   ├── config.py              # SurfConfig + 瓦片索引
│   ├── constants.py           # yomcst / yoethf 常数（离线覆盖值）
│   ├── model.py               # LandSurface nn.Module（完整前向步）
│   ├── state.py               # Forcing / Static / SurfState / SurfDiag
│   ├── forcing.py             # 理想化日变化强迫生成器（M2c）
│   ├── column_harness.py      # 多步单柱运行器 + 结果导出（M2c）
│   ├── utils.py               # qsat / foeew / tridiag_solve / 辅助函数
│   ├── tables/
│   │   ├── soil.py            # SUSSOIL：7 类 USDA + 派生量
│   │   ├── veg.py             # SUSVEG：20 类覆盖类型
│   │   ├── roots.py           # SRFROOTFR：Zeng (1998)
│   │   └── ritbl.py           # SUVEXCS：Monin-Obukhov 稳定度 LUT
│   └── physics/               # seb / diffusion / exchange / soil / snow /
│       │                        canalb（城市）/ snow_albedo（SRFSN_ASN）...
│       ├── canalb.py          # 城市冠层反照率（CANALB，哨兵门控）
│       └── snow_albedo.py     # SRFSN 传统路径 + SRFSN_ASN 冰川分支
├── coupling/                  # PyTorch↔OpenIFS 离线驱动器桥接（方案 A）
│   ├── build_offline_coupled.sh   # 构建 osmMASTER_coupled 可执行文件
└── tests/                     # 单元 + Fortran 对比（ULP）+ ecland + 耦合测试
```

## 安装

```bash
cd surf_pytorch
pip install -e .[test]
```

依赖 `torch>=2.0`、`numpy>=1.22`、Python ≥ 3.9。

## 快速上手

```python
import torch
from surf_pytorch import LandSurface, SurfConfig

model = LandSurface(SurfConfig(dtype=torch.float64))
# state, forcing, static = ...  # 见 surf_pytorch/state.py
# next_state, diag = model(state, forcing, static)  # 完整离线驱动器步
```

`SurfConfig` 默认值与离线单柱驱动器一致：4 层土壤、8 个瓦片、Jarvis 气孔阻抗
（非 Farquhar）、单层雪、`float64`。如需端到端多步运行，有两种入口：

* **理想化自洽强迫**（无需外部数据）：见
  [`examples/run_idealized_diurnal.py`](examples/run_idealized_diurnal.py)
  以及 `ColumnHarness` + `build_idealized_diurnal` API
  （`surf_pytorch/forcing.py`、`surf_pytorch/column_harness.py`）。
* **真实 ERA5/ecLand 站点强迫**：见 [`examples/run_ecland.py`](examples/run_ecland.py)
  （用法与驱动数据制作见下文 *ecLand ERA5 站点强迫* 一节）。

## 设计说明

### 与 Fortran 的 bit 级一致

* 每个标量常数都逐字抄录自 `surf/offline/driver/sucst.F90` 与 `suphec.F90`，包括
  **离线驱动器覆盖值**（与标准 IFS 取值不同的部分）：
  - `RVTMP2 = 0.0`（忽略湿空气 cp 修正）
  - `R2ES = 611.21 * RD/RV`
  - `RTICE = RTT - 23.0`
* 饱和函数（`foeew`、`foedelta`、`foealfa`、`foeewm`、`foede`、`foedesu`）逐行对应
  `fcttre.h` 语句函数。
* 肤温 Jacobian `dqsat_dt_ifs` 对应上游 `vsurf_mod.F90:284` 的 PDQS 形式
  （`PQS=FOEEW(T)/P`，`ZCOR=ZSALIN/(1-RETV*PQS)`，`PDQS=PQS*ZCOR*FOEDESU(T)`），
  即该导数使用 **ZCOR 的平方** 而非解析 dqs/dT。海洋盐度因子 `ZSALIN=0.98`
  （仅当 `LEOCSA=.T.` 时激活，否则为 1.0）通过 `qsat(..., salinity=)` 与
  `dqsat_dt_ifs(..., salinity=)` 贯通。`SurfModel` 中的 SEB 路径经此入口计算
  `dqsdT`，使残差匹配 Fortran 参考而非精确解析导数。该公式由 Fortran 核
  `SURF_REF_DQSDT`（`offline_driver/fortran/surf_ref.F90`）**独立** 验证——
  该核从上游 `fcttre.h` 语句函数重新推导 PDQS，而非消费 Python 值；
  `tests/test_dqsdt_ref.py` 在水/冰/盐度各区域断言 0-ULP 一致。（SEB 参考核
  `SURF_REF_SEB` 自身无法做到——它把 `DQSDT` 作为输入，因此那里的比较将退化为
  Python 公式的自我抄录。）
* 瓦片 / 土壤 / 植被查找表严格遵循 `SUSSOIL`、`SUSVEG`、`SRFROOTFR`、`SUVEXCS`，
  包括索引 0 哨兵——植被表对应类型 8（荒漠），土壤表对应类型 0（水体）。

### 端到端装配验证

上述逐核 Fortran 校验证明每个地表例程（VLAMSK / VEXCS / DQSDT / SEB）在隔离状态下
与上游算术一致。`tests/test_forward_e2e.py` 在其上再加一层：向 `LandSurface.forward`
与一个参考装配输入相同的 forcing / static / state——后者复用移植代码的输入构造胶水
（辐射单位换算、瓦片分数广播、alpha 闭合），但把每个物理核路由到 `libsurf_ref.so`。
肤温推进（ULP）与全部五项逐瓦片通量的一致性，即为**装配**（核之间的接线、
逐网格 → 逐瓦片广播、场到 `SebInputs` 的映射）内部自洽的证据。这类 bug 是逐核测试
捕获不到的（例如在装配中交换 `tskm1m`/`tsrf` 会让残差膨胀到约 10^16 ULP，而逐核测试
仍会保持绿色）。

### 批量化、网格原生

每个场的首轴是**网格 / 柱维度**（`Ngrid`）；瓦片与土壤层轴附加其后。单柱运行即
`Ngrid=1`。

### 按需可微

* 默认 `SurfConfig.differentiable=False`：物理在 `torch.no_grad()` 下运行以提速。
* `differentiable=True`：相同代码在 `enable_grad` 下运行；查找表索引与
  `min/max/sign` 分支通过 `torch.where` 解耦，使梯度图为分段光滑。
* 查找表默认是不可学习的 buffer；选定列可通过 `SurfConfig.learnable_soil` /
  `learnable_veg` 提升为 `nn.Parameter`。

## 验证

PyTorch 移植对照**独立 Fortran 参考实现**（`offline_driver/fortran/surf_ref.F90`）
验证——这是一份全新、gfortran 编译的上游 `vlamsk_mod.F90` 与 `surfseb_ctl_mod.F90`
抄录，与移植代码**不共享任何 Python 代码**。故意避开庞大的 `master1s` 驱动器：它
引入 MPI / NetCDF / namelist / 派生类型机制，会使精确的、核级定位的对比复杂化。

用 `offline_driver/build_ref.sh` 构建参考（仅 gfortran），然后运行 ULP 级对比测试：

```bash
cd surf_pytorch
offline_driver/build_ref.sh                        # 构建 libsurf_ref.so
python -m pytest tests/test_fortran_ref.py -v      # 17 项 SEB 测试，diff < 1e-9
```

每个 SEB 输出场（逐瓦片通量 + 肤温 + 网格均值最低层 `Sl`/`Ql`）在默认、随机化、
昼/夜、弱/强湍流与多植被类型等情形下，均与 Fortran 参考匹配到末位 ULP（通常精确为
`0.0`）。

同样的模式应用到每个其他已移植核——每个都有专门的 `libsurf_ref.so` 抄录
（`SURF_REF_VLAMSK`、`_VEXCS`、`_DQSDT`、`_STOMATA`、`_VEVAP`、`_VDFDIFH`、
`_SOILT`、`_SOILW`、`_SNOW`、`_FARQUHAR`，共 11 个核）与一个断言约 0-ULP 一致
（Farquhar 因含三角函数为 1e-12 相对容限）的 `test_*_ref.py`。在逐核校验
之上，`tests/test_forward_e2e.py` 向 `LandSurface.forward` 与一个将每个核路由进 `.so`
的参考装配输入相同 forcing/state，断言完整 `forward()` 肤温推进与逐瓦片通量一致
（ULP）。端到端物理闭合（水量/能量平衡到机器精度）由 `tests/test_m2c_harness.py`
经理想化日变化生成器 + 多步 `ColumnHarness` 锁定。完整设计说明见
`offline_driver/README.md`。

## 地表反照率增强

两个反照率子系统在 bit 级核心之上移植。两者都遵循**哨兵门控激活**模式：除非调用方
通过设置 `Static` 哨兵字段显式开启，否则它们休眠（且与增强前的构建 bit 级一致），
因此当哨兵缺失时既有 golden 快照与所有传统测试不变。

### 城市冠层反照率（`CANALB`）

`surf_pytorch/physics/canalb.py` 移植 `canalb_mod.F90`（Masson 2000 太阳辐射分配 +
Harman 2004 形状因子 + Porson 2010 参考通量 + 余因子矩阵求逆）。冠层反照率 `PCANALB`
在 `_urban_canyon_albedo`（`surfrad_ctl_mod.F90:597-602`）中通过道路占比 `RWRR`
与屋顶反照率 `RROOALB` 混合：

```python
alpha_urban = RROOALB/(1+RWRR) + PCANALB * RWRR/(1+RWRR)
```

激活条件：设置 `static.urban_frac`（PCUR）**并** 在 `n_tile >= 10` 下运行
（城市瓦片为 `TILE_URBAN`，0 基索引 9）。城市瓦片分数在 `_compute_tile_frac`
（`surfbc_ctl_mod.F90:385-396`）中装配——地表覆盖被 `(1-PCUR-ZFRTIT)` 重新缩放，以
保持和为 1 的不变量。冠层材料/几何字段（`roof_alb`、`road_alb`、`wall_alb`、
`urban_height_ratio`、`urban_road_ratio`）在未设置时回退到 YOS_URB 默认值
（`0.16 / 0.05 / 0.6 / 1.0 / 1.0`）。

当 `urban_frac=None` 时，休眠门 `has_urban = (ntile > TILE_URBAN) and
(static.urban_frac is not None)` 为 `False`，`_surface_albedo` 永不调用 CANALB——
城市列保持为 0。

### 冰盖感知雪反照率（`SRFSN_ASN`）

`snow_albedo.py` 现按 `SnowAlbedoInputs.pcil` 分派：

* `pcil is None` → 传统 `SRFSN` 路径（无雪重置 + 融化/低温衰减 + RSFRESH 硬重置），
  与原移植**逐 bit 一致**。
* 提供 `pcil` → 完整 `SRFSN_ASN` 方案（`srfsn_asn_mod.F90` 133-179）：新反照率拆分为
  **陆地部分** `ZASN_L`（相同的融化/低温衰减，但用*线性*降雪增亮取代 RSFRESH 硬重置）
  与**冰川部分** `ZASN_I`（向 `ZALBMIN=0.65` 融化衰减，`ZRTAUF_GL=0.085`；冷冰川保持
  `PASNM1M`；半球相关的降雪重置：北半球 5 mm / 南半球 4 mm），按
  `PASN = (1-PCIL)*ZASN_L + PCIL*ZASN_I` 混合。

冰川分支字段（`pcil`、`ptsnm1m_top`、`psnowf`、`ldnh`）仅当 `static.ice_land_frac`
设置时，在 `_update_snow_albedo` 中分别从 `static.ice_land_frac`、`state.snow_t`、
`forcing.snow/dt`、`static.lat>0` 接线。

### 测试

`tests/test_canalb.py`（12）、`tests/test_urban_albedo.py`（14）以及追加到
`tests/test_snow_albedo.py` 的 `SRFSN_ASN` 测试类（+15）覆盖：解析参考值、`[0,1]` 边界、
天顶角/长宽比依赖、瓦片分数和为 1 不变量、休眠 bit 级一致回退、SEB 反馈、冰川融化/低温
分支、线性降雪增亮、PCIL 混合。一个 `torch.linalg.solve` 神谕交叉校验 CANALB 余因子
逆。

## 真实数据与耦合验证

两条互补的验证轴在理想化 ULP 套件之外检验移植。两者均为**门控**：在缺少可选数据/构建的
环境中干净跳过，因此默认 `pytest tests/` 运行不受影响。

### ecLand ERA5 站点强迫（物理健康回归）

`tests/test_ecland_driven.py` 用 ecLand 教程数据集（ECMWF ecLand v2.0.0，
`thirdpaty/ecland-main/`）中的**真实** ERA5 近地面强迫驱动 `LandSurface.forward`。
两个站点切片均来自荷兰 Cabauw 站、按不同季节切片：冬季 `CA-001`（25 个逐时步，
1 月中旬，冷、短波低）与早春 `VG-000`（43 个逐时步）。每个站点的三组文件命名带不同
前缀（**注意不是统一通配符**），实际路径为：

```
thirdpaty/ecland-main/tutorials/ecland_input_tutorial/
├── forcing/LS-PRACTICALS/met_era5HT_<SITE>_2022-2022.nc      # ERA5-Hourly-Tiled 强迫
└── clim/LS-PRACTICALS/
    ├── surfclim_<SITE>_2022-2022.nc                           # 静态描述（lat/tvl/sotype…）
    └── surfinit_<SITE>_2022-2022.nc                           # 初始状态（SoilTemp/SoilMoist…）
```

强迫文件携带 `Tair / Qair / Wind_E / Wind_N / SWdown / LWdown / Rainf / Snowf / PSurf`
（注意是**速率**量：辐射 W/m^2、降水 kg/m^2/s），加上匹配的 `surfclim` 静态描述与
`surfinit` 初始状态。测试用 `_DT_SECONDS = 3600`（逐时）将速率换算为移植代码
`Forcing` 所需的**累积量**（J/m^2、kg/m^2），净长波由 `LWdown - emis·σ·T^4` 重构
（向下为正）。

由于 ecLand Fortran 模式在此**未构建**，该测试是*物理健康*回归（无 golden 文件）：
断言运行推进无 NaN、保持在物理边界内、闭合地表能量/水量平衡、并产生可区分的
Jarvis 与 Farquhar 肤温响应。单位/符号约定（累积量-速率辐射、净-向下 LW、1 基 ecLand
类型索引）在模块文档字符串中统一。

**两种运行方式：**

```bash
cd surf_pytorch

# (1) 测试套件（断言物理健康，无输出文件）
python -m pytest tests/test_ecland_driven.py -v

# (2) 独立运行脚本（驱动真实数据 + 导出 CSV 时序 + 打印能量/水量平衡）
python examples/run_ecland.py                              # 默认 CA-001
python examples/run_ecland.py --site VG-000                # 换站点
python examples/run_ecland.py --farquhar                   # Farquhar 闭合
python examples/run_ecland.py --out /tmp/run               # 自定义输出目录
```

`examples/run_ecland.py` 是 `tests/test_ecland_driven.py` 的可运行对应物：加载某站点的
三个 NetCDF（强迫 + 静态 + 初始状态），逐时步推进完整 `LandSurface.forward`，并把每步的
网格均值时序（肤温、4 层土壤温湿、四项通量 netLW/H/LE/G0、2 m T/Q、净辐射）写入
`trajectory.csv`，输入强迫（单位换算后）写入 `forcing_grid_mean.csv`。运行结束打印整段
运行的肤温日变幅、最亮步的能量平衡残差、累积水量平衡残差，便于肉眼判断物理健康。
脚本的所有数据路径均可配置（见下文 *制作新站点驱动数据*）。

**数据前置条件**（缺任一项则测试模块跳过、脚本友好报错退出）：

* `xarray` + 可读 NetCDF 栈（缺失时测试以 `importorskip` 跳过，脚本提示安装命令）；
* 上述三组 ecLand NetCDF 解压在 `thirdpaty/ecland-main/` 之下。该数据**不**由
  `thirdpaty/install_thirdparty.sh` 提供（该脚本只装 atlas/eccodes/ecbuild 等构建
  依赖，不含 ecland）；从仓库自带的 `thirdpaty/ecland-main.zip` 解压即可获得：

  ```bash
  cd thirdpaty
  unzip ecland-main.zip          # 生成 thirdpaty/ecland-main/
  ```
  或从上游 `git clone https://github.com/ecmwf-ifs/ecland --branch main` 后放入
  `thirdpaty/`（目录名需为 `ecland-main`）。

#### 制作新站点驱动数据

要跑自己的站点（而非打包的 Cabauw 切片），需要准备**三个 NetCDF 文件**，字段清单如下
（即 `examples/run_ecland.py` / `tests/test_ecland_driven.py` 加载的最小集，缺任一字段
会 `KeyError`）。命名沿用 ecland 约定：`met_era5HT_<SITE>_<年-年>.nc` /
`surfclim_<SITE>_<年-年>.nc` / `surfinit_<SITE>_<年-年>.nc`。

**① 强迫文件** `met_era5HT_<SITE>.nc`（逐时 ERA5-Hourly-Tiled，1-D `time` 维，**速率**量）

| 变量 | 单位 | 说明 |
|------|------|------|
| `Tair` | K | 2 m 气温 |
| `Qair` | kg/kg | 2 m 比湿 |
| `Wind_E`、`Wind_N` | m/s | 10 m 风东西/南北分量 |
| `PSurf` | Pa | 地面气压 |
| `SWdown` | W/m² | 向下短波（代码会 ×dt 转累积量） |
| `LWdown` | W/m² | 向下长波（代码用 `LWdown - emis·σ·T⁴` 重构净长波） |
| `Rainf` | kg/m²/s | 降雨率 |
| `Snowf` | kg/m²/s | 降雪率 |
| `time` | datetime64 | 中间时刻戳（UTC，用于算太阳天顶角 μ0） |

**② 静态文件** `surfclim_<SITE>.nc`（标量）：`lat`/`lon`（度）、`tvl`/`tvh`（低/高植被类型，
**1 基**索引）、`sotype`（土壤类型，1 基）、`cvl`/`cvh`/`cu`（低/高/城市覆盖分数）、
`landsea`（陆地分数）、`geopot`（地球位势）、`z0m`、`lz0h`（log 粗糙度）、`zuv`、`zphista`。

**③ 初始文件** `surfinit_<SITE>.nc`：`SoilTemp`（4 层土壤温度，K）、`SoilMoist`
（4 层土壤湿度，体积分数）、`AvgSurfT`（平均肤温，K）、`CanopInt`（冠层截留，kg/m²）、
`SWE`（雪水当量，kg/m²）。

**制作路径（按推荐度）：**

* **(A) 复用 ecland 官方 create_forcing 工具**（推荐）。仓库已附带
  `thirdpaty/ecland-main/tools/create_forcing/`，它从 ERA5（MARS 或 CDS）提取站点强迫，
  产出的文件命名与格式正好与本脚本期望一致：

  ```bash
  cd thirdpaty/ecland-main/tools/create_forcing
  python ecland_create_forcing.py --info          # 查看依赖与用法
  python ecland_create_forcing.py -c <config.yaml>  # 按 yaml 下载+制作
  ```
  依赖：eccodes、metview、metview-python、cds api、pyYAML、netcdf4、cdo、nco。

* **(B) 用 cdsapi 自行下载 ERA5 单点再对齐字段**。免装 metview/cdo，但需手动处理单位
  （ERA5 累积量 J/m²→W/m²、m of water→kg/m²/s）与构造 surfclim 的类型字段。字段名须重
  命名为上表（如 `t2m→Tair`、`sp→PSurf`、`tp→Rainf`）。

* **(C) 最快验证链路**：拷贝现有站点改名，先把加载/运行跑通再换真实数据：

  ```bash
  cd thirdpaty/ecland-main/tutorials/ecland_input_tutorial
  SITE=ZZ-999
  cp forcing/LS-PRACTICALS/met_era5HT_CA-001_2022-2022.nc forcing/LS-PRACTICALS/met_era5HT_${SITE}_2022-2022.nc
  cp clim/LS-PRACTICALS/surfclim_CA-001_2022-2022.nc      clim/LS-PRACTICALS/surfclim_${SITE}_2022-2022.nc
  cp clim/LS-PRACTICALS/surfinit_CA-001_2022-2022.nc      clim/LS-PRACTICALS/surfinit_${SITE}_2022-2022.nc
  python -c "import sys; sys.argv=['x','--site','$SITE']; exec(open('$(git rev-parse --show-toplevel 2>/dev/null)/surf_pytorch/examples/run_ecland.py').read())"
  ```
  或直接显式传三个路径：`python examples/run_ecland.py --met <p> --clim <p> --init <p>`。

**易错点**（单位/符号不对会算错但不报错）：辐射与降水必须是**速率**（W/m²、kg/m²/s），
代码乘 `dt=3600` 转累积量；`LWdown` 是**向下**长波，代码自己减 `emis·σ·T⁴` 重构净长波；
类型索引是 **1 基**，直接喂 LUT 不要 `-1`；`time` 维按 UTC 中间时刻戳用于算 μ0。


`make_offline_run_dir.py` 统一两个世界的单位/符号约定：它从 PyTorch 侧使用的同一个
`build_idealized_diurnal` 生成器发出强迫 NetCDF 模式（`sufcdf.F90` / `rdfvar.F90`），
使 Fortran 驱动器与模型看到*完全相同*的输入——这正是让对比有意义的关键。

## 路线图

| 里程碑 | 状态 | 内容 |
|--------|------|------|
| M1 | ✅ | 查找表、常数、饱和函数、脚手架 |
| M2 | ✅ | 独立 Fortran 参考（`libsurf_ref.so`）；ULP 级核 + `forward()` 装配对比工具 |
| M3 | ✅ | 地表能量平衡（Penman-Monteith）+ 肤层导热（VLAMSK） |
| M4 | ✅ | Louis/Beljaars Monin-Obukhov 交换系数（VEXCS）；隐式垂直扩散（VDFDIFH1S 封装） |
| M5 | ✅ | HTESSEL 土壤温度（SRFRCGS + SRFTS）与土壤湿度（SRFWEXC_VG + SRFWDIF + SRFWINC + SRFWNG）预报 |
| M6 | ✅ | 单层雪（SRFSN_LWIMPS）接入 `forward()`；FLake/KPP/Farquhar 冷启动桩 |
| M2c | ✅ | 理想化日变化强迫生成器 + 多步单柱工具（水量/能量平衡闭合、golden 快照、CSV 导出） |
| M2d/M2e | ✅ | 降雪穿透降水接线（`forcing.snow → PSSFL`）；单步 bit 级 + 多步物理闭合测试 |
| 反照率 | ✅ | 城市冠层反照率（`CANALB`）+ 冰盖感知雪反照率（`SRFSN_ASN`）；哨兵门控，休眠时 bit 级一致 |
| 验证 | ✅ | ecLand ERA5 站点强迫健康回归（`test_ecland_driven.py`）+ PyTorch 耦合离线驱动器 `o_gg` 对比（`test_offline_coupled.py`，由 `OIFS_OGG_PATH` 门控） |

> 📖 **智能陆面预报（研究方向）。**
> [`docs/INTELLIGENT_FORECASTING.md`](docs/INTELLIGENT_FORECASTING.md) 论证在可微
> `surf_pytorch` 上开展*物理–数据混合陆面建模*的科学方向：为何不采用端到端 ML、保持守恒律的
> 残差矫正构造、三层研究脉络（可微参数标定 → 经验式结构矫正 → 神经子模型替换）、对可学习 LUT
> 参数与经验方案的精确刻画（Jarvis F1/F2/F3、Van Genuchten、Monin-Obukhov PSI、雪反照率
> 老化），以及开放的方法学问题（非光滑算子下的伴随保真、守恒律作为可微约束、参数误差与结构
> 误差的辨识）。

> 📖 **四维变分资料同化。** 因为 `surf_pytorch` 是原生 `nn.Module`，autograd *就是* 4D-Var
> 所需的伴随算子 $M^*$——无需手写切线性/伴随代码。见
> [`docs/DATA_ASSIMILATION_4DVAR.md`](docs/DATA_ASSIMILATION_4DVAR.md) 的数学等价性论证、
> 控制变量/B 矩阵设计、观测算子 H 清单，以及 DA-1（toy 4D-Var + `gradcheck`）→ DA-4
> （弱约束 / 4D-EnKF）路线图。

## 许可证

Apache-2.0，与 OpenIFS 再分发条款一致。底层 OpenIFS 源码需要有效的 ECMWF / OpenIFS
许可证；支持信息见 `openifs-48r1/README`。
