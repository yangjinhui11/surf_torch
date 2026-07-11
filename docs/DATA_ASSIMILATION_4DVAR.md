# 基于 surf 的四维变分陆面资料同化 — 实施方案与建议

> 本文档对应"第二类工作"：在 `surf_pytorch`（OpenIFS 48r1 陆面包的 PyTorch 移植，
> 经 bit-level 验证）之上构建**四维变分（4D-Var）陆面资料同化系统**。
>
> 核心论点：**`surf_pytorch` 作为 PyTorch `nn.Module`，autograd 天然提供了 4D-Var
> 所需的切线性（TLM）与伴随（adjoint）算子**，免去了传统陆面同化最痛苦的"手写伴随"环节——
> 这是相对 Fortran 实现的根本性工程红利。但"天然"二字有几个必须正视的技术边界。
>
> 全部内容基于对 `surf_pytorch/` 现有代码的实测核验（行号均截至 2026-06-29）。
> 配套文档：[`INTELLIGENT_FORECASTING.md`](INTELLIGENT_FORECASTING.md)（第一类工作，
> 智能预报）——两者共享 90% 工程基建。

---

## 0. TL;DR — 给决策者

- **你的判断成立**：autograd 的反向传播**就是**伴随算子 $M^*$ 的数值实现，与 4D-Var
  伴随公式逐项对应——**数学上严格等价，不是近似**。
- **推荐增量 4D-Var（incremental 4D-Var）起步**：与第一类工作（参数标定）共用 90% 流水线，
  可审稿性最好，最充分利用 autograd 红利。
- **最小可行动作**：先做 DA-1 的 toy 4D-Var + `torch.autograd.gradcheck` 梯度检验，
  用合成观测（已知真值扰动）验证整条伴随链路自洽，**2–3 周可交付**，且会暴露所有非光滑算子陷阱。
- **两个同化特有的新增工程**：(a) B 矩阵设计；(b) 观测算子 H 接入。
  surf 的 `RunResult` 让前四类观测几乎是恒等映射，起步成本极低。

---

## 1. 数学等价性论证（你的核心判断对不对）

### 1.1 4D-Var 代价函数

强约束 4D-Var 最小化的代价函数（增量形式，围绕背景场 $x^b$ 线性化）：

$$
J(\delta x_0) \;=\; \underbrace{\tfrac12\,\delta x_0^{\mathsf T}\,\mathbf{B}^{-1}\,\delta x_0}_{J_b}
\;+\; \underbrace{\tfrac12\sum_{k=0}^{K}\big(d_k - H_k\,M_{0\to k}\,\delta x_0\big)^{\mathsf T}\,\mathbf{R}_k^{-1}\,\big(d_k - H_k\,M_{0\to k}\,\delta x_0\big)}_{J_o}
$$

- $\delta x_0$：控制变量（初始场增量）。
- $M_{0\to k}$：模型算子，surf 多步积分 $x_0 \to x_k$。
- $H_k$：观测算子（模型状态 → 观测空间）。
- $d_k = y_k - H_k(x_k^b)$：新息向量（innovation），观测减背景。
- $\mathbf{B}$、$\mathbf{R}_k$：背景/观测误差协方差矩阵。

### 1.2 为什么 autograd = 伴随

最小化 $J$ 需要其对控制变量的梯度。对 $J_o$ 做变分，伴随链给出：

$$
\nabla_{\delta x_0} J_o \;=\; \sum_{k=0}^{K} M_{0\to k}^{*}\,H_k^{*}\,\mathbf{R}_k^{-1}\,\big(H_k\,M_{0\to k}\,\delta x_0 - d_k\big)
$$

其中 $M^*$ 是 $M$ 的**伴随**（转置雅可比，$\partial M/\partial x$ 的共轭）。

**关键事实**：PyTorch reverse-mode autograd 反向传播算的正是：

$$
\bar{x}_0 \;=\; \frac{\partial \,\text{loss}}{\partial x_0} \;=\; \sum_k M_{0\to k}^{*}\,H_k^{*}\,\frac{\partial \,\text{loss}}{\partial (H_k x_k)}
$$

这与 4D-Var 伴随公式**逐项对应**。所以在 surf 上做 4D-Var 的最小代价循环就是：

```python
with torch.enable_grad():
    delta_x = ...                                          # 控制变量, requires_grad=True
    x0 = x_bg + delta_x                                    # 增量加到背景场
    result = harness.run(forcing_list, initial_state=x0)   # = M_{0→K}
    innov  = H(result) - d                                 # = H·M·δx - d
    Jo = 0.5 * sum(innov_k^T R_k^{-1} innov_k)             # 观测项
    Jb = 0.5 * delta_x^T Binv(delta_x)                     # 背景项
    (Jo + Jb).backward()                                   # = autograd 给的伴随 M*
    optimizer.step()                                       # 增量极小化
```

**无需任何手写伴随代码**——这是相对 Fortran 4D-Var 的根本红利。

### 1.3 两个必须澄清的等价性边界

1. **autograd 给的是精确伴随（exact adjoint），不一定等于 TLM 转置**。对连续可微算子两者一致；
   对**非光滑算子**（`torch.where` 硬切换、`argmax`、求根号在零点）两者可能发散。
   surf 里大量用 `torch.where`（见 §3.2 审计）。

2. **autograd 默认 reverse-mode（=伴随）**。若同化算法需要 **forward TLM**（如外层
   Hessian-vector 乘积、或 EnKF 的切线性扰动传播），要用
   `torch.autograd.functional.jvp`（前向模式）。两者 PyTorch 都支持，但工程取舍不同：
   - reverse-mode（伴随）：一次反向 = 全梯度。**4D-Var 的标准做法。**
   - forward-mode（TLM）：一次正向 = 一个扰动方向的雅可比向量积。集合扰动传播、
     Hessian-vector 乘积用。

---

## 2. 四种同化方案对比（你要做的选择）

| 方案 | 用 autograd 做什么 | 优点 | 缺点 | 与 surf 契合度 |
|------|-------------------|------|------|----------------|
| **A. 增量 4D-Var**（推荐） | 反传算 $\nabla_{\delta x}$，外层拟牛顿（LBFGS）/共轭梯度 | 理论成熟、可审稿；与 ECMWF/IFS 同化哲学一致 | 需设计 $\mathbf{B}$；长窗口反传显存大 | ★★★★★ 最自然 |
| **B. 弱约束 4D-Var** | 控制变量含模式误差 $q_k$，反传到每步 | 处理模式误差（surf 经验模型误差确实大） | 控制变量维度爆炸 | ★★★ 高阶选项 |
| **C. EnKF / EnKS**（集合卡尔曼） | **不用**反传，集合扰动前向传播估计 $\mathbf{P}$ | 无需伴随、天然非高斯、易并行 | 集合发散、需 localization；**没用到 autograd 优势** | ★★ 浪费 torch |
| **D. 4D-EnKF / Hessian-free 混合** | autograd 提供 $\mathbf{H}\cdot v$，集合提供 $\mathbf{B}$ | 兼顾集合与变分 | 实现复杂 | ★★★ 进阶 |

> **强烈推荐方案 A（增量 4D-Var）作为起点**——最充分地利用 surf 已是 `nn.Module`
> + autograd 的架构红利（零伴随代码），与第一类工作共用 90% 工程代码，
> 且可审稿性最好（传统 4D-Var 是同化金标准）。

---

## 3. surf 当前可微性现状（实测核验）

### 3.1 已经具备、对同化有利的

| 能力 | 位置 | 对同化的意义 |
|------|------|--------------|
| **全局 `differentiable` 开关** | `config.py:134` | 设 True 后整条前向跑在 `enable_grad` 下；每个物理子模型都尊重这个开关。**= 4D-Var 的 $M$ 算子开箱可微。** |
| **`ColumnHarness.run()` 纯前向、autograd 无关** | `column_harness.py:8-11,177` | 调用方包一层 `enable_grad` 即可让多步积分对 loss 可微——**= $M_{0\to K}$ 算子**。 |
| **VEXCS 割线迭代是固定循环次数** | `exchange.py:345`（`for _ in range(kitt)`） | **无数据依赖的提前退出**，迭代被完整展开成固定计算图，可微性不受影响——避免了"while+break"导致的动态图与伴随不一致的经典陷阱。 |
| **大量用 `torch.where`/`clamp` 而非 Python `if`** | 全仓库 285 处 | 分支两端都被求值，梯度能正确回流到"激活分支"。这是 PyTorch 可微条件的标准写法。 |
| **LUT 索引已 `.detach()`** | `soil.py:309`、`veg.py:722` | 把 LUT 当分段常数函数处理，梯度只回流到 LUT 值——**正是陆面 4D-Var 对离散索引的正确处理**。 |
| **`RunResult` 含全状态量 + 通量** | `column_harness.py:45-76` | 前四类观测的 $H$ 算子几乎是恒等映射（见 §5），同化起步成本极低。 |

### 3.2 ⚠️ 必须正视的"非光滑"陷阱（4D-Var 严格性的关键）

这是传统 Fortran 4D-Var 也要处理的，但 autograd 让它**隐蔽**了，需主动管理：

| 算子类型 | surf 中的位置 | 4D-Var 中的问题 | 处理方法 |
|----------|--------------|----------------|----------|
| **`torch.where` 硬切换** | 冻结判据 `t_soil[...,0] <= consts.rtt`（`stomata.py:295`）、雪/雨判据、稳定/不稳定层结切换 | 伴随在切换点有 $\delta$ 奇异；TLM 取激活分支线性化 | 接受（陆面多在平滑区运行），或对关键判据用 sigmoid 软化 |
| **查表 `.gather`** | 土壤/植被类型索引 | 经典"开关型"非线性，TLM/伴随在离散索引点上无定义 | surf 已 `.detach()`——**正确做法**，LUT 当分段常数 |
| **`qsat` 饱和函数** | 蒸发、F3、SEB 迭代 | 解析可微（FOEEW 多项式），无问题 | 无需处理 |
| **`argmin`/`round`/不可导索引** | 物理核心里**没有**硬阻断算子（已扫描确认） | — | 已安全 |
| **Secant 迭代不动点** | VEXCS | 严格 4D-Var 要求对收敛解做隐函数定理伴随 | autograd 对固定循环次数是"展开 N 步"，`vexcs_kitt=3` 足够时为合法近似 |

> **结论**：surf 的可微性**足以支撑 operational-grade 的增量 4D-Var**，前提是接受
> LUT 索引作分段常数、`torch.where` 在判据点上的线性化。这两点传统 Fortran 4D-Var
> 同样要接受，不是 autograd 引入的新限制。

---

## 4. 控制变量与 B 矩阵设计（同化特有的工程难点 #1）

这是同化比参数训练**多出来**的核心工作。

### 4.1 控制变量选择

基于实测的 `SurfState` 字段（`state.py:247-298`），陆面同化的增量 $\delta x_0$ 推荐为：

| 状态量 | 字段 | shape | 进控制变量？ | 理由 |
|--------|------|-------|:---:|------|
| 土壤温度（4 层） | `soil_t` | (Ngrid, 4) | ✅ **是** | 长记忆、LST 观测强约束 |
| 土壤液态水（4 层） | `soil_w_liq` | (Ngrid, 4) | ✅ **是** | 长记忆、SMAP/SMOS 强约束 |
| 皮肤温度（每 tile） | `skin_t` | (Ngrid, Ntile) | ✅ **是** | 直接被 LST 观测 |
| 雪水当量 | `snow_w` | (Ngrid,) | ⚠️ 可选 | 短记忆、雪覆盖区才有 |
| 雪覆盖率 | `snow_frac` | (Ngrid,) | ⚠️ 可选 | 同上 |
| 冠层截留水 | `skin_water` | (Ngrid,) | ❌ 否 | 短记忆（小时级），观测少 |
| 土壤冰含量 | `soil_w_ice` | (Ngrid, 4) | ❌ 否 | 由温度 + 总水诊断，非独立 |
| 雪反照率/雪温/雪密度 | `snow_alb`/`snow_t`/`snow_rho` | (Ngrid,) | ❌ 否 | 短记忆、作"被动预报"跟着走 |

**推荐最小控制变量**（单格点维度 = 4 + 4 + Ntile ≈ 13）：
$$
\delta x_0 = \big[\delta T_{\text{soil}}^{(1..4)},\;\delta W_{\text{soil}}^{(1..4)},\;\delta T_{\text{skin}}^{(\text{tiles})}\big]
$$

雪场若纳入，作为"被动预报"（有观测时进控制变量，无观测时跟着物理走）。

### 4.2 B 矩阵设计

这是同化的灵魂。陆面 B 矩阵的常见做法，按精度递增：

1. **对角 + 经验方差**（起点够用）：每个状态量给一个经验标准差，如
   $\sigma_{T_{\text{soil}}} \approx 2\,\text{K}$，$\sigma_{W_{\text{soil}}} \approx 0.05$（体积含水量）。
2. **NMC 法**：跑 surf 集合预报，用 24h vs 48h 预报差估计 $\mathbf{B}$。
3. **集合法**：用集合离散度直接估计 $\mathbf{B}$。

> **关键工程红利**：因为 surf 是 torch，**集合预报就是 batch 维前向**（把 ensemble 当 batch），
> B 矩阵估计天然并行——这是 Fortran 做不到的。一个 (Ngrid, Nens) 的 batch 前向即得
> $\mathbf{P}^b = \text{cov}(\text{ensemble})$。

### 4.3 变量变换（balance 控制）

标准做法是引入控制变量变换 $\delta x = \mathbf{L}\,\delta w$（$\mathbf{B}=\mathbf{L}\mathbf{L}^{\mathsf T}$），
在 $\delta w$ 空间用单位阵做先验。surf 已是 torch，$\mathbf{L}$ 可以是：
- 解析变换（如土壤温-水协方差的 Cholesky），
- 或 `nn.Linear`（若走数据驱动的平衡关系学习）。

**陆面特有的平衡关系**：表层土壤温 ↔ 皮肤温强相关；相邻土壤层温/水强相关。
这些应在 $\mathbf{L}$ 中显式编码。

---

## 5. 观测算子 H 与观测接入（同化特有的工程难点 #2）

参数训练用"轨迹 vs 观测 RMSE"即可，同化需要**显式观测算子 $H$**：

| 观测类型 | $H$ 算子 | surf 是否现成 | 备注 |
|----------|---------|:---:|------|
| 站点 LST / 皮肤温 | 格点插值 + 直接取 `skin_t` | ✅ | 几乎恒等映射 |
| SMAP/SMOS 土壤湿 | 取表层 `soil_w_liq[...,0]` + 体积含水量换算 | ⚠️ 需写换算 | 浅层加权（卫星探测深度 ~5cm） |
| 通量塔 LE/H | 取 surf 的 `e_flux`/`h_flux` 诊断 | ✅ | `RunResult` 已直接输出 |
| GPP | Farquhar 输出（需开 `use_farquhar`） | ✅ | 仅当 A-gs 光合路径开启 |
| 亮温（卫星） | 辐射传输模型（CMEM 等） | ❌ 需外接 | 留作后期，需外接 RTM |

> **起步成本极低**：`RunResult` 已含 `h_flux/e_flux/net_rad/ground_heat` + 全状态量
> （`column_harness.py:45-76`），所以前四类观测的 $H$ 几乎是恒等映射。
> 卫星亮温这类需外接辐射传输的，留作后期。

### 5.1 观测算子的可微性

$H$ 必须可微，否则 autograd 的伴随链断在 $H^*$ 处。
- 恒等/插值类 $H$：天然可微。
- 浅层加权换算：解析可微。
- 外接 RTM（CMEM）：需 RTM 自身可微，或用有限差分近似其雅可比——**这是后期的主要工程量**。

---

## 6. 分阶段实施路线

| 阶段 | 做什么 | 与第一类工作的关系 | 周期 |
|------|--------|---------------------|------|
| **DA-1** | 单点 toy 4D-Var：合成观测（背景轨迹+已知扰动）回灌，验证 $\nabla J$ 正确（`torch.autograd.gradcheck`） | 复用 `ColumnHarness` + optimizer，控制变量换成 $\delta x_0$ | 2–3 周 |
| **DA-2** | 真实观测 $H$ 接入 + B 矩阵（对角 NMC 起步） | 复用参数训练的观测数据加载器 | 3–4 周 |
| **DA-3** | 多窗口 cycling 同化循环（分析→预报→下次分析） | 独立工程，但验证脚本复用 offline-coupled 基线 | 3–5 周 |
| **DA-4** | 弱约束 / 4D-EnKF 混合（模式误差在线估计） | 与第一类 Phase 3 的 NN 替换耦合——NN 误差结构就是 $\mathbf{Q}$ 的来源 | 6–8 周 |

### 6.1 DA-1 详述（推荐的最小可行动作）

目标：**证明整条 4D-Var 链路在 surf 上数学自洽**，纯数值验证，不需要真实观测。

1. **构造合成观测（OSSE，Observing System Simulation Experiment）**：
   - 取背景轨迹 $x^b$（任意合理初值跑 `ColumnHarness`）。
   - 加已知增量 $\delta x^{\text{true}}$ 得"真值"轨迹 $x^{\text{true}}$。
   - 在 $x^{\text{true}}$ 上采样 + 加观测噪声 → 合成观测 $y$。
2. **梯度检验**：
   ```python
   # torch.autograd.gradcheck: autograd 梯度 vs 有限差分
   gradcheck(lambda dx: J(x_bg + dx, y), (delta_x.clone().requires_grad_(True),),
             eps=1e-6, atol=1e-4)
   ```
   通过则证明 autograd 给的 $\nabla J$ 是真梯度的精确伴随，伴随链无 bug。
3. **增量极小化**：LBFGS 最小化 $J$，看 $\delta x^{\text{recovered}}$ 是否收敛到 $\delta x^{\text{true}}$
   （在 $\mathbf{B}/\mathbf{R}$ 平衡范围内）。
4. **非光滑算子体检**：DA-1 会暴露 `torch.where` 切换点导致的梯度异常（gradcheck 失败处），
   逐个评估：接受 / sigmoid 软化 / 跳过该窗口。

**DA-1 验收门槛**：
- [ ] `gradcheck` 在典型窗口通过（伴随链自洽）。
- [ ] OSSE 的 $\delta x^{\text{recovered}}$ 收敛到 $\delta x^{\text{true}}$（同化可逆）。
- [ ] $\delta x_0 = 0$ 时与 Fortran ref bit-identical（sentinel 回归通过）。

### 6.2 DA-1 可能暴露的典型问题

| 问题 | 现象 | 对策 |
|------|------|------|
| 冻结点 `torch.where` 切换 | gradcheck 在过冷水窗口失败 | sigmoid 软化判据，或窗口避开相变天气 |
| 雪覆盖 on/off 切换 | `snow_frac` 跨 0/1 时梯度跳变 | 雪场不进控制变量（被动预报） |
| VEXCS 收敛性 | 极端稳定层结下 secant 不收敛 | 检查 `vexcs_kitt` 是否足够；必要时增到 4–5 |
| 长窗口显存 | 反传几十步 surf forward 爆显存 | `torch.utils.checkpoint` 对每步重计算（见 §7） |

---

## 7. 风险与防护（同化特有）

| 风险 | 防护 |
|------|------|
| **长窗口反传显存爆炸** | **梯度检查点** `torch.utils.checkpoint`：对每步 surf forward 重计算而非缓存；或截断窗口（TBPTT，只对最近 N 步反传）。一个 assimilation window 几十步，每步是完整 surf forward。 |
| **B 矩阵病态 / 初始增量发散** | 变量变换 $\delta x = \mathbf{L}\delta w$ + 增量范数正则；LBFGS 而非纯梯度下降；增量法围绕背景场线性化。 |
| **autograd 伴随与 TLM 在 `torch.where` 切换点不一致** | 对关键判据（冻结点、雪雨分界）做 sigmoid 软化；或在同化窗口避开相变天气。 |
| **观测算子 $H$ 非线性强导致 $J$ 非凸** | 增量法（围绕背景场线性化）+ 外层多次极小化（ECMWF 做法）；初值用背景场。 |
| **与 Fortran IFS 4D-Var 的一致性论证（审稿）** | 关键：证明"关掉 ML/同化就是原 surf"——sentinel-gated 设计保证 bit-identical 回退。 |
| **模式误差被强约束假设忽略** | DA-4 升级到弱约束 4D-Var；或用第一类工作摸清的误差结构估 $\mathbf{Q}$。 |

---

## 8. 与第一类工作（智能预报）的协同

两者共享 90% 工程基建，建议**并行设计、串行启动**：

- **共用流水线**：4D-Var 本质是"以初始场增量为可学习参数，对观测轨迹做梯度最小化"，
  与参数训练同构——`ColumnHarness` + optimizer + loss 基建完全复用。
- **误差知识反哺**：第一类工作 Phase 1–2（见
  [`INTELLIGENT_FORECASTING.md`](INTELLIGENT_FORECASTING.md)）摸清"哪些经验模型误差大"，
  直接喂给同化的 **$\mathbf{Q}$ 矩阵**（模式误差协方差）——这是强约束 4D-Var 最大的不确定来源。
- **闭环**：参数训练 → 同化产出分析场 → 分析场作为更优 target 反哺参数训练。

**推荐启动顺序**：先第一类 Phase 1（参数标定）拿到误差结构，再启动同化 DA-1。
若想更快验证同化可行性，可并行起 DA-1（它只需合成观测，不依赖真实数据加载器）。

---

## 9. 核心结论

1. **判断成立**：surf 作为 torch `nn.Module`，autograd **严格提供了 4D-Var 所需的伴随算子 $M^*$**，
   免去传统陆面同化最痛苦的"手写伴随"环节。这是相对 Fortran 实现的**根本性工程红利**。

2. **数学等价性成立但有边界**：对连续可微部分严格等价；对 `torch.where`/LUT 索引等非光滑部分，
   autograd 给精确伴随，与 TLM 转置在切换点可能分歧——这是传统 4D-Var 同样要处理的，
   不是新增限制。

3. **推荐增量 4D-Var（方案 A）起步**：与参数训练框架共用 90% 基建，可审稿性最好，
   最充分用 autograd 红利。

4. **两个同化特有的新增工程**：
   - (a) **B 矩阵设计**（集合法 + torch batch 天然并行）；
   - (b) **观测算子 H 接入**（surf 的 `RunResult` 让前四类观测几乎是恒等映射，起步成本极低）。

5. **最小可行动作**：DA-1 的 toy 4D-Var + `gradcheck` 梯度检验，2–3 周可交付，
   会暴露所有非光滑算子陷阱，为真实观测接入铺路。

---

## 附：关键文件/位置速查

| 用途 | 位置 |
|------|------|
| 可微开关 | `surf_pytorch/config.py:134`（`differentiable`）、`vexcs_kitt:149` |
| 轨迹生成器（= $M_{0\to K}$） | `surf_pytorch/column_harness.py:177`（`run()`） |
| RunResult（含状态量+通量，= $H$ 输入） | `surf_pytorch/column_harness.py:45-76` |
| SurfState（控制变量来源） | `surf_pytorch/state.py:247-298`（`soil_t:269`/`soil_w_liq:272`/`skin_t:266`） |
| 非光滑判据（冻结点） | `surf_pytorch/physics/stomata.py:295` |
| LUT 索引 detach（=分段常数处理） | `surf_pytorch/tables/soil.py:309`、`veg.py:722` |
| VEXCS 固定迭代（无动态图陷阱） | `surf_pytorch/physics/exchange.py:345` |
| 验证基线 | `coupling/`（offline-coupled）、`tests/test_ecland_driven.py` |
| 梯度检验 API | `torch.autograd.gradcheck`（reverse-mode 伴随 vs 有限差分） |
| 前向模式 TLM（Hessian-vector） | `torch.autograd.functional.jvp`、`vjp`、`hvp` |
| 梯度检查点（显存防护） | `torch.utils.checkpoint` |

### 配套文档
- [`INTELLIGENT_FORECASTING.md`](INTELLIGENT_FORECASTING.md) — 第一类工作（智能预报）：
  参数标定 → 经验矫正 → NN 替代 → 混合模型。含完整参数清单与经验方案清单。
