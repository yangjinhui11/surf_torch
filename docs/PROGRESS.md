# PyTorch 陆面方案 — 进度日志

## P1.6-4b — 逐瓦片交换系数（关闭：2026-06-27）

### 目标
弥合 PyTorch 移植与 Fortran 离线驱动器第 0 步参考之间，单步 `forward()` 肤温的
物理差距：

| 变体                  | Δskin (K) | 与 Fortran 的差距 |
|----------------------|-----------|----------------|
| Fortran `o_gg[1]-o_gg[0]` | **-6.038** | 参考           |
| 网格均值（传统）       | -3.6（约）  | ~1.9 K（差）   |
| 逐瓦片（本工作）       | -6.007 | **+0.031 K**    |

### 改动内容
- **`surf_pytorch/physics/exchange.py`** —— `psimu` 非选中支 NaN 修复。
  `torch.where(cond, A, B)` 会对*两条*分支反向传播；不稳定分支
  `inner = sqrt(1 - 8*eta)` 未加截断，导致水/冰瓦片（小 z0 → pzdl > 0.125）在
  前向产生 NaN，即使 psihs 是被选中支也会以 `0 * nan = nan` 传播。改为使用已截断的
  `sqrt_z`。
- **`surf_pytorch/model.py`**
  - `_per_tile_z0` —— 用 `torch.full(..., 0.0)` 替换 `torch.empty`（后者让瓦片 8
    湖 / 9 城市留下垃圾值 → n_tile=10 时前向 NaN），随后在逐瓦片覆写之前填入裸土
    默认值。
  - `_per_tile_monin_obukhov_coeffs` + `forward/_exchange_coeffs` —— 将
    (Ngrid, ntile) 展平为 (Ngrid*ntile,) 做一次向量化 VEXCS 调用，对应 Fortran
    `surfexcdrivers_ctl_mod.F90:388-398`（每瓦片一次 VEXCS）。
  - `SurfConfig.per_tile_exchange` 标志（默认开）。
- **测试** —— `test_diffusion.py`（RVDIFTS 1.5→1.0，ztpfac 2/3,1/3→1.0,0.0）、
  `test_model.py`（水瓦片精确钉在 SST）、`test_forward_m5.py`（强制不稳定柱以获得
  符号确定的冷却）、`test_forward_e2e.py`（`_reference_exchange` 在展平轴上经 Fortran
  `.so` 路由 VEXCS）、新增 `test_step0_physics.py`（绝对 Δskin ≈ -6.04 K）。

### 验证（最终）
- `scripts/diag_forward_lw.py` —— Fortran 取值修正后的残差差距：**+0.0309 K**
  （原为 Fortran -6.04K 对比旧网格均值 -3.6K）。
- 全套测试：**510 通过，8 跳过**（OIFS_OGG_PATH 未设置；门控的 Fortran 耦合测试）。

### 已知残差（+0.031 K）
逐瓦片移植将 Fortran 第 0 步 Δskin 复现到 **+0.031 K**。该残差在 P1.6-4c 中调查，
**并非移植 bug**：

1. **辐射装配精确**（`scripts/diag_step0_residual.py`）：抵达核的 DiffusionInputs
   携带 `pslrfl = -117.631200` W/m^2（与 Fortran `callpar1s:812-813` 取值 bit 相等）、
   `ptskrad = 288.0`（肤温基）、`pemis = 0.99`。
2. **核与 Fortran bit 相等**（`test_forward_e2e.py`）：在相同 DiffusionInputs 下，
   `new_state.skin_t` 与 `ref.vdfdifh(...).ptsktip1` 一致到 <2000 ULP（~4e-13 K）。
3. **PEMIS 本就正确**：离线驱动器硬编码 `REMISS = 0.99`（`surdi1s.F90:12`，经
   `callpar1s.F90:806` 的 `ZEMIS=REMISS` 贯通）。较早的 "0.9667" 数值是*完整 IFS* 的
   地表比辐射率，而非移植代码所对照验证的离线驱动器取值——因此无需修正。

这 +0.031 K 是**单步 `forward()`**（仅 vdfdifh1s + 肤温外推）与**完整离线驱动器第 0
步**（`o_gg[1]`，额外运行 `cpg1s` 辐射、全模式最低层 Dirichlet 耦合、以及驱动器自身
序列中的土壤/雪更新）之间的差距。这些超出可微单步模块的范围。残差稳定且小（<0.1 K），
因此记录在案而非继续追查。
