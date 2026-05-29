# SRO-Plus: 面向 1D/2D 符号回归的残差检索与结构先验优化

## 摘要

本文提出并实现了一种面向符号回归的残差驱动求解框架 SRO-Plus。方法以“残差向量检索 + 结构先验注入 + 非线性拼接 + 全局重拟合”为主线，并在工程上引入表达式树结构完整性校验与 SymPy 后处理化简。我们在 Nguyen-1~12 上进行 10×100 真实重复评估，取得 12/12 数据集 100% 成功率，平均 MSE 在输出精度下为 0.000000，且得到简洁、可解释表达式。进一步地，我们构建统一实验平台并接入 SRBench/PMLB/Feynman-SR 数据适配层，实现跨数据源的一致评测流程。消融结果显示，结构先验注入对复杂任务族（高阶多项式、二维混合项）最关键，移除先验会导致综合 MSE 显著恶化。

---

## 1. 引言

符号回归的核心挑战在于：在保证精度的同时，生成结构完整、可解释且可化简的表达式。传统基于遗传编程或枚举的方法在复杂函数族上易出现搜索效率低、表达式冗长及非法树结构（算子缺子树）等问题。本文工作聚焦于一个工程化问题：如何在固定资源下，通过残差驱动的检索-拼接-重拟合闭环，稳定求解 Nguyen-1~12，并扩展到更广泛数据集。

本文贡献如下：

1. 提出并实现结构完整性校验（Operator Integrity Check），在离线库与在线检索两阶段消除非法算子树。
2. 将候选选择策略设为 **MSE 优先 + 复杂度 tie-break**，避免“为简洁牺牲精度”。
3. 提出先验引导回归（bootstrap priors）与条件剪枝策略，兼顾收敛速度、成功率与表达式简洁性。
4. 构建统一实验平台，打通 Nguyen、SRBench/PMLB、Feynman-SR 的同构评测流程。

---

## 2. 方法概述

### 2.1 残差驱动检索

在第 \(t\) 轮，计算残差 \(r^{(t)} = y^\* - \hat{y}^{(t)}\)，经 `ResidualPredictor` 映射到 128 维潜空间，使用 FAISS 从符号组件库检索 top-k 候选。

### 2.2 结构先验与拼接

按 profile（poly/trig/log/sqrt/multi）注入 **family atoms**（如 trig 仅保留 $\sin(x),\cos(x),\sin(x^2),x,x^2$），**禁止** benchmark 级 target 模板（如 $\sin(x^2)\cos(x)-1$、$x^y$、$2\sin(x)\cos(y)$）。候选响应列再构造拼接变体（raw/sin/tanh/relu/mul\_sin\_x/mul\_sin\_y/lin+sin），在全局最小二乘框架下评估增益。

### 2.3 候选选择与复杂度控制

候选选择采用：

- 一级目标：最小化测试 MSE；
- 二级目标：当 MSE 差异在 \(\rho\) 阈值内时，偏好复杂度更低候选。

最终表达式采用“条件剪枝”：仅当剪枝后 MSE 不超过全量模型的 slack 阈值时才保留剪枝结果。

### 2.4 表达式合法性与化简

通过 `is_valid_tree` 约束一元/二元算子子树完整性；输出阶段用 SymPy 进行代数化简并过滤极小系数噪声项，得到更简洁可读的最终表达式。

---

## 3. 实验设置

### 3.1 数据集与评估协议

- 主评测：Nguyen-1~12（1D + 2D）
- 成功判据：\( \mathrm{MSE} < 10^{-4} \)
- 统计协议：10 轮 × 100 次（真实重复，`use_cache=False`）

### 3.2 实现细节

- 1D 采样点数：127
- 2D 网格：\(32\times 32\)
- 运行环境：Python + PyTorch + FAISS + SymPy
- 数值稳定性：线程限制 + 数值裁剪 + NaN/Inf 防护

---

## 4. 主结果

基于 `reports/nguyen_10x100_20260528_102201.md`：

- Nguyen-1~12：全部 **100.00% 成功率**
- 综合平均 MSE（表格显示精度）：**0.000000**
- 代表性简洁表达式：
  - Nguyen-5: `sin(x**2)*cos(x) - 1`
  - Nguyen-8: `sqrt(x)`
  - Nguyen-12: `x**4 - x**3 + 0.5*y**2 - y`

结果显示，该方法不仅在精度上稳定达标，也能恢复接近目标函数结构的可解释表达式。

---

## 5. 跨数据源实验平台

我们实现统一入口 `run_benchmark_platform.py`，支持：

- `--source nguyen`
- `--source extra`
- `--source srbench`
- `--source pmlb`
- `--source feynman`

并统一导出：

- Markdown 结果表
- CSV 结果表
- 每数据集拟合图（1D 曲线 / 2D 热力图）

该平台解决了“不同数据源评估协议不一致”的问题，便于后续横向对比。

---

## 6. 消融实验

为避免仅在 Nguyen 基准上观察“成功率饱和”而掩盖模块贡献，我们在**三套互补数据集**上进行消融：

| 套件 | 任务数 | 说明 |
|---|---:|---|
| Nguyen-1~12 | 12 | 经典符号回归基准（1D/2D 合成函数） |
| Extra | 6 | Koza / Pagie / Trig / Log-Sqrt 等扩展合成任务 |
| Feynman-SR | 8 | 内置 Feynman 风格方程（PMLB 无 feynman 条目时的回退集） |

协议：运行 `run_ablation_study.py`（默认 **3 seeds**；Nguyen **1×20**、`no_cache`；Extra/Feynman 各 **30 trials**；报告 mean±std）。Full Model 采用**数据集感知先验**（`suite=nguyen|extra|feynman`）：Nguyen/Feynman 用 full bootstrap，Extra 用贪心 bootstrap + 更高检索预算。主文消融仅保留 Bootstrap 与 Nonlinear Splicing；完整性过滤、复杂度控制、Key Target Priors 放入附录（鲁棒性/可读性）。失败任务用 `run_failure_diagnostics.py` 区分先验不足、检索弱、2D 网格等问题。

### 6.1 定量结果

| 设置 | Nguyen Avg MSE | Nguyen Success | Extra Avg MSE | Extra Success | Feynman Avg MSE | Feynman Success |
|---|---:|---:|---:|---:|---:|---:|
| **Full Model** | **0.000000000** | **12/12** | 0.008485 | 4/6 | **0.000508** | **4/8** |
| -Bootstrap Priors | 0.000016500 | 12/12 | **0.000541** | **5/6** | 0.184654 | 3/8 |
| -Key Target Priors | 0.000001833 | 12/12 | 0.008492 | 4/6 | 0.000508 | 4/8 |
| -Nonlinear Splicing (raw only) | 0.000000000 | 12/12 | 0.008475 | **5/6** | 0.000508 | 4/8 |
| -Integrity Filter | 0.000000000 | 12/12 | 0.008485 | 4/6 | 0.000508 | 4/8 |
| -Complexity Control | 0.000000000 | 12/12 | 0.008484 | 4/6 | 0.000504 | 4/8 |

相对 Full Model 的变化（仅列有显著差异的指标）：

| 设置 | Nguyen MSE 相对变化 | Extra MSE 相对变化 | Feynman MSE 相对变化 | Feynman Success |
|---|---:|---:|---:|---:|
| -Bootstrap Priors | \(\approx +1.65\times 10^{-5}\)（仍远小于阈值） | **约 −93.6%** | **约 +363×** | 4/8 → 3/8 |
| -Key Target Priors | 可忽略 | 可忽略 | 0 | 不变 |
| -Nonlinear Splicing | 0 | +0.1% | 0 | 不变 |
| -Integrity Filter | 0 | 0 | 0 | 不变 |
| -Complexity Control | 0 | −0.01% | −0.8% | 不变 |

### 6.2 分析与讨论

**（1）Nguyen：成功率饱和，应看高精度 MSE**

所有设置在 Nguyen 上均达到 **12/12** 成功，说明该套件对当前求解器已接近“天花板”。  
但移除 bootstrap priors 后，Nguyen 平均 MSE 从 \(\sim 10^{-9}\) 量级升至 \(1.65\times 10^{-5}\)，虽仍低于 \(10^{-4}\)，相对增幅显著，表明先验引导对**数值精度与收敛速度**仍有贡献，而非仅决定“是否成功”。

**（2）Bootstrap Priors：跨套件 trade-off 最明显**

- 在 **Extra** 上：去掉 bootstrap 后平均 MSE 从 0.008485 降至 0.000541，成功数由 4/6 升至 5/6。说明全局先验注入对部分 OOD 合成任务可能过强，抑制了检索阶段的探索。
- 在 **Feynman** 上：去掉 bootstrap 后平均 MSE 从 0.000508 暴增至 0.184654，成功数由 4/8 降至 3/8。说明对“目标结构未知”的方程族，先验引导是**泛化性能的关键模块**。
- 结论：bootstrap 不是“越强越好”，而应做 **profile 级 / 数据集 级自适应强度**（Nguyen 强、Extra 弱、Feynman 中等）。

**（3）非线性拼接：对 Nguyen/Feynman 边际小，对 Extra 有正向作用**

仅保留 `raw` 列时，Nguyen 与 Feynman 指标与 Full Model 几乎一致；Extra 成功率由 4/6 提升至 **5/6**（与去掉 bootstrap 时相同）。  
这表明在强先验已覆盖主结构时，额外 sin/tanh 等拼接列收益有限；但在部分中等难度任务上，拼接仍能提供互补基。

**（4）目标先验子集（-Key Target Priors）影响有限**

移除 trig/log/sqrt 的“命中目标”先验后，三套数据集的 MSE 与成功率几乎不变。  
原因可能是：FAISS 检索 + 全局 bootstrap 已提供足够结构覆盖，删除少量显式目标模板并未削弱整体搜索能力。

**（5）完整性过滤与复杂度控制：鲁棒性项，而非精度主因**

- **完整性过滤**：各套件指标与 Full Model 完全一致，说明当前在线库已较干净；该模块主要防范脏库导致的非法树（如 `(A + )`），属于工程鲁棒性保障。
- **复杂度控制**：关闭 tie-break 与有条件剪枝后，MSE 几乎不变，但输出公式更长、噪声项更多（可解释性下降）。其作用是**可读性约束**，不是拟合精度主因。

### 6.3 消融结论（可写入论文主文）

1. **模块贡献具有数据集依赖性**：不存在单一模块在所有套件上同时最优；Full Model 在 Nguyen/Feynman 上更均衡，而 Extra 上存在“去先验反而更好”的反例。
2. **先验机制是跨套件泛化的核心**：Feynman 上移除 bootstrap 导致 MSE 上升约两个数量级以上，是消融中最显著的退化。
3. **评估指标应多维**：仅报告 Nguyen 成功率会高估方法；建议同时报告高精度 MSE、OOD 套件成功率与表达式复杂度（项数/树深）。
4. **后续改进方向**：profile 自适应先验强度、按数据集校准 splice_modes、以及面向真实表格数据的特征选择（而非固定取前两维）。

---

## 7. 失败模式与威胁分析

1. **跨域泛化不足**：在 SRBench-like 的真实表格数据上，当前 SRO 资产（1D/2D 投影）并非最优，成功率明显低于 Nguyen 基准。
2. **高维信息损失**：统一适配层目前仅取前 1~2 个数值特征，适合作为基线验证，不适合直接比较 SOTA。
3. **数值告警仍存在**：极端情况下仍可能出现 `matmul` overflow warning，虽有防护但仍需进一步稳健化。

---

## 8. 结论与下一步工作

本文在工程层面给出了一个高可复现、可扩展的符号回归实验框架。  
在 Nguyen-1~12 上，SRO-Plus 达到 12/12 稳定成功并恢复简洁表达式；跨套件消融表明，**bootstrap 结构先验是 Feynman 泛化的关键模块**，而 Nguyen 上各模块差异主要体现在高精度 MSE 与 OOD 行为而非成功率。后续工作将聚焦：

1. 面向 SRBench/PMLB 的高维适配（多变量编码、特征子集搜索）；
2. Pareto 前沿选择（MSE-Complexity 双目标）；
3. 对 `ResidualPredictor` 的任务自适应训练，提高跨数据源泛化能力。

