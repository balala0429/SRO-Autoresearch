# Methods: Nguyen SRO（Sections 3.1–3.6）

## 3.1 Problem Setup

给定采样点上的观测值 $\mathbf{y}^\ast\in\mathbb{R}^{N}$，符号回归目标是寻找一个可解释的表达式 $F(\cdot)$，使其在采样点上的预测值与 $\mathbf{y}^\ast$ 之间的均方误差最小：

$$
\min_{F}\ \mathrm{MSE}(F)=\frac{1}{N}\left\|\mathbf{y}^\ast - F(\mathbf{x})\right\|_2^2.
$$

本项目使用“符号表达式树（symbolic expression tree）”进行表示。表达式由常数、变量叶子与算子节点构成，算子集合包含二元算子 $\{+,-,\times,/\}$ 与一元算子 $\{\sin,\cos,\exp,\log,\sqrt\}$。其中多变量任务（Nguyen-9~12）引入变量叶子 `y`，并对表达式进行多变量数值求值。

求解过程以残差形式迭代进行：在第 $t$ 轮维护当前预测 $\mathbf{y}^{(t)}$，并令残差 $\mathbf{r}^{(t)}=\mathbf{y}^\ast-\mathbf{y}^{(t)}$。系统通过“残差到组件语义”的方式检索候选符号组件，并将候选组件通过（可选的）非线性拼接后加入全局线性重拟合。

## 3.2 Library Construction

系统包含一个离线预构建的符号组件库（library），由两部分组成：

1. **符号树集合**：大量随机/安全生成的表达式子树（components），以树结构形式存储在 `library/library/symbolic_trees.pkl`。
2. **向量索引（FAISS）**：对每个组件的向量表征建立检索索引，存储在 `library/library/symbolic_index.bin`。

在在线阶段，ResidualSolver 加载：

- `library/library/symbolic_trees.pkl`（组件树）
- `library/library/symbolic_index.bin`（向量索引）

并在每轮通过 predictor 输出的潜向量查询 FAISS，得到 top-$k$ 候选组件。

### 3.2.1 Operator Integrity Check (结构完整性)

实际运行中发现，若库中存在结构不完整的表达式树（例如二元算子缺少左右子树），数值求值可能“默认为 0”而仍可参与拟合，但字符串化与符号化简会出现非法形式（例如 `(cos(sqrt(x)) + )`）。为保证模型可读性与拼接正确性，本文在**库构建**与**在线加载**阶段加入结构完整性校验（operator integrity check）：

- **叶子节点**：`x`, `y`, `const` 必须无子树；`const` 需有有限实数值。
- **一元算子**：$\{\sin,\cos,\exp,\log,\sqrt\}$ 必须且仅有一个子树。
- **二元算子**：$\{+,-,\times,/,\mathrm{pow}\}$ 必须同时具有左右子树。

在线阶段加载 `trees.pkl` 后，会过滤掉不合法树，并重建与过滤后树集合一一对应的 FAISS 索引（确保 `index[i]` 与 `trees[i]` 始终对齐），从而避免检索返回的候选在拼接与打印阶段产生结构缺失。

## 3.3 Residual Retrieval

ResidualSolver 由三步构成：探测（predictor）、检索（FAISS）与候选评估（beam + 全局重拟合）。

### 3.3.1 Residual Embedding (Predictor)

给定当前残差 $\mathbf{r}^{(t)}$，先进行归一化得到输入向量 $\hat{\mathbf{r}}^{(t)}$，再由神经网络 ResidualPredictor 映射到潜空间：

$$
\mathbf{v}^{(t)} = f_\theta(\hat{\mathbf{r}}^{(t)})\in\mathbb{R}^{128}.
$$

随后对 $\mathbf{v}^{(t)}$ 做 L2 归一化，并作为 FAISS 的 query。

### 3.3.2 FAISS Retrieval

令 $\mathcal{C}=\{T_j\}$ 为组件库中候选表达式树集合，系统从 FAISS 检索得到 top-$k$：

$$
\{T_{j_1},\dots,T_{j_k}\} \leftarrow \mathrm{FAISS}(\mathbf{v}^{(t)},k).
$$

同时引入 profile-specific 的结构先验（structural priors injection），将若干与目标函数族高度相关的子树直接加入候选队列，以补足向量检索对特定结构模式的漏检风险。

### 3.3.3 Candidate Evaluation with Beam and Global Refit

对候选组件 $T$，先在采样点上数值评估其响应列 $\phi(T)$，并构造非线性拼接后的候选列集合（见 3.4）。系统对“加入候选后的整体列基”进行最小二乘重拟合，从而计算该候选的测试 MSE：

$$
\mathrm{MSE}(T) = \frac{1}{N}\left\|\mathbf{y}^\ast - \mathbf{A}_{\mathrm{active}\cup\{\phi(T)\}} \mathbf{w}\right\|_2^2,
$$

其中 $\mathbf{A}$ 为当前 active 列集合与候选列拼接后的设计矩阵。由于候选组件复杂度可能导致过拟合，系统同时引入复杂度惩罚（以树节点数刻画）：

$$
\mathrm{score}(T)=\mathrm{MSE}(T)\cdot\left(1+\lambda\cdot\mathrm{size}(T)\right).
$$

在 beam 层面，对 top-$k$ 候选做去重与截断，仅保留前若干个多样性候选用于评估；随后进行**MSE 优先 + 复杂度打破平局（tie-break）**的选择策略：

- 首先选择使 $\mathrm{MSE}(T)$ 最小的候选；
- 若若干候选的 MSE 差距在阈值 $\rho$ 内（例如相对差距 $\le \rho$），则在这些“近似同精度”的候选中选择复杂度更小者（树节点更少、拼接列数更少）。

该策略的目的，是避免“为了简洁而牺牲达到成功阈值的精度”，同时仍能在精度接近时偏好更短、更可解释的表达式。

## 3.4 Nonlinear Splicing

传统符号回归中的组件组合通常是线性的：将候选组件输出列直接作为 OMP/最小二乘的字典列。为捕获目标函数中的强非线性结构，本系统在“列基构造”阶段引入非线性拼接（nonlinear splicing）。

对候选组件在采样点的响应列 $\phi(T)$，系统构造多种变体列，并在 OMP/最小二乘重拟合时让线性系数自动选择最有用的非线性基：

- **raw**：$\phi(T)$
- **sin**：$\sin(\phi(T))$
- **tanh**：$\tanh(\phi(T))$
- **relu**：$\max(\phi(T),0)$
- **mul\_sin\_x**：$\phi(T)\cdot \sin(x)$
- **mul\_sin\_y**（2D）：$\phi(T)\cdot \sin(y)$
- **exp\_mix**：$\exp(\phi(T))$（数值裁剪）
- **lin+sin**：对联合列集合 $\left[\phi(T),\sin(\phi(T))\right]$ 进行联合最小二乘评估

其中多变量任务同时支持带 `y` 参与的变量叶子表达；非线性拼接主要作用于组件响应列本身。

为抑制不必要的非线性爆炸，本系统通过不同 profile 控制允许的拼接算子集合（splice_modes）。

## 3.5 Multi-variable Extension (x,y)

为支持 Nguyen-9~12 的二维目标函数，系统对表达式树求值与采样点构造进行了多变量扩展：

1. **统一求值接口**：`eval_tree` 接收变量字典 `var_data={'x':..., 'y':...}`，递归计算树上 `x`、`y` 叶子与一元/二元算子的组合响应。
2. **采样点构造**：对于 dim=2 的任务，在 `x_range` 与 `y_range` 上分别取离散点集 $\{x_i\}_{i=1}^{n_x}$ 与 $\{y_j\}_{j=1}^{n_y}$，并构造二维网格 $\{(x_i,y_j)\}$。实际实现采用 `meshgrid` 形成 $n_x\times n_y$ 个点，再按行/列展平为长度 $N=n_x n_y$ 的向量，从而将二维函数评估为统一的一维响应列 $\mathbf{y}\in\mathbb{R}^{N}$。
3. **变量集合支持**：树表达式的叶子节点扩展为 `x` 与 `y`，并在 structural priors 与拼接策略中包含与二维结构相关的候选子树（例如包含 `sin(y)`、`cos(y)` 与 `sin(x)*cos(y)`）。

在该扩展下，系统可以直接对 Nguyen-9~12 进行与 1D 基准一致的求解与评估流程。

## 3.6 Evaluation Protocol

### 3.6.1 Datasets and Success Criterion

本方法在 Nguyen 系列基准上进行评估。对 1D 任务（Nguyen-1~8、部分约束任务），使用一维采样区间；对 2D 任务（Nguyen-9~12），使用二维采样区间 `x_range` 与 `y_range`。

成功判定采用：

$$
\mathrm{Success} \iff \mathrm{MSE} < 10^{-4}.
$$

### 3.6.2 Repeated Trials: 10×100

为了统计鲁棒性与成功率，采用“10 轮 × 每轮 100 次”的统计协议。实现上提供两种模式：

- **缓存模式（use_cache=True）**：对同一函数族求解一次，将得到的 MSE 复制到 total_trials，用于快速统计（适用于求解器对固定输入在实践中近似确定的情况）。
- **真实重复模式（use_cache=False）**：对同一函数族进行 total_trials 次独立求解，并计算成功次数与平均 MSE。

输出指标包括：

- **Success Rate**：Success 次数 / total_trials
- **Avg MSE**：平均 MSE
- **Best Expression**：对使用缓存时取该求解对应的最终化简表达式（SymPy 化简后字符串形式）

### 3.6.3 Profile Configuration (关键超参数)

系统对不同任务族启用不同的 profile 超参数（`PROFILE_CONFIG`），如下：

| profile | top_k | beam_limit | penalty_weight $\lambda$ | mse_tie_ratio $\rho$ | max_iters | tol | max_terms | max_tree_nodes | prune_weight | prune_mse_slack | bootstrap_priors | splice_modes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `poly` | 80  | 20 | 0.015 | 0.02 | 14 | 1e-4 | 12 | 40 | 0.005 | 1.05 | True | (raw) |
| `trig` | 120 | 25 | 0.04  | 0.03 | 25 | 1e-4 | 12 | 40 | 0.008 | 1.08 | False | (raw, sin, tanh, relu, mul_sin_x, lin+sin) |
| `log` | 100 | 22 | 0.035 | 0.03 | 20 | 1e-4 | 10 | 36 | 0.008 | 1.08 | False | (raw, sin, exp_mix, lin+sin) |
| `sqrt` | 90 | 20 | 0.03  | 0.03 | 18 | 1e-4 | 10 | 36 | 0.008 | 1.08 | True | (raw, sin, relu, mul_sin_x) |
| `multi` | 140 | 30 | 0.04  | 0.03 | 28 | 1e-4 | 14 | 40 | 0.008 | 1.08 | True | (raw, sin, tanh, mul_sin_x, mul_sin_y, lin+sin) |

结构先验（`_build_prior_trees(profile)`）也随 profile 注入，例如：

- `poly`：$\{x,x^2,x^3,x^4,x^5,x^6\}$ 的等价树结构
- `trig`：$\sin(x),\sin(x^2),\sin(x^2)\sin(x),x^2\sin(x),\sin(x+x^2)$
- `log`：$\{x,x^2,\exp(x),\exp(x^2),x^2\}$(按实现的树结构注入)
- `sqrt`：$\{x,x^2,x^2\cdot x\}$(按实现的树结构注入)
- `multi`：同时包含 `y` 变量结构（如 `sin(y)`, `cos(y)`, `sin(x)*cos(y)` 与 `x*y` 等）

其中 `bootstrap_priors=True` 的 profile（如 `poly`、`multi`）会在残差迭代前先将结构先验一次性加入全局最小二乘回归，用作“先验引导”（prior-guided bootstrap），以提高高阶多项式或二维多项式/混合项在早期迭代的收敛概率。

此外，最终输出公式采用“有条件剪枝”：仅当剪枝后的 MSE 不超过全量模型 MSE 的 `prune_mse_slack` 倍时，才采用更短的表达式；否则保留全量模型以确保满足成功阈值的精度要求。

