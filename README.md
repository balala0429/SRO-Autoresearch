# SRO_PRO: Symbolic Regression Optimization

SRO_PRO 是一个基于“残差检索 + 符号组件拼接 + 全局最小二乘重拟合”的符号回归项目。  
当前支持 1D/2D 的 Nguyen 基准（Nguyen-1 ~ Nguyen-12），并提供批量评估与可视化导出。

## 1. 核心思路

- **残差驱动**：每轮拟合当前残差 `r = y - y_pred`。
- **神经检索**：`ResidualPredictor` 将残差映射到向量空间，使用 FAISS 在符号库中检索候选子树。
- **结构先验**：按任务类型（poly/trig/log/sqrt/multi）注入先验子树。
- **非线性拼接**：支持 `raw/sin/tanh/relu/mul_sin_x/mul_sin_y/lin+sin` 等拼接模式。
- **全局重拟合**：候选加入后对全部 active 列进行最小二乘重拟合。
- **表达式后处理**：使用 SymPy 化简并清理极小系数噪声项。

---

## 2. 目录结构

- `train.py`：主求解逻辑 + Nguyen 基准评估入口
- `run_full_nguyen_report.py`：10×100 统计与图表导出脚本
- `Predictor/ResidualPredictor.py`：残差向量预测网络
- `tool/`：树结构、求值、可视化等工具
- `library/library/`：FAISS 索引与符号树库
- `reports/`：评估产出（Markdown/CSV 表格、拟合图）
- `benchmarks/extra_symbolic_benchmarks.py`：新增扩展数据集定义
- `run_extra_report.py`：扩展数据集实验脚本

---

## 3. 环境安装

推荐使用 conda（Apple Silicon / Mac 环境已验证）：

```bash
conda create -n sro python=3.10 -y
conda activate sro
pip install -U pip
pip install numpy sympy faiss-cpu tqdm matplotlib torch --index-url https://download.pytorch.org/whl/cpu
```

快速验证：

```bash
python -c "import torch,faiss,numpy,sympy; print(torch.__version__, faiss.__version__)"
```

---

## 4. 运行方式

### 4.1 Nguyen 重复基准（默认 10×100）

```bash
conda activate sro
python train.py --num_rounds 10 --num_trials 100
```

参数：

- `--num_rounds`：轮数（默认 10）
- `--num_trials`：每轮试验次数（默认 100）
- `--no_cache`：关闭缓存，真实逐次求解（更慢，但更严格）
- **你现在可直接运行**
  ### **SRBench-like**
  conda activate sro
  python run_unified_adapter_[report.py](http://report.py) --source srbench --max_count 10 --num_rounds 1 --num_trials 10
  ### **PMLB 指定数据集**
  python run_unified_adapter_[report.py](http://report.py) --source pmlb --datasets 1027_ESL,1028_SWD --num_rounds 1 --num_trials 10
  ### **Feynman-SR**
  python run_unified_adapter_[report.py](http://report.py) --source feynman --max_count 10 --num_rounds 1 --num_trials 10

### 4.2 一键导出表格 + 拟合图（推荐）

```bash
conda activate sro
python run_full_nguyen_report.py --num_rounds 10 --num_trials 100
```

输出：

- `reports/nguyen_<rounds>x<trials>_<timestamp>.md`
- `reports/nguyen_<rounds>x<trials>_<timestamp>.csv`
- `reports/plots_<timestamp>/Nguyen-*.png`

---

## 5. 扩展数据集实验

项目已补充 `benchmarks/extra_symbolic_benchmarks.py`，包含常见扩展基准（Koza/Pagie 等）用于对比测试。

运行：

```bash
conda activate sro
python run_extra_report.py --num_rounds 3 --num_trials 30
```

---

## 6. 统一实验平台（Nguyen + SRBench/PMLB/Feynman）

推荐统一入口（支持表格 + CSV + 拟合图）：

`run_benchmark_platform.py`

示例：

```bash
# Nguyen 全基准
python run_benchmark_platform.py --source nguyen --num_rounds 10 --num_trials 100

# 扩展基准（benchmarks/extra_symbolic_benchmarks.py）
python run_benchmark_platform.py --source extra --num_rounds 3 --num_trials 30

# SRBench-like
python run_benchmark_platform.py --source srbench --max_count 10 --num_rounds 1 --num_trials 10

# Feynman-SR（若 pmlb 无内置则自动使用内置方程生成）
python run_benchmark_platform.py --source feynman --max_count 10 --num_rounds 1 --num_trials 10

# 指定 PMLB 数据集
python run_benchmark_platform.py --source pmlb --datasets 1027_ESL,1028_SWD --num_rounds 1 --num_trials 10
```

输出统一落在 `reports/`：

- `<source>_<rounds>x<trials>_<timestamp>.md`
- `<source>_<rounds>x<trials>_<timestamp>.csv`
- `plots_<source>_<timestamp>/`

---

## 7. 统一数据适配层（SRBench / PMLB / Feynman-SR）

已提供统一适配模块：

- `benchmarks/unified_data_adapter.py`
- 统一入口：`run_unified_adapter_report.py`

### 6.1 安装额外依赖

```bash
pip install pmlb pandas
```

### 6.2 跑 SRBench-like 数据（自动发现）

```bash
python run_unified_adapter_report.py --source srbench --max_count 10 --num_rounds 1 --num_trials 10
```

### 6.3 跑 Feynman-SR（从 PMLB 中筛选 feynman*）

```bash
python run_unified_adapter_report.py --source feynman --max_count 10 --num_rounds 1 --num_trials 10
```

### 6.4 指定 PMLB 数据集

```bash
python run_unified_adapter_report.py --source pmlb --datasets 1027_ESL,529_pollen --num_rounds 1 --num_trials 10
```

说明：

- 统一适配层会把表格数据转换为当前 SRO 可接受任务格式。
- 由于当前模型资产限定为 1D(127) 和 2D(32×32)，高维数据会自动截取前 1~2 个数值特征。
- 1D 采用插值重采样；2D 采用最近邻映射到规则网格。
- 输出报告保存到 `reports/*_unified_*.md`。

---

## 8. 结果解释

- **Success Rate**：成功次数 / 总次数（成功阈值默认 `MSE < 1e-4`）
- **Avg MSE**：对应数据集平均 MSE
- **Best Expression**：最优表达式（SymPy 化简后）

---

## 9. 常见问题

### Q1: `python: command not found`

用 `python3` 或先激活 conda：

```bash
conda activate sro
python train.py
```

### Q2: `ModuleNotFoundError: torch`

说明环境没激活或依赖未安装，请重新执行第 3 节安装步骤。

### Q3: Mac 上偶发 segmentation fault

项目已在 `train.py` 中默认限制 BLAS/OpenMP 线程并启用 `faulthandler`，一般可稳定运行。

---

## 10. 后续可扩展方向

- 接入 SRBench/PMLB/Feynman 数据集进行跨数据集泛化评估
- 增加表达式复杂度与可解释性指标（项数、树深、符号复杂度）
- 引入 Pareto 前沿筛选（MSE vs Complexity）

