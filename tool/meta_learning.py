"""
R16: 元学习搜索策略 — 跨任务迁移搜索经验

核心思路：
1. 从历史任务中提取经验：任务特征向量 + 各策略的实际效果
2. 建立任务相似度模型：基于特征向量的 KNN 检索
3. 策略推荐：基于相似任务的历史表现，加权投票选择最优策略
4. 经验积累：每次求解后自动记录经验，持续改进策略选择

与 R14 的区别：
- R14 基于规则的静态评估（启发式打分）
- R16 基于历史数据的动态学习（数据驱动）
- R16 可作为 R14 的补充/替代，在积累足够经验后更准确
"""

import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict


@dataclass
class TaskExperience:
    """单条任务经验记录"""
    
    # 任务标识
    task_id: str  # 如 "livermore-11", "keijzer-6"
    suite: str    # 如 "klv", "nguyen"
    
    # 任务特征向量
    features: Dict[str, float]  # y_range, y_std, y_mean, y_max_ratio, complexity_demand(编码)
    
    # 各策略的实际效果
    strategy_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 格式: {"r11": {"success": True, "mse": 1e-10, "time_ms": 50}, ...}
    
    # 最终结果
    best_strategy: Optional[str] = None
    final_mse: float = float('inf')
    final_expr: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TaskExperience':
        return cls(**d)
    
    def to_feature_vector(self) -> np.ndarray:
        """转换为特征向量（用于相似度计算）"""
        complexity_map = {"low": 0.0, "medium": 0.5, "high": 1.0, "unknown": 0.5}
        return np.array([
            self.features.get("y_range", 0.0),
            self.features.get("y_std", 0.0),
            self.features.get("y_mean", 0.0),
            self.features.get("y_max_ratio", 0.0),
            complexity_map.get(self.features.get("complexity_demand", "unknown"), 0.5),
            self.features.get("bootstrap_mse", 0.5) if self.features.get("bootstrap_mse") is not None else 0.5,
        ])


class ExperienceStore:
    """经验存储与管理"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.experiences: List[TaskExperience] = []
        self.storage_path = storage_path
        
        # 加载已有经验
        if storage_path and os.path.exists(storage_path):
            self.load(storage_path)
        
        # 初始化内置经验（从 R11/R13 的成功案例中提取）
        if len(self.experiences) == 0:
            self._init_builtin_experiences()
    
    def _init_builtin_experiences(self):
        """初始化内置经验（从 R11/R13 成功案例中提取）"""
        
        # Livermore-11: x²*y²/(x+y) — R11 成功
        x = np.linspace(0.1, 2.0, 127)
        y = np.linspace(0.1, 2.0, 127)
        X, Y = np.meshgrid(x, y)
        y_obs = (X.flatten()**2 * Y.flatten()**2) / (X.flatten() + Y.flatten())
        
        y_range = float(np.max(y_obs) - np.min(y_obs))
        y_std = float(np.std(y_obs))
        y_mean = float(np.mean(y_obs))
        y_max_ratio = float(np.max(np.abs(y_obs)) / (np.abs(np.mean(y_obs)) + 1e-10))
        
        self.add(TaskExperience(
            task_id="livermore-11",
            suite="klv",
            features={
                "y_range": y_range,
                "y_std": y_std,
                "y_mean": y_mean,
                "y_max_ratio": y_max_ratio,
                "complexity_demand": "medium",
                "bootstrap_mse": None,
            },
            strategy_results={
                "r11": {"success": True, "mse": 2.65e-16},
                "r13": {"success": True, "mse": 2.65e-16},
                "bootstrap": {"success": False, "mse": 1e-2},
            },
            best_strategy="r13",
            final_mse=2.65e-16,
            final_expr="((x * x) * (y * y)) / (x + y)",
        ))
        
        # 模拟几个 KLV 任务的经验（基于典型特征）
        # 高复杂度任务（R12 更有效）
        self.add(TaskExperience(
            task_id="klv_high_complexity_example",
            suite="klv",
            features={
                "y_range": 100.0,
                "y_std": 30.0,
                "y_mean": 10.0,
                "y_max_ratio": 50.0,
                "complexity_demand": "high",
                "bootstrap_mse": 0.1,
            },
            strategy_results={
                "r11": {"success": False, "mse": 0.05},
                "r12": {"success": True, "mse": 1e-8},
                "r13": {"success": False, "mse": 0.1},
                "bootstrap": {"success": False, "mse": 0.5},
            },
            best_strategy="r12",
            final_mse=1e-8,
        ))
        
        # 低复杂度任务（Bootstrap 足够）
        self.add(TaskExperience(
            task_id="klv_low_complexity_example",
            suite="klv",
            features={
                "y_range": 2.0,
                "y_std": 0.5,
                "y_mean": 1.0,
                "y_max_ratio": 3.0,
                "complexity_demand": "low",
                "bootstrap_mse": 1e-8,
            },
            strategy_results={
                "r11": {"success": False, "mse": 1e-4},
                "r12": {"success": False, "mse": 1e-4},
                "r13": {"success": True, "mse": 1e-10},
                "bootstrap": {"success": True, "mse": 1e-8},
            },
            best_strategy="bootstrap",
            final_mse=1e-8,
        ))
    
    def add(self, experience: TaskExperience):
        """添加新经验"""
        # 检查是否已存在（按 task_id 去重）
        for i, exp in enumerate(self.experiences):
            if exp.task_id == experience.task_id:
                self.experiences[i] = experience  # 更新
                return
        
        self.experiences.append(experience)
    
    def find_similar(self, features: np.ndarray, k: int = 5) -> List[Tuple[TaskExperience, float]]:
        """
        查找最相似的 k 个任务
        
        Args:
            features: 查询任务的特征向量
            k: 返回的邻居数量
        
        Returns:
            List of (experience, distance)
        """
        if len(self.experiences) == 0:
            return []
        
        # 计算与所有经验的距离
        distances = []
        for exp in self.experiences:
            exp_vec = exp.to_feature_vector()
            dist = np.linalg.norm(features - exp_vec)
            distances.append((exp, dist))
        
        # 按距离排序，返回前 k 个
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def get_strategy_success_rate(self, strategy: str) -> float:
        """计算某策略的整体成功率"""
        total = 0
        success = 0
        for exp in self.experiences:
            if strategy in exp.strategy_results:
                total += 1
                if exp.strategy_results[strategy].get("success", False):
                    success += 1
        return success / total if total > 0 else 0.0
    
    def save(self, path: Optional[str] = None):
        """保存经验到文件"""
        path = path or self.storage_path
        if path is None:
            return
        
        data = [exp.to_dict() for exp in self.experiences]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, path: str):
        """从文件加载经验"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.experiences = [TaskExperience.from_dict(d) for d in data]
        except Exception as e:
            print(f"⚠️ R16: 加载经验失败: {e}")
            self.experiences = []
    
    def __len__(self) -> int:
        return len(self.experiences)


class MetaLearner:
    """元学习策略选择器"""
    
    def __init__(self, store: ExperienceStore, k_neighbors: int = 5):
        self.store = store
        self.k_neighbors = k_neighbors
    
    def predict_strategy_scores(
        self, 
        features: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        基于历史经验预测各策略的得分
        
        使用加权 KNN：距离越近的经验权重越大
        每个策略的得分 = 加权平均（成功率 * inverse_mse）
        """
        similar = self.store.find_similar(features, k=self.k_neighbors)
        
        if len(similar) == 0:
            return {}
        
        # 计算权重（距离越近权重越大）
        weights = []
        for exp, dist in similar:
            # 使用高斯核作为权重
            sigma = 1.0  # 带宽参数
            w = np.exp(-dist**2 / (2 * sigma**2))
            weights.append(w)
        
        weights = np.array(weights)
        if weights.sum() < 1e-10:
            weights = np.ones(len(weights)) / len(weights)  # 均匀权重
        else:
            weights = weights / weights.sum()
        
        # 收集所有出现过的策略
        all_strategies = set()
        for exp, _ in similar:
            all_strategies.update(exp.strategy_results.keys())
        
        # 对每个策略计算加权得分
        scores = {}
        for strategy in all_strategies:
            score = 0.0
            for (exp, dist), w in zip(similar, weights):
                if strategy in exp.strategy_results:
                    result = exp.strategy_results[strategy]
                    success = 1.0 if result.get("success", False) else 0.0
                    mse = result.get("mse", float('inf'))
                    
                    # 得分 = 成功率 + inverse_mse_bonus
                    inverse_mse_bonus = 0.0
                    if mse < float('inf') and mse > 0:
                        inverse_mse_bonus = min(1.0, 1.0 / (1.0 + mse))
                    
                    score += w * (0.6 * success + 0.4 * inverse_mse_bonus)
            
            scores[strategy] = score
        
        if verbose:
            print(f"📚 R16: 基于 {len(similar)} 个相似任务的经验:")
            for i, (exp, dist) in enumerate(similar[:3]):
                print(f"   [{i+1}] {exp.task_id} (dist={dist:.3f}, best={exp.best_strategy})")
        
        return scores
    
    def recommend_strategy(
        self, 
        features: np.ndarray,
        verbose: bool = True
    ) -> Tuple[str, Dict[str, float]]:
        """
        推荐最优策略
        
        Returns:
            (best_strategy_name, scores_dict)
        """
        scores = self.predict_strategy_scores(features, verbose)
        
        if len(scores) == 0:
            return "bootstrap", {}  # 默认回退到 bootstrap
        
        best_strategy = max(scores, key=scores.get)
        
        if verbose:
            print(f"   策略推荐得分:")
            for s, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"     {s}: {sc:.3f}")
            print(f"   推荐策略: {best_strategy}")
        
        return best_strategy, scores
    
    def get_confidence(self, features: np.ndarray) -> float:
        """
        计算推荐置信度
        
        基于：
        1. 相似任务的数量（越多越可靠）
        2. 最近邻距离（越近越可靠）
        3. 最佳策略的绝对得分（越高越可靠）
        """
        similar = self.store.find_similar(features, k=self.k_neighbors)
        
        if len(similar) == 0:
            return 0.0
        
        # 因子 1: 经验数量（越多越可靠）
        n_factor = min(1.0, len(self.store) / 10.0)
        
        # 因子 2: 最近距离（越小越可靠）
        min_dist = similar[0][1] if similar else float('inf')
        dist_factor = 1.0 / (1.0 + min_dist)
        
        # 因子 3: 最佳策略得分（越高越可靠）
        scores = self.predict_strategy_scores(features, verbose=False)
        if scores:
            best_score = max(scores.values())
        else:
            best_score = 0.0
        
        confidence = (n_factor + dist_factor + best_score) / 3.0
        return confidence


def r16_meta_learning_recommendation(
    y_obs: np.ndarray,
    task_id: str = "unknown",
    suite: str = "unknown",
    bootstrap_mse: Optional[float] = None,
    verbose: bool = True,
    storage_path: Optional[str] = None,
) -> Tuple[str, Dict[str, float], float]:
    """
    R16 主入口：元学习策略推荐
    
    Args:
        y_obs: 观测数据
        task_id: 任务标识
        suite: 数据集名称
        bootstrap_mse: Bootstrap 后的 MSE（可选）
        verbose: 是否打印日志
        storage_path: 经验存储路径
    
    Returns:
        (recommended_strategy, scores, confidence)
    """
    if verbose:
        print(f"\n📚 R16: 元学习策略推荐 (task={task_id})")
    
    # 1. 提取特征
    y_range = float(np.max(y_obs) - np.min(y_obs))
    y_std = float(np.std(y_obs))
    y_mean = float(np.mean(y_obs))
    y_max_ratio = float(np.max(np.abs(y_obs)) / (np.abs(np.mean(y_obs)) + 1e-10))
    
    # 复杂度需求编码
    if bootstrap_mse is None:
        complexity = "unknown"
        complexity_val = 0.5
    elif bootstrap_mse < 1e-6:
        complexity = "low"
        complexity_val = 0.0
    elif bootstrap_mse < 1e-3:
        complexity = "medium"
        complexity_val = 0.5
    else:
        complexity = "high"
        complexity_val = 1.0
    
    bootstrap_mse_val = bootstrap_mse if bootstrap_mse is not None else 0.5
    
    features = np.array([
        y_range, y_std, y_mean, y_max_ratio, complexity_val, bootstrap_mse_val
    ])
    
    if verbose:
        print(f"   特征: range={y_range:.3f}, std={y_std:.3f}, max_ratio={y_max_ratio:.3f}, complexity={complexity}")
    
    # 2. 元学习推荐
    store = ExperienceStore(storage_path)
    learner = MetaLearner(store)
    
    best_strategy, scores = learner.recommend_strategy(features, verbose)
    confidence = learner.get_confidence(features)
    
    if verbose:
        print(f"   置信度: {confidence:.3f}")
    
    return best_strategy, scores, confidence


def r16_record_experience(
    task_id: str,
    suite: str,
    y_obs: np.ndarray,
    strategy_results: Dict[str, Dict[str, Any]],
    best_strategy: str,
    final_mse: float,
    final_expr: Optional[str] = None,
    bootstrap_mse: Optional[float] = None,
    storage_path: Optional[str] = None,
) -> None:
    """
    记录任务经验
    
    在任务求解完成后调用，将结果保存到经验库
    """
    # 提取特征
    y_range = float(np.max(y_obs) - np.min(y_obs))
    y_std = float(np.std(y_obs))
    y_mean = float(np.mean(y_obs))
    y_max_ratio = float(np.max(np.abs(y_obs)) / (np.abs(np.mean(y_obs)) + 1e-10))
    
    if bootstrap_mse is None:
        complexity = "unknown"
    elif bootstrap_mse < 1e-6:
        complexity = "low"
    elif bootstrap_mse < 1e-3:
        complexity = "medium"
    else:
        complexity = "high"
    
    # 创建经验记录
    experience = TaskExperience(
        task_id=task_id,
        suite=suite,
        features={
            "y_range": y_range,
            "y_std": y_std,
            "y_mean": y_mean,
            "y_max_ratio": y_max_ratio,
            "complexity_demand": complexity,
            "bootstrap_mse": bootstrap_mse,
        },
        strategy_results=strategy_results,
        best_strategy=best_strategy,
        final_mse=final_mse,
        final_expr=final_expr,
    )
    
    # 保存到经验库
    store = ExperienceStore(storage_path)
    store.add(experience)
    store.save(storage_path)
    
    print(f"📝 R16: 记录经验 {task_id} -> {best_strategy} (MSE={final_mse:.2e})")
