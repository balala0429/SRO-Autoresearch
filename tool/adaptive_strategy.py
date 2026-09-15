"""
R14: 自适应搜索策略

核心思路：
1. 根据任务特征（数据分布、bootstrap MSE、复杂度需求等）动态选择搜索策略
2. 策略评估器：评估不同策略的适用性得分
3. 策略选择器：基于评估结果选择最优策略
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum


class SearchStrategy(Enum):
    """搜索策略枚举"""
    R11_RATIONAL = "r11_rational"  # 有理函数搜索
    R12_NESTED = "r12_nested"      # 嵌套有理函数搜索
    R13_PATTERN = "r13_pattern"    # 模式库匹配
    BOOTSTRAP = "bootstrap"        # 标准 bootstrap
    HYBRID = "hybrid"              # 混合策略


class TaskFeatures:
    """任务特征提取"""
    
    def __init__(self, y_obs: np.ndarray, bootstrap_mse: float = None):
        self.y_obs = y_obs
        self.bootstrap_mse = bootstrap_mse
        
        # 数据特征
        self.y_range = float(np.max(y_obs) - np.min(y_obs))
        self.y_std = float(np.std(y_obs))
        self.y_mean = float(np.mean(y_obs))
        self.y_max_ratio = float(np.max(np.abs(y_obs)) / (np.abs(np.mean(y_obs)) + 1e-10))
        
        # 复杂度需求（通过 bootstrap MSE 推断）
        self.complexity_demand = self._estimate_complexity_demand()
    
    def _estimate_complexity_demand(self) -> str:
        """根据 bootstrap MSE 估计复杂度需求"""
        if self.bootstrap_mse is None:
            return "unknown"
        
        if self.bootstrap_mse < 1e-6:
            return "low"
        elif self.bootstrap_mse < 1e-3:
            return "medium"
        else:
            return "high"
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "y_range": self.y_range,
            "y_std": self.y_std,
            "y_mean": self.y_mean,
            "y_max_ratio": self.y_max_ratio,
            "complexity_demand": self.complexity_demand,
            "bootstrap_mse": self.bootstrap_mse,
        }


class StrategyEvaluator:
    """策略评估器"""
    
    def __init__(self):
        self.strategy_scores: Dict[SearchStrategy, float] = {}
    
    def evaluate(self, task_features: TaskFeatures) -> Dict[SearchStrategy, float]:
        """评估所有策略的适用性得分"""
        
        scores = {}
        
        # R13 模式库匹配策略评估
        scores[SearchStrategy.R13_PATTERN] = self._evaluate_r13_pattern(task_features)
        
        # R11 有理函数搜索策略评估
        scores[SearchStrategy.R11_RATIONAL] = self._evaluate_r11_rational(task_features)
        
        # R12 嵌套有理函数搜索策略评估
        scores[SearchStrategy.R12_NESTED] = self._evaluate_r12_nested(task_features)
        
        # Bootstrap 策略评估
        scores[SearchStrategy.BOOTSTRAP] = self._evaluate_bootstrap(task_features)
        
        # 混合策略评估
        scores[SearchStrategy.HYBRID] = self._evaluate_hybrid(scores)
        
        self.strategy_scores = scores
        return scores
    
    def _evaluate_r13_pattern(self, features: TaskFeatures) -> float:
        """评估 R13 模式库匹配策略"""
        # 模式库匹配成本低，始终给予基础分
        score = 0.15
        
        # 如果 bootstrap MSE 很低，说明任务简单，模式库可能直接匹配
        if features.bootstrap_mse is not None and features.bootstrap_mse < 1e-6:
            score += 0.8
        
        # 数据范围适中时模式库更可能匹配
        if 0.1 < features.y_range < 10.0:
            score += 0.2
        elif features.y_range >= 10.0:
            # 大范围任务也可能匹配有理函数模式
            score += 0.1
        
        return score
    
    def _evaluate_r11_rational(self, features: TaskFeatures) -> float:
        """评估 R11 有理函数搜索策略"""
        score = 0.0
        
        # 如果 bootstrap MSE 中等，可能需要有理函数结构
        if features.bootstrap_mse is not None:
            if 1e-4 < features.bootstrap_mse < 1e-2:
                score += 0.6
        
        # 如果数据有较大的动态范围，可能需要有理函数
        if features.y_max_ratio > 5.0:
            score += 0.3
        
        # 如果复杂度需求中等，R11 可能适用
        if features.complexity_demand == "medium":
            score += 0.1
        
        return score
    
    def _evaluate_r12_nested(self, features: TaskFeatures) -> float:
        """评估 R12 嵌套有理函数搜索策略"""
        score = 0.0
        
        # 如果 bootstrap MSE 较高，可能需要更复杂的嵌套结构
        if features.bootstrap_mse is not None and features.bootstrap_mse > 1e-2:
            score += 0.5
        
        # 如果复杂度需求高，R12 可能适用
        if features.complexity_demand == "high":
            score += 0.4
        
        # 如果数据范围很大，可能需要嵌套结构
        if features.y_range > 10.0:
            score += 0.1
        
        return score
    
    def _evaluate_bootstrap(self, features: TaskFeatures) -> float:
        """评估 Bootstrap 策略"""
        score = 0.0
        
        # 如果 bootstrap MSE 已经很低，直接使用
        if features.bootstrap_mse is not None and features.bootstrap_mse < 1e-4:
            score += 0.9
        
        # 如果复杂度需求低，bootstrap 可能足够
        if features.complexity_demand == "low":
            score += 0.1
        
        return score
    
    def _evaluate_hybrid(self, individual_scores: Dict[SearchStrategy, float]) -> float:
        """评估混合策略"""
        # 混合策略的得分是其他策略得分的加权平均
        weights = {
            SearchStrategy.R13_PATTERN: 0.3,
            SearchStrategy.R11_RATIONAL: 0.3,
            SearchStrategy.R12_NESTED: 0.3,
            SearchStrategy.BOOTSTRAP: 0.1,
        }
        
        hybrid_score = 0.0
        for strategy, weight in weights.items():
            if strategy in individual_scores:
                hybrid_score += weight * individual_scores[strategy]
        
        return hybrid_score


class StrategySelector:
    """策略选择器"""
    
    def __init__(self, evaluator: StrategyEvaluator):
        self.evaluator = evaluator
    
    def select(self, task_features: TaskFeatures) -> Tuple[SearchStrategy, Dict[SearchStrategy, float]]:
        """选择最优策略"""
        scores = self.evaluator.evaluate(task_features)
        
        # 选择得分最高的策略
        best_strategy = max(scores, key=scores.get)
        
        return best_strategy, scores
    
    def select_with_fallback(self, task_features: TaskFeatures) -> List[SearchStrategy]:
        """选择策略并返回回退顺序"""
        scores = self.evaluator.evaluate(task_features)
        
        # 按得分排序
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回策略列表（按得分降序）
        return [strategy for strategy, _ in sorted_strategies]


def r14_adaptive_strategy_selection(
    y_obs: np.ndarray, 
    bootstrap_mse: float = None,
    verbose: bool = True
) -> Tuple[SearchStrategy, List[SearchStrategy], Dict[SearchStrategy, float]]:
    """
    R14 主入口：自适应策略选择
    
    Args:
        y_obs: 观测数据
        bootstrap_mse: Bootstrap 后的 MSE（可选）
        verbose: 是否打印日志
    
    Returns:
        (best_strategy, fallback_order, scores)
    """
    if verbose:
        print("\n🎯 R14: 自适应策略选择")
    
    # 1. 提取任务特征
    task_features = TaskFeatures(y_obs, bootstrap_mse)
    
    if verbose:
        print(f"   数据范围: {task_features.y_range:.4f}")
        print(f"   数据标准差: {task_features.y_std:.4f}")
        print(f"   最大比值: {task_features.y_max_ratio:.4f}")
        if bootstrap_mse is not None:
            print(f"   Bootstrap MSE: {bootstrap_mse:.6e}")
        print(f"   复杂度需求: {task_features.complexity_demand}")
    
    # 2. 策略评估
    evaluator = StrategyEvaluator()
    selector = StrategySelector(evaluator)
    
    best_strategy, scores = selector.select(task_features)
    fallback_order = selector.select_with_fallback(task_features)
    
    if verbose:
        print("\n   策略评估得分:")
        for strategy, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"     {strategy.value}: {score:.3f}")
        
        print(f"\n   最优策略: {best_strategy.value}")
        print(f"   回退顺序: {[s.value for s in fallback_order[:3]]}")
    
    return best_strategy, fallback_order, scores


def r14_execute_strategy(
    strategy: SearchStrategy,
    solver,
    y_obs: np.ndarray,
    verbose: bool = True
) -> Optional[Tuple[float, str]]:
    """
    执行指定策略
    
    Args:
        strategy: 搜索策略
        solver: ResidualSolver 实例
        y_obs: 观测数据
        verbose: 是否打印日志
    
    Returns:
        (mse, expr) 或 None
    """
    if verbose:
        print(f"\n   执行策略: {strategy.value}")
    
    # 根据策略类型执行不同的搜索方法
    if strategy == SearchStrategy.R13_PATTERN:
        # R13 已在 solve() 中集成，这里不需要额外操作
        pass
    
    elif strategy == SearchStrategy.R11_RATIONAL:
        # R11 已在 solve() 中集成
        pass
    
    elif strategy == SearchStrategy.R12_NESTED:
        # R12 已在 solve() 中集成
        pass
    
    elif strategy == SearchStrategy.BOOTSTRAP:
        # 标准 bootstrap，不需要特殊处理
        pass
    
    elif strategy == SearchStrategy.HYBRID:
        # 混合策略：按顺序尝试所有策略
        # 这已经在 solve() 的 R13 -> R11 -> R12 流程中实现
        pass
    
    return None  # 实际执行在 solve() 中完成
