"""
R13: 高价值模式库 - 已成功结构的复用与泛化

核心思路：
1. 存储已成功恢复的结构模式（如 Livermore-11 的 x²*y²/(x+y)）
2. 在新任务中优先尝试匹配这些模式
3. 通过模式相似度评估，选择最合适的候选
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from tool.Node import Node
from tool.tree_utils import tree_to_str, is_valid_tree


class HighValuePattern:
    """高价值模式定义"""
    
    def __init__(self, name: str, tree: Node, mse: float, metadata: Dict = None):
        self.name = name
        self.tree = tree
        self.mse = mse
        self.metadata = metadata or {}
        
    def __repr__(self):
        return f"HighValuePattern(name={self.name}, mse={self.mse:.2e})"


class PatternLibrary:
    """高价值模式库"""
    
    def __init__(self):
        self.patterns: List[HighValuePattern] = []
        self._init_builtin_patterns()
    
    def _init_builtin_patterns(self):
        """初始化内置模式（从 R11/R12 成功经验中提取）"""
        x = Node('x')
        y = Node('y')
        
        # Livermore-11: x²*y²/(x+y)
        x2y2 = Node('*', Node('*', x, x), Node('*', y, y))
        x_plus_y = Node('+', x, y)
        livermore_11 = Node('/', x2y2, x_plus_y)
        
        self.patterns.append(HighValuePattern(
            name="livermore-11",
            tree=livermore_11,
            mse=2.65e-16,
            metadata={
                "source": "R11",
                "type": "rational",
                "numerator_degree": 4,
                "denominator_degree": 1,
            }
        ))
        
        # 扩展模式：x*y/(x+y)
        xy = Node('*', x, y)
        ratio_1 = Node('/', xy, x_plus_y)
        
        self.patterns.append(HighValuePattern(
            name="xy_over_sum",
            tree=ratio_1,
            mse=0.0,
            metadata={
                "source": "R13_extension",
                "type": "rational",
                "numerator_degree": 2,
                "denominator_degree": 1,
            }
        ))
        
        # 扩展模式：(x²+y²)/(x+y)
        x2 = Node('*', x, x)
        y2 = Node('*', y, y)
        sum_squares = Node('+', x2, y2)
        ratio_2 = Node('/', sum_squares, x_plus_y)
        
        self.patterns.append(HighValuePattern(
            name="sum_squares_over_sum",
            tree=ratio_2,
            mse=0.0,
            metadata={
                "source": "R13_extension",
                "type": "rational",
                "numerator_degree": 2,
                "denominator_degree": 1,
            }
        ))
        
        # 扩展模式：x²*y/(x+y)
        x2y = Node('*', x2, y)
        ratio_3 = Node('/', x2y, x_plus_y)
        
        self.patterns.append(HighValuePattern(
            name="x2y_over_sum",
            tree=ratio_3,
            mse=0.0,
            metadata={
                "source": "R13_extension",
                "type": "rational",
                "numerator_degree": 3,
                "denominator_degree": 1,
            }
        ))
    
    def add_pattern(self, pattern: HighValuePattern):
        """添加新模式到库"""
        self.patterns.append(pattern)
        # 按 MSE 排序
        self.patterns.sort(key=lambda p: p.mse)
    
    def get_top_patterns(self, n: int = 10) -> List[HighValuePattern]:
        """获取前 N 个最佳模式"""
        return self.patterns[:n]
    
    def estimate_pattern_mse(self, pattern: HighValuePattern, 
                             eval_func, var_data: Dict) -> Tuple[float, float]:
        """
        评估模式在给定数据上的拟合质量
        
        Returns:
            (mse, weight): MSE 和最优权重
        """
        try:
            patch_y = eval_func(pattern.tree, var_data)
            y_obs = var_data.get('y_obs')
            
            if y_obs is None:
                return float('inf'), 0.0
            
            # 最小二乘拟合
            dot_prod = np.dot(patch_y, y_obs)
            norm_sq = np.dot(patch_y, patch_y)
            
            if norm_sq < 1e-12:
                return float('inf'), 0.0
            
            weight = dot_prod / norm_sq
            residual = y_obs - weight * patch_y
            mse = float(np.mean(residual ** 2))
            
            return mse, weight
            
        except Exception as e:
            return float('inf'), 0.0
    
    def find_best_pattern(self, eval_func, var_data: Dict, 
                         threshold: float = 1e-4) -> Optional[Tuple[HighValuePattern, float, float]]:
        """
        在模式库中查找最佳匹配
        
        Returns:
            (pattern, mse, weight) 或 None
        """
        best_pattern = None
        best_mse = float('inf')
        best_weight = 0.0
        
        for pattern in self.patterns:
            mse, weight = self.estimate_pattern_mse(pattern, eval_func, var_data)
            
            if mse < best_mse:
                best_mse = mse
                best_pattern = pattern
                best_weight = weight
                
                # 如果已经足够好，提前返回
                if mse < threshold:
                    break
        
        if best_mse < threshold and best_pattern is not None:
            return best_pattern, best_mse, best_weight
        
        return None


# 全局模式库实例
_pattern_library: Optional[PatternLibrary] = None


def get_pattern_library() -> PatternLibrary:
    """获取全局模式库实例"""
    global _pattern_library
    if _pattern_library is None:
        _pattern_library = PatternLibrary()
    return _pattern_library


def r13_pattern_matching(eval_func, var_data: Dict, 
                        threshold: float = 1e-4,
                        verbose: bool = True) -> Optional[Tuple[str, float, float]]:
    """
    R13 主入口：模式库匹配
    
    Args:
        eval_func: 树求值函数
        var_data: 变量数据（包含 'x', 'y', 'y_obs'）
        threshold: MSE 阈值
        verbose: 是否打印日志
    
    Returns:
        (expr_str, mse, weight) 或 None
    """
    library = get_pattern_library()
    
    if verbose:
        print(f"🔍 R13: 模式库匹配（{len(library.patterns)} 个模式）...")
    
    result = library.find_best_pattern(eval_func, var_data, threshold)
    
    if result is not None:
        pattern, mse, weight = result
        
        if verbose:
            expr_str = tree_to_str(pattern.tree)
            if abs(weight - 1.0) > 1e-6:
                full_expr = f"{weight:.6g}*{expr_str}"
            else:
                full_expr = expr_str
            
            print(f"🎯 R13 匹配成功: {pattern.name}")
            print(f"   Expression: {full_expr}")
            print(f"   MSE: {mse:.6e}")
            
            # 动态添加成功模式到库
            new_pattern = HighValuePattern(
                name=f"r13_discovered_{pattern.name}",
                tree=pattern.tree,
                mse=mse,
                metadata=pattern.metadata
            )
            library.add_pattern(new_pattern)
        
        expr_str = tree_to_str(pattern.tree)
        if abs(weight - 1.0) > 1e-6:
            full_expr = f"{weight:.6g}*{expr_str}"
        else:
            full_expr = expr_str
        
        return full_expr, mse, weight
    
    if verbose:
        print(f"❌ R13: 未找到匹配模式")
    
    return None
