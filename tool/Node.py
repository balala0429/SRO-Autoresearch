import random
import numpy as np

from tool.tree_utils import is_valid_tree
# 算子映射表
OP_MAP = {
    '+': 0, '-': 1, '*': 2, '/': 3, 'pow': 4,
    'sin': 5, 'cos': 6, 'exp': 7, 'log': 8, 'sqrt': 9,
    'x': 10, 'y': 11, 'const': 12
}
VOCAB_SIZE = len(OP_MAP)
EMBED_DIM = 128  # 特征向量维度

def generate_random_tree(depth, allow_y=False):
    if depth <= 0 or (depth < 2 and random.random() < 0.3):
        # 生成叶子节点
        leaf = random.choice(['x', 'y', 'const']) if allow_y else random.choice(['x', 'const'])
        return Node(leaf)

    # 随机选一个算子
    op = random.choice(['+', '-', '*', '/', 'pow', 'sin', 'cos', 'exp', 'log', 'sqrt'])
    if op in ['sin', 'cos', 'exp', 'log', 'sqrt']:  # 一元算子
        return Node(op, left=generate_random_tree(depth - 1, allow_y=allow_y))
    else:  # 二元算子
        return Node(
            op,
            left=generate_random_tree(depth - 1, allow_y=allow_y),
            right=generate_random_tree(depth - 1, allow_y=allow_y),
        )

class Node:
    def __init__(self, op, left=None, right=None, value=None):
        self.op = op  # 字符串，如 '+', 'sin', 'x', 'const'
        self.left = left
        self.right = right
        self.value = value  # 仅用于 'const' 类型

    def evaluate(self, var_data):
        """
        var_data: numpy 数组（视为 x）或 dict，如 {'x': x_array, 'y': y_array}
        """
        try:
            if isinstance(var_data, np.ndarray):
                var_data = {'x': var_data}
            x_array = var_data.get('x', next(iter(var_data.values())))
            if self.op == 'x':
                return var_data['x']
            if self.op == 'y':
                return var_data.get('y', var_data['x'])
            elif self.op == 'const':
                if self.value is None:
                    # 给 const 一个默认小常数，避免 None 导致 nan
                    self.value = float(np.random.uniform(-2.0, 2.0))
                return np.full_like(x_array, self.value)

            # 递归计算子节点
            l_val = self.left.evaluate(var_data) if self.left else None
            r_val = self.right.evaluate(var_data) if self.right else None

            # 算子映射到 numpy 函数
            if self.op == '+': return l_val + r_val
            if self.op == '-': return l_val - r_val
            if self.op == '*': return l_val * r_val
            if self.op == '/':
                # 防止分母为0，添加微小扰动
                return l_val / (r_val + 1e-8)
            if self.op == 'sin': return np.sin(l_val)
            if self.op == 'cos': return np.cos(l_val)
            if self.op == 'exp':
                # 防止 exp 爆炸，进行数值截断
                return np.exp(np.clip(l_val, -10, 10))
            if self.op == 'log':
                return np.log(np.abs(l_val) + 1e-8)
            if self.op == 'sqrt':
                return np.sqrt(np.abs(l_val))
            if self.op == 'pow':
                # 安全幂：避免负底数与非整数指数导致复数
                base = np.abs(l_val) + 1e-8
                exp = np.clip(r_val, -3.0, 3.0)
                return np.power(base, exp)

        except Exception:
            return np.full_like(x_array, np.nan)


def generate_safe_tree(depth, x_samples=None, var_data=None, allow_y=False):
    """
    生成一个在指定采样点上数值稳定的随机树
    """
    if var_data is None:
        if x_samples is None:
            raise ValueError("generate_safe_tree requires x_samples or var_data")
        var_data = x_samples
    for _ in range(100):  # 最多尝试100次直到生成一个合法的树
        tree = generate_random_tree(depth, allow_y=allow_y)
        if not is_valid_tree(tree):
            continue
        y = tree.evaluate(var_data)

        # 检查合法性：结构完整 + 无 NaN/Inf + 数值范围适中
        if not np.any(np.isnan(y)) and not np.any(np.isinf(y)):
            if np.max(np.abs(y)) < 1e6 and np.var(y) >= 1e-6:
                return tree, y
    return None, None




if __name__ == '__main__':
    # 1. 设定采样点
    x_test = np.linspace(-5, 5, 200)

    # 2. 生成并打印 5 个随机子树及其数值响应特征
    print(f"{'Structure':<25} | {'Mean Y':<10} | {'Std Y':<10}")
    print("-" * 50)

    for _ in range(5):
        tree, y = generate_safe_tree(depth=2, x_samples=x_test)
        if tree:
            # 简单打印结构（递归展示）
            def get_str(n):
                if n.op in ['x', 'const']: return n.op if n.op == 'x' else str(round(n.value, 2))
                if n.right: return f"({get_str(n.left)} {n.op} {get_str(n.right)})"
                return f"{n.op}({get_str(n.left)})"


            struct_str = get_str(tree)
            print(f"{struct_str[:25]:<25} | {np.mean(y):<10.2f} | {np.std(y):<10.2f}")

