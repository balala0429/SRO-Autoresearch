import random
import numpy as np
# 算子映射表
OP_MAP = {'+': 0, '-': 1, '*': 2, '/': 3, 'sin': 4, 'exp': 5, 'x': 6, 'const': 7}
VOCAB_SIZE = len(OP_MAP)
EMBED_DIM = 128  # 特征向量维度

def generate_random_tree(depth):
    if depth <= 0 or (depth < 2 and random.random() < 0.3):
        # 生成叶子节点
        leaf = random.choice(['x', 'const'])
        return Node(leaf)

    # 随机选一个算子
    op = random.choice(['+', '-', '*', '/', 'sin', 'exp'])
    if op in ['sin', 'exp']:  # 一元算子
        return Node(op, left=generate_random_tree(depth - 1))
    else:  # 二元算子
        return Node(op, left=generate_random_tree(depth - 1),
                    right=generate_random_tree(depth - 1))

class Node:
    def __init__(self, op, left=None, right=None, value=None):
        self.op = op  # 字符串，如 '+', 'sin', 'x', 'const'
        self.left = left
        self.right = right
        self.value = value  # 仅用于 'const' 类型

    def evaluate(self, x_array):
        """
        x_array: numpy 数组，代表采样点
        """
        try:
            if self.op == 'x':
                return x_array
            elif self.op == 'const':
                return np.full_like(x_array, self.value)

            # 递归计算子节点
            l_val = self.left.evaluate(x_array) if self.left else None
            r_val = self.right.evaluate(x_array) if self.right else None

            # 算子映射到 numpy 函数
            if self.op == '+': return l_val + r_val
            if self.op == '-': return l_val - r_val
            if self.op == '*': return l_val * r_val
            if self.op == '/':
                # 防止分母为0，添加微小扰动
                return l_val / (r_val + 1e-8)
            if self.op == 'sin': return np.sin(l_val)
            if self.op == 'exp':
                # 防止 exp 爆炸，进行数值截断
                return np.exp(np.clip(l_val, -10, 10))

        except Exception:
            return np.full_like(x_array, np.nan)


def generate_safe_tree(depth, x_samples):
    """
    生成一个在指定采样点上数值稳定的随机树
    """
    for _ in range(100):  # 最多尝试100次直到生成一个合法的树
        tree = generate_random_tree(depth)  # 使用之前定义的递归生成函数
        y = tree.evaluate(x_samples)

        # 检查合法性：没有 NaN，没有无穷大，且数值范围适中
        if not np.any(np.isnan(y)) and not np.any(np.isinf(y)):
            if np.max(np.abs(y)) < 1e6:  # 过滤掉数值过大的无效树
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

