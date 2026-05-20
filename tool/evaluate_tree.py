import numpy as np


def evaluate_tree(node, x):
    """
    递归评估符号树在给定采样点 x 上的数值。
    x: numpy array, 采样点序列
    """
    if node is None:
        return np.zeros_like(x)

    # 1. 处理叶子节点
    if node.op == 'x':
        return x

    if node.op == 'const':
        # 返回与 x 形状相同的常数数组
        return np.full_like(x, node.value)

    # 2. 递归获取左右子树的计算结果
    left_val = evaluate_tree(node.left, x) if node.left else None
    right_val = evaluate_tree(node.right, x) if node.right else None

    # 3. 执行算子逻辑 (包含安全保护)
    op = node.op

    try:
        if op == '+':
            return left_val + right_val
        elif op == '-':
            return left_val - right_val
        elif op == '*':
            return left_val * right_val
        elif op == '/':
            # 安全除法：防止分母为 0
            return left_val / (right_val + 1e-8)
        elif op == 'sin':
            return np.sin(left_val)
        elif op == 'cos':
            return np.cos(left_val)
        elif op == 'exp':
            # 限制范围防止数值爆炸 (如 exp(700) 会溢出)
            return np.exp(np.clip(left_val, -100, 100))
        elif op == 'log':
            # 安全对数：防止负数和零
            return np.log(np.abs(left_val) + 1e-8)
        elif op == 'pow2':
            return np.power(left_val, 2)
        elif op == 'sqrt':
            return np.sqrt(np.abs(left_val))
        else:
            raise ValueError(f"Unknown operator: {op}")

    except Exception as e:
        # 如果计算过程中出现溢出或无效值，返回全 0 或一个极大的惩罚值
        return np.zeros_like(x)