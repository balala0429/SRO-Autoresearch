"""符号树结构校验、安全字符串化与复杂度度量。"""
import numpy as np

UNARY_OPS = frozenset({'sin', 'cos', 'exp', 'log', 'sqrt'})
BINARY_OPS = frozenset({'+', '-', '*', '/', 'pow'})
LEAF_OPS = frozenset({'x', 'y', 'const'})


def is_valid_tree(node):
    """检查树结构是否完整：二元算子必须有左右子树，一元算子必须有左子树。"""
    if node is None:
        return False
    op = node.op
    if op in LEAF_OPS:
        if op == 'const':
            return node.value is not None and np.isfinite(float(node.value))
        return node.left is None and node.right is None
    if op in UNARY_OPS:
        return node.left is not None and is_valid_tree(node.left) and node.right is None
    if op in BINARY_OPS:
        return (
            node.left is not None
            and node.right is not None
            and is_valid_tree(node.left)
            and is_valid_tree(node.right)
        )
    return False


def get_tree_size(node):
    if node is None:
        return 0
    return 1 + get_tree_size(node.left) + get_tree_size(node.right)


def tree_to_str(node):
    """将合法树转为可读表达式；非法树返回空字符串。"""
    if not is_valid_tree(node):
        return ""
    if node.op in LEAF_OPS:
        if node.op == 'const':
            return f"{node.value:.4g}"
        return node.op
    if node.op in UNARY_OPS:
        inner = tree_to_str(node.left)
        return f"{node.op}({inner})"
    if node.op in BINARY_OPS:
        left = tree_to_str(node.left)
        right = tree_to_str(node.right)
        if not left or not right:
            return ""
        if node.op == 'pow':
            return f"({left})**({right})"
        return f"({left} {node.op} {right})"
    return ""


def entry_to_str(entry):
    """将 (tree, splice_mode) 或 tree 转为单项表达式片段。"""
    if isinstance(entry, tuple):
        tree, mode = entry
    else:
        tree, mode = entry, 'raw'
    base = tree_to_str(tree)
    if not base:
        return ""
    if mode == 'sin':
        return f"sin({base})"
    if mode == 'tanh':
        return f"tanh({base})"
    if mode == 'relu':
        return f"relu({base})"
    if mode == 'mul_sin_x':
        return f"({base})*sin(x)"
    if mode == 'mul_sin_y':
        return f"({base})*sin(y)"
    if mode == 'exp_mix':
        return f"exp({base})"
    return base


def format_weighted_term(weight, entry):
    """生成带符号的加权项，避免 '+ -15.1027 * ...' 这种可读性差的形式。"""
    term = entry_to_str(entry)
    if not term:
        return None
    w = float(weight)
    if abs(w) < 1e-12:
        return None
    if w >= 0:
        return f"{w:.4g}*({term})"
    return f"-{abs(w):.4g}*({term})"


def format_formula(weights, entries):
    parts = []
    for w, e in zip(weights, entries):
        s = format_weighted_term(w, e)
        if s:
            parts.append(s)
    if not parts:
        return "0"
    return " + ".join(parts)


def count_active_complexity(entries):
    """估计最终公式复杂度：项数 + 各基树节点数之和。"""
    n_terms = len(entries)
    n_nodes = 0
    for entry in entries:
        if isinstance(entry, tuple):
            tree = entry[0]
        else:
            tree = entry
        n_nodes += get_tree_size(tree)
    return n_terms, n_nodes
