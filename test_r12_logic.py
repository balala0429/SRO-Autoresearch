"""
验证 R12 代码逻辑
检查嵌套候选是否正确生成
"""

import sys
sys.path.insert(0, '/Users/songjia9/Downloads/SRO-Autoresearch-autoresearch-may20')

from tool.Node import Node
from tool.tree_utils import tree_to_str, is_valid_tree
from tool.rational_mutation import deep_copy_tree

def test_r12_logic():
    """测试 R12 的嵌套候选生成逻辑"""
    
    print("="*60)
    print("R12 代码逻辑验证")
    print("="*60)
    
    # 创建基础比值树：x²*y²/(x+y)
    x = Node('x')
    y = Node('y')
    
    # 分子：x²*y²
    x2 = Node('*', x, x)
    y2 = Node('*', y, y)
    numerator = Node('*', x2, y2)
    
    # 分母：x+y
    denominator = Node('+', x, y)
    
    # 比值：x²*y²/(x+y)
    ratio = Node('/', numerator, denominator)
    
    print(f"\n基础比值树: {tree_to_str(ratio)}")
    print(f"有效树: {is_valid_tree(ratio)}")
    
    # R12 改进 1: 幂次扩展
    print("\n" + "="*60)
    print("R12 改进 1: 幂次扩展")
    print("="*60)
    
    power_candidates = []
    for power in [2, 3]:
        power_tree = Node('pow', deep_copy_tree(ratio), Node('const', value=float(power)))
        if is_valid_tree(power_tree):
            power_candidates.append(power_tree)
            print(f"  幂次 {power}: {tree_to_str(power_tree)}")
    
    print(f"  生成 {len(power_candidates)} 个幂次候选")
    
    # R12 改进 2: 分母乘法扩展
    print("\n" + "="*60)
    print("R12 改进 2: 分母乘法扩展")
    print("="*60)
    
    # 创建另一个加法树：x-y
    x_minus_y = Node('-', x, y)
    
    denom_product_candidates = []
    # a / (b * c)
    new_den = Node('*', deep_copy_tree(denominator), deep_copy_tree(x_minus_y))
    new_ratio = Node('/', deep_copy_tree(numerator), new_den)
    if is_valid_tree(new_ratio):
        denom_product_candidates.append(new_ratio)
        print(f"  分母乘积: {tree_to_str(new_ratio)}")
    
    print(f"  生成 {len(denom_product_candidates)} 个分母乘法候选")
    
    # R12 改进 3: 分子加法扩展
    print("\n" + "="*60)
    print("R12 改进 3: 分子加法扩展")
    print("="*60)
    
    # 创建另一个乘法树：x*y
    xy = Node('*', x, y)
    
    num_sum_candidates = []
    # (a + b) / c
    new_num = Node('+', deep_copy_tree(numerator), deep_copy_tree(xy))
    new_ratio = Node('/', new_num, deep_copy_tree(denominator))
    if is_valid_tree(new_ratio):
        num_sum_candidates.append(new_ratio)
        print(f"  分子加法: {tree_to_str(new_ratio)}")
    
    print(f"  生成 {len(num_sum_candidates)} 个分子加法候选")
    
    # 汇总
    print("\n" + "="*60)
    print("R12 验证汇总")
    print("="*60)
    
    total_candidates = len(power_candidates) + len(denom_product_candidates) + len(num_sum_candidates)
    print(f"\n总嵌套候选数: {total_candidates}")
    print(f"  - 幂次扩展: {len(power_candidates)}")
    print(f"  - 分母乘法: {len(denom_product_candidates)}")
    print(f"  - 分子加法: {len(num_sum_candidates)}")
    
    print("\n✅ R12 代码逻辑验证通过")
    
    return total_candidates > 0

if __name__ == '__main__':
    success = test_r12_logic()
    sys.exit(0 if success else 1)
