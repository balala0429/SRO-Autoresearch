"""
R13 测试脚本：验证高价值模式库功能

测试场景：
1. 直接匹配 Livermore-11 模式（x²*y²/(x+y)）
2. 模式库扩展能力（动态添加新发现的模式）
3. 失败回退（当模式库匹配失败时，正确回退到 R11/R12）
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/songjia9/Downloads/SRO-Autoresearch-autoresearch-may20')

from tool.Node import Node
from tool.tree_utils import tree_to_str
from tool.pattern_library import (
    PatternLibrary, 
    HighValuePattern, 
    get_pattern_library,
    r13_pattern_matching
)
from train import eval_tree


def test_pattern_library_initialization():
    """测试 1: 模式库初始化"""
    print("="*60)
    print("测试 1: 模式库初始化")
    print("="*60)
    
    library = get_pattern_library()
    
    print(f"模式库大小: {len(library.patterns)}")
    print("\n初始模式:")
    for i, pattern in enumerate(library.patterns):
        print(f"  {i+1}. {pattern.name}: {tree_to_str(pattern.tree)}")
        print(f"     MSE: {pattern.mse:.2e}")
    
    assert len(library.patterns) > 0, "模式库应该至少有一个内置模式"
    print("\n✅ 测试 1 通过")
    return True


def test_pattern_matching_livermore11():
    """测试 2: 匹配 Livermore-11 模式"""
    print("\n" + "="*60)
    print("测试 2: 匹配 Livermore-11 模式")
    print("="*60)
    
    # 创建测试数据：x²*y²/(x+y)
    x = np.linspace(0.1, 2.0, 127)
    y = np.linspace(0.1, 2.0, 127)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # 目标函数
    y_obs = (X_flat**2 * Y_flat**2) / (X_flat + Y_flat)
    
    var_data = {
        'x': X_flat,
        'y': Y_flat,
        'y_obs': y_obs
    }
    
    # 执行模式匹配
    result = r13_pattern_matching(eval_tree, var_data, threshold=1e-4, verbose=True)
    
    if result is not None:
        expr_str, mse, weight = result
        print(f"\n匹配结果:")
        print(f"  表达式: {expr_str}")
        print(f"  MSE: {mse:.6e}")
        print(f"  权重: {weight:.6f}")
        
        assert mse < 1e-4, f"MSE 应该小于 1e-4，但实际为 {mse}"
        print("\n✅ 测试 2 通过")
        return True
    else:
        print("\n❌ 测试 2 失败：未找到匹配模式")
        return False


def test_pattern_library_extension():
    """测试 3: 模式库扩展能力"""
    print("\n" + "="*60)
    print("测试 3: 模式库扩展能力")
    print("="*60)
    
    library = get_pattern_library()
    initial_size = len(library.patterns)
    
    # 添加新模式
    x = Node('x')
    y = Node('y')
    x3 = Node('*', Node('*', x, x), x)
    y2 = Node('*', y, y)
    x3_y2 = Node('*', x3, y2)
    x_plus_y = Node('+', x, y)
    new_pattern_tree = Node('/', x3_y2, x_plus_y)
    
    new_pattern = HighValuePattern(
        name="x3y2_over_sum",
        tree=new_pattern_tree,
        mse=1e-10,
        metadata={"source": "test"}
    )
    
    library.add_pattern(new_pattern)
    
    print(f"初始模式数: {initial_size}")
    print(f"添加后模式数: {len(library.patterns)}")
    
    assert len(library.patterns) == initial_size + 1, "模式数应该增加 1"
    print("\n✅ 测试 3 通过")
    return True


def test_pattern_matching_failure():
    """测试 4: 匹配失败（无法匹配的模式）"""
    print("\n" + "="*60)
    print("测试 4: 匹配失败回退")
    print("="*60)
    
    # 创建一个无法匹配的数据：sin(x) * cos(y)
    x = np.linspace(0.1, 2.0, 127)
    y = np.linspace(0.1, 2.0, 127)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    y_obs = np.sin(X_flat) * np.cos(Y_flat)
    
    var_data = {
        'x': X_flat,
        'y': Y_flat,
        'y_obs': y_obs
    }
    
    # 执行模式匹配（应该失败）
    result = r13_pattern_matching(eval_tree, var_data, threshold=1e-4, verbose=True)
    
    if result is None:
        print("\n✅ 测试 4 通过：正确返回 None（无匹配）")
        return True
    else:
        expr_str, mse, weight = result
        print(f"\n❌ 测试 4 失败：意外匹配到 {expr_str} (MSE={mse:.6e})")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("R13 测试套件")
    print("="*60)
    
    tests = [
        test_pattern_library_initialization,
        test_pattern_matching_livermore11,
        test_pattern_library_extension,
        test_pattern_matching_failure,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n❌ 测试 {test.__name__} 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 通过")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
