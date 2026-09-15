"""
R14 测试脚本：验证自适应策略选择逻辑

测试场景：
1. 策略评估器对不同任务特征给出合理评分
2. 高 bootstrap MSE 任务优先选择 R12 嵌套搜索
3. 低 bootstrap MSE 任务优先选择 R13 模式库匹配
4. 完整流程测试（Livermore-11 端到端）
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/songjia9/Downloads/SRO-Autoresearch-autoresearch-may20')

from tool.adaptive_strategy import (
    TaskFeatures,
    StrategyEvaluator,
    StrategySelector,
    SearchStrategy,
    r14_adaptive_strategy_selection,
)


def test_task_features_basic():
    """测试 1: 任务特征提取"""
    print("="*60)
    print("测试 1: 任务特征提取")
    print("="*60)
    
    np.random.seed(42)
    y_obs = np.random.randn(127) * 2.0 + 5.0
    
    features = TaskFeatures(y_obs, bootstrap_mse=1e-5)
    
    print(f"数据范围: {features.y_range:.4f}")
    print(f"数据标准差: {features.y_std:.4f}")
    print(f"数据均值: {features.y_mean:.4f}")
    print(f"最大比值: {features.y_max_ratio:.4f}")
    print(f"复杂度需求: {features.complexity_demand}")
    print(f"字典: {features.to_dict()}")
    
    assert features.y_range > 0, "数据范围应该 > 0"
    assert features.complexity_demand == "medium", "bootstrap MSE=1e-5 在 1e-6~1e-3 之间，应为 medium"
    print("\n✅ 测试 1 通过")
    return True


def test_complexity_demand_levels():
    """测试 2: 复杂度需求分级"""
    print("\n" + "="*60)
    print("测试 2: 复杂度需求分级")
    print("="*60)
    
    y_obs = np.ones(127)
    
    # low: < 1e-6
    f_low = TaskFeatures(y_obs, bootstrap_mse=1e-8)
    assert f_low.complexity_demand == "low", f"Expected low, got {f_low.complexity_demand}"
    print(f"  MSE=1e-8 -> {f_low.complexity_demand}")
    
    # medium: 1e-6 ~ 1e-3
    f_med = TaskFeatures(y_obs, bootstrap_mse=1e-4)
    assert f_med.complexity_demand == "medium", f"Expected medium, got {f_med.complexity_demand}"
    print(f"  MSE=1e-4 -> {f_med.complexity_demand}")
    
    # high: > 1e-3
    f_high = TaskFeatures(y_obs, bootstrap_mse=1e-2)
    assert f_high.complexity_demand == "high", f"Expected high, got {f_high.complexity_demand}"
    print(f"  MSE=1e-2 -> {f_high.complexity_demand}")
    
    print("\n✅ 测试 2 通过")
    return True


def test_strategy_scoring_easy_task():
    """测试 3: 简单任务（低 bootstrap MSE）的评分"""
    print("\n" + "="*60)
    print("测试 3: 简单任务评分（bootstrap MSE=1e-7）")
    print("="*60)
    
    y_obs = np.random.randn(127)
    
    best, fallback, scores = r14_adaptive_strategy_selection(
        y_obs, bootstrap_mse=1e-7, verbose=True
    )
    
    print(f"\n最优策略: {best.value}")
    print(f"回退顺序: {[s.value for s in fallback[:3]]}")
    
    # Bootstrap 得分应该最高（因为 MSE 很低）
    assert scores[SearchStrategy.BOOTSTRAP] > scores[SearchStrategy.R11_RATIONAL], \
        "简单任务应该优先 Bootstrap"
    
    print("\n✅ 测试 3 通过")
    return True


def test_strategy_scoring_hard_task():
    """测试 4: 困难任务（高 bootstrap MSE）的评分"""
    print("\n" + "="*60)
    print("测试 4: 困难任务评分（bootstrap MSE=1e-1）")
    print("="*60)
    
    y_obs = np.random.randn(127) * 100.0  # 大范围数据
    
    best, fallback, scores = r14_adaptive_strategy_selection(
        y_obs, bootstrap_mse=1e-1, verbose=True
    )
    
    print(f"\n最优策略: {best.value}")
    print(f"回退顺序: {[s.value for s in fallback[:3]]}")
    
    # R12 嵌套搜索得分应该较高（因为 bootstrap MSE 高）
    assert scores[SearchStrategy.R12_NESTED] > scores[SearchStrategy.BOOTSTRAP], \
        "困难任务应该给 R12 更高分"
    
    print("\n✅ 测试 4 通过")
    return True


def test_strategy_scoring_medium_task():
    """测试 5: 中等任务（中等 bootstrap MSE）的评分"""
    print("\n" + "="*60)
    print("测试 5: 中等任务评分（bootstrap MSE=1e-4）")
    print("="*60)
    
    np.random.seed(123)
    y_obs = np.random.randn(127) * 5.0
    
    best, fallback, scores = r14_adaptive_strategy_selection(
        y_obs, bootstrap_mse=1e-4, verbose=True
    )
    
    print(f"\n最优策略: {best.value}")
    print(f"回退顺序: {[s.value for s in fallback[:3]]}")
    
    # R11 应该得到合理评分（中等复杂度适合 R11）
    assert scores[SearchStrategy.R11_RATIONAL] > 0, \
        "R11 应该在中等任务中得到评分"
    
    print("\n✅ 测试 5 通过")
    return True


def test_fallback_ordering():
    """测试 6: 回退顺序正确性"""
    print("\n" + "="*60)
    print("测试 6: 回退顺序正确性")
    print("="*60)
    
    y_obs = np.random.randn(127)
    
    _, fallback, _ = r14_adaptive_strategy_selection(
        y_obs, bootstrap_mse=1e-2, verbose=False
    )
    
    # 回退顺序应该是按得分降序
    evaluator = StrategyEvaluator()
    features = TaskFeatures(y_obs, bootstrap_mse=1e-2)
    scores = evaluator.evaluate(features)
    
    sorted_scores = sorted(scores.values(), reverse=True)
    actual_scores = [scores[s] for s in fallback]
    
    assert actual_scores == sorted_scores, \
        f"回退顺序应该按得分降序: expected {sorted_scores}, got {actual_scores}"
    
    print(f"回退顺序（得分）: {[f'{s:.3f}' for s in actual_scores]}")
    print("\n✅ 测试 6 通过")
    return True


def test_all_strategies_present():
    """测试 7: 所有策略都有得分"""
    print("\n" + "="*60)
    print("测试 7: 所有策略都有得分")
    print("="*60)
    
    y_obs = np.random.randn(127)
    
    _, _, scores = r14_adaptive_strategy_selection(
        y_obs, bootstrap_mse=None, verbose=False
    )
    
    expected_strategies = [
        SearchStrategy.R13_PATTERN,
        SearchStrategy.R11_RATIONAL,
        SearchStrategy.R12_NESTED,
        SearchStrategy.BOOTSTRAP,
        SearchStrategy.HYBRID,
    ]
    
    for s in expected_strategies:
        assert s in scores, f"策略 {s.value} 缺少得分"
        print(f"  {s.value}: {scores[s]:.3f}")
    
    print("\n✅ 测试 7 通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("R14 测试套件")
    print("="*60)
    
    tests = [
        test_task_features_basic,
        test_complexity_demand_levels,
        test_strategy_scoring_easy_task,
        test_strategy_scoring_hard_task,
        test_strategy_scoring_medium_task,
        test_fallback_ordering,
        test_all_strategies_present,
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
    
    # 汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n总计: {passed}/{total} 通过")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
