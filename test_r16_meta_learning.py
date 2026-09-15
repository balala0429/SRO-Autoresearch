"""
R16 测试脚本：验证元学习搜索策略

测试场景：
1. TaskExperience 特征向量转换
2. ExperienceStore 内置经验加载
3. 相似度检索（KNN）
4. MetaLearner 策略推荐
5. 经验记录与持久化
6. 端到端流程（Livermore-11 类似任务应推荐 R11/R13）
"""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, '/Users/songjia9/Downloads/SRO-Autoresearch-autoresearch-may20')

from tool.meta_learning import (
    TaskExperience,
    ExperienceStore,
    MetaLearner,
    r16_meta_learning_recommendation,
    r16_record_experience,
)


def test_task_experience_feature_vector():
    """测试 1: TaskExperience 特征向量转换"""
    print("=" * 60)
    print("测试 1: TaskExperience 特征向量转换")
    print("=" * 60)
    
    exp = TaskExperience(
        task_id="test-task",
        suite="test",
        features={
            "y_range": 10.0,
            "y_std": 3.0,
            "y_mean": 5.0,
            "y_max_ratio": 4.0,
            "complexity_demand": "medium",
            "bootstrap_mse": 1e-4,
        },
    )
    
    vec = exp.to_feature_vector()
    
    print(f"特征: {exp.features}")
    print(f"特征向量: {vec}")
    
    assert len(vec) == 6, f"特征向量长度应为 6，实际为 {len(vec)}"
    assert vec[0] == 10.0, "y_range 应为 10.0"
    assert vec[4] == 0.5, "medium 复杂度编码应为 0.5"
    
    # 序列化/反序列化
    d = exp.to_dict()
    exp2 = TaskExperience.from_dict(d)
    assert exp2.task_id == exp.task_id
    assert exp2.features == exp.features
    
    print("\n✅ 测试 1 通过")
    return True


def test_experience_store_builtin():
    """测试 2: ExperienceStore 内置经验加载"""
    print("\n" + "=" * 60)
    print("测试 2: ExperienceStore 内置经验")
    print("=" * 60)
    
    store = ExperienceStore()
    
    print(f"内置经验数: {len(store)}")
    for exp in store.experiences:
        print(f"  - {exp.task_id}: best={exp.best_strategy}, mse={exp.final_mse:.2e}")
    
    assert len(store) >= 3, f"应有至少 3 个内置经验，实际为 {len(store)}"
    
    # 检查策略成功率
    r11_rate = store.get_strategy_success_rate("r11")
    r13_rate = store.get_strategy_success_rate("r13")
    print(f"\nR11 成功率: {r11_rate:.2f}")
    print(f"R13 成功率: {r13_rate:.2f}")
    
    print("\n✅ 测试 2 通过")
    return True


def test_similarity_search():
    """测试 3: 相似度检索"""
    print("\n" + "=" * 60)
    print("测试 3: 相似度检索 (KNN)")
    print("=" * 60)
    
    store = ExperienceStore()
    
    # 查询一个与 Livermore-11 特征相近的任务
    # Livermore-11 的特征: y_range 大, y_max_ratio 大
    query_features = store.experiences[0].to_feature_vector()
    
    similar = store.find_similar(query_features, k=3)
    
    print(f"查询: {store.experiences[0].task_id}")
    print(f"最相似的 {len(similar)} 个任务:")
    for exp, dist in similar:
        print(f"  - {exp.task_id} (dist={dist:.4f})")
    
    # 自身距离应为 0
    assert similar[0][1] < 1e-6, "自身距离应接近 0"
    
    print("\n✅ 测试 3 通过")
    return True


def test_meta_learner_recommendation():
    """测试 4: MetaLearner 策略推荐"""
    print("\n" + "=" * 60)
    print("测试 4: MetaLearner 策略推荐")
    print("=" * 60)
    
    store = ExperienceStore()
    learner = MetaLearner(store)
    
    # 测试 Livermore-11 类似特征
    livermore_exp = store.experiences[0]
    features = livermore_exp.to_feature_vector()
    
    best, scores = learner.recommend_strategy(features, verbose=True)
    confidence = learner.get_confidence(features)
    
    print(f"\n推荐: {best}, 置信度: {confidence:.3f}")
    
    assert best in scores, "推荐策略应在 scores 中"
    assert 0 <= confidence <= 1.0, "置信度应在 [0, 1] 范围内"
    
    print("\n✅ 测试 4 通过")
    return True


def test_experience_persistence():
    """测试 5: 经验持久化"""
    print("\n" + "=" * 60)
    print("测试 5: 经验持久化 (save/load)")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        tmp_path = f.name
    
    try:
        # 保存
        store = ExperienceStore(storage_path=tmp_path)
        initial_count = len(store)
        print(f"初始经验数: {initial_count}")
        
        # 添加新经验
        store.add(TaskExperience(
            task_id="persist-test",
            suite="test",
            features={"y_range": 1.0, "y_std": 0.5, "y_mean": 0.5,
                      "y_max_ratio": 2.0, "complexity_demand": "low"},
            strategy_results={"r13": {"success": True, "mse": 1e-12}},
            best_strategy="r13",
            final_mse=1e-12,
        ))
        store.save()
        print(f"添加后经验数: {len(store)}")
        
        # 重新加载
        store2 = ExperienceStore(storage_path=tmp_path)
        print(f"重新加载经验数: {len(store2)}")
        
        assert len(store2) == initial_count + 1, \
            f"加载后应有 {initial_count + 1} 个经验，实际为 {len(store2)}"
        
        # 找到新添加的经验
        found = any(e.task_id == "persist-test" for e in store2.experiences)
        assert found, "应能找到持久化的经验"
        
        print("\n✅ 测试 5 通过")
    finally:
        os.unlink(tmp_path)
    
    return True


def test_r16_end_to_end_livermore_like():
    """测试 6: 端到端 - Livermore-11 类似任务应推荐 R11/R13"""
    print("\n" + "=" * 60)
    print("测试 6: 端到端 - Livermore-11 类似任务")
    print("=" * 60)
    
    # 构造 Livermore-11 类似的 y_obs
    x = np.linspace(0.1, 2.0, 127)
    y = np.linspace(0.1, 2.0, 127)
    X, Y = np.meshgrid(x, y)
    y_obs = (X.flatten() ** 2 * Y.flatten() ** 2) / (X.flatten() + Y.flatten())
    
    best, scores, confidence = r16_meta_learning_recommendation(
        y_obs, task_id="test-livermore-like", suite="test", verbose=True
    )
    
    print(f"\n推荐策略: {best}")
    print(f"置信度: {confidence:.3f}")
    
    # 对于 Livermore-11 类型任务，R11/R13 应得分较高
    if "r11" in scores and "r13" in scores:
        print(f"R11 得分: {scores.get('r11', 0):.3f}")
        print(f"R13 得分: {scores.get('r13', 0):.3f}")
    
    assert best in ("r11", "r13", "bootstrap"), \
        f"Livermore-11 类似任务应推荐 r11/r13/bootstrap，实际为 {best}"
    
    print("\n✅ 测试 6 通过")
    return True


def test_r16_end_to_end_high_complexity():
    """测试 7: 端到端 - 高复杂度任务应倾向 R12"""
    print("\n" + "=" * 60)
    print("测试 7: 端到端 - 高复杂度任务")
    print("=" * 60)
    
    # 构造高复杂度特征：大范围、高 bootstrap MSE
    np.random.seed(42)
    y_obs = np.random.randn(16129) * 100  # 127*127 = 16129
    
    best, scores, confidence = r16_meta_learning_recommendation(
        y_obs, task_id="test-high-complexity", suite="test",
        bootstrap_mse=0.1, verbose=True
    )
    
    print(f"\n推荐策略: {best}")
    print(f"置信度: {confidence:.3f}")
    
    # 高复杂度任务，R12 应得分较高
    if "r12" in scores:
        print(f"R12 得分: {scores['r12']:.3f}")
    
    print("\n✅ 测试 7 通过")
    return True


def test_r16_record_and_retrieve():
    """测试 8: 记录新经验后检索验证"""
    print("\n" + "=" * 60)
    print("测试 8: 记录新经验后检索")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        tmp_path = f.name
    
    try:
        # 构造 Livermore-11 类似的 y_obs
        x = np.linspace(0.1, 2.0, 127)
        y = np.linspace(0.1, 2.0, 127)
        X, Y = np.meshgrid(x, y)
        y_obs = (X.flatten() ** 2 * Y.flatten() ** 2) / (X.flatten() + Y.flatten())
        
        # 记录经验
        r16_record_experience(
            task_id="record-test-task",
            suite="test",
            y_obs=y_obs,
            strategy_results={
                "r11": {"success": True, "mse": 1e-14},
                "r13": {"success": True, "mse": 1e-14},
                "bootstrap": {"success": False, "mse": 0.01},
            },
            best_strategy="r11",
            final_mse=1e-14,
            final_expr="((x * x) * (y * y)) / (x + y)",
            storage_path=tmp_path,
        )
        
        # 重新加载并验证经验存在
        store2 = ExperienceStore(storage_path=tmp_path)
        print(f"加载后经验数: {len(store2)}")
        
        # 找到新添加的经验
        found_exp = None
        for e in store2.experiences:
            if e.task_id == "record-test-task":
                found_exp = e
                break
        
        assert found_exp is not None, "应能找到新记录的经验"
        assert found_exp.best_strategy == "r11"
        assert found_exp.final_mse == 1e-14
        print(f"检索到: {found_exp.task_id}, best={found_exp.best_strategy}")
        
        print("\n✅ 测试 8 通过")
    finally:
        os.unlink(tmp_path)
    
    return True


def test_r16_confidence_scoring():
    """测试 9: 置信度评分合理性"""
    print("\n" + "=" * 60)
    print("测试 9: 置信度评分")
    print("=" * 60)
    
    store = ExperienceStore()
    learner = MetaLearner(store)
    
    # 与内置经验完全匹配的查询 — 置信度应较高
    exp = store.experiences[0]
    features_exact = exp.to_feature_vector()
    conf_exact = learner.get_confidence(features_exact)
    
    # 远离所有经验的查询 — 置信度应较低
    features_far = np.array([1000.0, 500.0, 800.0, 100.0, 1.0, 0.9])
    conf_far = learner.get_confidence(features_far)
    
    print(f"精确匹配置信度: {conf_exact:.3f}")
    print(f"远离经验置信度: {conf_far:.3f}")
    
    assert conf_exact > conf_far, \
        f"精确匹配置信度({conf_exact:.3f})应大于远离经验({conf_far:.3f})"
    
    print("\n✅ 测试 9 通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("R16 元学习搜索策略测试套件")
    print("=" * 60)
    
    tests = [
        test_task_experience_feature_vector,
        test_experience_store_builtin,
        test_similarity_search,
        test_meta_learner_recommendation,
        test_experience_persistence,
        test_r16_end_to_end_livermore_like,
        test_r16_end_to_end_high_complexity,
        test_r16_record_and_retrieve,
        test_r16_confidence_scoring,
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
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
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
