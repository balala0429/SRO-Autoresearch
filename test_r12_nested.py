"""
测试 R12: 嵌套有理函数搜索
验证三种改进：
1. 幂次扩展：(a/b)², (a/b)³
2. 分母乘法扩展：a/(b*c)
3. 分子加法扩展：(a+b)/c
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/songjia9/Downloads/SRO-Autoresearch-autoresearch-may20')

from train import ResidualSolver, build_var_data

def test_nested_rational():
    """测试嵌套有理函数恢复"""
    
    test_cases = [
        # 幂次结构
        {
            'name': 'power_structure',
            'expr': '(x**2 * y**2 / (x + y))**2',
            'target': lambda x, y: (x**2 * y**2 / (x + y))**2,
            'x_range': (0.1, 2.0),
            'y_range': (0.1, 2.0),
        },
        # 分母乘法结构
        {
            'name': 'denominator_product',
            'expr': 'x**2 * y**2 / ((x + y) * (x - y))',
            'target': lambda x, y: x**2 * y**2 / ((x + y) * (x - y + 0.1)),  # 避免除零
            'x_range': (0.5, 2.0),
            'y_range': (0.5, 2.0),
        },
        # 分子加法结构
        {
            'name': 'numerator_sum',
            'expr': '(x**2 * y**2 + x * y) / (x + y)',
            'target': lambda x, y: (x**2 * y**2 + x * y) / (x + y),
            'x_range': (0.1, 2.0),
            'y_range': (0.1, 2.0),
        },
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"测试用例 {i+1}: {case['name']}")
        print(f"目标表达式: {case['expr']}")
        print(f"{'='*60}")
        
        # 准备数据
        N_GRID = 127
        x = np.linspace(case['x_range'][0], case['x_range'][1], N_GRID)
        y = np.linspace(case['y_range'][0], case['y_range'][1], N_GRID)
        X, Y = np.meshgrid(x, y)
        
        var_data = {
            'x': X.flatten(),
            'y': Y.flatten()
        }
        
        y_obs = case['target'](X, Y).flatten()
        
        # 创建求解器
        solver = ResidualSolver()
        solver.var_data = var_data
        
        # 求解
        try:
            mse, expr = solver.solve(
                y_obs=y_obs,
                profile='multi',
                suite='nguyen',
                verbose=True,
                cfg_override={
                    'top_k': 120,
                    'beam_limit': 25,
                    'penalty_weight': 0.015,
                    'prune_weight': 0.005,
                }
            )
            
            print(f"\n结果:")
            print(f"  MSE: {mse:.6e}")
            print(f"  恢复表达式: {expr}")
            
            success = mse < 1e-4
            print(f"  状态: {'✅ 成功' if success else '❌ 失败'}")
            
            results.append({
                'name': case['name'],
                'target': case['expr'],
                'recovered': expr,
                'mse': mse,
                'success': success
            })
            
        except Exception as e:
            print(f"  ❌ 异常: {str(e)}")
            results.append({
                'name': case['name'],
                'target': case['expr'],
                'recovered': None,
                'mse': float('inf'),
                'success': False
            })
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("R12 测试结果汇总")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"\n成功率: {success_count}/{total_count}")
    
    for r in results:
        status = '✅' if r['success'] else '❌'
        print(f"\n{status} {r['name']}")
        print(f"  目标: {r['target']}")
        print(f"  恢复: {r['recovered']}")
        print(f"  MSE: {r['mse']:.6e}")
    
    return results

if __name__ == '__main__':
    results = test_nested_rational()
