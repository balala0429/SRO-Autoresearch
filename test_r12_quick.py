"""
快速测试 R12: 验证嵌套有理函数搜索的基本功能
只测试一个用例：幂次结构 (x²*y²/(x+y))²
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/songjia9/Downloads/SRO-Autoresearch-autoresearch-may20')

from train import ResidualSolver

def test_power_structure():
    """测试幂次结构恢复"""
    
    print("="*60)
    print("R12 快速测试: 幂次结构 (x²*y²/(x+y))²")
    print("="*60)
    
    # 准备数据（使用 1D 数据以匹配 predictor 期望的 127 个点）
    N_POINTS = 127
    x = np.linspace(0.1, 2.0, N_POINTS)
    y = np.linspace(0.1, 2.0, N_POINTS)
    
    # 使用简单的 1D 数据（x 和 y 相同）
    var_data = {
        'x': x,
        'y': y
    }
    
    # 目标函数: (x²*y²/(x+y))²
    y_obs = (x**2 * y**2 / (x + y))**2
    
    print(f"\n数据点数量: {len(y_obs)}")
    print(f"y_obs 范围: [{y_obs.min():.4f}, {y_obs.max():.4f}]")
    
    # 创建求解器
    solver = ResidualSolver()
    solver.var_data = var_data
    
    # 求解
    print("\n开始求解...")
    try:
        mse, expr = solver.solve(
            y_obs=y_obs,
            profile='multi',
            suite='nguyen',
            verbose=True
        )
        
        print(f"\n{'='*60}")
        print("结果:")
        print(f"{'='*60}")
        print(f"MSE: {mse:.6e}")
        print(f"恢复表达式: {expr}")
        
        success = mse < 1e-4
        print(f"\n状态: {'✅ 成功' if success else '❌ 失败'}")
        
        return success, mse, expr
        
    except Exception as e:
        print(f"\n❌ 异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, float('inf'), None

if __name__ == '__main__':
    success, mse, expr = test_power_structure()
    sys.exit(0 if success else 1)
