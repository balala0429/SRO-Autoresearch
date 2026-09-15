#!/usr/bin/env python3
"""
Test R11 functionality on Nguyen-12 benchmark.
Target: x⁴ - x³ + 0.5*y² - y
Expected: MSE ≈ 0 (exact match)
"""

import sys
sys.path.insert(0, '/Users/songjia9/Downloads/SRO-Autoresearch-autoresearch-may20')

import numpy as np
from train import ResidualSolver, build_var_data

def test_nguyen_12():
    """Test Nguyen-12: x⁴ - x³ + 0.5*y² - y"""
    print("=" * 60)
    print("Testing R11: Nguyen-12 (x⁴ - x³ + 0.5*y² - y)")
    print("=" * 60)
    
    # Nguyen-12 definition
    case = {
        "name": "nguyen-12",
        "dim": 2,
        "profile": "multi",
        "function_str": "x**4 - x**3 + 0.5*y**2 - y",
        "x_range": (-1, 1),
        "y_range": (-1, 1),
        "target": lambda x, y: x**4 - x**3 + 0.5*y**2 - y,
    }
    
    N_GRID = 127
    var_data = build_var_data(case, N_GRID)
    
    solver = ResidualSolver()
    solver.var_data = var_data
    
    y_obs = case['target'](var_data['x'], var_data['y'])
    y_obs = np.asarray(y_obs, dtype=np.float64)
    
    mse, expr = solver.solve(
        y_obs=y_obs,
        profile=case.get('profile', 'multi'),
        suite='nguyen',
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print(f"=== Nguyen-12 Result ===")
    print(f"MSE: {mse:.6e}")
    print(f"Expression: {expr}")
    print(f"Ground Truth: {case['function_str']}")
    print("=" * 60)
    
    # Check if successful
    if mse < 1e-4:
        print("✅ SUCCESS: Exact match achieved!")
        return True
    else:
        print("❌ FAILED: Could not achieve exact match")
        return False

def test_simple_rational():
    """Test a simple rational function: x²*y²/(x+y)"""
    print("=" * 60)
    print("Testing R11: Simple Rational (x²*y²/(x+y))")
    print("=" * 60)
    
    # Simple rational function
    case = {
        "name": "simple-rational",
        "dim": 2,
        "profile": "multi",
        "function_str": "x**2 * y**2 / (x + y)",
        "x_range": (0.1, 2),
        "y_range": (0.1, 2),
        "target": lambda x, y: (x**2 * y**2) / (x + y),
    }
    
    N_GRID = 127
    var_data = build_var_data(case, N_GRID)
    
    solver = ResidualSolver()
    solver.var_data = var_data
    
    y_obs = case['target'](var_data['x'], var_data['y'])
    y_obs = np.asarray(y_obs, dtype=np.float64)
    
    mse, expr = solver.solve(
        y_obs=y_obs,
        profile=case.get('profile', 'multi'),
        suite='nguyen',
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print(f"=== Simple Rational Result ===")
    print(f"MSE: {mse:.6e}")
    print(f"Expression: {expr}")
    print(f"Ground Truth: {case['function_str']}")
    print("=" * 60)
    
    # Check if successful
    if mse < 1e-4:
        print("✅ SUCCESS: Exact match achieved!")
        return True
    else:
        print("❌ FAILED: Could not achieve exact match")
        return False

if __name__ == '__main__':
    success1 = test_nguyen_12()
    success2 = test_simple_rational()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Nguyen-12: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  Simple Rational: {'✅ PASS' if success2 else '❌ FAIL'}")
    print("=" * 60)
    
    sys.exit(0 if (success1 and success2) else 1)
