#!/usr/bin/env python3
"""
Test R11 functionality on Livermore-11 benchmark.
Target: x²*y²/(x+y)
Expected: MSE ≈ 0 (exact match)
"""

import sys
sys.path.insert(0, '/Users/songjia9/Downloads/SRO-Autoresearch-autoresearch-may20')

from benchmarks.livermore_keijzer_vladislavleva import LIVERMORE_BENCHMARKS
from train import ResidualSolver, build_var_data
import numpy as np

def test_livermore_11():
    """Test Livermore-11: x²*y²/(x+y)"""
    print("=" * 60)
    print("Testing R11: Livermore-11 (x²*y²/(x+y))")
    print("=" * 60)
    
    case = LIVERMORE_BENCHMARKS[10]
    N_GRID = 127
    var_data = build_var_data(case, N_GRID)
    
    solver = ResidualSolver()
    solver.var_data = var_data
    
    y_obs = case['target'](var_data['x'], var_data['y'])
    y_obs = np.asarray(y_obs, dtype=np.float64)
    
    cfg_override = {
        'top_k': 120,
        'beam_limit': 25,
        'penalty_weight': 0.015,
        'prune_weight': 0.005,
    }
    
    mse, expr = solver.solve(
        y_obs=y_obs,
        profile=case.get('profile', 'multi'),
        suite='nguyen',
        verbose=True,
        cfg_override=cfg_override,
    )
    
    print("\n" + "=" * 60)
    print(f"=== Livermore-11 Result ===")
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
    success = test_livermore_11()
    sys.exit(0 if success else 1)
