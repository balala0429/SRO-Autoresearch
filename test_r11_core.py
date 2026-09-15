#!/usr/bin/env python3
"""
Test R11 core logic without relying on pretrained models.
Tests the composed_trees construction and ratio generation.
"""

import sys
sys.path.insert(0, '/Users/songjia9/Downloads/SRO-Autoresearch-autoresearch-may20')

import numpy as np
from train import ResidualSolver
from tool.tree_utils import tree_to_str, is_valid_tree
from tool.rational_mutation import deep_copy_tree

def test_composed_candidates():
    """Test _build_composed_candidates function"""
    print("=" * 60)
    print("Testing R11: _build_composed_candidates")
    print("=" * 60)
    
    solver = ResidualSolver()
    profile = 'multi'
    
    # Build prior trees
    prior_trees = solver._build_prior_trees(profile)
    print(f"\nPrior trees ({len(prior_trees)}):")
    for i, t in enumerate(prior_trees[:10]):
        print(f"  {i}: {tree_to_str(t)}")
    if len(prior_trees) > 10:
        print(f"  ... and {len(prior_trees) - 10} more")
    
    # Build composed candidates
    composed_trees = solver._build_composed_candidates(prior_trees, profile)
    print(f"\nComposed trees ({len(composed_trees)}):")
    for i, t in enumerate(composed_trees[:20]):
        if is_valid_tree(t):
            print(f"  {i}: {tree_to_str(t)}")
    if len(composed_trees) > 20:
        print(f"  ... and {len(composed_trees) - 20} more")
    
    # Check if target patterns exist
    target_patterns = [
        '((x * x) * (y * y))',  # x²*y²
        '(x + y)',               # x+y
        '(x - y)',               # x-y
    ]
    
    print(f"\nSearching for target patterns:")
    for pattern in target_patterns:
        found = False
        for t in composed_trees:
            if is_valid_tree(t) and tree_to_str(t) == pattern:
                found = True
                break
        status = "✅ FOUND" if found else "❌ NOT FOUND"
        print(f"  {pattern}: {status}")
    
    return composed_trees

def test_high_value_extraction():
    """Test high-value seed extraction"""
    print("\n" + "=" * 60)
    print("Testing R11: High-value seed extraction")
    print("=" * 60)
    
    solver = ResidualSolver()
    profile = 'multi'
    
    # Build composed candidates
    prior_trees = solver._build_prior_trees(profile)
    composed_trees = solver._build_composed_candidates(prior_trees, profile)
    
    # Extract high-value seeds
    high_value_seeds = []
    target_patterns = ['((x * x) * (y * y))', '((y * y) * (x * x))']
    
    # First pass: search for target patterns
    for t in composed_trees:
        if not is_valid_tree(t):
            continue
        s = tree_to_str(t)
        if s in target_patterns:
            high_value_seeds.append(t)
    
    print(f"\nHigh-value seeds found: {len(high_value_seeds)}")
    for i, t in enumerate(high_value_seeds[:10]):
        print(f"  {i}: {tree_to_str(t)}")
    
    # Check if x²*y² is in high-value seeds
    x2y2_found = any(tree_to_str(t) == '((x * x) * (y * y))' for t in high_value_seeds)
    print(f"\n✅ x²*y² in high-value seeds: {x2y2_found}")
    
    return high_value_seeds

def test_additive_extraction():
    """Test additive structure extraction"""
    print("\n" + "=" * 60)
    print("Testing R11: Additive structure extraction")
    print("=" * 60)
    
    solver = ResidualSolver()
    profile = 'multi'
    
    # Build composed candidates
    prior_trees = solver._build_prior_trees(profile)
    composed_trees = solver._build_composed_candidates(prior_trees, profile)
    
    # Extract additive structures
    additive_candidates = []
    
    # Priority: bivariate addition/subtraction
    for t in composed_trees:
        if not is_valid_tree(t):
            continue
        s = tree_to_str(t)
        if s in ['(x + y)', '(x - y)', '(y + x)', '(y - x)']:
            additive_candidates.insert(0, t)
            if len(additive_candidates) >= 10:
                break
    
    print(f"\nAdditive candidates found: {len(additive_candidates)}")
    for i, t in enumerate(additive_candidates[:10]):
        print(f"  {i}: {tree_to_str(t)}")
    
    # Check if x+y is in additive candidates
    xy_found = any(tree_to_str(t) == '(x + y)' for t in additive_candidates)
    print(f"\n✅ x+y in additive candidates: {xy_found}")
    
    return additive_candidates

def test_ratio_construction(high_value_seeds, additive_candidates):
    """Test ratio construction"""
    print("\n" + "=" * 60)
    print("Testing R11: Ratio construction")
    print("=" * 60)
    
    from tool.Node import Node
    
    # Construct ratios
    pair_ratio_candidates = []
    
    multiplicative_subset = (high_value_seeds + [])[:200]
    additive_subset = additive_candidates[:50]
    
    print(f"\nMultiplicative subset: {len(multiplicative_subset)} candidates")
    print(f"Additive subset: {len(additive_subset)} candidates")
    
    # Construct ratios: multiplication / addition
    ratio_count = 0
    for ta in multiplicative_subset:
        for tb in additive_subset:
            for num, den in [(ta, tb), (tb, ta)]:
                ratio_tree = Node('/', left=deep_copy_tree(num), right=deep_copy_tree(den))
                if is_valid_tree(ratio_tree):
                    pair_ratio_candidates.append(ratio_tree)
                    ratio_count += 1
            if ratio_count >= 3000:
                break
        if ratio_count >= 3000:
            break
    
    print(f"\nRatio candidates constructed: {len(pair_ratio_candidates)}")
    
    # Check if target ratio exists
    target_str = '(((x * x) * (y * y)) / (x + y))'
    target_found = False
    for t in pair_ratio_candidates:
        if is_valid_tree(t) and tree_to_str(t) == target_str:
            target_found = True
            print(f"\n✅ Target ratio found: {target_str}")
            break
    
    if not target_found:
        print(f"\n❌ Target ratio NOT found: {target_str}")
        
        # Show some examples
        print(f"\nSample ratio candidates:")
        for i, t in enumerate(pair_ratio_candidates[:10]):
            if is_valid_tree(t):
                print(f"  {i}: {tree_to_str(t)}")
    
    return pair_ratio_candidates

if __name__ == '__main__':
    # Test composed candidates
    composed_trees = test_composed_candidates()
    
    # Test high-value extraction
    high_value_seeds = test_high_value_extraction()
    
    # Test additive extraction
    additive_candidates = test_additive_extraction()
    
    # Test ratio construction
    pair_ratio_candidates = test_ratio_construction(high_value_seeds, additive_candidates)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Composed trees: {len(composed_trees)}")
    print(f"  High-value seeds: {len(high_value_seeds)}")
    print(f"  Additive candidates: {len(additive_candidates)}")
    print(f"  Ratio candidates: {len(pair_ratio_candidates)}")
    print("=" * 60)
