"""
R11: Predictor-guided rational mutation operations for symbolic regression.
Implements subtree crossover, operator mutation, and structural combinations
to guide the search towards rational function structures.
"""

import random
import copy
import numpy as np
from typing import List, Tuple, Optional
from .Node import Node
from .tree_utils import tree_to_str, is_valid_tree


def deep_copy_tree(node: Node) -> Node:
    """Create a deep copy of a tree node."""
    if node is None:
        return None
    
    if node.op in ('x', 'y'):
        return Node(node.op)
    elif node.op.isdigit() or '.' in node.op:
        return Node(node.op)
    elif node.left is None and node.right is None:
        return Node(node.op)
    else:
        new_node = Node(node.op)
        new_node.left = deep_copy_tree(node.left)
        new_node.right = deep_copy_tree(node.right)
        return new_node


def subtree_crossover(parent1: Node, parent2: Node, max_depth: int = 10) -> Optional[Node]:
    """
    Perform subtree crossover between two parent trees.
    Randomly select a subtree from parent1 and replace it with a subtree from parent2.
    """
    if parent1 is None or parent2 is None:
        return None
    
    # Collect all subtrees
    def collect_subtrees(node: Node, depth: int = 0) -> List[Tuple[Node, int, str]]:
        if node is None or depth > max_depth:
            return []
        
        subtrees = [(node, depth, 'root')]
        if node.left is not None:
            subtrees.extend(collect_subtrees(node.left, depth + 1))
        if node.right is not None:
            subtrees.extend(collect_subtrees(node.right, depth + 1))
        return subtrees
    
    subtrees1 = collect_subtrees(parent1)
    subtrees2 = collect_subtrees(parent2)
    
    if not subtrees1 or not subtrees2:
        return None
    
    # Randomly select subtrees
    subtree1, _, _ = random.choice(subtrees1)
    subtree2, _, _ = random.choice(subtrees2)
    
    # Create new tree by replacing subtree1 with subtree2
    def replace_subtree(node: Node, target: Node, replacement: Node, is_first: bool = True) -> Node:
        if node is None:
            return None
        
        if node is target and is_first:
            return deep_copy_tree(replacement)
        
        new_node = Node(node.op)
        new_node.left = replace_subtree(node.left, target, replacement, is_first and node.left is not target)
        new_node.right = replace_subtree(node.right, target, replacement, is_first and node.right is not target)
        return new_node
    
    result = replace_subtree(parent1, subtree1, subtree2, True)
    
    if is_valid_tree(result):
        return result
    return None


def operator_mutation(tree: Node, mutation_rate: float = 0.1) -> Optional[Node]:
    """
    Mutate operators in the tree with a given probability.
    Swap similar operators (e.g., + <-> -, * <-> /).
    """
    if tree is None:
        return None
    
    operator_map = {
        '+': '-',
        '-': '+',
        '*': '/',
        '/': '*',
    }
    
    def mutate_node(node: Node) -> Node:
        if node is None:
            return None
        
        new_node = Node(node.op)
        
        # Mutate operator with probability
        if node.op in operator_map and random.random() < mutation_rate:
            new_node = Node(operator_map[node.op])
        
        new_node.left = mutate_node(node.left)
        new_node.right = mutate_node(node.right)
        return new_node
    
    result = mutate_node(tree)
    
    if is_valid_tree(result):
        return result
    return None


def constant_perturbation(tree: Node, perturbation_scale: float = 0.1) -> Optional[Node]:
    """
    Perturb constant values in the tree by a small random amount.
    """
    if tree is None:
        return None
    
    def perturb_node(node: Node) -> Node:
        if node is None:
            return None
        
        new_node = Node(node.op)
        
        # Perturb constants
        try:
            const_val = float(node.op)
            perturbation = random.gauss(0, perturbation_scale)
            new_val = const_val + perturbation
            new_node = Node(str(new_val))
        except (ValueError, AttributeError):
            pass
        
        new_node.left = perturb_node(node.left)
        new_node.right = perturb_node(node.right)
        return new_node
    
    result = perturb_node(tree)
    
    if is_valid_tree(result):
        return result
    return None


def insert_division_node(tree: Node) -> Optional[Node]:
    """
    Insert a division node at a random position in the tree.
    This helps discover rational function structures.
    """
    if tree is None:
        return None
    
    # Collect all nodes
    def collect_nodes(node: Node) -> List[Node]:
        if node is None:
            return []
        nodes = [node]
        nodes.extend(collect_nodes(node.left))
        nodes.extend(collect_nodes(node.right))
        return nodes
    
    nodes = collect_nodes(tree)
    if not nodes:
        return None
    
    # Randomly select a node to wrap in division
    target_node = random.choice(nodes)
    
    # Create a simple denominator (x + y or similar)
    denominator_options = [
        Node('+', left=Node('x'), right=Node('y')),
        Node('-', left=Node('x'), right=Node('y')),
    ]
    denominator = random.choice(denominator_options)
    
    # Create division node
    div_node = Node('/', left=deep_copy_tree(target_node), right=deep_copy_tree(denominator))
    
    # Replace target with division in the tree
    def replace_node(node: Node, target: Node, replacement: Node, replaced: bool = False) -> Node:
        if node is None or replaced:
            return node
        
        if node is target:
            return deep_copy_tree(replacement)
        
        new_node = Node(node.op)
        new_left = replace_node(node.left, target, replacement, replaced)
        new_right = replace_node(node.right, target, replacement, replaced or new_left is not node.left)
        new_node.left = new_left
        new_node.right = new_right
        return new_node
    
    result = replace_node(tree, target_node, div_node, False)
    
    if is_valid_tree(result):
        return result
    return None


def predictor_guided_mutate(tree: Node, predictor_score: float, 
                           mutation_type: str = 'crossover',
                           partner_tree: Optional[Node] = None) -> Optional[Node]:
    """
    Apply predictor-guided mutation based on the predictor's score.
    Higher scores indicate more promising structures.
    
    Args:
        tree: The tree to mutate
        predictor_score: Score from the residual predictor (higher = better)
        mutation_type: Type of mutation to apply
        partner_tree: Partner tree for crossover (if applicable)
    
    Returns:
        Mutated tree or None if mutation failed
    """
    if tree is None:
        return None
    
    # Adapt mutation strategy based on predictor score
    if predictor_score > 0.8:
        # High score: minimal mutation to preserve good structure
        mutation_rate = 0.05
    elif predictor_score > 0.5:
        # Medium score: moderate mutation
        mutation_rate = 0.15
    else:
        # Low score: aggressive mutation
        mutation_rate = 0.3
    
    # Apply mutation
    if mutation_type == 'crossover' and partner_tree is not None:
        return subtree_crossover(tree, partner_tree)
    elif mutation_type == 'operator':
        return operator_mutation(tree, mutation_rate)
    elif mutation_type == 'constant':
        return constant_perturbation(tree, mutation_rate)
    elif mutation_type == 'division':
        return insert_division_node(tree)
    
    return None
