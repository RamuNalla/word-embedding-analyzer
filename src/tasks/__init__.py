"""
Task implementations package.
"""

from .similarity import (
    calculate_similarity,
    get_most_similar,
    compare_similarities,
    batch_similarity,
    similarity_matrix,
    find_odd_one_out,
    semantic_distance
)

from .analogy import (
    solve_analogy,
    simple_analogy,
    compare_analogies,
    batch_analogies,
    evaluate_analogy,
    create_analogy_string
)

__all__ = [
    # Similarity
    'calculate_similarity',
    'get_most_similar',
    'compare_similarities',
    'batch_similarity',
    'similarity_matrix',
    'find_odd_one_out',
    'semantic_distance',
    # Analogy
    'solve_analogy',
    'simple_analogy',
    'compare_analogies',
    'batch_analogies',
    'evaluate_analogy',
    'create_analogy_string'
]
