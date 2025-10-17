
"""
Word analogy solving tasks.
"""

from typing import List, Tuple, Dict
import logging

from ..embeddings.base_embedding import BaseEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def solve_analogy(embedding: BaseEmbedding, 
                  positive: List[str], 
                  negative: List[str], 
                  topn: int = 5) -> List[Tuple[str, float]]:
    """
    Solve word analogy using vector arithmetic.
    
    Example: king - man + woman = ?
    positive = ['king', 'woman'], negative = ['man']
    
    Args:
        embedding: Embedding model to use
        positive: List of positive words
        negative: List of negative words
        topn: Number of results to return
        
    Returns:
        List of (word, score) tuples
    """
    return embedding.analogy(positive, negative, topn=topn)

def simple_analogy(embedding: BaseEmbedding, 
                   a: str, b: str, c: str, 
                   topn: int = 5) -> List[Tuple[str, float]]:
    """
    Solve simple analogy: a is to b as c is to ?
    
    Example: king is to queen as man is to ?
    a='king', b='queen', c='man' -> woman
    
    Args:
        embedding: Embedding model to use
        a: First word of analogy
        b: Second word of analogy
        c: Third word of analogy
        topn: Number of results to return
        
    Returns:
        List of (word, score) tuples
    """
    # a:b :: c:? -> b - a + c
    return embedding.analogy(positive=[b, c], negative=[a], topn=topn)

def compare_analogies(positive: List[str], 
                     negative: List[str], 
                     embeddings: Dict[str, BaseEmbedding],
                     topn: int = 5) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compare analogy results across different embedding models.
    
    Args:
        positive: List of positive words
        negative: List of negative words
        embeddings: Dictionary mapping model names to embedding objects
        topn: Number of results to return per model
        
    Returns:
        Dictionary mapping model names to their analogy results
    """
    results = {}
    
    for model_name, embedding in embeddings.items():
        try:
            analogy_results = solve_analogy(embedding, positive, negative, topn=topn)
            results[model_name] = analogy_results
        except Exception as e:
            logger.error(f"Error solving analogy with {model_name}: {e}")
            results[model_name] = []
    
    return results

def batch_analogies(embedding: BaseEmbedding, 
                   analogy_list: List[Dict[str, any]],
                   topn: int = 1) -> List[Dict[str, any]]:
    """
    Solve multiple analogies in batch.
    
    Args:
        embedding: Embedding model to use
        analogy_list: List of dictionaries with 'positive' and 'negative' keys
        topn: Number of results per analogy
        
    Returns:
        List of results with predictions
    """
    results = []
    
    for analogy in analogy_list:
        positive = analogy.get('positive', [])
        negative = analogy.get('negative', [])
        
        try:
            predictions = solve_analogy(embedding, positive, negative, topn=topn)
            results.append({
                'positive': positive,
                'negative': negative,
                'predictions': predictions,
                'model': embedding.get_model_name()
            })
        except Exception as e:
            logger.error(f"Error solving analogy {positive} - {negative}: {e}")
            results.append({
                'positive': positive,
                'negative': negative,
                'predictions': [],
                'error': str(e),
                'model': embedding.get_model_name()
            })
    
    return results


def evaluate_analogy(embedding: BaseEmbedding,
                    positive: List[str],
                    negative: List[str],
                    expected: str,
                    topn: int = 5) -> Dict[str, any]:
    """
    Evaluate an analogy by checking if expected answer is in top predictions.
    
    Args:
        embedding: Embedding model to use
        positive: List of positive words
        negative: List of negative words
        expected: Expected answer
        topn: Number of predictions to check
        
    Returns:
        Dictionary with evaluation results
    """
    predictions = solve_analogy(embedding, positive, negative, topn=topn)
    
    # Check if expected answer is in predictions
    found_at = None
    for i, (word, score) in enumerate(predictions, 1):
        if word.lower() == expected.lower():
            found_at = i
            break
    
    return {
        'positive': positive,
        'negative': negative,
        'expected': expected,
        'predictions': predictions,
        'found': found_at is not None,
        'rank': found_at,
        'model': embedding.get_model_name()
    }

def create_analogy_string(positive: List[str], negative: List[str], 
                         result: str = None) -> str:
    """
    Create a human-readable analogy string.
    
    Args:
        positive: List of positive words
        negative: List of negative words
        result: Optional result word
        
    Returns:
        Formatted analogy string
    """
    if len(positive) >= 2 and len(negative) >= 1:
        # Standard format: a - b + c = ?
        pos_str = ' + '.join(positive)
        neg_str = ' - '.join(negative)
        base = f"{positive[0]} - {negative[0]} + {positive[1]}"
        
        if result:
            return f"{base} = {result}"
        return f"{base} = ?"
    else:
        # General format
        pos_str = ' + '.join(positive) if positive else ''
        neg_str = ' - '.join(negative) if negative else ''
        
        parts = []
        if pos_str:
            parts.append(pos_str)
        if neg_str:
            parts.append(f"- {neg_str}")
        
        base = ' '.join(parts)
        if result:
            return f"{base} = {result}"
        return f"{base} = ?"
