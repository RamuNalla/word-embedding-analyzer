
"""
Word similarity calculation tasks.
"""

from typing import List, Tuple, Dict
import numpy as np
import logging

from ..embeddings.base_embedding import BaseEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_similarity(embedding: BaseEmbedding, word1: str, word2: str) -> float:
    """
    Calculate similarity between two words.
    
    Args:
        embedding: Embedding model to use
        word1: First word
        word2: Second word
        
    Returns:
        Similarity score between -1 and 1
    """
    return embedding.similarity(word1, word2)

def get_most_similar(embedding: BaseEmbedding, word: str, topn: int = 10) -> List[Tuple[str, float]]:
    """
    Get most similar words to a given word.
    
    Args:
        embedding: Embedding model to use
        word: Target word
        topn: Number of similar words to return
        
    Returns:
        List of (word, similarity_score) tuples
    """
    return embedding.most_similar(word, topn=topn)

def compare_similarities(word1: str, word2: str, embeddings: Dict[str, BaseEmbedding]) -> Dict[str, float]:
    """
    Compare similarity scores across different embedding models.
    
    Args:
        word1: First word
        word2: Second word
        embeddings: Dictionary mapping model names to embedding objects
        
    Returns:
        Dictionary mapping model names to similarity scores
    """
    results = {}
    
    for model_name, embedding in embeddings.items():
        try:
            score = calculate_similarity(embedding, word1, word2)
            results[model_name] = score
        except Exception as e:
            logger.error(f"Error calculating similarity with {model_name}: {e}")
            results[model_name] = None
    
    return results


def batch_similarity(embedding: BaseEmbedding, word_pairs: List[Tuple[str, str]]) -> List[Dict[str, any]]:
    """
    Calculate similarities for multiple word pairs.
    
    Args:
        embedding: Embedding model to use
        word_pairs: List of (word1, word2) tuples
        
    Returns:
        List of dictionaries containing word pairs and their similarities
    """
    results = []
    
    for word1, word2 in word_pairs:
        try:
            score = calculate_similarity(embedding, word1, word2)
            results.append({
                'word1': word1,
                'word2': word2,
                'similarity': score,
                'model': embedding.get_model_name()
            })
        except Exception as e:
            logger.error(f"Error calculating similarity for ({word1}, {word2}): {e}")
            results.append({
                'word1': word1,
                'word2': word2,
                'similarity': None,
                'error': str(e),
                'model': embedding.get_model_name()
            })
    
    return results


def similarity_matrix(embedding: BaseEmbedding, words: List[str]) -> np.ndarray:
    """
    Create a similarity matrix for a list of words.
    
    Args:
        embedding: Embedding model to use
        words: List of words
        
    Returns:
        NxN similarity matrix where N is the number of words
    """
    n = len(words)
    matrix = np.zeros((n, n))
    
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i == j:
                matrix[i][j] = 1.0
            elif i < j:
                score = calculate_similarity(embedding, word1, word2)
                matrix[i][j] = score
                matrix[j][i] = score
    
    return matrix

def find_odd_one_out(embedding: BaseEmbedding, words: List[str]) -> Tuple[str, float]:
    """
    Find the word that doesn't belong in a group (least similar to others).
    
    Args:
        embedding: Embedding model to use
        words: List of words
        
    Returns:
        Tuple of (odd_word, average_similarity_score)
    """
    if len(words) < 2:
        raise ValueError("Need at least 2 words")
    
    avg_similarities = {}
    
    for word in words:
        other_words = [w for w in words if w != word]
        similarities = [calculate_similarity(embedding, word, other) for other in other_words]
        avg_similarities[word] = np.mean(similarities)
    
    # Find word with lowest average similarity
    odd_word = min(avg_similarities, key=avg_similarities.get)
    
    return odd_word, avg_similarities[odd_word]

def semantic_distance(embedding: BaseEmbedding, word1: str, word2: str) -> float:
    """
    Calculate semantic distance (inverse of similarity).
    
    Args:
        embedding: Embedding model to use
        word1: First word
        word2: Second word
        
    Returns:
        Distance score (0 = identical, 2 = opposite)
    """
    similarity = calculate_similarity(embedding, word1, word2)
    # Convert similarity [-1, 1] to distance [0, 2]
    return 1 - similarity