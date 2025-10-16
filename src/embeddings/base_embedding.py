from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

"""
Base class for word embedding models.
Defines the interface that all embedding implementations must follow.
"""

class BaseEmbedding(ABC):
    """Abstract base class for word embedding models."""
    
    def __init__(self, model_path: str):
        """
        Initialize the embedding model.
        Args:
            model_path: Path to the pre-trained embedding file
        """
        self.model_path = model_path
        self.model = None
        self.vector_size = None

    @abstractmethod
    def load(self) -> None:
        """Load the pre-trained embedding model."""
        pass

    @abstractmethod
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get the embedding vector for a word.
        Args:
            word: The word to get the vector for
            
        Returns:
            Embedding vector or None if word not in vocabulary
        """
        pass

    @abstractmethod
    def similarity(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words.           
        Returns:
            Similarity score between -1 and 1
        """
        pass

    @abstractmethod
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words to the given word.
        Args:
            word: Target word
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        pass

    @abstractmethod
    def analogy(self, positive: List[str], negative: List[str], 
                topn: int = 1) -> List[Tuple[str, float]]:
        """
        Solve word analogies: positive[0] - negative[0] + positive[1] = ?
        
        Args:
            positive: List of positive words
            negative: List of negative words
            topn: Number of results to return
            
        Returns:
            List of (word, score) tuples
        """
        pass

    @abstractmethod
    def contains(self, word: str) -> bool:
        """
        Check if word is in the vocabulary.
        
        Args:
            word: Word to check
        Returns:
            True if word is in vocabulary, False otherwise
        """
        pass

    @abstractmethod
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            Number of words in vocabulary
        """
        pass

    def get_model_name(self) -> str:
        """
        Get the name of the embedding model.
        
        Returns:
            Model name
        """
        return self.__class__.__name__.replace('Embedding', '')
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))