"""
GloVe embedding implementation using TorchText.
"""

from typing import List, Tuple, Optional
import numpy as np
import logging
import torch
from torchtext.vocab import GloVe as TorchTextGloVe

from .base_embedding import BaseEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TorchTextGloVeEmbedding(BaseEmbedding):
    """GloVe embedding model implementation using TorchText."""
    
    def __init__(self, name: str = '6B', dim: int = 100, cache: str = '.vector_cache'):
        """
        Initialize TorchText GloVe embedding.
        
        Args:
            name: GloVe corpus name ('6B', '42B', '840B', 'twitter.27B')
            dim: Embedding dimension (50, 100, 200, 300 for 6B)
            cache: Directory to cache downloaded vectors
        """
        # Create a path string for compatibility with base class
        model_path = f"torchtext_glove_{name}_{dim}d"
        super().__init__(model_path)
        self.name = name
        self.dim = dim
        self.cache = cache
        self.load()

    def load(self) -> None:
        """Load the pre-trained GloVe embeddings from TorchText."""
        try:
            logger.info(f"Loading GloVe {self.name} {self.dim}d from TorchText...")
            
            # Download and load GloVe vectors
            self.model = TorchTextGloVe(
                name=self.name,
                dim=self.dim,
                cache=self.cache
            )
            
            self.vector_size = self.dim
            
            logger.info(f"Loaded {len(self.model.vectors)} word vectors")
            logger.info(f"Vector dimension: {self.vector_size}")
            
        except Exception as e:
            logger.error(f"Error loading TorchText GloVe embeddings: {e}")
            raise

    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get the embedding vector for a word.
        
        Args:
            word: The word to get the vector for
            
        Returns:
            Embedding vector or None if word not in vocabulary
        """
        try:
            # Get the index of the word
            idx = self.model.stoi.get(word.lower())
            if idx is None:
                return None
            
            # Get the vector and convert to numpy
            vec = self.model.vectors[idx].numpy()
            return vec
        except Exception:
            return None
        
    def similarity(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between -1 and 1
        """
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        if vec1 is None or vec2 is None:
            logger.warning(f"One or both words not in vocabulary: {word1}, {word2}")
            return 0.0
        
        return self.cosine_similarity(vec1, vec2)
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words to the given word.
        
        Args:
            word: Target word
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        target_vec = self.get_vector(word)
        
        if target_vec is None:
            logger.warning(f"Word '{word}' not in vocabulary")
            return []
        
        # Convert to torch tensor
        target_tensor = torch.from_numpy(target_vec)
        
        # Calculate cosine similarity with all vectors
        # Normalize vectors
        target_norm = target_tensor / torch.norm(target_tensor)
        vectors_norm = self.model.vectors / torch.norm(self.model.vectors, dim=1, keepdim=True)
        
        # Compute similarities
        similarities = torch.matmul(vectors_norm, target_norm)
        
        # Get top k indices (excluding the word itself)
        word_idx = self.model.stoi.get(word.lower())
        if word_idx is not None:
            similarities[word_idx] = -float('inf')
        
        top_indices = torch.topk(similarities, k=min(topn, len(similarities))).indices
        
        # Convert to list of (word, score) tuples
        results = []
        for idx in top_indices:
            word_str = self.model.itos[idx.item()]
            score = similarities[idx].item()
            results.append((word_str, float(score)))
        
        return results
    

    def analogy(self, positive: List[str], negative: List[str], 
                topn: int = 1) -> List[Tuple[str, float]]:
        """
        Solve word analogies using vector arithmetic.
        
        Example: king - man + woman = queen
        positive = ['king', 'woman'], negative = ['man']
        
        Args:
            positive: List of positive words
            negative: List of negative words
            topn: Number of results to return
            
        Returns:
            List of (word, score) tuples
        """
        # Get vectors for all words
        pos_vecs = [self.get_vector(w) for w in positive]
        neg_vecs = [self.get_vector(w) for w in negative]
        
        # Check if all words are in vocabulary
        if None in pos_vecs or None in neg_vecs:
            logger.warning("One or more words not in vocabulary")
            return []
        
        # Calculate result vector: sum(positive) - sum(negative)
        result_vec = np.sum(pos_vecs, axis=0) - np.sum(neg_vecs, axis=0)
        result_tensor = torch.from_numpy(result_vec)
        
        # Normalize
        result_norm = result_tensor / torch.norm(result_tensor)
        vectors_norm = self.model.vectors / torch.norm(self.model.vectors, dim=1, keepdim=True)
        
        # Compute similarities
        similarities = torch.matmul(vectors_norm, result_norm)
        
        # Exclude input words
        exclude_words = set(w.lower() for w in positive + negative)
        for word in exclude_words:
            word_idx = self.model.stoi.get(word)
            if word_idx is not None:
                similarities[word_idx] = -float('inf')
        
        # Get top k
        top_indices = torch.topk(similarities, k=min(topn, len(similarities))).indices
        
        results = []
        for idx in top_indices:
            word_str = self.model.itos[idx.item()]
            score = similarities[idx].item()
            results.append((word_str, float(score)))
        
        return results
    
    def contains(self, word: str) -> bool:
        """
        Check if word is in the vocabulary.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is in vocabulary, False otherwise
        """
        return word.lower() in self.model.stoi
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            Number of words in vocabulary
        """
        return len(self.model.vectors)
    
    def get_words(self, limit: Optional[int] = None) -> List[str]:
        """
        Get list of words in vocabulary.
        
        Args:
            limit: Maximum number of words to return
            
        Returns:
            List of words
        """
        words = self.model.itos
        if limit:
            return words[:limit]
        return words
    
    def get_model_name(self) -> str:
        """
        Get the name of the embedding model.
        
        Returns:
            Model name
        """
        return f"GloVe-{self.name}-{self.dim}d"
