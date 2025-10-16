from typing import List, Tuple, Optional
import numpy as np
import logging
import torch
from torchtext.vocab import FastText as TorchTextFastText

from .base_embedding import BaseEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TorchTextFastTextEmbedding(BaseEmbedding):
    """
    FastText embedding model implementation using TorchText.
    FastText is similar to Word2Vec but handles out-of-vocabulary words better.
    """

    def __init__(self, language: str = 'en', cache: str = '.vector_cache'):
        """
        Initialize TorchText FastText embedding.
        
        Args:
            language: Language code ('en', 'simple' for English)
            cache: Directory to cache downloaded vectors
        """
        model_path = f"torchtext_fasttext_{language}"
        super().__init__(model_path)
        self.language = language
        self.cache = cache
        self.load()


    def load(self) -> None:
        """Load the pre-trained FastText embeddings from TorchText."""
        try:
            logger.info(f"Loading FastText {self.language} from TorchText...")
            logger.info("Note: This downloads pre-trained FastText vectors (~1GB)")
            
            # Download and load FastText vectors
            self.model = TorchTextFastText(
                language=self.language,
                cache=self.cache
            )
            
            self.vector_size = self.model.dim
            
            logger.info(f"Loaded {len(self.model.vectors)} word vectors")
            logger.info(f"Vector dimension: {self.vector_size}")
            
        except Exception as e:
            logger.error(f"Error loading TorchText FastText embeddings: {e}")
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
            # FastText can handle OOV words through subword information
            idx = self.model.stoi.get(word.lower())
            if idx is None:
                # For FastText, we could compute subword embeddings
                # but for compatibility, return None
                return None
            
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
        target_norm = target_tensor / torch.norm(target_tensor)
        vectors_norm = self.model.vectors / torch.norm(self.model.vectors, dim=1, keepdim=True)
        
        similarities = torch.matmul(vectors_norm, target_norm)
        
        # Exclude the word itself
        word_idx = self.model.stoi.get(word.lower())
        if word_idx is not None:
            similarities[word_idx] = -float('inf')
        
        top_indices = torch.topk(similarities, k=min(topn, len(similarities))).indices
        
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
        
        Args:
            positive: List of positive words
            negative: List of negative words
            topn: Number of results to return
            
        Returns:
            List of (word, score) tuples
        """
        pos_vecs = [self.get_vector(w) for w in positive]
        neg_vecs = [self.get_vector(w) for w in negative]
        
        if None in pos_vecs or None in neg_vecs:
            logger.warning("One or more words not in vocabulary")
            return []
        
        result_vec = np.sum(pos_vecs, axis=0) - np.sum(neg_vecs, axis=0)
        result_tensor = torch.from_numpy(result_vec)
        
        result_norm = result_tensor / torch.norm(result_tensor)
        vectors_norm = self.model.vectors / torch.norm(self.model.vectors, dim=1, keepdim=True)
        
        similarities = torch.matmul(vectors_norm, result_norm)
        
        # Exclude input words
        exclude_words = set(w.lower() for w in positive + negative)
        for word in exclude_words:
            word_idx = self.model.stoi.get(word)
            if word_idx is not None:
                similarities[word_idx] = -float('inf')
        
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
        return f"FastText-{self.language}"