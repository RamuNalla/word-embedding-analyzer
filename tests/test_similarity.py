
"""
Unit tests for similarity tasks.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.tasks.similarity import (
    calculate_similarity,
    get_most_similar,
    compare_similarities,
    batch_similarity,
    similarity_matrix,
    find_odd_one_out,
    semantic_distance
)

@pytest.fixture
def mock_embedding():
    """Create a mock (fake) embedding for testing."""
    embedding = Mock()
    embedding.get_model_name.return_value = "MockEmbedding"
    embedding.contains.return_value = True
    embedding.similarity.return_value = 0.75
    embedding.most_similar.return_value = [
        ("similar1", 0.9),
        ("similar2", 0.8),
        ("similar3", 0.7)
    ]
    return embedding

def test_calculate_similarity(mock_embedding):          # Any test function that lists mock_embedding as an argument will automatically receive the output of this function. It's our pre-built testing bench.
    """Test similarity calculation."""
    result = calculate_similarity(mock_embedding, "word1", "word2")
    
    assert result == 0.75
    mock_embedding.similarity.assert_called_once_with("word1", "word2")     # Did our function correctly call the similarity method on the embedding object with the right arguments?

def test_get_most_similar(mock_embedding):
    """Test getting most similar words."""
    result = get_most_similar(mock_embedding, "test", topn=3)
    
    assert len(result) == 3
    assert result[0] == ("similar1", 0.9)
    mock_embedding.most_similar.assert_called_once_with("test", topn=3)

def test_compare_similarities():                # No mock embedding
    """Test comparing similarities across models."""
    emb1 = Mock()
    emb1.similarity.return_value = 0.8
    
    emb2 = Mock()
    emb2.similarity.return_value = 0.6
    
    embeddings = {"model1": emb1, "model2": emb2}
    
    result = compare_similarities("word1", "word2", embeddings)
    
    assert result["model1"] == 0.8
    assert result["model2"] == 0.6


def test_batch_similarity(mock_embedding):
    """Test batch similarity calculation."""
    word_pairs = [("king", "queen"), ("man", "woman")]
    
    result = batch_similarity(mock_embedding, word_pairs)
    
    assert len(result) == 2
    assert result[0]["word1"] == "king"
    assert result[0]["word2"] == "queen"
    assert result[0]["similarity"] == 0.75

def test_similarity_matrix(mock_embedding):
    """Test similarity matrix creation."""
    words = ["word1", "word2", "word3"]
    
    result = similarity_matrix(mock_embedding, words)
    
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 1.0)  # Diagonal should be 1
    assert result[0, 1] == result[1, 0]  # Should be symmetric

def test_find_odd_one_out(mock_embedding):
    """Test finding odd word out."""
    mock_embedding.similarity.side_effect = [0.9, 0.8, 0.3, 0.85, 0.4, 0.87]
    
    words = ["king", "queen", "car"]
    odd_word, score = find_odd_one_out(mock_embedding, words)
    
    assert odd_word in words
    assert isinstance(score, float)

def test_semantic_distance(mock_embedding):
    """Test semantic distance calculation."""
    result = semantic_distance(mock_embedding, "word1", "word2")
    
    assert result == 0.25  # 1 - 0.75
    assert 0 <= result <= 2

def test_similarity_with_missing_word():
    """Test similarity with word not in vocabulary."""
    embedding = Mock()
    embedding.contains.side_effect = [False, True]
    embedding.similarity.return_value = 0.0
    
    # Should handle gracefully
    result = calculate_similarity(embedding, "missing", "word")
    assert isinstance(result, float)


