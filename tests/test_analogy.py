"""
Unit tests for analogy tasks.
"""

import pytest
from unittest.mock import Mock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.tasks.analogy import (
    solve_analogy,
    simple_analogy,
    compare_analogies,
    batch_analogies,
    evaluate_analogy,
    create_analogy_string
)


@pytest.fixture
def mock_embedding():
    """Create a mock embedding for testing."""
    embedding = Mock()
    embedding.get_model_name.return_value = "MockEmbedding"
    embedding.contains.return_value = True
    embedding.analogy.return_value = [
        ("queen", 0.85),
        ("monarch", 0.72),
        ("princess", 0.68)
    ]
    return embedding

def test_solve_analogy(mock_embedding):
    """Test solving analogy."""
    result = solve_analogy(
        mock_embedding,
        positive=["king", "woman"],
        negative=["man"],
        topn=3
    )
    
    assert len(result) == 3
    assert result[0] == ("queen", 0.85)
    mock_embedding.analogy.assert_called_once()  

def test_simple_analogy(mock_embedding):
    """Test simple analogy format."""
    result = simple_analogy(
        mock_embedding,
        a="king",
        b="queen",
        c="man",
        topn=5
    )
    
    assert len(result) == 3
    mock_embedding.analogy.assert_called_once()
    
    # Check that it was called with correct parameters
    call_args = mock_embedding.analogy.call_args
    assert "queen" in call_args[1]["positive"]
    assert "man" in call_args[1]["positive"]
    assert "king" in call_args[1]["negative"]


def test_compare_analogies():
    """Test comparing analogies across models."""
    emb1 = Mock()
    emb1.analogy.return_value = [("queen", 0.9)]
    
    emb2 = Mock()
    emb2.analogy.return_value = [("queen", 0.8)]
    
    embeddings = {"model1": emb1, "model2": emb2}
    
    result = compare_analogies(
        positive=["king", "woman"],
        negative=["man"],
        embeddings=embeddings,
        topn=1
    )
    
    assert "model1" in result
    assert "model2" in result
    assert result["model1"][0][0] == "queen" 

def test_batch_analogies(mock_embedding):
    """Test batch analogy solving."""
    analogy_list = [
        {"positive": ["king", "woman"], "negative": ["man"]},
        {"positive": ["paris", "italy"], "negative": ["france"]}
    ]
    
    result = batch_analogies(mock_embedding, analogy_list, topn=3)
    
    assert len(result) == 2
    assert "predictions" in result[0]
    assert len(result[0]["predictions"]) == 3


def test_evaluate_analogy(mock_embedding):
    """Test analogy evaluation."""
    result = evaluate_analogy(
        mock_embedding,
        positive=["king", "woman"],
        negative=["man"],
        expected="queen",
        topn=5
    )
    
    assert result["found"] is True
    assert result["rank"] == 1
    assert result["expected"] == "queen"

def test_evaluate_analogy_not_found(mock_embedding):
    """Test analogy evaluation when expected word not in results."""
    mock_embedding.analogy.return_value = [
        ("monarch", 0.72),
        ("princess", 0.68)
    ]
    
    result = evaluate_analogy(
        mock_embedding,
        positive=["king", "woman"],
        negative=["man"],
        expected="queen",
        topn=5
    )
    
    assert result["found"] is False
    assert result["rank"] is None

def test_create_analogy_string_simple():
    """Test creating analogy string in simple format."""
    result = create_analogy_string(
        positive=["king", "woman"],
        negative=["man"],
        result="queen"
    )
    
    assert "king" in result
    assert "man" in result
    assert "woman" in result
    assert "queen" in result
    assert "=" in result

def test_create_analogy_string_no_result():
    """Test creating analogy string without result."""
    result = create_analogy_string(
        positive=["king", "woman"],
        negative=["man"]
    )
    
    assert "?" in result
    assert "=" in result

def test_create_analogy_string_complex():
    """Test creating analogy string with multiple words."""
    result = create_analogy_string(
        positive=["word1", "word2", "word3"],
        negative=["word4", "word5"]
    )
    
    assert "+" in result or " " in result
    assert "-" in result
