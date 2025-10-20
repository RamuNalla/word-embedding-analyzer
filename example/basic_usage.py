"""
Basic usage examples for Word Embedding Analyzer with TorchText.

This script demonstrates the core functionality with automatic model downloading.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings.torchtext_glove_embeddings import TorchTextGloVeEmbedding
from src.tasks.similarity import calculate_similarity, get_most_similar
from src.tasks.analogy import solve_analogy


def example_1_load_model():
    """Example 1: Load a model (downloads automatically)."""
    print("\n" + "="*60)
    print("Example 1: Loading GloVe Model")
    print("="*60)
    
    # Load GloVe 6B 100d (downloads to .vector_cache/ on first use)
    print("\nLoading GloVe 6B 100d...")
    print("(This will download ~330MB on first run)")
    
    glove = TorchTextGloVeEmbedding(name='6B', dim=100)
    
    print(f"\n✓ Model loaded successfully!")
    print(f"  - Vocabulary size: {glove.get_vocabulary_size():,} words")
    print(f"  - Vector dimension: {glove.vector_size}")
    print(f"  - Model name: {glove.get_model_name()}")
    
    return glove


def example_2_word_similarity(glove):
    """Example 2: Calculate word similarity."""
    print("\n" + "="*60)
    print("Example 2: Word Similarity")
    print("="*60)
    
    # Test various word pairs
    word_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("dog", "cat"),
        ("computer", "laptop"),
        ("car", "automobile"),
        ("good", "bad")
    ]
    
    print("\nCalculating similarities:")
    print(f"{'Word Pair':<25} {'Similarity':<12} {'Interpretation'}")
    print("-" * 60)
    
    for word1, word2 in word_pairs:
        similarity = calculate_similarity(glove, word1, word2)
        
        # Interpret similarity
        if similarity > 0.7:
            interp = "Very similar"
        elif similarity > 0.5:
            interp = "Similar"
        elif similarity > 0.3:
            interp = "Somewhat related"
        else:
            interp = "Not very related"
        
        print(f"{word1}-{word2:<20} {similarity:.4f}        {interp}")


def example_3_most_similar(glove):
    """Example 3: Find most similar words."""
    print("\n" + "="*60)
    print("Example 3: Most Similar Words")
    print("="*60)
    
    test_words = ["king", "computer", "happy"]
    
    for word in test_words:
        print(f"\nMost similar to '{word}':")
        similar = get_most_similar(glove, word, topn=5)
        
        for i, (similar_word, score) in enumerate(similar, 1):
            print(f"  {i}. {similar_word:<15} (similarity: {score:.4f})")


def example_4_analogies(glove):
    """Example 4: Solve word analogies."""
    print("\n" + "="*60)
    print("Example 4: Word Analogies")
    print("="*60)
    
    analogies = [
        {
            "description": "king - man + woman = ?",
            "positive": ["king", "woman"],
            "negative": ["man"],
            "expected": "queen"
        },
        {
            "description": "paris - france + italy = ?",
            "positive": ["paris", "italy"],
            "negative": ["france"],
            "expected": "rome"
        },
        {
            "description": "good - bad + ugly = ?",
            "positive": ["good", "ugly"],
            "negative": ["bad"],
            "expected": "beautiful"
        }
    ]
    
    for analogy in analogies:
        print(f"\n{analogy['description']}")
        print(f"Expected: {analogy['expected']}")
        
        results = solve_analogy(
            glove,
            analogy['positive'],
            analogy['negative'],
            topn=5
        )
        
        print("Results:")
        for i, (word, score) in enumerate(results, 1):
            marker = "✓" if word.lower() == analogy['expected'].lower() else " "
            print(f"  {marker} {i}. {word:<15} (score: {score:.4f})")


def example_5_vocabulary_check(glove):
    """Example 5: Check vocabulary."""
    print("\n" + "="*60)
    print("Example 5: Vocabulary Check")
    print("="*60)
    
    test_words = [
        "computer",
        "smartphone",  # might not be in older embeddings
        "covid",       # definitely not in pre-2020 embeddings
        "internet",
        "cryptocurrency"
    ]
    
    print("\nChecking if words are in vocabulary:")
    for word in test_words:
        in_vocab = glove.contains(word)
        status = "✓ Found" if in_vocab else "✗ Not found"
        print(f"  {word:<20} {status}")
        
        if in_vocab:
            # Show a similar word
            similar = glove.most_similar(word, topn=1)
            if similar:
                print(f"    → Most similar: {similar[0][0]}")


def example_6_semantic_relationships(glove):
    """Example 6: Explore semantic relationships."""
    print("\n" + "="*60)
    print("Example 6: Semantic Relationships")
    print("="*60)
    
    # Country-Capital relationships
    print("\nCountry-Capital relationships:")
    countries_capitals = [
        ("france", "paris"),
        ("italy", "rome"),
        ("japan", "tokyo"),
        ("germany", "berlin")
    ]
    
    for country, capital in countries_capitals:
        similarity = calculate_similarity(glove, country, capital)
        print(f"  {country.capitalize()}-{capital.capitalize()}: {similarity:.4f}")
    
    # Gender relationships
    print("\nGender pair relationships:")
    gender_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("father", "mother")
    ]
    
    for male, female in gender_pairs:
        similarity = calculate_similarity(glove, male, female)
        print(f"  {male}-{female}: {similarity:.4f}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Word Embedding Analyzer - Basic Usage Examples")
    print("="*60)
    print("\nThis script demonstrates core functionality.")
    print("Models download automatically on first use.\n")
    
    try:
        # Example 1: Load model
        glove = example_1_load_model()
        
        # Example 2: Word similarity
        example_2_word_similarity(glove)
        
        # Example 3: Most similar words
        example_3_most_similar(glove)
        
        # Example 4: Word analogies
        example_4_analogies(glove)
        
        # Example 5: Vocabulary check
        example_5_vocabulary_check(glove)
        
        # Example 6: Semantic relationships
        example_6_semantic_relationships(glove)
        
        # Success message
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
        print("\nNext steps:")
        print("  1. Try the API: uvicorn api.main:app --reload")
        print("  2. Try Streamlit: streamlit run app/streamlit_app.py")
        print("  3. Read TORCHTEXT_GUIDE.md for more options")
        print("  4. Explore notebooks/exploration.ipynb")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Make sure requirements are installed: pip install -r requirements.txt")
        print("  2. Check internet connection (for model download)")
        print("  3. Try running: python scripts/test_torchtext.py")


if __name__ == "__main__":
    main()