"""
Test script for TorchText embeddings.
Quick verification that TorchText models work correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings.torchtext_glove_embedding import TorchTextGloVeEmbedding
from src.embeddings.torchtext_word2vec_embedding import TorchTextFastTextEmbedding
from src.tasks.similarity import calculate_similarity
from src.tasks.analogy import solve_analogy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_glove():
    """Test TorchText GloVe embedding."""
    print("\n" + "="*60)
    print("Testing TorchText GloVe")
    print("="*60)
    
    try:
        # Load GloVe (will download if not cached)
        logger.info("Loading GloVe 6B 100d...")
        glove = TorchTextGloVeEmbedding(name='6B', dim=100)
        
        # Test vocabulary
        print(f"\n✓ Vocabulary size: {glove.get_vocabulary_size():,}")
        print(f"✓ Vector dimension: {glove.vector_size}")
        
        # Test word presence
        test_words = ['king', 'queen', 'computer', 'python']
        print(f"\n✓ Testing vocabulary:")
        for word in test_words:
            exists = glove.contains(word)
            print(f"  - '{word}': {'✓ Found' if exists else '✗ Not found'}")
        
        # Test similarity
        print(f"\n✓ Testing similarity:")
        word_pairs = [
            ('king', 'queen'),
            ('man', 'woman'),
            ('computer', 'laptop')
        ]
        for w1, w2 in word_pairs:
            sim = calculate_similarity(glove, w1, w2)
            print(f"  - {w1} <-> {w2}: {sim:.4f}")
        
        # Test most similar
        print(f"\n✓ Testing most similar words:")
        similar = glove.most_similar('king', topn=5)
        print(f"  Most similar to 'king':")
        for word, score in similar:
            print(f"    - {word}: {score:.4f}")
        
        # Test analogy
        print(f"\n✓ Testing analogy:")
        results = solve_analogy(glove, ['king', 'woman'], ['man'], topn=3)
        print(f"  king - man + woman =")
        for word, score in results:
            print(f"    - {word}: {score:.4f}")
        
        print("\n✅ GloVe tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ GloVe test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fasttext():
    """Test TorchText FastText embedding."""
    print("\n" + "="*60)
    print("Testing TorchText FastText")
    print("="*60)
    
    try:
        # Load FastText (will download if not cached - this is large!)
        logger.info("Loading FastText English...")
        logger.warning("Warning: FastText model is ~7GB. This may take a while...")
        
        response = input("\nDownload FastText (~7GB)? This will take time. (y/n): ")
        if response.lower() != 'y':
            print("Skipping FastText test.")
            return None
        
        fasttext = TorchTextFastTextEmbedding(language='en')
        
        # Test vocabulary
        print(f"\n✓ Vocabulary size: {fasttext.get_vocabulary_size():,}")
        print(f"✓ Vector dimension: {fasttext.vector_size}")
        
        # Test similarity
        print(f"\n✓ Testing similarity:")
        sim = calculate_similarity(fasttext, 'king', 'queen')
        print(f"  - king <-> queen: {sim:.4f}")
        
        # Test most similar
        print(f"\n✓ Testing most similar words:")
        similar = fasttext.most_similar('computer', topn=5)
        print(f"  Most similar to 'computer':")
        for word, score in similar:
            print(f"    - {word}: {score:.4f}")
        
        print("\n✅ FastText tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ FastText test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_models():
    """Compare GloVe and FastText on same tasks."""
    print("\n" + "="*60)
    print("Comparing Models")
    print("="*60)
    
    try:
        # Load both models
        print("\nLoading models...")
        glove = TorchTextGloVeEmbedding(name='6B', dim=100)
        print("✓ GloVe loaded")
        
        # Test on same word pairs
        test_pairs = [
            ('king', 'queen'),
            ('man', 'woman'),
            ('paris', 'france'),
            ('computer', 'laptop')
        ]
        
        print("\n" + "-"*60)
        print(f"{'Word Pair':<25} {'GloVe':<15}")
        print("-"*60)
        
        for w1, w2 in test_pairs:
            glove_sim = calculate_similarity(glove, w1, w2)
            print(f"{w1}-{w2:<20} {glove_sim:.4f}")
        
        print("\n✅ Comparison complete!")
        
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TorchText Embeddings Test Suite")
    print("="*60)
    
    print("\nThis script tests TorchText embedding functionality.")
    print("Models will be downloaded automatically if not cached.")
    print("\nCache location: .vector_cache/")
    
    # Test GloVe (smaller, faster)
    glove_success = test_glove()
    
    # Test FastText (optional, very large)
    fasttext_success = test_fasttext()
    
    # Compare if GloVe worked
    if glove_success:
        compare_models()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"GloVe:    {'✅ PASSED' if glove_success else '❌ FAILED'}")
    if fasttext_success is not None:
        print(f"FastText: {'✅ PASSED' if fasttext_success else '❌ FAILED'}")
    else:
        print(f"FastText: ⊘ SKIPPED")
    print("="*60)
    
    if glove_success:
        print("\n✅ Core functionality working!")
        print("\nNext steps:")
        print("  1. Run API: uvicorn api.main:app --reload")
        print("  2. Run Streamlit: streamlit run app/streamlit_app.py")
        print("  3. See TORCHTEXT_GUIDE.md for more options")
    else:
        print("\n❌ Some tests failed. Check error messages above.")


if __name__ == "__main__":
    main()