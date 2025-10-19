"""
Script to train Word2Vec model on custom corpus.
"""

import os
import sys
import logging
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_word2vec(corpus_path: str, 
                   output_path: str,
                   vector_size: int = 300,
                   window: int = 5,
                   min_count: int = 5,
                   workers: int = 4,
                   epochs: int = 5,
                   sg: int = 0):
    """
    Train Word2Vec model.
    
    Args:
        corpus_path: Path to corpus file (one sentence per line)
        output_path: Path to save the trained model
        vector_size: Dimensionality of word vectors
        window: Maximum distance between current and predicted word
        min_count: Ignores words with frequency lower than this
        workers: Number of worker threads
        epochs: Number of training epochs
        sg: Training algorithm: 1 for skip-gram; 0 for CBOW
    """

    logger.info("=" * 60)
    logger.info("Training Word2Vec Model")
    logger.info("=" * 60)
    
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Vector size: {vector_size}")
    logger.info(f"Window: {window}")
    logger.info(f"Min count: {min_count}")
    logger.info(f"Algorithm: {'Skip-gram' if sg else 'CBOW'}")
    logger.info(f"Epochs: {epochs}")

    # Check if corpus exists
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file not found: {corpus_path}")
        return
    
    # Load sentences from corpus
    logger.info("Loading corpus...")
    sentences = LineSentence(corpus_path)

    # Train model
    logger.info("Training model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs
    )

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Saving model to {output_path}...")
    
    # Save in word2vec format (can be loaded by KeyedVectors)
    model.wv.save_word2vec_format(output_path, binary=True)

    # Print model info
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Vocabulary size: {len(model.wv)}")
    logger.info(f"Model saved to: {output_path}")
    
    # Test model
    logger.info("\nTesting model with sample queries...")
    
    test_words = ['king', 'computer', 'good', 'happy']
    for word in test_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=3)
            logger.info(f"{word}: {[w for w, _ in similar]}")


def create_sample_corpus(output_path: str = "data/sample_corpus.txt"):
    """
    Create a sample corpus for demonstration.
    
    Args:
        output_path: Path to save sample corpus
    """
    sample_text = """
    The king and queen lived in a beautiful castle.
    The man and woman walked through the park.
    A boy and girl played with their toys.
    The prince and princess attended the royal ball.
    The father and mother took care of their children.
    Paris is the capital of France.
    Rome is the capital of Italy.
    London is the capital of England.
    Tokyo is the capital of Japan.
    The computer and laptop are electronic devices.
    The cat and dog are popular pets.
    The sun rises in the east and sets in the west.
    Good food makes people happy and healthy.
    The teacher and student discussed the mathematics problem.
    The doctor and nurse worked at the hospital.
    The car and truck drove on the highway.
    The airplane and helicopter flew in the sky.
    The apple and banana are types of fruit.
    The book and magazine were on the table.
    The phone and tablet are modern gadgets.
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(sample_text.strip())
    
    logger.info(f"Sample corpus created at: {output_path}")
    return output_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Word2Vec model')
    parser.add_argument('--corpus', type=str, default=None,
                       help='Path to corpus file (one sentence per line)')
    parser.add_argument('--output', type=str, default='models/word2vec.model',
                       help='Output path for trained model')
    parser.add_argument('--vector-size', type=int, default=100,
                       help='Dimensionality of word vectors')
    parser.add_argument('--window', type=int, default=5,
                       help='Maximum distance between current and predicted word')
    parser.add_argument('--min-count', type=int, default=1,
                       help='Minimum word frequency')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker threads')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--sg', type=int, default=0, choices=[0, 1],
                       help='Training algorithm: 1=skip-gram, 0=CBOW')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create and use sample corpus')
    
    args = parser.parse_args()
    
    # Create sample corpus if requested
    if args.create_sample or args.corpus is None:
        logger.info("Creating sample corpus for demonstration...")
        corpus_path = create_sample_corpus()
        logger.warning("\nNOTE: This is a tiny sample corpus for demonstration only.")
        logger.warning("For production use, provide a large corpus with --corpus flag.\n")
    else:
        corpus_path = args.corpus
    
    # Train model
    train_word2vec(
        corpus_path=corpus_path,
        output_path=args.output,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
        sg=args.sg
    )


if __name__ == "__main__":
    main()
