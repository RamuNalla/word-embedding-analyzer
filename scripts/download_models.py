"""
Script to download pre-trained embedding models.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: str, description: str = "Downloading"):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        description: Description for progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    logger.info(f"Downloaded: {destination}")


def extract_zip(zip_path: str, extract_to: str):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    logger.info(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info(f"Extracted to {extract_to}")


def download_glove(models_dir: str = "models"):
    """
    Download GloVe embeddings.
    
    Args:
        models_dir: Directory to save models
    """
    logger.info("=" * 50)
    logger.info("Downloading GloVe Embeddings")
    logger.info("=" * 50)
    
    # GloVe 6B (Wikipedia 2014 + Gigaword 5)
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    glove_zip = os.path.join(models_dir, "glove.6B.zip")
    
    if not os.path.exists(os.path.join(models_dir, "glove.6B.100d.txt")):
        download_file(glove_url, glove_zip, "GloVe 6B")
        extract_zip(glove_zip, models_dir)
        
        # Remove zip file to save space
        os.remove(glove_zip)
        logger.info("GloVe download complete!")
    else:
        logger.info("GloVe embeddings already exist. Skipping download.")


def download_word2vec_instructions():
    """
    Provide instructions for downloading Word2Vec.
    """
    logger.info("=" * 50)
    logger.info("Word2Vec Model Instructions")
    logger.info("=" * 50)
    
    logger.info("\nWord2Vec models are large (~3.5GB). You have two options:\n")
    
    logger.info("Option 1: Download pre-trained Google News model")
    logger.info("-" * 50)
    logger.info("1. Visit: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM")
    logger.info("2. Download GoogleNews-vectors-negative300.bin.gz")
    logger.info("3. Extract the file to the 'models' directory")
    logger.info("4. Rename to 'word2vec.model' or update config.yaml")
    
    logger.info("\nOption 2: Train your own Word2Vec model")
    logger.info("-" * 50)
    logger.info("Run: python scripts/train_word2vec.py")
    logger.info("This will train a model on a sample corpus (requires corpus data)")


def main():
    """Main function to download models."""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("Word Embedding Model Downloader")
    print("=" * 50 + "\n")
    
    # Download GloVe
    try:
        download_glove(models_dir)
    except Exception as e:
        logger.error(f"Error downloading GloVe: {e}")
    
    # Word2Vec instructions
    print("\n")
    download_word2vec_instructions()
    
    print("\n" + "=" * 50)
    print("Download process complete!")
    print("=" * 50)
    
    # Summary
    print("\nDownloaded files in 'models' directory:")
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(('.txt', '.bin', '.model')):
                file_path = os.path.join(models_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()