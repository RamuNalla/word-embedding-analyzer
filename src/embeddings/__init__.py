"""
Embedding models package.
"""

from .base_embedding import BaseEmbedding
from .torchtext_glove_embeddings import TorchTextGloVeEmbedding
from .torchtext_word2vec_embeddings import TorchTextFastTextEmbedding

__all__ = [
    'BaseEmbedding',
    'TorchTextGloVeEmbedding',
    'TorchTextFastTextEmbedding'
]
