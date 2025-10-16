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