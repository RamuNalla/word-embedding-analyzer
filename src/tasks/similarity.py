
"""
Word similarity calculation tasks.
"""

from typing import List, Tuple, Dict
import numpy as np
import logging

from ..embeddings.base_embedding import BaseEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

