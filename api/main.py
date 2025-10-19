from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import yaml
import logging
from pathlib import Path
# import sys
# sys.path.append('..')

from src.embeddings.torchtext_glove_embeddings import TorchTextGloVeEmbedding
from src.embeddings.torchtext_word2vec_embeddings import TorchTextFastTextEmbedding
from src.tasks.similarity import calculate_similarity, get_most_similar, compare_similarities
from src.tasks.analogy import solve_analogy, compare_analogies

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path("config/config.yaml")
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {
        'api': {
            'title': 'Word Embedding Analyzer API',
            'description': 'API for word similarity and analogy tasks',
            'version': '1.0.0'
        }
    }

# Lifespan context manager to load models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: load embedding models on startup and perform optional cleanup on shutdown."""
    logger.info("Loading embedding models...")
    try:
        # Load TorchText GloVe (automatic download)
        if config.get('embeddings', {}).get('torchtext_glove', {}).get('enabled', True):
            try:
                glove_config = config.get('embeddings', {}).get('torchtext_glove', {})
                name = glove_config.get('name', '6B')
                dim = glove_config.get('dim', 100)
                cache = glove_config.get('cache', '.vector_cache')
                
                embeddings['glove'] = TorchTextGloVeEmbedding(name=name, dim=dim, cache=cache)
                logger.info(f"✅ TorchText GloVe {name} {dim}d loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load TorchText GloVe: {e}")
        
        # Load TorchText FastText (automatic download)
        if config.get('embeddings', {}).get('torchtext_fasttext', {}).get('enabled', True):
            try:
                fasttext_config = config.get('embeddings', {}).get('torchtext_fasttext', {})
                language = fasttext_config.get('language', 'en')
                cache = fasttext_config.get('cache', '.vector_cache')
                
                embeddings['fasttext'] = TorchTextFastTextEmbedding(language=language, cache=cache)
                logger.info(f"✅ TorchText FastText {language} loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load TorchText FastText: {e}")
        
        # Legacy: Try to load local Word2Vec
        if config.get('embeddings', {}).get('word2vec', {}).get('enabled', False):
            try:
                w2v_path = config.get('embeddings', {}).get('word2vec', {}).get('path', 'models/word2vec.model')
                binary = config.get('embeddings', {}).get('word2vec', {}).get('binary', True)
                embeddings['word2vec'] = TorchTextFastTextEmbedding(w2v_path, binary=binary)
                logger.info("✅ Word2Vec model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load Word2Vec model: {e}")
        
        # Legacy: Try to load local GloVe
        if config.get('embeddings', {}).get('glove', {}).get('enabled', False):
            try:
                glove_path = config.get('embeddings', {}).get('glove', {}).get('path', 'models/glove.6B.100d.txt')
                embeddings['glove_local'] = TorchTextGloVeEmbedding(glove_path)
                logger.info("✅ Local GloVe model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load local GloVe model: {e}")
        
        if not embeddings:
            logger.error("❌ No embedding models could be loaded!")
            logger.info("TorchText models will be downloaded automatically on first use.")
    except Exception as e:
        logger.error(f"Error during model loading: {e}")

    # yield control to let the app run
    yield


# Initialize FastAPI app with lifespan
app = FastAPI(
    title=config['api']['title'],
    description=config['api']['description'],
    version=config['api']['version'],
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for embeddings
embeddings: Dict[str, any] = {}

# Pydantic models for request/response
class SimilarityRequest(BaseModel):
    word1: str = Field(..., description="First word")
    word2: str = Field(..., description="Second word")
    model_type: str = Field("word2vec", description="Model type: 'word2vec' or 'glove'")


class SimilarityResponse(BaseModel):
    word1: str
    word2: str
    similarity: float
    model: str


class MostSimilarRequest(BaseModel):
    word: str = Field(..., description="Target word")
    topn: int = Field(10, description="Number of similar words", ge=1, le=100)
    model_type: str = Field("word2vec", description="Model type: 'word2vec' or 'glove'")

class MostSimilarResponse(BaseModel):
    word: str
    similar_words: List[Dict[str, float]]
    model: str

class AnalogyRequest(BaseModel):
    positive: List[str] = Field(..., description="Positive words")
    negative: List[str] = Field(..., description="Negative words")
    topn: int = Field(5, description="Number of results", ge=1, le=50)
    model_type: str = Field("word2vec", description="Model type: 'word2vec' or 'glove'")

class AnalogyResponse(BaseModel):
    positive: List[str]
    negative: List[str]
    results: List[Dict[str, float]]
    model: str


class CompareRequest(BaseModel):
    word1: str = Field(..., description="First word")
    word2: str = Field(..., description="Second word")

class CompareResponse(BaseModel):
    word1: str
    word2: str
    comparisons: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]


def get_embedding(model_type: str):
    """Get embedding model by type."""
    model_type = model_type.lower()
    if model_type not in embeddings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_type}' not loaded. Available models: {list(embeddings.keys())}"
        )
    return embeddings[model_type]

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Word Embedding Analyzer API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(embeddings.keys())
    }


@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(request: SimilarityRequest):
    """
    Calculate similarity between two words.
    """
    try:
        embedding = get_embedding(request.model_type)
        
        if not embedding.contains(request.word1):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Word '{request.word1}' not in vocabulary"
            )
        
        if not embedding.contains(request.word2):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Word '{request.word2}' not in vocabulary"
            )
        
        score = calculate_similarity(embedding, request.word1, request.word2)
        
        return {
            "word1": request.word1,
            "word2": request.word2,
            "similarity": score,
            "model": embedding.get_model_name()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
@app.post("/most-similar", response_model=MostSimilarResponse)
async def most_similar(request: MostSimilarRequest):
    """
    Find most similar words to a given word.
    """
    try:
        embedding = get_embedding(request.model_type)
        
        if not embedding.contains(request.word):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Word '{request.word}' not in vocabulary"
            )
        
        results = get_most_similar(embedding, request.word, topn=request.topn)
        
        similar_words = [{"word": word, "score": float(score)} for word, score in results]
        
        return {
            "word": request.word,
            "similar_words": similar_words,
            "model": embedding.get_model_name()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar words: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
@app.post("/analogy", response_model=AnalogyResponse)
async def analogy(request: AnalogyRequest):
    """
    Solve word analogy using vector arithmetic.
    Example: king - man + woman = queen
    """
    try:
        embedding = get_embedding(request.model_type)
        
        # Check if all words are in vocabulary
        all_words = request.positive + request.negative
        for word in all_words:
            if not embedding.contains(word):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Word '{word}' not in vocabulary"
                )
        
        results = solve_analogy(embedding, request.positive, request.negative, topn=request.topn)
        
        analogy_results = [{"word": word, "score": float(score)} for word, score in results]
        
        return {
            "positive": request.positive,
            "negative": request.negative,
            "results": analogy_results,
            "model": embedding.get_model_name()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error solving analogy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """
    Compare similarity scores across all available models.
    """
    try:
        if not embeddings:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No embedding models are loaded"
            )
        
        results = compare_similarities(request.word1, request.word2, embeddings)
        
        return {
            "word1": request.word1,
            "word2": request.word2,
            "comparisons": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/models")
async def list_models():
    """List available embedding models."""
    return {
        "available_models": list(embeddings.keys()),
        "model_info": {
            name: {
                "vocabulary_size": emb.get_vocabulary_size(),
                "vector_size": emb.vector_size
            }
            for name, emb in embeddings.items()
        }
    }


@app.get("/vocabulary/{model_type}")
async def check_vocabulary(model_type: str, word: str):
    """Check if a word is in the vocabulary."""
    try:
        embedding = get_embedding(model_type)
        exists = embedding.contains(word)
        
        return {
            "word": word,
            "model": model_type,
            "in_vocabulary": exists
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
