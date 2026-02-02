import os
from typing import Any, Dict, List
import json
import numpy as np
DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1 
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

BM25_K1 = 1.5
BM25_B = 0.75
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
MOVIE_CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
METADATA_CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

def cosine_similarity(vec1, vec2) -> float:
    """Calculate cosine similarity between two vectors.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))

def load_metadata_chunk_embeddings():
    with open(METADATA_CHUNK_EMBEDDINGS_PATH, "r") as f:
        data = json.load(f)
        return data

def load_chunk_embeddings():
    with open(MOVIE_CHUNK_EMBEDDINGS_PATH, "rb") as f:
        return np.load(f)

def load_movies() -> List[Dict[str, Any]]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        return data["movies"]
        
def load_stop_words() -> List[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stop_words = f.read().split("\n")
    return stop_words
    
def load_movie_embeddings():
    with open(MOVIE_EMBEDDINGS_PATH, "rb") as f:
        return np.load(f)
