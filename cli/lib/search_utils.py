import os
from typing import Any, Dict, List
import json
import numpy as np
DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3

BM25_K1 = 1.5
BM25_B = 0.75
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
MOVIE_CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
METADATA_CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

def dot(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have the same length. Got {len(vec1)} and {len(vec2)}.") 
    total = 0.0
    for i in range(len(vec1)):
        total += vec1[i] * vec2[i]

def euclidean_norm(vec):
    total = 0.0
    for x in vec:
        total += x**2

    return total**0.5

def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have the same length. Got {len(vec1)} and {len(vec2)}.")
    
    dot_product = dot(vec1, vec2)
    magnitude1 = euclidean_norm(vec1)
    magnitude2 = euclidean_norm(vec2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

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