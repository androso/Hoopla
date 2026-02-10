import json
from typing import Any, Dict, List

import numpy as np

from .constants import (
    DATA_PATH,
    METADATA_CHUNK_EMBEDDINGS_PATH,
    MOVIE_CHUNK_EMBEDDINGS_PATH,
    MOVIE_EMBEDDINGS_PATH,
    STOPWORDS_PATH,
)


def load_metadata_chunk_embeddings() -> Dict[str, Any]:
    with open(METADATA_CHUNK_EMBEDDINGS_PATH, "r") as f:
        return json.load(f)


def load_chunk_embeddings():
    with open(MOVIE_CHUNK_EMBEDDINGS_PATH, "rb") as f:
        return np.load(f)


def load_movies() -> List[Dict[str, Any]]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        return data["movies"]


def load_stop_words() -> List[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().split("\n")


def load_movie_embeddings():
    with open(MOVIE_EMBEDDINGS_PATH, "rb") as f:
        return np.load(f)
