import os
from typing import Any, Dict, List
import json

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

def load_movies() -> List[Dict[str, Any]]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        return data["movies"]
        
def load_stop_words() -> List[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stop_words = f.read().split("\n")
    return stop_words