from .search.loaders import load_movies
from .hybrid_search import HybridSearch
from gemini import generate_answer
from .search.constants import DEFAULT_K

def rag_command(query: str, docs, k=DEFAULT_K, ):
    movies = load_movies()
    search = HybridSearch(movies)

    results = search.rrf_search(query, k, 5)
    if not results:
        return {
            "query": query,
            "results": [],
            "errors": "No results found"
        }

    answer = generate_answer(query, docs)

    return {
        "query": query,
        "k": k,
        "answer": answer
    } 