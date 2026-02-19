from .search.loaders import load_movies
from .hybrid_search import HybridSearch
from gemini import generate_answer, generate_summary
from .search.constants import DEFAULT_K

def rag_command(query: str, k=DEFAULT_K):
    movies = load_movies()
    search = HybridSearch(movies)

    results = search.rrf_search(query, k, 5)
    if not results:
        return {
            "query": query,
            "results": [],
            "errors": "No results found"
        }

    answer = generate_answer(query, results)

    return {
        "query": query,
        "answer": answer,
        "results": results
    } 


def summarize_command(query: str, limit: int):
    movies = load_movies()
    search = HybridSearch(movies)

    res = search.rrf_search(query, k=DEFAULT_K, limit=limit * 500)
    if not res:
        return {
            "query": query,
            "search_results": [],
            "errors": "No results found"
        }

    summary = generate_summary(query, res, limit)

    return {
        "query": query,
        "summary": summary,
        "search_results": res[:limit]
    }