import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies, DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA, format_search_results

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha = DEFAULT_ALPHA, limit=DEFAULT_SEARCH_LIMIT):
        keyword_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        results = combine_search_results(keyword_results, semantic_results, alpha)
        return results[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_scores(scores: list[float]):
    if not scores:
        return []
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    normalized_scores = []
    
    for score in scores:
        normalized = (score - min_score) / (max_score - min_score)
        normalized_scores.append(normalized)

    return normalized_scores

def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = DEFAULT_ALPHA
) -> list[dict]:
    bm25_normalized = normalize_scores([r['score'] for r in bm25_results])
    semantic_normalized = normalize_scores([r['score'] for r in semantic_results])
    
    combined_scores: dict[int, dict] = {}
    
    for doc, norm in zip(bm25_results, bm25_normalized):
        doc_id = doc["id"]
        entry = combined_scores.setdefault(doc_id, {
            "id": doc_id,
            "title": doc["title"],
            "document": doc["document"],
            "bm25_score": 0.0,
            "semantic_score": 0.0
        })

        entry["bm25_score"] = max(entry["bm25_score"], norm)
    
    for doc, norm in zip(semantic_results, semantic_normalized):
        doc_id = doc["id"]
        entry = combined_scores.setdefault(doc_id, {
            "id": doc_id,
            "title": doc["title"],
            "document": doc["document"],
            "bm25_score": 0.0,
            "semantic_score": 0.0
        })

        entry["semantic_score"] = max(entry["semantic_score"], norm)
    
    hybrid_results = []
    for entry in combined_scores.values():
        score = hybrid_score(entry["bm25_score"], entry["semantic_score"], alpha)
        result = format_search_results(
            doc_id=entry["id"],
            title=entry["title"],
            document=entry["document"],
            score=score,
            bm25_score=entry["bm25_score"],
            semantic_score=entry["semantic_score"]
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)

def hybrid_score(bm25_score, semantic_score, alpha = DEFAULT_ALPHA):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def hybrid_search_command(query: str, alpha:float = DEFAULT_ALPHA, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    search = HybridSearch(movies)
    results = search.weighted_search(query, alpha, limit)
    return results