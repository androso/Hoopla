import os
from dataclasses import dataclass
from typing import Protocol, Optional
import time
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search.constants import DEFAULT_ALPHA, DEFAULT_K, DEFAULT_SEARCH_LIMIT
from .search.formatting import format_search_results
from .search.loaders import load_movies
from sentence_transformers import CrossEncoder

class FusionStrategy(Protocol):
    def fuse(
        self,
        bm25_results: list[dict],
        semantic_results: list[dict],
        limit: int,
    ) -> list[dict]:
        ...

@dataclass(frozen=True)
class WeightedFusion:
    alpha: float = DEFAULT_ALPHA

    def fuse(
        self,
        bm25_results: list[dict],
        semantic_results: list[dict],
        limit: int,
    ) -> list[dict]:
        return combine_search_results(bm25_results, semantic_results, self.alpha)[:limit]

@dataclass(frozen=True)
class RRFFusion:
    k: int = DEFAULT_K

    def fuse(
        self,
        bm25_results: list[dict],
        semantic_results: list[dict],
        limit: int,
    ) -> list[dict]:

        return combine_rrf_results(bm25_results, semantic_results, self.k)[:limit]


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

    def _retrieve(self, query: str, limit: int) -> tuple[list[dict], list[dict]]:
        if limit <= 0:
            return [], []
        limit *= 500
        bm25_results = self._bm25_search(query, limit)
        semantic_results = self.semantic_search.search_chunks(query, limit)
        return bm25_results, semantic_results

    def search(
        self,
        query: str,
        fusion_strategy: FusionStrategy,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[dict]:
        bm25_results, semantic_results = self._retrieve(query, limit)
        return fusion_strategy.fuse(bm25_results, semantic_results, limit)

    def weighted_search(
        self, query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[dict]:
        return self.search(query, WeightedFusion(alpha), limit)

    def rrf_search(self, query: str, k: int, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        return self.search(query, RRFFusion(k), limit)



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


def combine_rrf_results(
    bm25_results: list[dict], semantic_results: list[dict], k: int = DEFAULT_K
) -> list[dict]:
    doc_ranks_by_id = {}

    for rank, doc in enumerate(bm25_results, 1):
        doc_id = doc["id"]
        doc_ranks_by_id[doc_id] = format_search_results(
            doc_id=doc["id"],
            title=doc["title"],
            document=doc["document"],
            score=0,
            rrf_score=1 / (k + rank),
            semantic_rank=None,
            bm25_rank=rank,
        )

    for rank, doc in enumerate(semantic_results, 1):
        doc_id = doc["id"]
        if doc_id not in doc_ranks_by_id:
            doc_ranks_by_id[doc_id] = format_search_results(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["document"],
                score=0,
                rrf_score=1 / (k + rank),
                semantic_rank=rank,
                bm25_rank=None,
            )
        else:
            doc_ranks_by_id[doc_id] = format_search_results(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["document"],
                score=0,
                rrf_score=doc_ranks_by_id[doc_id]["metadata"]["rrf_score"]
                + (1 / (k + rank)),
                semantic_rank=rank,
                bm25_rank=doc_ranks_by_id[doc_id]["metadata"]["bm25_rank"],
            )

    return sorted(
        doc_ranks_by_id.values(),
        key=lambda doc: doc["metadata"]["rrf_score"],
        reverse=True,
    )


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

def rrf_search_command(query: str, k=DEFAULT_K, enhance: Optional[str] = None, limit = DEFAULT_SEARCH_LIMIT, rerank_method: Optional[str] = None):
    movies = load_movies()
    search = HybridSearch(movies)
    original_query = query
    enhanced_query = None

    print(f"[DEBUG] Original query: {query}")

    if enhance:
        from cli.gemini import enhance_query
        enhanced_query = enhance_query(query, enhance)
        query = enhanced_query
        print(f"[DEBUG] Enhanced query ({enhance}): {query}")
    else:
        print("[DEBUG] No query enhancement applied")

    if rerank_method:
        search_limit = limit * 5
        results = search.rrf_search(query, k, search_limit)

        print(f"[DEBUG] RRF search returned {len(results)} results (search_limit={search_limit}, k={k})")
        for i, r in enumerate(results[:10]):
            print(f"  [DEBUG] RRF result #{i + 1}: id={r['id']} title={r['title']} rrf_score={r['metadata']['rrf_score']:.4f}")

        if rerank_method == "individual":
            from cli.gemini import score_document
            for result in results:
                result["metadata"]["reranked_score"] = score_document(query, result)         
                time.sleep(3)            
            results = sorted(results, key=lambda doc: doc["metadata"]["reranked_score"], reverse=True)[:limit]
        elif rerank_method=="cross_encoder":
            pairs = []
            for doc in results:
                pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"]) 
            encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            scores = encoder.predict(pairs)       
            for score_idx, score in enumerate(scores):
                results[score_idx]["metadata"]["cross_encoder_score"] = float(score)

            results = sorted(results, key=lambda doc: doc["metadata"]["cross_encoder_score"], reverse=True)[:limit]
        else:
            from cli.gemini import score_batch_documents
            results = score_batch_documents(query, results)[:limit]

        print(f"[DEBUG] Final results after re-ranking ({rerank_method}):")
        for i, r in enumerate(results):
            print(f"  [DEBUG] Final #{i + 1}: id={r['id']} title={r['title']} rrf_score={r['metadata']['rrf_score']:.4f}")

    else:
        results = search.rrf_search(query, k, limit)

        print(f"[DEBUG] RRF search returned {len(results)} results (no re-ranking)")
        for i, r in enumerate(results):
            print(f"  [DEBUG] RRF result #{i + 1}: id={r['id']} title={r['title']} rrf_score={r['metadata']['rrf_score']:.4f}")

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhanced_method": enhance,
        "query": query,
        "k": k,
        "results": results
    }
