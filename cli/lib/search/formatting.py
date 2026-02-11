from .constants import SCORE_PRECISION


def format_search_results(
    doc_id: str,
    title: str,
    document: str,
    score: float,
    **metadata,
) -> dict:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
