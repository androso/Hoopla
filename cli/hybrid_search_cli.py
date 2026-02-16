import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from lib.hybrid_search import (
    normalize_scores,
    hybrid_search_command,
    rrf_search_command
)

from gemini import spellcheck_query, evaluate_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize")
    normalize_parser.add_argument("scores", type=float, nargs="+")

    weighted_search_parser = subparser.add_parser("weighted-search")
    weighted_search_parser.add_argument("query", type=str, help="query to use for search")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Alpha")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit of results for the search")

    rrf_search_parser = subparser.add_parser("rrf-search")
    rrf_search_parser.add_argument("query", type=str, help="Query to use for search")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="Constant to use in the ranking")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit of search results")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Use LLM based reranking for better results")
    rrf_search_parser.add_argument("--evaluate", action="store_true", help="Run Evaluation") 

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores(args.scores)
            if len(scores) >= 1:
                for score in scores:
                    print(f"* {score:.4f}")
        case "weighted-search":
            results = hybrid_search_command(args.query, args.alpha, args.limit)
            
            for i, result in enumerate(results, 1):
               print(f"{i}. {result['title']}") 
               print(f"     Hybrid score: {result['score']}")
               print(f"     BM25: {result['metadata']['bm25_score']}, Semantic: {result['metadata']['semantic_score']}")
               print(f"     {result['document'][:100]}...")

        case "rrf-search":
            result = rrf_search_command(args.query, args.k, args.enhance, args.limit, args.rerank_method)
                
            if result["enhanced_query"]:
                print(f"Enhanced query ({result['enhanced_method']}): '{result['original_query']} -> '{result['enhanced_query']}")

            for i, res in enumerate(result["results"], 1):
               print(f"{i}. {res['title']}") 
               if args.rerank_method == "cross_encoder":
                    print(f"Cross Encoder Score: {res["metadata"]["cross_encoder_score"]}")
               if args.rerank_method == "batch":
                    print(f"     Rerank Rank: {i}")
               if args.rerank_method == "individual": 
                    print(f"     Rerank Score: {res["metadata"]['reranked_score']}")
                
               print(f"     RRF Score: {res["metadata"]['rrf_score']:.3f}")
               print(f"     BM25 Rank: {res["metadata"]['bm25_rank']}, Semantic Rank: {res["metadata"]['semantic_rank']}")
               print(f"     {res['document'][:100]}...")
            
            if args.evaluate:
                # print(result)
                results_evaluated = evaluate_results(result['query'], result["results"])
                for doc_idx, doc in enumerate(results_evaluated, 1):
                    print(f"{doc_idx}. {doc['title']}: {doc['metadata']['llm_score']}/3")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
