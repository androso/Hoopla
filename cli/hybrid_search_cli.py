import argparse

from lib.hybrid_search import (
    normalize_scores,
    hybrid_search_command,
    rrf_search_command
)

from gemini import spellcheck_query

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
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell"], help="Query enhancement method")
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
               print(f"     {result['document']}...")

        case "rrf-search":
            if args.enhance:
                enhanced_query = spellcheck_query(args.query)
                print(f"Enhanced query ({args.enhance}): '{args.query} -> '{enhanced_query}")
                results = rrf_search_command(enhanced_query, args.k, args.limit)
            else:
                results = rrf_search_command(args.query, args.k, args.limit) 

            for i, result in enumerate(results, 1):
               print(f"{i}. {result['title']}") 
               print(f"     RRF Score: {result["metadata"]['rrf_score']}")
               print(f"     BM25 Rank: {result["metadata"]['bm25_rank']}, Semantic Rank: {result["metadata"]['semantic_rank']}")
               print(f"     {result['document']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
