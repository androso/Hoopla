import argparse
from lib.hybrid_search import rrf_search_command
from lib.augmented_generation import rag_command, summarize_command

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag",
        help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize movies results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to get results")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Limit of documents to search for")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_response = rag_command(query, search_result["results"])

            print(f"Search results:")
            for doc in rag_response['results']:
                print(f"    - {doc['title']}")

            print(f"RAG Response:")
            print(rag_response["answer"])

        case "summarize":
            query = args.query
            limit = args.limit
            response = summarize_command(query, limit)

            print("Search Results")
            for doc in response["results"]:
                print(f"    - {doc['title']}")
            print("LLM Summary:")
            print(response["answer"])

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()