import argparse
from lib.augmented_generation import rag_command, summarize_command, citations_command

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
        help="Generate multi-document summary"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Maximum number of documents to summarize")

    citations_parser = subparsers.add_parser(
        "citations",
        help="Perform RAG with citations"
    )

    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument("--limit", type=int, default=5, help="Maximum number of documents to use")


    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_response = rag_command(query)

            print(f"Search results:")
            for doc in rag_response['results']:
                print(f"    - {doc['title']}")

            if 'answer' in rag_response:
                print(f"RAG Response:")
                print(rag_response["answer"])

        case "summarize":
            query = args.query
            limit = args.limit
            response = summarize_command(query, limit)

            print("Search Results")
            for doc in response["search_results"]:
                print(f"    - {doc['title']}")
            if 'errors' in response:
                print(response["errors"])
            else:
                print("LLM Summary:")
                print(response["summary"])

        case "citations":
            response = citations_command(args.query, args.limit)

            if 'errors' in response:
                print(response['errors'])
            else:
                print("Search Results:")
                for res in response["search_results"]:
                    print(f" -     {res['title']}")
                print(response['answer'])
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()