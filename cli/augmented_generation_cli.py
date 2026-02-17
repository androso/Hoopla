import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gemini import rag_generation
from lib.hybrid_search import rrf_search_command

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag",
        help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do the rag stuff here
            search_result = rrf_search_command(query, limit=5)
            rag_response = rag_generation(query, search_result["results"])

            print(f"Search results:")
            for doc in search_result['results']:
                print(f"    - {doc['title']}")

            print(f"RAG Response:")
            print(rag_response)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()