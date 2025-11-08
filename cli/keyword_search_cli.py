import argparse
import pickle

from lib.keyword_search import InvertedIndex, search_movie
from lib.search_utils import load_movies, load_stop_words

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command",
                                       help="Available commands")

    search_parser = subparsers.add_parser("search",
                                          help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    stop_words = load_stop_words()

    match args.command:
        case "build":
            print("Building inverted index...")
            movies = load_movies()
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
            print("Index built and saved!")
            
        case "search":
            print("Searching for:", args.query)
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except(FileNotFoundError, pickle.UnpicklingError):
                print("Error: Index not found. Please run 'build' command first") 
                return
                
            results = search_movie(inverted_index, args.query)
            print(results) 
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
