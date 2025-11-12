import argparse
import pickle
import math
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
    
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a given doc")
    tf_parser.add_argument("doc_id", type=str, help="document id")
    tf_parser.add_argument("term", type=str, help="Term to get the count for")

    idf_parser = subparsers.add_parser("idf", help="Calculates the IDF for a given term")
    idf_parser.add_argument("term", type=str, help="the term to get the idf for")
    
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
        case "idf":
            print("Calculating IDF")
            inverted_index = InvertedIndex()
            inverted_index.load()

            docs_count = len(inverted_index.docmap) + 1
            term_count = len(search_movie(inverted_index, args.term)) + 1
            
            idf = math.log(docs_count / term_count)
            
            print(f"Inverse document frequency of '{args.term}': {idf:.2}") 
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
            
        case "tf":
            inverted_index = InvertedIndex()
            
            inverted_index.load()
            count = inverted_index.get_tf(args.doc_id, args.term)
            print(count)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
