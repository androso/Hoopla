import argparse
import pickle
from lib.keyword_search import InvertedIndex, search_movie
from lib.search_utils import BM25_B, BM25_K1, load_movies, load_stop_words

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

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF - IDF") 
    tfidf_parser.add_argument("doc_id", help="document id")
    tfidf_parser.add_argument("term", help="term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")                                        

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter") 

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")    
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Limit to documents returned")

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
            idf = inverted_index.get_idf(args.term)
            
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}") 
            
        case "tfidf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            tf_idf = inverted_index.get_tf_idf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            
        case "bm25idf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            bm25idf = inverted_index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            
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
            
        case "bm25search":
            inverted_index = InvertedIndex()
            inverted_index.load()
            results = inverted_index.bm25_search(args.query, args.limit)
            
            for index, result in enumerate(results):
                print(f"{index + 1}. ({result["id"]}) {result["title"]} - Score: {result["score"]:.2f}")
                
        case "tf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            count = inverted_index.get_tf(args.doc_id, args.term)
            print(count)
            
        case "bm25tf":
            inverted_index = InvertedIndex() 
            inverted_index.load()
            bm25_tf = inverted_index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()