import argparse
import json
from typing import Any, Dict, List
import string

MOVIES_FILE = "data/movies.json"
MAX_RESULTS = 5

def remove_punctuation_from_str(text: str) -> str:
    table = str.maketrans("", "", string.punctuation) 
    cleanText = text.translate(table)
    return cleanText
    
def load_movies(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r") as f:
        data = json.load(f)
        return data["movies"]

def search_movie(moviesList: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    result = []
    for movie in moviesList:
       movieTitle = remove_punctuation_from_str(movie["title"].lower())
       if query in movieTitle:
          result.append(movie) 
    return result
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    args.query = remove_punctuation_from_str(args.query.lower())
    
    match args.command:
        case "search":
            print("Searching for:", args.query)
            movies = load_movies("data/movies.json")
            results = search_movie(movies, args.query) 
            for index, movie in enumerate(results):
                if (index > 4):
                    break
                print(f"{index + 1}.", movie["title"])
        case _:
            parser.print_help()
            
if __name__ == "__main__":
    main()