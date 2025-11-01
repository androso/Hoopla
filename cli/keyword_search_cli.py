import argparse
import json
from typing import Any, Dict, List

MOVIES_FILE = "data/movies.json"
MAX_RESULTS = 5

def load_movies(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r") as f:
        data = json.load(f)
        return data["movies"]

def search_movie(moviesList: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    result = []
    for movie in moviesList:
       if query in movie["title"]:
          result.append(movie) 
    return result
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    
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