import argparse
from collections import defaultdict
import json
from typing import Any, DefaultDict, Dict, List, Set
import string
from nltk.stem import PorterStemmer
import os 
import pickle

MOVIES_FILE = "data/movies.json"
MAX_RESULTS = 5
stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self, movies: List[Dict[str, Any]], stop_words: List[str]):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, Dict[str, Any]] = {}
        self.stop_words = stop_words
        self.movies = movies
        
    def __add_document(self, doc_id: int, text:str):
        # tokenize
        text = remove_punctuation_from_str(text.lower()) 
        tokens = text.split()
        tokens = remove_stop_words_and_stem(tokens, self.stop_words) 

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            
    def get_documents(self, term: str) -> List[int]:
        term = term.lower()
        if term in self.index:
            return sorted(list(self.index[term]))
        return []
        
    def build(self) -> None:
        for doc_id, movie in enumerate(self.movies):
            self.docmap[doc_id] = movie

            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        
    def search(self, query):
        return
        
def remove_punctuation_from_str(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    cleanText = text.translate(table)
    return cleanText

def load_movies(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r") as f:
        data = json.load(f)
        return data["movies"]

def remove_stop_words_and_stem(tknList, stopwords):
    result = []
    for tkn in tknList:
        if tkn in stopwords:
            pass
        else:
            tkn = stemmer.stem(tkn)
            result.append(tkn)
            
    return result


def search_movie(moviesList: List[Dict[str, Any]], query: str,
                 stop_words: List[str]) -> List[Dict[str, Any]]:
    result = []
    queryTokenized = remove_stop_words_and_stem(query.split(" "), stop_words)

    for movie in moviesList:
        movieTitle = remove_punctuation_from_str(movie["title"].lower())
        movieTitleTokens = remove_stop_words_and_stem(set(movieTitle.split(" ")),
                                             stop_words)
        for tkn in queryTokenized:
            if any(tkn in movieTkn for movieTkn in movieTitleTokens):
                result.append(movie)
                break

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command",
                                       help="Available commands")

    search_parser = subparsers.add_parser("search",
                                          help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")
    
    args = parser.parse_args()

    with open("data/stopwords.txt", "r") as f:
        stop_words = f.read().split("\n")

    match args.command:
        case "search":
            print("Searching for:", args.query)
            args.query = remove_punctuation_from_str(args.query.lower())
            movies = load_movies("data/movies.json")
            results = search_movie(movies, args.query, stop_words)

            for index, movie in enumerate(results):
                if (index > 4):
                    break
                print(f"{index + 1}.", movie["title"])
        case "build": 
            print("Building inverted index...")
            movies = load_movies("data/movies.json")
            inverted_index = InvertedIndex(movies, stop_words)
            inverted_index.build()
            inverted_index.save()
            print("Index built and saved!")

            merida_docs = inverted_index.get_documents("merida")
            if merida_docs:
                first_id = merida_docs[0]
                print(f"First document ID for 'merida': {first_id} ({inverted_index.docmap[first_id]['title']})")
                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
