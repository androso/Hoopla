from collections import Counter
import os
import pickle
import string
from typing import Any, Dict, List, Set
from nltk import tokenize
from nltk.stem import PorterStemmer
from lib.search_utils import CACHE_DIR, load_movies, load_stop_words
import math

MAX_RESULTS = 5
stemmer = PorterStemmer()

def tokenize_text(text: str) -> List[str]:
    text = text.lower()
    text = remove_punctuation_from_str(text)
    tokens = text.split()
    stop_words = load_stop_words()
    tokens = remove_stop_words(tokens, stop_words)
    tokens = stem_tokens(tokens)
    
    return tokens
    
class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, Dict[str, Any]] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies = {}
        self.term_freq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl") 
        
    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1 

    def get_tf(self, doc_id, term):
        token = tokenize_text(term)
        
        if len(token) > 1:
            raise Exception("you must only give one term")

        return self.term_frequencies[int(doc_id)][token[0]]
        
    def get_idf(self, term):
        token = tokenize_text(term)
        if len(token) > 1:
            raise Exception("you must only give one term")
        docs_count = len(self.docmap) + 1
        term_count = len(self.get_documents(token[0])) + 1
        idf = math.log(docs_count / term_count)
        return idf
        
    def get_tf_idf(self, doc_id, term):
        return self.get_tf(doc_id, term) * self.get_idf(term) 

    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_text(term)
        if len(token) > 1:
            raise Exception("you must only give one term")
        
        df = len(self.get_documents(token[0]))
        idf = math.log((len(self.docmap) - df + 0.5) / (df + 0.5) + 1) 
        return idf
        
    def get_documentea(self, term: str, max_results=None):
        document_ids = []
        result = []
        
        if term in self.index:
            document_ids = sorted(list(self.index[term]))
            
        for id in document_ids:
            if (max_results and len(result) >= max_results):
               break 
            else:
                result.append(self.docmap[id])   
                
        return result

    def build(self) -> None:
        movies = load_movies()
        
        for idx, movie in enumerate(movies):
            doc_id = movie['id']
            self.docmap[doc_id] = movie

            doc_description = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, doc_description)
            
    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as file:
                self.index = pickle.load(file)
            with open(self.docmap_path, "rb") as file:
                self.docmap = pickle.load(file) 
            with open(self.term_freq_path, "rb") as file: 
                self.term_frequencies = pickle.load(file)
        except FileNotFoundError:
            print("Cached files not found. Run 'build' command")
            raise
        except pickle.UnpicklingError:
            print("Cache corrupted. Run 'build' command again")
            raise
        
    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_freq_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
            
    def search(self, query):
        return

def remove_punctuation_from_str(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(table)
    return clean_text

def remove_stop_words(tkn_list, stop_words):
    result = []
    for tkn in tkn_list:
        if tkn in stop_words:
            pass
        else:
            result.append(tkn)
    return result


def stem_tokens(tkn_list):
    result = []
    for tkn in tkn_list:
        result.append(stemmer.stem(tkn))
    return result


def search_movie(inverted_index: InvertedIndex, query: str):
    query_tokenized = tokenize_text(query)
    document_ids = []
    
    for token in query_tokenized:
        document_ids.extend(inverted_index.get_documents(token, MAX_RESULTS))
        
    return document_ids 
