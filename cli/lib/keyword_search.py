from collections import Counter
import os
import pickle
import string
from typing import Any, Dict, List, Set
from nltk.stem import PorterStemmer
from lib.search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words, 
    format_search_results
)
import math

stemmer = PorterStemmer()

def tokenize_text(text: str) -> List[str]:
    text = text.lower()
    text = remove_punctuation_from_str(text)
    tokens = text.split()
    tokens = remove_stop_words(tokens)
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
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        
    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1 
        self.doc_lengths[doc_id] = len(tokens)
        
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
    
    def get_bm25_tf(self, doc_id, term, k1 = BM25_K1, b = BM25_B):
        raw_tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        
        return bm25_tf
        
    def get_documents(self, term: str, max_results=None):
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
            with open(self.doc_lengths_path, "rb") as file:
                self.doc_lengths = pickle.load(file)
        except FileNotFoundError:
            print("Cached files not found. Run 'build' command")
            raise
        except pickle.UnpicklingError:
            print("Cache corrupted. Run 'build' command again")
            raise
            
    def  __get_avg_doc_length(self):
        if len(self.docmap) > 0:
            sum = 0.0
            for doc_length in self.doc_lengths:
               sum += self.doc_lengths[doc_length] 
            avg = sum / len(self.doc_lengths)
            return avg
        return 0.0
        
    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_freq_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        score = tf * idf
        
        return score
        
    def bm25_search(self, query, limit = 5):
        query_tokenized = tokenize_text(query)
        scores_dict = {}

        relevant_docs = set()
        for token in query_tokenized:
            if token in self.index:
               relevant_docs.update(self.index[token]) 
        for doc_id in relevant_docs:
            scores_dict[doc_id] = 0.0
            for token in query_tokenized:
                score = self.bm25(doc_id, token)
                scores_dict[doc_id] += score

        sorted_results = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:limit]

        doc_results = []
        for doc_id, score in sorted_results:
            doc = self.docmap[doc_id]
            formatted_result = format_search_results(
                doc_id=doc['id'],
                title=doc['title'],
                document=doc['description'],
                score=score
            )
            doc_results.append(formatted_result)

        return doc_results 
        
def remove_punctuation_from_str(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(table)
    return clean_text

def remove_stop_words(tkn_list):
    stop_words = load_stop_words()
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
def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_docs = idx.get_documents(query_token)
        for doc in matching_docs:
            doc_id = doc["id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(doc)
            if len(results) >= limit:
                return results

    return results

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)
