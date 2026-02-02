from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import json 
import re

from numpy.typing import NDArray
from lib.search_utils import MOVIE_EMBEDDINGS_PATH, load_movie_embeddings, load_movies, load_chunk_embeddings, load_metadata_chunk_embeddings, MOVIE_CHUNK_EMBEDDINGS_PATH, METADATA_CHUNK_EMBEDDINGS_PATH, cosine_similarity, SCORE_PRECISION

_semantic_search_instance: Optional['SemanticSearch'] = None
_chunked_semantic_search_instance: Optional['ChunkedSemanticSearch'] = None

def get_chunked_semantic_search() -> 'ChunkedSemanticSearch':
    global _chunked_semantic_search_instance
    if _chunked_semantic_search_instance is None:
        _chunked_semantic_search_instance = ChunkedSemanticSearch()

    return _chunked_semantic_search_instance

def get_semantic_search() -> 'SemanticSearch':
    """Get or create the singleton SemanticSearch instance."""
    global _semantic_search_instance
    if _semantic_search_instance is None:
        _semantic_search_instance = SemanticSearch()
    return _semantic_search_instance

def get_semantic_chunks(text, max_chunk_size, overlap) -> list[str]:
    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, text) 
    step = max_chunk_size - (overlap or 0) 
    chunks = []

    for chunk_idx in range(0, len(sentences), step):
        chunk_sentences = sentences[chunk_idx:chunk_idx + max_chunk_size]
        chunk = " ".join(chunk_sentences)
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(chunk)

    return chunks

class SemanticSearch:
    """Handles semantic search operations using sentence transformers."""
    
    model_name: str
    model: SentenceTransformer
    embeddings: Optional[NDArray[np.floating[Any]]]
    documents: Optional[List[Dict[str, Any]]]
    document_map: Dict[Any, Dict[str, Any]]
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the semantic search with a pre-trained model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        try:
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        
    def build_embeddings(self, documents: List[Dict[str, Any]]) -> NDArray[np.floating[Any]]:
        """Build embeddings for a list of documents.
        
        Args:
            documents: List of document dictionaries containing 'id', 'title', and 'description'
            
        Returns:
            Numpy array of embeddings
            
        Raises:
            ValueError: If documents list is empty
            RuntimeError: If encoding fails
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        self.documents = documents
        doc_descriptions: List[str] = []
 
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_descriptions.append(f"{doc['title']}: {doc['description']}")
        try:
            self.embeddings = self.model.encode(doc_descriptions, show_progress_bar=True)
        except Exception as e:
            raise RuntimeError(f"Failed to encode documents: {str(e)}")

        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings) 
        return self.embeddings
        
    def load_or_create_embeddings(self, documents: List[Dict[str, Any]]) -> NDArray[np.floating[Any]]:
        """Load existing embeddings or create new ones if needed.
        
        Args:
            documents: List of document dictionaries containing 'id', 'title', and 'description'
            
        Returns:
            Numpy array of embeddings
            
        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
            
        try: 
            self.embeddings = load_movie_embeddings()
        except FileNotFoundError:
            return self.build_embeddings(documents)
            
        if len(self.embeddings) != len(documents):
            return self.build_embeddings(documents)
        return self.embeddings
        
    def generate_embedding(self, text: str) -> NDArray[np.floating[Any]]:
        """Generate an embedding vector for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array containing the embedding vector
            
        Raises:
            ValueError: If text is empty or whitespace-only
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace-only")
        
        embedding: NDArray[np.floating[Any]] = self.model.encode([text])[0]
        return embedding
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            dict with model_name and max_seq_length
        """
        return {
            "model_name": self.model_name,
            "max_seq_length": self.model.max_seq_length
        }
        
    def search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for documents similar to the query.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            
        Returns:
            List of result dictionaries with 'score', 'title', and 'description'
            
        Raises:
            ValueError: If no embeddings are loaded
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)        
        
        results: List[tuple[float, Dict[str, Any]]] = []
        for idx, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            if self.documents:
                doc = self.documents[idx] 
                results.append((similarity, doc))
            
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)[:limit]
        formatted_results: List[Dict[str, Any]] = []
        for score, doc in sorted_results:
            result = {
                'score': score,
                'title': doc['title'],
                'description': doc['description'] 
            }
            formatted_results.append(result) 
        return formatted_results

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        if not documents:
            raise ValueError("Documents list cannot be empty")

        self.documents = documents 
        chunks = []
        chunks_meta = []
        total_doc_chunks = 0
        for movie_idx, doc in enumerate(documents):
            if not doc["description"]:
                continue
            self.document_map[doc["id"]] = doc
            doc_chunks = get_semantic_chunks(doc["description"], 4, 1)
            total_doc_chunks += len(doc_chunks) 
            chunks.extend(doc_chunks)
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunks_meta.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(doc_chunks)
                })

        print(total_doc_chunks)
        try:
            self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
            self.chunk_metadata = {
                "chunks": chunks_meta,
                "total_chunks": len(chunks)
            } 
        except Exception as e:
            raise RuntimeError(f"Failed to encode documents: {str(e)}")
        np.save(MOVIE_CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings) 
        with open(METADATA_CHUNK_EMBEDDINGS_PATH, "w") as f:
            json.dump(self.chunk_metadata, f, indent=2)
            # json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if not documents:
            raise ValueError("Document list cannot be empty")
        self.documents = documents 
        for doc in documents:
            self.document_map[doc["id"]] = doc

        try:
            self.chunk_embeddings = load_chunk_embeddings()
            self.chunk_metadata = load_metadata_chunk_embeddings()
            return self.chunk_embeddings
        except FileNotFoundError:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query:str, limit:int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None: 
            raise ValueError("No embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        print("[search_chunks] Generating query embedding")
        query_embedding = self.generate_embedding(query) 
        print("[search_chunks] Scoring chunks")
        movie_score = {}
        print(f"[search_chunks] Total chunks: {len(self.chunk_embeddings)}")
        for chunk_idx, chunk in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk) 
            meta = self.chunk_metadata["chunks"][chunk_idx]
            chunk_scores.append({
                "chunk_idx": chunk_idx,
                "movie_idx": meta["movie_idx"],
                "score": similarity
            })
            score = movie_score.get(meta["movie_idx"], float("-inf"))
            if similarity > score:
                movie_score[meta["movie_idx"]] = similarity

        print(f"[search_chunks] Aggregated scores for {len(movie_score)} movies")
        print("[search_chunks] Sorting results")
        sorted_scores = sorted(movie_score.items(), key=lambda kv: kv[1], reverse=True)
        top_items = sorted_scores[:limit] 
        print(f"[search_chunks] Preparing top {len(top_items)} results")
        results = []
        for movie_idx, score in top_items:
            for doc in [self.documents[movie_idx]]:
                results.append({
                    "id": doc["id"],
                    "title": doc.get("title", ""),
                    "document": (doc.get("description") or "")[:100],
                    "score": round(score, SCORE_PRECISION),
                    "metadata": doc.get("metadata") or {}
                })
        return results
        


def embed_text(text: str) -> NDArray[np.floating[Any]]:
    """Generate embedding for text using the cached model instance.
    
    Args:
        text: Input text to embed
        
    Returns:
        numpy array containing the embedding vector
    """
    search = get_semantic_search()
    return search.generate_embedding(text)


def get_model_info() -> Dict[str, Any]:
    """Get information about the cached model instance.
    
    Returns:
        dict with model_name and max_seq_length
    """
    search = get_semantic_search()
    return search.get_model_info()

def embed_query_text(query: str) -> None:
    """Embed a query and print its properties.
    
    Args:
        query: Query text to embed
    """
    search = get_semantic_search()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}") 
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")



## COMMANDS ACTIONS
def verify_model():
    search_instance = get_semantic_search()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")
    
def verify_embeddings() -> None:
    """Verify embeddings by loading/creating them and printing their shape."""
    search = get_semantic_search()
    movies = load_movies()
    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of docs: {len(movies)}") 
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]}")

def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0

    step = chunk_size - (overlap or 0) 
    for chunk_idx in range(0, len(words), step):
        chunk_words = words[chunk_idx : chunk_idx + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP):
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def semantic_chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP):
    chunks = get_semantic_chunks(text, chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")

    for idx, chunk in enumerate(chunks, 1):
        print(f"{idx}. {chunk}")

def embed_chunks_command():
    movies = load_movies()
    instance = get_chunked_semantic_search() 
    embeddings = instance.load_or_create_chunk_embeddings(movies)
    return embeddings

def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    searcher = get_chunked_semantic_search()
    searcher.load_or_create_chunk_embeddings(movies)
    results = instance.search_chunks(args.query, args.limit)
    return {"query": query, "results": results}

def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    search = get_semantic_search()
    movies = load_movies()
    search.load_or_create_embeddings(movies)
    results = search.search(query, limit)

    for idx, result in enumerate(results):
        print(f"{idx + 1}. {result['title']} (score: {result['score']})") 