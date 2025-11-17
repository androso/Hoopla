from sentence_transformers import SentenceTransformer


_semantic_search_instance = None


def get_semantic_search():
    """Get or create the singleton SemanticSearch instance."""
    global _semantic_search_instance
    if _semantic_search_instance is None:
        print("building semantic search")
        _semantic_search_instance = SemanticSearch()
    return _semantic_search_instance


class SemanticSearch:
    """Handles semantic search operations using sentence transformers."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the semantic search with a pre-trained model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        try:
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")
    
    def generate_embedding(self, text):
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
        
        embedding = self.model.encode([text])[0]
        return embedding
    
    def get_model_info(self):
        """Get information about the loaded model.
        
        Returns:
            dict with model_name and max_seq_length
        """
        return {
            "model_name": self.model_name,
            "max_seq_length": self.model.max_seq_length
        }


def embed_text(text):
    """Generate embedding for text using the cached model instance.
    
    Args:
        text: Input text to embed
        
    Returns:
        numpy array containing the embedding vector
    """
    search = get_semantic_search()
    return search.generate_embedding(text)


def get_model_info():
    """Get information about the cached model instance.
    
    Returns:
        dict with model_name and max_seq_length
    """
    search = get_semantic_search()
    return search.get_model_info()
