from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model_name = "all-MiniLM-L6-v2"
    def generate_embedding(self, text):
        text.replace(" ", "")
        if text == "":
            raise ValueError("Empty space when generating embedding")
        embedding = self.model.encode([text])[0]   
        return embedding

def embed_text(text):
    semantic_search = SemanticSearch() 
    text_embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {text_embedding[:3]}")
    print(f"Dimensions: {text_embedding.shape[0]}")
    
def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model_name}") 
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")