from PIL import Image
from sentence_transformers import SentenceTransformer
from .search.scoring import cosine_similarity
from .search.loaders import load_movies

class MultimodalSearch:
    def __init__(self, model_name= "clip-ViT-B-32", documents = []):
        self.model = SentenceTransformer(model_name)
        self.documents = documents 
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)


    def embed_image(self, path: str):
        img = Image.open(path)
        img_embeddings = self.model.encode([img])
        return img_embeddings[0]

    def search_with_image(self, path: str): 

        img_embedding = self.embed_image(path)
        similarities = [
            (doc, cosine_similarity(text_embedding, img_embedding))
            for doc, text_embedding in zip(self.documents, self.text_embeddings)
        ] 
        similarities.sort(key=lambda s: s[1], reverse=True)

        return similarities[:5]


        
def verify_image_embedding(img_path: str):
    search = MultimodalSearch()

    embedding = search.embed_image(img_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(img_path: str):
    movies = load_movies()
    search = MultimodalSearch(documents=movies)
    res = search.search_with_image(img_path)

    return res