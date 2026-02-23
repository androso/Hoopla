import os
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer

from .search.constants import DEFAULT_SEARCH_LIMIT, SCORE_PRECISION
from .search.formatting import format_search_results
from .search.loaders import load_movies
from .search.scoring import cosine_similarity


class MultimodalSearch:
    """Handles multimodal (image-to-text) search using CLIP embeddings."""

    model_name: str
    model: SentenceTransformer
    documents: List[Dict[str, Any]]
    text_embeddings: Optional[NDArray[np.floating[Any]]]

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        documents: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Initialize multimodal search with a CLIP model.

        Args:
            model_name: Name of the CLIP sentence transformer model to use.
            documents: List of document dicts with 'id', 'title', and 'description'.
        """
        try:
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")

        self.documents = documents or []
        if self.documents:
            texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
            self.text_embeddings = self.model.encode(texts, show_progress_bar=True)
        else:
            self.text_embeddings = None

    def embed_image(self, path: str) -> NDArray[np.floating[Any]]:
        """Generate a CLIP embedding for an image file.

        Args:
            path: Filesystem path to the image.

        Returns:
            1-D numpy array representing the image embedding.

        Raises:
            FileNotFoundError: If the image path does not exist.
            ValueError: If the file cannot be opened as an image.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            img = Image.open(path)
        except Exception as e:
            raise ValueError(f"Unable to open image '{path}': {e}")

        img_embeddings = self.model.encode([img])
        return img_embeddings[0]

    def search_with_image(
        self, path: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Dict[str, Any]]:
        """Search documents by similarity to an image.

        Args:
            path: Filesystem path to the query image.
            limit: Maximum number of results to return.

        Returns:
            List of formatted result dicts sorted by descending similarity.

        Raises:
            ValueError: If no documents have been indexed.
        """
        if self.text_embeddings is None or not self.documents:
            raise ValueError("No documents indexed. Pass documents to the constructor first.")

        img_embedding = self.embed_image(path)

        scored = [
            (doc, cosine_similarity(text_emb, img_embedding))
            for doc, text_emb in zip(self.documents, self.text_embeddings)
        ]
        scored.sort(key=lambda s: s[1], reverse=True)

        return [
            format_search_results(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            for doc, score in scored[:limit]
        ]


def verify_image_embedding(img_path: str) -> None:
    """Load a CLIP model and print the dimensionality of an image embedding."""
    search = MultimodalSearch()
    embedding = search.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(
    img_path: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> List[Dict[str, Any]]:
    """Run an image-based search over the movie catalogue."""
    movies = load_movies()
    search = MultimodalSearch(documents=movies)
    return search.search_with_image(img_path, limit=limit)