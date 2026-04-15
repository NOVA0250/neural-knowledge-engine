from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents = []

    def build_index(self, documents):
        self.documents = documents
        embeddings = self.model.encode(documents)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def search(self, query, top_k=5):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append((self.documents[idx], distances[0][i]))

        return results
