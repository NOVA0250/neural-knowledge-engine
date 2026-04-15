from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, embedding_manager, documents, top_k=5):
        self.embedding_manager = embedding_manager
        self.documents = documents
        self.top_k = top_k

        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def hybrid_search(self, query):
        dense_results = self.embedding_manager.search(query, self.top_k)

        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        combined = []

        for i, doc in enumerate(self.documents):
            score = bm25_scores[i]
            combined.append((doc, score))

        combined.sort(key=lambda x: x[1], reverse=True)

        return combined[:self.top_k]
