import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        embedder_name: str,
    ):

        self.embedder = SentenceTransformer(embedder_name)

    def build_index(self, chunks: list[dict]):

        self.chunks = chunks

        texts = [chunk["text"] for chunk in self.chunks]

        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        dimension = embeddings.shape[-1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

    def retrieve(self, query: str, topn: int = 20):

        if not hasattr(self, "index"):
            raise RuntimeError("Index is not built. Call build_index() or load().")

        query_embedding = self.embedder.encode(query)
        query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, topn)

        indices = indices.squeeze()
        distances = distances.squeeze()

        retrieves_chunks = []

        for index, distance in zip(indices, distances):
            chunk = self.chunks[index].copy()
            chunk["index"] = index
            chunk["score"] = distance
            retrieves_chunks.append(chunk)

        return retrieves_chunks

    def save(self, index_path="storage/index.faiss", chunks_path="storage/chunks.pkl"):

        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

        faiss.write_index(self.index, index_path)

        return "Данные успешно сохранены в chunks.pkl и index.faiss"

    def load(self, index_path="storage/index.faiss", chunks_path="storage/chunks.pkl"):
        assert os.path.exists(chunks_path), "Файл chunks.pkl не найден"
        assert os.path.exists(index_path), "Файл index.faiss не найден"

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
