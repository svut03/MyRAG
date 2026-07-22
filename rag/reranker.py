from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModel
from typing import Any
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm


class CrossEncoderReranker:
    def __init__(self, cross_encoder_name: str, topk: int = 3):

        self.model = CrossEncoder(cross_encoder_name)
        self.topk = topk

    def __call__(self, question: str, retrieved_chunks: list[dict[str, Any]]):

        question_chunks = [(question, chunk["text"]) for chunk in retrieved_chunks]

        scores = self.model.predict(question_chunks)

        reranked = []

        for chunk, score in zip(retrieved_chunks, scores):
            chunk = chunk.copy()
            chunk["score"] = float(score)
            reranked.append(chunk)

        reranked = sorted(reranked, key=lambda chunk: chunk["score"], reverse=True)

        return reranked[: self.topk]


class ColBertReranker:
    def __init__(
        self,
        transformer: str,  # google-bert/bert-base-uncased
        tokenizer: str,  # google-bert/bert-base-uncased
        topk: int = 3,
    ):
        self.transformer = AutoModel.from_pretrained(transformer)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.topk = topk
        self.transformer.eval()

    def __call__(self, question: str, retrieved_chunks: list[dict[str, Any]]):

        embedding_query = self.encode(question).squeeze(0)
        embeddings_chunks = [
            self.embeddings[chunk["index"]] for chunk in retrieved_chunks
        ]

        scores = []

        for emb_chunk in embeddings_chunks:
            similarity_matrix = self._compute_similarity(embedding_query, emb_chunk)
            score = self._max_sim(similarity_matrix)
            scores.append(score.item())

        reranked = []

        for chunk, score in zip(retrieved_chunks, scores):
            chunk = chunk.copy()
            chunk["score"] = float(score)
            reranked.append(chunk)

        reranked = sorted(reranked, key=lambda chunk: chunk["score"], reverse=True)

        return reranked[: self.topk]

    def _compute_similarity(self, emb_query, emb_doc):

        similarity = emb_query @ emb_doc.T

        return similarity

    def _max_sim(self, similarity_matrix):
        return similarity_matrix.max(dim=1).values.sum()

    def encode(self, texts: list[str] | str):

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.transformer(**inputs)

        return F.normalize(outputs.last_hidden_state, p=2, dim=-1)

    def build_tensors(
        self,
        chunks: list[dict[str, Any]],
        embeddings_path="storage/token_embeddings.pt",
        batch_size: int = 32,
    ):

        embeddings = {}

        for start in tqdm(range(0, len(chunks), batch_size),
                          desc="Encoding documents"):
            batch_chunks = chunks[start : start + batch_size]

            texts = [chunk["text"] for chunk in batch_chunks]

            embedded = self.encode(texts)

            for idx, emb in enumerate(embedded, start=start):
                embeddings[idx] = emb.cpu()

        assert len(chunks) == len(embeddings)

        torch.save(embeddings, embeddings_path)

        return f"Данные успешно сохранены в {embeddings_path}"

    def load_tensors(self, embeddings_path="storage/token_embeddings.pt"):
        assert os.path.exists(embeddings_path), f"Файл {embeddings_path} не найден"

        self.embeddings = torch.load(embeddings_path, map_location="cpu")
