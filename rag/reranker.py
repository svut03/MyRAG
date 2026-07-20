import numpy as np
from sentence_transformers import CrossEncoder
from typing import Any

class CrossEncoderReranker():
    def __init__(
        self,
        cross_encoder_name: str,
        topk : int = 3):
        
        self.model = CrossEncoder(cross_encoder_name)
        self.topk = topk

        
    def __call__(self, question : str, retrieved_chunks: list[dict[str, Any]]):
        
        question_chunks = [(question, chunk['text']) for chunk in retrieved_chunks]
        
        scores = self.model.predict(question_chunks)

        reranked = []

        for chunk, score in zip(retrieved_chunks, scores):
            chunk = chunk.copy()
            chunk['score'] = float(score)
            reranked.append(chunk)

        reranked = sorted(reranked, key=lambda chunk: chunk['score'], reverse=True)
        
        return reranked[:self.topk]