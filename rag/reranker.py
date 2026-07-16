import numpy as np
from sentence_transformers import CrossEncoder

class CrossEncoderReranker():
    def __init__(
        self,
        cross_encoder_name: str,
        topk : int = 3):
        
        self.model = CrossEncoder(cross_encoder_name)
        self.topk = topk

        
    def rerank(self, question : str, retrieved_chunks: list[str, np.ndarray]):
        
        question_chunks = [(question, chunk[0]) for chunk in retrieved_chunks]
        
        scores = self.model.predict(question_chunks)

        indices = [ids[1] for ids in retrieved_chunks]

        reranked = sorted(zip(scores, indices), reverse=True)[:self.topk]
        
        return reranked