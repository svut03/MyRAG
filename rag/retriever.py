import faiss
from sentence_transformers import SentenceTransformer

class Retriever():
    def __init__(self, 
                 embedder_name: str):
        
        self.embedder = SentenceTransformer(embedder_name)
    
    def build_index(self, chunks: list[dict]):

        texts = [chunk['text'] for chunk in chunks]

        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        dimension = embeddings.shape[-1]
        
        self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embeddings)

    
    def search(self, 
                   query:str, 
                   topn: int = 20):
        
        query_embedding = self.embedder.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, topn)

        indices = indices.squeeze() 

        return distances, indices

    
    def retrieve_chunks(self, query:str, 
                        chunks: list[dict]):
        retrieves_chunks = []
        
        _, indices = self.search(query)

        for doc_id, i in enumerate(indices):
            retrieves_chunks.append([chunks[i]['text'], i])
        
        return retrieves_chunks