from transformers import AutoTokenizer
import datasets

class Chunker():
    def __init__(self, 
                 tokenizer_name: str,
                 chunk_size: int,
                 overlap: int,
                ):
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunk_size = chunk_size
        self.overlap = overlap
        

    def chunk(self,
                 dataset: datasets.arrow_dataset.Dataset
                 ): 
    
        chunks = []

        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller yhan chank_size")
        
        for doc_id, doc in enumerate(dataset):

            ids = self.tokenizer.encode(doc['text'], add_special_tokens=False)
            url = doc['source']

            for chunk_id, start in enumerate(range(0, len(ids), self.chunk_size - self.overlap)):
                
                chunk = ids[start : start + self.chunk_size]
                chunks.append({
                    'text': self.tokenizer.decode(
                    chunk, 
                    skip_special_tokens=True),
                               'doc_id': doc_id,
                               'chunk_id': chunk_id,
                               'url': url})
                
        return chunks