from .chunker import Chunker
from .retriever import Retriever
from .generator import Generator
from .prompt_builder import PromptBuilder
from .reranker import CrossEncoderReranker

class RAG():
    def __init__(self,
                 tokenizer_name: str,
                 embedder_name: str,
                 tokenizer_r_name: str,
                 reader_name: str,
                 reranker: str = None,
                 chunk_size: int = 200,
                 overlap: int = 20,
                 topk: int = 3,
                ):
        
        self.chunker = Chunker(tokenizer_name=tokenizer_name,
                               chunk_size=chunk_size,
                               overlap=overlap)
        self.retriever = Retriever(embedder_name=embedder_name)
        self.prompt_builder = PromptBuilder(tokenizer_r_name=tokenizer_r_name)
        self.generator = Generator(tokenizer_r_name=tokenizer_r_name,
                                   reader_name=reader_name)
        
        self.reranker = reranker
        
        if self.reranker is not None:
            self.reranker = CrossEncoderReranker(reranker, topk)

    def ask(self, question):

        retrieved_chunks = self.retriever.retrieve(question)

        if self.reranker is not None:
            
            reranked = self.reranker(question, retrieved_chunks)

            prompt = self.prompt_builder.build_prompt(question, reranked)

        else:
            
            prompt = self.prompt_builder.build_prompt(question, retrieved_chunks)

        answer = self.generator.generate(prompt)

        return answer
    
    def load(self):
        self.retriever.load()