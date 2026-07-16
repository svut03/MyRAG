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
                 cross_encoder_name: str,
                 chunk_size: int = 200,
                 overlap: int = 20,
                 topk: int = 3,
                 reranking: bool = True
                ):
        
        self.chunker = Chunker(tokenizer_name=tokenizer_name,
                               chunk_size=chunk_size,
                               overlap=overlap)
        self.retriever = Retriever(embedder_name=embedder_name)
        self.prompt_builder = PromptBuilder(tokenizer_r_name=tokenizer_r_name)
        self.generator = Generator(tokenizer_r_name=tokenizer_r_name,
                                   reader_name=reader_name)
        self.reranker = CrossEncoderReranker(cross_encoder_name, topk)
        self.reranking = reranking

    def ask(self, question, dataset):

        chunks = self.chunker.chunk(dataset)

        self.retriever.build_index(chunks)

        _, indices = self.retriever.search(question)

        if self.reranking:
            
            retrieved_chunks = self.retriever.retrieve_chunks(question)

            scores, indices = self.reranker.rerank(question, retrieved_chunks)

            context = self.prompt_builder.retrieve(question, indices)

        else:
            
            _, indices = self.retriever.search(question)
            
            context = self.prompt_builder.retrieve(question, indices)

        prompt = self.prompt_builder.build(
            question,
            context,
            chunks
            )

        answer = self.generator.generate(prompt)

        return answer