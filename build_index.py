from rag.chunker import Chunker
from rag.retriever import Retriever
from datasets import load_dataset

ds = load_dataset("m-ric/huggingface_doc", split="train[:10%]")

chunker = Chunker(
    tokenizer_name="sentence-transformers/all-MiniLM-L6-v2", chunk_size=200, overlap=20
)

retriever = Retriever(embedder_name="sentence-transformers/all-MiniLM-L6-v2")

chunks = chunker.chunk(ds)

retriever.build_index(chunks)

retriever.save()


