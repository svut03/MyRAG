
from rag.retriever import Retriever
from rag.reranker import ColBertReranker

reranker = ColBertReranker(transformer='google-bert/bert-base-uncased',
                           tokenizer='google-bert/bert-base-uncased')
retriever = Retriever(embedder_name="sentence-transformers/all-MiniLM-L6-v2")

retriever.load()

reranker.build_tensors(retriever.chunks)






