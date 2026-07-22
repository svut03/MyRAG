from rag import RAG

rag = RAG(
    tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
    embedder_name="sentence-transformers/all-MiniLM-L6-v2",
    tokenizer_r_name="Qwen/Qwen2.5-1.5B-Instruct",
    reader_name="Qwen/Qwen2.5-1.5B-Instruct",
    reranker='google-bert/bert-base-uncased',
)

rag.load()

while True:
    question = input("> ")

    if question.lower() in {"exit", "quit"}:
        break

    if question.lower() == ":chunks":
        print(rag.retriever.chunks)
        continue

    if question.startswith(":topn "):
        query = question[len(":topn "):]
        print(rag.retriever.retrieve(query))
        continue

    if question.lower() == ":topk":
        query = question[len(":topk "):]
        print(rag.reranker(question, rag.retriever.retrieve(query)))
        continue

    answer = rag.ask(question)

    print()
    print(answer)
    print()
