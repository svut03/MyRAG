from transformers import AutoTokenizer
import numpy as np
from typing import Any

class PromptBuilder():
    def __init__(self, tokenizer_r_name: str):
        
        self.tokenizer_r = AutoTokenizer.from_pretrained(tokenizer_r_name)

        self.prompt_template = [
    {
        "role" : "system",
        "content" : """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer."""
    },
    {
        "role" : "user",
        "content" : """Context: 
        {context}
        -----
        Now here is the question you need to answer.
        Question: {question}""",
    },
]

        
    def build_prompt(self, query:str, chunks: list[dict[str, Any]]):

        documents = []
        
        for doc_id, chunk in enumerate(chunks):

            documents.append(
                f"""Document + {doc_id + 1}
        Source: {chunk['url']}
                
        {chunk['text']}"""
            )

        context = "\n\n---------------------\n\n".join(documents)
        
        rag_prompt_template = self.tokenizer_r.apply_chat_template(self.prompt_template, tokenize=False, add_generation_prompt=True)
        
        prompt = rag_prompt_template.format(question=query, context=context)
        
        return prompt