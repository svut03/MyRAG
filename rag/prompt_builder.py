from transformers import AutoTokenizer
import numpy as np

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

    def retrieve(self, query:str,
                 indices: np.ndarray,
                 chunks: list[dict]):

        documents = []
        
        for doc_id, i in enumerate(indices):

            documents.append(
                f"""Document + {doc_id + 1}
        Source: {chunks[i]['url']}
                
        {chunks[i]['text']}"""
            )

        context = "\n\n---------------------\n\n".join(documents)
        
        return context

        
    def build_prompt(self, query:str, context:str):
        
        rag_prompt_template = self.tokenizer_r.apply_chat_template(self.prompt_template, tokenize=False, add_generation_prompt=True)
        
        prompt = rag_prompt_template.format(question=query, context=context)
        
        return prompt